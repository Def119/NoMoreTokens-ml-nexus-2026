"""Generate evidence from completed nested predictions and their actual fitted models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from .features import ID, TARGET, LABS, CATS
from .train import dump, metrics, FAMILIES, METHODS
from .models import fit_model


def row_loss(y,p):
    p=np.clip(p,1e-7,1-1e-7)
    return -(np.asarray(y)*np.log(p)+(1-np.asarray(y))*np.log1p(-p))


def interval(values, rng, repeats=2000):
    values=np.asarray(values)
    stats=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(repeats)])
    return np.quantile(stats,[.025,.975]).tolist()


def bundle_predict(bundle, method, X):
    if method.startswith('tabm'):
        from .neural import predict_bundle
        return predict_bundle(bundle,method,X)
    if method in FAMILIES:
        return bundle['models'][bundle['chosen'][method]['name']].predict(X)
    if method=='reference':
        return np.mean([m.predict(X) for n,m in bundle['models'].items() if n.startswith('reference')],axis=0)
    if method=='selected_inner':
        return bundle['models'][bundle['chosen'][bundle['best_inner_family']]['name']].predict(X)
    matrix=np.column_stack([bundle['models'][bundle['chosen'][f]['name']].predict(X) for f in FAMILIES])
    return bundle['combiners'][method].predict(matrix)


def report(out, learning=True):
    out=Path(out);dest=out/'report';dest.mkdir(exist_ok=True)
    s=json.loads((out/'summary.json').read_text());winner=s['winner']
    tr=pd.read_csv('train.csv');te=pd.read_csv('test.csv');oof=pd.read_csv(out/'oof.csv')
    assert oof[ID].equals(tr[ID])
    y=oof[TARGET].to_numpy();p=oof[winner].to_numpy();ref=oof.reference.to_numpy()
    seed=json.loads((out/'manifest.json').read_text())['seed']
    rng=np.random.default_rng(seed)
    delta=row_loss(y,p)-row_loss(y,ref)
    evidence=metrics(y,p)
    evidence.update(average_precision=float(average_precision_score(y,p)),delta_logloss=float(delta.mean()),
                    delta_ci95=interval(delta,rng),winner=winner,
                    fold_deltas=[float(delta[oof.fold==f].mean()) for f in sorted(oof.fold.unique())])
    calibration=LogisticRegression(C=1e6,max_iter=2000).fit(logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1),y)
    evidence.update(calibration_slope=float(calibration.coef_[0,0]),calibration_intercept=float(calibration.intercept_[0]))
    tn,fp,fn,tp=confusion_matrix(y,p>=.15).ravel()
    evidence.update(threshold=.15,sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)))
    # Patient bootstrap intervals condition on the fitted OOF predictions.
    groups={'sex':tr.sex,'rurality':tr.rurality,'hospital_type':tr.hospital_type,
            'age_group':pd.cut(tr.age,[-np.inf,40,60,80,np.inf],labels=['<40','40-59','60-79','80+'],right=False)}
    rows=[]
    for dimension,values in groups.items():
        for group in sorted(values.dropna().unique()):
            mask=np.asarray(values==group);gy=y[mask];gp=p[mask]
            ci=interval(row_loss(gy,gp),rng,500)
            rows.append(dict(dimension=dimension,group=str(group),n=int(mask.sum()),positives=int(gy.sum()),
                             prevalence=float(gy.mean()),log_loss=float(log_loss(gy,gp,labels=[0,1])),
                             ll_lower=ci[0],ll_upper=ci[1],brier=float(brier_score_loss(gy,gp)),
                             auc=float(roc_auc_score(gy,gp)) if len(np.unique(gy))==2 else None))
    subgroup=pd.DataFrame(rows);subgroup.to_csv(dest/'subgroups.csv',index=False)
    entropy=-(p*np.log2(p)+(1-p)*np.log2(1-p))
    order=np.argsort(entropy,kind='stable');retained=order[:int(len(p)*.9)];referred=order[int(len(p)*.9):]
    evidence['referral']=dict(retained_n=len(retained),referred_n=len(referred),retained=metrics(y[retained],p[retained]),
                              retained_prevalence=float(y[retained].mean()),referred_prevalence=float(y[referred].mean()))
    num=tr.select_dtypes(include='number').columns.drop(TARGET)
    shift=pd.DataFrame({'feature':num,'train_missing':[tr[c].isna().mean() for c in num],
                        'test_missing':[te[c].isna().mean() for c in num]})
    shift.to_csv(dest/'missingness.csv',index=False)
    masked=np.full(len(tr),np.nan);importance=[];local=[];curves=[];recipes=[]
    records=[int(np.argmin(p)),int(np.argmax(p))]
    feature_cols=[c for c in tr if c not in [ID,TARGET]]
    for f in sorted(oof.fold.unique()):
        bundle=joblib.load(out/'models'/f'fold_{f}.joblib')
        bundle['best_inner_family']=s['folds'][int(f)]['best_inner_family']
        detail=dict(fold=int(f),recipe=winner,chosen=bundle['chosen'],
                    fitted_iterations={name:model.iterations for name,model in bundle['models'].items()})
        comb=bundle['neural_combiners'].get(winner) if winner.startswith('tabm_') else bundle['combiners'].get(winner)
        if comb is not None:
            detail['combiner']=dict(kind=comb.kind,calibrated=comb.calibrate,
                                    input_families=FAMILIES+(['tabm'] if winner.startswith('tabm_') else []))
            if comb.kind=='stack':
                detail['combiner'].update(logit_coefficients=comb.stack_.coef_[0].tolist(),intercept=float(comb.stack_.intercept_[0]))
            else:detail['combiner']['probability_weights']=comb.weights_.tolist()
            if comb.calibrate:
                detail['combiner']['calibration']=dict(slope=float(comb.calibrator_.coef_[0,0]),intercept=float(comb.calibrator_.intercept_[0]))
        if 'neural' in bundle:detail['neural']=dict(config=bundle['neural'].config,epochs=bundle['neural'].iterations)
        recipes.append(detail)
        idx=bundle['outer_validation'];trainidx=bundle['outer_train'];vx=tr.iloc[idx].copy();vy=y[idx]
        actual=bundle_predict(bundle,winner,vx)
        np.testing.assert_allclose(actual,p[idx],atol=1e-12)
        corrupt=vx.copy()
        for c in LABS:
            mask=(rng.random(len(vx))<.02)&corrupt[c].notna().to_numpy()
            corrupt.loc[corrupt.index[mask],c]=np.nan
        masked[idx]=bundle_predict(bundle,winner,corrupt)
        for c in feature_cols:
            changes=[]
            for repeat in range(3):
                altered=vx.copy();altered[c]=rng.permutation(altered[c].to_numpy())
                changes.append(log_loss(vy,bundle_predict(bundle,winner,altered))-log_loss(vy,actual))
            importance.append(dict(fold=int(f),feature=c,delta_logloss=float(np.mean(changes))))
        for record in records:
            if record not in idx:continue
            original=tr.iloc[[record]].copy();base=float(bundle_predict(bundle,winner,original)[0])
            for c in feature_cols:
                altered=original.copy()
                replacement=tr.iloc[trainidx][c].mode().iloc[0] if c in CATS else tr.iloc[trainidx][c].median()
                altered[c]=replacement
                changed=float(bundle_predict(bundle,winner,altered)[0])
                local.append(dict(patient_id=original[ID].iloc[0],risk='low' if record==records[0] else 'high',
                                  predicted=base,actual=int(y[record]),feature=c,observed=str(original[c].iloc[0]),
                                  reference_value=str(replacement),probability_change=changed-base))
        if learning:
            # Diagnose the dominant candidate family on each outer fold without retuning on it.
            family='tabm' if winner=='tabm' else winner if winner in FAMILIES else s['folds'][int(f)]['best_inner_family']
            config=bundle['neural'].config if family=='tabm' else bundle['chosen'][family]
            fitted=bundle['neural'] if family=='tabm' else bundle['models'][config['name']]
            for fraction in [.5,.75,1.]:
                chosen=trainidx if fraction==1 else train_test_split(trainidx,train_size=fraction,stratify=y[trainidx],random_state=321+int(f))[0]
                with threadpool_limits(limits=4):
                    if family=='tabm':
                        from .neural import fit_neural
                        model=fitted if fraction==1 else fit_neural(config,tr.iloc[chosen],y[chosen],seed=seed+int(f)*100,epochs=fitted.iterations)
                    else:
                        model=fitted if fraction==1 else fit_model(config,tr.iloc[chosen],y[chosen],seed=seed+int(f)*100,
                              threads=4,backend=json.loads((out/'backend.json').read_text())['selected'],
                              iterations=fitted.iterations if family in ['lgb','cb'] else None)
                curves.append(dict(fold=int(f),family=family,fraction=fraction,n=len(chosen),
                                   train_logloss=float(log_loss(y[chosen],model.predict(tr.iloc[chosen]))),
                                   validation_logloss=float(log_loss(vy,model.predict(vx)))))
        print(f'Report: fold {f+1} robustness, held-out importance, local sensitivity complete',flush=True)
    evidence['missingness_stress']=metrics(y,masked)
    evidence['missingness_delta']=float(log_loss(y,masked)-log_loss(y,p))
    pd.DataFrame(importance).to_csv(dest/'permutation_importance.csv',index=False)
    pd.DataFrame(local).to_csv(dest/'local_sensitivity.csv',index=False)
    pd.DataFrame(curves).to_csv(dest/'learning_curves.csv',index=False)
    dump(dest/'evidence.json',evidence)
    dump(dest/'recipe_details.json',recipes)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for label,pred in [('Selected',p),('Reference',ref)]:
        obs,expected=calibration_curve(y,pred,n_bins=10,strategy='quantile')
        axs[0,0].plot(expected,obs,'o-',label=label)
    axs[0,0].plot([0,.6],[0,.6],'k--',alpha=.4);axs[0,0].legend();axs[0,0].set(title='OOF calibration',xlabel='Predicted probability',ylabel='Observed frequency')
    comparison=pd.read_csv(out/'comparison.csv').sort_values('log_loss')
    axs[0,1].barh(comparison.method,comparison.log_loss);axs[0,1].invert_yaxis();axs[0,1].set(title='Nested OOF log loss',xlim=(comparison.log_loss.min()-.002,comparison.log_loss.max()+.002))
    imp=pd.DataFrame(importance).groupby('feature').delta_logloss.mean().nlargest(10).sort_values()
    axs[1,0].barh(imp.index,imp.values);axs[1,0].set(title='Held-out permutation importance',xlabel='Increase in log loss')
    if curves:
        curve=pd.DataFrame(curves).groupby('fraction')[['train_logloss','validation_logloss']].mean()
        axs[1,1].plot(curve.index,curve.train_logloss,'o-',label='Training');axs[1,1].plot(curve.index,curve.validation_logloss,'o-',label='Validation')
        axs[1,1].legend();axs[1,1].set(title='Learning curves (selected component)',xlabel='Fraction of outer training data',ylabel='Log loss')
    fig.savefig(dest/'diagnostics.png',dpi=160);plt.close(fig)
    make_trust_card(out,evidence,subgroup,imp)
    make_notebook(out)
    print(json.dumps(evidence,indent=2),flush=True)


def make_trust_card(out,e,subgroup,imp):
    summary=json.loads((out/'summary.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    ci=e['delta_ci95'];nwin=sum(d<0 for d in e['fold_deltas'])
    best=e['winner']
    lines=[
        '# Trust Card: ML & AI Nexus 2026',
        '\n## 1. Model Summary',
        f'Selected recipe: **{best}**. Final submission averages five outer-fold models, each trained on 80% of the supplied training set. Family and hyperparameters are selected using inner cross-validation. No pretrained models or external datasets were used.',
        '\n## 2. Validation Strategy',
        f'Five stratified outer folds (seed {manifest["seed"]}), each with three inner folds. Preprocessing, candidate selection, stopping iterations and ensemble fitting are restricted to the outer training partition. Outer labels are used for scoring only. Choosing a final recipe from several outer scores still introduces selection optimism; historical development also used these labels. This is not a pristine external validation dataset.',
        '\n## 3. Predictive Performance',
        f'Nested OOF log loss: **{e["log_loss"]:.7f}**; Brier: {e["brier"]:.7f}; ROC-AUC: {e["auc"]:.5f}; average precision: {e["average_precision"]:.5f}. At the historical illustrative threshold 0.15, sensitivity is {e["sensitivity"]:.3f} and specificity is {e["specificity"]:.3f}. This threshold is not clinically validated. Kaggle score for this new submission: **not yet available**.',
        '\n## 4. Calibration',
        f'OOF calibration slope: {e["calibration_slope"]:.4f}; intercept: {e["calibration_intercept"]:.4f}. These describe the OOF predictions and do not alter them. Calibration candidates, where used, were fitted inside outer training partitions. See diagnostics.png for reliability curves.',
        '\n## 5. Robustness and Stability',
        f'Additional random masking of 2% of observed lab/follow-up entries changed OOF log loss by {e["missingness_delta"]:+.7f}. Test missingness is higher than training missingness. Learning curves use fixed selected hyperparameters on 50%, 75%, and 100% of outer training data; they diagnose a selected component, not necessarily the whole ensemble. Fold-level metrics and effective parameters are saved with the run.',
        '\n## 6. Subgroup Reliability',
        'See report/subgroups.csv for sex, rurality, hospital type and age-group sample sizes, positive counts, prevalence, log loss with conditional bootstrap intervals, Brier and AUC. Differences may reflect case mix and sampling uncertainty; they do not establish fairness or causal discrimination.',
        subgroup[['dimension','group','n','positives','log_loss','ll_lower','ll_upper','auc']].to_csv(index=False),
        '\n## 7. Uncertainty and Human Oversight',
        f'The highest-entropy 10% ({e["referral"]["referred_n"]} rows) are referred in a retrospective simulation. Retained-set log loss is {e["referral"]["retained"]["log_loss"]:.6f}; retained prevalence {e["referral"]["retained_prevalence"]:.4f}, referred prevalence {e["referral"]["referred_prevalence"]:.4f}. Entropy measures predictive ambiguity, not epistemic uncertainty. Removing higher-risk rows changes case mix; a lower retained loss does not prove safer care or improved calibration.',
        '\n## 8. Explainability',
        'Whole-recipe held-out permutation importance identifies: '+', '.join(reversed(imp.index.tolist()))+'. Correlated predictors can share or conceal importance. Local sensitivity for one high-risk and one low-risk OOF patient replaces one raw feature with an outer-training median/mode and measures the resulting probability change. These are non-additive model sensitivities, not causal interventions or clinical recommendations. See local_sensitivity.csv. EBM missing-value effects are retained in the model; its standard visualizations may omit those effects.',
        '\n## 9. Model Comparison',
        f'Compared with the reference recipe evaluated on the same outer folds, log-loss difference is {e["delta_logloss"]:+.7f}; paired patient-bootstrap 95% interval [{ci[0]:+.7f}, {ci[1]:+.7f}]. Improvement occurs in {nwin}/5 folds. These intervals condition on fitted OOF predictions and do not capture all training/tuning uncertainty. See comparison.csv for every evaluated recipe. Historical v4 saved OOF=0.3491526 and recorded Kaggle=0.33574 use a different protocol; the historical leaderboard value is not independently verified.',
        '\n## 10. Failure Modes and Limitations',
        'The dataset is synthetic and contains only 881 positive examples. Changes in missingness, follow-up scheduling or care-pathway composition may weaken performance. Unobserved factors and synthetic-data artifacts limit transfer. Public-leaderboard adaptation and repeated development on these labels can overfit model selection. Patient IDs are excluded. No claim of causal effects, fairness guarantees or clinical safety is made.',
        '\n## 11. Deployment / Use Recommendation',
        '**Requires additional model development.** This competition model has only retrospective validation on synthetic data. External real-world validation, prospective evaluation, calibration monitoring and clinical review are required before any patient-care use. The reported scores do not justify deployment.',
        '\n## 12. Reproducibility',
        'The run manifest records package versions, source/data SHA-256 hashes, complete candidate definitions, seeds and fold counts. Per-fit checkpoints, selected fitted models, OOF/test probabilities and persistent training logs are saved. CatBoost GPU floating-point operations are nondeterministic. AI assistance: the historical repository was reported by the team as developed using Gemini and Claude; this refinement used OpenAI Codex for code, research, experiments and reporting. The team must understand and defend these choices.',
        '\nSee submission_notebook.ipynb for the exact seed, configuration and reproduction commands. Neural comparisons, when present, use randomly initialized TabM models with per-member training loss and probability averaging.',
        '\nCompetition logistics: detailed rulebook limit is five submissions before midnight and five after; notebook and Trust Card due September 14 at 08:00. The weekday on one Trust Card deadline line conflicts with the date. The official template linked on Kaggle was not supplied; this report follows the twelve headings in the provided PDF.'
    ]
    (out/'TRUST_CARD.md').write_text('\n\n'.join(lines),encoding='utf-8')


def make_notebook(out):
    import nbformat
    nb=nbformat.v4.new_notebook()
    rel=out.as_posix()
    seed=json.loads((out/'manifest.json').read_text())['seed']
    nb.cells=[nbformat.v4.new_markdown_cell('# ML & AI Nexus 2026: reproducible refinement\n\nThis notebook audits saved experimental evidence and supplies the full rerun command. Execute from the repository root using the project virtual environment. No external data or pretrained weights are used.'),
              nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd, numpy as np\nfrom sklearn.metrics import log_loss, brier_score_loss, roc_auc_score\nROOT = Path.cwd()\nwhile not (ROOT / 'train.csv').exists() and ROOT != ROOT.parent:\n    ROOT = ROOT.parent\nassert (ROOT / 'train.csv').exists(), 'Run inside the repository'\nRUN = ROOT / '"+rel+"'\nsummary = json.loads((RUN / 'summary.json').read_text())\nprint('Selected recipe:', summary['winner'])\nprint('Evaluation limitation:', summary['evaluation_note'])"),
              nbformat.v4.new_code_cell("train = pd.read_csv(ROOT / 'train.csv')\ntest = pd.read_csv(ROOT / 'test.csv')\nprint('Shapes:', train.shape, test.shape)\nprint('Positive labels:', int(train.readmitted_30d.sum()))\ndisplay(pd.read_csv(RUN / 'comparison.csv'))"),
              nbformat.v4.new_code_cell("oof = pd.read_csv(RUN / 'oof.csv')\nassert oof.patient_id.equals(train.patient_id)\np = oof[summary['winner']]\nprint('Recomputed OOF log loss:', log_loss(oof.readmitted_30d, p))\ndisplay(pd.read_csv(RUN / 'report/subgroups.csv'))"),
              nbformat.v4.new_code_cell("from IPython.display import Image, display, Markdown\ndisplay(Image(filename=str(RUN / 'report/diagnostics.png')))\ndisplay(Markdown((RUN / 'TRUST_CARD.md').read_text(encoding='utf-8')))"),
              nbformat.v4.new_code_cell("submission = pd.read_csv(RUN / 'submission_recommended.csv')\nsample = pd.read_csv(ROOT / 'sample_submission.csv')\nassert submission.columns.tolist() == sample.columns.tolist()\nassert submission.patient_id.equals(sample.patient_id)\nassert len(submission) == 3000\nassert np.isfinite(submission.readmitted_30d).all()\nassert submission.readmitted_30d.between(0, 1).all()\nprint('Submission validated; Kaggle score requires manual upload.')"),
              nbformat.v4.new_markdown_cell('## Reproduce training\nInstall `requirements-lock.txt`, then run the following cell with `RUN_TRAINING=True`. A fresh output directory recomputes everything. This can take substantial time; the default only audits existing saved evidence.'),
              nbformat.v4.new_code_cell("RUN_TRAINING = False\nif RUN_TRAINING:\n    import subprocess, sys\n    subprocess.run([sys.executable, '-u', '-m', 'nexus.train', '--seed', '"+str(seed)+"', '--config', str(RUN / 'resolved_config.json'), '--output', 'runs/reproduction', '--deadline', ''], cwd=ROOT, check=True)")]
    if (out/'neural_manifest.json').exists():
        nb.cells.append(nbformat.v4.new_markdown_cell('### Optional neural comparison\nInstall `requirements-neural.txt` as well; CUDA is required. This trains TabM from scratch using the reproduced base partitions, then combines compatible evidence. GPU numerical nondeterminism may cause small changes.'))
        nb.cells.append(nbformat.v4.new_code_cell("if RUN_TRAINING:\n    subprocess.run([sys.executable, '-u', '-m', 'nexus.neural', '--base', 'runs/reproduction', '--output', 'runs/reproduction_neural', '--deadline', ''], cwd=ROOT, check=True)\n    subprocess.run([sys.executable, 'scripts/combine_neural_evidence.py', '--base', 'runs/reproduction', '--neural', 'runs/reproduction_neural', '--output', 'runs/reproduction_combined'], cwd=ROOT, check=True)"))
    nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    nbformat.write(nb,out/'submission_notebook.ipynb')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='runs/nested_v7');ap.add_argument('--skip-learning',action='store_true')
    args=ap.parse_args();report(args.output,not args.skip_learning)
