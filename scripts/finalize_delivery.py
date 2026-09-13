"""Create the submission-ready evidence directory from the frozen 20-fold result."""
import hashlib
import json
import shutil
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs'/'final20_v11'
BASE=ROOT/'runs'/'combined_v8'
DEST=ROOT/'runs'/'final_delivery_v11'
ID='patient_id'; TARGET='readmitted_30d'


def copy(source, target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def main():
    assert json.loads((RUN/'summary.json').read_text())['complete']
    if DEST.exists() and any(DEST.iterdir()):
        raise RuntimeError(f'Use an empty output directory: {DEST}')
    DEST.mkdir(parents=True)
    summary=json.loads((RUN/'summary.json').read_text())
    train=pd.read_csv(ROOT/'train.csv');sample=pd.read_csv(ROOT/'sample_submission.csv')
    oof=pd.read_csv(RUN/'oof.csv');sub=pd.read_csv(RUN/'submission_equal_cal.csv')
    assert oof[ID].equals(train[ID]) and oof[TARGET].equals(train[TARGET])
    assert sub.columns.tolist()==sample.columns.tolist() and sub[ID].equals(sample[ID])
    assert len(sub)==3000 and np.isfinite(sub[TARGET]).all() and sub[TARGET].between(0,1,inclusive='neither').all()
    p=oof.equal_cal.to_numpy();y=oof[TARGET].to_numpy()
    calibration=LogisticRegression(C=1e6,max_iter=2000).fit(logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1),y)
    evidence=dict(recipe='fixed five-family equal-weight ensemble with inner-Oof sigmoid calibration',
                  outer_folds=20,inner_folds=3,model_training_fraction=.95,
                  log_loss=float(log_loss(y,p)),brier=float(brier_score_loss(y,p)),auc=float(roc_auc_score(y,p)),
                  average_precision=float(average_precision_score(y,p)),
                  calibration_slope=float(calibration.coef_[0,0]),calibration_intercept=float(calibration.intercept_[0]),
                  paired_bootstrap=summary['paired_bootstrap'],evaluation_note=summary['evaluation_note'],
                  submission_sha256=hashlib.sha256((RUN/'submission_equal_cal.csv').read_bytes()).hexdigest())
    (DEST/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    for name in ['manifest.json','resolved_config.json','summary.json','folds.csv']:
        copy(RUN/name,DEST/name)
    copy(RUN/'submission_equal_cal.csv',DEST/'submission_recommended.csv')
    copy(RUN/'submission_equal.csv',DEST/'submission_uncalibrated_fallback.csv')
    copy(ROOT/'submission_v4.csv',DEST/'submission_historical_v4.csv')
    comparison=pd.DataFrame([
        dict(recipe='20-fold fixed equal_cal (recommended)',log_loss=evidence['log_loss'],training_fraction=.95,notes='20 outer folds; calibration trained inside each outer training partition'),
        dict(recipe='10-fold fixed equal_cal',log_loss=0.3489490319796503,training_fraction=.90,notes='same frozen five components'),
        dict(recipe='5-fold selected equal_cal',log_loss=0.3491452963957503,training_fraction=.80,notes='same primary campaign; component diagnostics source'),
        dict(recipe='TabM comparison',log_loss=0.3493042,training_fraction=.80,notes='trained from scratch; not selected'),
        dict(recipe='count-effect logistic',log_loss=0.34989240511763553,training_fraction=.80,notes='separate fresh partition; not directly comparable'),
        dict(recipe='historical saved v4 OOF',log_loss=0.3491526,training_fraction=np.nan,notes='different, biased historical protocol; contextual only'),
    ])
    comparison.to_csv(DEST/'comparison.csv',index=False)
    # These diagnostics are deliberately labelled as component diagnostics: they were generated
    # for the related 5-fold base recipe and must not be presented as a separate final20 analysis.
    shutil.copytree(BASE/'report',DEST/'component_diagnostics')
    ci=summary['paired_bootstrap']['final10_equal_cal']['equal_cal']['ci95']
    delta=summary['paired_bootstrap']['final10_equal_cal']['equal_cal']['delta_log_loss']
    trust=f'''# Trust Card: ML & AI Nexus 2026

## 1. Model Summary

Selected submission: a fixed, equal-weight five-component ensemble (regularized logistic regression, cubic-spline logistic regression, LightGBM, CatBoost and EBM), calibrated by a sigmoid logistic model. Twenty outer models are averaged; each is trained on 95% of the supplied rows. No pretrained models or external data were used.

## 2. Validation Strategy

The final campaign used 20 stratified outer folds, seed 20260914, with three inner folds inside each outer-training partition. Preprocessing, category vocabularies, numeric medians, feature representations, tree stopping iterations and calibration are learned only from the relevant training partition. The five component configurations and calibration C=1 were frozen from the preceding fixed 10-fold campaign. Outer labels score only their held-out rows.

## 3. Predictive Performance

Nested OOF log loss: **{evidence['log_loss']:.7f}**; Brier: {evidence['brier']:.7f}; ROC-AUC: {evidence['auc']:.5f}; average precision: {evidence['average_precision']:.5f}. This is a local validation value, not a Kaggle score. Kaggle score: **not yet available; manual upload required**.

## 4. Calibration

OOF calibration slope: {evidence['calibration_slope']:.4f}; intercept: {evidence['calibration_intercept']:.4f}. Each fold's calibrator is fit on inner OOF probabilities from its outer-training rows. Calibration is not clinical validation.

## 5. Robustness and Stability

Fold metrics, exact fold assignments, configurations, package versions, data hashes, inner checkpoints and fitted models are saved in the final20_v11 run. The 20-fold score is {delta:+.7f} versus the related 10-fold calibrated recipe; its conditional paired-bootstrap 95% interval is [{ci[0]:+.7f}, {ci[1]:+.7f}], which includes zero. The improvement is not a claim of statistical superiority.

## 6. Subgroup Reliability

Subgroup counts and component-level subgroup metrics are in `component_diagnostics/subgroups.csv`. They are from the related five-fold component evaluation, not a substitute for dedicated final20 subgroup validation. Differences may reflect case mix and sampling uncertainty; they do not establish fairness.

## 7. Uncertainty and Human Oversight

The uncertainty interval conditions on saved OOF predictions and excludes training/tuning uncertainty, overlapping-fold dependence and adaptive research choices. Entropy-referral diagnostics in `component_diagnostics/evidence.json` are retrospective component analyses only; they do not prove safety.

## 8. Explainability

`component_diagnostics/permutation_importance.csv` and `local_sensitivity.csv` explain the related five-fold component recipe. They are non-causal model sensitivities, not explanations of every final20 ensemble prediction. Correlated predictors can share or conceal importance.

## 9. Model Comparison

See `comparison.csv`. The 20-fold recipe is selected as the lowest completed local OOF score. The 20-fold expansion was requested after inspecting earlier results, so the score is adaptively selected and not independent confirmation. Historical v4 OOF and reported Kaggle values use a different, biased protocol and are contextual only.

## 10. Failure Modes and Limitations

This is a synthetic dataset with 881 positive labels. Higher test missingness, changes in care pathways, feature distributions or missingness mechanisms may weaken performance. IDs are excluded. No causal, fairness or clinical-safety claim is made.

## 11. Deployment / Use Recommendation

**Requires additional model development.** This competition model has retrospective synthetic-data validation only. External validation, prospective evaluation, calibration monitoring and clinical review are required before patient-care use.

## 12. Reproducibility

`manifest.json` records data/source hashes, package versions, seed, fixed configurations and fold design. `summary.json` records all fold metrics and bootstrap comparisons. `submission_recommended.csv` has exactly 3,000 rows in sample-submission ID order; SHA-256: `{evidence['submission_sha256']}`. AI assistance: historical code was reported by the team as Gemini/Claude-generated; this refinement used OpenAI Codex. The team must understand and defend the work.
'''
    (DEST/'TRUST_CARD.md').write_text(trust,encoding='utf-8')
    nb=nbformat.v4.new_notebook()
    rel='runs/final_delivery_v11'
    nb.cells=[nbformat.v4.new_markdown_cell('# ML & AI Nexus 2026: final submission audit\n\nAudits the frozen 20-fold final artifact. The Kaggle upload is manual.'),
              nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd, numpy as np\nfrom sklearn.metrics import log_loss, brier_score_loss, roc_auc_score\nROOT=Path.cwd()\nwhile not (ROOT/'train.csv').exists() and ROOT!=ROOT.parent: ROOT=ROOT.parent\nRUN=ROOT/'"+rel+"'\nsummary=json.loads((RUN/'summary.json').read_text()); evidence=json.loads((RUN/'evidence.json').read_text())\nprint('Recipe:',evidence['recipe']); print('OOF log loss:',evidence['log_loss']); print(summary['evaluation_note'])"),
              nbformat.v4.new_code_cell("train=pd.read_csv(ROOT/'train.csv'); oof=pd.read_csv(ROOT/'runs/final20_v11/oof.csv')\nassert oof.patient_id.equals(train.patient_id)\nprint('Recomputed:',log_loss(oof.readmitted_30d,oof.equal_cal),brier_score_loss(oof.readmitted_30d,oof.equal_cal),roc_auc_score(oof.readmitted_30d,oof.equal_cal))\ndisplay(pd.read_csv(RUN/'comparison.csv'))"),
              nbformat.v4.new_code_cell("submission=pd.read_csv(RUN/'submission_recommended.csv'); sample=pd.read_csv(ROOT/'sample_submission.csv')\nassert submission.columns.tolist()==sample.columns.tolist() and submission.patient_id.equals(sample.patient_id)\nassert len(submission)==3000 and np.isfinite(submission.readmitted_30d).all() and submission.readmitted_30d.between(0,1).all()\nprint('Submission validated; upload manually to Kaggle.')"),
              nbformat.v4.new_code_cell("from IPython.display import display, Markdown, Image\ndisplay(Markdown((RUN/'TRUST_CARD.md').read_text()))\ndisplay(Image(filename=str(RUN/'component_diagnostics/diagnostics.png')))"),
              nbformat.v4.new_markdown_cell('## Reproduce\nInstall `requirements-lock.txt` and run `python -u -m nexus.final_refit20` from the repository root. This recomputes the frozen 20-fold experiment; GPU CatBoost may not be bitwise deterministic.')]
    nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    nbformat.write(nb,DEST/'submission_notebook.ipynb')
    print(DEST)


if __name__=='__main__':main()
