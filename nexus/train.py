"""Deadline-aware nested evaluation and resumable artifact generation.

python -u -m nexus.train --output runs/nested_v7 --resume
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from .features import ID, TARGET
from .models import candidates, fit_model
from .ensemble import Combiner

FAMILIES=['lr','spline','lgb','cb','ebm']
METHODS=['equal','convex','stack','equal_cal','convex_cal','stack_cal']


def dump(path, value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                if path.name=='status.json': return  # Live UI status must never kill training.
                raise
            time.sleep(.1*(attempt+1))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y,p):
    return dict(log_loss=float(log_loss(y,p)), brier=float(brier_score_loss(y,p)), auc=float(roc_auc_score(y,p)))


class Logger:
    def __init__(self,out):
        self.out=out; self.start=time.monotonic()
        self.file=(out/'training.log').open('a',encoding='utf-8',buffering=1)
    def __call__(self,message,**state):
        now=datetime.now().astimezone()
        elapsed=time.monotonic()-self.start
        line=f'[{now:%H:%M:%S}] elapsed={elapsed/60:.1f}m {message}'
        print(line,flush=True); self.file.write(line+'\n')
        dump(self.out/'status.json',dict(timestamp=now.isoformat(),elapsed_seconds=elapsed,message=message,**state))


def benchmark_backend(X,y,configs,threads,log):
    c=next(c for c in configs if c['name']=='reference_cb').copy()
    c['params']=dict(iterations=100,depth=4,learning_rate=0.04)
    times={}
    for backend in ['CPU','GPU']:
        try:
            m=fit_model(c,X.iloc[:2500],y.iloc[:2500],threads=threads,backend=backend)
            times[backend]=m.seconds
            log(f'CatBoost {backend} benchmark: {m.seconds:.2f}s')
        except Exception as e:
            log(f'CatBoost {backend} unavailable: {e}')
    if not times: raise RuntimeError('No CatBoost backend works')
    backend=min(times,key=times.get)
    log(f'CatBoost backend selected: {backend}; GPU benchmark is not a validation score')
    return backend,times


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'cache').mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True)
    log=Logger(out)
    tr=pd.read_csv('train.csv');te=pd.read_csv('test.csv');sample=pd.read_csv('sample_submission.csv')
    assert tr[ID].is_unique and te[ID].is_unique
    assert te[ID].equals(sample[ID])
    X=tr.drop(columns=[TARGET]);y=tr[TARGET]
    assert set(y)=={0,1} and TARGET not in te
    best=json.loads(Path('best_params_optuna.json').read_text())
    configs=candidates(best)
    if args.config:
        spec=json.loads(Path(args.config).read_text())
        configs=spec['candidates']
    if args.smoke:
        configs=[next(c for c in configs if c['family']==f and not c['name'].startswith('reference')) for f in FAMILIES]+[c for c in configs if c['name'].startswith('reference')]
        for c in configs:
            if c['family']=='lgb': c['params']['n_estimators']=30
            if c['family']=='cb': c['params']['iterations']=30
            if c['family']=='ebm': c['params'].update(max_rounds=30,outer_bags=2)
        tr=tr.groupby(TARGET,group_keys=False).sample(frac=0.08,random_state=42).reset_index(drop=True)
        X=tr.drop(columns=[TARGET]);y=tr[TARGET]
    manifest=dict(data={f:digest(f) for f in ['train.csv','test.csv','sample_submission.csv','best_params_optuna.json']},
                  code={f:digest('nexus/'+f) for f in ['features.py','models.py','ensemble.py','train.py']},
                  candidates=configs,outer_folds=args.outer_folds,inner_folds=args.inner_folds,seed=args.seed,
                  threads=args.threads,smoke=args.smoke,
                  python=sys.version,platform=platform.platform(),
                  packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','lightgbm','catboost','interpret-core']})
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if not args.resume: raise ValueError('Existing run: use --resume or a fresh output directory')
        old=json.loads(manifest_path.read_text())
        for k in ['data','code','candidates','outer_folds','inner_folds','seed','smoke','packages']:
            if old[k]!=manifest[k]:raise ValueError(f'Resume mismatch in {k}; use a new output directory')
    else: dump(manifest_path,manifest)
    dump(out/'resolved_config.json',{'candidates':configs})
    shutil.copy2('submission_v4.csv',out/'submission_reference.csv')
    backend_path=out/'backend.json'
    if backend_path.exists():backend=json.loads(backend_path.read_text())['selected']
    else:
        backend,times=benchmark_backend(X,y,configs,args.threads,log)
        dump(backend_path,dict(selected=backend,seconds=times))
    outer=list(StratifiedKFold(args.outer_folds,shuffle=True,random_state=args.seed).split(X,y))
    fold_ids=np.zeros(len(X),int)
    for f,(_,va) in enumerate(outer): fold_ids[va]=f
    pd.DataFrame({ID:tr[ID],'fold':fold_ids}).to_csv(out/'folds.csv',index=False)
    deadline=datetime.fromisoformat(args.deadline) if args.deadline else None
    if deadline is not None and deadline.tzinfo is None: deadline=deadline.astimezone()
    all_names=FAMILIES+METHODS+['reference','selected_inner']
    oof={k:np.full(len(X),np.nan) for k in all_names}
    tests={k:[] for k in all_names}
    fold_results=[]
    measured=[]
    total_tasks=args.outer_folds*len(configs)*args.inner_folds
    completed=0
    for f,(ot,ov) in enumerate(outer):
        finish_path=out/'cache'/f'outer_{f}.joblib'
        if finish_path.exists():
            if not (out/'models'/f'fold_{f}.joblib').exists():
                raise RuntimeError(f'Fold {f} completion exists without its model bundle; use a fresh output directory')
            saved=joblib.load(finish_path)
            for name in all_names:oof[name][ov]=saved['outer'][name];tests[name].append(saved['test'][name])
            fold_results.append(saved['result']);completed+=len(configs)*args.inner_folds
            log(f'RESUME outer fold {f+1}/{args.outer_folds} complete')
            continue
        tx=X.iloc[ot].reset_index(drop=True);ty=y.iloc[ot].reset_index(drop=True)
        inner=list(StratifiedKFold(args.inner_folds,shuffle=True,random_state=args.seed+f+1).split(tx,ty))
        scores={};inner_preds={};iterations={};train_losses={}
        for ci,c in enumerate(configs):
            ip=np.full(len(tx),np.nan);its=[];tls=[]
            for j,(it,iv) in enumerate(inner):
                cache=out/'cache'/f'o{f}_{c["name"]}_i{j}.npz'
                if cache.exists():
                    z=np.load(cache);ip[iv]=z['prediction'];its.append(float(z['iterations']));tls.append(float(z['train_loss']))
                    completed+=1;continue
                estimate=max(measured[-6:],default=15)
                if deadline and (deadline-datetime.now().astimezone()).total_seconds()<estimate+180:
                    log('DEADLINE checkpoint: search stopped safely; resume supported',phase='paused')
                    return False
                eta=(total_tasks-completed)*np.mean(measured[-20:] or [15])/60
                log(f'TRAIN outer={f+1}/{args.outer_folds} trial={ci+1}/{len(configs)} {c["name"]} inner={j+1}/{args.inner_folds} estimated_remaining={eta:.1f}m',phase='training',outer_fold=f+1,trial=ci+1,model=c['name'])
                with threadpool_limits(limits=args.threads):
                    model=fit_model(c,tx.iloc[it],ty.iloc[it],seed=args.seed+f*100+j,threads=args.threads,backend=backend,
                                    validation=(tx.iloc[iv],ty.iloc[iv]) if c['family'] in ['lgb','cb'] else None)
                pred=model.predict(tx.iloc[iv]);tl=log_loss(ty.iloc[it],model.predict(tx.iloc[it]))
                ip[iv]=pred;its.append(model.iterations or np.nan);tls.append(tl);measured.append(model.seconds);completed+=1
                np.savez_compressed(cache,prediction=pred,iterations=its[-1],train_loss=tl)
                log(f'INNER RESULT {c["name"]} fold={j+1} logloss={log_loss(ty.iloc[iv],pred):.7f} train={tl:.7f} iterations={model.iterations} fit={model.seconds:.1f}s (tuning score)')
            scores[c['name']]=log_loss(ty,ip);inner_preds[c['name']]=ip
            iterations[c['name']]=int(np.nanmedian(its)) if np.isfinite(its).any() else None
            train_losses[c['name']]=float(np.mean(tls))
            log(f'CANDIDATE COMPLETE outer={f+1} {c["name"]} inner_OOF={scores[c["name"]]:.7f} best_inner={min(scores.values()):.7f}')
        chosen={fam:min([c for c in configs if c['family']==fam and not c['name'].startswith('reference')],key=lambda c:scores[c['name']]) for fam in FAMILIES}
        selected=list(chosen.values())+[c for c in configs if c['name'].startswith('reference')]
        outer_pred={};test_pred={}; fitted={}
        for c in selected:
            log(f'REFIT outer={f+1} {c["name"]} using inner-selected iterations={iterations[c["name"]]}; no outer labels passed')
            with threadpool_limits(limits=args.threads):
                model=fit_model(c,tx,ty,seed=args.seed+f*100,threads=args.threads,backend=backend,
                                iterations=iterations[c['name']] if c['family'] in ['lgb','cb'] else None)
            outer_pred[c['name']]=model.predict(X.iloc[ov]);test_pred[c['name']]=model.predict(te)
            fitted[c['name']]=model
        a=np.column_stack([inner_preds[chosen[fam]['name']] for fam in FAMILIES])
        b=np.column_stack([outer_pred[chosen[fam]['name']] for fam in FAMILIES])
        t=np.column_stack([test_pred[chosen[fam]['name']] for fam in FAMILIES])
        pp={fam:b[:,i] for i,fam in enumerate(FAMILIES)}
        tp={fam:t[:,i] for i,fam in enumerate(FAMILIES)}
        combiners={}
        for name in METHODS:
            comb=Combiner(name.replace('_cal',''),name.endswith('_cal')).fit(a,ty)
            pp[name]=comb.predict(b);tp[name]=comb.predict(t);combiners[name]=comb
        # Inner-only winner across individual families: an independently evaluated selection policy.
        best_family=min(FAMILIES,key=lambda fam:scores[chosen[fam]['name']])
        pp['selected_inner']=pp[best_family];tp['selected_inner']=tp[best_family]
        refs=[c['name'] for c in configs if c['name'].startswith('reference')]
        pp['reference']=np.mean([outer_pred[n] for n in refs],axis=0)
        tp['reference']=np.mean([test_pred[n] for n in refs],axis=0)
        result=dict(fold=f,inner_scores=scores,chosen={k:v['name'] for k,v in chosen.items()},
                    iterations=iterations,inner_train_losses=train_losses,
                    outer_metrics={name:metrics(y.iloc[ov],p) for name,p in pp.items()},best_inner_family=best_family)
        saved=dict(outer=pp,test=tp,result=result)
        joblib.dump(dict(models=fitted,chosen=chosen,combiners=combiners,outer_train=ot,outer_validation=ov),out/'models'/f'fold_{f}.joblib')
        # The completion marker is written last so resume cannot skip a fold whose model bundle is absent.
        joblib.dump(saved,finish_path)
        dump(out/f'fold_{f}_results.json',result)
        for name in all_names:oof[name][ov]=pp[name];tests[name].append(tp[name])
        fold_results.append(result)
        ranked=sorted(result['outer_metrics'].items(),key=lambda kv:kv[1]['log_loss'])
        log('OUTER HELD-OUT RESULT '+str(f+1)+': '+' | '.join(f'{n}={m["log_loss"]:.7f}' for n,m in ranked),phase='outer_complete')
    frame=pd.DataFrame({ID:tr[ID],TARGET:y,'fold':fold_ids,**oof})
    frame.to_csv(out/'oof.csv',index=False)
    summary={name:metrics(y,p) for name,p in oof.items()}
    winner=min(summary,key=lambda n:summary[n]['log_loss'])
    for name in all_names:
        pd.DataFrame({ID:te[ID],TARGET:np.mean(tests[name],axis=0)}).to_csv(out/f'submission_{name}.csv',index=False)
    # Preserve original uploaded reference separately from the freshly evaluated reference recipe.
    shutil.copy2('submission_v4.csv',out/'submission_historical_v4.csv')
    shutil.copy2(out/f'submission_{winner}.csv',out/'submission_recommended.csv')
    pd.DataFrame(summary).T.sort_values('log_loss').to_csv(out/'comparison.csv',index_label='method')
    dump(out/'summary.json',dict(winner=winner,metrics=summary,folds=fold_results,
                               evaluation_note='Each recipe is nested evaluated; choosing the best outer result introduces residual selection optimism. Historical development used these labels.',
                               complete=True,smoke=args.smoke))
    log(f'CAMPAIGN COMPLETE best={winner} nested_OOF_logloss={summary[winner]["log_loss"]:.7f}; historical saved OOF=0.3491526 is a different protocol',phase='complete',winner=winner,metrics=summary)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='runs/nested_v7');p.add_argument('--config')
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--outer-folds',type=int,default=5);p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--seed',type=int,default=20260913);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--deadline',default='2026-09-13T18:45:00+05:30')
    args=p.parse_args()
    try: run(args)
    except Exception:
        import traceback
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

if __name__=='__main__':main()
