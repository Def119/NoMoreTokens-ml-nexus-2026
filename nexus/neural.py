"""Optional TabM trained from scratch, evaluated inside the existing outer partitions.

Uses the base campaign's *inner* predictions for combination fitting. It never uses
global OOF predictions as training features for an outer-fold meta model.
"""
import argparse
import copy
import importlib.metadata
import json
import time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from tabm import TabM
from rtdl_num_embeddings import LinearReLUEmbeddings
from .features import ID, TARGET, CATS, LABS
from .ensemble import Combiner
from .train import Logger, dump, digest, metrics, FAMILIES

BINARY=['diabetes','hypertension','chronic_kidney_disease','heart_failure']


class NeuralPreprocessor:
    def fit(self,X):
        self.num_cols=[c for c in X if c not in [ID,TARGET]+CATS+BINARY]
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(X[self.num_cols])
        self.scaler=StandardScaler().fit(self.imputer.transform(X[self.num_cols]))
        self.encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1).fit(self.categorical(X))
        self.cardinalities=[len(c)+1 for c in self.encoder.categories_]
        return self

    def categorical(self,X):
        frame=X[CATS+BINARY].astype(str).copy()
        for c in LABS:frame['missing_'+c]=X[c].isna().astype(str)
        return frame

    def transform(self,X):
        num=self.scaler.transform(self.imputer.transform(X[self.num_cols])).astype('float32')
        cat=(self.encoder.transform(self.categorical(X))+1).astype('int64')
        return num,cat


class NeuralModel:
    def __init__(self,prep,model,config,epochs):
        self.prep,self.model,self.config,self.iterations=prep,model.cpu(),config,epochs

    def predict(self,X):
        num,cat=self.prep.transform(X)
        self.model.eval()
        with torch.no_grad():
            values=[self.model(torch.from_numpy(num[i:i+512]),torch.from_numpy(cat[i:i+512])).squeeze(-1).sigmoid().mean(1).numpy() for i in range(0,len(X),512)]
        return np.clip(np.concatenate(values),1e-7,1-1e-7)


def fit_neural(config,X,y,seed,validation=None,epochs=None,log=None):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
    prep=NeuralPreprocessor().fit(X)
    num,cat=prep.transform(X)
    p=config['params']
    model=TabM.make(n_num_features=num.shape[1],cat_cardinalities=prep.cardinalities,d_out=1,k=16,
                    n_blocks=2,d_block=128,dropout=p['dropout'],
                    num_embeddings=LinearReLUEmbeddings(num.shape[1],d_embedding=8) if p['embeddings'] else None).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=p['weight_decay'])
    nt=torch.from_numpy(num).cuda();ct=torch.from_numpy(cat).cuda();yt=torch.tensor(np.asarray(y),dtype=torch.float32,device='cuda')
    if validation is not None:
        vn,vc=prep.transform(validation[0]);vn=torch.from_numpy(vn).cuda();vc=torch.from_numpy(vc).cuda()
    best=float('inf');best_epoch=0;best_state=None;start=time.monotonic()
    for epoch in range(1,(epochs or 180)+1):
        model.train()
        order=torch.randperm(len(X),device='cuda')
        for idx in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            logits=model(nt[idx],ct[idx]).squeeze(-1)
            # Separate losses per ensemble member, as specified by the TabM authors.
            loss=F.binary_cross_entropy_with_logits(logits,yt[idx,None].expand_as(logits))
            loss.backward();optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                pred=torch.cat([model(vn[i:i+512],vc[i:i+512]).squeeze(-1).sigmoid().mean(1) for i in range(0,len(vn),512)]).cpu().numpy()
            score=log_loss(validation[1],pred)
            if score<best-1e-6:
                best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
            if epoch%20==0 and log:log(f'TabM epoch={epoch} inner_logloss={score:.7f} best={best:.7f} elapsed_fit={time.monotonic()-start:.1f}s')
            if epoch-best_epoch>=25:break
    if validation is not None:model.load_state_dict(best_state)
    fitted=NeuralModel(prep,model,config,best_epoch if validation is not None else epochs)
    del nt,ct,yt,optimizer
    torch.cuda.empty_cache()
    return fitted


def predict_bundle(bundle,method,X):
    from .report import bundle_predict
    q=bundle['neural'].predict(X)
    if method=='tabm':return q
    base=bundle['base_bundle']
    matrix=np.column_stack([bundle_predict(base,f,X) for f in FAMILIES]+[q])
    return bundle['neural_combiners'][method].predict(matrix)


def run(base,out,deadline='2026-09-13T18:45:00+05:30'):
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'models').mkdir(exist_ok=True);(out/'cache').mkdir(exist_ok=True)
    logger=Logger(out)
    assert torch.cuda.is_available(), 'CUDA-enabled PyTorch required'
    logger('TabM device='+torch.cuda.get_device_name(0)+'; weights randomly initialized; no pretraining')
    source=json.loads((base/'manifest.json').read_text());summary=json.loads((base/'summary.json').read_text())
    assert summary.get('complete') and not source.get('smoke'), 'A complete full-data base run is required'
    for filename,expected in source['data'].items():
        assert digest(filename)==expected, f'Current data differs from base run: {filename}'
    for filename in ['features.py','models.py','ensemble.py']:
        assert digest(Path(__file__).parent/filename)==source['code'][filename], f'Base model source changed: {filename}'
    dependencies=[base/'summary.json',base/'folds.csv']
    for f,fold in enumerate(summary['folds']):
        dependencies.append(base/'models'/f'fold_{f}.joblib')
        for name in fold['chosen'].values():
            dependencies.extend(base/'cache'/f'o{f}_{name}_i{j}.npz' for j in range(source['inner_folds']))
    candidates=[dict(name='tabm_plain',family='tabm',mode='raw',params=dict(embeddings=False,dropout=.15,weight_decay=.01)),
                dict(name='tabm_embedded',family='tabm',mode='raw',params=dict(embeddings=True,dropout=.1,weight_decay=.001))]
    manifest=dict(base=str(base.resolve()),source_manifest_sha256=digest(base/'manifest.json'),
                                base_artifacts={str(p.relative_to(base)):digest(p) for p in dependencies},
                                code_sha256=digest(__file__),packages={p:importlib.metadata.version(p) for p in ['torch','tabm','rtdl-num-embeddings']},candidates=candidates,seed=source['seed'])
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text())==manifest, 'Neural checkpoint manifest mismatch; use a fresh output directory'
    else:dump(out/'manifest.json',manifest)
    tr=pd.read_csv('train.csv');te=pd.read_csv('test.csv');folds=pd.read_csv(base/'folds.csv')
    assert folds[ID].equals(tr[ID]), 'Base fold rows are not aligned with current training IDs'
    assert te[ID].equals(pd.read_csv('sample_submission.csv')[ID]), 'Submission ID order mismatch'
    oof=pd.DataFrame({ID:tr[ID],TARGET:tr[TARGET],'fold':folds.fold})
    methods=['tabm','tabm_convex','tabm_equal','tabm_stack'];test={m:[] for m in methods};results=[]
    fit_times=[];completed=0;total=source['outer_folds']*len(candidates)*source['inner_folds']
    for f in sorted(folds.fold.unique()):
        finish=out/'cache'/f'fold_{f}.joblib'
        if finish.exists():
            saved=joblib.load(finish);completed+=len(candidates)*source['inner_folds']
        else:
            b=joblib.load(base/'models'/f'fold_{f}.joblib');ot=b['outer_train'];ov=b['outer_validation']
            tx=tr.iloc[ot].reset_index(drop=True);ty=tx[TARGET]
            inner=list(StratifiedKFold(source['inner_folds'],shuffle=True,random_state=source['seed']+int(f)+1).split(tx,ty))
            ps={};eps={}
            for c in candidates:
                ip=np.zeros(len(tx));es=[]
                for j,(it,iv) in enumerate(inner):
                    file=out/'cache'/f'o{f}_{c["name"]}_i{j}.npz'
                    if file.exists():
                        z=np.load(file);ip[iv]=z['prediction'];es.append(int(z['epochs']));completed+=1;continue
                    if deadline and (datetime.fromisoformat(deadline)-datetime.now().astimezone()).total_seconds()<300:
                        logger('Neural search cutoff: completed checkpoints retained');return
                    eta=(total-completed)*np.mean(fit_times or [60])/60
                    logger(f'TRAIN TabM outer={f+1}/5 candidate={c["name"]} inner={j+1}/3 estimated_remaining={eta:.1f}m plus refits')
                    start=time.monotonic()
                    model=fit_neural(c,tx.iloc[it],ty.iloc[it],source['seed']+int(f)*100+j,
                                     validation=(tx.iloc[iv],ty.iloc[iv]),log=logger)
                    ip[iv]=model.predict(tx.iloc[iv]);es.append(model.iterations)
                    fit_times.append(time.monotonic()-start);completed+=1
                    np.savez_compressed(file,prediction=ip[iv],epochs=model.iterations)
                    logger(f'INNER RESULT {c["name"]} logloss={log_loss(ty.iloc[iv],ip[iv]):.7f} best_epoch={model.iterations}')
                ps[c['name']]=ip;eps[c['name']]=int(np.median(es))
            selected=min(candidates,key=lambda c:log_loss(ty,ps[c['name']]))
            logger(f'REFIT TabM outer={f+1} selected={selected["name"]} epochs={eps[selected["name"]]}')
            model=fit_neural(selected,tx,ty,source['seed']+int(f)*100,epochs=eps[selected['name']])
            outerp=model.predict(tr.iloc[ov]);testp=model.predict(te)
            # Reconstruct exactly the inner prediction rows of the base campaign.
            bi=[]
            for fam in FAMILIES:
                ip=np.zeros(len(tx));name=b['chosen'][fam]['name']
                for j,(_,iv) in enumerate(inner):ip[iv]=np.load(base/'cache'/f'o{f}_{name}_i{j}.npz')['prediction']
                bi.append(ip)
            from .report import bundle_predict
            a=np.column_stack(bi+[ps[selected['name']]])
            v=np.column_stack([bundle_predict(b,fam,tr.iloc[ov]) for fam in FAMILIES]+[outerp])
            t=np.column_stack([bundle_predict(b,fam,te) for fam in FAMILIES]+[testp])
            pp={'tabm':outerp};tp={'tabm':testp};combiners={}
            for name in methods[1:]:
                comb=Combiner(name.replace('tabm_','')).fit(a,ty);pp[name]=comb.predict(v);tp[name]=comb.predict(t);combiners[name]=comb
            result=dict(fold=int(f),selected=selected,epochs=eps,inner_scores={n:float(log_loss(ty,p)) for n,p in ps.items()},
                        outer_metrics={n:metrics(tr[TARGET].iloc[ov],p) for n,p in pp.items()})
            saved=dict(outer=pp,test=tp,result=result,indices=ov)
            joblib.dump(dict(base_bundle=b,neural=model,neural_combiners=combiners,outer_train=ot,outer_validation=ov),out/'models'/f'fold_{f}.joblib')
            joblib.dump(saved,finish)
        for m in methods:oof.loc[saved['indices'],m]=saved['outer'][m];test[m].append(saved['test'][m])
        results.append(saved['result'])
        logger('OUTER HELD-OUT RESULT '+str(f+1)+': '+json.dumps(saved['result']['outer_metrics']))
    stats={m:metrics(tr[TARGET],oof[m]) for m in methods};winner=min(stats,key=lambda m:stats[m]['log_loss'])
    oof.to_csv(out/'oof.csv',index=False)
    for m in methods:pd.DataFrame({ID:te[ID],TARGET:np.mean(test[m],axis=0)}).to_csv(out/f'submission_{m}.csv',index=False)
    pd.DataFrame(stats).T.sort_values('log_loss').to_csv(out/'comparison.csv',index_label='method')
    dump(out/'summary.json',dict(winner=winner,metrics=stats,folds=results,complete=True,base_run=str(base)))
    logger(f'TABM COMPLETE best={winner} nested_OOF={stats[winner]["log_loss"]:.7f}')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='runs/nested_v7');ap.add_argument('--output',default='runs/tabm_v7')
    ap.add_argument('--deadline',default='2026-09-13T18:45:00+05:30')
    # Import under the package name so persisted NeuralModel objects remain loadable.
    from nexus.neural import run as run_imported
    args=ap.parse_args();run_imported(args.base,args.output,args.deadline)
