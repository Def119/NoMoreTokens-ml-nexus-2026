"""Nested, out-of-fold stack of completed campaign predictions.

This never refits a component model.  Every input probability is from a model
which did not see that patient's label; only meta-models are fitted here.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs'/'stack_v12'
ID='patient_id';TARGET='readmitted_30d';SEED=20260916
SOURCES=[
 ('final20',ROOT/'runs'/'final20_v11'/'oof.csv',ROOT/'runs'/'final20_v11'/'submission_equal_cal.csv','equal_cal'),
 ('final10',ROOT/'runs'/'final10_v9'/'oof.csv',ROOT/'runs'/'final10_v9'/'submission_equal_cal.csv','equal_cal'),
 ('smooth',ROOT/'runs'/'smooth_v8'/'oof.csv',ROOT/'runs'/'smooth_v8'/'submission_equal_cal.csv','equal_cal'),
 ('count',ROOT/'runs'/'count_v10'/'oof.csv',ROOT/'runs'/'count_v10'/'submission_selected.csv','selected'),
 ('tabm',ROOT/'runs'/'tabm_v8'/'oof.csv',ROOT/'runs'/'tabm_v8'/'submission_tabm_equal.csv','tabm_equal'),]
RECIPES=('equal_all','equal_without_tabm','convex_all','stack_c0.01','stack_c0.1','stack_c1')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clip(x):return np.clip(np.asarray(x,float),1e-7,1-1e-7)
def metrics(y,p):return dict(log_loss=float(log_loss(y,p)),brier=float(brier_score_loss(y,p)),auc=float(roc_auc_score(y,p)))

class Meta:
 def __init__(self,name):self.name=name
 def fit(self,x,y):
  if self.name=='equal_all':self.weights=np.ones(x.shape[1])/x.shape[1]
  elif self.name=='equal_without_tabm':
   self.weights=np.ones(x.shape[1]);self.weights[-1]=0;self.weights/=self.weights.sum()
  elif self.name=='convex_all':
   initial=np.ones(x.shape[1])/x.shape[1]
   def objective(w):return log_loss(y,clip(x@w))
   res=minimize(objective,initial,method='SLSQP',bounds=[(0,1)]*x.shape[1],constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-11,'maxiter':1000})
   if not res.success:raise RuntimeError(res.message)
   self.weights=np.maximum(res.x,0);self.weights/=self.weights.sum()
  else:
   c=float(self.name.rsplit('c',1)[1]);self.model=LogisticRegression(C=c,max_iter=4000).fit(logit(clip(x)),y)
  return self
 def predict(self,x):
  return clip(self.model.predict_proba(logit(clip(x)))[:,1] if hasattr(self,'model') else x@self.weights)

def load():
 train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv');sample=pd.read_csv(ROOT/'sample_submission.csv')
 assert test[ID].equals(sample[ID])
 cols=[];tcols=[];hashes={}
 for name,oof_path,sub_path,col in SOURCES:
  o=pd.read_csv(oof_path);s=pd.read_csv(sub_path)
  assert o[ID].equals(train[ID]) and o[TARGET].equals(train[TARGET]);assert s[ID].equals(sample[ID])
  cols.append(clip(o[col]));tcols.append(clip(s[TARGET]));hashes[name]=dict(oof=digest(oof_path),submission=digest(sub_path),column=col)
 return train,test,sample,np.column_stack(cols),np.column_stack(tcols),hashes

def bootstrap(y,p,base):
 d=-(y*np.log(clip(p))+(1-y)*np.log1p(-clip(p))) - (-(y*np.log(clip(base))+(1-y)*np.log1p(-clip(base))))
 rng=np.random.default_rng(SEED);z=np.array([d[rng.integers(len(d),size=len(d))].mean() for _ in range(5000)])
 return dict(delta_log_loss=float(d.mean()),ci95=np.quantile(z,[.025,.975]).tolist(),repeats=5000,seed=SEED)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'summary.json').exists():raise RuntimeError('Use a new output directory')
 train,test,sample,x,tx,hashes=load();y=train[TARGET].to_numpy();base=x[:,0]
 manifest=dict(seed=SEED,outer_folds=5,inner_folds=3,recipes=RECIPES,source_hashes=hashes,
   data={p:digest(ROOT/p) for p in ['train.csv','test.csv','sample_submission.csv']},
   note='All component probabilities are out-of-fold for their row. Meta candidate selection occurs inside each outer training partition. Components were developed adaptively, so this is not independent confirmation.')
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 splits=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(x,y));fold=np.full(len(y),-1);oof=np.full(len(y),np.nan);test_parts=[];details=[]
 for f,(ot,ov) in enumerate(splits):
  fold[ov]=f;inner=list(StratifiedKFold(3,shuffle=True,random_state=SEED+f+1).split(x[ot],y[ot]));scores={}
  for recipe in RECIPES:
   ip=np.full(len(ot),np.nan)
   for it,iv in inner:ip[iv]=Meta(recipe).fit(x[ot][it],y[ot][it]).predict(x[ot][iv])
   scores[recipe]=float(log_loss(y[ot],ip))
  chosen=min(scores,key=scores.get);model=Meta(chosen).fit(x[ot],y[ot]);oof[ov]=model.predict(x[ov]);test_parts.append(model.predict(tx))
  parameters=(getattr(model,'weights',None).tolist() if hasattr(model,'weights') else dict(coefficients=model.model.coef_[0].tolist(),intercept=float(model.model.intercept_[0])))
  detail=dict(fold=f,chosen=chosen,inner_scores=scores,outer=metrics(y[ov],oof[ov]),parameters=parameters)
  details.append(detail);joblib.dump(model,OUT/f'meta_fold_{f}.joblib');print(f'OUTER {f+1}/5 {chosen} {detail["outer"]["log_loss"]:.7f}',flush=True)
 assert np.isfinite(oof).all();pred=np.mean(test_parts,axis=0);sub=pd.DataFrame({ID:sample[ID],TARGET:clip(pred)});sub.to_csv(OUT/'submission_policy.csv',index=False)
 pd.DataFrame({ID:train[ID],TARGET:y,'fold':fold,'stack_policy':oof,'final20_equal_cal':base,**{n:x[:,i] for i,(n,*_) in enumerate(SOURCES)}}).to_csv(OUT/'oof.csv',index=False)
 summ=dict(complete=True,metrics=dict(stack_policy=metrics(y,oof),final20_equal_cal=metrics(y,base)),paired_bootstrap=bootstrap(y,oof,base),folds=details,evaluation_note=manifest['note'],test_policy='Average five outer-fitted meta models; no global outer-score recipe selection.')
 (OUT/'summary.json').write_text(json.dumps(summ,indent=2),encoding='utf-8')
 print(json.dumps(summ['metrics'],indent=2),flush=True)

if __name__=='__main__':main()
