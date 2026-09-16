import numpy as np
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


def clipped(p):
    return np.clip(np.asarray(p, dtype=float), 1e-7, 1-1e-7)


class Combiner:
    def __init__(self, kind='equal', calibrate=False):
        self.kind, self.calibrate = kind, calibrate

    def fit(self, p, y):
        p, y = clipped(p), np.asarray(y)
        self.weights_ = np.full(p.shape[1], 1/p.shape[1])
        if self.kind == 'convex':
            def loss(w):
                q=clipped(p@w)
                return -np.mean(y*np.log(q)+(1-y)*np.log1p(-q))
            def jac(w):
                q=clipped(p@w)
                return p.T@((q-y)/(q*(1-q)))/len(y)
            res=minimize(loss,self.weights_,jac=jac,method='SLSQP',bounds=[(0,1)]*p.shape[1],
                         constraints={'type':'eq','fun':lambda w:w.sum()-1}, options={'ftol':1e-11,'maxiter':1000})
            if not res.success:
                raise RuntimeError(f'Blend failed: {res.message}')
            self.weights_=np.maximum(res.x,0); self.weights_/=self.weights_.sum()
        elif self.kind == 'stack':
            self.stack_=LogisticRegression(C=0.1, max_iter=2000).fit(logit(p),y)
        if self.calibrate:
            self.calibrator_=LogisticRegression(C=1.0,max_iter=2000).fit(logit(self.raw(p)).reshape(-1,1),y)
        return self

    def raw(self,p):
        return clipped(self.stack_.predict_proba(logit(clipped(p)))[:,1] if self.kind=='stack' else clipped(p)@self.weights_)

    def predict(self,p):
        q=self.raw(p)
        return clipped(self.calibrator_.predict_proba(logit(q).reshape(-1,1))[:,1] if self.calibrate else q)
