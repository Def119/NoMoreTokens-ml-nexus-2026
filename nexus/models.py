"""Small-data candidate library. All fits accept only their own training labels."""
import copy
import time
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from .features import CATS, FrameAdapter, engineer, linear_preprocessor


def candidates(best):
    configs = []
    def add(name, family, mode='raw', **params):
        configs.append(dict(name=name, family=family, mode=mode, params=params))
    for c in [0.01, 0.1, 1.0]:
        add(f'lr_raw_{c}', 'lr', C=c, l1_ratio=0.0)
    for c in [0.03, 0.3, 3.0]:
        add(f'lr_engineered_{c}', 'lr', 'engineered', C=c, l1_ratio=0.5)
    for knots in [3, 5]:
        for c in [0.01, 0.1, 1.0]:
            add(f'spline_k{knots}_c{c}', 'spline', knots=knots, C=c, l1_ratio=0.0)
    for mode in ['raw','engineered']:
        for depth, leaves, child, reg in [(2,4,100,5),(3,8,80,10),(4,16,120,15)]:
            add(f'lgb_{mode}_d{depth}', 'lgb', mode, learning_rate=0.025, n_estimators=2000,
                max_depth=depth, num_leaves=leaves, min_child_samples=child, reg_lambda=reg,
                colsample_bytree=0.85, subsample=0.85, subsample_freq=1)
        for depth, reg in [(3,5),(4,10),(5,20)]:
            add(f'cb_{mode}_d{depth}', 'cb', mode, iterations=1600, depth=depth,
                learning_rate=0.035, l2_leaf_reg=reg, random_strength=1,
                bootstrap_type='Bayesian', bagging_temperature=0.5)
    for interactions in [0, 5]:
        add(f'ebm_i{interactions}', 'ebm', interactions=interactions, outer_bags=4,
            max_bins=128, max_interaction_bins=16, learning_rate=0.025,
            smoothing_rounds=200, max_rounds=4000, min_samples_leaf=30)
    add('reference_lgb', 'lgb', 'engineered', **best['lightgbm'], n_estimators=2000, subsample_freq=0)
    add('reference_cb', 'cb', 'engineered', **best['catboost'], iterations=2000, border_count=128)
    add('reference_lr', 'lr', 'engineered', **best['logistic_regression'], legacy=True)
    return configs


class FittedModel:
    def __init__(self, adapter, preprocessor, model, config, iterations, seconds):
        self.adapter, self.preprocessor, self.model = adapter, preprocessor, model
        self.config, self.iterations, self.seconds = config, iterations, seconds

    def transform(self, X):
        x = self.adapter.transform(X)
        return self.preprocessor.transform(x) if self.preprocessor is not None else x

    def predict(self, X):
        return np.clip(self.model.predict_proba(self.transform(X))[:,1], 1e-7, 1-1e-7)


def fit_model(config, X, y, *, seed=42, threads=4, backend='CPU', validation=None, iterations=None):
    """Outer evaluation calls this with iterations from inner CV, never outer labels."""
    start = time.monotonic()
    c = copy.deepcopy(config)
    p = c['params'].copy()
    fam = c['family']
    adapter = FrameAdapter(fam, c['mode']).fit(X)
    xt = adapter.transform(X)
    xv = adapter.transform(validation[0]) if validation is not None else None
    pre = None
    n_iter = None
    if fam in ['lr', 'spline']:
        pre = linear_preprocessor(xt, spline=fam=='spline', knots=p.pop('knots',4), legacy=p.pop('legacy',False))
        xt = pre.fit_transform(xt)
        model = LogisticRegression(solver='saga' if p.get('l1_ratio',0)>0 else 'lbfgs',
                                   max_iter=4000, random_state=seed, **p)
        model.fit(xt, y)
        n_iter = int(np.max(model.n_iter_))
    elif fam == 'lgb':
        if iterations is not None:
            p['n_estimators'] = int(iterations)
        model = lgb.LGBMClassifier(**p, random_state=seed, n_jobs=threads, verbosity=-1)
        kw = {} if validation is None else dict(eval_X=xv, eval_y=validation[1], callbacks=[lgb.early_stopping(100,verbose=False)])
        model.fit(xt, y, **kw)
        n_iter = int(model.best_iteration_ or model.n_estimators_)
    elif fam == 'cb':
        if iterations is not None:
            p['iterations'] = int(iterations)
        model = CatBoostClassifier(**p, task_type=backend, cat_features=CATS, loss_function='Logloss',
                                   random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False)
        kw = {} if validation is None else dict(eval_set=(xv,validation[1]), early_stopping_rounds=100)
        model.fit(xt,y,**kw)
        n_iter = int(model.tree_count_)
    elif fam == 'ebm':
        from interpret.glassbox import ExplainableBoostingClassifier
        model = ExplainableBoostingClassifier(**p, random_state=seed, n_jobs=threads)
        model.fit(xt, y)
    else:
        raise ValueError(f'Unknown family {fam}')
    return FittedModel(adapter, pre, model, c, n_iter, time.monotonic()-start)
