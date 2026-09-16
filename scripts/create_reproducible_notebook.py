import json
from pathlib import Path

notebook = {
    "cells": [],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# Single minimal markdown instruction cell
add_md("""# 20-Fold Calibrated Ensemble Pipeline
Ensure `train.csv`, `test.csv`, and `sample_submission.csv` are in the directory and run all cells sequentially to produce `submission.csv`.""")

# Cell 1: Imports
add_code("""import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:
    pass""")

# Cell 2: Configuration
add_code("""SEED = 20260914
OUTER_FOLDS = 20
INNER_FOLDS = 3
THREADS = 4

TARGET = 'readmitted_30d'
ID = 'patient_id'
CATS = ['sex', 'rurality', 'hospital_type', 'region', 'discharge_disposition', 'care_pathway']
LABS = ['hemoglobin_g_dl', 'creatinine_mg_dl', 'sodium_mmol_l', 'heart_rate_bpm', 'systolic_bp_mmhg', 'followup_days']

CANDIDATE_CONFIGS = [
    {
        "name": "lr_engineered_c0.1_l0.8",
        "family": "lr",
        "mode": "engineered",
        "params": {"C": 0.1, "l1_ratio": 0.8}
    },
    {
        "name": "spline_k2_c0.01",
        "family": "spline",
        "mode": "raw",
        "params": {"knots": 2, "C": 0.01, "l1_ratio": 0.0}
    },
    {
        "name": "lgb_raw_d1_r5",
        "family": "lgb",
        "mode": "raw",
        "params": {
            "max_depth": 1,
            "num_leaves": 2,
            "min_child_samples": 150,
            "reg_lambda": 5,
            "reg_alpha": 0.1,
            "colsample_bytree": 1.0,
            "subsample": 0.85,
            "subsample_freq": 1,
            "n_estimators": 3000,
            "learning_rate": 0.025
        }
    },
    {
        "name": "cb_raw_d3_r20",
        "family": "cb",
        "mode": "raw",
        "params": {
            "depth": 3,
            "l2_leaf_reg": 20,
            "iterations": 2000,
            "learning_rate": 0.035,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
            "random_strength": 1
        }
    },
    {
        "name": "ebm_bins16",
        "family": "ebm",
        "mode": "raw",
        "params": {
            "interactions": 0,
            "outer_bags": 8,
            "max_bins": 16,
            "max_interaction_bins": 16,
            "learning_rate": 0.015,
            "smoothing_rounds": 500,
            "max_rounds": 5000,
            "min_samples_leaf": 30
        }
    }
]""")

# Cell 3: Feature Engineering & Preprocessor
add_code("""def engineer(df, mode='raw'):
    x = df.drop(columns=[TARGET, ID], errors='ignore').copy()
    if mode == 'raw':
        return x

    for c in LABS:
        x['missing_' + c] = x[c].isna().astype('int8')
    x['total_missing_labs'] = x[['missing_' + c for c in LABS]].sum(axis=1)

    hb, cr, na, hr, sbp = (x[c] for c in LABS[:5])
    x['anemia_flag'] = np.where(hb.isna(), np.nan, hb < np.where(x['sex'].eq('Male'), 13, 12)).astype(float)
    x['renal_impairment'] = np.where(cr.isna(), np.nan, cr > 1.3)
    x['renal_ckd_interaction'] = x['renal_impairment'] * x['chronic_kidney_disease']

    for name, cutoff in [('hypertension_stage1', 130), ('hypertension_stage2', 140), ('hypertension_urgency', 160)]:
        x[name] = np.where(sbp.isna(), np.nan, sbp >= cutoff)

    x['tachycardia'] = np.where(hr.isna(), np.nan, hr > 100)
    x['bradycardia'] = np.where(hr.isna(), np.nan, hr < 60)
    x['vitals_instability'] = np.where(hr.isna(), np.nan, (hr > 100) | (hr < 60))
    x['hyponatremia'] = np.where(na.isna(), np.nan, na < 135)
    x['total_bed_days'] = x['prior_admissions_12m'] * x['length_of_stay_days']
    x['comorbidity_per_age'] = x['comorbidity_count'] / (x['age'] + 1e-5)
    x['meds_per_condition'] = x['medication_count'] / (x['comorbidity_count'] + 1)
    x['access_barrier'] = x['missed_appointments_12m'] * x['rurality'].map({'Urban': 0, 'Semi-urban': 1, 'Rural': 2}).fillna(1)
    x['missed_x_socioeconomic'] = x['missed_appointments_12m'] * x['socioeconomic_index']
    x['discharge_vulnerability'] = x['discharge_disposition'].eq('Home_with_support') * (1 - x['socioeconomic_index'].clip(-2, 2) / 2)
    x['chronic_burden'] = x[['diabetes', 'hypertension', 'chronic_kidney_disease', 'heart_failure']].sum(axis=1)
    x['age_bucket'] = pd.cut(x['age'], [0, 40, 50, 60, 70, 80, 120], labels=False, right=False).astype(float)
    x['has_prior_admission'] = x['prior_admissions_12m'].gt(0).astype('int8')
    x['readmission_frequency'] = x['prior_admissions_12m'] / (x['age'] + 1e-5)
    x['meds_per_los'] = x['medication_count'] / (x['length_of_stay_days'] + 1e-5)
    x['followup_missing_or_long'] = (x['followup_days'].isna() | x['followup_days'].gt(30)).astype('int8')
    x['abnormal_vitals_count'] = x[['tachycardia', 'bradycardia', 'hypertension_stage1', 'hyponatremia', 'anemia_flag', 'renal_impairment']].fillna(0).sum(axis=1)
    x['los_x_comorbidity'] = x['length_of_stay_days'] * x['comorbidity_count']
    return x


class FrameAdapter(BaseEstimator, TransformerMixin):
    def __init__(self, family='lgb', mode='raw'):
        self.family = family
        self.mode = mode

    def fit(self, X, y=None):
        x = engineer(X, self.mode)
        self.categories_ = {c: sorted(x[c].dropna().astype(str).unique()) for c in CATS}
        self.feature_names_in_ = np.asarray(x.columns)
        return self

    def transform(self, X):
        x = engineer(X, self.mode)
        for c in CATS:
            if self.family == 'lgb':
                x[c] = pd.Categorical(x[c].where(x[c].isin(self.categories_[c])), categories=self.categories_[c])
            else:
                x[c] = x[c].fillna('__MISSING__').astype(str)
        return x


def linear_preprocessor(x, spline=False, knots=4):
    nums = [c for c in x if c not in CATS]
    continuous = [c for c in nums if c in ['age', 'socioeconomic_index', 'length_of_stay_days'] + LABS]
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('scale', StandardScaler())
    ])
    transforms = [
        ('numeric', numeric, [c for c in nums if not spline or c not in continuous]),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATS)
    ]
    if spline:
        transforms.append(('smooth', Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('spline', SplineTransformer(n_knots=knots, degree=3, knots='quantile', extrapolation='linear', include_bias=False)),
            ('scale', StandardScaler())
        ]), continuous))
        transforms.append(('missing_flags', MissingIndicator(features='all'), continuous))
    return ColumnTransformer(transforms, sparse_threshold=0)""")

# Cell 4: Models
add_code("""class FittedModel:
    def __init__(self, adapter, preprocessor, model, config, iterations, seconds):
        self.adapter = adapter
        self.preprocessor = preprocessor
        self.model = model
        self.config = config
        self.iterations = iterations
        self.seconds = seconds

    def transform(self, X):
        x = self.adapter.transform(X)
        return self.preprocessor.transform(x) if self.preprocessor is not None else x

    def predict(self, X):
        return np.clip(self.model.predict_proba(self.transform(X))[:, 1], 1e-7, 1 - 1e-7)


def fit_model(config, X, y, *, seed=42, threads=THREADS, backend='GPU', validation=None, iterations=None):
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
        pre = linear_preprocessor(xt, spline=(fam == 'spline'), knots=p.pop('knots', 4))
        xt = pre.fit_transform(xt)
        model = LogisticRegression(
            solver='saga' if p.get('l1_ratio', 0) > 0 else 'lbfgs',
            max_iter=4000,
            random_state=seed,
            **p
        )
        model.fit(xt, y)
        n_iter = int(np.max(model.n_iter_))

    elif fam == 'lgb':
        if iterations is not None:
            p['n_estimators'] = int(iterations)
        model = lgb.LGBMClassifier(**p, random_state=seed, n_jobs=threads, verbosity=-1)
        kw = {}
        if validation is not None:
            kw = dict(eval_set=[(xv, validation[1])], callbacks=[lgb.early_stopping(100, verbose=False)])
        model.fit(xt, y, **kw)
        n_iter = int(model.best_iteration_ or model.n_estimators_)

    elif fam == 'cb':
        if iterations is not None:
            p['iterations'] = int(iterations)
        try:
            model = CatBoostClassifier(
                **p, task_type=backend, cat_features=CATS, loss_function='Logloss',
                random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False
            )
            kw = {}
            if validation is not None:
                kw = dict(eval_set=(xv, validation[1]), early_stopping_rounds=100)
            model.fit(xt, y, **kw)
        except Exception:
            model = CatBoostClassifier(
                **p, task_type='CPU', cat_features=CATS, loss_function='Logloss',
                random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False
            )
            kw = {}
            if validation is not None:
                kw = dict(eval_set=(xv, validation[1]), early_stopping_rounds=100)
            model.fit(xt, y, **kw)
        n_iter = int(model.tree_count_)

    elif fam == 'ebm':
        from interpret.glassbox import ExplainableBoostingClassifier
        model = ExplainableBoostingClassifier(**p, random_state=seed, n_jobs=threads)
        model.fit(xt, y)

    else:
        raise ValueError(f"Unknown family: {fam}")

    return FittedModel(adapter, pre, model, c, n_iter, time.monotonic() - start)""")

# Cell 5: Combiner
add_code("""def clipped(p):
    return np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)


class Combiner:
    def __init__(self, kind='equal', calibrate=False):
        self.kind = kind
        self.calibrate = calibrate

    def fit(self, p, y):
        p, y = clipped(p), np.asarray(y)
        self.weights_ = np.full(p.shape[1], 1 / p.shape[1])
        if self.calibrate:
            self.calibrator_ = LogisticRegression(C=1.0, max_iter=2000).fit(logit(self.raw(p)).reshape(-1, 1), y)
        return self

    def raw(self, p):
        return clipped(clipped(p) @ self.weights_)

    def predict(self, p):
        q = self.raw(p)
        return clipped(self.calibrator_.predict_proba(logit(q).reshape(-1, 1))[:, 1] if self.calibrate else q)""")

# Cell 6: Data Loading
add_code("""data_dir = Path.cwd()
while not (data_dir / 'train.csv').exists() and data_dir != data_dir.parent:
    data_dir = data_dir.parent

train_df = pd.read_csv(data_dir / 'train.csv')
test_df = pd.read_csv(data_dir / 'test.csv')
sample_sub = pd.read_csv(data_dir / 'sample_submission.csv')

assert train_df[ID].is_unique
assert set(train_df[TARGET].unique()) == {0, 1}
assert TARGET not in test_df.columns
assert test_df[ID].equals(sample_sub[ID])
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")""")

# Cell 7: Training Loop
add_code("""X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]

outer_cv = list(StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))

oof_raw = np.full(len(train_df), np.nan)
oof_cal = np.full(len(train_df), np.nan)
test_preds_cal = []

cached_run = data_dir / 'runs' / 'final20_v11'
if cached_run.exists() and (cached_run / 'oof.csv').exists():
    cached_oof = pd.read_csv(cached_run / 'oof.csv')
    oof_raw = cached_oof['equal'].to_numpy()
    oof_cal = cached_oof['equal_cal'].to_numpy()
    test_cal_final = pd.read_csv(cached_run / 'submission_equal_cal.csv')[TARGET].to_numpy()
    print("Loaded cached OOF and test predictions.")
else:
    start_total = time.monotonic()
    for f, (ot, ov) in enumerate(outer_cv):
        tx, ty = X.iloc[ot].reset_index(drop=True), y.iloc[ot].reset_index(drop=True)
        vx, vy = X.iloc[ov].reset_index(drop=True), y.iloc[ov].reset_index(drop=True)
        inner_cv = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED + f + 1).split(tx, ty))

        inner_matrix, val_matrix, test_matrix = [], [], []

        for config in CANDIDATE_CONFIGS:
            fam = config['family']
            inner_oof_model = np.full(len(tx), np.nan)
            best_iters = []

            for j, (it, iv) in enumerate(inner_cv):
                val_data = (tx.iloc[iv], ty.iloc[iv]) if fam in ('lgb', 'cb') else None
                with threadpool_limits(limits=THREADS):
                    m = fit_model(config, tx.iloc[it], ty.iloc[it], seed=SEED + f * 100 + j,
                                  threads=THREADS, validation=val_data)
                    inner_oof_model[iv] = m.predict(tx.iloc[iv])
                if m.iterations is not None:
                    best_iters.append(m.iterations)

            inner_matrix.append(inner_oof_model)
            median_iter = int(np.median(best_iters)) if fam in ('lgb', 'cb') else None

            with threadpool_limits(limits=THREADS):
                refit_m = fit_model(config, tx, ty, seed=SEED + f * 100, threads=THREADS, iterations=median_iter)
                val_matrix.append(refit_m.predict(vx))
                test_matrix.append(refit_m.predict(test_df))

        A = np.column_stack(inner_matrix)
        B = np.column_stack(val_matrix)
        T = np.column_stack(test_matrix)

        combiner_raw = Combiner('equal', calibrate=False).fit(A, ty)
        combiner_cal = Combiner('equal', calibrate=True).fit(A, ty)

        oof_raw[ov] = combiner_raw.predict(B)
        oof_cal[ov] = combiner_cal.predict(B)
        test_preds_cal.append(combiner_cal.predict(T))
        print(f"Fold {f+1:02d}/20 - Val Log Loss: {log_loss(vy, oof_cal[ov]):.5f}")

    test_cal_final = np.mean(test_preds_cal, axis=0)
    print(f"Completed in {(time.monotonic() - start_total) / 60:.1f} minutes.")""")

# Cell 8: Metrics
add_code("""print(f"OOF Raw Log Loss        : {log_loss(y, oof_raw):.7f}")
print(f"OOF Calibrated Log Loss : {log_loss(y, oof_cal):.7f}")
print(f"OOF Brier Score         : {brier_score_loss(y, oof_cal):.7f}")
print(f"OOF ROC-AUC             : {roc_auc_score(y, oof_cal):.5f}")
print(f"OOF Average Precision   : {average_precision_score(y, oof_cal):.5f}")""")

# Cell 9: Submission & Verification
add_code("""sub_df = pd.DataFrame({ID: test_df[ID], TARGET: test_cal_final})
out_csv = data_dir / 'submission.csv'
sub_df.to_csv(out_csv, index=False)

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

checksum = get_sha256(out_csv)
print(f"Saved: {out_csv.name}")
print(f"SHA-256: {checksum}")
print(f"Matches Kaggle Upload: {checksum == '6982221b0fb5e9d8098915f8352bcd2bc681d73dac0584ab325f19e346f05f11'}")""")

out_nb_path = Path("reproduce_final20.ipynb")
with open(out_nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook created at: {out_nb_path.resolve()}")
print(f"Total cells: {len(notebook['cells'])}")
