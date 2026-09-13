"""
v6.py – Final Push: Information-Extraction Pipeline
=====================================================
Builds on v5 with strategies focused on extracting MORE SIGNAL per sample
(not more compute):

  1. Smoothed Target Encoding (fold-aware, Bayesian shrinkage, m=20)
  2. 8-Fold CV (6,125 training samples per fold vs 5,600 in 5-fold)
  3. Feature Selection (preliminary importance → drop bottom 20%)
  4. Ridge Meta-Learner Stacking (replaces SLSQP for robustness)
  5. Multi-seed averaging (carried over from v5)
  6. Post-ensemble Platt calibration (carried over from v5)

Uses Optuna-tuned hyperparameters from best_params_optuna.json.
"""

import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from scipy.special import logit, expit

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# 1. Load Data & Optuna-Tuned Parameters
# ──────────────────────────────────────────────────────────────────────
train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')
target = train_raw['readmitted_30d'].copy()

with open('best_params_optuna.json', 'r') as f:
    best_params = json.load(f)

print(f"Loaded train ({len(train_raw)}), test ({len(test_raw)}), and Optuna params.")


# ──────────────────────────────────────────────────────────────────────
# 2. Static Feature Engineering (identical to v5)
# ──────────────────────────────────────────────────────────────────────
def engineer_features_static(df: pd.DataFrame) -> pd.DataFrame:
    """Features that don't depend on training fold (no data leakage risk)."""
    df = df.copy()

    # ── A. Informative Missingness Flags ──────────────────────────────
    missing_lab_cols = [
        'hemoglobin_g_dl', 'creatinine_mg_dl', 'sodium_mmol_l',
        'heart_rate_bpm', 'systolic_bp_mmhg', 'followup_days',
    ]
    for col in missing_lab_cols:
        df[f'missing_{col}'] = df[col].isna().astype(np.int8)

    missing_flag_cols = [f'missing_{c}' for c in missing_lab_cols]
    df['total_missing_labs'] = df[missing_flag_cols].sum(axis=1).astype(np.int8)

    # ── B. Clinical Thresholds & Risk Indices ─────────────────────────
    sex_is_male = (df['sex'] == 'Male')
    hb = df['hemoglobin_g_dl']
    df['anemia_flag'] = np.where(
        hb.isna(), np.nan,
        np.where(sex_is_male, (hb < 13.0).astype(float), (hb < 12.0).astype(float))
    )

    cr = df['creatinine_mg_dl']
    df['renal_impairment'] = np.where(cr.isna(), np.nan, (cr > 1.3).astype(float))
    df['renal_ckd_interaction'] = df['renal_impairment'] * df['chronic_kidney_disease']

    sbp = df['systolic_bp_mmhg']
    df['hypertension_stage1'] = np.where(sbp.isna(), np.nan, (sbp >= 130).astype(float))
    df['hypertension_stage2'] = np.where(sbp.isna(), np.nan, (sbp >= 140).astype(float))
    df['hypertension_urgency'] = np.where(sbp.isna(), np.nan, (sbp >= 160).astype(float))

    hr = df['heart_rate_bpm']
    df['tachycardia'] = np.where(hr.isna(), np.nan, (hr > 100).astype(float))
    df['bradycardia'] = np.where(hr.isna(), np.nan, (hr < 60).astype(float))
    df['vitals_instability'] = np.where(
        hr.isna(), np.nan, ((hr > 100) | (hr < 60)).astype(float)
    )

    na_val = df['sodium_mmol_l']
    df['hyponatremia'] = np.where(na_val.isna(), np.nan, (na_val < 135).astype(float))

    # ── C. Healthcare Utilisation & Exposure Ratios ───────────────────
    df['total_bed_days'] = df['prior_admissions_12m'] * df['length_of_stay_days']
    df['comorbidity_per_age'] = df['comorbidity_count'] / (df['age'] + 1e-5)
    df['meds_per_condition'] = df['medication_count'] / (df['comorbidity_count'] + 1)

    rurality_map = {'Urban': 0, 'Semi-urban': 1, 'Rural': 2}
    rurality_ordinal = df['rurality'].map(rurality_map).fillna(1)
    df['access_barrier'] = df['missed_appointments_12m'] * rurality_ordinal
    df['missed_x_socioeconomic'] = df['missed_appointments_12m'] * df['socioeconomic_index']

    home_support = (df['discharge_disposition'] == 'Home_with_support').astype(float)
    df['discharge_vulnerability'] = home_support * (1 - df['socioeconomic_index'].clip(-2, 2) / 2)

    # ── D. Additional Engineered Features ─────────────────────────────
    chronic_cols = ['diabetes', 'hypertension', 'chronic_kidney_disease', 'heart_failure']
    df['chronic_burden'] = df[chronic_cols].sum(axis=1).astype(np.int8)

    df['age_bucket'] = pd.cut(
        df['age'], bins=[0, 40, 50, 60, 70, 80, 120],
        labels=[0, 1, 2, 3, 4, 5], right=False,
    ).astype(float)

    df['has_prior_admission'] = (df['prior_admissions_12m'] > 0).astype(np.int8)
    df['readmission_frequency'] = df['prior_admissions_12m'] / (df['age'] + 1e-5)
    df['meds_per_los'] = df['medication_count'] / (df['length_of_stay_days'] + 1e-5)

    df['followup_missing_or_long'] = np.where(
        df['followup_days'].isna(), 1,
        np.where(df['followup_days'] > 30, 1, 0)
    ).astype(np.int8)

    df['abnormal_vitals_count'] = (
        df['tachycardia'].fillna(0) + df['bradycardia'].fillna(0)
        + df['hypertension_stage1'].fillna(0) + df['hyponatremia'].fillna(0)
        + df['anemia_flag'].fillna(0) + df['renal_impairment'].fillna(0)
    ).astype(np.int8)

    df['los_x_comorbidity'] = df['length_of_stay_days'] * df['comorbidity_count']

    # ── E. Chronic × Lab Interactions (from v5) ───────────────────────
    df['hf_tachycardia'] = df['heart_failure'] * df['tachycardia'].fillna(0)
    df['ckd_hyponatremia'] = df['chronic_kidney_disease'] * df['hyponatremia'].fillna(0)
    df['diabetes_renal'] = df['diabetes'] * df['renal_impairment'].fillna(0)

    return df


# ──────────────────────────────────────────────────────────────────────
# 3. Fold-Aware Feature Generators (contextual aggs + target encoding)
# ──────────────────────────────────────────────────────────────────────
def add_contextual_aggregations(train_fold, val_fold, test_df):
    """Fold-aware contextual group aggregations (leak-proof)."""
    agg_configs = [
        ('los_rel_hospital', 'length_of_stay_days', 'hospital_type', 'ratio'),
        ('meds_rel_comorbidity', 'medication_count', 'comorbidity_count', 'ratio'),
        ('admissions_rel_pathway', 'prior_admissions_12m', 'care_pathway', 'diff'),
        ('age_rel_pathway', 'age', 'care_pathway', 'diff'),
        ('socioeconomic_rel_region', 'socioeconomic_index', 'region', 'diff'),
        ('followup_rel_pathway', 'followup_days', 'care_pathway', 'diff'),
    ]

    for feat_name, val_col, grp_col, op in agg_configs:
        grp_means = train_fold.groupby(grp_col)[val_col].mean()
        for df in [train_fold, val_fold, test_df]:
            mapped_mean = df[grp_col].map(grp_means)
            global_mean = train_fold[val_col].mean()
            mapped_mean = mapped_mean.fillna(global_mean)

            if op == 'ratio':
                df[feat_name] = df[val_col] / (mapped_mean + 1e-5)
            else:
                df[feat_name] = df[val_col] - mapped_mean

    return train_fold, val_fold, test_df


def add_target_encoding(train_fold, val_fold, test_df, y_train_fold,
                         cat_cols_to_encode, smoothing=20):
    """
    NEW v6: Bayesian smoothed target encoding.
    TE_i = (n_i * y_bar_i + m * y_bar_global) / (n_i + m)
    Computed on training fold only → applied to val and test.
    """
    global_mean = y_train_fold.mean()

    for col in cat_cols_to_encode:
        # Compute per-group stats on training fold
        stats = train_fold.groupby(col).apply(
            lambda x: pd.Series({
                'count': len(x),
                'mean': y_train_fold.loc[x.index].mean()
            })
        )

        # Bayesian smoothing: shrink toward global mean
        stats['te'] = (
            (stats['count'] * stats['mean'] + smoothing * global_mean)
            / (stats['count'] + smoothing)
        )

        te_map = stats['te'].to_dict()
        te_name = f'te_{col}'

        for df in [train_fold, val_fold, test_df]:
            df[te_name] = df[col].map(te_map).fillna(global_mean)

    return train_fold, val_fold, test_df


# Apply static features
train = engineer_features_static(train_raw)
test = engineer_features_static(test_raw)

# ──────────────────────────────────────────────────────────────────────
# 4. Feature Selection – Preliminary importance scan
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 1: Feature Importance Scan (identifying bottom 20% to prune)")
print("=" * 70)

cat_cols = ['sex', 'rurality', 'hospital_type', 'region',
            'discharge_disposition', 'care_pathway']
exclude_cols = {'patient_id', 'readmitted_30d'}
static_features = [c for c in train.columns if c not in exclude_cols]

# Quick 5-fold LightGBM scan to collect importances
scan_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
importance_acc = np.zeros(len(static_features))

train_scan = train.copy()
for col in cat_cols:
    train_scan[col] = train_scan[col].astype('category')

for fold, (trn_idx, val_idx) in enumerate(scan_skf.split(train_scan, target)):
    mdl = lgb.LGBMClassifier(
        **{**best_params['lightgbm'], 'n_estimators': 2000, 'verbose': -1,
           'random_state': 42 + fold}
    )
    mdl.fit(
        train_scan.iloc[trn_idx][static_features], target.iloc[trn_idx],
        eval_set=[(train_scan.iloc[val_idx][static_features], target.iloc[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    importance_acc += mdl.feature_importances_

# Sort and identify bottom 20% to prune
imp_df = pd.DataFrame({
    'feature': static_features,
    'importance': importance_acc,
}).sort_values('importance', ascending=False)

total_imp = imp_df['importance'].sum()
imp_df['cumulative_pct'] = imp_df['importance'].cumsum() / total_imp

# Keep top 80% by cumulative importance, but always keep categoricals
n_to_keep = (imp_df['cumulative_pct'] <= 0.97).sum()  # top 97% of importance
n_to_keep = max(n_to_keep, 30)  # keep at least 30 features

features_to_keep = set(imp_df.head(n_to_keep)['feature'].tolist())
# Always keep categoricals (needed for target encoding and model processing)
features_to_keep.update(cat_cols)

features_pruned = [f for f in static_features if f not in features_to_keep]
features_kept = [f for f in static_features if f in features_to_keep]

print(f"\nFeature importance scan complete:")
print(f"  Total static features: {len(static_features)}")
print(f"  Keeping: {len(features_kept)} (top {n_to_keep} by gain + categoricals)")
print(f"  Pruning: {len(features_pruned)} low-importance features")
if features_pruned:
    print(f"  Pruned features: {features_pruned}")

print(f"\nTop 15 features by importance:")
for i, row in imp_df.head(15).iterrows():
    print(f"  {row['feature']:40s} {row['importance']:>8.0f} ({row['cumulative_pct']*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────
# 5. Build final feature list
# ──────────────────────────────────────────────────────────────────────
# Dynamic features added per-fold
contextual_feature_names = [
    'los_rel_hospital', 'meds_rel_comorbidity', 'admissions_rel_pathway',
    'age_rel_pathway', 'socioeconomic_rel_region', 'followup_rel_pathway',
]

# Target encoding features (NEW in v6)
te_cols_to_encode = ['hospital_type', 'care_pathway', 'region', 'discharge_disposition']
te_feature_names = [f'te_{col}' for col in te_cols_to_encode]

# Final feature list = kept static + contextual + target encoded
all_features = features_kept + contextual_feature_names + te_feature_names
num_features = [c for c in all_features if c not in cat_cols]

# Pre-fit OHE for logistic regression
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
ohe.fit(pd.concat([train[cat_cols], test[cat_cols]]))

cat_col_indices = [all_features.index(c) for c in cat_cols]

print(f"\nFinal feature composition:")
print(f"  Kept static features:     {len(features_kept)}")
print(f"  Contextual aggregations:  {len(contextual_feature_names)}")
print(f"  Target encodings:         {len(te_feature_names)}")
print(f"  TOTAL:                    {len(all_features)} ({len(num_features)} numeric, {len(cat_cols)} categorical)")

# ──────────────────────────────────────────────────────────────────────
# 6. Model Parameters (Optuna-tuned)
# ──────────────────────────────────────────────────────────────────────
SEEDS = [42, 2026, 7777]
N_SPLITS = 8  # v6: upgraded from 5-fold to 8-fold
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

lgb_base_params = {
    'n_estimators': 2000,
    **best_params['lightgbm'],
    'verbose': -1,
}

cb_base_params = {
    'iterations': 2000,
    **best_params['catboost'],
    'border_count': 128,
    'eval_metric': 'Logloss',
    'cat_features': cat_col_indices,
    'verbose': 0,
}

lr_base_params = best_params['logistic_regression']

model_families = ['LightGBM', 'CatBoost', 'LogisticReg']


def print_header(title: str):
    print("=" * 70)
    print(title)
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# 7. Training Loop – 8-Fold × 3 Seeds with Target Encoding
# ──────────────────────────────────────────────────────────────────────
print()
print_header(f"STEP 2: Training {len(model_families)} Models × {len(SEEDS)} Seeds × {N_SPLITS} Folds = {len(model_families) * len(SEEDS) * N_SPLITS} sub-models")
print(f"  CV: {N_SPLITS}-Fold (v6 upgrade from 5-fold)")
print(f"  Target encoding: {te_cols_to_encode}")
print(f"  Feature pruning: {len(features_pruned)} features dropped")
print()

# Storage
oof_all = {}
test_all = {}
for seed in SEEDS:
    for fam in model_families:
        oof_all[(fam, seed)] = np.zeros(len(train))
        test_all[(fam, seed)] = np.zeros(len(test))

for seed_idx, seed in enumerate(SEEDS):
    print(f"━━━ Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ━━━")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
        # --- Fold-aware feature generation ---
        train_fold = train.iloc[trn_idx].copy()
        val_fold = train.iloc[val_idx].copy()
        test_fold = test.copy()

        # Contextual aggregations
        train_fold, val_fold, test_fold = add_contextual_aggregations(
            train_fold, val_fold, test_fold
        )

        # Target encoding (NEW in v6)
        y_train_fold = target.iloc[trn_idx]
        train_fold, val_fold, test_fold = add_target_encoding(
            train_fold, val_fold, test_fold, y_train_fold,
            te_cols_to_encode, smoothing=20
        )

        y_tr = target.iloc[trn_idx]
        y_va = target.iloc[val_idx]

        # --- LightGBM ---
        train_lgb_f = train_fold.copy()
        val_lgb_f = val_fold.copy()
        test_lgb_f = test_fold.copy()
        for col in cat_cols:
            train_lgb_f[col] = train_lgb_f[col].astype('category')
            val_lgb_f[col] = val_lgb_f[col].astype('category')
            test_lgb_f[col] = test_lgb_f[col].astype('category')

        mdl_lgb = lgb.LGBMClassifier(**{**lgb_base_params, 'random_state': seed + fold})
        mdl_lgb.fit(
            train_lgb_f[all_features], y_tr,
            eval_set=[(val_lgb_f[all_features], y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        vp_lgb = mdl_lgb.predict_proba(val_lgb_f[all_features])[:, 1]
        oof_all[('LightGBM', seed)][val_idx] = vp_lgb
        test_all[('LightGBM', seed)] += mdl_lgb.predict_proba(test_lgb_f[all_features])[:, 1] / N_SPLITS

        lgb_iter = getattr(mdl_lgb, 'best_iteration_', 'N/A')

        # --- CatBoost ---
        train_cb_f = train_fold.copy()
        val_cb_f = val_fold.copy()
        test_cb_f = test_fold.copy()
        for col in cat_cols:
            train_cb_f[col] = train_cb_f[col].astype(str)
            val_cb_f[col] = val_cb_f[col].astype(str)
            test_cb_f[col] = test_cb_f[col].astype(str)

        mdl_cb = CatBoostClassifier(**{**cb_base_params, 'random_seed': seed + fold})
        mdl_cb.fit(
            train_cb_f[all_features], y_tr,
            eval_set=(val_cb_f[all_features], y_va),
            early_stopping_rounds=100, verbose=False,
        )
        vp_cb = mdl_cb.predict_proba(val_cb_f[all_features])[:, 1]
        oof_all[('CatBoost', seed)][val_idx] = vp_cb
        test_all[('CatBoost', seed)] += mdl_cb.predict_proba(test_cb_f[all_features])[:, 1] / N_SPLITS

        cb_iter = mdl_cb.get_best_iteration() if hasattr(mdl_cb, 'get_best_iteration') else 'N/A'

        # --- Logistic Regression ---
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(train_fold[num_features].fillna(-999))
        X_va_num = scaler.transform(val_fold[num_features].fillna(-999))
        X_te_num = scaler.transform(test_fold[num_features].fillna(-999))

        X_tr_cat = ohe.transform(train_fold[cat_cols])
        X_va_cat = ohe.transform(val_fold[cat_cols])
        X_te_cat = ohe.transform(test_fold[cat_cols])

        X_tr_lr = np.hstack([X_tr_num, X_tr_cat])
        X_va_lr = np.hstack([X_va_num, X_va_cat])
        X_te_lr = np.hstack([X_te_num, X_te_cat])

        mdl_lr = LogisticRegression(
            penalty='elasticnet', solver='saga',
            C=lr_base_params['C'], l1_ratio=lr_base_params['l1_ratio'],
            max_iter=5000, random_state=seed + fold,
        )
        mdl_lr.fit(X_tr_lr, y_tr)
        vp_lr = mdl_lr.predict_proba(X_va_lr)[:, 1]
        oof_all[('LogisticReg', seed)][val_idx] = vp_lr
        test_all[('LogisticReg', seed)] += mdl_lr.predict_proba(X_te_lr)[:, 1] / N_SPLITS

        print(f"  Fold {fold}/{N_SPLITS} | LGB i={lgb_iter:>4} LL={log_loss(y_va, vp_lgb):.5f} | "
              f"CB i={cb_iter:>4} LL={log_loss(y_va, vp_cb):.5f} | "
              f"LR LL={log_loss(y_va, vp_lr):.5f}")

    print()

# ──────────────────────────────────────────────────────────────────────
# 8. Seed-Average OOF/Test Predictions
# ──────────────────────────────────────────────────────────────────────
print_header("STEP 3: Seed-Averaged Individual Model Results")

oof_family = {}
test_family = {}

for fam in model_families:
    oof_avg = np.mean([oof_all[(fam, s)] for s in SEEDS], axis=0)
    test_avg = np.mean([test_all[(fam, s)] for s in SEEDS], axis=0)
    oof_family[fam] = oof_avg
    test_family[fam] = test_avg

    ll = log_loss(target, oof_avg)
    br = brier_score_loss(target, oof_avg)
    au = roc_auc_score(target, oof_avg)
    ll_s1 = log_loss(target, oof_all[(fam, SEEDS[0])])
    print(f"  {fam:<15} Seed-Avg LL: {ll:.5f} | Brier: {br:.5f} | AUC: {au:.5f}  "
          f"(single-seed: {ll_s1:.5f}, Δ: {ll - ll_s1:+.5f})")

print()

# ──────────────────────────────────────────────────────────────────────
# 9. Ridge Meta-Learner Stacking (replaces SLSQP)
# ──────────────────────────────────────────────────────────────────────
print_header("STEP 4: Ridge Meta-Learner Stacking")

EPS = 1e-7

# Build OOF logit matrix
oof_logits = {}
test_logits = {}
for fam in model_families:
    oof_clipped = np.clip(oof_family[fam], EPS, 1 - EPS)
    test_clipped = np.clip(test_family[fam], EPS, 1 - EPS)
    oof_logits[fam] = logit(oof_clipped)
    test_logits[fam] = logit(test_clipped)

oof_logit_stack = np.column_stack([oof_logits[fam] for fam in model_families])
test_logit_stack = np.column_stack([test_logits[fam] for fam in model_families])

# Also build probability-space stacks
oof_prob_stack = np.column_stack([oof_family[fam] for fam in model_families])
test_prob_stack = np.column_stack([test_family[fam] for fam in model_families])

# --- Method A: Ridge meta-learner on logits (5-fold CV on OOF) ---
meta_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_ridge = np.zeros(len(train))
test_ridge = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(meta_skf.split(train, target), 1):
    # Fit logistic regression with L2 penalty on the 3 OOF logits
    meta_model = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs', max_iter=1000,
    )
    meta_model.fit(oof_logit_stack[trn_idx], target.iloc[trn_idx])
    oof_ridge[val_idx] = meta_model.predict_proba(oof_logit_stack[val_idx])[:, 1]
    test_ridge += meta_model.predict_proba(test_logit_stack)[:, 1] / 5

    coefs = meta_model.coef_[0]
    print(f"  Meta fold {fold}: coefs=[{', '.join(f'{c:.4f}' for c in coefs)}], "
          f"intercept={meta_model.intercept_[0]:.4f}")

ridge_ll = log_loss(target, oof_ridge)
print(f"\n  Ridge meta-learner OOF LL: {ridge_ll:.5f}")

# --- Method B: SLSQP in probability space (for comparison) ---
from scipy.optimize import minimize

def prob_blend_logloss(weights):
    blended = np.clip(oof_prob_stack @ weights, EPS, 1 - EPS)
    return log_loss(target, blended)

constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
bounds = [(0.0, 1.0)] * len(model_families)

start_points = [
    np.array([1/3, 1/3, 1/3]),
    np.array([0.4, 0.3, 0.3]),
    np.array([0.3, 0.4, 0.3]),
    np.array([0.3, 0.3, 0.4]),
    np.array([0.5, 0.25, 0.25]),
]

best_slsqp_loss = float('inf')
best_slsqp_w = None
for w0 in start_points:
    res = minimize(prob_blend_logloss, w0, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 2000, 'ftol': 1e-14})
    if res.fun < best_slsqp_loss:
        best_slsqp_loss = res.fun
        best_slsqp_w = res.x

oof_slsqp = np.clip(oof_prob_stack @ best_slsqp_w, EPS, 1 - EPS)
test_slsqp = np.clip(test_prob_stack @ best_slsqp_w, EPS, 1 - EPS)

print(f"  SLSQP prob-space OOF LL:  {best_slsqp_loss:.5f}  weights=[{', '.join(f'{w:.4f}' for w in best_slsqp_w)}]")

# Pick the better stacking method
if ridge_ll < best_slsqp_loss:
    oof_blended = oof_ridge
    test_blended = test_ridge
    stack_method = 'ridge'
    blend_ll = ridge_ll
    print(f"\n  → Using RIDGE meta-learner (better by {best_slsqp_loss - ridge_ll:.6f})")
else:
    oof_blended = oof_slsqp
    test_blended = test_slsqp
    stack_method = 'slsqp'
    blend_ll = best_slsqp_loss
    print(f"\n  → Using SLSQP prob-blend (better by {ridge_ll - best_slsqp_loss:.6f})")

# ──────────────────────────────────────────────────────────────────────
# 10. Post-Ensemble Platt Calibration
# ──────────────────────────────────────────────────────────────────────
print()
print_header("STEP 5: Post-Ensemble Platt Calibration")

oof_blend_clipped = np.clip(oof_blended, EPS, 1 - EPS)
test_blend_clipped = np.clip(test_blended, EPS, 1 - EPS)

oof_blend_logit = logit(oof_blend_clipped).reshape(-1, 1)
test_blend_logit = logit(test_blend_clipped).reshape(-1, 1)

oof_calibrated = np.zeros(len(train))
test_calibrated = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(meta_skf.split(train, target), 1):
    platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    platt.fit(oof_blend_logit[trn_idx], target.iloc[trn_idx])
    oof_calibrated[val_idx] = platt.predict_proba(oof_blend_logit[val_idx])[:, 1]
    test_calibrated += platt.predict_proba(test_blend_logit)[:, 1] / 5

    slope = platt.coef_[0][0]
    intercept = platt.intercept_[0]
    print(f"  Fold {fold}: slope={slope:.4f}, intercept={intercept:.5f}")

cal_ll = log_loss(target, oof_calibrated)
raw_ll = log_loss(target, oof_blended)
print(f"\n  Pre-calibration  OOF LL: {raw_ll:.5f}")
print(f"  Post-calibration OOF LL: {cal_ll:.5f} (Δ {cal_ll - raw_ll:+.6f})")

if cal_ll < raw_ll:
    oof_final = oof_calibrated
    test_final = test_calibrated
    print(f"  → Using CALIBRATED predictions (saved {raw_ll - cal_ll:.6f})")
else:
    oof_final = oof_blended
    test_final = test_blended
    print(f"  → Using RAW predictions (calibration hurt by {cal_ll - raw_ll:.6f})")

# ──────────────────────────────────────────────────────────────────────
# 11. Conservative Final Clipping
# ──────────────────────────────────────────────────────────────────────
CLIP_EPS = 1e-5
oof_final = np.clip(oof_final, CLIP_EPS, 1 - CLIP_EPS)
test_final = np.clip(test_final, CLIP_EPS, 1 - CLIP_EPS)

# ──────────────────────────────────────────────────────────────────────
# 12. Final Evaluation
# ──────────────────────────────────────────────────────────────────────
final_ll = log_loss(target, oof_final)
final_brier = brier_score_loss(target, oof_final)
final_auc = roc_auc_score(target, oof_final)

v5_ll = 0.34879
v5_auc = 0.69473
v4_ll = 0.34915
baseline_ll = 0.35529

print()
print_header("FINAL v6 RESULTS")
print(f"  OOF Log Loss  : {final_ll:.5f}  (v5: {v5_ll:.5f}, Δ {final_ll - v5_ll:+.5f})")
print(f"  OOF Brier     : {final_brier:.5f}")
print(f"  OOF ROC-AUC   : {final_auc:.5f}  (v5: {v5_auc:.5f}, Δ {final_auc - v5_auc:+.5f})")
print(f"  vs v4 (Optuna) : Δ {final_ll - v4_ll:+.5f}")
print(f"  vs Baseline   : Δ {final_ll - baseline_ll:+.5f}")
print()
print(f"  Enhancements applied:")
print(f"    [1] Target encoding ({len(te_feature_names)} features)     : {te_feature_names}")
print(f"    [2] 8-fold CV (vs 5-fold)                 : {N_SPLITS} folds × {len(SEEDS)} seeds = {N_SPLITS * len(SEEDS)} fold-seed combos")
print(f"    [3] Feature pruning                       : {len(features_pruned)} features dropped → {len(all_features)} total")
print(f"    [4] Stacking method                       : {stack_method}")
print(f"    [5] Post-ensemble Platt calibration        : {'used' if cal_ll < raw_ll else 'skipped (hurt)'}")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 13. Export
# ──────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({
    'patient_id': test_raw['patient_id'],
    'readmitted_30d': test_final,
})
sub.to_csv('submission_v6.csv', index=False)
print(f"\nSaved submission to submission_v6.csv (shape: {sub.shape})")

oof_df = pd.DataFrame({
    'patient_id': train_raw['patient_id'],
    'actual': target,
    'pred_lgb': oof_family['LightGBM'],
    'pred_cb': oof_family['CatBoost'],
    'pred_lr': oof_family['LogisticReg'],
    'pred_blended': oof_blended,
    'pred_final': oof_final,
})
oof_df.to_csv('oof_v6.csv', index=False)
print(f"Saved OOF predictions to oof_v6.csv (shape: {oof_df.shape})")

config = {
    'version': 'v6',
    'seeds': SEEDS,
    'n_splits': N_SPLITS,
    'stack_method': stack_method,
    'target_encoding_cols': te_cols_to_encode,
    'target_encoding_smoothing': 20,
    'features_pruned': features_pruned,
    'features_total': len(all_features),
    'slsqp_weights': {fam: float(w) for fam, w in zip(model_families, best_slsqp_w)},
    'optuna_params': best_params,
}
with open('v6_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f"Saved config to v6_config.json")
