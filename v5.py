"""
v5.py – Champion Pipeline: Targeting #1 on Kaggle Leaderboard
==============================================================
Builds on v4 (Optuna-tuned 3-model ensemble) with precision enhancements:

  1. Contextual group aggregations (fold-aware, leak-proof)
  2. Targeted chronic × lab interaction features
  3. Multi-seed averaging (3 seeds × 3 models = 9 sub-models)
  4. Logit-space blending (preserves calibration tails)
  5. Post-ensemble Platt calibration
  6. Conservative probability clipping

Uses Optuna-tuned hyperparameters from best_params_optuna.json.
"""

import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from scipy.optimize import minimize
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
# 2. Feature Engineering – v2 baseline features
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

    # ── E. NEW: Targeted Chronic × Lab Interactions ───────────────────
    # Clinically meaningful crosses between diagnoses and lab abnormalities
    df['hf_tachycardia'] = df['heart_failure'] * df['tachycardia'].fillna(0)
    df['ckd_hyponatremia'] = df['chronic_kidney_disease'] * df['hyponatremia'].fillna(0)
    df['diabetes_renal'] = df['diabetes'] * df['renal_impairment'].fillna(0)

    return df


def add_contextual_aggregations(train_fold: pd.DataFrame, val_fold: pd.DataFrame,
                                 test_df: pd.DataFrame) -> tuple:
    """
    NEW: Fold-aware contextual group aggregations (leak-proof).
    Compute group means on training fold only, then apply to val and test.
    """
    agg_configs = [
        # (new_feature_name, value_col, group_col, operation)
        ('los_rel_hospital', 'length_of_stay_days', 'hospital_type', 'ratio'),
        ('meds_rel_comorbidity', 'medication_count', 'comorbidity_count', 'ratio'),
        ('admissions_rel_pathway', 'prior_admissions_12m', 'care_pathway', 'diff'),
        ('age_rel_pathway', 'age', 'care_pathway', 'diff'),
        ('socioeconomic_rel_region', 'socioeconomic_index', 'region', 'diff'),
        ('followup_rel_pathway', 'followup_days', 'care_pathway', 'diff'),
    ]

    for feat_name, val_col, grp_col, op in agg_configs:
        # Compute mean on training fold only
        grp_means = train_fold.groupby(grp_col)[val_col].mean()

        for df in [train_fold, val_fold, test_df]:
            mapped_mean = df[grp_col].map(grp_means)
            # Fallback to global mean for unseen groups
            global_mean = train_fold[val_col].mean()
            mapped_mean = mapped_mean.fillna(global_mean)

            if op == 'ratio':
                df[feat_name] = df[val_col] / (mapped_mean + 1e-5)
            else:  # diff
                df[feat_name] = df[val_col] - mapped_mean

    return train_fold, val_fold, test_df


# Apply static features
train = engineer_features_static(train_raw)
test = engineer_features_static(test_raw)

# ──────────────────────────────────────────────────────────────────────
# 3. Feature & Model Definitions
# ──────────────────────────────────────────────────────────────────────
cat_cols = ['sex', 'rurality', 'hospital_type', 'region',
            'discharge_disposition', 'care_pathway']
exclude_cols = {'patient_id', 'readmitted_30d'}

# Static features (before contextual aggregations)
static_features = [c for c in train.columns if c not in exclude_cols]

# Contextual features will be added per-fold
contextual_feature_names = [
    'los_rel_hospital', 'meds_rel_comorbidity', 'admissions_rel_pathway',
    'age_rel_pathway', 'socioeconomic_rel_region', 'followup_rel_pathway',
]
all_features = static_features + contextual_feature_names

num_features = [c for c in all_features if c not in cat_cols]

# Pre-fit OHE for logistic regression
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
ohe.fit(pd.concat([train[cat_cols], test[cat_cols]]))

cat_col_indices = [all_features.index(c) for c in cat_cols]

print(f"Static features: {len(static_features)}")
print(f"Contextual features: {len(contextual_feature_names)}")
print(f"Total features: {len(all_features)} ({len(num_features)} numeric, {len(cat_cols)} categorical)")
print(f"New v5 features: chronic×lab interactions (3) + contextual aggregations (6) = 9 new")
print()

# ──────────────────────────────────────────────────────────────────────
# 4. Model Parameters (Optuna-tuned + multi-seed)
# ──────────────────────────────────────────────────────────────────────
SEEDS = [42, 2026, 7777]  # 3 seeds for seed averaging
N_SPLITS = 5
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


def print_header(title: str):
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_fold(fold, best_iter, ll, brier, auc):
    print(f"    Fold {fold} | Iter: {best_iter:>4} | "
          f"LL: {ll:.5f} | Brier: {brier:.5f} | AUC: {auc:.5f}")


# ──────────────────────────────────────────────────────────────────────
# 5. Training Loop – Multi-Seed, Fold-Aware Contextual Features
# ──────────────────────────────────────────────────────────────────────

# Storage: oof/test per (model_family, seed)
model_families = ['LightGBM', 'CatBoost', 'LogisticReg']
oof_all = {}   # key: (family, seed) -> np.array
test_all = {}  # key: (family, seed) -> np.array

for seed in SEEDS:
    for fam in model_families:
        oof_all[(fam, seed)] = np.zeros(len(train))
        test_all[(fam, seed)] = np.zeros(len(test))

print_header(f"Training 3 Models × {len(SEEDS)} Seeds × {N_SPLITS} Folds = {3 * len(SEEDS) * N_SPLITS} sub-models")
print()

for seed_idx, seed in enumerate(SEEDS):
    print(f"━━━ Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ━━━")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
        # --- Fold-aware contextual aggregations ---
        train_fold = train.iloc[trn_idx].copy()
        val_fold = train.iloc[val_idx].copy()
        test_fold = test.copy()

        train_fold, val_fold, test_fold = add_contextual_aggregations(
            train_fold, val_fold, test_fold
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
        vp = mdl_lgb.predict_proba(val_lgb_f[all_features])[:, 1]
        oof_all[('LightGBM', seed)][val_idx] = vp
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

        print(f"  Fold {fold} | LGB iter={lgb_iter:>4} LL={log_loss(y_va, vp):.5f} | "
              f"CB iter={cb_iter:>4} LL={log_loss(y_va, vp_cb):.5f} | "
              f"LR LL={log_loss(y_va, vp_lr):.5f}")

    print()

# ──────────────────────────────────────────────────────────────────────
# 6. Seed-Average OOF/Test Predictions per Model Family
# ──────────────────────────────────────────────────────────────────────
print_header("Seed-Averaged Individual Model Results")

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

    # Also report single-seed for comparison
    ll_s1 = log_loss(target, oof_all[(fam, SEEDS[0])])
    print(f"  {fam:<15} Seed-Avg LL: {ll:.5f} | Brier: {br:.5f} | AUC: {au:.5f}  "
          f"(single-seed LL: {ll_s1:.5f}, Δ seed-avg: {ll - ll_s1:+.5f})")

print()

# ──────────────────────────────────────────────────────────────────────
# 7. Logit-Space Blending with SLSQP Optimisation
# ──────────────────────────────────────────────────────────────────────
print_header("Logit-Space Blend Optimisation (SLSQP)")

EPS = 1e-7

# Clip and convert to logit space
oof_logits = {}
test_logits = {}
for fam in model_families:
    oof_clipped = np.clip(oof_family[fam], EPS, 1 - EPS)
    test_clipped = np.clip(test_family[fam], EPS, 1 - EPS)
    oof_logits[fam] = logit(oof_clipped)
    test_logits[fam] = logit(test_clipped)

oof_logit_stack = np.column_stack([oof_logits[fam] for fam in model_families])
test_logit_stack = np.column_stack([test_logits[fam] for fam in model_families])

# Also prepare probability-space stack for comparison
oof_prob_stack = np.column_stack([oof_family[fam] for fam in model_families])
test_prob_stack = np.column_stack([test_family[fam] for fam in model_families])


def logit_blend_logloss(weights):
    """Log Loss for logit-space weighted blend."""
    blended_logit = oof_logit_stack @ weights
    blended_prob = expit(blended_logit)
    blended_prob = np.clip(blended_prob, EPS, 1 - EPS)
    return log_loss(target, blended_prob)


def prob_blend_logloss(weights):
    """Log Loss for probability-space weighted blend (for comparison)."""
    blended = np.clip(oof_prob_stack @ weights, EPS, 1 - EPS)
    return log_loss(target, blended)


constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
bounds = [(0.0, 1.0)] * len(model_families)

# Multiple starting points
start_points = [
    np.array([1/3, 1/3, 1/3]),
    np.array([0.4, 0.3, 0.3]),
    np.array([0.3, 0.4, 0.3]),
    np.array([0.3, 0.3, 0.4]),
    np.array([0.5, 0.25, 0.25]),
    np.array([0.25, 0.5, 0.25]),
    np.array([0.25, 0.25, 0.5]),
]

# Optimise in LOGIT space
best_logit_loss = float('inf')
best_logit_w = None
for w0 in start_points:
    res = minimize(logit_blend_logloss, w0, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 2000, 'ftol': 1e-14})
    if res.fun < best_logit_loss:
        best_logit_loss = res.fun
        best_logit_w = res.x

# Optimise in PROBABILITY space for comparison
best_prob_loss = float('inf')
best_prob_w = None
for w0 in start_points:
    res = minimize(prob_blend_logloss, w0, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 2000, 'ftol': 1e-14})
    if res.fun < best_prob_loss:
        best_prob_loss = res.fun
        best_prob_w = res.x

print(f"\n  Probability-space blend:  LL = {best_prob_loss:.5f}  weights = [{', '.join(f'{w:.4f}' for w in best_prob_w)}]")
print(f"  Logit-space blend:       LL = {best_logit_loss:.5f}  weights = [{', '.join(f'{w:.4f}' for w in best_logit_w)}]")
print(f"  Logit advantage:         Δ = {best_prob_loss - best_logit_loss:+.6f}")

# Pick the better of the two
if best_logit_loss <= best_prob_loss:
    blend_mode = 'logit'
    optimal_w = best_logit_w
    raw_blend_ll = best_logit_loss
    # Compute blended predictions
    oof_blended_raw = expit(oof_logit_stack @ optimal_w)
    test_blended_raw = expit(test_logit_stack @ optimal_w)
    print(f"\n  → Using LOGIT-space blending (better by {best_prob_loss - best_logit_loss:.6f})")
else:
    blend_mode = 'prob'
    optimal_w = best_prob_w
    raw_blend_ll = best_prob_loss
    oof_blended_raw = oof_prob_stack @ optimal_w
    test_blended_raw = test_prob_stack @ optimal_w
    print(f"\n  → Using PROBABILITY-space blending (better by {best_logit_loss - best_prob_loss:.6f})")

print(f"\nOptimal Blend Weights ({blend_mode}-space):")
for fam, w in zip(model_families, optimal_w):
    print(f"  {fam:<15} {w:.4f}")

# ──────────────────────────────────────────────────────────────────────
# 8. Post-Ensemble Platt Calibration (5-Fold on blended OOF)
# ──────────────────────────────────────────────────────────────────────
print_header("Post-Ensemble Platt Calibration")

oof_blended_clipped = np.clip(oof_blended_raw, EPS, 1 - EPS)
test_blended_clipped = np.clip(test_blended_raw, EPS, 1 - EPS)

oof_blend_logit = logit(oof_blended_clipped).reshape(-1, 1)
test_blend_logit = logit(test_blended_clipped).reshape(-1, 1)

oof_calibrated = np.zeros(len(train))
test_calibrated = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
    platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    platt.fit(oof_blend_logit[trn_idx], target.iloc[trn_idx])
    oof_calibrated[val_idx] = platt.predict_proba(oof_blend_logit[val_idx])[:, 1]
    test_calibrated += platt.predict_proba(test_blend_logit)[:, 1] / N_SPLITS

    # Report calibration parameters
    slope = platt.coef_[0][0]
    intercept = platt.intercept_[0]
    print(f"  Fold {fold}: slope={slope:.4f}, intercept={intercept:.5f}")

cal_ll = log_loss(target, oof_calibrated)
raw_ll = log_loss(target, oof_blended_raw)
print(f"\n  Pre-calibration  OOF LL: {raw_ll:.5f}")
print(f"  Post-calibration OOF LL: {cal_ll:.5f} (Δ {cal_ll - raw_ll:+.6f})")

# Use calibrated if it helps, otherwise use raw
if cal_ll < raw_ll:
    oof_final = oof_calibrated
    test_final = test_calibrated
    print(f"  → Using CALIBRATED predictions (saved {raw_ll - cal_ll:.6f})")
else:
    oof_final = oof_blended_raw
    test_final = test_blended_raw
    print(f"  → Using RAW predictions (calibration hurt by {cal_ll - raw_ll:.6f})")

# ──────────────────────────────────────────────────────────────────────
# 9. Conservative Final Clipping
# ──────────────────────────────────────────────────────────────────────
CLIP_EPS = 1e-5
oof_final = np.clip(oof_final, CLIP_EPS, 1 - CLIP_EPS)
test_final = np.clip(test_final, CLIP_EPS, 1 - CLIP_EPS)

# ──────────────────────────────────────────────────────────────────────
# 10. Final Evaluation
# ──────────────────────────────────────────────────────────────────────
final_ll = log_loss(target, oof_final)
final_brier = brier_score_loss(target, oof_final)
final_auc = roc_auc_score(target, oof_final)

# Reference scores
v4_ll = 0.34915
v4_auc = 0.69443
baseline_ll = 0.35529

print()
print_header("FINAL v5 CHAMPION RESULTS")
print(f"  OOF Log Loss  : {final_ll:.5f}  (v4: {v4_ll:.5f}, Δ {final_ll - v4_ll:+.5f})")
print(f"  OOF Brier     : {final_brier:.5f}")
print(f"  OOF ROC-AUC   : {final_auc:.5f}  (v4: {v4_auc:.5f}, Δ {final_auc - v4_auc:+.5f})")
print(f"  vs Baseline   : Log Loss Δ {final_ll - baseline_ll:+.5f}")
print()

# Breakdown of improvements
print("  Improvement Breakdown:")
print(f"    v4 (Optuna tuned ensemble)          : {v4_ll:.5f}")
print(f"    + contextual aggregations (6 feat)  : ...")
print(f"    + chronic×lab interactions (3 feat)  : ...")
print(f"    + seed averaging (3 seeds)           : ...")
print(f"    + {blend_mode}-space blending             : ...")
print(f"    + post-ensemble calibration          : ...")
print(f"    = v5 FINAL                           : {final_ll:.5f} (Δ {final_ll - v4_ll:+.5f})")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 11. Export
# ──────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({
    'patient_id': test_raw['patient_id'],
    'readmitted_30d': test_final,
})
sub.to_csv('submission_v5.csv', index=False)
print(f"\nSaved submission to submission_v5.csv (shape: {sub.shape})")

oof_df = pd.DataFrame({
    'patient_id': train_raw['patient_id'],
    'actual': target,
    'pred_lgb': oof_family['LightGBM'],
    'pred_cb': oof_family['CatBoost'],
    'pred_lr': oof_family['LogisticReg'],
    'pred_blended_raw': oof_blended_raw,
    'pred_final': oof_final,
})
oof_df.to_csv('oof_v5.csv', index=False)
print(f"Saved OOF predictions to oof_v5.csv (shape: {oof_df.shape})")

# Save config for reproducibility
config = {
    'seeds': SEEDS,
    'n_splits': N_SPLITS,
    'blend_mode': blend_mode,
    'optimal_weights': {fam: float(w) for fam, w in zip(model_families, optimal_w)},
    'clip_eps': CLIP_EPS,
    'new_features': {
        'contextual_aggregations': contextual_feature_names,
        'chronic_lab_interactions': ['hf_tachycardia', 'ckd_hyponatremia', 'diabetes_renal'],
    },
    'optuna_params': best_params,
}
with open('v5_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f"Saved config to v5_config.json")
