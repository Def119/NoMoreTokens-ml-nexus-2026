"""
v3.py – Multi-Model Ensemble with Optimised Blending
=====================================================
Phase 3 from future_improvements.md: train 4 diverse model families,
calibrate each with Platt scaling, then find optimal blend weights
that minimise OOF Log Loss.

Models:
  1. Regularised LightGBM  (leaf-wise, native categoricals)
  2. CatBoost              (oblivious/symmetric trees, native categoricals)
  3. XGBoost               (level-wise trees, label-encoded categoricals)
  4. Logistic Regression    (ElasticNet, standardised + one-hot encoded)

Ensemble:
  - Calibrate each model's OOF via 5-fold Platt scaling
  - Scipy SLSQP optimisation to find blend weights minimising Log Loss
  - Final submission = weighted average of calibrated test predictions
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target = train['readmitted_30d'].copy()


# ──────────────────────────────────────────────────────────────────────
# 2. Feature Engineering (identical to v2.py)
# ──────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transforms and return df."""
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

    return df


train = engineer_features(train)
test = engineer_features(test)

# ──────────────────────────────────────────────────────────────────────
# 3. Feature Definitions
# ──────────────────────────────────────────────────────────────────────
exclude_cols = {'patient_id', 'readmitted_30d'}
all_features = [c for c in train.columns if c not in exclude_cols]

cat_cols = ['sex', 'rurality', 'hospital_type', 'region',
            'discharge_disposition', 'care_pathway']

# For tree models using native categoricals
tree_features = all_features  # all features including categoricals

# For LightGBM: convert to category dtype
train_lgb = train.copy()
test_lgb = test.copy()
for col in cat_cols:
    train_lgb[col] = train_lgb[col].astype('category')
    test_lgb[col] = test_lgb[col].astype('category')

# For CatBoost: keep as string (CatBoost handles natively)
train_cb = train.copy()
test_cb = test.copy()
for col in cat_cols:
    train_cb[col] = train_cb[col].astype(str)
    test_cb[col] = test_cb[col].astype(str)
cat_col_indices = [tree_features.index(c) for c in cat_cols]

# For XGBoost: label-encode categoricals
from sklearn.preprocessing import LabelEncoder
train_xgb = train.copy()
test_xgb = test.copy()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_xgb[col], test_xgb[col]]).astype(str))
    train_xgb[col] = le.transform(train_xgb[col].astype(str))
    test_xgb[col] = le.transform(test_xgb[col].astype(str))
    label_encoders[col] = le

# For Logistic Regression: standardise numerics + one-hot categoricals
num_features = [c for c in all_features if c not in cat_cols]

print(f"Total features: {len(all_features)} ({len(num_features)} numeric, {len(cat_cols)} categorical)")
print()

# ──────────────────────────────────────────────────────────────────────
# 4. CV Setup
# ──────────────────────────────────────────────────────────────────────
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Storage for OOF and test predictions per model
model_names = ['LightGBM', 'CatBoost', 'XGBoost', 'LogisticReg']
oof_dict = {name: np.zeros(len(train)) for name in model_names}
test_dict = {name: np.zeros(len(test)) for name in model_names}


# ──────────────────────────────────────────────────────────────────────
# 5. Model Definitions
# ──────────────────────────────────────────────────────────────────────

# --- 5a. LightGBM (regularised, same as v2) ---
lgb_params = dict(
    n_estimators=2000, learning_rate=0.018,
    num_leaves=16, max_depth=5, min_child_samples=80,
    colsample_bytree=0.75, subsample=0.80, subsample_freq=1,
    reg_alpha=0.5, reg_lambda=3.0,
    random_state=42, verbose=-1,
)

# --- 5b. CatBoost (oblivious trees, native categoricals) ---
cb_params = dict(
    iterations=2000, learning_rate=0.025,
    depth=5, l2_leaf_reg=5.0,
    random_strength=0.5, bagging_temperature=0.3,
    border_count=128,
    random_seed=42, verbose=0,
    eval_metric='Logloss', cat_features=cat_col_indices,
)

# --- 5c. XGBoost (level-wise trees) ---
xgb_params = dict(
    n_estimators=2000, learning_rate=0.020,
    max_depth=4, min_child_weight=80,
    colsample_bytree=0.75, subsample=0.80,
    reg_alpha=0.5, reg_lambda=3.0,
    random_state=42, verbosity=0,
    eval_metric='logloss', tree_method='hist',
    enable_categorical=False,  # using label-encoded
)


def print_header(title: str):
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_fold(fold, best_iter, ll, brier, auc):
    print(f"  Fold {fold} | Best Iter: {best_iter:>4} | "
          f"Log Loss: {ll:.5f} | Brier: {brier:.5f} | ROC-AUC: {auc:.5f}")


def print_oof(name, ll, brier, auc):
    print(f"  → {name} OOF Log Loss: {ll:.5f} | Brier: {brier:.5f} | ROC-AUC: {auc:.5f}")


# ──────────────────────────────────────────────────────────────────────
# 6. Training Loop – LightGBM
# ──────────────────────────────────────────────────────────────────────
print_header("Model 1/4: Regularised LightGBM")

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
    X_tr, y_tr = train_lgb.iloc[trn_idx][tree_features], target.iloc[trn_idx]
    X_va, y_va = train_lgb.iloc[val_idx][tree_features], target.iloc[val_idx]

    model = lgb.LGBMClassifier(**{**lgb_params, 'random_state': 42 + fold})
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

    val_p = model.predict_proba(X_va)[:, 1]
    oof_dict['LightGBM'][val_idx] = val_p
    test_dict['LightGBM'] += model.predict_proba(test_lgb[tree_features])[:, 1] / N_SPLITS

    print_fold(fold, getattr(model, 'best_iteration_', 'N/A'),
               log_loss(y_va, val_p), brier_score_loss(y_va, val_p), roc_auc_score(y_va, val_p))

ll = log_loss(target, oof_dict['LightGBM'])
br = brier_score_loss(target, oof_dict['LightGBM'])
au = roc_auc_score(target, oof_dict['LightGBM'])
print_oof('LightGBM', ll, br, au)
print()

# ──────────────────────────────────────────────────────────────────────
# 7. Training Loop – CatBoost
# ──────────────────────────────────────────────────────────────────────
print_header("Model 2/4: CatBoost")

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
    X_tr, y_tr = train_cb.iloc[trn_idx][tree_features], target.iloc[trn_idx]
    X_va, y_va = train_cb.iloc[val_idx][tree_features], target.iloc[val_idx]

    model_cb = CatBoostClassifier(**{**cb_params, 'random_seed': 42 + fold})
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)

    val_p = model_cb.predict_proba(X_va)[:, 1]
    oof_dict['CatBoost'][val_idx] = val_p
    test_dict['CatBoost'] += model_cb.predict_proba(test_cb[tree_features])[:, 1] / N_SPLITS

    best_iter = model_cb.get_best_iteration() if hasattr(model_cb, 'get_best_iteration') else 'N/A'
    print_fold(fold, best_iter,
               log_loss(y_va, val_p), brier_score_loss(y_va, val_p), roc_auc_score(y_va, val_p))

ll = log_loss(target, oof_dict['CatBoost'])
br = brier_score_loss(target, oof_dict['CatBoost'])
au = roc_auc_score(target, oof_dict['CatBoost'])
print_oof('CatBoost', ll, br, au)
print()

# ──────────────────────────────────────────────────────────────────────
# 8. Training Loop – XGBoost
# ──────────────────────────────────────────────────────────────────────
print_header("Model 3/4: XGBoost")

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
    X_tr, y_tr = train_xgb.iloc[trn_idx][tree_features], target.iloc[trn_idx]
    X_va, y_va = train_xgb.iloc[val_idx][tree_features], target.iloc[val_idx]

    # Fill NaN for XGBoost (it can handle NaN internally, but label-encoded cats should be int)
    model_xgb = XGBClassifier(**{**xgb_params, 'random_state': 42 + fold,
                                 'early_stopping_rounds': 100})
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  verbose=False)

    val_p = model_xgb.predict_proba(X_va)[:, 1]
    oof_dict['XGBoost'][val_idx] = val_p
    test_dict['XGBoost'] += model_xgb.predict_proba(test_xgb[tree_features])[:, 1] / N_SPLITS

    best_iter = model_xgb.best_iteration if hasattr(model_xgb, 'best_iteration') else 'N/A'
    print_fold(fold, best_iter,
               log_loss(y_va, val_p), brier_score_loss(y_va, val_p), roc_auc_score(y_va, val_p))

ll = log_loss(target, oof_dict['XGBoost'])
br = brier_score_loss(target, oof_dict['XGBoost'])
au = roc_auc_score(target, oof_dict['XGBoost'])
print_oof('XGBoost', ll, br, au)
print()

# ──────────────────────────────────────────────────────────────────────
# 9. Training Loop – Logistic Regression (ElasticNet)
# ──────────────────────────────────────────────────────────────────────
print_header("Model 4/4: Logistic Regression (ElasticNet)")

# Pre-fit one-hot encoder on full dataset for consistency
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
ohe.fit(pd.concat([train[cat_cols], test[cat_cols]]))

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
    # Numeric features: standardise per fold
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(train.iloc[trn_idx][num_features].fillna(-999))
    X_va_num = scaler.transform(train.iloc[val_idx][num_features].fillna(-999))
    X_te_num = scaler.transform(test[num_features].fillna(-999))

    # Categorical features: one-hot encode
    X_tr_cat = ohe.transform(train.iloc[trn_idx][cat_cols])
    X_va_cat = ohe.transform(train.iloc[val_idx][cat_cols])
    X_te_cat = ohe.transform(test[cat_cols])

    # Combine
    X_tr_lr = np.hstack([X_tr_num, X_tr_cat])
    X_va_lr = np.hstack([X_va_num, X_va_cat])
    X_te_lr = np.hstack([X_te_num, X_te_cat])

    y_tr = target.iloc[trn_idx]
    y_va = target.iloc[val_idx]

    lr_model = LogisticRegression(
        penalty='elasticnet', solver='saga',
        l1_ratio=0.5, C=0.5,
        max_iter=5000, random_state=42 + fold,
    )
    lr_model.fit(X_tr_lr, y_tr)

    val_p = lr_model.predict_proba(X_va_lr)[:, 1]
    oof_dict['LogisticReg'][val_idx] = val_p
    test_dict['LogisticReg'] += lr_model.predict_proba(X_te_lr)[:, 1] / N_SPLITS

    print_fold(fold, 'N/A',
               log_loss(y_va, val_p), brier_score_loss(y_va, val_p), roc_auc_score(y_va, val_p))

ll = log_loss(target, oof_dict['LogisticReg'])
br = brier_score_loss(target, oof_dict['LogisticReg'])
au = roc_auc_score(target, oof_dict['LogisticReg'])
print_oof('LogisticReg', ll, br, au)
print()

# ──────────────────────────────────────────────────────────────────────
# 10. Per-Model Summary Table
# ──────────────────────────────────────────────────────────────────────
print_header("Individual Model OOF Summary")
print(f"{'Model':<15} {'Log Loss':>10} {'Brier':>10} {'ROC-AUC':>10}")
print("-" * 50)
for name in model_names:
    ll = log_loss(target, oof_dict[name])
    br = brier_score_loss(target, oof_dict[name])
    au = roc_auc_score(target, oof_dict[name])
    print(f"{name:<15} {ll:>10.5f} {br:>10.5f} {au:>10.5f}")
print()

# ──────────────────────────────────────────────────────────────────────
# 11. Platt Calibration of each model's OOF predictions
# ──────────────────────────────────────────────────────────────────────
print_header("Platt Calibration (5-Fold on OOF)")

oof_calibrated = {}
test_calibrated = {}

for name in model_names:
    oof_raw = oof_dict[name].copy()
    test_raw = test_dict[name].copy()

    # Clip to avoid log(0) issues
    oof_raw = np.clip(oof_raw, 1e-15, 1 - 1e-15)
    test_raw = np.clip(test_raw, 1e-15, 1 - 1e-15)

    # Transform to log-odds for Platt scaling
    oof_logit = np.log(oof_raw / (1 - oof_raw)).reshape(-1, 1)
    test_logit = np.log(test_raw / (1 - test_raw)).reshape(-1, 1)

    cal_oof = np.zeros(len(train))
    cal_test = np.zeros(len(test))

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
        platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
        platt.fit(oof_logit[trn_idx], target.iloc[trn_idx])

        cal_oof[val_idx] = platt.predict_proba(oof_logit[val_idx])[:, 1]
        cal_test += platt.predict_proba(test_logit)[:, 1] / N_SPLITS

    oof_calibrated[name] = cal_oof
    test_calibrated[name] = cal_test

    ll_raw = log_loss(target, oof_dict[name])
    ll_cal = log_loss(target, cal_oof)
    print(f"  {name:<15} Raw LL: {ll_raw:.5f} → Calibrated LL: {ll_cal:.5f} (Δ {ll_cal - ll_raw:+.5f})")

print()

# ──────────────────────────────────────────────────────────────────────
# 12. Optimal Blend Weights (minimise OOF Log Loss)
# ──────────────────────────────────────────────────────────────────────
print_header("Optimising Blend Weights (scipy SLSQP)")

oof_stack = np.column_stack([oof_calibrated[name] for name in model_names])
test_stack = np.column_stack([test_calibrated[name] for name in model_names])


def blend_logloss(weights):
    """Compute log loss for a weighted blend of calibrated OOF predictions."""
    blended = np.clip(oof_stack @ weights, 1e-15, 1 - 1e-15)
    return log_loss(target, blended)


# Constraints: weights sum to 1, each weight >= 0
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
bounds = [(0.0, 1.0)] * len(model_names)
initial_weights = np.ones(len(model_names)) / len(model_names)

# Run optimiser from multiple starting points for robustness
best_result = None
best_loss = float('inf')

start_points = [
    initial_weights,
    np.array([0.4, 0.3, 0.2, 0.1]),
    np.array([0.3, 0.4, 0.2, 0.1]),
    np.array([0.25, 0.25, 0.25, 0.25]),
    np.array([0.5, 0.2, 0.2, 0.1]),
    np.array([0.2, 0.5, 0.2, 0.1]),
]

for w0 in start_points:
    result = minimize(blend_logloss, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})
    if result.fun < best_loss:
        best_loss = result.fun
        best_result = result

optimal_weights = best_result.x
print("\nOptimal Blend Weights:")
for name, w in zip(model_names, optimal_weights):
    print(f"  {name:<15} {w:.4f}")

# ──────────────────────────────────────────────────────────────────────
# 13. Final Blended OOF Evaluation
# ──────────────────────────────────────────────────────────────────────
oof_blended = np.clip(oof_stack @ optimal_weights, 1e-15, 1 - 1e-15)
test_blended = np.clip(test_stack @ optimal_weights, 1e-15, 1 - 1e-15)

oof_ll = log_loss(target, oof_blended)
oof_brier = brier_score_loss(target, oof_blended)
oof_auc = roc_auc_score(target, oof_blended)

# Reference scores
v2_ll = 0.35113
v2_auc = 0.68505
baseline_ll = 0.35529

print()
print_header("Final Ensemble OOF Results")
print(f"Blended OOF Log Loss : {oof_ll:.5f}  (v2: {v2_ll:.5f}, Δ {oof_ll - v2_ll:+.5f})")
print(f"Blended OOF Brier    : {oof_brier:.5f}")
print(f"Blended OOF ROC-AUC  : {oof_auc:.5f}  (v2: {v2_auc:.5f}, Δ {oof_auc - v2_auc:+.5f})")
print(f"Improvement vs basic.py baseline: Log Loss {oof_ll - baseline_ll:+.5f}")
print("=" * 70)

# Also report equal-weight blend as sanity check
oof_equal = np.clip(oof_stack @ initial_weights, 1e-15, 1 - 1e-15)
eq_ll = log_loss(target, oof_equal)
print(f"\n(Equal-weight blend Log Loss: {eq_ll:.5f} for reference)")

# ──────────────────────────────────────────────────────────────────────
# 14. Export
# ──────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({
    'patient_id': test['patient_id'],
    'readmitted_30d': test_blended,
})
sub.to_csv('submission_v3.csv', index=False)
print(f"\nSaved submission to submission_v3.csv (shape: {sub.shape})")

oof_df = pd.DataFrame({
    'patient_id': train['patient_id'],
    'actual': target,
    'pred_lgb': oof_calibrated['LightGBM'],
    'pred_cb': oof_calibrated['CatBoost'],
    'pred_xgb': oof_calibrated['XGBoost'],
    'pred_lr': oof_calibrated['LogisticReg'],
    'pred_blended': oof_blended,
})
oof_df.to_csv('oof_v3.csv', index=False)
print(f"Saved OOF predictions to oof_v3.csv (shape: {oof_df.shape})")

# Save weights for reproducibility
weights_df = pd.DataFrame({
    'model': model_names,
    'weight': optimal_weights,
})
weights_df.to_csv('blend_weights_v3.csv', index=False)
print(f"Saved blend weights to blend_weights_v3.csv")
