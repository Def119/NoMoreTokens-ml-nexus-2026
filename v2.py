"""
v2.py – Feature-Engineered LightGBM with Regularized Trees
==========================================================
Builds on basic.py by implementing feature engineering ideas from
future_improvements.md (Phase 1 A/B/C) and tree regularisation (Phase 2).

Feature engineering groups:
  A. Informative missingness flags  (MNAR signals)
  B. Clinical thresholds & risk indices
  C. Healthcare utilisation & exposure ratios
  +  Additional engineered features (age buckets, chronic burden, vitals composite)

Tree regularisation:
  - Shallower trees (num_leaves=16, max_depth=5)
  - Higher min_child_samples (80)
  - Feature & row subsampling
  - L1/L2 penalties
  - Slower learning rate with more boosting rounds
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

warnings.filterwarnings('ignore', category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target = train['readmitted_30d'].copy()


# ──────────────────────────────────────────────────────────────────────
# 2. Feature Engineering (applied identically to train & test)
# ──────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transforms in-place and return df."""
    df = df.copy()

    # ── A. Informative Missingness Flags ──────────────────────────────
    # In clinical data, missingness is MNAR – tests are ordered only
    # when clinically indicated, so a missing lab carries signal.
    missing_lab_cols = [
        'hemoglobin_g_dl', 'creatinine_mg_dl', 'sodium_mmol_l',
        'heart_rate_bpm', 'systolic_bp_mmhg', 'followup_days',
    ]
    for col in missing_lab_cols:
        df[f'missing_{col}'] = df[col].isna().astype(np.int8)

    # Aggregate count of missing labs per patient
    missing_flag_cols = [f'missing_{c}' for c in missing_lab_cols]
    df['total_missing_labs'] = df[missing_flag_cols].sum(axis=1).astype(np.int8)

    # ── B. Clinical Thresholds & Risk Indices ─────────────────────────

    # Anemia indicator (sex-specific thresholds)
    #   Female: hemoglobin < 12.0,  Male: hemoglobin < 13.0
    sex_is_male = (df['sex'] == 'Male')
    hb = df['hemoglobin_g_dl']
    df['anemia_flag'] = np.where(
        hb.isna(), np.nan,
        np.where(sex_is_male, (hb < 13.0).astype(float), (hb < 12.0).astype(float))
    )

    # Renal impairment marker: creatinine > 1.3
    cr = df['creatinine_mg_dl']
    df['renal_impairment'] = np.where(cr.isna(), np.nan, (cr > 1.3).astype(float))

    # Renal impairment × CKD interaction
    df['renal_ckd_interaction'] = df['renal_impairment'] * df['chronic_kidney_disease']

    # Hypertension staging
    sbp = df['systolic_bp_mmhg']
    df['hypertension_stage1'] = np.where(sbp.isna(), np.nan, (sbp >= 130).astype(float))
    df['hypertension_stage2'] = np.where(sbp.isna(), np.nan, (sbp >= 140).astype(float))
    df['hypertension_urgency'] = np.where(sbp.isna(), np.nan, (sbp >= 160).astype(float))

    # Tachycardia / bradycardia (vitals instability)
    hr = df['heart_rate_bpm']
    df['tachycardia'] = np.where(hr.isna(), np.nan, (hr > 100).astype(float))
    df['bradycardia'] = np.where(hr.isna(), np.nan, (hr < 60).astype(float))
    df['vitals_instability'] = np.where(
        hr.isna(), np.nan,
        ((hr > 100) | (hr < 60)).astype(float)
    )

    # Electrolyte disturbance: hyponatremia (sodium < 135)
    na_val = df['sodium_mmol_l']
    df['hyponatremia'] = np.where(na_val.isna(), np.nan, (na_val < 135).astype(float))

    # ── C. Healthcare Utilisation & Exposure Ratios ───────────────────

    # Cumulative hospital exposure
    df['total_bed_days'] = df['prior_admissions_12m'] * df['length_of_stay_days']

    # Comorbidity density (burden relative to age)
    df['comorbidity_per_age'] = df['comorbidity_count'] / (df['age'] + 1e-5)

    # Polypharmacy risk ratio
    df['meds_per_condition'] = df['medication_count'] / (df['comorbidity_count'] + 1)

    # Access / non-adherence barrier  (missed appts × rurality interaction)
    # Encode rurality as ordinal: Urban=0, Semi-urban=1, Rural=2
    rurality_map = {'Urban': 0, 'Semi-urban': 1, 'Rural': 2}
    rurality_ordinal = df['rurality'].map(rurality_map).fillna(1)  # default mid
    df['access_barrier'] = df['missed_appointments_12m'] * rurality_ordinal

    # Missed appointments × socioeconomic index interaction
    df['missed_x_socioeconomic'] = df['missed_appointments_12m'] * df['socioeconomic_index']

    # Discharge vulnerability:
    #   Home_with_support combined with low socioeconomic index
    home_support = (df['discharge_disposition'] == 'Home_with_support').astype(float)
    df['discharge_vulnerability'] = home_support * (1 - df['socioeconomic_index'].clip(-2, 2) / 2)

    # ── D. Additional Engineered Features ─────────────────────────────

    # Total chronic conditions (sum of 4 binary diagnosis flags)
    chronic_cols = ['diabetes', 'hypertension', 'chronic_kidney_disease', 'heart_failure']
    df['chronic_burden'] = df[chronic_cols].sum(axis=1).astype(np.int8)

    # Age buckets (binned for non-linear risk thresholds)
    df['age_bucket'] = pd.cut(
        df['age'],
        bins=[0, 40, 50, 60, 70, 80, 120],
        labels=[0, 1, 2, 3, 4, 5],
        right=False,
    ).astype(float)

    # Prior admissions intensity (any prior admission flag + count interaction)
    df['has_prior_admission'] = (df['prior_admissions_12m'] > 0).astype(np.int8)
    df['readmission_frequency'] = df['prior_admissions_12m'] / (df['age'] + 1e-5)

    # Medication load relative to LOS
    df['meds_per_los'] = df['medication_count'] / (df['length_of_stay_days'] + 1e-5)

    # Follow-up gap risk: missing or very long follow-up is risky
    df['followup_missing_or_long'] = np.where(
        df['followup_days'].isna(), 1,
        np.where(df['followup_days'] > 30, 1, 0)
    ).astype(np.int8)

    # Vitals composite score (z-score-free: just count of abnormal vitals)
    df['abnormal_vitals_count'] = (
        df['tachycardia'].fillna(0)
        + df['bradycardia'].fillna(0)
        + df['hypertension_stage1'].fillna(0)
        + df['hyponatremia'].fillna(0)
        + df['anemia_flag'].fillna(0)
        + df['renal_impairment'].fillna(0)
    ).astype(np.int8)

    # LOS × comorbidity interaction
    df['los_x_comorbidity'] = df['length_of_stay_days'] * df['comorbidity_count']

    # ── Convert categoricals to category dtype ────────────────────────
    cat_cols = ['sex', 'rurality', 'hospital_type', 'region',
                'discharge_disposition', 'care_pathway']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    return df


# Apply to both splits
train = engineer_features(train)
test = engineer_features(test)

# Define feature list (exclude identifiers and target)
exclude_cols = {'patient_id', 'readmitted_30d'}
features = [c for c in train.columns if c not in exclude_cols]

print(f"Total features after engineering: {len(features)}")
print(f"  Original raw features: 23")
print(f"  Engineered features:   {len(features) - 23}")
print()

# ──────────────────────────────────────────────────────────────────────
# 3. Regularised LightGBM Hyperparameters (Phase 2)
# ──────────────────────────────────────────────────────────────────────
lgb_params = dict(
    n_estimators=2000,
    learning_rate=0.018,
    num_leaves=16,
    max_depth=5,
    min_child_samples=80,
    colsample_bytree=0.75,
    subsample=0.80,
    subsample_freq=1,
    reg_alpha=0.5,
    reg_lambda=3.0,
    random_state=42,
    verbose=-1,
)

# ──────────────────────────────────────────────────────────────────────
# 4. Validation Framework
# ──────────────────────────────────────────────────────────────────────
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof = np.zeros(len(train))
preds = np.zeros(len(test))

print("=" * 70)
print("Starting 5-Fold Stratified CV  (v2: Feature Engineering + Regularised LightGBM)")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 5. Training Loop
# ──────────────────────────────────────────────────────────────────────
fold_scores = []
for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
    X_train, y_train = train.iloc[trn_idx][features], target.iloc[trn_idx]
    X_val, y_val = train.iloc[val_idx][features], target.iloc[val_idx]

    model = lgb.LGBMClassifier(**{**lgb_params, 'random_state': 42 + fold})

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    oof[val_idx] = val_preds
    preds += model.predict_proba(test[features])[:, 1] / N_SPLITS

    fold_ll = log_loss(y_val, val_preds)
    fold_brier = brier_score_loss(y_val, val_preds)
    fold_auc = roc_auc_score(y_val, val_preds)
    best_iter = getattr(model, 'best_iteration_', 'N/A')

    fold_scores.append((fold_ll, fold_brier, fold_auc))
    print(
        f"Fold {fold} | Best Iter: {best_iter:>4} | "
        f"Log Loss: {fold_ll:.5f} | Brier: {fold_brier:.5f} | "
        f"ROC-AUC: {fold_auc:.5f}"
    )

# ──────────────────────────────────────────────────────────────────────
# 6. Overall OOF Evaluation
# ──────────────────────────────────────────────────────────────────────
oof_ll = log_loss(target, oof)
oof_brier = brier_score_loss(target, oof)
oof_auc = roc_auc_score(target, oof)

# Compare against baseline
baseline_ll = 0.35529
baseline_auc = 0.66819

print("=" * 70)
print(f"Overall OOF Log Loss : {oof_ll:.5f}  (baseline: {baseline_ll:.5f}, delta: {oof_ll - baseline_ll:+.5f})")
print(f"Overall OOF Brier    : {oof_brier:.5f}")
print(f"Overall OOF ROC-AUC  : {oof_auc:.5f}  (baseline: {baseline_auc:.5f}, delta: {oof_auc - baseline_auc:+.5f})")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 7. Feature Importance (top 25)
# ──────────────────────────────────────────────────────────────────────
print("\nTop 25 Feature Importances (gain, last fold):")
imp = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)
for i, row in imp.head(25).iterrows():
    print(f"  {row['feature']:40s} {row['importance']:>6}")

# ──────────────────────────────────────────────────────────────────────
# 8. Export Submissions & OOF Predictions
# ──────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({
    'patient_id': test['patient_id'],
    'readmitted_30d': preds,
})
sub.to_csv('submission_v2.csv', index=False)
print(f"\nSaved submission to submission_v2.csv (shape: {sub.shape})")

oof_df = pd.DataFrame({
    'patient_id': train['patient_id'],
    'actual': target,
    'pred_prob': oof,
})
oof_df.to_csv('oof_v2.csv', index=False)
print(f"Saved OOF predictions to oof_v2.csv (shape: {oof_df.shape})")
