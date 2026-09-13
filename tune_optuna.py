"""
tune_optuna.py – Bayesian Hyperparameter Tuning via Optuna
==========================================================
Tunes hyperparameters for the top contributors of the v3 ensemble:
  1. LightGBM (47% blend weight in v3)
  2. Logistic Regression ElasticNet (40% blend weight in v3)
  3. CatBoost (13% blend weight in v3)

Uses the exact same 54 engineered features and 5-Fold Stratified CV splits
(random_state=42) to minimize Out-of-Fold Binary Log Loss.

Results are saved to best_params_optuna.json.
"""

import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Optuna is not installed. Please run: pip install optuna")

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
# 1. Feature Engineering (Identical to v2.py and v3.py)
# ──────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Informative Missingness Flags
    missing_lab_cols = [
        'hemoglobin_g_dl', 'creatinine_mg_dl', 'sodium_mmol_l',
        'heart_rate_bpm', 'systolic_bp_mmhg', 'followup_days',
    ]
    for col in missing_lab_cols:
        df[f'missing_{col}'] = df[col].isna().astype(np.int8)

    missing_flag_cols = [f'missing_{c}' for c in missing_lab_cols]
    df['total_missing_labs'] = df[missing_flag_cols].sum(axis=1).astype(np.int8)

    # Clinical Thresholds & Risk Indices
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
        hr.isna(), np.nan,
        ((hr > 100) | (hr < 60)).astype(float)
    )

    na_val = df['sodium_mmol_l']
    df['hyponatremia'] = np.where(na_val.isna(), np.nan, (na_val < 135).astype(float))

    # Healthcare Utilisation & Exposure Ratios
    df['total_bed_days'] = df['prior_admissions_12m'] * df['length_of_stay_days']
    df['comorbidity_per_age'] = df['comorbidity_count'] / (df['age'] + 1e-5)
    df['meds_per_condition'] = df['medication_count'] / (df['comorbidity_count'] + 1)

    rurality_map = {'Urban': 0, 'Semi-urban': 1, 'Rural': 2}
    rurality_ordinal = df['rurality'].map(rurality_map).fillna(1)
    df['access_barrier'] = df['missed_appointments_12m'] * rurality_ordinal
    df['missed_x_socioeconomic'] = df['missed_appointments_12m'] * df['socioeconomic_index']

    home_support = (df['discharge_disposition'] == 'Home_with_support').astype(float)
    df['discharge_vulnerability'] = home_support * (1 - df['socioeconomic_index'].clip(-2, 2) / 2)

    # Additional Engineered Features
    chronic_cols = ['diabetes', 'hypertension', 'chronic_kidney_disease', 'heart_failure']
    df['chronic_burden'] = df[chronic_cols].sum(axis=1).astype(np.int8)

    df['age_bucket'] = pd.cut(
        df['age'],
        bins=[0, 40, 50, 60, 70, 80, 120],
        labels=[0, 1, 2, 3, 4, 5],
        right=False,
    ).astype(float)

    df['has_prior_admission'] = (df['prior_admissions_12m'] > 0).astype(np.int8)
    df['readmission_frequency'] = df['prior_admissions_12m'] / (df['age'] + 1e-5)
    df['meds_per_los'] = df['medication_count'] / (df['length_of_stay_days'] + 1e-5)

    df['followup_missing_or_long'] = np.where(
        df['followup_days'].isna(), 1,
        np.where(df['followup_days'] > 30, 1, 0)
    ).astype(np.int8)

    df['abnormal_vitals_count'] = (
        df['tachycardia'].fillna(0)
        + df['bradycardia'].fillna(0)
        + df['hypertension_stage1'].fillna(0)
        + df['hyponatremia'].fillna(0)
        + df['anemia_flag'].fillna(0)
        + df['renal_impairment'].fillna(0)
    ).astype(np.int8)

    df['los_x_comorbidity'] = df['length_of_stay_days'] * df['comorbidity_count']

    return df


def load_and_prepare_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    target = train['readmitted_30d'].copy()

    train = engineer_features(train)
    test = engineer_features(test)

    exclude_cols = {'patient_id', 'readmitted_30d'}
    all_features = [c for c in train.columns if c not in exclude_cols]
    cat_cols = ['sex', 'rurality', 'hospital_type', 'region', 'discharge_disposition', 'care_pathway']
    num_features = [c for c in all_features if c not in cat_cols]

    # Pre-encode for LightGBM
    train_lgb = train.copy()
    test_lgb = test.copy()
    for col in cat_cols:
        train_lgb[col] = train_lgb[col].astype('category')
        test_lgb[col] = test_lgb[col].astype('category')

    # Cat features indices for CatBoost
    cat_col_indices = [all_features.index(c) for c in cat_cols]

    # OneHotEncoder for Logistic Regression
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    ohe.fit(pd.concat([train[cat_cols], test[cat_cols]]))

    return {
        'train': train, 'test': test, 'target': target,
        'all_features': all_features, 'cat_cols': cat_cols, 'num_features': num_features,
        'train_lgb': train_lgb, 'test_lgb': test_lgb,
        'cat_col_indices': cat_col_indices, 'ohe': ohe,
    }


# ──────────────────────────────────────────────────────────────────────
# 2. Optuna Objective: LightGBM
# ──────────────────────────────────────────────────────────────────────
def tune_lightgbm(data, n_trials=35):
    print("=" * 70)
    print(f"TUNING LIGHTGBM WITH OPTUNA ({n_trials} Trials)")
    print("=" * 70)

    train_lgb = data['train_lgb']
    target = data['target']
    features = data['all_features']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.012, 0.030),
            'num_leaves': trial.suggest_int('num_leaves', 10, 24),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 140),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 0.85),
            'subsample': trial.suggest_float('subsample', 0.65, 0.90),
            'subsample_freq': 1,
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
            'random_state': 42,
            'verbose': -1,
        }

        oof = np.zeros(len(train_lgb))
        for fold, (trn_idx, val_idx) in enumerate(skf.split(train_lgb, target)):
            X_tr, y_tr = train_lgb.iloc[trn_idx][features], target.iloc[trn_idx]
            X_va, y_va = train_lgb.iloc[val_idx][features], target.iloc[val_idx]

            model = lgb.LGBMClassifier(**{**params, 'random_state': 42 + fold})
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
            )
            oof[val_idx] = model.predict_proba(X_va)[:, 1]

        score = log_loss(target, oof)
        return score

    study = optuna.create_study(direction="minimize", study_name="lgb_tuning")
    
    def callback(study, trial):
        print(f"  [Trial {trial.number+1:>2}/{n_trials}] Log Loss: {trial.value:.5f} (Best: {study.best_value:.5f}) | lr={trial.params['learning_rate']:.3f}, leaves={trial.params['num_leaves']}, min_child={trial.params['min_child_samples']}")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    print("\n[+] LightGBM Tuning Complete!")
    print(f"    Best OOF Log Loss: {study.best_value:.5f} (v2 baseline was 0.35113)")
    print("    Best Parameters:")
    for k, v in study.best_params.items():
        print(f"      {k}: {v}")

    return study.best_params, study.best_value


# ──────────────────────────────────────────────────────────────────────
# 3. Optuna Objective: Logistic Regression (ElasticNet)
# ──────────────────────────────────────────────────────────────────────
def tune_logistic_regression(data, n_trials=25):
    print("\n" + "=" * 70)
    print(f"TUNING LOGISTIC REGRESSION (ELASTICNET) WITH OPTUNA ({n_trials} Trials)")
    print("=" * 70)

    train = data['train']
    target = data['target']
    num_features = data['num_features']
    cat_cols = data['cat_cols']
    ohe = data['ohe']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Pre-scale numerics per fold outside the trial loop for speed
    fold_data = []
    for trn_idx, val_idx in skf.split(train, target):
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(train.iloc[trn_idx][num_features].fillna(-999))
        X_va_num = scaler.transform(train.iloc[val_idx][num_features].fillna(-999))

        X_tr_cat = ohe.transform(train.iloc[trn_idx][cat_cols])
        X_va_cat = ohe.transform(train.iloc[val_idx][cat_cols])

        X_tr_lr = np.hstack([X_tr_num, X_tr_cat])
        X_va_lr = np.hstack([X_va_num, X_va_cat])

        fold_data.append((X_tr_lr, X_va_lr, target.iloc[trn_idx], target.iloc[val_idx], val_idx))

    def objective(trial):
        C = trial.suggest_float('C', 0.01, 5.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

        oof = np.zeros(len(train))
        for fold, (X_tr, X_va, y_tr, y_va, val_idx) in enumerate(fold_data):
            model = LogisticRegression(
                penalty='elasticnet', solver='saga',
                C=C, l1_ratio=l1_ratio,
                max_iter=3000, random_state=42 + fold,
            )
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict_proba(X_va)[:, 1]

        return log_loss(target, oof)

    study = optuna.create_study(direction="minimize", study_name="lr_tuning")

    def callback(study, trial):
        print(f"  [Trial {trial.number+1:>2}/{n_trials}] Log Loss: {trial.value:.5f} (Best: {study.best_value:.5f}) | C={trial.params['C']:.4f}, l1_ratio={trial.params['l1_ratio']:.3f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    print("\n[+] Logistic Regression Tuning Complete!")
    print(f"    Best OOF Log Loss: {study.best_value:.5f} (v3 baseline was 0.35209)")
    print("    Best Parameters:")
    for k, v in study.best_params.items():
        print(f"      {k}: {v}")

    return study.best_params, study.best_value


# ──────────────────────────────────────────────────────────────────────
# 4. Optuna Objective: CatBoost
# ──────────────────────────────────────────────────────────────────────
def tune_catboost(data, n_trials=20):
    print("\n" + "=" * 70)
    print(f"TUNING CATBOOST WITH OPTUNA ({n_trials} Trials)")
    print("=" * 70)

    train = data['train']
    target = data['target']
    features = data['all_features']
    cat_col_indices = data['cat_col_indices']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Convert string columns to category/str
    train_cb = train.copy()
    for col in data['cat_cols']:
        train_cb[col] = train_cb[col].astype(str)

    def objective(trial):
        params = {
            'iterations': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.035),
            'depth': trial.suggest_int('depth', 4, 6),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.1, 2.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': 128,
            'eval_metric': 'Logloss',
            'cat_features': cat_col_indices,
            'random_seed': 42,
            'verbose': 0,
        }

        oof = np.zeros(len(train_cb))
        for fold, (trn_idx, val_idx) in enumerate(skf.split(train_cb, target)):
            X_tr, y_tr = train_cb.iloc[trn_idx][features], target.iloc[trn_idx]
            X_va, y_va = train_cb.iloc[val_idx][features], target.iloc[val_idx]

            model = CatBoostClassifier(**{**params, 'random_seed': 42 + fold})
            model.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                early_stopping_rounds=80,
                verbose=False
            )
            oof[val_idx] = model.predict_proba(X_va)[:, 1]

        return log_loss(target, oof)

    study = optuna.create_study(direction="minimize", study_name="cb_tuning")

    def callback(study, trial):
        print(f"  [Trial {trial.number+1:>2}/{n_trials}] Log Loss: {trial.value:.5f} (Best: {study.best_value:.5f}) | depth={trial.params['depth']}, lr={trial.params['learning_rate']:.3f}, l2={trial.params['l2_leaf_reg']:.2f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    print("\n[+] CatBoost Tuning Complete!")
    print(f"    Best OOF Log Loss: {study.best_value:.5f} (v3 baseline was 0.35144)")
    print("    Best Parameters:")
    for k, v in study.best_params.items():
        print(f"      {k}: {v}")

    return study.best_params, study.best_value


# ──────────────────────────────────────────────────────────────────────
# 5. Full Evaluation & Ensemble Re-Blending with Tuned Params
# ──────────────────────────────────────────────────────────────────────
def run_tuned_ensemble(data, best_params_all):
    print("\n" + "=" * 70)
    print("EVALUATING TUNED MODELS & OPTIMIZING ENSEMBLE BLEND (v4 Tuned)")
    print("=" * 70)

    train = data['train']
    test = data['test']
    target = data['target']
    features = data['all_features']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_tuned = {}
    test_tuned = {}

    # --- 1. Tuned LightGBM ---
    print("\n[1/3] Training Tuned LightGBM...")
    train_lgb = data['train_lgb']
    test_lgb = data['test_lgb']
    lgb_p = {
        'n_estimators': 2000,
        **best_params_all.get('lightgbm', {}),
        'verbose': -1,
    }
    oof_lgb = np.zeros(len(train))
    test_lgb_preds = np.zeros(len(test))
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_lgb, target), 1):
        model = lgb.LGBMClassifier(**{**lgb_p, 'random_state': 42 + fold})
        model.fit(
            train_lgb.iloc[trn_idx][features], target.iloc[trn_idx],
            eval_set=[(train_lgb.iloc[val_idx][features], target.iloc[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        oof_lgb[val_idx] = model.predict_proba(train_lgb.iloc[val_idx][features])[:, 1]
        test_lgb_preds += model.predict_proba(test_lgb[features])[:, 1] / 5

    oof_tuned['LightGBM'] = oof_lgb
    test_tuned['LightGBM'] = test_lgb_preds
    print(f"    Tuned LightGBM OOF Log Loss: {log_loss(target, oof_lgb):.5f} | AUC: {roc_auc_score(target, oof_lgb):.4f}")

    # --- 2. Tuned Logistic Regression ---
    print("\n[2/3] Training Tuned Logistic Regression...")
    num_features = data['num_features']
    cat_cols = data['cat_cols']
    ohe = data['ohe']
    lr_p = best_params_all.get('logistic_regression', {'C': 0.5, 'l1_ratio': 0.5})

    oof_lr = np.zeros(len(train))
    test_lr_preds = np.zeros(len(test))
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(train.iloc[trn_idx][num_features].fillna(-999))
        X_va_num = scaler.transform(train.iloc[val_idx][num_features].fillna(-999))
        X_te_num = scaler.transform(test[num_features].fillna(-999))

        X_tr_cat = ohe.transform(train.iloc[trn_idx][cat_cols])
        X_va_cat = ohe.transform(train.iloc[val_idx][cat_cols])
        X_te_cat = ohe.transform(test[cat_cols])

        X_tr = np.hstack([X_tr_num, X_tr_cat])
        X_va = np.hstack([X_va_num, X_va_cat])
        X_te = np.hstack([X_te_num, X_te_cat])

        model = LogisticRegression(
            penalty='elasticnet', solver='saga',
            C=lr_p['C'], l1_ratio=lr_p['l1_ratio'],
            max_iter=5000, random_state=42 + fold
        )
        model.fit(X_tr, target.iloc[trn_idx])
        oof_lr[val_idx] = model.predict_proba(X_va)[:, 1]
        test_lr_preds += model.predict_proba(X_te)[:, 1] / 5

    oof_tuned['LogisticReg'] = oof_lr
    test_tuned['LogisticReg'] = test_lr_preds
    print(f"    Tuned LogisticReg OOF Log Loss: {log_loss(target, oof_lr):.5f} | AUC: {roc_auc_score(target, oof_lr):.4f}")

    # --- 3. Tuned CatBoost ---
    print("\n[3/3] Training Tuned CatBoost...")
    train_cb = train.copy()
    test_cb = test.copy()
    for col in data['cat_cols']:
        train_cb[col] = train_cb[col].astype(str)
        test_cb[col] = test_cb[col].astype(str)

    cb_p = {
        'iterations': 2000,
        **best_params_all.get('catboost', {}),
        'border_count': 128,
        'eval_metric': 'Logloss',
        'cat_features': data['cat_col_indices'],
        'verbose': 0,
    }
    oof_cb = np.zeros(len(train))
    test_cb_preds = np.zeros(len(test))
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_cb, target), 1):
        model = CatBoostClassifier(**{**cb_p, 'random_seed': 42 + fold})
        model.fit(
            train_cb.iloc[trn_idx][features], target.iloc[trn_idx],
            eval_set=(train_cb.iloc[val_idx][features], target.iloc[val_idx]),
            early_stopping_rounds=100,
            verbose=False
        )
        oof_cb[val_idx] = model.predict_proba(train_cb.iloc[val_idx][features])[:, 1]
        test_cb_preds += model.predict_proba(test_cb[features])[:, 1] / 5

    oof_tuned['CatBoost'] = oof_cb
    test_tuned['CatBoost'] = test_cb_preds
    print(f"    Tuned CatBoost OOF Log Loss: {log_loss(target, oof_cb):.5f} | AUC: {roc_auc_score(target, oof_cb):.4f}")

    # --- SLSQP Blend Optimization ---
    print("\nOptimizing Tuned Blend Weights (SLSQP)...")
    model_keys = ['LightGBM', 'LogisticReg', 'CatBoost']
    oof_matrix = np.column_stack([oof_tuned[k] for k in model_keys])
    test_matrix = np.column_stack([test_tuned[k] for k in model_keys])

    def loss_func(weights):
        weights = np.array(weights)
        weights /= weights.sum()
        blended = oof_matrix @ weights
        blended = np.clip(blended, 1e-7, 1 - 1e-7)
        return log_loss(target, blended)

    init_w = [1.0 / len(model_keys)] * len(model_keys)
    bounds = [(0.0, 1.0)] * len(model_keys)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    res = minimize(loss_func, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_w = res.x / np.sum(res.x)

    blended_oof = oof_matrix @ optimal_w
    blended_test = test_matrix @ optimal_w

    final_ll = log_loss(target, blended_oof)
    final_brier = brier_score_loss(target, blended_oof)
    final_auc = roc_auc_score(target, blended_oof)

    print("\n" + "=" * 70)
    print("TUNED ENSEMBLE RESULTS (v4)")
    print("=" * 70)
    for k, w in zip(model_keys, optimal_w):
        print(f"  {k:<16}: {w*100:>5.1f}%")
    print("-" * 70)
    print(f"Tuned OOF Log Loss : {final_ll:.5f}  (v3 was 0.34968, Δ {final_ll - 0.34968:+.5f})")
    print(f"Tuned OOF Brier    : {final_brier:.5f}")
    print(f"Tuned OOF ROC-AUC  : {final_auc:.5f}  (v3 was 0.69256, Δ {final_auc - 0.69256:+.5f})")
    print("=" * 70)

    # Save artifacts
    sub = pd.DataFrame({
        'patient_id': test['patient_id'],
        'readmitted_30d': blended_test
    })
    sub.to_csv('submission_v4.csv', index=False)
    print(f"Saved submission to submission_v4.csv (shape: {sub.shape})")

    oof_df = pd.DataFrame({
        'patient_id': train['patient_id'],
        'actual': target,
        'pred_prob': blended_oof,
        'pred_blended': blended_oof,
    })
    oof_df.to_csv('oof_v4.csv', index=False)
    print(f"Saved OOF predictions to oof_v4.csv (shape: {oof_df.shape})")


# ──────────────────────────────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument("--model", type=str, default="all", choices=["lgb", "lr", "cb", "all"],
                        help="Model to tune: 'lgb', 'lr', 'cb', or 'all' (default: all)")
    parser.add_argument("--trials", type=int, default=None,
                        help="Number of trials (overrides default per model)")
    args = parser.parse_args()

    print("=" * 70)
    print("LOADING DATA & EXTRACTING 54 FEATURES...")
    data = load_and_prepare_data()
    print(f"Loaded {len(data['train'])} train and {len(data['test'])} test rows.")
    print("=" * 70)

    # Load existing best params if available
    params_file = "best_params_optuna.json"
    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            best_params = json.load(f)
    else:
        best_params = {}

    if args.model in ["lgb", "all"]:
        n_t = args.trials if args.trials else 35
        best_p, _ = tune_lightgbm(data, n_trials=n_t)
        best_params["lightgbm"] = best_p
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=2)

    if args.model in ["lr", "all"]:
        n_t = args.trials if args.trials else 25
        best_p, _ = tune_logistic_regression(data, n_trials=n_t)
        best_params["logistic_regression"] = best_p
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=2)

    if args.model in ["cb", "all"]:
        n_t = args.trials if args.trials else 20
        best_p, _ = tune_catboost(data, n_trials=n_t)
        best_params["catboost"] = best_p
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=2)

    print(f"\n[+] Saved all winning parameters to {params_file}")

    # If 'all' was tuned, run the final tuned ensemble
    if args.model == "all":
        run_tuned_ensemble(data, best_params)


if __name__ == "__main__":
    main()
