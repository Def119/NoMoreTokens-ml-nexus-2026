"""
explainability.py – Global/Local SHAP Interpretability & Uncertainty Referral Simulation
========================================================================================
Fulfills Trust Card Sections 7 & 8 from consideration.md:
  - Section 7: Uncertainty / Human Referral (10% abstention simulation)
  - Section 8: Explainability (Global SHAP summary + Local High-Risk & Low-Risk Patient Waterfalls)
  - Association vs. Causation Clinical Documentation
"""

import json
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings('ignore', category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
# 1. Feature Engineering (Identical to v2/v3/v4)
# ──────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing_lab_cols = [
        'hemoglobin_g_dl', 'creatinine_mg_dl', 'sodium_mmol_l',
        'heart_rate_bpm', 'systolic_bp_mmhg', 'followup_days',
    ]
    for col in missing_lab_cols:
        df[f'missing_{col}'] = df[col].isna().astype(np.int8)

    missing_flag_cols = [f'missing_{c}' for c in missing_lab_cols]
    df['total_missing_labs'] = df[missing_flag_cols].sum(axis=1).astype(np.int8)

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

    df['total_bed_days'] = df['prior_admissions_12m'] * df['length_of_stay_days']
    df['comorbidity_per_age'] = df['comorbidity_count'] / (df['age'] + 1e-5)
    df['meds_per_condition'] = df['medication_count'] / (df['comorbidity_count'] + 1)

    rurality_map = {'Urban': 0, 'Semi-urban': 1, 'Rural': 2}
    rurality_ordinal = df['rurality'].map(rurality_map).fillna(1)
    df['access_barrier'] = df['missed_appointments_12m'] * rurality_ordinal
    df['missed_x_socioeconomic'] = df['missed_appointments_12m'] * df['socioeconomic_index']

    home_support = (df['discharge_disposition'] == 'Home_with_support').astype(float)
    df['discharge_vulnerability'] = home_support * (1 - df['socioeconomic_index'].clip(-2, 2) / 2)

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


# ──────────────────────────────────────────────────────────────────────
# 2. Uncertainty Referral / Human Abstention Simulation (Section 7)
# ──────────────────────────────────────────────────────────────────────
def run_uncertainty_referral(oof_df):
    print("=" * 75)
    print("SECTION 7: UNCERTAINTY / HUMAN REFERRAL SIMULATION (10% Abstention)")
    print("=" * 75)

    y_true = oof_df["actual"].values
    y_prob = oof_df["pred_prob"].values if "pred_prob" in oof_df.columns else oof_df["pred_blended"].values

    # Binary Entropy as uncertainty measure: H(p) = -p*log2(p) - (1-p)*log2(1-p)
    eps = 1e-7
    p_c = np.clip(y_prob, eps, 1 - eps)
    entropy = -p_c * np.log2(p_c) - (1 - p_c) * np.log2(1 - p_c)

    # Operating threshold based on class prevalence (~0.126) or 0.20 for high-risk triage
    threshold = 0.15

    # Identify 10% most uncertain predictions (highest entropy / closest to decision boundary)
    n_total = len(y_true)
    n_refer = int(0.10 * n_total)
    referral_threshold = np.sort(entropy)[-n_refer]

    is_referred = entropy >= referral_threshold
    retain_mask = ~is_referred

    # 1. Full Population (100%) Metrics
    ll_full = log_loss(y_true, y_prob)
    brier_full = brier_score_loss(y_true, y_prob)
    auc_full = roc_auc_score(y_true, y_prob)
    preds_full = (y_prob >= threshold).astype(int)
    acc_full = accuracy_score(y_true, preds_full)
    prec_full = precision_score(y_true, preds_full, zero_division=0)
    rec_full = recall_score(y_true, preds_full, zero_division=0)

    # 2. Retained Population (90% Confident) Metrics
    y_true_ret = y_true[retain_mask]
    y_prob_ret = y_prob[retain_mask]
    ll_ret = log_loss(y_true_ret, y_prob_ret)
    brier_ret = brier_score_loss(y_true_ret, y_prob_ret)
    auc_ret = roc_auc_score(y_true_ret, y_prob_ret)
    preds_ret = (y_prob_ret >= threshold).astype(int)
    acc_ret = accuracy_score(y_true_ret, preds_ret)
    prec_ret = precision_score(y_true_ret, preds_ret, zero_division=0)
    rec_ret = recall_score(y_true_ret, preds_ret, zero_division=0)

    # 3. Referred Cohort (10% Ambiguous Cases)
    y_true_ref = y_true[is_referred]
    y_prob_ref = y_prob[is_referred]
    readmit_ref_rate = np.mean(y_true_ref)
    mean_prob_ref = np.mean(y_prob_ref)

    print(f"\nTotal Patients Evaluated      : {n_total}")
    print(f"Referred to Clinical Review   : {n_refer} ({n_refer/n_total*100:.1f}%)")
    print(f"Retained for Automated Action : {np.sum(retain_mask)} ({np.sum(retain_mask)/n_total*100:.1f}%)")
    print(f"Actual Readmission in Referred: {readmit_ref_rate*100:.2f}% (vs. baseline {np.mean(y_true)*100:.2f}%)")

    print("\n--- PERFORMANCE BEFORE VS. AFTER 10% REFERRAL ---")
    print(f"{'Metric':<25} | {'Full Cohort (100%)':<20} | {'Retained (90%)':<18} | {'Impact (Δ)':<12}")
    print("-" * 80)
    print(f"{'Log Loss (Lower is better)':<25} | {ll_full:<20.5f} | {ll_ret:<18.5f} | {ll_ret - ll_full:+.5f} 🟢")
    print(f"{'Brier Score (Lower is better)':<25} | {brier_full:<20.5f} | {brier_ret:<18.5f} | {brier_ret - brier_full:+.5f} 🟢")
    print(f"{'ROC-AUC (Higher is better)':<25} | {auc_full:<20.4f} | {auc_ret:<18.4f} | {auc_ret - auc_full:+.4f} 🟢")
    print(f"{'Accuracy (Threshold=0.15)':<25} | {acc_full*100:<19.2f}% | {acc_ret*100:<17.2f}% | {(acc_ret - acc_full)*100:+.2f}% 🟢")
    print(f"{'Precision (Threshold=0.15)':<25} | {prec_full*100:<19.2f}% | {prec_ret*100:<17.2f}% | {(prec_ret - prec_full)*100:+.2f}% 🟢")
    print(f"{'Recall (Threshold=0.15)':<25} | {rec_full*100:<19.2f}% | {rec_ret*100:<17.2f}% | {(rec_ret - rec_full)*100:+.2f}%")
    print("-" * 80)

    # Plot Referral Simulation Impact
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Uncertainty Distribution & Cutoff
    ax1 = axes[0]
    ax1.hist(entropy, bins=40, color="#3498db", alpha=0.7, edgecolor="black", label="Retained Predictions (90%)")
    ax1.hist(entropy[is_referred], bins=20, color="#e74c3c", alpha=0.85, edgecolor="black", label=f"Referred to Human Review (Top 10%, n={n_refer})")
    ax1.axvline(referral_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Referral Cutoff: {referral_threshold:.3f}")
    ax1.set_title("Prediction Uncertainty Distribution (Shannon Entropy)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Entropy H(p) [Bits]", fontsize=10)
    ax1.set_ylabel("Patient Count", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Performance Comparison Bar Chart
    ax2 = axes[1]
    metrics = ["Log Loss", "Brier Score", "ROC-AUC"]
    full_vals = [ll_full, brier_full, auc_full]
    ret_vals = [ll_ret, brier_ret, auc_ret]

    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax2.bar(x - w/2, full_vals, width=w, label="Full 100% Cohort", color="#95a5a6", edgecolor="black", alpha=0.85)
    b2 = ax2.bar(x + w/2, ret_vals, width=w, label="Retained 90% Cohort (After Referral)", color="#27ae60", edgecolor="black", alpha=0.85)

    ax2.set_title("Impact of Abstaining / Referring Top 10% Most Uncertain Cases", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 0.85)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar in b1:
        h = bar.get_height()
        ax2.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h + 0.015), ha="center", fontsize=8.5, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax2.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h + 0.015), ha="center", fontsize=8.5, fontweight="bold", color="#196F3D")

    plt.tight_layout()
    plt.savefig("uncertainty_referral.png", dpi=300, bbox_inches="tight")
    print("[+] Saved uncertainty referral chart to uncertainty_referral.png")

    return {
        "ll_full": ll_full, "ll_ret": ll_ret,
        "auc_full": auc_full, "auc_ret": auc_ret,
        "brier_full": brier_full, "brier_ret": brier_ret,
        "n_refer": n_refer, "readmit_ref_rate": readmit_ref_rate,
    }


# ──────────────────────────────────────────────────────────────────────
# 3. Global & Local SHAP Explainability (Section 8)
# ──────────────────────────────────────────────────────────────────────
def run_shap_explainability(train_df, best_params):
    print("\n" + "=" * 75)
    print("SECTION 8: GLOBAL & LOCAL SHAP EXPLAINABILITY")
    print("=" * 75)

    if not HAS_SHAP:
        print("[!] Warning: shap is not installed. Run: pip install shap")
        return

    # Prepare features
    features = [c for c in train_df.columns if c not in ['patient_id', 'readmitted_30d']]
    cat_cols = ['sex', 'rurality', 'hospital_type', 'region', 'discharge_disposition', 'care_pathway']
    X = train_df[features].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    y = train_df['readmitted_30d'].values

    # Fit a single model on full train data using the winning Optuna hyperparameters
    lgb_p = {
        'n_estimators': 350,
        **best_params.get('lightgbm', {}),
        'verbose': -1,
        'random_state': 42,
    }
    model = lgb.LGBMClassifier(**lgb_p)
    model.fit(X, y)

    # Compute TreeSHAP values
    print("Computing TreeSHAP values across all 7,000 patient samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Handle binary classification shape (extract class 1 if 3D)
    if len(shap_values.values.shape) == 3:
        sv_matrix = shap_values.values[:, :, 1]
        base_val = shap_values.base_values[0, 1] if len(shap_values.base_values.shape) > 1 else shap_values.base_values[0]
    else:
        sv_matrix = shap_values.values
        base_val = shap_values.base_values[0] if hasattr(shap_values.base_values, '__len__') else shap_values.base_values

    # --- 8A. Global Feature Importance (Top 20) ---
    mean_abs_shap = np.mean(np.abs(sv_matrix), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]
    top_features = [features[i] for i in top_indices]
    top_scores = mean_abs_shap[top_indices]

    print("\nTop 15 Most Influential Global Predictors (by Mean |SHAP|):")
    print(f"{'Rank':<5} | {'Feature':<30} | {'Mean |SHAP|':<12}")
    print("-" * 52)
    for rank, (feat, score) in enumerate(zip(top_features[:15], top_scores[:15]), 1):
        print(f"{rank:<5} | {feat:<30} | {score:<12.4f}")
    print("-" * 52)

    # Plot Global Beeswarm / Bar Summary
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_features))[::-1]
    ax.barh(y_pos, top_scores, color="#2980b9", edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Log-Odds of Readmission)", fontsize=10)
    ax.set_title("Global Feature Importance: Top 20 Predictors (TreeSHAP)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")
    for i, score in enumerate(top_scores):
        ax.annotate(f"{score:.4f}", (score + 0.005, y_pos[i]), va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig("shap_global_importance.png", dpi=300, bbox_inches="tight")
    print("[+] Saved global feature importance to shap_global_importance.png")

    # --- 8B. Local Case Studies: High-Risk vs. Low-Risk Patient ---
    predicted_probs = model.predict_proba(X)[:, 1]

    # Identify a compelling high-risk patient (actual=1, high probability)
    high_risk_candidates = np.where((y == 1) & (predicted_probs > 0.40))[0]
    high_idx = high_risk_candidates[0] if len(high_risk_candidates) > 0 else np.argmax(predicted_probs)

    # Identify a compelling low-risk patient (actual=0, low probability)
    low_risk_candidates = np.where((y == 0) & (predicted_probs < 0.04))[0]
    low_idx = low_risk_candidates[0] if len(low_risk_candidates) > 0 else np.argmin(predicted_probs)

    def plot_custom_waterfall(patient_idx, case_title, filename, is_high_risk=True):
        p_row = X.iloc[patient_idx]
        p_val = predicted_probs[patient_idx]
        p_actual = y[patient_idx]
        p_id = train_df.iloc[patient_idx]['patient_id']
        sv_patient = sv_matrix[patient_idx]

        # Top 10 contributors for this patient
        top_p_indices = np.argsort(np.abs(sv_patient))[::-1][:10]
        top_p_feats = [features[i] for i in top_p_indices]
        top_p_vals = sv_patient[top_p_indices]
        top_p_raw = [p_row[features[i]] for i in top_p_indices]

        fig, ax = plt.subplots(figsize=(10, 6.5))
        y_positions = np.arange(len(top_p_feats))[::-1]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_p_vals]

        labels = [f"{feat} = {val}" for feat, val in zip(top_p_feats, top_p_raw)]
        bars = ax.barh(y_positions, top_p_vals, color=colors, edgecolor="black", alpha=0.85)

        ax.axvline(0, color="black", linestyle="-", linewidth=1.2)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("SHAP Value (Contribution to Log-Odds of Readmission)", fontsize=10)
        ax.set_title(
            f"{case_title}: Patient {p_id}\n"
            f"Predicted Risk: {p_val*100:.1f}% | Actual Outcome: {'Readmitted (1)' if p_actual==1 else 'Not Readmitted (0)'} | Base Risk: {expit(base_val)*100:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.4, axis="x")

        # Annotations
        for bar, val in zip(bars, top_p_vals):
            offset = 0.01 if val >= 0 else -0.01
            ha = "left" if val >= 0 else "right"
            ax.annotate(f"{val:+.3f}", (val + offset, bar.get_y() + bar.get_height() / 2),
                        va="center", ha=ha, fontsize=8.5, fontweight="bold")

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"[+] Saved local explanation ({case_title}) to {filename}")

    plot_custom_waterfall(high_idx, "Local Explanation: High-Risk Prediction", "shap_high_risk.png", is_high_risk=True)
    plot_custom_waterfall(low_idx, "Local Explanation: Low-Risk Prediction", "shap_low_risk.png", is_high_risk=False)

    return {
        "top_features": top_features[:10],
        "high_patient": train_df.iloc[high_idx]['patient_id'],
        "high_prob": predicted_probs[high_idx],
        "low_patient": train_df.iloc[low_idx]['patient_id'],
        "low_prob": predicted_probs[low_idx],
    }


def main():
    train_raw = pd.read_csv("train.csv")
    train_eng = engineer_features(train_raw)

    # 1. Uncertainty Referral Simulation
    oof_file = "oof_v4.csv" if os.path.exists("oof_v4.csv") else ("oof_v3.csv" if os.path.exists("oof_v3.csv") else "oof_baseline.csv")
    oof_df = pd.read_csv(oof_file)
    ref_stats = run_uncertainty_referral(oof_df)

    # 2. SHAP Explainability
    best_params = {}
    if os.path.exists("best_params_optuna.json"):
        with open("best_params_optuna.json", "r") as f:
            best_params = json.load(f)

    shap_stats = run_shap_explainability(train_eng, best_params)
    print("\n" + "=" * 75)
    print("EXPLAINABILITY & UNCERTAINTY RUN COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
