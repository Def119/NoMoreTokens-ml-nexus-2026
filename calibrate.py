import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def compute_calibration_stats(y_true, y_prob):
    """Computes calibration intercept, slope, and Expected Calibration Error (ECE)."""
    eps = 1e-6
    p_clipped = np.clip(y_prob, eps, 1 - eps)
    logits = logit(p_clipped).reshape(-1, 1)

    # Calibration intercept and slope via logistic regression: logit(y) ~ a + b * logit(p)
    lr = LogisticRegression(C=1e5, solver="lbfgs")
    lr.fit(logits, y_true)
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    # Expected Calibration Error (ECE) using 10 equal-width bins
    bin_edges = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
    ece = 0.0
    for b in range(1, 11):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / len(y_prob)) * np.abs(bin_acc - bin_conf)

    return intercept, slope, ece


def print_decile_table(y_true, y_prob, title="Decile Reliability Table"):
    """Prints a binned calibration reliability table."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decile"] = pd.qcut(df["y_prob"], q=10, duplicates="drop")
    grouped = (
        df.groupby("decile", observed=False)
        .agg(
            Count=("y_true", "count"),
            Mean_Predicted=("y_prob", "mean"),
            Observed_Rate=("y_true", "mean"),
        )
        .reset_index()
    )
    grouped["Abs_Error"] = (
        grouped["Mean_Predicted"] - grouped["Observed_Rate"]
    ).abs()

    print(f"\n--- {title} ---")
    print(
        f"{'Decile':<20} | {'Count':<6} | {'Mean Pred':<10} | {'Observed':<10} | {'Abs Error':<10}"
    )
    print("-" * 65)
    for _, row in grouped.iterrows():
        print(
            f"{str(row['decile']):<20} | {int(row['Count']):<6} | {row['Mean_Predicted']:<10.4f} | {row['Observed_Rate']:<10.4f} | {row['Abs_Error']:<10.4f}"
        )


def main():
    print("=" * 70)
    print("PROBABILITY CALIBRATION EXPERIMENT")
    print("=" * 70)

    # 1. Load Data
    oof_df = pd.read_csv("oof_baseline.csv")
    sub_df = pd.read_csv("submission.csv")

    y_true = oof_df["actual"].values
    raw_probs = oof_df["pred_prob"].values
    test_probs = sub_df["readmitted_30d"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Cross-Validated Calibration
    platt_oof = np.zeros(len(y_true))
    isotonic_oof = np.zeros(len(y_true))

    eps = 1e-6
    raw_logits = logit(np.clip(raw_probs, eps, 1 - eps)).reshape(-1, 1)

    for trn_idx, val_idx in skf.split(raw_logits, y_true):
        # Platt Scaling (Logistic Regression on Log-Odds)
        platt = LogisticRegression(C=1e5, solver="lbfgs")
        platt.fit(raw_logits[trn_idx], y_true[trn_idx])
        platt_oof[val_idx] = platt.predict_proba(raw_logits[val_idx])[:, 1]

        # Isotonic Regression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_probs[trn_idx], y_true[trn_idx])
        isotonic_oof[val_idx] = iso.predict(raw_probs[val_idx])

    # 3. Compute Metrics for Comparison
    results = []

    models = [
        ("Raw LightGBM (Uncalibrated)", raw_probs),
        ("Platt Scaling (Logistic CV)", platt_oof),
        ("Isotonic Regression (CV)", isotonic_oof),
    ]

    for name, p in models:
        # Clip to prevent log(0)
        p_eval = np.clip(p, 1e-7, 1 - 1e-7)
        ll = log_loss(y_true, p_eval)
        brier = brier_score_loss(y_true, p_eval)
        auc = roc_auc_score(y_true, p_eval)
        intercept, slope, ece = compute_calibration_stats(y_true, p_eval)
        results.append(
            {
                "Method": name,
                "Log Loss": ll,
                "Brier Score": brier,
                "ROC-AUC": auc,
                "Calib Intercept": intercept,
                "Calib Slope": slope,
                "ECE": ece,
            }
        )

    res_df = pd.DataFrame(results)

    print("\nCalibration Performance Comparison:")
    print(
        f"{'Method':<28} | {'Log Loss':<9} | {'Brier':<8} | {'AUC':<7} | {'Slope':<7} | {'Intercept':<9} | {'ECE':<7}"
    )
    print("-" * 85)
    for _, r in res_df.iterrows():
        print(
            f"{r['Method']:<28} | {r['Log Loss']:<9.5f} | {r['Brier Score']:<8.5f} | {r['ROC-AUC']:<7.5f} | {r['Calib Slope']:<7.3f} | {r['Calib Intercept']:<9.3f} | {r['ECE']:<7.4f}"
        )

    # 4. Decile breakdown before vs after
    print_decile_table(y_true, raw_probs, "Raw LightGBM Reliability")

    # Pick the best method by Log Loss
    best_method = (
        "Platt" if res_df.loc[1, "Log Loss"] < res_df.loc[2, "Log Loss"] else "Isotonic"
    )
    best_oof = platt_oof if best_method == "Platt" else isotonic_oof
    print_decile_table(
        y_true, best_oof, f"{best_method} Calibrated Reliability"
    )

    # 5. Apply Best Calibrator to Test Predictions
    if best_method == "Platt":
        full_calibrator = LogisticRegression(C=1e5, solver="lbfgs")
        full_calibrator.fit(raw_logits, y_true)
        test_logits = logit(np.clip(test_probs, eps, 1 - eps)).reshape(-1, 1)
        calibrated_test_preds = full_calibrator.predict_proba(test_logits)[:, 1]
    else:
        full_calibrator = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0
        )
        full_calibrator.fit(raw_probs, y_true)
        calibrated_test_preds = full_calibrator.predict(test_probs)

    calibrated_test_preds = np.clip(calibrated_test_preds, 1e-6, 1 - 1e-6)

    # Save calibrated submission
    sub_cal = pd.DataFrame(
        {"patient_id": sub_df["patient_id"], "readmitted_30d": calibrated_test_preds}
    )
    sub_cal.to_csv("submission_calibrated.csv", index=False)
    print("\n" + "=" * 70)
    print(
        f"Saved calibrated predictions ({best_method}) to submission_calibrated.csv"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
