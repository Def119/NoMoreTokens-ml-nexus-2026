import os
import matplotlib
matplotlib.use('Agg')  # Headless backend for WSL / remote environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_calibration_metrics(y_true, y_prob):
    """Computes calibration slope, intercept, and Expected Calibration Error (ECE)."""
    eps = 1e-6
    p_clipped = np.clip(y_prob, eps, 1 - eps)
    logits = logit(p_clipped).reshape(-1, 1)

    lr = LogisticRegression(C=1e5, solver="lbfgs")
    lr.fit(logits, y_true)
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    bin_edges = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
    ece = 0.0
    for b in range(1, 11):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / len(y_prob)) * np.abs(bin_acc - bin_conf)

    return slope, intercept, ece


def main():
    print("=" * 90)
    print("3-VERSION MODEL PROGRESSION & ENSEMBLE DIAGNOSTICS")
    print("=" * 90)

    # 1. Load Ground Truth
    train_df = pd.read_csv("train.csv")
    y_true = train_df["readmitted_30d"].values
    prior_prob = np.mean(y_true)

    # 2. Define Version Registry
    model_registry = [
        {
            "id": "Model 0",
            "name": "Model 0: Baseline LightGBM (basic.py)",
            "short_name": "M0: Baseline",
            "file": "oof_baseline.csv",
            "pred_col": "pred_prob",
            "color": "#e74c3c",      # Coral Red
            "linestyle": "--",
            "marker": "s",
        },
        {
            "id": "Model 1",
            "name": "Model 1: Feature-Eng + Reg LightGBM (v2.py)",
            "short_name": "M1: v2 (FE + Reg)",
            "file": "oof_v2.csv",
            "pred_col": "pred_prob",
            "color": "#2980b9",      # Vibrant Blue
            "linestyle": "-.",
            "marker": "^",
        },
        {
            "id": "Model 2",
            "name": "Model 2: Calibrated 4-Model Ensemble (v3.py)",
            "short_name": "M2: v3 (Ensemble)",
            "file": "oof_v3.csv",
            "pred_col": "pred_blended",
            "color": "#27ae60",      # Emerald Green
            "linestyle": "-",
            "marker": "o",
        },
        {
            "id": "Model 3",
            "name": "Model 3: Optuna-Tuned Ensemble (v4)",
            "short_name": "M3: v4 (Tuned)",
            "file": "oof_v4.csv",
            "pred_col": "pred_blended",
            "color": "#8e44ad",      # Royal Purple
            "linestyle": "-",
            "marker": "D",
        },
    ]

    # Load and validate available models
    active_models = []
    for m in model_registry:
        if os.path.exists(m["file"]):
            df = pd.read_csv(m["file"])
            # Support either pred_col or fallback
            pred_col = m["pred_col"] if m["pred_col"] in df.columns else ("pred_blended" if "pred_blended" in df.columns else "pred_prob")
            merged = pd.merge(train_df[["patient_id", "readmitted_30d"]], df, on="patient_id")
            preds = merged[pred_col].values
            m["preds"] = preds
            m["df"] = df
            m["actual_col"] = pred_col
            active_models.append(m)
        else:
            print(f"[-] Notice: {m['file']} not found on disk. Skipping {m['name']}.")

    if not active_models:
        print("Error: No OOF prediction files found! Please run basic.py, v2.py, or v3.py first.")
        return

    # 3. Compute Metrics & Print Side-by-Side Progression Table
    summary_data = []
    baseline_ll = None
    baseline_auc = None
    prev_ll = None
    prev_auc = None

    for idx, m in enumerate(active_models):
        p = m["preds"]
        ll = log_loss(y_true, p)
        brier = brier_score_loss(y_true, p)
        auc = roc_auc_score(y_true, p)
        ap = average_precision_score(y_true, p)
        slope, intercept, ece = compute_calibration_metrics(y_true, p)

        if idx == 0:
            baseline_ll = ll
            baseline_auc = auc
            delta_base_ll = 0.0
            delta_base_auc = 0.0
            delta_prev_ll = 0.0
            delta_prev_auc = 0.0
        else:
            delta_base_ll = ll - baseline_ll
            delta_base_auc = auc - baseline_auc
            delta_prev_ll = ll - prev_ll
            delta_prev_auc = auc - prev_auc

        prev_ll = ll
        prev_auc = auc

        m["log_loss"] = ll
        m["brier"] = brier
        m["auc"] = auc
        m["ap"] = ap
        m["slope"] = slope
        m["intercept"] = intercept
        m["ece"] = ece
        m["delta_base_ll"] = delta_base_ll
        m["delta_base_auc"] = delta_base_auc
        m["delta_prev_ll"] = delta_prev_ll

        summary_data.append({
            "Version": m["short_name"],
            "Log Loss": ll,
            "Δ vs Base": delta_base_ll,
            "Δ vs Prev": delta_prev_ll,
            "Brier": brier,
            "ROC-AUC": auc,
            "Δ AUC Base": delta_base_auc,
            "PR-AUC (AP)": ap,
            "Slope": slope,
            "Intercept": intercept,
            "ECE": ece,
        })

    sum_df = pd.DataFrame(summary_data)

    print("\n" + "=" * 115)
    print("PROGRESSION SCORECARD ACROSS ALL VERSIONS")
    print("=" * 115)
    print(f"{'Version':<18} | {'Log Loss':<9} | {'Δ vs Base':<9} | {'Δ vs Prev':<9} | {'Brier':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Slope':<6} | {'ECE':<6}")
    print("-" * 115)
    for _, r in sum_df.iterrows():
        db_str = f"{r['Δ vs Base']:+.5f}" if r['Δ vs Base'] != 0 else "Baseline"
        dp_str = f"{r['Δ vs Prev']:+.5f}" if r['Δ vs Prev'] != 0 else "—"
        print(f"{r['Version']:<18} | {r['Log Loss']:<9.5f} | {db_str:<9} | {dp_str:<9} | {r['Brier']:<8.5f} | {r['ROC-AUC']:<8.4f} | {r['PR-AUC (AP)']:<8.4f} | {r['Slope']:<6.3f} | {r['ECE']:<6.4f}")
    print("=" * 115)

    # If v3 exists, check blend weights and individual model performances
    has_v3_breakdown = False
    v3_components = {}
    if os.path.exists("oof_v3.csv"):
        v3_df = pd.read_csv("oof_v3.csv")
        sub_cols = [c for c in ["pred_lgb", "pred_cb", "pred_xgb", "pred_lr"] if c in v3_df.columns]
        if len(sub_cols) >= 2:
            has_v3_breakdown = True
            name_map = {"pred_lgb": "LightGBM", "pred_cb": "CatBoost", "pred_xgb": "XGBoost", "pred_lr": "LogisticReg"}
            print("\n--- MODEL 2 (v3) ENSEMBLE COMPONENT PERFORMANCE ---")
            print(f"{'Component Model':<18} | {'OOF Log Loss':<12} | {'OOF Brier':<10} | {'OOF ROC-AUC':<12}")
            print("-" * 60)
            for sc in sub_cols:
                c_name = name_map.get(sc, sc)
                c_p = v3_df[sc].values
                c_ll = log_loss(y_true, c_p)
                c_brier = brier_score_loss(y_true, c_p)
                c_auc = roc_auc_score(y_true, c_p)
                v3_components[c_name] = {"ll": c_ll, "brier": c_brier, "auc": c_auc}
                print(f"{c_name:<18} | {c_ll:<12.5f} | {c_brier:<10.5f} | {c_auc:<12.4f}")
            print("-" * 60)

    # -----------------------------------------------------------------
    # FIGURE 1: COMPREHENSIVE 6-PANEL PROGRESSION DASHBOARD
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(16, 17))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # --- Panel 1: Core Metric Progression Across Versions ---
    ax_bar = axes[0, 0]
    names = [m["short_name"] for m in active_models]
    lls = [m["log_loss"] for m in active_models]
    aucs = [m["auc"] for m in active_models]
    briers = [m["brier"] for m in active_models]

    x = np.arange(len(names))
    width = 0.25

    rects1 = ax_bar.bar(x - width, lls, width, label="Log Loss (lower is better)", color="#e74c3c", alpha=0.85, edgecolor="black")
    rects2 = ax_bar.bar(x, aucs, width, label="ROC-AUC (higher is better)", color="#2980b9", alpha=0.85, edgecolor="black")
    rects3 = ax_bar.bar(x + width, briers, width, label="Brier Score (lower is better)", color="#f39c12", alpha=0.85, edgecolor="black")

    ax_bar.set_title("1. Core Metric Progression Across Versions", fontsize=12, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax_bar.set_ylim(0.0, 0.85)
    ax_bar.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_bar.legend(loc="upper right", fontsize=9)

    # Value annotations on bars
    for rect in rects1:
        h = rect.get_height()
        ax_bar.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width() / 2, h + 0.012), ha="center", fontsize=8, fontweight="bold", color="#900C3F")
    for rect in rects2:
        h = rect.get_height()
        ax_bar.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width() / 2, h + 0.012), ha="center", fontsize=8, fontweight="bold", color="#1A5276")
    for rect in rects3:
        h = rect.get_height()
        ax_bar.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width() / 2, h + 0.012), ha="center", fontsize=8, fontweight="bold", color="#9A7D0A")

    # --- Panel 2: Comparative ROC Curves (Discrimination) ---
    ax_roc = axes[0, 1]
    ax_roc.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.5000)", alpha=0.5, linewidth=1.5)
    for m in active_models:
        fpr, tpr, _ = roc_curve(y_true, m["preds"])
        ax_roc.plot(
            fpr, tpr,
            label=f"{m['short_name']} (AUC = {m['auc']:.4f})",
            color=m["color"],
            linestyle=m["linestyle"],
            linewidth=2.2
        )
    ax_roc.set_title("2. Comparative ROC Curves (Discrimination Lift)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 3: Comparative Precision-Recall Curves ---
    ax_pr = axes[1, 0]
    ax_pr.plot([0, 1], [prior_prob, prior_prob], "k--", label=f"Prior Prevalence (AP = {prior_prob:.4f})", alpha=0.5, linewidth=1.5)
    for m in active_models:
        prec, rec, _ = precision_recall_curve(y_true, m["preds"])
        ax_pr.plot(
            rec, prec,
            label=f"{m['short_name']} (AP = {m['ap']:.4f})",
            color=m["color"],
            linestyle=m["linestyle"],
            linewidth=2.2
        )
    ax_pr.set_title("3. Comparative Precision-Recall Curves (Positive Predictive Value)", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax_pr.set_ylabel("Precision (PPV)", fontsize=10)
    ax_pr.legend(loc="upper right", fontsize=9)
    ax_pr.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 4: Comparative Calibration Reliability Curves ---
    ax_cal = axes[1, 1]
    ax_cal.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.6, linewidth=1.5)
    for m in active_models:
        prob_true, prob_pred = calibration_curve(y_true, m["preds"], n_bins=10, strategy="uniform")
        ax_cal.plot(
            prob_pred, prob_true,
            marker=m["marker"],
            label=f"{m['short_name']} (Slope: {m['slope']:.3f}, ECE: {m['ece']:.4f})",
            color=m["color"],
            linestyle=m["linestyle"],
            linewidth=2.0
        )
    ax_cal.set_xlim([0.0, 0.65])
    ax_cal.set_ylim([0.0, 0.65])
    ax_cal.set_title("4. Calibration Reliability Curves", fontsize=12, fontweight="bold")
    ax_cal.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax_cal.set_ylabel("Observed Fraction of Positives", fontsize=10)
    ax_cal.legend(loc="upper left", fontsize=9)
    ax_cal.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 5: Decile Risk Stratification Progression ---
    ax_dec = axes[2, 0]
    decile_indices = np.arange(1, 11)
    bar_w = 0.25 if len(active_models) >= 3 else 0.35
    for i, m in enumerate(active_models):
        temp_df = pd.DataFrame({"y_true": y_true, "p": m["preds"]})
        temp_df["decile"] = pd.qcut(temp_df["p"], q=10, labels=False) + 1
        decile_observed = temp_df.groupby("decile")["y_true"].mean().values
        offset = (i - (len(active_models) - 1) / 2) * bar_w
        ax_dec.bar(
            decile_indices + offset,
            decile_observed * 100,
            width=bar_w,
            label=f"{m['short_name']}",
            color=m["color"],
            alpha=0.85,
            edgecolor="black"
        )
    ax_dec.axhline(prior_prob * 100, color="gray", linestyle=":", label=f"Average Prevalence ({prior_prob*100:.1f}%)")
    ax_dec.set_title("5. Decile Risk Stratification (Observed % by Predicted Decile)", fontsize=12, fontweight="bold")
    ax_dec.set_xlabel("Predicted Risk Decile (1 = Lowest, 10 = Highest)", fontsize=10)
    ax_dec.set_ylabel("Observed Readmission Rate (%)", fontsize=10)
    ax_dec.set_xticks(decile_indices)
    ax_dec.legend(loc="upper left", fontsize=8.5)
    ax_dec.grid(True, linestyle="--", alpha=0.4, axis="y")

    # --- Panel 6: Ensemble Weights & Diversity Breakdown ---
    ax_ens = axes[2, 1]
    if os.path.exists("blend_weights_v3.csv"):
        bw_df = pd.read_csv("blend_weights_v3.csv")
        models_bw = bw_df["model"].values
        weights_bw = bw_df["weight"].values * 100

        colors_pie = ["#2980b9", "#8e44ad", "#e67e22", "#16a085"]
        y_pos = np.arange(len(models_bw))
        bars = ax_ens.barh(y_pos, weights_bw, color=colors_pie[:len(models_bw)], edgecolor="black", alpha=0.85)
        ax_ens.set_yticks(y_pos)
        ax_ens.set_yticklabels(models_bw, fontsize=10, fontweight="bold")
        ax_ens.set_xlabel("Ensemble Blend Weight (%)", fontsize=10)
        ax_ens.set_xlim(0, 60)
        ax_ens.set_title("6. Optimal Blend Weights in Model 2 (v3 SLSQP)", fontsize=12, fontweight="bold")
        ax_ens.grid(True, linestyle="--", alpha=0.4, axis="x")

        for bar in bars:
            w = bar.get_width()
            ax_ens.annotate(f"{w:.1f}%", (w + 1.0, bar.get_y() + bar.get_height() / 2), va="center", fontsize=9, fontweight="bold")
    else:
        # Fallback to probability spread boxplot
        plot_data = []
        labels_box = []
        colors_box = []
        for m in active_models:
            pos_p = m["preds"][y_true == 1]
            neg_p = m["preds"][y_true == 0]
            plot_data.extend([neg_p, pos_p])
            labels_box.extend([f"{m['short_name']}\nNon-Readmit", f"{m['short_name']}\nReadmitted"])
            colors_box.extend(["#bdc3c7", m["color"]])

        bp = ax_ens.boxplot(plot_data, patch_artist=True, tick_labels=labels_box, showmeans=True)
        for patch, col in zip(bp["boxes"], colors_box):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)
        ax_ens.set_title("6. Predicted Probability Separation by True Class", fontsize=12, fontweight="bold")
        ax_ens.set_ylabel("Predicted Probability", fontsize=10)
        ax_ens.tick_params(axis="x", rotation=25, labelsize=8)
        ax_ens.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle(
        "Evolution Across 3 Iterations: Model 0 (Baseline) → Model 1 (FE) → Model 2 (Ensemble)",
        fontsize=16,
        fontweight="bold",
        y=0.99
    )
    plt.savefig("model_progression_dashboard.png", dpi=300, bbox_inches="tight")
    print("\n[+] Saved 6-panel progression dashboard to model_progression_dashboard.png")

    # -----------------------------------------------------------------
    # FIGURE 2: SUBGROUP PROGRESSION AUDIT (Trust Card Parity Across M0, M1, M2)
    # -----------------------------------------------------------------
    subgroups = {
        "Sex": "sex",
        "Rurality": "rurality",
        "Hospital Type": "hospital_type",
    }

    subgroup_records = []
    for m in active_models:
        m_merged = pd.merge(train_df, m["df"], on="patient_id")
        pred_col = m["actual_col"]
        for cat_name, col in subgroups.items():
            for val in sorted(m_merged[col].unique()):
                subset = m_merged[m_merged[col] == val]
                y_sub = subset["actual"].values
                p_sub = subset[pred_col].values
                n_sub = len(subset)
                ll_sub = log_loss(y_sub, p_sub)
                auc_sub = roc_auc_score(y_sub, p_sub) if len(np.unique(y_sub)) > 1 else np.nan
                subgroup_records.append({
                    "Model": m["short_name"],
                    "Category": cat_name,
                    "Subgroup": str(val),
                    "N": n_sub,
                    "Prevalence": np.mean(y_sub),
                    "Log Loss": ll_sub,
                    "ROC-AUC": auc_sub,
                })

    sub_df = pd.DataFrame(subgroup_records)

    # Print Subgroup Evolution Table
    print("\n" + "=" * 115)
    print("SUBGROUP PERFORMANCE AUDIT ACROSS ALL 3 VERSIONS")
    print("=" * 115)
    piv_ll = sub_df.pivot(index=["Category", "Subgroup"], columns="Model", values="Log Loss").reset_index()
    piv_auc = sub_df.pivot(index=["Category", "Subgroup"], columns="Model", values="ROC-AUC").reset_index()

    comp_sub = pd.merge(piv_ll, piv_auc, on=["Category", "Subgroup"], suffixes=(" (LogLoss)", " (AUC)"))
    print(comp_sub.to_string(index=False))
    print("=" * 115)

    # Plot Subgroup Comparison
    fig_sub, axes_sub = plt.subplots(1, 2, figsize=(18, 6.5))

    subgroup_labels = [f"{r['Category']}: {r['Subgroup']}" for _, r in piv_ll.iterrows()]
    x_sub = np.arange(len(subgroup_labels))
    w_sub = 0.25 if len(active_models) >= 3 else 0.35

    # Panel 1: Log Loss across subgroups
    ax_s1 = axes_sub[0]
    for i, m in enumerate(active_models):
        m_ll_vals = piv_ll[m["short_name"]].values
        offset = (i - (len(active_models) - 1) / 2) * w_sub
        bars = ax_s1.bar(x_sub + offset, m_ll_vals, width=w_sub, label=m["short_name"], color=m["color"], alpha=0.85, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax_s1.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h + 0.003), ha="center", fontsize=7, rotation=45)

    ax_s1.set_title("Subgroup Log Loss: Progression M0 → M1 → M2 (Lower is better)", fontsize=12, fontweight="bold")
    ax_s1.set_xticks(x_sub)
    ax_s1.set_xticklabels(subgroup_labels, rotation=40, ha="right", fontsize=9)
    ax_s1.set_ylabel("Log Loss", fontsize=10)
    ax_s1.set_ylim(0.28, 0.44)
    ax_s1.legend(loc="upper right", fontsize=9)
    ax_s1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 2: ROC-AUC across subgroups
    ax_s2 = axes_sub[1]
    for i, m in enumerate(active_models):
        m_auc_vals = piv_auc[m["short_name"]].values
        offset = (i - (len(active_models) - 1) / 2) * w_sub
        bars = ax_s2.bar(x_sub + offset, m_auc_vals, width=w_sub, label=m["short_name"], color=m["color"], alpha=0.85, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax_s2.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h + 0.005), ha="center", fontsize=7, rotation=45)

    ax_s2.set_title("Subgroup ROC-AUC: Progression M0 → M1 → M2 (Higher is better)", fontsize=12, fontweight="bold")
    ax_s2.set_xticks(x_sub)
    ax_s2.set_xticklabels(subgroup_labels, rotation=40, ha="right", fontsize=9)
    ax_s2.set_ylabel("ROC-AUC", fontsize=10)
    ax_s2.set_ylim(0.55, 0.77)
    ax_s2.legend(loc="upper left", fontsize=9)
    ax_s2.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    plt.savefig("subgroup_progression.png", dpi=300, bbox_inches="tight")
    print("[+] Saved subgroup progression audit to subgroup_progression.png")
    print("=" * 90)


if __name__ == "__main__":
    main()
