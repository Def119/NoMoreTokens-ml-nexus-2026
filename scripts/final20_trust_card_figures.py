"""Build final20-specific evidence, figures, Trust Card markdown, and notebook.

This script uses only the frozen final20 OOF predictions and the supplied training
labels. It deliberately does not use v6, target-encoded outputs, or the untested
20-fold five-seed candidate.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "final20_v11"
DEST = ROOT / "deliverables_final"
FIG = DEST / "figures_final20"
OUT_PDF = ROOT / "output" / "pdf"
TARGET = "readmitted_30d"
ID = "patient_id"
SEED = 20260914


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)


def _calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    x = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1 - 1e-7))
    model = LogisticRegression(C=1e6, max_iter=2000).fit(x.reshape(-1, 1), y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _subgroup_frame(train: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    merged = train.merge(oof[[ID, "equal_cal"]], on=ID, validate="one_to_one")
    specs: list[tuple[str, str]] = []
    for col in ["sex", "rurality", "hospital_type"]:
        if col in merged.columns:
            specs.append((col, col))
    if "age" in merged.columns:
        merged["age_group"] = pd.cut(
            merged["age"], bins=[-np.inf, 39, 59, 74, np.inf], labels=["<40", "40-59", "60-74", "75+"]
        )
        specs.append(("age_group", "age group"))
    rows = []
    for col, label in specs:
        for value, frame in merged.groupby(col, dropna=False, observed=False):
            y = frame[TARGET].to_numpy()
            p = frame["equal_cal"].to_numpy()
            rows.append(
                {
                    "variable": label,
                    "group": "Missing" if pd.isna(value) else str(value),
                    "n": int(len(frame)),
                    "positives": int(y.sum()),
                    "prevalence": float(y.mean()),
                    "log_loss": float(log_loss(y, p, labels=[0, 1])),
                    "brier": float(brier_score_loss(y, p)),
                }
            )
    return pd.DataFrame(rows)


def build_evidence() -> dict:
    train = pd.read_csv(ROOT / "train.csv")
    oof = pd.read_csv(RUN / "oof.csv")
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    if not oof[ID].equals(train[ID]) or not oof[TARGET].equals(train[TARGET]):
        raise ValueError("final20 OOF rows are not aligned to train.csv")
    y = oof[TARGET].to_numpy(dtype=int)
    p_raw = oof["equal"].to_numpy(dtype=float)
    p = oof["equal_cal"].to_numpy(dtype=float)
    threshold = 0.15
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    entropy = _entropy(p)
    cutoff = np.quantile(entropy, 0.90)
    referred = entropy >= cutoff
    retained = ~referred
    slope, intercept = _calibration_slope_intercept(y, p)
    fold = oof.groupby("fold", as_index=False).apply(
        lambda f: pd.Series(
            {
                "equal_log_loss": log_loss(f[TARGET], f["equal"], labels=[0, 1]),
                "equal_cal_log_loss": log_loss(f[TARGET], f["equal_cal"], labels=[0, 1]),
                "n": len(f),
            }
        ),
        include_groups=False,
    ).reset_index().rename(columns={"level_0": "fold"})
    fold["fold"] = fold["fold"].astype(int)
    subgroup = _subgroup_frame(train, oof)
    subgroup.to_csv(DEST / "final20_subgroup_metrics.csv", index=False)
    fold.to_csv(DEST / "final20_fold_metrics.csv", index=False)
    evidence = {
        "recipe": "fixed equal-weight five-component ensemble with inner-OOF sigmoid calibration",
        "components": ["regularized logistic regression", "cubic-spline logistic regression", "LightGBM", "CatBoost", "EBM"],
        "outer_folds": 20,
        "inner_folds": 3,
        "training_fraction": 0.95,
        "seed": SEED,
        "n_train": int(len(train)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "metrics": {
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "uncalibrated_log_loss": float(log_loss(y, p_raw, labels=[0, 1])),
        },
        "calibration": {"slope": slope, "intercept": intercept},
        "threshold_0_15": {
            "threshold": threshold,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        },
        "referral_10_percent": {
            "referred_n": int(referred.sum()),
            "retained_n": int(retained.sum()),
            "retained_fraction": float(retained.mean()),
            "referred_prevalence": float(y[referred].mean()),
            "retained_log_loss": float(log_loss(y[retained], p[retained], labels=[0, 1])),
            "retained_brier": float(brier_score_loss(y[retained], p[retained])),
            "retained_auc": float(roc_auc_score(y[retained], p[retained])),
        },
        "fold_log_loss": {
            "mean": float(fold.equal_cal_log_loss.mean()),
            "std": float(fold.equal_cal_log_loss.std(ddof=1)),
            "min": float(fold.equal_cal_log_loss.min()),
            "max": float(fold.equal_cal_log_loss.max()),
        },
        "paired_bootstrap_vs_final10": summary["paired_bootstrap"]["final10_equal_cal"]["equal_cal"],
        "public_kaggle_score": 0.33507,
        "public_kaggle_score_note": "User-reported public leaderboard result; not produced or independently verified by this repository.",
        "submission_sha256": hashlib.sha256((RUN / "submission_equal_cal.csv").read_bytes()).hexdigest(),
    }
    (DEST / "final20_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence, train, oof, fold, subgroup


def make_figures(train: pd.DataFrame, oof: pd.DataFrame, fold: pd.DataFrame, subgroup: pd.DataFrame) -> None:
    _set_style()
    y = oof[TARGET].to_numpy()
    p_raw = oof["equal"].to_numpy()
    p = oof["equal_cal"].to_numpy()
    colors = {"raw equal-weight": "#9aa5b1", "calibrated equal-weight": "#0f766e"}

    # Calibration and prediction distribution.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})
    for name, prob in [("raw equal-weight", p_raw), ("calibrated equal-weight", p)]:
        frac, mean = calibration_curve(y, prob, n_bins=10, strategy="quantile")
        ax1.plot(mean, frac, marker="o", linewidth=2, label=name, color=colors[name])
    ax1.plot([0, 1], [0, 1], "--", color="#374151", linewidth=1, label="ideal")
    ax1.set(title="OOF calibration", xlabel="Mean predicted probability", ylabel="Observed event rate")
    ax1.legend(frameon=False, fontsize=8)
    ax2.hist(p_raw, bins=20, alpha=0.45, label="raw", color=colors["raw equal-weight"])
    ax2.hist(p, bins=20, alpha=0.55, label="calibrated", color=colors["calibrated equal-weight"])
    ax2.set(title="OOF probability distribution", xlabel="Predicted probability", ylabel="Patients")
    ax2.legend(frameon=False, fontsize=8)
    fig.suptitle("Final20 probability calibration", x=0.03, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "final20_calibration.png")

    # Fold stability.
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.plot(fold["fold"], fold["equal_log_loss"], marker="o", label="raw equal-weight", color=colors["raw equal-weight"])
    ax.plot(fold["fold"], fold["equal_cal_log_loss"], marker="o", label="calibrated equal-weight", color=colors["calibrated equal-weight"])
    ax.axhline(fold["equal_cal_log_loss"].mean(), color=colors["calibrated equal-weight"], linestyle="--", linewidth=1)
    ax.set(title="Held-out log loss across the 20 outer folds", xlabel="Outer fold", ylabel="Binary log loss")
    ax.set_xticks(fold["fold"])
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "final20_fold_stability.png")

    # Threshold operating characteristics.
    thresholds = np.linspace(0.02, 0.45, 88)
    sens, spec = [], []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(thresholds, sens, label="Sensitivity", color="#b45309", linewidth=2)
    ax.plot(thresholds, spec, label="Specificity", color="#1d4ed8", linewidth=2)
    ax.axvline(0.15, color="#374151", linestyle="--", linewidth=1, label="Reporting threshold 0.15")
    ax.set(title="Operating characteristics on final20 OOF predictions", xlabel="Probability threshold", ylabel="Rate")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    _save(fig, "final20_threshold_tradeoff.png")

    # Subgroup reliability.
    plot_df = subgroup.copy()
    plot_df["label"] = plot_df["variable"] + " = " + plot_df["group"]
    plot_df = plot_df.sort_values(["variable", "log_loss"])
    fig_h = max(4.4, 0.28 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10.0, fig_h))
    ax.barh(plot_df["label"], plot_df["log_loss"], color="#0f766e")
    for i, row in enumerate(plot_df.itertuples()):
        ax.text(row.log_loss + 0.001, i, f"n={row.n}, prev={row.prevalence:.1%}", va="center", fontsize=8)
    ax.set(title="Final20 OOF log loss by observed subgroup", xlabel="Binary log loss", ylabel="Subgroup")
    ax.set_xlim(0, max(plot_df["log_loss"].max() * 1.25, 0.4))
    fig.tight_layout()
    _save(fig, "final20_subgroup_logloss.png")


def build_trust_card(evidence: dict) -> str:
    m = evidence["metrics"]
    cal = evidence["calibration"]
    th = evidence["threshold_0_15"]
    ref = evidence["referral_10_percent"]
    fold = evidence["fold_log_loss"]
    ci = evidence["paired_bootstrap_vs_final10"]["ci95"]
    delta = evidence["paired_bootstrap_vs_final10"]["delta_log_loss"]
    return f"""# Trust Card: ML and AI Nexus 2026 Final20 Model

**Decision:** Use the frozen final20 calibrated ensemble as the current champion and as the reference for future experiments. The untested 20-fold five-seed candidate is intentionally not selected here.

## 1. Model summary

The selected model is a fixed, equal-weight ensemble of regularized logistic regression, cubic-spline logistic regression, LightGBM, CatBoost, and EBM. A sigmoid logistic calibrator is fit within each outer training partition and applied to the held-out prediction. Twenty outer models are averaged, with each model trained on 95% of the supplied rows. IDs are excluded; no external data or pretrained models are used.

The five component configurations and calibration setting were frozen before this final20 run. This keeps the selected model reproducible, while acknowledging that the choice of the 20-fold campaign was made after earlier results had been inspected.

## 2. Validation design

The campaign uses 20 stratified outer folds and three inner folds inside each outer-training partition, with seed {evidence['seed']}. Preprocessing, category vocabularies, numeric medians, feature representations, tree stopping iterations, and calibration are learned only from the relevant training partition. Outer labels score only their held-out rows.

The training set contains {evidence['n_train']:,} rows and {evidence['positive_count']:,} positive labels ({evidence['positive_rate']:.2%}). The outer-fold predictions are saved in `runs/final20_v11/oof.csv` and are the basis for the metrics below.

## 3. Predictive performance

On the nested out-of-fold predictions, binary log loss is **{m['log_loss']:.7f}**, Brier score is **{m['brier']:.7f}**, ROC-AUC is **{m['roc_auc']:.5f}**, and average precision is **{m['average_precision']:.5f}**. The uncalibrated equal-weight log loss is {m['uncalibrated_log_loss']:.7f}; calibration improves it by {m['uncalibrated_log_loss'] - m['log_loss']:.7f} on these OOF predictions.

At a probability threshold of 0.15, reported for operational context rather than leaderboard optimization, OOF sensitivity is **{th['sensitivity']:.3f}** and specificity is **{th['specificity']:.3f}**. The competition metric remains log loss, so this threshold should not be used to judge the Kaggle submission.

The user-reported public Kaggle score for the final20 upload is **0.33507**. This value is included as external leaderboard context and was not generated or independently verified by the repository workflow.

## 4. Calibration

The OOF calibration slope is **{cal['slope']:.4f}** and intercept is **{cal['intercept']:.4f}**. Calibration is fit from inner OOF predictions inside each outer training partition, not from the outer holdout labels. Figure `figures_final20/final20_calibration.png` shows the reliability curve and probability distribution. Calibration is not clinical validation.

## 5. Robustness

Across the 20 held-out folds, calibrated log loss has mean **{fold['mean']:.7f}**, standard deviation **{fold['std']:.7f}**, minimum **{fold['min']:.7f}**, and maximum **{fold['max']:.7f}**. Figure `figures_final20/final20_fold_stability.png` shows fold-level variation.

Against the related fixed 10-fold calibrated recipe, the final20 paired OOF difference is **{delta:+.7f}** with a conditional paired-bootstrap 95% interval of **[{ci[0]:+.7f}, {ci[1]:+.7f}]**. The interval includes zero, so the local improvement is not a claim of statistical superiority. The interval conditions on saved OOF predictions and omits training/tuning uncertainty, overlapping-fold dependence, and adaptive research choices.

## 6. Subgroup reliability

Final20-specific subgroup counts, prevalence, log loss, and Brier scores are in `final20_subgroup_metrics.csv`, covering sex, rurality, hospital type, and age group when those columns are present. Figure `figures_final20/final20_subgroup_logloss.png` summarizes the subgroup log loss values and sample sizes.

Subgroup differences can reflect case mix and sampling uncertainty; they do not establish fairness, discrimination, or causation. Dedicated confidence intervals and prospective subgroup monitoring are still required before any real-world use.

## 7. Uncertainty and human referral

Uncertainty is defined by predictive entropy on final20 OOF probabilities. Referring the 10% most uncertain OOF cases leaves {ref['retained_n']:,} cases with log loss **{ref['retained_log_loss']:.7f}**, Brier score **{ref['retained_brier']:.7f}**, and ROC-AUC **{ref['retained_auc']:.5f}**. The retained set contains {ref['retained_fraction']:.1%} of rows; the referred set has observed prevalence {ref['referred_prevalence']:.1%}.

This is a retrospective simulation of a review queue, not proof that referral improves safety. The entropy cutoff and reported performance should be re-estimated prospectively.

## 8. Explainability

The existing SHAP and sensitivity assets are component-level diagnostics for a related five-fold recipe, not complete explanations of every final20 ensemble prediction. They may be used as contextual figures only when labeled that way. The final20 package therefore emphasizes reproducible calibration, fold stability, threshold trade-offs, and subgroup log loss rather than presenting component explanations as final-model explanations.

Predictive association is not causation. Correlated variables can share or hide importance, and a local explanation does not establish that changing a feature would change the outcome.

## 9. Model comparison

The final20 recipe remains the champion for this package because it is the best model with a user-confirmed public Kaggle score of 0.33507. The 20-fold five-seed model has lower local OOF log loss in the research log but has not been tested on Kaggle, so it is not promoted here. The later 30-fold five-seed run was effectively tied with that untested candidate and is also not selected. v6 and any v6-derived blend are excluded because the user reported that they underperformed on Kaggle.

## 10. Failure modes and limitations

The model may fail when the deployment population differs from the synthetic training distribution, when missingness mechanisms or care pathways change, or when rare combinations are underrepresented. It may also be overconfident for patients outside the training support, and subgroup estimates may be unstable where sample sizes are small. The public Kaggle score is a single leaderboard observation and is not external clinical validation. No causal, fairness, or clinical-safety claim is made.

## 11. Deployment recommendation

**Requires additional model development.** This is retrospective validation on synthetic competition data. Before patient-care use, require external validation, prospective silent-mode evaluation, calibration monitoring, subgroup review, drift checks, and clinical governance. In the competition setting, keep final20 as the fallback while any untested candidate is evaluated independently.

## 12. Reproducibility

The frozen source run is `runs/final20_v11`. Its `manifest.json`, `resolved_config.json`, `summary.json`, fold results, OOF predictions, cached predictions, and fitted models are retained. The recommended submission is `runs/final20_v11/submission_equal_cal.csv`; SHA-256 is `{evidence['submission_sha256']}`. The reproducible figure and report builder is `scripts/final20_trust_card_figures.py`. Seed: {evidence['seed']}. AI assistance: this refinement used OpenAI Codex; the team must understand and defend the work.

## 13. One-sentence conclusion

We trust this model only when its frozen recipe, calibration, uncertainty limits, subgroup behavior, and current data distribution are checked, and when it is used as decision support with human oversight rather than as an autonomous clinical decision.

### Figure index

- `figures_final20/final20_calibration.png` - reliability curves and probability distribution.
- `figures_final20/final20_fold_stability.png` - held-out log loss across all 20 outer folds.
- `figures_final20/final20_threshold_tradeoff.png` - sensitivity and specificity as the reporting threshold changes.
- `figures_final20/final20_subgroup_logloss.png` - final20 OOF log loss by observed subgroup.
"""


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# ML and AI Nexus 2026 final20 audit\n\n"
            "This notebook audits the selected final20 model only. The untested five-seed candidate and all v6-derived outputs are excluded from model selection."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json, hashlib\nimport numpy as np\nimport pandas as pd\nfrom sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, average_precision_score\n\nROOT = Path.cwd()\nwhile not (ROOT / 'train.csv').exists() and ROOT != ROOT.parent:\n    ROOT = ROOT.parent\nRUN = ROOT / 'runs' / 'final20_v11'\nevidence = json.loads((ROOT / 'deliverables_final' / 'final20_evidence.json').read_text())\ntrain = pd.read_csv(ROOT / 'train.csv')\noof = pd.read_csv(RUN / 'oof.csv')\nprint(evidence['recipe'])\nprint('Public Kaggle context:', evidence['public_kaggle_score'])"
        ),
        nbf.v4.new_code_cell(
            "assert oof.patient_id.equals(train.patient_id)\nassert oof.readmitted_30d.equals(train.readmitted_30d)\ny = oof.readmitted_30d\np = oof.equal_cal\nprint({\n    'log_loss': log_loss(y, p),\n    'brier': brier_score_loss(y, p),\n    'roc_auc': roc_auc_score(y, p),\n    'average_precision': average_precision_score(y, p),\n})"
        ),
        nbf.v4.new_code_cell(
            "folds = pd.read_csv(ROOT / 'deliverables_final' / 'final20_fold_metrics.csv')\ndisplay(folds)\nprint('Mean calibrated fold log loss:', folds.equal_cal_log_loss.mean())\nprint('Std calibrated fold log loss:', folds.equal_cal_log_loss.std(ddof=1))"
        ),
        nbf.v4.new_code_cell(
            "subgroup = pd.read_csv(ROOT / 'deliverables_final' / 'final20_subgroup_metrics.csv')\ndisplay(subgroup)"
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import display, Image, Markdown\ndisplay(Markdown((ROOT / 'deliverables_final' / 'TRUST_CARD.md').read_text()))\nfor name in ['final20_calibration.png', 'final20_fold_stability.png', 'final20_threshold_tradeoff.png', 'final20_subgroup_logloss.png']:\n    display(Image(filename=str(ROOT / 'deliverables_final' / 'figures_final20' / name)))"
        ),
        nbf.v4.new_code_cell(
            "sample = pd.read_csv(ROOT / 'sample_submission.csv')\nsubmission = pd.read_csv(RUN / 'submission_equal_cal.csv')\nassert submission.columns.tolist() == sample.columns.tolist()\nassert submission.patient_id.equals(sample.patient_id)\nassert len(submission) == len(sample) == 3000\nassert np.isfinite(submission.readmitted_30d).all()\nassert submission.readmitted_30d.between(0, 1).all()\nprint('Final20 submission validated for manual Kaggle upload.')"
        ),
        nbf.v4.new_markdown_cell(
            "## Reproduction\n\n"
            "Run `python scripts/final20_trust_card_figures.py` from the repository root to rebuild the evidence tables, figures, Trust Card, and this notebook. The frozen model campaign itself is retained under `runs/final20_v11`."
        ),
    ]
    nbf.write(nb, DEST / "final20_submission_notebook.ipynb")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    OUT_PDF.mkdir(parents=True, exist_ok=True)
    evidence, train, oof, fold, subgroup = build_evidence()
    make_figures(train, oof, fold, subgroup)
    (DEST / "TRUST_CARD.md").write_text(build_trust_card(evidence), encoding="utf-8")
    build_notebook()
    print(json.dumps({"dest": str(DEST), "figures": sorted(p.name for p in FIG.glob("*.png")), "log_loss": evidence["metrics"]["log_loss"]}, indent=2))


if __name__ == "__main__":
    main()
