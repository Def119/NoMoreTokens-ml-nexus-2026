# Trust Card: ML and AI Nexus 2026 Final20 Model

**Decision:** Use the frozen final20 calibrated ensemble as the current champion and as the reference for future experiments. The untested 20-fold five-seed candidate is intentionally not selected here.

## 1. Model summary

The selected model is a fixed, equal-weight ensemble of regularized logistic regression, cubic-spline logistic regression, LightGBM, CatBoost, and EBM. A sigmoid logistic calibrator is fit within each outer training partition and applied to the held-out prediction. Twenty outer models are averaged, with each model trained on 95% of the supplied rows. IDs are excluded; no external data or pretrained models are used.

The five component configurations and calibration setting were frozen before this final20 run. This keeps the selected model reproducible, while acknowledging that the choice of the 20-fold campaign was made after earlier results had been inspected.

## 2. Validation design

The campaign uses 20 stratified outer folds and three inner folds inside each outer-training partition, with seed 20260914. Preprocessing, category vocabularies, numeric medians, feature representations, tree stopping iterations, and calibration are learned only from the relevant training partition. Outer labels score only their held-out rows.

The training set contains 7,000 rows and 881 positive labels (12.59%). The outer-fold predictions are saved in `runs/final20_v11/oof.csv` and are the basis for the metrics below.

## 3. Predictive performance

On the nested out-of-fold predictions, binary log loss is **0.3487788**, Brier score is **0.1020847**, ROC-AUC is **0.69366**, and average precision is **0.28188**. The uncalibrated equal-weight log loss is 0.3488812; calibration improves it by 0.0001024 on these OOF predictions.

At a probability threshold of 0.15, reported for operational context rather than leaderboard optimization, OOF sensitivity is **0.511** and specificity is **0.765**. The competition metric remains log loss, so this threshold should not be used to judge the Kaggle submission.

The user-reported public Kaggle score for the final20 upload is **0.33507**. This value is included as external leaderboard context and was not generated or independently verified by the repository workflow.

## 4. Calibration

The OOF calibration slope is **0.9864** and intercept is **-0.0252**. Calibration is fit from inner OOF predictions inside each outer training partition, not from the outer holdout labels. Figure `figures_final20/final20_calibration.png` shows the reliability curve and probability distribution. Calibration is not clinical validation.

## 5. Robustness

Across the 20 held-out folds, calibrated log loss has mean **0.3487788**, standard deviation **0.0125862**, minimum **0.3107431**, and maximum **0.3813199**. Figure `figures_final20/final20_fold_stability.png` shows fold-level variation.

Against the related fixed 10-fold calibrated recipe, the final20 paired OOF difference is **-0.0001703** with a conditional paired-bootstrap 95% interval of **[-0.0006262, +0.0002724]**. The interval includes zero, so the local improvement is not a claim of statistical superiority. The interval conditions on saved OOF predictions and omits training/tuning uncertainty, overlapping-fold dependence, and adaptive research choices.

## 6. Subgroup reliability

Final20-specific subgroup counts, prevalence, log loss, and Brier scores are in `final20_subgroup_metrics.csv`, covering sex, rurality, hospital type, and age group when those columns are present. Figure `figures_final20/final20_subgroup_logloss.png` summarizes the subgroup log loss values and sample sizes.

Subgroup differences can reflect case mix and sampling uncertainty; they do not establish fairness, discrimination, or causation. Dedicated confidence intervals and prospective subgroup monitoring are still required before any real-world use.

## 7. Uncertainty and human referral

Uncertainty is defined by predictive entropy on final20 OOF probabilities. Referring the 10% most uncertain OOF cases leaves 6,300 cases with log loss **0.3171099**, Brier score **0.0888771**, and ROC-AUC **0.63474**. The retained set contains 90.0% of rows; the referred set has observed prevalence 35.0%.

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

The frozen source run is `runs/final20_v11`. Its `manifest.json`, `resolved_config.json`, `summary.json`, fold results, OOF predictions, cached predictions, and fitted models are retained. The recommended submission is `runs/final20_v11/submission_equal_cal.csv`; SHA-256 is `6982221b0fb5e9d8098915f8352bcd2bc681d73dac0584ab325f19e346f05f11`. The reproducible figure and report builder is `scripts/final20_trust_card_figures.py`. Seed: 20260914. AI assistance: this refinement used OpenAI Codex; the team must understand and defend the work.

## 13. One-sentence conclusion

We trust this model only when its frozen recipe, calibration, uncertainty limits, subgroup behavior, and current data distribution are checked, and when it is used as decision support with human oversight rather than as an autonomous clinical decision.

### Figure index

- `figures_final20/final20_calibration.png` - reliability curves and probability distribution.
- `figures_final20/final20_fold_stability.png` - held-out log loss across all 20 outer folds.
- `figures_final20/final20_threshold_tradeoff.png` - sensitivity and specificity as the reporting threshold changes.
- `figures_final20/final20_subgroup_logloss.png` - final20 OOF log loss by observed subgroup.
