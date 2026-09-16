# Trust Card: ML and AI Nexus 2026 Final20 Model

The current Trust Card is the frozen final20 calibrated ensemble. The detailed, figure-backed version is maintained at [`deliverables_final/TRUST_CARD.md`](deliverables_final/TRUST_CARD.md) and converted to [`output/pdf/TRUST_CARD_FINAL20.pdf`](output/pdf/TRUST_CARD_FINAL20.pdf).

## Current champion

- Model: fixed equal-weight ensemble of regularized logistic regression, cubic-spline logistic regression, LightGBM, CatBoost, and EBM, with inner-OOF sigmoid calibration.
- Validation: 20 stratified outer folds, 3 inner folds, 95% training fraction, seed `20260914`.
- Local nested OOF log loss: **0.3487788**.
- Local Brier score: **0.1020847**.
- Local ROC-AUC: **0.69366**.
- User-reported public Kaggle score: **0.33507**.

The 20-fold five-seed candidate is not selected because it has not been tested on Kaggle. v6-derived outputs and blends are excluded because they underperformed on the public leaderboard.

Use the detailed card in `deliverables_final` for calibration, fold stability, subgroup reliability, uncertainty referral, failure modes, reproducibility, and the final20-specific figures.
