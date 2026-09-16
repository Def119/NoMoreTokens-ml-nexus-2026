# Future Improvements & Strategic Roadmap: Path to #1 on Kaggle

> Historical roadmap. Its claims of "zero overfitting risk," guaranteed benefits from seed averaging, and underconfidence inferred from the CV-to-leaderboard gap are not established. The new `nexus` workflow tests those assumptions using nested validation, corrected preprocessing, and fresh model comparisons. Current outputs and evidence are stored under `runs/` and explained in `README.md`.

This document outlines the prioritized improvements for the 30-day hospital readmission prediction pipeline, specifically aligned with the **Binary Log Loss competition metric** and the **Trust Card requirements** from `consideration.md`.

---

## 1. Executive Summary & Current Competitive Standing

Our baseline-to-ensemble progression has brought us to the very top echelon of the competition leaderboard:
- **Baseline OOF Log Loss:** `0.35529` (naive prior baseline was `0.37849`).
- **Feature Engineering (v2):** `0.35113` (31 clinical features, shallow tree regularization).
- **Ensemble (v3):** `0.34968` (LightGBM + CatBoost + Logistic Regression).
- **Optuna Tuned Ensemble (v4):** **`0.34915` OOF Log Loss** | **`0.69443` ROC-AUC** | **`0.10233` Brier**.
- **Kaggle Public Leaderboard Submission (`submission_v4.csv`):** **`0.33574`**.
- **Leaderboard Standings:**
  - **OC Benchmark / Test Submission:** `0.33518` (Δ `0.00056` ahead).
  - **Top 2 Competing Teams:** `0.33538` (Δ `0.00036` ahead).
  - **Our Current Score:** **`0.33574`**.
- **Strategic Focus:** Because daily Kaggle submissions are strictly limited, all next-generation modifications must first show a statistically confirmed drop in local 5-fold CV (from `0.34915` $\rightarrow$ `< 0.3480`) before burning a submission.

---

## 2. Completed Phases

### Phase 1: Clinical & Tabular Feature Engineering (`v2.py`) ✅ Complete
- [x] **Informative Missingness Flags:** Binary indicators for `hemoglobin`, `creatinine`, `sodium`, `heart_rate`, `systolic_bp`, and `followup_days`, plus aggregate `total_missing_labs` count.
- [x] **Clinical Thresholds & Risk Indices:**
  - Sex-specific anemia flag (`hemoglobin < 12/13`).
  - Renal impairment (`creatinine > 1.3`) and interaction with `chronic_kidney_disease`.
  - 3-stage hypertension flags (`systolic_bp >= 130/140/160`).
  - Tachycardia (`> 100`) and Bradycardia (`< 60`).
  - Hyponatremia flag (`sodium < 135`).
- [x] **Healthcare Utilization & Exposure Ratios:**
  - Cumulative bed exposure (`total_bed_days = prior_admissions * length_of_stay`).
  - Comorbidity velocity (`comorbidity_per_age = comorbidity / age`).
  - Medication density (`meds_per_condition = meds / (comorbidities + 1)`).
  - Access barrier (`missed_appointments * rurality`).
  - Discharge vulnerability (`Home_with_support * (1 - socioeconomic_index/2)`).

### Phase 2: Tree Regularization & Architecture Dynamics (`v2.py`, `tune_optuna.py`) ✅ Complete
- [x] **Constrained Tree Complexity:** Reduced `num_leaves` to 16–23 and fixed `max_depth=4` to eliminate noise fitting.
- [x] **Smoothed Leaf Estimates:** Increased `min_child_samples` to 80–140 patients per leaf.
- [x] **Subsampling & Shrinkage:** `colsample_bytree = 0.61 - 0.75`, `subsample = 0.80 - 0.84`, `learning_rate = 0.014 - 0.033`.
- [x] **L1/L2 Penalties:** Heavy regularization applied (`reg_lambda = 8.19` on LightGBM; `l2_leaf_reg = 1.17` on CatBoost).

### Phase 3: Model Diversity & Bayesian Tuning (`v3.py`, `tune_optuna.py`) ✅ Complete
- [x] **Multi-Model Diversity:** Leaf-wise trees (LightGBM) + symmetric oblivious trees (CatBoost) + sparse ElasticNet linear probabilities (Logistic Regression).
- [x] **Bayesian TPE Optimization:** 75 Optuna trials across all 3 model families.
- [x] **Synergistic Blend:** Reached equal triad weights (33.3% LightGBM / 33.3% CatBoost / 33.3% Logistic Regression).

### Phase 4: Trust Card & Responsible AI Tasks (`explainability.py`, `trust_card.md`) ✅ Complete
- [x] **Subgroup Audit:** Proven monotonic improvement and demographic parity across Sex, Rurality, and Hospital Type.
- [x] **10% Uncertainty Referral Simulation:** Demonstrated that isolating the top 10% highest-entropy cases captures a cohort with 35.29% readmission rate and lowers automated Log Loss to `0.31631`.
- [x] **TreeSHAP Global & Local Interpretability:** Top predictors identified (6 of top 10 are engineered features) and local waterfall plots generated for high- and low-risk cases.

---

## 3. Phase 5: v5 Champion Pipeline (`v5.py`) ✅ Complete

Implemented four precision enhancements plus two additional strategies to push OOF from `0.34915` → **`0.34879`** (−0.00036):

### A. Contextual & Institutional Group Aggregations (Relative Clinical Features)
- [x] `los_rel_hospital` — LOS / mean(LOS by hospital_type)
- [x] `meds_rel_comorbidity` — meds / mean(meds by comorbidity_count)
- [x] `admissions_rel_pathway` — prior_admissions − mean(admissions by care_pathway)
- [x] `age_rel_pathway` — age − mean(age by care_pathway)
- [x] `socioeconomic_rel_region` — socioeconomic_index − mean(socioeconomic by region) *(new)*
- [x] `followup_rel_pathway` — followup_days − mean(followup by care_pathway) *(new)*
- **All fold-aware** (means computed on training fold only; leak-proof).

### B. Advanced Probability Ensembling in Logit Space
- [x] Implemented SLSQP in both logit-space and probability-space.
- **Result:** Nearly identical (Δ 0.000008). Models are well-calibrated enough that blending space doesn't matter.
- Prob-space used as final (marginally better).

### C. Post-Ensemble Calibration & Prior Adaptation
- [x] Platt scaling on blended OOF log-odds: slope ~1.11, saved **0.000065** LL.
- [x] Conservative clipping at $\epsilon = 10^{-5}$.

### D. Seed Averaging (NEW — not in original blueprint)
- [x] 3 seeds (42, 2026, 7777) × 3 models = 9 sub-models per fold, 45 total.
- **CatBoost gained the most:** −0.00068 LL from seed averaging (oblivious trees have high seed variance on small data).

### E. Chronic × Lab Interaction Features (NEW)
- [x] `hf_tachycardia`, `ckd_hyponatremia`, `diabetes_renal`.

### F. XGBoost Re-integration
- [x] **Decision: NOT included.** XGBoost was redundant with LightGBM in v3 (weight ≈ 0) and would dilute the clean 3-model ensemble. Keeping it out is also better for Trust Card interpretability.

---

## 4. Phase 6: v6 Information-Extraction (`v6.py`) ⚠️ OVERFIT — LB Regression

### Results:
- **Local OOF LL:** 0.34763 (−0.00116 vs v5 ✅)
- **Kaggle LB:** `0.33609` (**+0.00035 worse** than v4's `0.33574` ❌)
- **Verdict:** Local CV improved but LB regressed. Classic overfitting to training distribution.

### What Was Tried & Outcome:

| Enhancement | Local Impact | LB Impact | Verdict |
| :--- | :--- | :--- | :--- |
| Target Encoding (4 features, m=20) | LGB −0.00160 LL ✅ | 🔴 Suspected overfit | **❌ ABANDON** — few-category TE encodes noisy group rates |
| 8-Fold CV (vs 5-fold) | Trees trained deeper (694 iters) ✅ | 🟡 Unclear | **⚠️ TEST SEPARATELY** |
| Feature Pruning (26 dropped → 41 total) | Fewer noise dims ✅ | 🟡 May have removed regularising features | **❌ ABANDON** — LGB importance ≠ CatBoost/LR importance |
| Ridge Meta-Learner (unconstrained) | −0.000126 vs SLSQP ✅ | 🟡 Coefs >1.0 + intercept rescale | **❌ REVERT to SLSQP** — sum-to-1 constraint is safer |
| Post-Ensemble Platt Cal | Hurt (+0.000226) | N/A | Correctly skipped |

### Why "More Training" Won't Work (Overfitting Analysis)
With **7,000 rows, 881 positives, and 5,600 training samples per 5-fold split**, the information ceiling is real:
- Trees stop at 200–400 iterations because there's no more signal to extract at the current regularisation level.
- Forcing deeper training requires loosening regularisation (more leaves, lower min_child_samples), which fits noise → **hurts generalisation on 3,000 test samples**.
- The v4→v5 gain (−0.00036) is smaller than v3→v4 (−0.00053), confirming **diminishing returns**.
- The CV-to-LB gap (0.34915 → 0.33574) shows our models are already **under-confident** (test is easier), which is the healthy direction. Overfit models show the opposite.

### 🔑 Key Lesson Learned:
**Local CV improvement ≠ LB improvement.** Any enhancement that injects target-dependent information (even fold-aware target encoding) or removes features that regularise secondary models is risky on 7,000-sample datasets. The safest gains come from **target-independent** strategies: seed averaging, constrained blending, and simple ratio/diff features.

---

## 5. Phase 7: v7 Conservative Recovery — v4 Base + Safe Enhancements Only

Instead of chasing aggressive local CV gains, v7 returns to our **proven LB base** (v4: `0.33574`) and applies ONLY enhancements with **zero overfitting risk**.

### Strategy: "Do No Harm" — Additive Variance Reduction

| Enhancement | Why it's safe | Expected Impact |
| :--- | :--- | :--- |
| **Seed averaging (3 seeds)** | Pure variance reduction across random initialisations. Cannot inject false signal. | 0.0002–0.0005 LL |
| **Contextual aggregations (6 features)** | Simple ratio/diff of raw features vs group means. No target leakage. | 0.0001–0.0003 LL |
| **SLSQP blending (constrained)** | Sum-to-1 constraint prevents probability rescaling. Known-safe from v4. | Baseline (proven) |
| **Full feature set (NO pruning)** | Keep all 54 v2 features. Low-importance features may regularise CatBoost/LR. | Safety |
| **5-Fold CV (proven)** | Same splits as v4 for direct LB comparability. | Baseline (proven) |

### What NOT to include:
- [ ] ~~Target encoding~~ — confirmed overfit risk on few-category features.
- [ ] ~~Feature pruning~~ — LGB importance ≠ ensemble importance.
- [ ] ~~Ridge/unconstrained meta-learner~~ — coefs >1.0 rescale probabilities.
- [ ] ~~8-fold CV~~ — test separately if needed, but not with other changes.

### Additional Safe Ideas:
- [ ] **Logit-space SLSQP blending:** Compare logit vs prob-space. In v5 they were nearly identical, but worth re-checking with new features.
- [ ] **Post-ensemble Platt calibration:** Test if small calibration slope adjustment helps after SLSQP blend.
- [ ] **Repeated CV significance test (3×5-fold):** Before submitting, verify the improvement is statistically significant (>2× standard deviation across repeats).

### Target:
- Local CV: Improve from v4's `0.34915` while staying conservative.
- **LB: Must beat `0.33574`** — any change that doesn't must be reverted.

---

## 6. Implementation Milestones

| Milestone | Action Items | Success Metric (Local CV) | Kaggle Public LB | Status |
| :--- | :--- | :--- | :--- | :--- |
| **M0: Baseline** | 5-Fold Stratified LightGBM + Calibration | Log Loss: `0.35527`, Brier: `0.10378` | ~0.342 | **Complete** |
| **M1: Diagnostics** | Generate ROC, PR, Calibration, and Subgroup plots | Visual inspection of weaknesses | — | **Complete** |
| **M2: Feature Eng.** | Missingness flags + clinical risk ratios + interaction terms (`v2.py`) | Log Loss: `0.35113`, ROC-AUC: `0.68505` | ~0.338 | **Complete** |
| **M3: Model Diversity** | Train CatBoost + XGBoost + Logistic Regression (`v3.py`) | 4 diverse model families | — | **Complete** |
| **M4: Blending** | SLSQP-optimised blend weights minimising OOF Log Loss (`v3.py`) | Log Loss: `0.34968`, ROC-AUC: `0.69256` | — | **Complete** |
| **M5: Trust Card** | Subgroup audits, 10% uncertainty referral, SHAP explanations (`trust_card.md`) | Full report per `consideration.md` | — | **Complete** |
| **M6: Optuna Tuning** | Bayesian TPE tuning of LGBM, CatBoost, LogReg (`tune_optuna.py`) | Log Loss: `0.34915`, ROC-AUC: `0.69443` | **`0.33574`** | **Complete** |
| **M7: v5 Champion** | Contextual aggs + chronic×lab + 3-seed avg + Platt cal (`v5.py`) | Log Loss: **`0.34879`**, ROC-AUC: `0.69473` | — | **Complete** |
| **M8: v6 Final Push** | Target encoding + 8-fold CV + feature pruning + ridge stacking | Log Loss: **`0.34763`** | **`0.33609`** ⚠️ Regressed | **⚠️ Overfit** |
| **M9: v7 Recovery** | v4 base + seed avg + contextual aggs + SLSQP (safe only) | Target: **`< 0.3488`** | Target: **`< 0.33574`** | **Up Next** |
