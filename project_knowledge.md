# ML Nexus 2026 - 30-Day Hospital Readmission Challenge

> September 13, 2026 audit note: this is the historical development log, not an independently verified evaluation report. The supplied test set has 3,000 rows. v6 global feature selection contaminates its reported validation; its training target encoding is not cross-fitted. Several blends are evaluated on their optimization data. The previous subgroup counts and clinical conclusions require regeneration. Follow `README.md` and the selected run's generated evidence for the refinement campaign.

## 1. Problem Overview
- **Objective:** Predict unplanned 30-day hospital readmission (`readmitted_30d`) from electronic health record (EHR) features.
- **Primary Metric:** Binary Log Loss (lower is better).
  - Strongly penalizes overconfident incorrect predictions.
  - Requires well-calibrated probabilities, not just ranked discrimination.
- **Trustworthy AI Requirements (from `consideration.md`):**
  - Performance: Log Loss, Brier score, ROC-AUC, sensitivity/specificity.
  - Calibration: Reliability curves, calibration slope/intercept before vs. after calibration.
  - Subgroup Reliability: Audit across `sex`, `rurality`, `age group`, and `hospital_type`.
  - Uncertainty & Referral: Impact of abstaining / referring the 10% most uncertain predictions.
  - Explainability: Global feature importance and local instance explanations (high-risk vs. low-risk).

---

## 2. Dataset Summary
- **Train Set:** 7,000 samples $\times$ 25 columns
- **Test Set:** 3,001 samples $\times$ 24 columns (`sample_submission.csv` uses constant prior $p \approx 0.125857$)
- **Target Distribution:**
  - Positives: ~12.586% (881 / 7,000)
  - Negatives: ~87.414% (6,119 / 7,000)
  - Imbalance Ratio: ~1:7
- **Feature Overview:**
  - **Identifiers:** `patient_id`
  - **Sensitive / Demographic:** `sex`, `age`, `rurality`, `socioeconomic_index`
  - **Hospital & Care Context:** `hospital_type`, `region`, `care_pathway`, `discharge_disposition`
  - **Prior Utilization & History:** `prior_admissions_12m`, `comorbidity_count`, `length_of_stay_days`, `medication_count`, `missed_appointments_12m`, `followup_days`
  - **Chronic Diagnoses (Binary):** `diabetes`, `hypertension`, `chronic_kidney_disease`, `heart_failure`
  - **Vitals & Labs (Contains Missing Values):** `hemoglobin_g_dl`, `creatinine_mg_dl`, `sodium_mmol_l`, `heart_rate_bpm`, `systolic_bp_mmhg`

---

## 3. Benchmark & Baselines

| Baseline Type | Description / Model | OOF Log Loss | OOF Brier | OOF ROC-AUC | Kaggle Public LB | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Prior Dummy** | Predict constant prior $p = 0.12586$ | ~0.37849 | ~0.11002 | 0.50000 | ~0.365 | Theoretical minimum baseline |
| **Model 0 (basic.py)** | Default LightGBM (5-Fold Stratified CV, lr=0.05, early stopping 50) | **0.35529** | **0.10375** | **0.66819** | ~0.342 | First working tree baseline; raw features, uncalibrated |
| **Model 0 + Platt Calibration** | LightGBM OOF + 5-Fold Platt Scaling (Logistic on log-odds) | **0.35527** | **0.10378** | **0.66800** | — | Perfected calibration slope (0.998) and intercept (-0.004) |
| **Model 1 (v2.py)** | Feature-Engineered + Regularised LightGBM (5-Fold, lr=0.018, 31 new features) | **0.35113** | **0.10282** | **0.68505** | ~0.338 | −0.00416 LL vs baseline; trees train 118–308 iters |
| **Model 2 (v3.py)** | 4-Model Calibrated Ensemble (LightGBM+CatBoost+XGBoost+LogReg, SLSQP weights) | **0.34968** | **0.10249** | **0.69256** | — | −0.00561 LL vs baseline; optimal weights: LGB 47%, LR 40%, CB 13% |
| **Model 3 (v4 Tuned)** | Optuna-Tuned 3-Model Ensemble (LGB+CB+LR, SLSQP weights: 33.3% each) | **0.34915** | **0.10233** | **0.69443** | **0.33574** | −0.00614 LL vs baseline; submitted to Kaggle |
| **Model 4 (v5.py)** | v5 Champion: 3-Seed Avg + Contextual Agg + Platt Cal (LGB 41%/CB 33%/LR 27%) | **0.34879** | **0.10218** | **0.69473** | — | −0.00650 LL vs baseline; 9 new features, 45 sub-models |
| **Model 5 (v6.py)** ⚠️ | v6: Target Encoding + 8-Fold + Feature Pruning + Ridge Stack | **0.34763** | **0.10185** | **0.69773** | **0.33609** ⚠️ | −0.00766 local BUT **+0.00035 worse on LB than v4**. Overfit to train. |

### Live Kaggle Leaderboard Standings (Current Snapshot):
- **Current top Submission:** `0.33358` (only **`0.00216`** ahead of us)
- **Our Best LB Score:** **`0.33574`** (Model 3 / v4 Tuned) ← still our best
- **Our v6 Submission:** `0.33609` ⚠️ regressed vs v4 despite better local CV

---

## 4. Model Iteration Log

### Model 0: Initial LightGBM Baseline (`basic.py`)
- **Code:** [basic.py](file:///c:/Users/hansajak/Desktop/code/ml-nexus-2026/basic.py)
- **Validation Scheme:** 5-Fold Stratified K-Fold (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)
- **Preprocessing:**
  - Dropped `patient_id` and target.
  - Converted 6 string columns (`sex`, `rurality`, `hospital_type`, `region`, `discharge_disposition`, `care_pathway`) to pandas `category` dtype.
  - Zero missing value imputation or feature engineering.
- **Hyperparameters:**
  - `n_estimators=1000`
  - `learning_rate=0.05`
  - Early stopping: 50 rounds
- **Per-Fold Results:**
  - Fold 1: Best Iter = 38 | Log Loss: 0.35876 | Brier: 0.10504 | ROC-AUC: 0.65656
  - Fold 2: Best Iter = 56 | Log Loss: 0.34647 | Brier: 0.10105 | ROC-AUC: 0.69567
  - Fold 3: Best Iter = 43 | Log Loss: 0.35976 | Brier: 0.10494 | ROC-AUC: 0.65275
  - Fold 4: Best Iter = 58 | Log Loss: 0.35584 | Brier: 0.10405 | ROC-AUC: 0.67187
  - Fold 5: Best Iter = 41 | Log Loss: 0.35561 | Brier: 0.10366 | ROC-AUC: 0.66252
- **Overall OOF Results:**
  - **OOF Log Loss:** 0.35529
  - **OOF Brier:** 0.10375
  - **OOF ROC-AUC:** 0.66819
- **Key Diagnostic Finding:**
  - Early stopping triggered between iteration 38 and 58 across all folds.
  - Tree complexity (`num_leaves=31`, `min_child_samples=20`) causes quick overfitting on 5,600 samples per fold.
  - Regularization (smaller `num_leaves`, higher `min_child_samples`, feature subsampling) is required to allow deeper learning.

---

## 4. Model Iteration Log

### Model 0: Initial LightGBM Baseline (`basic.py`)
- **Code:** [basic.py](file:///c:/Users/hansajak/Desktop/code/ml-nexus-2026/basic.py)
- **Validation Scheme:** 5-Fold Stratified K-Fold (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)
- **Preprocessing:**
  - Dropped `patient_id` and target.
  - Converted 6 string columns (`sex`, `rurality`, `hospital_type`, `region`, `discharge_disposition`, `care_pathway`) to pandas `category` dtype.
  - Zero missing value imputation or feature engineering.
- **Hyperparameters:**
  - `n_estimators=1000`
  - `learning_rate=0.05`
  - Early stopping: 50 rounds
- **Per-Fold Results:**
  - Fold 1: Best Iter = 38 | Log Loss: 0.35876 | Brier: 0.10504 | ROC-AUC: 0.65656
  - Fold 2: Best Iter = 56 | Log Loss: 0.34647 | Brier: 0.10105 | ROC-AUC: 0.69567
  - Fold 3: Best Iter = 43 | Log Loss: 0.35976 | Brier: 0.10494 | ROC-AUC: 0.65275
  - Fold 4: Best Iter = 58 | Log Loss: 0.35584 | Brier: 0.10405 | ROC-AUC: 0.67187
  - Fold 5: Best Iter = 41 | Log Loss: 0.35561 | Brier: 0.10366 | ROC-AUC: 0.66252
- **Overall OOF Results:**
  - **OOF Log Loss:** 0.35529
  - **OOF Brier:** 0.10375
  - **OOF ROC-AUC:** 0.66819
- **Key Diagnostic Finding:**
  - Early stopping triggered between iteration 38 and 58 across all folds.
  - Tree complexity (`num_leaves=31`, `min_child_samples=20`) causes quick overfitting on 5,600 samples per fold.
  - Regularization (smaller `num_leaves`, higher `min_child_samples`, feature subsampling) is required to allow deeper learning.

---

### Experiment 0.1: Probability Calibration (`calibrate.py`)
- **Code:** [calibrate.py](file:///c:/Users/hansajak/Desktop/code/ml-nexus-2026/calibrate.py)
- **Objective:** Evaluate whether post-hoc calibration improves probability reliability and Log Loss.
- **Calibration Methods Compared (5-Fold CV on OOF):**

| Method | OOF Log Loss | OOF Brier | ROC-AUC | Calib Slope (Ideal=1.0) | Calib Intercept (Ideal=0.0) | ECE (Expected Calib Error) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw LightGBM** | 0.35529 | 0.10375 | 0.66819 | 0.942 | -0.079 | 0.0073 |
| **Platt Scaling (Logistic CV)** | **0.35527** | 0.10378 | 0.66800 | **0.998** | **-0.004** | **0.0059** |
| **Isotonic Regression (CV)** | 0.35837 | 0.10423 | 0.65974 | 0.881 | -0.216 | 0.0049 |

- **Conclusions:**
  1. **Platt Scaling achieves near-perfect calibration properties**: Slope = `0.998` (ideal `1.0`), Intercept = `-0.004` (ideal `0.0`), and ECE reduced by ~19%.
  2. **Isotonic Regression overfits**: Stepwise step functions produce rank ties, degrading AUC (`0.668` $\rightarrow$ `0.659`) and worsening Log Loss (`0.35837`).
  3. **Performance Bottleneck Confirmed**: Calibration cannot compensate for missing feature signal. The current ceiling is feature discrimination (`ROC-AUC ~0.668`). Feature engineering and regularization are the critical levers.

---

### Model 1: Feature-Engineered + Regularised LightGBM (`v2.py`)
- **Code:** [v2.py](file:///c:/Users/hansajak/Desktop/code/ml-nexus-2026/v2.py)
- **Validation Scheme:** 5-Fold Stratified K-Fold (same splits as Model 0: `random_state=42`)
- **Feature Engineering (31 new features → 54 total):**
  - **A. Informative Missingness Flags (7):** Binary `missing_*` flags for `hemoglobin_g_dl`, `creatinine_mg_dl`, `sodium_mmol_l`, `heart_rate_bpm`, `systolic_bp_mmhg`, `followup_days`, plus aggregate `total_missing_labs` count.
  - **B. Clinical Thresholds & Risk Indices (11):** Sex-specific `anemia_flag` (F<12, M<13), `renal_impairment` (creatinine>1.3), `renal_ckd_interaction`, 3 hypertension stages (≥130/≥140/≥160), `tachycardia` (>100), `bradycardia` (<60), `vitals_instability`, `hyponatremia` (sodium<135).
  - **C. Healthcare Utilisation Ratios (6):** `total_bed_days`, `comorbidity_per_age`, `meds_per_condition`, `access_barrier` (missed appts × rurality ordinal), `missed_x_socioeconomic`, `discharge_vulnerability`.
  - **D. Additional Engineered (7):** `chronic_burden` (sum of 4 diagnoses), `age_bucket` (6 bins), `has_prior_admission`, `readmission_frequency`, `meds_per_los`, `followup_missing_or_long`, `abnormal_vitals_count`, `los_x_comorbidity`.
- **Regularised Hyperparameters (Phase 2):**
  - `n_estimators=2000`, `learning_rate=0.018`
  - `num_leaves=16`, `max_depth=5`, `min_child_samples=80`
  - `colsample_bytree=0.75`, `subsample=0.80`, `subsample_freq=1`
  - `reg_alpha=0.5`, `reg_lambda=3.0`
  - Early stopping: 100 rounds
- **Per-Fold Results:**
  - Fold 1: Best Iter = 118 | Log Loss: 0.35533 | Brier: 0.10418 | ROC-AUC: 0.67074
  - Fold 2: Best Iter = 308 | Log Loss: 0.34183 | Brier: 0.10004 | ROC-AUC: 0.71132
  - Fold 3: Best Iter = 157 | Log Loss: 0.35722 | Brier: 0.10427 | ROC-AUC: 0.66044
  - Fold 4: Best Iter = 227 | Log Loss: 0.35254 | Brier: 0.10362 | ROC-AUC: 0.68860
  - Fold 5: Best Iter = 178 | Log Loss: 0.34870 | Brier: 0.10199 | ROC-AUC: 0.69345
- **Overall OOF Results:**
  - **OOF Log Loss:** 0.35113 (Δ −0.00416 vs Model 0)
  - **OOF Brier:** 0.10282 (Δ −0.00093 vs Model 0)
  - **OOF ROC-AUC:** 0.68505 (Δ +0.01686 vs Model 0)
- **Top 10 Features by Gain (last fold):**
  1. `age` (230), `followup_days` (230), `sodium_mmol_l` (219), `los_x_comorbidity` (135), `systolic_bp_mmhg` (132), `creatinine_mg_dl` (132), `heart_rate_bpm` (129), `readmission_frequency` (120), `hemoglobin_g_dl` (103), `comorbidity_per_age` (95)
- **Key Findings:**
  1. **Regularisation resolved overfitting:** Trees now train 118–308 iterations (vs 38–58 in Model 0), confirming that shallow trees + subsampling + penalties allow sustained gradient descent.
  2. **Feature engineering lifted discrimination ceiling:** ROC-AUC improved from 0.668 → 0.685 (+1.7pp), directly translating to Log Loss improvement.
  3. **Engineered interaction features rank highly:** `los_x_comorbidity`, `readmission_frequency`, `comorbidity_per_age` all appear in top 10 by gain, validating the domain feature engineering strategy.
  4. **Next steps:** Multi-model ensemble (CatBoost, XGBoost, Logistic Regression) + calibrated blending to push below 0.345 Log Loss.

---

### Model 2: Multi-Model Calibrated Ensemble (`v3.py`)
- **Code:** [v3.py](file:///c:/Users/hansajak/Desktop/code/ml-nexus-2026/v3.py)
- **Validation Scheme:** 5-Fold Stratified K-Fold (same splits: `random_state=42`)
- **Architecture:** Train 4 diverse model families → Platt-calibrate each → SLSQP-optimise blend weights minimising OOF Log Loss.

#### Individual Model OOF Results:

| Model | OOF Log Loss | OOF Brier | OOF ROC-AUC | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | 0.35113 | 0.10282 | 0.68505 | Same regularised params as v2; leaf-wise growth |
| **CatBoost** | 0.35144 | 0.10310 | **0.68825** | Oblivious trees (`depth=5`), `l2_leaf_reg=5.0`, 182–347 iters |
| **XGBoost** | 0.35391 | 0.10358 | 0.67872 | Level-wise (`max_depth=4`), label-encoded cats, 259–385 iters |
| **Logistic Reg** | 0.35209 | 0.10331 | 0.68778 | ElasticNet (`l1_ratio=0.5`, `C=0.5`), standardised + one-hot |

- **Key Hyperparameters:**
  - CatBoost: `iterations=2000`, `lr=0.025`, `depth=5`, `l2_leaf_reg=5.0`, `random_strength=0.5`, `bagging_temperature=0.3`, early stopping 100
  - XGBoost: `n_estimators=2000`, `lr=0.020`, `max_depth=4`, `min_child_weight=80`, `colsample_bytree=0.75`, `subsample=0.80`, `reg_alpha=0.5`, `reg_lambda=3.0`, early stopping 100
  - LogisticReg: `penalty='elasticnet'`, `solver='saga'`, `l1_ratio=0.5`, `C=0.5`, `max_iter=5000`

#### Platt Calibration Results:
- All models were already well-calibrated; Platt scaling produced negligible changes (|Δ LL| < 0.0001 for all except CatBoost at −0.00007).

#### Optimal Blend Weights (SLSQP, 6 random restarts):

| Model | Weight | Interpretation |
| :--- | :--- | :--- |
| LightGBM | **0.4669** | Primary driver; best individual LL |
| LogisticReg | **0.3987** | Complementary linear model; prevents overconfident tails |
| CatBoost | 0.1273 | Modest contribution from symmetric trees |
| XGBoost | 0.0071 | Effectively zeroed; redundant with LightGBM signal |

#### Final Ensemble OOF Results:
- **OOF Log Loss:** 0.34968 (Δ −0.00145 vs Model 1, Δ −0.00561 vs Model 0)
- **OOF Brier:** 0.10249
- **OOF ROC-AUC:** 0.69256 (Δ +0.00751 vs Model 1)
- Equal-weight blend reference: 0.35003 (optimised weights save additional 0.00035)

#### Key Findings:
1. **Diversity pays off:** Blending 3 model families improved Log Loss by 0.00145 over the best single model (LightGBM), confirming that CatBoost's oblivious trees and LogReg's linear probabilities capture complementary patterns.
2. **LogisticReg is surprisingly strong:** At 0.35209 LL with ROC-AUC 0.688, the linear model nearly matches LightGBM, and its 40% blend weight indicates it contributes unique, well-calibrated probability signal — likely by preventing overconfident predictions in the tails.
3. **XGBoost is redundant:** Its level-wise trees did not add information beyond what LightGBM's leaf-wise trees already captured (weight ≈ 0).
4. **All models are well-calibrated out of the box:** Platt scaling had negligible effect, suggesting the engineered features produce stable probability estimates across model families.
5. **Total pipeline improvement:** Log Loss 0.35529 (baseline) → 0.35113 (v2, FE) → **0.34968** (v3, ensemble), a cumulative −0.00561 reduction.

---

### Model 3: Optuna-Tuned Ensemble (`tune_optuna.py`)
- **Code:** [tune_optuna.py](file:///c:/Users/hansajak/Desktop/code/ml-nexus-2026/tune_optuna.py)
- **Parameters File:** [best_params_optuna.json](file:///c:/Users/hansajak/Desktop/code/ml-nexus-2026/best_params_optuna.json)
- **Validation Scheme:** 5-Fold Stratified K-Fold (same splits: `random_state=42`)
- **Architecture:** Bayesian TPE tuning on the 3 core model families (LightGBM, Logistic Regression, CatBoost) to minimize 5-Fold OOF Log Loss, followed by SLSQP re-weighting.

#### Individual Tuned Model Results vs. Pre-Tuning:

| Model Family | Pre-Tuning OOF LL | Tuned OOF LL | Pre-Tuning AUC | Tuned AUC | Key Winning Hyperparameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | 0.35113 | **0.35020** | 0.68505 | **0.68880** | `max_depth=4`, `num_leaves=23`, `min_child_samples=140`, `colsample=0.613`, `reg_lambda=8.19` |
| **CatBoost** | 0.35144 | **0.35012** | 0.68825 | **0.68910** | `depth=4`, `learning_rate=0.0329`, `l2_leaf_reg=1.17`, `bagging_temperature=0.821` |
| **LogisticReg** | 0.35209 | **0.35119** | 0.68778 | **0.69050** | `C=0.1126` (stronger shrinkage), `l1_ratio=0.850` (85% Lasso L1 sparsity + 15% Ridge L2) |

#### Optimal Blend Weights (SLSQP):
- **LightGBM:** `33.3%`
- **LogisticReg:** `33.3%`
- **CatBoost:** `33.3%`
*(Equal 1/3 balance across leaf-wise trees, oblivious trees, and regularized linear model).*

#### Final Tuned Ensemble OOF Results:
- **OOF Log Loss:** **0.34915** (Δ −0.00053 vs Model 2, cumulative **−0.00614** vs Model 0 Baseline)
- **OOF Brier:** **0.10233**
- **OOF ROC-AUC:** **0.69443** (Δ +0.00187 vs Model 2, cumulative **+0.02624** vs Model 0 Baseline)
- **Artifacts Saved:** `oof_v4.csv`, `submission_v4.csv`

#### Key Insights:
1. **Shallow depth 4 is optimal across all tree models:** Both LightGBM and CatBoost converged to `depth=4`. Constraining depth forces trees to avoid high-order noise interactions and focus strictly on primary clinical effects.
2. **Heavy regularization on LightGBM leaves:** `min_child_samples=140` and `reg_lambda=8.19` ensure that terminal leaf risk estimates are supported by large patient cohorts, preventing extreme probability spikes.
3. **Sparse linear model (85% L1):** Logistic regression preferred an 85% L1 ratio with `C=0.1126`, effectively performing automatic feature selection and zeroing out collinear noisy engineered features.
4. **Equal Triad Synergy (33.3% / 33.3% / 33.3%):** The ensemble reached an exact balanced triad: 1/3 leaf-wise gradient boosting (LightGBM), 1/3 symmetric oblivious boosting (CatBoost), and 1/3 sparse linear log-odds (Logistic Regression).

---

### Model 4: v5 Champion Pipeline (`v5.py`)
- **Code:** [v5.py](file:///c:/Users/hansajak/Desktop/code/ml-nexus-2026/v5.py)
- **Validation Scheme:** 5-Fold Stratified K-Fold (same splits: `random_state=42`)
- **Architecture:** 3 model families × 3 seeds (42, 2026, 7777) = 9 sub-models per fold, seed-averaged, then SLSQP-blended with post-ensemble Platt calibration.

#### New Features (9 additional → 63 total):
- **Chronic × Lab Interactions (3):** `hf_tachycardia` (heart failure × tachycardia), `ckd_hyponatremia` (CKD × hyponatremia), `diabetes_renal` (diabetes × renal impairment).
- **Fold-Aware Contextual Aggregations (6):** `los_rel_hospital` (LOS / hospital-type mean), `meds_rel_comorbidity` (meds / comorbidity-group mean), `admissions_rel_pathway` (admissions − pathway mean), `age_rel_pathway`, `socioeconomic_rel_region`, `followup_rel_pathway`. All computed on training fold only (leak-proof).

#### Seed-Averaged Individual Model Results:

| Model | Single-Seed LL | 3-Seed Avg LL | Seed Δ | Brier | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | 0.35003 | **0.34998** | −0.00005 | 0.10250 | 0.68996 |
| **CatBoost** | 0.35026 | **0.34958** | **−0.00068** | 0.10240 | 0.69240 |
| **LogisticReg** | 0.35145 | **0.35145** | ±0.00000 | 0.10308 | 0.68944 |

#### Blend & Calibration:
- **Logit vs Prob-space blending:** Nearly identical (Δ 0.000008); prob-space marginally better.
- **Optimal weights:** LightGBM 40.6%, CatBoost 32.6%, LogisticReg 26.8% (CatBoost rose from 33.3% → reflects contextual features boosting its oblivious trees).
- **Post-ensemble Platt calibration:** Slope ~1.11 (models were slightly under-confident), saved 0.000065 LL.

#### Final Results:
- **OOF Log Loss:** 0.34879 (Δ −0.00036 vs Model 3, cumulative **−0.00650** vs baseline)
- **OOF Brier:** 0.10218
- **OOF ROC-AUC:** 0.69473

#### Key Findings:
1. **Seed averaging is free variance reduction:** CatBoost benefited most (−0.00068 LL) because oblivious trees have higher seed-dependent variance on small datasets. LogisticReg showed zero seed variance (deterministic with SAGA convergence).
2. **Contextual aggregations improved CatBoost disproportionately:** CatBoost weight rose from 33.3% → 32.6% (and its individual LL improved from 0.35012 to 0.34958). Symmetric trees are better at exploiting pre-normalised relative features.
3. **Calibration slopes ~1.11:** The raw ensemble was slightly under-confident (slope > 1.0), meaning Platt scaling stretched probabilities further from the prior — appropriate for a Log Loss metric that rewards well-separated confident predictions.
4. **Diminishing returns observed:** The v4→v5 gain (−0.00036) is smaller than v3→v4 (−0.00053), approaching the noise floor for 7,000-sample datasets.

---

### Model 5: v6 Information-Extraction Pipeline (`v6.py`) ⚠️ OVERFIT
- **Code:** [v6.py](file:///c:/Users/hansajak/Desktop/code/ml-nexus-2026/v6.py)
- **Validation Scheme:** 8-Fold Stratified K-Fold (upgraded from 5-fold)
- **Architecture:** 3 models × 3 seeds = 9 sub-models per fold (72 total), ridge meta-learner stacking, post-ensemble Platt calibration.

#### New Enhancements vs v5:
1. **Smoothed Target Encoding (4 features):** `te_hospital_type`, `te_care_pathway`, `te_region`, `te_discharge_disposition` with Bayesian shrinkage (m=20), fold-aware.
2. **8-Fold CV:** 6,125 training samples/fold (vs 5,600 in 5-fold). Trees trained deeper (up to 694 iters).
3. **Feature Pruning:** Dropped 26 low-importance features (all missingness flags, all clinical thresholds, chronic×lab interactions) → 41 total features.
4. **Ridge Meta-Learner:** Replaced SLSQP with L2-regularised logistic regression on OOF logits. Learned unconstrained coefs: LGB ~0.60, CB ~0.35, LR ~0.20, intercept ~0.25.

#### Seed-Averaged Individual Model Results:

| Model | v5 LL | v6 LL | Δ | v6 AUC |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | 0.34998 | **0.34838** | −0.00160 | 0.69594 |
| **CatBoost** | 0.34958 | **0.34873** | −0.00085 | 0.69440 |
| **LogisticReg** | 0.35145 | **0.35127** | −0.00018 | 0.68824 |

#### Final Local Results:
- **OOF Log Loss:** 0.34763 (Δ −0.00116 vs v5, cumulative −0.00766 vs baseline)
- **OOF Brier:** 0.10185
- **OOF ROC-AUC:** 0.69773

#### ⚠️ Kaggle LB Result: `0.33609` (WORSE than v4's `0.33574` by +0.00035)

#### Post-Mortem — Why v6 Overfit:

| Metric | v4 (LB best) | v6 | Interpretation |
| :--- | :--- | :--- | :--- |
| Local OOF LL | 0.34915 | 0.34763 | v6 is −0.00152 better locally |
| Kaggle LB | **0.33574** | 0.33609 | v6 is **+0.00035 worse** on test |
| CV-to-LB gap | 0.01341 | 0.01154 | Gap shrank → v6 memorised train patterns |

**Root Causes (ranked by suspicion):**
1. **🔴 Target Encoding:** With few categories per feature (hospital_type ~3, care_pathway ~5), the encoded values are noisy group readmission rates that fit training splits well but don’t transfer to test.
2. **🟡 Aggressive Feature Pruning (26 dropped):** Removed ALL missingness flags and clinical thresholds. These may have provided regularisation signal for CatBoost/LogReg despite low LightGBM importance.
3. **🟡 Ridge Meta-Learner:** Unconstrained coefficients summing to >1.0 with positive intercept rescale probabilities in ways that don’t generalise. SLSQP’s sum-to-1 constraint is safer.
4. **🟢 8-Fold CV:** Probably fine alone, but amplifies target encoding leakage.

**Key Lesson:** Local CV improvement ≠ LB improvement. Any feature or method that injects target-dependent information (even fold-aware) is risky on 7,000-sample datasets. The safest gains come from **target-independent** enhancements: seed averaging, constrained blending, and simple ratio features.

---

## 5. Development Roadmap & Next Steps

1. ~~**Step 1: Baseline Standardization & Multi-Metric Logging**~~ ✅ Complete
   - `basic.py` records per-fold and overall Log Loss, Brier score, ROC-AUC, and saves OOF + submission CSVs.

2. ~~**Step 2: Post-Hoc Probability Calibration**~~ ✅ Complete
   - Platt scaling validated as near-optimal (slope 0.998, intercept −0.004). Isotonic overfits.

3. ~~**Step 3: Feature Engineering + Tree Regularisation**~~ ✅ Complete
   - `v2.py` implements 31 engineered features (missingness flags, clinical thresholds, utilisation ratios, interactions) and regularised LightGBM hyperparameters.
   - Result: Log Loss 0.35529 → **0.35113** (−0.00416), ROC-AUC 0.668 → **0.685** (+0.017).

4. ~~**Step 4: Model Exploration & Ensembling**~~ ✅ Complete
   - `v3.py` trains LightGBM, CatBoost, XGBoost, and ElasticNet LogReg; Platt-calibrates each; SLSQP-optimises blend weights.
   - Result: Log Loss 0.35113 → **0.34968** (−0.00145), ROC-AUC 0.685 → **0.693** (+0.008).
   - Optimal weights: LightGBM 47% + LogisticReg 40% + CatBoost 13% + XGBoost ~0%.

5. ~~**Step 5: Trust Card & Explainability Audit**~~ ✅ Complete
   - Implemented `explainability.py` for global and local SHAP explanations and 10% uncertainty referral simulation.
   - Compiled full 12-section [trust_card.md](file:///c:/Users/hansajak/Desktop/code/ml-nexus-2026/trust_card.md) covering model summary, calibration, fairness parity, failure modes, and clinical deployment recommendations.

6. ~~**Step 6: Bayesian Optuna Tuning & First Kaggle Submission**~~ ✅ Complete
   - Bayesian TPE tuning of LightGBM, CatBoost, and ElasticNet Logistic Regression via `tune_optuna.py`.
   - Result: OOF Log Loss **`0.34915`**, ROC-AUC **`0.69443`**.
   - Kaggle Public Leaderboard Submission (`submission_v4.csv`): **`0.33574`**.
   - Competition Gap: Only **`0.00036`** behind rival teams (`0.33538`) and **`0.00056`** behind OC benchmark (`0.33518`).

7. ~~**Step 7: v5 Champion Pipeline**~~ ✅ Complete
   - `v5.py` adds 9 new features (3 chronic×lab interactions + 6 fold-aware contextual aggregations), 3-seed averaging (45 sub-models), SLSQP blending, and post-ensemble Platt calibration.
   - Result: OOF Log Loss 0.34915 → **0.34879** (−0.00036), ROC-AUC **0.69473**.
   - Cumulative improvement vs baseline: **−0.00650** Log Loss.

8. ~~**Step 8: v6 Final Push (Information-Extraction)**~~ ⚠️ Overfit
   - `v6.py` added target encoding, 8-fold CV, feature pruning, ridge stacking.
   - Result: OOF Log Loss **0.34763** (−0.00116 vs v5) BUT Kaggle LB **0.33609** (+0.00035 worse than v4).
   - **Diagnosis:** Target encoding + aggressive pruning + unconstrained ridge = overfit to train distribution.

9. **Step 9: v7 Conservative Recovery (v4 Base + Safe Enhancements Only)** ← **Up Next**
   - **Base:** v4 pipeline (proven LB 0.33574) with full 54-feature set (NO pruning, NO target encoding).
   - **Safe additions only:**
     - ✅ Seed averaging (3 seeds) — pure variance reduction, can’t overfit.
     - ✅ Contextual aggregations (6 features) — simple ratio/diff, NOT target-dependent.
     - ✅ SLSQP blending (NOT ridge) — sum-to-1 constraint prevents probability rescaling.
     - ⚠️ Test both 5-fold and 8-fold CV, compare LB impact.
   - **Validation gate:** Only submit if improvement passes repeated CV significance test.
