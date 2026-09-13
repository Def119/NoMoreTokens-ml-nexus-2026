# ML & AI Nexus 2026: probability-model refinement

Predict unplanned 30-day readmission on the supplied synthetic data. The competition metric is binary log loss (lower is better). This repository is a research workflow, not a clinical application.

## Environment

Use Python 3.12 and install `requirements-lock.txt` for the exact experiment environment, or `requirements-research.txt` for the declared dependencies. The local Windows setup has a project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -u -m nexus.train --output runs/nested_v7 --resume
.\.venv\Scripts\python.exe -u -m nexus.report --output runs/nested_v7
```

Run from the repository root. For a new reproduction after the competition, use a fresh output directory and `--deadline ""`. The default training cutoff is September 13, 2026 at 18:45 +05:30. `--config` accepts the `candidates` list in a run's `resolved_config.json`; `--threads` caps threads per model. A smoke test is available with `--smoke --outer-folds 2 --inner-folds 2 --output runs/smoke_new`.

To view progress without interrupting training:

```powershell
powershell -NoExit -File scripts/watch_training.ps1 -RunDirectory runs/nested_v7
```

Closing the viewer does not stop the independent training process. Progress estimates initially have little timing evidence and can fluctuate across model families. `training.log` is persistent; `status.json` is a best-effort live summary.

## Validation and interpretation

Each of five stratified outer folds uses three inner folds to choose configurations. All learned preprocessing and stopping-iteration selection are restricted to outer training data. Base model and combination recipes are evaluated on outer held-out rows. Candidate families are logistic regression, cubic-spline logistic regression, LightGBM, CatBoost and Explainable Boosting Machines. Equal, constrained convex and regularized logistic combinations are compared, with and without sigmoid calibration.

The chosen recipe has the lowest completed outer OOF log loss. Selection among these outer scores has residual optimism; prior development also used this dataset. Bootstrap intervals condition on saved predictions and do not account for every source of training/tuning variance. Compare the newly evaluated reference on identical folds rather than treating its score as interchangeable with historical saved OOF scores.

CatBoost CPU and GPU are benchmarked before a campaign. GPU results are not bitwise deterministic. No external data or pretrained weights are used. Patient identifiers and target labels cannot enter feature transformations.

## Artifacts

Each run contains the original-data/source hashes and software versions in `manifest.json`, full candidate configurations, row-level fold assignments, inner checkpoints, fitted outer models, per-fold selection/results, complete OOF predictions, candidate submissions and `comparison.csv`. Checkpoint/model directories are Git-ignored. Never combine caches across data, preprocessing, candidate, split or model changes. Resume requires matching data, source, candidates, splits and packages.

`submission_recommended.csv` is selected from completed candidates; `submission_historical_v4.csv` preserves the previous competition submission. Submissions have exactly the sample submission's ID order and columns. New Kaggle scores are unknown until manual upload.

The report produces calibration and learning-curve plots, held-out whole-recipe permutation importance, local feature-replacement sensitivity, missingness stress tests and subgroup confidence intervals. Local sensitivity is non-additive and non-causal. Entropy-based referral changes the retained population and does not establish safer care. A generated Trust Card follows the twelve headings in the supplied rulebook; the competition's linked template was not available locally.

`submission_notebook.ipynb` recomputes metrics from artifacts and provides an opt-in full training reproduction command. Use the project kernel when executing it.

## Historical audit findings

- v6 globally selects features before its later CV, contaminating evaluation.
- Its target-encoding map includes each training row's own label.
- Several historical scripts evaluate blends on the same OOF data used to optimize their weights.
- Final historical LightGBM fits omit `subsample_freq=1` used during tuning.
- Historical linear preprocessing uses `-999` numeric imputation, which distorts scaling.
- Existing Trust Card subgroup counts and several causal/overfitting interpretations are unsupported by supplied data.

Historical scripts and files are preserved. New evidence and the new Trust Card live in the selected run directory.

## Research references

- [Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Target-encoder cross-fitting](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html)
- [Spline transformations](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)
- [Explainable Boosting Classifier](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html)
- [CatBoost GPU behavior](https://catboost.ai/docs/en/features/training-on-gpu)
- [TabM: parameter-efficient tabular ensembles](https://github.com/yandex-research/tabm)

## Additional bounded campaigns

`configs/smooth_models.json` tests stronger shrinkage, simpler splines, shallower trees and coarser EBM bins. The fresh-partition campaign uses `--seed 20260914`; the original primary campaign uses `20260913`. Compare each against its own identically partitioned reference.

An optional CUDA TabM campaign is available after installing `requirements-neural.txt`. Run `python -u -m nexus.neural --base runs/nested_v7 --output runs/tabm_v7`. It trains from random initialization and reconstructs base **inner** OOF predictions for fitting its ensembles. Do not run it concurrently with GPU CatBoost. The existing base run remains untouched. `scripts/combine_neural_evidence.py` can combine completed compatible evidence; this is a comparison, not a fit to outer labels.

Use `scripts/review_campaigns.py` to collect completed scores, and `scripts/package_delivery.py --run <completed-run>` to package a verified report and reproduction sources. Run `scripts/execute_notebook.py <notebook>` and `scripts/validate_submission.py <submission>` before handoff. Uploaded Kaggle scores must be recorded separately from local validation.

## Competition logistics

The detailed supplied rules allocate 25 marks to leaderboard performance, 15 to the Trust Card, 40 to presentation and 20 to viva. They allow five submissions before midnight and five between midnight and September 14 at 08:00; a summary table conflicts with that detailed limit. The user requested local deliverables by September 13 at 20:00 Sri Lanka time. No automatic uploads are performed.
