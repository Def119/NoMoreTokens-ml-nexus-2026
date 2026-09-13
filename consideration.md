Focusing on both predictive accuracy as well as interpretability

## Kaggle metric
The live Kaggle competition should use Binary Log Loss.

Lower is better.

Why Log Loss? Because this challenge is about trustworthy probabilities, not only correct 0/1 classifications. Very confident wrong predictions are penalized heavily.

## Statistical evidence
Whenever you claim that one model is better than another, avoid relying on a single validation score. Useful evidence includes:

repeated cross
validation
bootstrap confidence intervals
paired bootstrap comparisons
calibration curves/intercept/slope
Brier score
subgroup confidence intervals
sensitivity analyses

## trusr card details

1. Model summary
Final model(s):
Key preprocessing:
Key hyperparameters:
Why this model was selected:

2. Validation design
Describe train/validation strategy.
Explain why it is appropriate.
Report uncertainty or variability across splits/resamples.

3. Performance
Report at minimum:

Log Loss
Brier score
ROC-AUC
Sensitivity / specificity at your chosen operating threshold
Do not report only the best single split.

4. Calibration
Include:

calibration plot or summary
calibration method, if used
evidence before vs after calibration

5. Robustness
What might change between development and deployment populations?
What sensitivity tests did you run?
Which variables or modeling choices appeared unstable?

6. Subgroup reliability
At minimum examine:

sex
rurality
age group
hospital type
State group sizes. Report uncertainty where possible.
Do not interpret differences automatically as discrimination or causation.

7. Uncertainty / human referral
How did you identify predictions that should be treated cautiously?
If you abstain/refer the 10% most uncertain cases, what happens to performance on the remaining 90%?

8. Explainability
List the most influential predictors.

Provide one local explanation for:

a high-risk prediction
a low-risk prediction
Clearly distinguish predictive association from causation.

9. Failure modes
Give at least three concrete ways your model may fail.

10. Deployment recommendation
Choose one:

Ready for limited prospective validation
Requires additional model development
Should not be deployed
Justify your choice in 100 words or fewer.

11. Reproducibility
Software / package versions:
Random seed(s):
Approximate training time:
AI-assistant use, if permitted:

12. One-sentence conclusion
Complete: We trust this model only when…