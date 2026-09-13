import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

warnings.filterwarnings('ignore', category=UserWarning)

# 1. Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

features = [c for c in train.columns if c not in ['patient_id', 'readmitted_30d']]
target = train['readmitted_30d']

# Convert categorical columns to category dtype
cat_cols = ['sex', 'rurality', 'hospital_type', 'region', 'discharge_disposition', 'care_pathway']
for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# 2. Validation Framework
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train))
preds = np.zeros(len(test))

print("=" * 60)
print("Starting 5-Fold Stratified Cross-Validation (LightGBM Baseline)")
print("=" * 60)

# 3. Training Loop
fold_scores = []
for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target), 1):
    X_train, y_train = train.iloc[trn_idx][features], target.iloc[trn_idx]
    X_val, y_val = train.iloc[val_idx][features], target.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42 + fold,
        verbose=-1
    )
    
    # Fit with early stopping
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    val_preds = model.predict_proba(X_val)[:, 1]
    oof[val_idx] = val_preds
    preds += model.predict_proba(test[features])[:, 1] / skf.n_splits
    
    fold_ll = log_loss(y_val, val_preds)
    fold_brier = brier_score_loss(y_val, val_preds)
    fold_auc = roc_auc_score(y_val, val_preds)
    best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else 'N/A'
    
    fold_scores.append((fold_ll, fold_brier, fold_auc))
    print(f"Fold {fold} | Best Iter: {best_iter:>4} | Log Loss: {fold_ll:.5f} | Brier: {fold_brier:.5f} | ROC-AUC: {fold_auc:.5f}")

# 4. Overall OOF Evaluation
oof_ll = log_loss(target, oof)
oof_brier = brier_score_loss(target, oof)
oof_auc = roc_auc_score(target, oof)

print("=" * 60)
print(f"Overall OOF Log Loss : {oof_ll:.5f}")
print(f"Overall OOF Brier    : {oof_brier:.5f}")
print(f"Overall OOF ROC-AUC  : {oof_auc:.5f}")
print("=" * 60)

# 5. Export Submissions & OOF Predictions
sub = pd.DataFrame({
    'patient_id': test['patient_id'],
    'readmitted_30d': preds
})
sub.to_csv('submission.csv', index=False)
print(f"Saved submission to submission.csv (shape: {sub.shape})")

oof_df = pd.DataFrame({
    'patient_id': train['patient_id'],
    'actual': target,
    'pred_prob': oof
})
oof_df.to_csv('oof_baseline.csv', index=False)
print(f"Saved OOF predictions to oof_baseline.csv (shape: {oof_df.shape})")
