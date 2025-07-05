#!/usr/bin/env python3
"""
PURPOSE: Create baseline model without any null features
HYPOTHESIS: Null features cause overfitting - test pure baseline
EXPECTED: Lower CV but better generalization
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
from datetime import datetime

print(f"Baseline model started at: {datetime.now()}")

# Load data
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Prepare features - ONLY original features
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                'Post_frequency']

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()
y_train = (train_df['Personality'] == 'Extrovert').astype(int)

# Convert categorical
for col in ['Stage_fear', 'Drained_after_socializing']:
    X_train[col] = X_train[col].map({'Yes': 1, 'No': 0})
    X_test[col] = X_test[col].map({'Yes': 1, 'No': 0})

print(f"Features: {len(feature_cols)}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Simple imputation - use mode for binary, median for numeric
print("\n=== IMPUTATION ===")
# For binary features (0/1), use mode
binary_imputer = SimpleImputer(strategy='most_frequent')
binary_cols = ['Stage_fear', 'Drained_after_socializing']

# For numeric features, use median
numeric_imputer = SimpleImputer(strategy='median')
numeric_cols = [col for col in feature_cols if col not in binary_cols]

# Apply imputation
X_train[binary_cols] = binary_imputer.fit_transform(X_train[binary_cols])
X_test[binary_cols] = binary_imputer.transform(X_test[binary_cols])

X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

print("Imputation complete")

# Very simple XGBoost
print("\n=== TRAINING SIMPLE XGBOOST ===")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,  # Very shallow
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    cv_scores.append(score)
    print(f"Fold {fold+1}: {score:.6f}")

print(f"\nMean CV: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")

# Train final model
model_final = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model_final.fit(X_train, y_train)

# Feature importance
feature_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feature_imp)

# Predictions
probs = model_final.predict_proba(X_test)[:, 1]
predictions = (probs >= 0.5).astype(int)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in predictions]
})

# Save
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)
submission_path = f'subm/DATE_{date_str}/20250705_0300_baseline_no_nulls.csv'
submission.to_csv(submission_path, index=False)

print(f"\n=== SUMMARY ===")
print(f"CV Score: {np.mean(cv_scores):.6f}")
print(f"Predictions: {predictions.sum()}/{len(predictions)} Extroverts ({predictions.mean()*100:.1f}%)")
print(f"Submission saved to: {submission_path}")

# RESULT: [To be filled after execution]