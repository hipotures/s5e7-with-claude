#!/usr/bin/env python3
"""
PURPOSE: Fix overfitting issue - 98% CV but only 91.5% on leaderboard
HYPOTHESIS: Class-aware imputation and complex models cause overfitting
EXPECTED: More robust model with better generalization
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import os
from datetime import datetime

print(f"Fixing overfitting started at: {datetime.now()}")

# Load raw data
print("Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Prepare features
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                'Post_frequency']

X_train_raw = train_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()
y_train = (train_df['Personality'] == 'Extrovert').astype(int)

# Convert categorical
for col in ['Stage_fear', 'Drained_after_socializing']:
    X_train_raw[col] = X_train_raw[col].map({'Yes': 1, 'No': 0})
    X_test_raw[col] = X_test_raw[col].map({'Yes': 1, 'No': 0})

print("\n=== CREATING SIMPLE NULL FEATURES ===")

def create_simple_null_features(df):
    """Create only the most important null features"""
    df_features = df.copy()
    
    # Only key null indicators
    df_features['has_drained_null'] = df['Drained_after_socializing'].isna().astype(int)
    df_features['has_stage_null'] = df['Stage_fear'].isna().astype(int)
    df_features['null_count'] = df.isna().sum(axis=1)
    df_features['has_no_nulls'] = (df_features['null_count'] == 0).astype(int)
    
    return df_features

# Create features
X_train = create_simple_null_features(X_train_raw)
X_test = create_simple_null_features(X_test_raw)

print(f"Features: {X_train.shape[1]}")

print("\n=== SIMPLE IMPUTATION (NO CLASS INFO) ===")

# Use median imputation - more robust than mean
imputer = SimpleImputer(strategy='median')
feature_cols_to_impute = feature_cols

# Fit on train and transform both
X_train[feature_cols_to_impute] = imputer.fit_transform(X_train[feature_cols_to_impute])
X_test[feature_cols_to_impute] = imputer.transform(X_test[feature_cols_to_impute])

print("Imputation complete")

print("\n=== CONSERVATIVE MODEL TRAINING ===")

# More conservative parameters to prevent overfitting
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,  # Reduced from 6-10
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,  # More regularization
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)

cat_model = CatBoostClassifier(
    iterations=300,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=5,  # More regularization
    random_seed=42,
    verbose=False
)

# Cross-validation
print("\n=== CROSS-VALIDATION ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {'xgb': [], 'lgb': [], 'cat': [], 'ensemble': []}

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"\nFold {fold + 1}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train without sample weights to avoid overfitting
    xgb_model.fit(X_tr, y_tr)
    lgb_model.fit(X_tr, y_tr)
    cat_model.fit(X_tr, y_tr)
    
    # Predictions
    xgb_pred = xgb_model.predict(X_val)
    lgb_pred = lgb_model.predict(X_val)
    cat_pred = cat_model.predict(X_val)
    
    # Simple average ensemble
    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
    cat_probs = cat_model.predict_proba(X_val)[:, 1]
    
    ensemble_probs = (xgb_probs + lgb_probs + cat_probs) / 3
    ensemble_pred = (ensemble_probs >= 0.5).astype(int)
    
    # Scores
    cv_scores['xgb'].append(accuracy_score(y_val, xgb_pred))
    cv_scores['lgb'].append(accuracy_score(y_val, lgb_pred))
    cv_scores['cat'].append(accuracy_score(y_val, cat_pred))
    cv_scores['ensemble'].append(accuracy_score(y_val, ensemble_pred))

print("\n=== CV RESULTS ===")
for model, scores in cv_scores.items():
    print(f"{model}: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")

# Train final models on full data
print("\n=== TRAINING FINAL MODELS ===")
xgb_final = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0,
    random_state=42, use_label_encoder=False, eval_metric='logloss'
)

lgb_final = lgb.LGBMClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0,
    random_state=42, verbose=-1
)

cat_final = CatBoostClassifier(
    iterations=300, depth=4, learning_rate=0.05,
    l2_leaf_reg=5, random_seed=42, verbose=False
)

# Train without sample weights
xgb_final.fit(X_train, y_train)
lgb_final.fit(X_train, y_train)
cat_final.fit(X_train, y_train)

print("\n=== GENERATING PREDICTIONS ===")

# Get probabilities
xgb_probs = xgb_final.predict_proba(X_test)[:, 1]
lgb_probs = lgb_final.predict_proba(X_test)[:, 1]
cat_probs = cat_final.predict_proba(X_test)[:, 1]

# Simple average
ensemble_probs = (xgb_probs + lgb_probs + cat_probs) / 3

# Standard threshold
predictions = (ensemble_probs >= 0.5).astype(int)

# Apply only the most reliable null rules
print("\nApplying conservative null rules...")
rules_applied = 0

# Only apply rule for missing Drained with very low probability
drained_null_mask = X_test['has_drained_null'] == 1
for i in np.where(drained_null_mask)[0]:
    if ensemble_probs[i] < 0.4:  # More conservative threshold
        predictions[i] = 0
        rules_applied += 1

print(f"Applied rules to {rules_applied} samples")

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in predictions]
})

# Save
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)
submission_path = f'subm/DATE_{date_str}/20250705_0250_fix_overfitting.csv'
submission.to_csv(submission_path, index=False)

print(f"\n=== SUMMARY ===")
print(f"Best CV: {max([np.mean(scores) for scores in cv_scores.values()]):.6f}")
print(f"Prediction distribution: {predictions.sum()}/{len(predictions)} Extroverts ({predictions.mean()*100:.1f}%)")
print(f"Submission saved to: {submission_path}")

print(f"\nCompleted at: {datetime.now()}")

# RESULT: [To be filled after execution]