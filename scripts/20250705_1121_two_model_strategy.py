#!/usr/bin/env python3
"""
TWO-MODEL STRATEGY FOR PERSONALITY PREDICTION
=============================================

This script implements a two-model strategy based on the discovery that
records with no missing values have a different pattern than those with missing values.

- Model A: Expert on records with NO missing values.
- Model B: Expert on records WITH missing values.

Author: Claude & Gemini
Date: 2025-07-05 11:21
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 TWO-MODEL STRATEGY")
print("="*60)

# --- 1. Load and Preprocess Data ---
print("1. Loading and preprocessing data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Convert target variable
train_df['is_extrovert'] = (train_df['Personality'] == 'Extrovert').astype(int)

# Feature columns
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
                'Post_frequency']

# Create null_count feature for both train and test
train_df['null_count'] = train_df[feature_cols].isnull().sum(axis=1)
test_df['null_count'] = test_df[feature_cols].isnull().sum(axis=1)

# Preprocess categorical features (Yes/No -> 1/0)
for col in ['Stage_fear', 'Drained_after_socializing']:
    train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})
    test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})

# --- 2. Split Data into Subsets ---
print("2. Splitting data based on null patterns...")
train_no_nulls = train_df[train_df['null_count'] == 0].copy()
train_with_nulls = train_df[train_df['null_count'] > 0].copy()

test_no_nulls = test_df[test_df['null_count'] == 0].copy()
test_with_nulls = test_df[test_df['null_count'] > 0].copy()

print(f"  Train (no nulls): {len(train_no_nulls)} records")
print(f"  Train (with nulls): {len(train_with_nulls)} records")
print(f"  Test (no nulls): {len(test_no_nulls)} records")
print(f"  Test (with nulls): {len(test_with_nulls)} records")

# --- 3. Train Model A (No-Nulls Expert) ---
print("3. Training Model A (No-Nulls Expert)...")

# For no-null data, all features are available. No imputation needed.
X_train_A = train_no_nulls[feature_cols]
y_train_A = train_no_nulls['is_extrovert']

model_A = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model_A.fit(X_train_A, y_train_A)
print("  Model A trained.")

# --- 4. Train Model B (With-Nulls Expert) ---
print(" 4. Training Model B (With-Nulls Expert)...")

# For this model, null indicators are important features
X_train_B = train_with_nulls[feature_cols].copy()
y_train_B = train_with_nulls['is_extrovert']

# Create null indicator features for Model B
for col in feature_cols:
    if X_train_B[col].isnull().any():
        X_train_B[f'{col}_is_null'] = X_train_B[col].isnull().astype(int)

# Impute missing values (XGBoost can handle NaNs natively)
# No explicit imputation needed for XGBoost, it handles it internally.

model_B_features = list(X_train_B.columns) # Get all features including indicators
model_B = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model_B.fit(X_train_B, y_train_B)
print("  Model B trained.")

# --- 5. Make Predictions ---
print(" 5. Making predictions on test subsets...")

# Predict on no-nulls test set with Model A
X_test_A = test_no_nulls[feature_cols]
predictions_A = model_A.predict(X_test_A)
test_no_nulls['prediction'] = predictions_A
print(f"  Predicted {len(test_no_nulls)} records with Model A.")

# Predict on with-nulls test set with Model B
X_test_B = test_with_nulls[feature_cols].copy()
# Create the same null indicator features for the test set
for col in feature_cols:
    if X_test_B[col].isnull().any():
        X_test_B[f'{col}_is_null'] = X_test_B[col].isnull().astype(int)

# Ensure columns match between train and test for Model B
missing_cols = set(model_B_features) - set(X_test_B.columns)
for c in missing_cols:
    X_test_B[c] = 0
X_test_B = X_test_B[model_B_features] # Ensure order is the same

predictions_B = model_B.predict(X_test_B)
test_with_nulls['prediction'] = predictions_B
print(f"  Predicted {len(test_with_nulls)} records with Model B.")

# --- 6. Combine and Save Submission ---
print(" 6. Combining predictions and saving submission...")

# Combine predictions
final_predictions_df = pd.concat([
    test_no_nulls[['id', 'prediction']],
    test_with_nulls[['id', 'prediction']]
]).sort_values('id')

# Map back to original labels
final_predictions_df['Personality'] = final_predictions_df['prediction'].map({1: 'Extrovert', 0: 'Introvert'})

# Create submission file
submission_df = final_predictions_df[['id', 'Personality']]
submission_path = 'submission_two_model_strategy.csv'
submission_df.to_csv(submission_path, index=False)

print(f" ✓ Submission saved to {submission_path}")
print(" Distribution of final predictions:")
print(submission_df['Personality'].value_counts(normalize=True))

print(" " + "="*60)
print("SCRIPT COMPLETE")
print("="*60)
