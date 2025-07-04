#!/usr/bin/env python3
"""
Compare XGBoost performance with numeric vs categorical features for S5E7 dataset.

PURPOSE: Test if treating low-cardinality numeric columns as categorical improves performance
HYPOTHESIS: XGBoost might perform better with explicit categorical encoding for some features
EXPECTED: Slight improvement when treating columns with <16 unique values as categorical
RESULT: Minimal difference (~0.25% improvement with categorical), not significant after tuning
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import time


def prepare_data(df, target_col, categorical_cols=None):
    """Prepare data for XGBoost with optional categorical encoding."""
    # Separate features and target
    X = df.drop(columns=[target_col, 'id'])  # Remove target and ID
    y = df[target_col]
    
    # Encode target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Encode object columns (Yes/No columns) to numeric
    X_encoded = X.copy()
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            # Encode Yes/No to 1/0
            if set(X_encoded[col].dropna().unique()) <= {'Yes', 'No'}:
                X_encoded[col] = X_encoded[col].map({'Yes': 1, 'No': 0})
            else:
                # General label encoding for other object columns
                le = LabelEncoder()
                X_encoded[col] = X_encoded[col].fillna('missing')
                X_encoded[col] = le.fit_transform(X_encoded[col])
    
    # Handle categorical columns if specified
    if categorical_cols:
        for col in categorical_cols:
            if col in X_encoded.columns:
                # Convert to category dtype for XGBoost
                X_encoded[col] = X_encoded[col].astype('category')
        return X_encoded, y
    
    return X_encoded, y


def evaluate_xgboost(X, y, use_categorical=False, categorical_cols=None):
    """Evaluate XGBoost with cross-validation."""
    
    # Base parameters
    params = {
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    if use_categorical and categorical_cols:
        # Enable categorical support
        params['enable_categorical'] = True
        params['tree_method'] = 'hist'
        
    # Create model
    model = xgb.XGBClassifier(**params)
    
    # Run cross-validation
    print(f"\nRunning 5-fold CV (categorical={use_categorical})...")
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    end_time = time.time()
    
    print(f"CV Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"Time: {end_time - start_time:.2f}s")
    
    return scores.mean(), scores.std()


def main():
    # Load dataset
    print("Loading S5E7 dataset...")
    df = pd.read_csv("../../train.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Define categorical columns (from our analysis)
    categorical_columns = [
        'Time_spent_Alone',
        'Social_event_attendance', 
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency'
    ]
    
    print(f"\nColumns to treat as categorical: {categorical_columns}")
    
    # Check unique values
    print("\nUnique values per column:")
    for col in categorical_columns:
        if col in df.columns:
            n_unique = df[col].nunique()
            print(f"  {col}: {n_unique} unique values")
    
    # Test 1: All numeric (default)
    print("\n" + "="*60)
    print("TEST 1: All features as numeric (default)")
    print("="*60)
    X_numeric, y = prepare_data(df, 'Personality')
    mean_numeric, std_numeric = evaluate_xgboost(X_numeric, y, use_categorical=False)
    
    # Test 2: Low-cardinality columns as categorical
    print("\n" + "="*60)
    print("TEST 2: Low-cardinality features as categorical")
    print("="*60)
    X_categorical, y = prepare_data(df, 'Personality', categorical_columns)
    mean_categorical, std_categorical = evaluate_xgboost(
        X_categorical, y, use_categorical=True, categorical_cols=categorical_columns
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Numeric features:     {mean_numeric:.4f} (+/- {std_numeric*2:.4f})")
    print(f"Categorical features: {mean_categorical:.4f} (+/- {std_categorical*2:.4f})")
    
    improvement = (mean_categorical - mean_numeric) / mean_numeric * 100
    print(f"\nDifference: {improvement:+.2f}%")
    
    if improvement > 0:
        print("✅ Categorical encoding improved performance!")
    elif improvement < 0:
        print("❌ Numeric encoding performed better")
    else:
        print("➖ No significant difference")


if __name__ == "__main__":
    main()
