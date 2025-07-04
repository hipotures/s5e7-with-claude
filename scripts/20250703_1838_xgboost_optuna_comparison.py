#!/usr/bin/env python3
"""
Compare XGBoost performance with numeric vs categorical features using Optuna hyperparameter optimization.

PURPOSE: Test if hyperparameter optimization changes the numeric vs categorical feature performance gap
HYPOTHESIS: With proper tuning, the performance difference between approaches might diminish
EXPECTED: Similar performance after optimization, suggesting encoding choice is less critical than tuning
RESULT: After optimization, categorical encoding still slightly better but difference reduced to ~0.1%
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import optuna
import time
import warnings
warnings.filterwarnings('ignore')

# Silence Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def objective_numeric(trial, X, y):
    """Optuna objective for numeric features."""
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
    return scores.mean()


def objective_categorical(trial, X, y):
    """Optuna objective for categorical features."""
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'enable_categorical': True,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
    return scores.mean()


def optimize_and_evaluate(X, y, use_categorical=False, n_trials=50):
    """Optimize hyperparameters using Optuna and evaluate."""
    
    print(f"\nOptimizing hyperparameters with Optuna ({n_trials} trials)...")
    start_time = time.time()
    
    # Create study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    # Select objective function
    if use_categorical:
        study.optimize(lambda trial: objective_categorical(trial, X, y), n_trials=n_trials)
    else:
        study.optimize(lambda trial: objective_numeric(trial, X, y), n_trials=n_trials)
    
    optimization_time = time.time() - start_time
    
    # Get best parameters
    best_params = study.best_params
    best_params.update({
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 0
    })
    
    if use_categorical:
        best_params['enable_categorical'] = True
        best_params['tree_method'] = 'hist'
    
    print(f"Best score during optimization: {study.best_value:.4f}")
    print(f"Optimization time: {optimization_time:.1f}s")
    
    # Final evaluation with best parameters
    print("\nFinal evaluation with best parameters...")
    model = xgb.XGBClassifier(**best_params)
    
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
    eval_time = time.time() - start_time
    
    print(f"CV Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"Evaluation time: {eval_time:.2f}s")
    
    # Show best parameters
    print("\nBest parameters:")
    for param, value in sorted(best_params.items()):
        if param not in ['random_state', 'n_jobs', 'objective', 'eval_metric', 'verbosity', 'enable_categorical', 'tree_method']:
            print(f"  {param}: {value}")
    
    return scores.mean(), scores.std(), study.best_value, best_params


def main():
    # Load dataset
    print("Loading S5E7 dataset...")
    df = pd.read_csv("../../train.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Define categorical columns
    categorical_columns = [
        'Time_spent_Alone',
        'Social_event_attendance', 
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency'
    ]
    
    print(f"\nColumns to treat as categorical: {categorical_columns}")
    
    # Test 1: All numeric with Optuna
    print("\n" + "="*60)
    print("TEST 1: All features as numeric (with Optuna)")
    print("="*60)
    X_numeric, y = prepare_data(df, 'Personality')
    mean_numeric, std_numeric, best_numeric, params_numeric = optimize_and_evaluate(
        X_numeric, y, use_categorical=False, n_trials=50
    )
    
    # Test 2: Low-cardinality columns as categorical with Optuna
    print("\n" + "="*60)
    print("TEST 2: Low-cardinality features as categorical (with Optuna)")
    print("="*60)
    X_categorical, y = prepare_data(df, 'Personality', categorical_columns)
    mean_categorical, std_categorical, best_categorical, params_categorical = optimize_and_evaluate(
        X_categorical, y, use_categorical=True, n_trials=50
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY (with hyperparameter optimization)")
    print("="*60)
    print(f"Numeric features:     {mean_numeric:.4f} (+/- {std_numeric*2:.4f})")
    print(f"Categorical features: {mean_categorical:.4f} (+/- {std_categorical*2:.4f})")
    
    improvement = (mean_categorical - mean_numeric) / mean_numeric * 100
    print(f"\nDifference: {improvement:+.2f}%")
    
    if improvement > 0.1:
        print("✅ Categorical encoding improved performance!")
    elif improvement < -0.1:
        print("❌ Numeric encoding performed better")
    else:
        print("➖ No significant difference")
    
    # Compare with default parameters results
    print("\n" + "="*60)
    print("COMPARISON: Default vs Optimized Parameters")
    print("="*60)
    print("Default parameters (from previous run):")
    print("  Numeric:     0.9617 (+/- 0.0073)")
    print("  Categorical: 0.9642 (+/- 0.0051)")
    print("  Difference:  +0.25%")
    print("\nOptimized parameters:")
    print(f"  Numeric:     {mean_numeric:.4f} (+/- {std_numeric*2:.4f})")
    print(f"  Categorical: {mean_categorical:.4f} (+/- {std_categorical*2:.4f})")
    print(f"  Difference:  {improvement:+.2f}%")


if __name__ == "__main__":
    main()