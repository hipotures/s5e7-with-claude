#!/usr/bin/env python3
"""
PURPOSE: Test models without ambiguous sample weights to reduce overfitting
HYPOTHESIS: Removing high weights on 600 ambiguous samples will improve generalization
EXPECTED: Lower CV score but better leaderboard performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

print("="*80)
print("TESTING MODELS WITHOUT AMBIGUOUS SAMPLE WEIGHTS")
print("="*80)

# Configuration
DATA_DIR = Path(".")
OUTPUT_DIR = Path("output")
SCORES_DIR = Path("../scores")
TARGET_COLUMN = "Personality"
ID_COLUMN = "id"
CV_FOLDS = 5
CV_RANDOM_STATE = 42

# Best performing corrected datasets
DATASETS_TO_TEST = [
    "train_corrected_07.csv",  # Best overall
    "train_corrected_04.csv",  # Psychological contradictions
    "train_corrected_01.csv",  # Extreme introverts
]

def load_and_preprocess_data(dataset_name=None):
    """Load and preprocess data"""
    if dataset_name:
        train_path = OUTPUT_DIR / dataset_name
        train_df = pd.read_csv(train_path)
    else:
        train_df = pd.read_csv("../../train.csv")
    
    test_df = pd.read_csv("../../test.csv")
    
    # Features
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency']
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    
    # Create null indicators BEFORE filling
    null_cols = numerical_cols
    for col in null_cols:
        train_df[f'{col}_is_null'] = train_df[col].isnull().astype(float)
        test_df[f'{col}_is_null'] = test_df[col].isnull().astype(float)
    
    train_df['null_count'] = train_df[null_cols].isnull().sum(axis=1)
    test_df['null_count'] = test_df[null_cols].isnull().sum(axis=1)
    train_df['has_nulls'] = (train_df['null_count'] > 0).astype(float)
    test_df['has_nulls'] = (test_df['null_count'] > 0).astype(float)
    
    # Weighted null score
    null_weights = {
        'Time_spent_Alone': 2.5,
        'Social_event_attendance': 1.8,
        'Going_outside': 1.5,
        'Friends_circle_size': 1.2,
        'Post_frequency': 1.0
    }
    
    train_df['weighted_null_score'] = sum(
        train_df[f'{col}_is_null'] * weight 
        for col, weight in null_weights.items()
    )
    test_df['weighted_null_score'] = sum(
        test_df[f'{col}_is_null'] * weight 
        for col, weight in null_weights.items()
    )
    
    # Class-aware imputation for numerical features
    for col in numerical_cols:
        if col in train_df.columns:
            intro_mean = train_df[train_df[TARGET_COLUMN] == 'Introvert'][col].mean()
            extro_mean = train_df[train_df[TARGET_COLUMN] == 'Extrovert'][col].mean()
            
            # For test set, use overall mean
            test_mean = train_df[col].mean()
            test_df[col] = test_df[col].fillna(test_mean)
            
            # For train, use class means
            intro_mask = train_df[TARGET_COLUMN] == 'Introvert'
            extro_mask = train_df[TARGET_COLUMN] == 'Extrovert'
            
            train_df.loc[intro_mask, col] = train_df.loc[intro_mask, col].fillna(intro_mean)
            train_df.loc[extro_mask, col] = train_df.loc[extro_mask, col].fillna(extro_mean)
    
    # Convert categorical
    for col in categorical_cols:
        if col in train_df.columns:
            mapping = {'Yes': 1, 'No': 0}
            train_df[col] = train_df[col].map(mapping)
            test_df[col] = test_df[col].map(mapping)
            
            mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 0.5
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)
    
    # Simple interaction features
    train_df['social_interaction'] = (
        train_df['Social_event_attendance'].fillna(0) * 
        train_df['Friends_circle_size'].fillna(0)
    )
    test_df['social_interaction'] = (
        test_df['Social_event_attendance'].fillna(0) * 
        test_df['Friends_circle_size'].fillna(0)
    )
    
    train_df['introvert_score'] = (
        train_df['Time_spent_Alone'].fillna(5) + 
        (10 - train_df['Social_event_attendance'].fillna(5)) +
        (15 - train_df['Friends_circle_size'].fillna(7.5))
    ) / 3
    test_df['introvert_score'] = (
        test_df['Time_spent_Alone'].fillna(5) + 
        (10 - test_df['Social_event_attendance'].fillna(5)) +
        (15 - test_df['Friends_circle_size'].fillna(7.5))
    ) / 3
    
    # Target
    train_df['target'] = (train_df[TARGET_COLUMN] == 'Extrovert').astype(int)
    
    # Feature columns
    feature_cols = (numerical_cols + categorical_cols + 
                   [f'{col}_is_null' for col in null_cols] +
                   ['null_count', 'has_nulls', 'weighted_null_score',
                    'social_interaction', 'introvert_score'])
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    
    return X_train, y_train, X_test, train_df, test_df

def train_and_evaluate(model_type, dataset_name=None):
    """Train model without ambiguous weights and evaluate"""
    print(f"\nTraining {model_type} on {dataset_name or 'original train.csv'}...")
    
    # Load data
    X_train, y_train, X_test, train_df, test_df = load_and_preprocess_data(dataset_name)
    
    # Model parameters (simplified, no ambig_weight)
    if model_type == 'xgb':
        model_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
            'gamma': 0.0,
            'min_child_weight': 1,
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'use_label_encoder': False,
            'random_state': CV_RANDOM_STATE,
            'n_jobs': 1
        }
        model_class = xgb.XGBClassifier
        
    elif model_type == 'gbm':
        model_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': -1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
            'min_child_weight': 1,
            'min_split_gain': 0.0,
            'device': 'gpu',
            'gpu_device_id': 1,
            'objective': 'binary',
            'metric': 'binary_error',
            'verbosity': -1,
            'random_state': CV_RANDOM_STATE,
            'n_jobs': 1
        }
        model_class = lgb.LGBMClassifier
        
    elif model_type == 'cat':
        model_params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'border_count': 128,
            'random_strength': 1.0,
            'bagging_temperature': 1.0,
            'thread_count': 16,
            'random_state': CV_RANDOM_STATE,
            'verbose': False
        }
        model_class = cb.CatBoostClassifier
    
    # Cross-validation WITHOUT sample weights
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train WITHOUT sample weights
        model = model_class(**model_params)
        
        if model_type == 'cat':
            model.fit(X_fold_train, y_fold_train, 
                     eval_set=(X_fold_val, y_fold_val), 
                     early_stopping_rounds=50)
        else:
            model.fit(X_fold_train, y_fold_train)
        
        # Predict
        pred_proba = model.predict_proba(X_fold_val)[:, 1]
        pred = (pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_fold_val, pred)
        cv_scores.append(accuracy)
        print(f"  Fold {fold}: {accuracy:.6f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    print(f"  CV Score: {mean_score:.6f} (+/- {std_score:.6f})")
    
    # Train final model on full data
    final_model = model_class(**model_params)
    final_model.fit(X_train, y_train)
    
    # Make predictions
    test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_pred_proba > 0.5).astype(int)
    
    # Create submission
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_short = dataset_name.replace('train_corrected_', 'tc').replace('.csv', '') if dataset_name else 'orig'
    
    submission = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        TARGET_COLUMN: test_pred
    })
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].map({1: 'Extrovert', 0: 'Introvert'})
    
    filename = f"subm-{mean_score:.6f}-{timestamp}-{model_type}-{dataset_short}-no_ambig_weight.csv"
    submission.to_csv(SCORES_DIR / filename, index=False)
    print(f"  Saved: {filename}")
    
    return mean_score, model

# Main execution
print("\nTesting models WITHOUT ambiguous sample weights...")
print("This should reduce CV scores but potentially improve LB performance")

results = []

# Test on original dataset first
for model_type in ['xgb', 'gbm', 'cat']:
    score, _ = train_and_evaluate(model_type, None)
    results.append(('original', model_type, score))

# Test on corrected datasets
for dataset in DATASETS_TO_TEST:
    for model_type in ['xgb', 'gbm', 'cat']:
        score, _ = train_and_evaluate(model_type, dataset)
        results.append((dataset, model_type, score))

# Summary
print("\n" + "="*80)
print("RESULTS SUMMARY (WITHOUT AMBIGUOUS WEIGHTS)")
print("="*80)

print("\nDataset                  Model    CV Score    Expected Impact")
print("-"*60)
for dataset, model, score in sorted(results, key=lambda x: x[2], reverse=True):
    dataset_short = dataset.replace('train_corrected_', 'tc').replace('.csv', '') if dataset != 'original' else 'original'
    print(f"{dataset_short:<20} {model:<8} {score:.6f}    Lower CV, better LB?")

print("\nKey observations:")
print("1. CV scores should be lower than with ambig_weight")
print("2. But generalization to test set should improve")
print("3. The gap between CV and LB should be smaller")

print("\nAnalysis complete!")