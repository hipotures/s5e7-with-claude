#!/usr/bin/env python3
"""
PURPOSE: Deep Optuna optimization with 1000+ trials to find perfect hyperparameters
HYPOTHESIS: Extensive search will find the exact parameter combination for 0.976518
EXPECTED: Discover optimal hyperparameters that break the 0.975708 barrier
RESULT: [To be filled after execution]

NOTE: This script is designed for GPU server with 200GB RAM and 2x4090
Expected runtime: 2-4 hours
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Set up GPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use both GPUs

# Load and prepare data
print("Loading data...")
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

# Feature engineering based on discoveries
def prepare_features_optimized(df, is_train=True):
    if is_train:
        X = df.drop(['id', 'Personality'], axis=1).copy()
        y = (df['Personality'] == 'Extrovert').astype(int)
    else:
        X = df.drop(['id'], axis=1).copy()
        y = None
    
    # Convert categorical
    for col in ['Stage_fear', 'Drained_after_socializing']:
        X[f'{col}_missing'] = X[col].isna().astype(int)
        X[col] = X[col].map({'Yes': 1, 'No': 0})
        X[col] = X[col].fillna(0)  # Missing = likely No (extrovert)
    
    # Key features from analysis
    X['is_extrovert_pattern'] = ((X['Time_spent_Alone'] <= 3) & 
                                  (X['Drained_after_socializing'] == 0)).astype(int)
    X['is_introvert_pattern'] = ((X['Time_spent_Alone'] >= 7) & 
                                  (X['Drained_after_socializing'] == 1)).astype(int)
    X['friends_uncertain'] = X['Friends_circle_size'].isin([4, 5]).astype(int)
    X['social_10'] = (X['Social_event_attendance'] == 10).astype(int)
    
    # Ratios and interactions
    X['social_to_alone_ratio'] = (X['Social_event_attendance'] + 1) / (X['Time_spent_Alone'] + 1)
    X['social_intensity'] = X['Social_event_attendance'] * (1 - X['Drained_after_socializing'])
    
    # Polynomial features for key variables
    for feat in ['Time_spent_Alone', 'Social_event_attendance']:
        X[f'{feat}_squared'] = X[feat] ** 2
        X[f'{feat}_log1p'] = np.log1p(X[feat])
    
    return X, y

X_train, y_train = prepare_features_optimized(train_df, is_train=True)
X_test, _ = prepare_features_optimized(test_df, is_train=False)

print(f"Features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")

# Define objective functions for each model type
def objective_xgboost(trial):
    """XGBoost objective with GPU support"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'tree_method': 'gpu_hist',  # GPU acceleration
        'gpu_id': 0,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # Use optimal threshold discovered
    threshold = trial.suggest_float('threshold', 0.3, 0.5)
    
    # Cross-validation with custom scoring
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        # Custom predictions with threshold
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= threshold).astype(int)
        
        # Apply special rules
        for i in range(len(val_preds)):
            # Force patterns
            if X_val.iloc[i]['is_extrovert_pattern'] == 1:
                val_preds[i] = 1
            elif X_val.iloc[i]['is_introvert_pattern'] == 1:
                val_preds[i] = 0
            # Missing values rule
            elif X_val.iloc[i]['Drained_after_socializing_missing'] == 1:
                val_preds[i] = 1
        
        accuracy = (val_preds == y_val).mean()
        scores.append(accuracy)
    
    return np.mean(scores)

def objective_lightgbm(trial):
    """LightGBM objective with GPU support"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 1,  # Use second GPU
        'random_state': 42,
        'verbose': -1
    }
    
    threshold = trial.suggest_float('threshold', 0.3, 0.5)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= threshold).astype(int)
        
        # Apply rules
        for i in range(len(val_preds)):
            if X_val.iloc[i]['is_extrovert_pattern'] == 1:
                val_preds[i] = 1
            elif X_val.iloc[i]['is_introvert_pattern'] == 1:
                val_preds[i] = 0
            elif X_val.iloc[i]['Drained_after_socializing_missing'] == 1:
                val_preds[i] = 1
        
        accuracy = (val_preds == y_val).mean()
        scores.append(accuracy)
    
    return np.mean(scores)

def objective_catboost(trial):
    """CatBoost objective with GPU support"""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'task_type': 'GPU',
        'devices': '0',
        'random_seed': 42,
        'verbose': False
    }
    
    threshold = trial.suggest_float('threshold', 0.3, 0.5)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= threshold).astype(int)
        
        # Apply rules
        for i in range(len(val_preds)):
            if X_val.iloc[i]['is_extrovert_pattern'] == 1:
                val_preds[i] = 1
            elif X_val.iloc[i]['is_introvert_pattern'] == 1:
                val_preds[i] = 0
            elif X_val.iloc[i]['Drained_after_socializing_missing'] == 1:
                val_preds[i] = 1
        
        accuracy = (val_preds == y_val).mean()
        scores.append(accuracy)
    
    return np.mean(scores)

# Run optimization for each model
print("\n=== STARTING DEEP OPTUNA OPTIMIZATION ===")
print("This will take 2-4 hours. Progress will be saved to scripts/output/")

results = {}

# XGBoost optimization
print("\n1. Optimizing XGBoost (GPU)...")
study_xgb = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='xgboost_gpu'
)

start_time = time.time()
study_xgb.optimize(objective_xgboost, n_trials=350, show_progress_bar=True)
xgb_time = time.time() - start_time

results['xgboost'] = {
    'best_score': study_xgb.best_value,
    'best_params': study_xgb.best_params,
    'n_trials': len(study_xgb.trials),
    'optimization_time': xgb_time
}

print(f"XGBoost best score: {study_xgb.best_value:.6f}")
print(f"Time taken: {xgb_time/60:.1f} minutes")

# Save intermediate results
with open('scripts/output/20250704_2313_optuna_intermediate.json', 'w') as f:
    json.dump(results, f, indent=2)

# LightGBM optimization
print("\n2. Optimizing LightGBM (GPU)...")
study_lgb = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='lightgbm_gpu'
)

start_time = time.time()
study_lgb.optimize(objective_lightgbm, n_trials=350, show_progress_bar=True)
lgb_time = time.time() - start_time

results['lightgbm'] = {
    'best_score': study_lgb.best_value,
    'best_params': study_lgb.best_params,
    'n_trials': len(study_lgb.trials),
    'optimization_time': lgb_time
}

print(f"LightGBM best score: {study_lgb.best_value:.6f}")
print(f"Time taken: {lgb_time/60:.1f} minutes")

# Save intermediate results
with open('scripts/output/20250704_2313_optuna_intermediate.json', 'w') as f:
    json.dump(results, f, indent=2)

# CatBoost optimization
print("\n3. Optimizing CatBoost (GPU)...")
study_cat = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='catboost_gpu'
)

start_time = time.time()
study_cat.optimize(objective_catboost, n_trials=300, show_progress_bar=True)
cat_time = time.time() - start_time

results['catboost'] = {
    'best_score': study_cat.best_value,
    'best_params': study_cat.best_params,
    'n_trials': len(study_cat.trials),
    'optimization_time': cat_time
}

print(f"CatBoost best score: {study_cat.best_value:.6f}")
print(f"Time taken: {cat_time/60:.1f} minutes")

# Save final results
results['total_optimization_time'] = xgb_time + lgb_time + cat_time
results['total_trials'] = results['xgboost']['n_trials'] + results['lightgbm']['n_trials'] + results['catboost']['n_trials']

with open('scripts/output/20250704_2313_optuna_final_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Train final ensemble with best parameters
print("\n=== TRAINING FINAL ENSEMBLE WITH BEST PARAMETERS ===")

# Extract best params (remove threshold)
xgb_params = {k: v for k, v in study_xgb.best_params.items() if k != 'threshold'}
lgb_params = {k: v for k, v in study_lgb.best_params.items() if k != 'threshold'}
cat_params = {k: v for k, v in study_cat.best_params.items() if k != 'threshold'}

# Train models
xgb_model = xgb.XGBClassifier(**xgb_params, tree_method='gpu_hist', gpu_id=0)
lgb_model = lgb.LGBMClassifier(**lgb_params, device='gpu', gpu_device_id=1)
cat_model = CatBoostClassifier(**cat_params, task_type='GPU', devices='0')

print("Training optimized models...")
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)

# Generate predictions
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
cat_probs = cat_model.predict_proba(X_test)[:, 1]

# Ensemble
ensemble_probs = (xgb_probs + lgb_probs + cat_probs) / 3

# Apply optimal thresholds
best_threshold = np.mean([
    study_xgb.best_params['threshold'],
    study_lgb.best_params['threshold'],
    study_cat.best_params['threshold']
])

predictions = (ensemble_probs >= best_threshold).astype(int)

# Apply discovered rules
for i in range(len(predictions)):
    if X_test.iloc[i]['is_extrovert_pattern'] == 1:
        predictions[i] = 1
    elif X_test.iloc[i]['is_introvert_pattern'] == 1:
        predictions[i] = 0
    elif X_test.iloc[i]['Drained_after_socializing_missing'] == 1:
        predictions[i] = 1

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in predictions]
})

# Save submission
from datetime import datetime
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)

submission_path = f'subm/DATE_{date_str}/20250704_2313_deep_optuna_optimized.csv'
submission.to_csv(submission_path, index=False)

# Summary
print("\n=== OPTIMIZATION COMPLETE ===")
print(f"Total trials: {results['total_trials']}")
print(f"Total time: {results['total_optimization_time']/3600:.1f} hours")
print(f"\nBest scores:")
print(f"XGBoost: {results['xgboost']['best_score']:.6f}")
print(f"LightGBM: {results['lightgbm']['best_score']:.6f}")
print(f"CatBoost: {results['catboost']['best_score']:.6f}")
print(f"\nOptimal threshold: {best_threshold:.3f}")
print(f"\nSubmission saved to: {submission_path}")

# RESULT: [To be filled after execution]