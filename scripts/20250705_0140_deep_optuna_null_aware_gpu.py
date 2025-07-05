#!/usr/bin/env python3
"""
PURPOSE: Deep Optuna optimization of null-aware models on GPU for maximum performance
HYPOTHESIS: With proper optimization, we can push beyond 98% accuracy
EXPECTED: Find optimal hyperparameters for all models and ensemble weights
RESULT: [To be filled after execution]

NOTE: Designed for GPU server with extended runtime (2-4 hours)
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optimization parameters
N_TRIALS_XGB = 200
N_TRIALS_LGB = 100
N_TRIALS_CAT = 100
N_TRIALS_ENS = 50

print(f"Deep optimization started at: {datetime.now()}")

# Load engineered features
print("Loading null-aware engineered features...")
train_df = pd.read_csv('output/20250705_0126_train_engineered.csv')
test_df = pd.read_csv('output/20250705_0126_test_engineered.csv')

feature_cols = [col for col in train_df.columns if col not in ['id', 'Personality']]
X_train = train_df[feature_cols]
y_train = train_df['Personality']
X_test = test_df[feature_cols]

print(f"Features: {len(feature_cols)}")
print(f"Training samples: {len(X_train)}")

# Load always-wrong samples
try:
    always_wrong_df = pd.read_csv('output/20250704_2318_always_wrong_samples.csv')
    always_wrong_ids = set(always_wrong_df['id'].values)
    always_wrong_indices = train_df[train_df['id'].isin(always_wrong_ids)].index.tolist()
    print(f"Loaded {len(always_wrong_indices)} always-wrong samples")
except:
    always_wrong_indices = []

print("\n=== DEEP OPTUNA OPTIMIZATION ===")

# Global CV settings
N_SPLITS = 3  # Balanced between speed and accuracy
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Create sample weights
sample_weights = np.ones(len(X_train))
for idx in always_wrong_indices:
    sample_weights[idx] = 10.0  # Increased weight

def objective_xgboost(trial):
    """Deep optimization for XGBoost with GPU"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        w_tr = sample_weights[train_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        
        y_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    
    return np.mean(scores)

def objective_lightgbm(trial):
    """Deep optimization for LightGBM with GPU"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 1,
        'random_state': 42,
        'verbose': -1
    }
    
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        w_tr = sample_weights[train_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        y_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    
    return np.mean(scores)

def objective_catboost(trial):
    """Deep optimization for CatBoost with GPU"""
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'task_type': 'GPU',
        'devices': '0',
        'random_seed': 42,
        'verbose': False
    }
    
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        w_tr = sample_weights[train_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        y_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    
    return np.mean(scores)

def objective_ensemble(trial, xgb_params, lgb_params, cat_params):
    """Optimize ensemble weights and threshold"""
    # Ensemble weights
    w_xgb = trial.suggest_float('w_xgb', 0, 1)
    w_lgb = trial.suggest_float('w_lgb', 0, 1)
    w_cat = trial.suggest_float('w_cat', 0, 1)
    
    # Normalize weights
    total = w_xgb + w_lgb + w_cat
    if total == 0:
        return 0
    w_xgb, w_lgb, w_cat = w_xgb/total, w_lgb/total, w_cat/total
    
    # Decision threshold
    threshold = trial.suggest_float('threshold', 0.25, 0.5)
    
    # Sample weight multiplier for difficult cases
    difficult_weight = trial.suggest_float('difficult_weight', 5, 20)
    
    # Create modified sample weights
    mod_weights = sample_weights.copy()
    for idx in always_wrong_indices:
        mod_weights[idx] = difficult_weight
    
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        w_tr = mod_weights[train_idx]
        
        # Train models
        xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, use_label_encoder=False)
        lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42, verbose=-1)
        cat_model = CatBoostClassifier(**cat_params, random_seed=42, verbose=False)
        
        xgb_model.fit(X_tr, y_tr, sample_weight=w_tr)
        lgb_model.fit(X_tr, y_tr, sample_weight=w_tr)
        cat_model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        # Get probabilities
        xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
        lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
        cat_probs = cat_model.predict_proba(X_val)[:, 1]
        
        # Weighted ensemble
        ensemble_probs = w_xgb * xgb_probs + w_lgb * lgb_probs + w_cat * cat_probs
        
        # Apply threshold
        y_pred = (ensemble_probs >= threshold).astype(int)
        
        scores.append(accuracy_score(y_val, y_pred))
    
    return np.mean(scores)

# Run optimizations
results = {}
start_time = time.time()

# 1. Optimize XGBoost
print("\n1. Optimizing XGBoost (GPU)...")
study_xgb = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='xgboost_null_aware'
)
study_xgb.optimize(objective_xgboost, n_trials=N_TRIALS_XGB, show_progress_bar=True)
results['xgboost'] = {
    'best_score': study_xgb.best_value,
    'best_params': study_xgb.best_params,
    'time': time.time() - start_time
}
print(f"XGBoost best: {study_xgb.best_value:.6f}")

# 2. Optimize LightGBM
print("\n2. Optimizing LightGBM (GPU)...")
start_lgb = time.time()
study_lgb = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='lightgbm_null_aware'
)
study_lgb.optimize(objective_lightgbm, n_trials=N_TRIALS_LGB, show_progress_bar=True)
results['lightgbm'] = {
    'best_score': study_lgb.best_value,
    'best_params': study_lgb.best_params,
    'time': time.time() - start_lgb
}
print(f"LightGBM best: {study_lgb.best_value:.6f}")

# 3. Optimize CatBoost
print("\n3. Optimizing CatBoost (GPU)...")
start_cat = time.time()
study_cat = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='catboost_null_aware'
)
study_cat.optimize(objective_catboost, n_trials=N_TRIALS_CAT, show_progress_bar=True)
results['catboost'] = {
    'best_score': study_cat.best_value,
    'best_params': study_cat.best_params,
    'time': time.time() - start_cat
}
print(f"CatBoost best: {study_cat.best_value:.6f}")

# 4. Optimize ensemble
print("\n4. Optimizing ensemble weights and threshold...")
start_ens = time.time()
study_ensemble = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='ensemble_null_aware'
)

# Use best params from individual optimizations
ensemble_objective = lambda trial: objective_ensemble(
    trial,
    study_xgb.best_params,
    study_lgb.best_params,
    study_cat.best_params
)

study_ensemble.optimize(ensemble_objective, n_trials=N_TRIALS_ENS, show_progress_bar=True)
results['ensemble'] = {
    'best_score': study_ensemble.best_value,
    'best_params': study_ensemble.best_params,
    'time': time.time() - start_ens
}
print(f"Ensemble best: {study_ensemble.best_value:.6f}")

# Save results
results['total_time'] = time.time() - start_time
results['total_trials'] = N_TRIALS_XGB + N_TRIALS_LGB + N_TRIALS_CAT + N_TRIALS_ENS

with open('output/20250705_0140_deep_optuna_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Train final optimized models
print("\n=== TRAINING FINAL OPTIMIZED MODELS ===")

# Extract best parameters
xgb_best = study_xgb.best_params
lgb_best = study_lgb.best_params
cat_best = study_cat.best_params
ens_best = study_ensemble.best_params

# Create optimized sample weights
final_weights = np.ones(len(X_train))
for idx in always_wrong_indices:
    final_weights[idx] = ens_best.get('difficult_weight', 10)

# Train on full data
print("Training optimized XGBoost...")
xgb_final = xgb.XGBClassifier(**xgb_best, random_state=42, use_label_encoder=False, tree_method='gpu_hist')
xgb_final.fit(X_train, y_train, sample_weight=final_weights)

print("Training optimized LightGBM...")
lgb_final = lgb.LGBMClassifier(**lgb_best, random_state=42, verbose=-1, device='gpu')
lgb_final.fit(X_train, y_train, sample_weight=final_weights)

print("Training optimized CatBoost...")
cat_final = CatBoostClassifier(**cat_best, random_seed=42, verbose=False, task_type='GPU')
cat_final.fit(X_train, y_train, sample_weight=final_weights)

# Generate predictions
print("\nGenerating optimized predictions...")
xgb_probs = xgb_final.predict_proba(X_test)[:, 1]
lgb_probs = lgb_final.predict_proba(X_test)[:, 1]
cat_probs = cat_final.predict_proba(X_test)[:, 1]

# Apply optimized ensemble
w_xgb = ens_best['w_xgb'] / (ens_best['w_xgb'] + ens_best['w_lgb'] + ens_best['w_cat'])
w_lgb = ens_best['w_lgb'] / (ens_best['w_xgb'] + ens_best['w_lgb'] + ens_best['w_cat'])
w_cat = ens_best['w_cat'] / (ens_best['w_xgb'] + ens_best['w_lgb'] + ens_best['w_cat'])

ensemble_probs = w_xgb * xgb_probs + w_lgb * lgb_probs + w_cat * cat_probs

# Apply optimized threshold
optimal_threshold = ens_best['threshold']
predictions = (ensemble_probs >= optimal_threshold).astype(int)

# Apply null-based rules
rules_applied = 0
drained_null_mask = test_df['Drained_after_socializing_is_null'] == 1
for i in np.where(drained_null_mask)[0]:
    if ensemble_probs[i] < 0.45:
        predictions[i] = 0
        rules_applied += 1

no_nulls_mask = test_df['has_no_nulls'] == 1
for i in np.where(no_nulls_mask)[0]:
    if ensemble_probs[i] > 0.65:
        predictions[i] = 1
        rules_applied += 1

print(f"Applied rules to {rules_applied} samples")

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in predictions]
})

# Save
import os
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)
submission.to_csv(f'subm/DATE_{date_str}/20250705_0140_deep_optuna_optimized.csv', index=False)

print(f"\n=== OPTIMIZATION SUMMARY ===")
print(f"Total optimization time: {results['total_time']/3600:.2f} hours")
print(f"Total trials: {results['total_trials']}")
print(f"\nBest scores:")
print(f"  XGBoost: {results['xgboost']['best_score']:.6f}")
print(f"  LightGBM: {results['lightgbm']['best_score']:.6f}")
print(f"  CatBoost: {results['catboost']['best_score']:.6f}")
print(f"  Ensemble: {results['ensemble']['best_score']:.6f}")
print(f"\nOptimized parameters:")
print(f"  Threshold: {optimal_threshold:.3f}")
print(f"  Weights: XGB={w_xgb:.3f}, LGB={w_lgb:.3f}, CAT={w_cat:.3f}")
print(f"  Difficult sample weight: {ens_best.get('difficult_weight', 10):.1f}x")

print(f"\nOptimization completed at: {datetime.now()}")

# RESULT: [To be filled after execution]
