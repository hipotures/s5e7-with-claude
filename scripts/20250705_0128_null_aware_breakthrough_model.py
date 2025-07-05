#!/usr/bin/env python3
"""
PURPOSE: Build the final null-aware model to break 0.976518 barrier
HYPOTHESIS: Combining null discoveries with optimized ensemble will achieve breakthrough
EXPECTED: Generate submission that achieves 0.976518+ on Kaggle leaderboard
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import VotingClassifier
import optuna
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print(f"Breakthrough model training started at: {datetime.now()}")

# Load engineered features
print("Loading engineered features...")
train_df = pd.read_csv('output/20250705_0126_train_engineered.csv')
test_df = pd.read_csv('output/20250705_0126_test_engineered.csv')

# Prepare features
feature_cols = [col for col in train_df.columns if col not in ['id', 'Personality']]
X_train = train_df[feature_cols]
y_train = train_df['Personality']
X_test = test_df[feature_cols]

print(f"Features: {len(feature_cols)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Load always-wrong samples for special handling
try:
    always_wrong_df = pd.read_csv('output/20250704_2318_always_wrong_samples.csv')
    always_wrong_ids = set(always_wrong_df['id'].values)
    always_wrong_indices = train_df[train_df['id'].isin(always_wrong_ids)].index.tolist()
    print(f"\nLoaded {len(always_wrong_indices)} always-wrong samples for special handling")
except:
    always_wrong_indices = []

print("\n=== OPTIMIZING MODELS ===")

# Quick Optuna optimization for best parameters
def optimize_xgboost(trial, X, y, cv_folds=3):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, verbose=False)
        
        y_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    
    return np.mean(scores)

# Run quick optimization
print("\nOptimizing XGBoost...")
study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(lambda trial: optimize_xgboost(trial, X_train, y_train), n_trials=50, show_progress_bar=True)

best_xgb_params = study_xgb.best_params
print(f"Best XGBoost score: {study_xgb.best_value:.6f}")

print("\n=== TRAINING FINAL MODELS ===")

# 1. XGBoost with optimized parameters
print("\n1. Training optimized XGBoost...")
xgb_model = xgb.XGBClassifier(
    **best_xgb_params,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 2. LightGBM with strong parameters
print("2. Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)

# 3. CatBoost for categorical handling
print("3. Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=False
)

# Cross-validation with special focus on difficult samples
print("\n=== CROSS-VALIDATION WITH SAMPLE WEIGHTS ===")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {'xgb': [], 'lgb': [], 'cat': [], 'ensemble': []}

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"\nFold {fold + 1}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Create sample weights - higher weight for always-wrong samples
    sample_weights = np.ones(len(X_tr))
    for idx in always_wrong_indices:
        if idx in train_idx:
            pos = np.where(train_idx == idx)[0][0]
            sample_weights[pos] = 5.0  # 5x weight for difficult samples
    
    # Train models
    xgb_model.fit(X_tr, y_tr, sample_weight=sample_weights)
    lgb_model.fit(X_tr, y_tr, sample_weight=sample_weights)
    cat_model.fit(X_tr, y_tr, sample_weight=sample_weights)
    
    # Individual predictions
    xgb_pred = xgb_model.predict(X_val)
    lgb_pred = lgb_model.predict(X_val)
    cat_pred = cat_model.predict(X_val)
    
    # Ensemble predictions (majority vote)
    ensemble_pred = np.array([xgb_pred, lgb_pred, cat_pred]).mean(axis=0) >= 0.5
    
    # Scores
    cv_scores['xgb'].append(accuracy_score(y_val, xgb_pred))
    cv_scores['lgb'].append(accuracy_score(y_val, lgb_pred))
    cv_scores['cat'].append(accuracy_score(y_val, cat_pred))
    cv_scores['ensemble'].append(accuracy_score(y_val, ensemble_pred))
    
    print(f"  XGBoost: {cv_scores['xgb'][-1]:.6f}")
    print(f"  LightGBM: {cv_scores['lgb'][-1]:.6f}")
    print(f"  CatBoost: {cv_scores['cat'][-1]:.6f}")
    print(f"  Ensemble: {cv_scores['ensemble'][-1]:.6f}")

print("\n=== CV SUMMARY ===")
for model_name, scores in cv_scores.items():
    print(f"{model_name}: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")

# Train final models on full data with sample weights
print("\n=== TRAINING FINAL MODELS ON FULL DATA ===")

# Create sample weights for full training
full_sample_weights = np.ones(len(X_train))
for idx in always_wrong_indices:
    full_sample_weights[idx] = 5.0

# Train all models
xgb_final = xgb.XGBClassifier(**best_xgb_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
lgb_final = lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, num_leaves=63, 
                                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
cat_final = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=3, 
                               random_seed=42, verbose=False)

print("Training final XGBoost...")
xgb_final.fit(X_train, y_train, sample_weight=full_sample_weights)

print("Training final LightGBM...")
lgb_final.fit(X_train, y_train, sample_weight=full_sample_weights)

print("Training final CatBoost...")
cat_final.fit(X_train, y_train, sample_weight=full_sample_weights)

print("\n=== GENERATING PREDICTIONS ===")

# Get probabilities from each model
xgb_probs = xgb_final.predict_proba(X_test)[:, 1]
lgb_probs = lgb_final.predict_proba(X_test)[:, 1]
cat_probs = cat_final.predict_proba(X_test)[:, 1]

# Weighted ensemble (based on CV performance)
weights = {
    'xgb': np.mean(cv_scores['xgb']),
    'lgb': np.mean(cv_scores['lgb']),
    'cat': np.mean(cv_scores['cat'])
}
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print(f"\nEnsemble weights: XGB={weights['xgb']:.3f}, LGB={weights['lgb']:.3f}, CAT={weights['cat']:.3f}")

# Weighted average
ensemble_probs = (
    weights['xgb'] * xgb_probs + 
    weights['lgb'] * lgb_probs + 
    weights['cat'] * cat_probs
)

# Apply optimal threshold and special rules
print("\n=== APPLYING SPECIAL RULES ===")

# Base predictions with optimal threshold
optimal_threshold = 0.35  # From our analysis
predictions = (ensemble_probs >= optimal_threshold).astype(int)

# Apply null-based rules
rules_applied = 0

# Rule 1: Missing Drained_after_socializing → likely Introvert
drained_null_mask = test_df['Drained_after_socializing_is_null'] == 1
for i in np.where(drained_null_mask)[0]:
    if ensemble_probs[i] < 0.5:  # If already leaning introvert
        predictions[i] = 0
        rules_applied += 1

# Rule 2: No nulls → likely Extrovert
no_nulls_mask = test_df['has_no_nulls'] == 1
for i in np.where(no_nulls_mask)[0]:
    if ensemble_probs[i] > 0.6:  # If already leaning extrovert
        predictions[i] = 1
        rules_applied += 1

# Rule 3: High weighted null score → likely Introvert
high_null_score = test_df['weighted_null_score'] > 0.15
for i in np.where(high_null_score)[0]:
    if ensemble_probs[i] < 0.45:
        predictions[i] = 0
        rules_applied += 1

print(f"Applied special rules to {rules_applied} samples")

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in predictions]
})

# Save submission
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)

submission_path = f'subm/DATE_{date_str}/20250705_0128_null_aware_breakthrough.csv'
submission.to_csv(submission_path, index=False)

# Save detailed results
results = {
    'cv_scores': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'scores': [float(s) for s in v]} 
                  for k, v in cv_scores.items()},
    'ensemble_weights': weights,
    'optimal_threshold': optimal_threshold,
    'rules_applied': rules_applied,
    'prediction_distribution': {
        'introverts': int((predictions == 0).sum()),
        'extroverts': int((predictions == 1).sum()),
        'extrovert_ratio': float((predictions == 1).mean())
    },
    'optuna_results': {
        'best_score': study_xgb.best_value,
        'best_params': best_xgb_params,
        'n_trials': len(study_xgb.trials)
    }
}

with open('output/20250705_0128_breakthrough_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Analyze predictions
print("\n=== PREDICTION ANALYSIS ===")
print(f"Prediction distribution:")
print(f"  Introverts: {(predictions == 0).sum()} ({(predictions == 0).mean()*100:.1f}%)")
print(f"  Extroverts: {(predictions == 1).sum()} ({(predictions == 1).mean()*100:.1f}%)")
print(f"\nProbability distribution:")
print(f"  Min: {ensemble_probs.min():.4f}")
print(f"  Max: {ensemble_probs.max():.4f}")
print(f"  Mean: {ensemble_probs.mean():.4f}")
print(f"  Samples < 0.35: {(ensemble_probs < 0.35).sum()}")
print(f"  Samples > 0.65: {(ensemble_probs > 0.65).sum()}")

print(f"\n=== FINAL SUMMARY ===")
print(f"Best CV score: {max([np.mean(v) for v in cv_scores.values()]):.6f}")
print(f"Models trained: 3 (XGBoost, LightGBM, CatBoost)")
print(f"Ensemble method: Weighted average based on CV performance")
print(f"Special rules applied: {rules_applied}")
print(f"Submission saved to: {submission_path}")

print(f"\nBreakthrough model training completed at: {datetime.now()}")

# RESULT: BREAKTHROUGH ACHIEVED!
# 1. CatBoost achieved 97.997% mean CV score - highest yet!
# 2. XGBoost: 97.717%, LightGBM: 97.663%, Ensemble: 97.868%
# 3. Applied null-based rules to 2,828 samples (45.8% of test set)
# 4. Final prediction: 87.4% Extroverts (higher than training 74%)
# 5. Submission ready: 20250705_0128_null_aware_breakthrough.csv
# This should break the 0.976518 barrier!