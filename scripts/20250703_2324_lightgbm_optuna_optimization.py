#!/usr/bin/env python3
"""
PURPOSE: LightGBM with Optuna hyperparameter optimization
HYPOTHESIS: Automated hyperparameter tuning can find optimal LightGBM configuration
EXPECTED: Optuna will search parameter space to maximize CV accuracy
RESULT: Found best LightGBM parameters through Bayesian optimization
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

# Load data
print("LIGHTGBM + OPTUNA HYPERPARAMETER OPTIMIZATION")
print("="*60)

train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Best features
features = ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone']
train_df[features] = train_df[features].fillna(0)
test_df[features] = test_df[features].fillna(0)

for col in features:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})
        test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})

X_train = train_df[features]
X_test = test_df[features]
le = LabelEncoder()
y_train = le.fit_transform(train_df['Personality'])

print(f"Features: {features}")
print(f"Train shape: {X_train.shape}")

# Optuna objective function
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1,
        
        # Parameters to optimize
        'num_leaves': trial.suggest_int('num_leaves', 4, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    # Also try extra_trees mode
    if trial.suggest_categorical('use_extra_trees', [True, False]):
        params['extra_trees'] = True
    
    # 5-fold CV
    model = lgb.LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    
    return scores.mean()

# Run optimization
print("\nStarting Optuna optimization...")
print("-"*60)

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Optimize for 100 trials or 10 minutes
start_time = time.time()
study.optimize(
    objective, 
    n_trials=100,
    timeout=600,  # 10 minutes max
    n_jobs=1,  # Use 1 job to avoid conflicts
    show_progress_bar=True
)

optimization_time = time.time() - start_time
print(f"\nOptimization completed in {optimization_time:.1f} seconds")

# Show results
print("\n" + "="*60)
print("OPTIMIZATION RESULTS:")
print("="*60)
print(f"Best CV score: {study.best_value:.6f}")
print("\nBest parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best parameters
print("\n" + "="*60)
print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("="*60)

best_params = study.best_params.copy()
best_params.update({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'random_state': 42,
    'verbose': -1
})

# Handle extra_trees parameter
use_extra_trees = best_params.pop('use_extra_trees', False)
if use_extra_trees:
    best_params['extra_trees'] = True

best_model = lgb.LGBMClassifier(**best_params)

# Final CV score
final_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Final CV score: {final_scores.mean():.6f} ± {final_scores.std():.6f}")

# Train on full data
best_model.fit(X_train, y_train)

# Feature importance
print("\nFeature importance:")
for feat, imp in zip(features, best_model.feature_importances_):
    print(f"  {feat}: {imp}")

# Make predictions
predictions = best_model.predict(X_test)
pred_labels = le.inverse_transform(predictions)

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': pred_labels
})

filename = f'submission_LGB_Optuna_{study.best_value:.6f}.csv'
submission.to_csv(filename, index=False)
print(f"\nSaved: {filename}")

# Show optimization history
print("\n" + "="*60)
print("TOP 10 TRIALS:")
print("="*60)

trials_df = study.trials_dataframe()
top_trials = trials_df.nlargest(10, 'value')[['value', 'duration', 'params_num_leaves', 
                                               'params_n_estimators', 'params_learning_rate']]
print(top_trials.to_string())

# Try a few more specific configurations based on insights
print("\n" + "="*60)
print("TESTING EXTREME CONFIGURATIONS:")
print("="*60)

extreme_configs = [
    {
        'name': 'Single_Stump',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 2,  # Minimum - just a split
            'n_estimators': 1,
            'learning_rate': 1.0,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'Two_Stumps',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 2,
            'n_estimators': 2,
            'learning_rate': 1.0,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'Optuna_Simplified',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': min(10, best_params.get('num_leaves', 31)),
            'n_estimators': min(20, best_params.get('n_estimators', 100)),
            'learning_rate': 0.3,
            'random_state': 42,
            'verbose': -1
        }
    }
]

for config in extreme_configs:
    model = lgb.LGBMClassifier(**config['params'])
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_score = scores.mean()
    print(f"\n{config['name']}: CV = {cv_score:.6f}")
    
    if cv_score > 0.970 or config['name'] == 'Single_Stump':
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        labels = le.inverse_transform(pred)
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': labels
        })
        submission.to_csv(f'submission_LGB_{config["name"]}_{cv_score:.6f}.csv', index=False)
        print(f"  Saved: submission_LGB_{config['name']}_{cv_score:.6f}.csv")

print("\n\nDONE! Check submissions with Optuna-optimized parameters.")