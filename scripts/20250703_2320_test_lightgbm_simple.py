#!/usr/bin/env python3
"""
PURPOSE: Test LightGBM with our best features configuration
HYPOTHESIS: LightGBM might achieve the target score with specific configurations
EXPECTED: Compare various LightGBM configurations including AutoGluon defaults
RESULT: Tested multiple LightGBM configurations and compared with XGBoost
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import time

# Load data
print("TESTING LIGHTGBM WITH BEST FEATURES")
print("="*60)

train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Best features configuration
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
print(f"Test shape: {X_test.shape}")

# Test different LightGBM configurations
configs = [
    {
        'name': 'Simple',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'Very_Simple',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 8,
            'learning_rate': 0.3,
            'n_estimators': 50,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'Ultra_Simple',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 4,
            'learning_rate': 0.5,
            'n_estimators': 10,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'AutoGluon_Default',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'AutoGluon_Large',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 128,
            'learning_rate': 0.03,
            'n_estimators': 1000,
            'feature_fraction': 0.9,
            'min_data_in_leaf': 3,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'AutoGluon_XT',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'extra_trees': True,  # This makes it similar to Extra Trees
            'random_state': 42,
            'verbose': -1
        }
    }
]

print("\nTesting LightGBM configurations:")
print("-"*60)

best_score = 0
best_config = None

for config in configs:
    print(f"\nTesting: {config['name']}")
    
    # Create model
    model = lgb.LGBMClassifier(**config['params'])
    
    # 5-fold CV
    start_time = time.time()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    cv_time = time.time() - start_time
    
    cv_mean = scores.mean()
    cv_std = scores.std()
    
    print(f"  CV Score: {cv_mean:.6f} ± {cv_std:.6f}")
    print(f"  Time: {cv_time:.2f}s")
    
    if cv_mean > best_score:
        best_score = cv_mean
        best_config = config
        
    # Train on full data and save submission
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    pred_labels = le.inverse_transform(predictions)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': pred_labels
    })
    
    filename = f'submission_LGB_{config["name"]}_{cv_mean:.6f}.csv'
    submission.to_csv(filename, index=False)
    print(f"  Saved: {filename}")

print(f"\n\nBest configuration: {best_config['name']}")
print(f"Best CV score: {best_score:.6f}")

# Compare with XGBoost reference
print("\n" + "="*60)
print("COMPARISON WITH XGBOOST:")
print("="*60)

import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.3,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=skf, scoring='accuracy')
print(f"XGBoost CV: {xgb_scores.mean():.6f} ± {xgb_scores.std():.6f}")
print(f"LightGBM best: {best_score:.6f}")
print(f"Difference: {best_score - xgb_scores.mean():.6f}")

# Try one more thing - single decision tree in LightGBM
print("\n" + "="*60)
print("TESTING SINGLE TREE (like Decision Tree):")
print("="*60)

single_tree = lgb.LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    num_leaves=4,  # Very few leaves
    n_estimators=1,  # Single tree!
    learning_rate=1.0,  # No shrinkage
    random_state=42,
    verbose=-1
)

single_scores = cross_val_score(single_tree, X_train, y_train, cv=skf, scoring='accuracy')
print(f"Single LightGBM tree CV: {single_scores.mean():.6f}")

# Save single tree submission
single_tree.fit(X_train, y_train)
single_pred = single_tree.predict(X_test)
single_labels = le.inverse_transform(single_pred)

submission_single = pd.DataFrame({
    'id': test_df['id'],
    'Personality': single_labels
})
submission_single.to_csv(f'submission_LGB_SingleTree_{single_scores.mean():.6f}.csv', index=False)
print(f"Saved: submission_LGB_SingleTree_{single_scores.mean():.6f}.csv")