#!/usr/bin/env python3
"""
PURPOSE: Test XGBoost with early stopping to find optimal number of iterations
HYPOTHESIS: Early stopping can prevent overfitting and find the ideal number of boosting rounds
EXPECTED: Identify the optimal number of iterations before performance degrades
RESULT: Found optimal iteration count using validation set and early stopping
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print("TESTING EARLY STOPPING")
print("="*60)

# Prepare data (best configuration so far)
features = ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone']
train_df[features] = train_df[features].fillna(0)
test_df[features] = test_df[features].fillna(0)

# Encode
for col in features:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})
        test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})

X = train_df[features]
y = LabelEncoder().fit_transform(train_df['Personality'])
X_test_final = test_df[features]

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")

# Test different configurations with early stopping
configs = [
    {
        'name': 'Conservative',
        'params': {
            'n_estimators': 5000,  # High number, will stop early
            'max_depth': 3,
            'learning_rate': 0.01,
            'early_stopping_rounds': 50,
            'eval_metric': 'logloss',
            'random_state': 42
        }
    },
    {
        'name': 'Moderate',
        'params': {
            'n_estimators': 5000,
            'max_depth': 4,
            'learning_rate': 0.05,
            'early_stopping_rounds': 30,
            'eval_metric': 'logloss',
            'random_state': 42
        }
    },
    {
        'name': 'Aggressive',
        'params': {
            'n_estimators': 5000,
            'max_depth': 5,
            'learning_rate': 0.1,
            'early_stopping_rounds': 20,
            'eval_metric': 'logloss',
            'random_state': 42
        }
    },
    {
        'name': 'Very_Simple',
        'params': {
            'n_estimators': 5000,
            'max_depth': 2,
            'learning_rate': 0.3,
            'early_stopping_rounds': 10,
            'eval_metric': 'logloss',
            'random_state': 42
        }
    }
]

best_score = 0
best_config = None
best_n_estimators = 0

for config in configs:
    print(f"\n\nTesting: {config['name']}")
    print("-"*40)
    
    # Create model
    model = xgb.XGBClassifier(**config['params'])
    
    # Fit with early stopping
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Get results
    results = model.evals_result()
    train_scores = results['validation_0']['logloss']
    val_scores = results['validation_1']['logloss']
    
    # Best iteration
    best_iteration = model.best_iteration
    val_accuracy = model.score(X_val, y_val)
    
    print(f"Best iteration: {best_iteration}")
    print(f"Validation accuracy: {val_accuracy:.6f}")
    print(f"Final train logloss: {train_scores[best_iteration]:.6f}")
    print(f"Final val logloss: {val_scores[best_iteration]:.6f}")
    
    if val_accuracy > best_score:
        best_score = val_accuracy
        best_config = config
        best_n_estimators = best_iteration
        
        # Train final model with optimal iterations
        final_model = xgb.XGBClassifier(
            n_estimators=best_iteration + 1,
            max_depth=config['params']['max_depth'],
            learning_rate=config['params']['learning_rate'],
            random_state=42
        )
        final_model.fit(X, y)
        
        # Predict
        predictions = final_model.predict(X_test_final)
        pred_labels = LabelEncoder().fit(train_df['Personality']).inverse_transform(predictions)
        
        # Save
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': pred_labels
        })
        filename = f'submission_early_stop_{config["name"]}_{val_accuracy:.6f}_iter{best_iteration}.csv'
        submission.to_csv(filename, index=False)
        print(f"Saved: {filename}")

print(f"\n\nBEST RESULT:")
print(f"Configuration: {best_config['name']}")
print(f"Validation accuracy: {best_score:.6f}")
print(f"Optimal iterations: {best_n_estimators}")

# Try one more approach - very minimal model
print("\n\nTESTING MINIMAL MODELS (might be overfitting with complex models)")
print("-"*60)

minimal_configs = [
    {'n_estimators': 10, 'max_depth': 2},
    {'n_estimators': 20, 'max_depth': 2},
    {'n_estimators': 30, 'max_depth': 2},
    {'n_estimators': 50, 'max_depth': 2},
    {'n_estimators': 10, 'max_depth': 3},
    {'n_estimators': 20, 'max_depth': 3},
    {'n_estimators': 30, 'max_depth': 3},
    {'n_estimators': 50, 'max_depth': 3},
]

for mc in minimal_configs:
    model = xgb.XGBClassifier(
        n_estimators=mc['n_estimators'],
        max_depth=mc['max_depth'],
        learning_rate=0.3,
        random_state=42
    )
    
    # Cross-validation score
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_score = scores.mean()
    
    print(f"n_estimators={mc['n_estimators']:3d}, max_depth={mc['max_depth']}: CV={cv_score:.6f}")
    
    if cv_score > 0.975:
        print("  ⭐ HIGH SCORE FOUND!")
        
        # Train and save
        model.fit(X, y)
        predictions = model.predict(X_test_final)
        pred_labels = LabelEncoder().fit(train_df['Personality']).inverse_transform(predictions)
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': pred_labels
        })
        filename = f'submission_minimal_n{mc["n_estimators"]}_d{mc["max_depth"]}_{cv_score:.6f}.csv'
        submission.to_csv(filename, index=False)
        print(f"  Saved: {filename}")