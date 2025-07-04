#!/usr/bin/env python3
"""
PURPOSE: Test if 0.975708 comes from a specific validation method
HYPOTHESIS: Different validation strategies (KFold, StratifiedKFold, etc.) might produce the exact score
EXPECTED: Find which validation method produces exactly 0.975708 accuracy
RESULT: Tested multiple validation approaches to reproduce the target score
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, 
    RepeatedStratifiedKFold, cross_val_score
)
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print("TESTING DIFFERENT VALIDATION METHODS")
print("="*60)

# Prepare data
features = ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone']
train_df[features] = train_df[features].fillna(0)
test_df[features] = test_df[features].fillna(0)

for col in features:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})
        test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})

X = train_df[features]
y = LabelEncoder().fit_transform(train_df['Personality'])
X_test = test_df[features]

target_score = 0.975708

# Standard model config
model_params = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.3,
    'random_state': 42,
    'eval_metric': 'logloss',
    'verbosity': 0
}

print(f"\nTarget score: {target_score}")
print(f"Using model: XGBoost with {model_params}\n")

# Test 1: Regular K-Fold (not stratified)
print("TEST 1: Regular K-Fold Cross-Validation")
print("-"*40)

for n_splits in [3, 4, 5, 10]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = xgb.XGBClassifier(**model_params)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    mean_score = scores.mean()
    print(f"{n_splits}-fold: {mean_score:.8f}")
    if abs(mean_score - target_score) < 0.00001:
        print(f"⭐ EXACT MATCH!")

# Test 2: Repeated Stratified K-Fold
print("\n\nTEST 2: Repeated Stratified K-Fold")
print("-"*40)

for n_splits in [5, 10]:
    for n_repeats in [2, 3]:
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        model = xgb.XGBClassifier(**model_params)
        scores = cross_val_score(model, X, y, cv=rskf, scoring='accuracy')
        mean_score = scores.mean()
        print(f"{n_splits}-fold, {n_repeats} repeats: {mean_score:.8f}")
        if abs(mean_score - target_score) < 0.00001:
            print(f"⭐ EXACT MATCH!")

# Test 3: Single validation split
print("\n\nTEST 3: Single Train/Validation Split")
print("-"*40)

for test_size in [0.1, 0.15, 0.2, 0.25, 0.3, 0.33]:
    for rs in range(50):  # Try first 50 random states
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=test_size, random_state=rs, stratify=y
        )
        
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_tr, y_tr)
        val_score = accuracy_score(y_val, model.predict(X_val))
        
        if abs(val_score - target_score) < 0.00001:
            print(f"⭐ EXACT MATCH! test_size={test_size}, random_state={rs}")
            print(f"   Validation score: {val_score:.8f}")
            print(f"   Val set size: {len(X_val)}")
            
            # Save this configuration
            model_final = xgb.XGBClassifier(**model_params)
            model_final.fit(X, y)
            predictions = model_final.predict(X_test)
            pred_labels = LabelEncoder().fit(train_df['Personality']).inverse_transform(predictions)
            
            submission = pd.DataFrame({
                'id': test_df['id'],
                'Personality': pred_labels
            })
            submission.to_csv(f'submission_VAL_SPLIT_{test_size}_{rs}.csv', index=False)
            print(f"   Saved submission!")
        elif val_score > 0.975:
            print(f"Close: test_size={test_size}, rs={rs} -> {val_score:.8f}")

# Test 4: Maybe it's the training accuracy?
print("\n\nTEST 4: Training Set Performance")
print("-"*40)

# Try different model complexities
test_configs = [
    {'n_estimators': 5, 'max_depth': 2},
    {'n_estimators': 10, 'max_depth': 2},
    {'n_estimators': 20, 'max_depth': 2},
    {'n_estimators': 50, 'max_depth': 3},
    {'n_estimators': 100, 'max_depth': 3},
    {'n_estimators': 200, 'max_depth': 4},
]

for config in test_configs:
    model = xgb.XGBClassifier(
        **config,
        learning_rate=0.3,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    
    model.fit(X, y)
    train_score = accuracy_score(y, model.predict(X))
    
    print(f"Config {config}: Train accuracy = {train_score:.8f}")
    
    if abs(train_score - target_score) < 0.00001:
        print(f"⭐ EXACT MATCH on training set!")
        # This would mean overfitting to training data

# Test 5: Maybe it's Out-of-Bag score?
print("\n\nTEST 5: Bootstrap/OOB-like validation")
print("-"*40)

from sklearn.model_selection import cross_val_predict

model = xgb.XGBClassifier(**model_params)

# Get OOB predictions using cross_val_predict
oob_predictions = cross_val_predict(model, X, y, cv=5, method='predict')
oob_score = accuracy_score(y, oob_predictions)

print(f"OOB-like score (5-fold): {oob_score:.8f}")

if abs(oob_score - target_score) < 0.00001:
    print(f"⭐ EXACT MATCH with OOB predictions!")

# Summary
print("\n\nSUMMARY")
print("="*60)
print("If exact match was found, check the submissions created.")
print("If not, the score might be:")
print("1. From a specific preprocessing (e.g., different NaN handling)")
print("2. From a specific feature subset")
print("3. From a different model (not XGBoost)")
print("4. The actual leaderboard score (not CV)")
print("5. From a specific seed/configuration we haven't tried yet")