#!/usr/bin/env python3
"""
PURPOSE: Check if 0.975708 score could be from a specific validation method
HYPOTHESIS: The exact score 0.975708 might come from a particular CV split or validation strategy
EXPECTED: Reproduce the exact score by finding the right validation configuration
RESULT: Explored various validation methods to match the target accuracy
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv("../../train.csv")

# Prepare (best config)
features = ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone']
train_df[features] = train_df[features].fillna(0)

for col in features:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})

X = train_df[features]
y = LabelEncoder().fit_transform(train_df['Personality'])

print("CHECKING EXACT VALIDATION METHODS")
print("="*60)

# Calculate exact accuracy: 0.975708
target_score = 0.975708
target_correct = int(round(target_score * len(X)))
print(f"\nTarget score: {target_score}")
print(f"This means {target_correct} correct out of {len(X)} samples")
print(f"Or {len(X) - target_correct} errors")

# Try different validation splits
print("\n\nTESTING DIFFERENT TRAIN/VAL SPLITS:")
print("-"*40)

test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3]
random_states = [0, 1, 42, 123, 2024]

best_val_score = 0
best_split_info = None

for test_size in test_sizes:
    for rs in random_states:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=rs, stratify=y
        )
        
        # Simple model
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        val_score = model.score(X_val, y_val)
        
        if abs(val_score - target_score) < 0.0001:
            print(f"⭐ EXACT MATCH! test_size={test_size}, random_state={rs}: {val_score:.6f}")
            best_split_info = (test_size, rs)
        elif val_score > 0.975:
            print(f"Close: test_size={test_size}, random_state={rs}: {val_score:.6f}")
            
        if val_score > best_val_score:
            best_val_score = val_score

print(f"\nBest validation score found: {best_val_score:.6f}")

# Check if it could be from cross-validation
print("\n\nCHECKING K-FOLD VARIATIONS:")
print("-"*40)

for n_splits in [3, 4, 5, 10]:
    for rs in [42, 0, 1]:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
        
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            
            score = model.score(X_val, y_val)
            scores.append(score)
        
        mean_score = np.mean(scores)
        
        if abs(mean_score - target_score) < 0.0001:
            print(f"⭐ EXACT MATCH! {n_splits}-fold, rs={rs}: {mean_score:.6f}")
        elif mean_score > 0.975:
            print(f"Close: {n_splits}-fold, rs={rs}: {mean_score:.6f}")

# Maybe it's the training accuracy?
print("\n\nCHECKING TRAINING ACCURACY:")
print("-"*40)

model_train = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
model_train.fit(X, y)
train_score = model_train.score(X, y)
print(f"Training accuracy: {train_score:.6f}")

if abs(train_score - target_score) < 0.0001:
    print("⭐ EXACT MATCH! This is the training accuracy!")

# Check with different model complexities
print("\n\nCHECKING DIFFERENT MODEL COMPLEXITIES:")
print("-"*40)

for n_est in [10, 20, 50, 100, 200]:
    for depth in [2, 3, 4]:
        model_test = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
        
        # Check training accuracy
        model_test.fit(X, y)
        train_acc = model_test.score(X, y)
        
        # Check CV
        scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]
            
            model_cv = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
            model_cv.fit(X_tr, y_tr)
            scores.append(model_cv.score(X_vl, y_vl))
        
        cv_acc = np.mean(scores)
        
        if abs(train_acc - target_score) < 0.0001:
            print(f"⭐ TRAIN MATCH! n_est={n_est}, depth={depth}: train={train_acc:.6f}")
        if abs(cv_acc - target_score) < 0.0001:
            print(f"⭐ CV MATCH! n_est={n_est}, depth={depth}: cv={cv_acc:.6f}")
            
        if train_acc > 0.975 or cv_acc > 0.975:
            print(f"n_est={n_est}, depth={depth}: train={train_acc:.6f}, cv={cv_acc:.6f}")