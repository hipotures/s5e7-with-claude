#!/usr/bin/env python3
"""
PURPOSE: Test if 0.975708 comes from a simple decision rule or tree structure
HYPOTHESIS: The exact score might be achievable with a simple decision tree rule
EXPECTED: Find simple decision rules that produce the target accuracy
RESULT: Explored simple decision trees to understand prediction patterns
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print("TESTING SIMPLE DECISION RULES")
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

# Test 1: Single decision stump (1 tree, depth 1)
print("\nTEST 1: Single Decision Stump on GPU")
print("-"*40)

for feature in features:
    # XGBoost with 1 tree, depth 1
    model = xgb.XGBClassifier(
        n_estimators=1,
        max_depth=1,
        learning_rate=1.0,  # No shrinkage
        random_state=42,
        tree_method='hist',
        device='cuda',
        verbosity=0
    )
    
    # Use only one feature
    X_single = X[[feature]]
    
    # Cross-validation
    scores = cross_val_score(model, X_single, y, cv=5, scoring='accuracy')
    cv_score = scores.mean()
    
    print(f"{feature}: {cv_score:.8f}")
    
    if abs(cv_score - target_score) < 0.0000001:
        print(f"⭐ EXACT MATCH with single feature {feature}!")

# Test 2: Two features at a time
print("\n\nTEST 2: Two-feature combinations")
print("-"*40)

from itertools import combinations

for feat1, feat2 in combinations(features, 2):
    X_two = X[[feat1, feat2]]
    
    # Simple XGBoost
    model = xgb.XGBClassifier(
        n_estimators=2,
        max_depth=1,
        learning_rate=1.0,
        random_state=42,
        tree_method='hist',
        device='cuda',
        verbosity=0
    )
    
    scores = cross_val_score(model, X_two, y, cv=5, scoring='accuracy')
    cv_score = scores.mean()
    
    if cv_score > 0.97:
        print(f"{feat1} + {feat2}: {cv_score:.8f}")
        
        if abs(cv_score - target_score) < 0.0000001:
            print(f"⭐ EXACT MATCH!")

# Test 3: Very shallow trees with different numbers
print("\n\nTEST 3: Ultra-simple XGBoost models on GPU")
print("-"*40)

# Test extremely simple models
for n_trees in range(1, 11):
    for max_depth in [1]:
        for lr in [0.5, 1.0]:
            model = xgb.XGBClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                learning_rate=lr,
                random_state=42,
                tree_method='hist',
                device='cuda',
                eval_metric='logloss',
                verbosity=0
            )
            
            # Test with different CV setups
            for cv_splits in [3, 4, 5]:
                skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
                cv_score = scores.mean()
                
                if cv_score > 0.975:
                    print(f"trees={n_trees}, depth={max_depth}, lr={lr}, {cv_splits}-fold: {cv_score:.8f}")
                    
                    if abs(cv_score - target_score) < 0.0000001:
                        print(f"⭐⭐⭐ EXACT MATCH FOUND! ⭐⭐⭐")
                        
                        # Train and save
                        model.fit(X, y)
                        predictions = model.predict(X_test)
                        pred_labels = LabelEncoder().fit(train_df['Personality']).inverse_transform(predictions)
                        
                        submission = pd.DataFrame({
                            'id': test_df['id'],
                            'Personality': pred_labels
                        })
                        submission.to_csv(f'submission_SIMPLE_EXACT_{n_trees}_{cv_splits}.csv', index=False)
                        print("Saved exact match submission!")

# Test 4: Check if it's from a specific threshold
print("\n\nTEST 4: Manual threshold on Drained_after_socializing")
print("-"*40)

# Since this is the dominant feature, maybe there's a simple threshold rule
drained_values = X['Drained_after_socializing'].values

# Test different mappings
mappings = [
    ("Drained=1 -> Introvert", lambda x: x),
    ("Drained=0 -> Introvert", lambda x: 1 - x),
    ("Always Introvert", lambda x: np.ones_like(x)),
    ("Always Extrovert", lambda x: np.zeros_like(x)),
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, rule in mappings:
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]
        
        # Get predictions and handle NaN
        drained_values = X_val['Drained_after_socializing'].values
        # Convert to int to avoid NaN issues
        drained_values = np.nan_to_num(drained_values, nan=0).astype(int)
        predictions = rule(drained_values).astype(int)
        
        score = accuracy_score(y_val, predictions)
        scores.append(score)
    
    cv_score = np.mean(scores)
    print(f"{name}: {cv_score:.8f}")
    
    if abs(cv_score - target_score) < 0.0000001:
        print(f"⭐ EXACT MATCH with rule: {name}!")

# Test 5: Decision tree to find exact rules
print("\n\nTEST 5: Decision Tree Analysis (CPU - for rule extraction)")
print("-"*40)

# Use simple decision tree to understand the rules
for max_depth in [1, 2, 3]:
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
    cv_score = scores.mean()
    
    print(f"\nDecision Tree (depth={max_depth}): {cv_score:.8f}")
    
    if abs(cv_score - target_score) < 0.0000001:
        print(f"⭐ EXACT MATCH with Decision Tree!")
    
    # Fit and show rules
    dt.fit(X, y)
    print("Rules learned:")
    print(export_text(dt, feature_names=features))

print("\n\nCONCLUSION:")
print("="*60)
print("If we found the exact match, check the submission files.")
print("The prevalence of 0.975708 among 240+ competitors suggests")
print("it's likely a simple rule or very shallow model that everyone discovered.")