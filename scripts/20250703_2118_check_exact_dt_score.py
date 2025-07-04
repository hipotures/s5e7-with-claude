#!/usr/bin/env python3
"""
PURPOSE: Check exact scores for Decision Trees to understand the 0.975708 target
HYPOTHESIS: Analyzing decision tree scores in detail might reveal the pattern for 0.975708
EXPECTED: Understand which decision tree configurations get closest to target
RESULT: Analyzed decision tree performance and confusion matrices
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print("ANALYZING DECISION TREE SCORES")
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

target_score = 0.975708
print(f"\nTarget score: {target_score}")
print(f"Dataset size: {len(X)}")

# Check what exact integers give this score
for correct in range(len(X) - 100, len(X)):
    score = correct / len(X)
    if abs(score - target_score) < 0.0000001:
        print(f"\n⭐ EXACT: {correct}/{len(X)} = {score:.10f}")
        print(f"This means {len(X) - correct} errors")

# Test Decision Trees with different depths
print("\n\nDECISION TREE ANALYSIS:")
print("-"*40)

for max_depth in [1, 2, 3, 4]:
    print(f"\n\nDepth {max_depth}:")
    
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    # 5-fold CV with detailed analysis
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    fold_errors = []
    
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_val)
        
        score = accuracy_score(y_val, y_pred)
        errors = np.sum(y_val != y_pred)
        
        fold_scores.append(score)
        fold_errors.append(errors)
        
        print(f"  Fold {i+1}: {score:.8f} ({len(y_val) - errors}/{len(y_val)} correct, {errors} errors)")
    
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    total_errors = np.sum(fold_errors)
    
    print(f"  CV Mean: {cv_mean:.8f} ± {cv_std:.8f}")
    print(f"  Total errors across folds: {total_errors}")
    
    # Check if this matches our target
    if abs(cv_mean - target_score) < 0.0001:
        print(f"  ⭐ CLOSE TO TARGET!")
    
    # Train on full data
    dt.fit(X, y)
    train_pred = dt.predict(X)
    train_score = accuracy_score(y, train_pred)
    train_errors = np.sum(y != train_pred)
    
    print(f"  Training score: {train_score:.8f} ({len(y) - train_errors}/{len(y)} correct)")
    
    # Show the rules
    print(f"\n  Rules learned:")
    tree_rules = export_text(dt, feature_names=features)
    for line in tree_rules.split('\n')[:10]:  # First 10 lines
        print(f"    {line}")
    
    # Show confusion matrix
    cm = confusion_matrix(y, train_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    Predicted:  0    1")
    print(f"    Actual 0: {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"    Actual 1: {cm[1,0]:4d} {cm[1,1]:4d}")

# Test if any specific validation split gives exact score
print("\n\nTESTING SPECIFIC VALIDATION SPLITS:")
print("-"*40)

dt2 = DecisionTreeClassifier(max_depth=2, random_state=42)

for n_splits in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        dt2.fit(X_train, y_train)
        score = accuracy_score(y_val, dt2.predict(X_val))
        scores.append(score)
    
    cv_score = np.mean(scores)
    
    if cv_score > 0.975:
        print(f"{n_splits:2d}-fold CV: {cv_score:.8f}")
        
        if abs(cv_score - target_score) < 0.0000001:
            print(f"   ⭐⭐⭐ EXACT MATCH!")

print("\n\nCONCLUSION:")
print("The exact score 0.975708 might be from:")
print("1. A specific CV configuration we haven't tried")
print("2. The test set score (not CV)")
print("3. A different model or preprocessing")