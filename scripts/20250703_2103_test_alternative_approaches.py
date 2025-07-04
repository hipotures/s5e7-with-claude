#!/usr/bin/env python3
"""
PURPOSE: Test alternative ML approaches that might yield 0.975708 score
HYPOTHESIS: Different algorithms (Decision Trees, Random Forest, Logistic Regression) might achieve the target
EXPECTED: Find which algorithm and configuration produces the exact score
RESULT: Evaluated multiple ML algorithms to match the target accuracy
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print("TESTING ALTERNATIVE APPROACHES")
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

# Test 1: Different models (not XGBoost)
print("\nTEST 1: Different ML Models")
print("-"*40)

models = [
    ('Decision Tree (depth=2)', DecisionTreeClassifier(max_depth=2, random_state=42)),
    ('Decision Tree (depth=3)', DecisionTreeClassifier(max_depth=3, random_state=42)),
    ('Decision Tree (depth=4)', DecisionTreeClassifier(max_depth=4, random_state=42)),
    ('Random Forest (10 trees)', RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
    ('Random Forest (50 trees)', RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)),
    ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
    ('Logistic Regression (C=0.1)', LogisticRegression(C=0.1, random_state=42, max_iter=1000)),
    ('Logistic Regression (C=10)', LogisticRegression(C=10, random_state=42, max_iter=1000)),
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models:
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    
    cv_score = np.mean(scores)
    print(f"{name}: {cv_score:.8f}")
    
    if abs(cv_score - target_score) < 0.00001:
        print(f"⭐ EXACT MATCH with {name}!")

# Test 2: Using only subset of data
print("\n\nTEST 2: Using Data Subsets")
print("-"*40)

# Maybe the score is from a specific subset
subset_sizes = [0.5, 0.6, 0.7, 0.8, 0.9]

for subset_size in subset_sizes:
    n_samples = int(len(X) * subset_size)
    X_subset = X.iloc[:n_samples]
    y_subset = y[:n_samples]
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    scores = []
    
    skf_subset = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf_subset.split(X_subset, y_subset):
        X_tr, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
        y_tr, y_val = y_subset[train_idx], y_subset[val_idx]
        
        model.fit(X_tr, y_tr)
        scores.append(model.score(X_val, y_val))
    
    cv_score = np.mean(scores)
    print(f"Subset {subset_size*100:.0f}% ({n_samples} samples): {cv_score:.8f}")
    
    if abs(cv_score - target_score) < 0.00001:
        print(f"⭐ EXACT MATCH with subset!")

# Test 3: Different scoring metrics
print("\n\nTEST 3: Different Scoring Metrics")
print("-"*40)

model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)

# Calculate different metrics
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'f1': 'f1',
    'f1_weighted': 'f1_weighted',
    'f1_macro': 'f1_macro',
    'precision': 'precision',
    'recall': 'recall'
}

cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric, scores in cv_results.items():
    if metric.startswith('test_'):
        metric_name = metric.replace('test_', '')
        mean_score = scores.mean()
        print(f"{metric_name}: {mean_score:.8f}")
        
        if abs(mean_score - target_score) < 0.00001:
            print(f"⭐ EXACT MATCH with {metric_name}!")

# Test 4: Manual rule-based approach
print("\n\nTEST 4: Rule-Based Classification")
print("-"*40)

# Since Drained_after_socializing is the dominant feature
# Let's test simple rules

def rule_based_predict(X):
    # Simple rule: if Drained_after_socializing == 1, predict Introvert (1)
    predictions = np.where(X['Drained_after_socializing'] == 1, 1, 0)
    return predictions

# Test on CV
scores = []
for train_idx, val_idx in skf.split(X, y):
    X_val = X.iloc[val_idx]
    y_val = y[val_idx]
    
    y_pred = rule_based_predict(X_val)
    scores.append(accuracy_score(y_val, y_pred))

rule_score = np.mean(scores)
print(f"Simple rule (Drained->Introvert): {rule_score:.8f}")

# Try inverse rule
def rule_based_predict_inverse(X):
    # Inverse: if Drained_after_socializing == 0, predict Introvert (1)
    predictions = np.where(X['Drained_after_socializing'] == 0, 1, 0)
    return predictions

scores_inv = []
for train_idx, val_idx in skf.split(X, y):
    X_val = X.iloc[val_idx]
    y_val = y[val_idx]
    
    y_pred = rule_based_predict_inverse(X_val)
    scores_inv.append(accuracy_score(y_val, y_pred))

rule_score_inv = np.mean(scores_inv)
print(f"Inverse rule (NotDrained->Introvert): {rule_score_inv:.8f}")

# Test 5: Check class distribution
print("\n\nTEST 5: Class Distribution Analysis")
print("-"*40)

# Check if 0.975708 could be from always predicting one class
class_counts = np.bincount(y)
majority_class_ratio = class_counts.max() / len(y)
print(f"Majority class ratio: {majority_class_ratio:.8f}")

if abs(majority_class_ratio - target_score) < 0.00001:
    print(f"⭐ EXACT MATCH! This is the majority class baseline!")

# Check distribution in each fold
print("\nClass distribution per fold:")
for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    y_val = y[val_idx]
    val_counts = np.bincount(y_val)
    val_ratio = val_counts[1] / len(y_val) if len(val_counts) > 1 else 0
    print(f"Fold {i+1}: Class 1 ratio = {val_ratio:.6f}")

print("\n\nFINAL ANALYSIS")
print("="*60)
print("The exact score 0.975708 shared by 240+ competitors suggests:")
print("1. A deterministic solution exists")
print("2. It might be a simple rule or threshold")
print("3. It could be the result of overfitting to specific validation")
print("4. Or a specific model/parameter combination we haven't found yet")