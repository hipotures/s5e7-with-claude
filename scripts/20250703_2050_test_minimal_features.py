#!/usr/bin/env python3
"""
PURPOSE: Test with minimal feature sets to find the optimal combination for classification
HYPOTHESIS: A smaller subset of features might perform better than using all features
EXPECTED: Identify the most predictive feature combinations for personality classification
RESULT: Found optimal minimal feature sets that achieve competitive performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import itertools

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Encode Yes/No
for col in train_df.columns:
    if train_df[col].dtype == 'object' and col not in ['Personality', 'id']:
        train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})
        test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})

# Prepare target
le = LabelEncoder()
y_train = le.fit_transform(train_df['Personality'])

print("TESTING MINIMAL FEATURE COMBINATIONS")
print("="*60)

# Test different feature combinations
feature_sets = [
    # Single features
    ['Drained_after_socializing'],
    ['Stage_fear'],
    ['Time_spent_Alone'],
    
    # Two features
    ['Drained_after_socializing', 'Stage_fear'],
    ['Drained_after_socializing', 'Time_spent_Alone'],
    ['Drained_after_socializing', 'Friends_circle_size'],
    
    # Three features
    ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone'],
    ['Drained_after_socializing', 'Stage_fear', 'Friends_circle_size'],
    
    # All original features
    ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
     'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
     'Post_frequency']
]

results = []

for features in feature_sets:
    X_train = train_df[features]
    X_test = test_df[features]
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_score = scores.mean()
    
    # Fit and predict
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Count predictions
    pred_counts = pd.Series(predictions).value_counts()
    
    results.append({
        'features': features,
        'n_features': len(features),
        'cv_score': cv_score,
        'introvert_count': pred_counts.get(1, 0),
        'extrovert_count': pred_counts.get(0, 0)
    })
    
    print(f"\nFeatures: {', '.join(features)}")
    print(f"CV Score: {cv_score:.6f}")
    print(f"Predictions: Introvert={pred_counts.get(1, 0)}, Extrovert={pred_counts.get(0, 0)}")

# Summary
print("\n" + "="*60)
print("SUMMARY (sorted by CV score)")
print("="*60)
results_sorted = sorted(results, key=lambda x: x['cv_score'], reverse=True)

for r in results_sorted[:5]:
    print(f"Score: {r['cv_score']:.6f} | Features: {', '.join(r['features'])}")

# Test decision tree to see rules
print("\n" + "="*60)
print("DECISION TREE ANALYSIS (to see rules)")
print("="*60)

from sklearn.tree import DecisionTreeClassifier, export_text

# Use best features
best_features = results_sorted[0]['features']
X_train = train_df[best_features]

# Simple decision tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

print(f"\nDecision tree rules for features: {best_features}")
tree_rules = export_text(dt, feature_names=best_features)
print(tree_rules)

# Feature importance from tree
print("\nFeature importance from decision tree:")
for feat, imp in zip(best_features, dt.feature_importances_):
    print(f"{feat:30} {imp:.4f}")