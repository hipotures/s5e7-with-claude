#!/usr/bin/env python3
"""
PURPOSE: Search for hidden patterns in feature interactions that could help achieve 0.976518
HYPOTHESIS: The winner found specific feature combinations or ratios that better identify ambiverts
EXPECTED: Discover non-linear relationships and interaction effects
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
from itertools import combinations
import json

# Load data
print("Loading data...")
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

# Prepare base features
X = train_df.drop(['id', 'Personality'], axis=1).copy()
y = (train_df['Personality'] == 'Extrovert').astype(int)

X_test = test_df.drop(['id'], axis=1).copy()

# Convert categorical columns
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    X[col] = X[col].map({'Yes': 1, 'No': 0})
    X[col] = X[col].fillna(0.5)
    X_test[col] = X_test[col].map({'Yes': 1, 'No': 0})
    X_test[col] = X_test[col].fillna(0.5)

print("Original shape:", X.shape)

# Create feature interactions
print("\n=== CREATING FEATURE INTERACTIONS ===")

# 1. Key ratios based on domain knowledge
print("\n1. Creating ratios...")
X['social_to_alone_ratio'] = (X['Social_event_attendance'] + 1) / (X['Time_spent_Alone'] + 1)
X['friends_to_posts_ratio'] = (X['Friends_circle_size'] + 1) / (X['Post_frequency'] + 1)
X['outdoor_to_alone_ratio'] = (X['Going_outside'] + 1) / (X['Time_spent_Alone'] + 1)
X['social_activity_index'] = X['Social_event_attendance'] * X['Going_outside'] / 10

X_test['social_to_alone_ratio'] = (X_test['Social_event_attendance'] + 1) / (X_test['Time_spent_Alone'] + 1)
X_test['friends_to_posts_ratio'] = (X_test['Friends_circle_size'] + 1) / (X_test['Post_frequency'] + 1)
X_test['outdoor_to_alone_ratio'] = (X_test['Going_outside'] + 1) / (X_test['Time_spent_Alone'] + 1)
X_test['social_activity_index'] = X_test['Social_event_attendance'] * X_test['Going_outside'] / 10

# 2. Polynomial features for key variables
print("2. Creating polynomial features...")
key_features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']
for feat in key_features:
    X[f'{feat}_squared'] = X[feat] ** 2
    X[f'{feat}_sqrt'] = np.sqrt(X[feat])
    X[f'{feat}_log1p'] = np.log1p(X[feat])
    
    X_test[f'{feat}_squared'] = X_test[feat] ** 2
    X_test[f'{feat}_sqrt'] = np.sqrt(X_test[feat])
    X_test[f'{feat}_log1p'] = np.log1p(X_test[feat])

# 3. Interaction features between key pairs
print("3. Creating interaction features...")
interaction_pairs = [
    ('Drained_after_socializing', 'Social_event_attendance'),
    ('Stage_fear', 'Friends_circle_size'),
    ('Time_spent_Alone', 'Drained_after_socializing'),
    ('Going_outside', 'Post_frequency')
]

for feat1, feat2 in interaction_pairs:
    X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
    X[f'{feat1}_plus_{feat2}'] = X[feat1] + X[feat2]
    X[f'{feat1}_minus_{feat2}'] = X[feat1] - X[feat2]
    
    X_test[f'{feat1}_x_{feat2}'] = X_test[feat1] * X_test[feat2]
    X_test[f'{feat1}_plus_{feat2}'] = X_test[feat1] + X_test[feat2]
    X_test[f'{feat1}_minus_{feat2}'] = X_test[feat1] - X_test[feat2]

# 4. Special indicators for ambiverts
print("4. Creating ambivert indicators...")

# Indicator for moderate social behavior
X['moderate_social'] = ((X['Social_event_attendance'] >= 4) & 
                        (X['Social_event_attendance'] <= 7)).astype(int)
X['moderate_friends'] = ((X['Friends_circle_size'] >= 6) & 
                         (X['Friends_circle_size'] <= 12)).astype(int)
X['low_alone_time'] = (X['Time_spent_Alone'] <= 3).astype(int)

X_test['moderate_social'] = ((X_test['Social_event_attendance'] >= 4) & 
                             (X_test['Social_event_attendance'] <= 7)).astype(int)
X_test['moderate_friends'] = ((X_test['Friends_circle_size'] >= 6) & 
                              (X_test['Friends_circle_size'] <= 12)).astype(int)
X_test['low_alone_time'] = (X_test['Time_spent_Alone'] <= 3).astype(int)

# Ambivert score based on multiple indicators
X['ambivert_score'] = (
    X['moderate_social'] * 0.3 +
    X['moderate_friends'] * 0.3 +
    X['low_alone_time'] * 0.2 +
    (1 - X['Drained_after_socializing']) * 0.2
)

X_test['ambivert_score'] = (
    X_test['moderate_social'] * 0.3 +
    X_test['moderate_friends'] * 0.3 +
    X_test['low_alone_time'] * 0.2 +
    (1 - X_test['Drained_after_socializing']) * 0.2
)

print(f"\nFeatures after engineering: {X.shape[1]}")

# Evaluate models with different feature sets
print("\n=== EVALUATING FEATURE SETS ===")

# Baseline model with original features
X_baseline = X[train_df.columns.drop(['id', 'Personality']).tolist()]
model_baseline = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
cv_baseline = cross_val_score(model_baseline, X_baseline, y, cv=5, scoring='accuracy')
print(f"\nBaseline (original features): {cv_baseline.mean():.6f} (+/- {cv_baseline.std():.6f})")

# Model with all engineered features
model_full = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
cv_full = cross_val_score(model_full, X, y, cv=5, scoring='accuracy')
print(f"With all engineered features: {cv_full.mean():.6f} (+/- {cv_full.std():.6f})")

# Train final model to analyze feature importance
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
model_full.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features:")
print(feature_importance.head(20))

# Test different probability thresholds
print("\n=== THRESHOLD OPTIMIZATION ===")
train_probs = model_full.predict_proba(X)[:, 1]

best_threshold = 0.5
best_accuracy = 0
for threshold in np.arange(0.35, 0.65, 0.01):
    preds = (train_probs >= threshold).astype(int)
    accuracy = (preds == y).mean()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f}")
print(f"Best accuracy: {best_accuracy:.6f}")

# Analyze predictions on ambiguous cases
print("\n=== AMBIGUOUS CASE ANALYSIS ===")

# Define ambiguous cases based on probability
ambiguous_mask = (train_probs >= 0.35) & (train_probs <= 0.65)
print(f"Ambiguous cases: {ambiguous_mask.sum()} ({ambiguous_mask.sum()/len(y)*100:.1f}%)")
print(f"Extrovert rate in ambiguous: {y[ambiguous_mask].mean():.1%}")

# Look for patterns in very confident wrong predictions
very_confident_wrong = ((train_probs > 0.9) & (y == 0)) | ((train_probs < 0.1) & (y == 1))
print(f"\nVery confident wrong predictions: {very_confident_wrong.sum()}")

# Generate predictions for test set
print("\n=== GENERATING TEST PREDICTIONS ===")
test_probs = model_full.predict_proba(X_test)[:, 1]

# Apply refined strategy
predictions = np.zeros(len(test_probs))

# Use dynamic thresholds based on ambivert score
for i in range(len(test_probs)):
    if X_test.iloc[i]['ambivert_score'] > 0.5:  # Likely ambivert
        threshold = 0.42  # Lower threshold for ambiverts
    else:
        threshold = best_threshold
    
    predictions[i] = int(test_probs[i] >= threshold)

# Save insights and prepare submission
insights = {
    'baseline_cv': float(cv_baseline.mean()),
    'engineered_cv': float(cv_full.mean()),
    'improvement': float(cv_full.mean() - cv_baseline.mean()),
    'best_threshold': float(best_threshold),
    'ambiguous_cases_pct': float(ambiguous_mask.sum()/len(y)*100),
    'top_10_features': feature_importance.head(10).to_dict('records')
}

with open('scripts/output/20250704_2251_feature_interactions.json', 'w') as f:
    json.dump(insights, f, indent=2)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in predictions.astype(int)]
})

# Create submission directory with date
import os
from datetime import datetime
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)

submission.to_csv(f'subm/DATE_{date_str}/20250704_2251_feature_interactions.csv', index=False)

print(f"\n=== SUMMARY ===")
print(f"Feature engineering improved CV by: {(cv_full.mean() - cv_baseline.mean())*100:.3f}%")
print(f"Best new features: {feature_importance.head(5)['feature'].tolist()}")
print(f"Submission saved to: subm/DATE_{date_str}/20250704_2251_feature_interactions.csv")

# RESULT: [To be filled after execution]