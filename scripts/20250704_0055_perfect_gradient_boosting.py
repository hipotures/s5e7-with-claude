#!/usr/bin/env python3
"""Use Gradient Boosting like the 94.4% accuracy project, but with our insights."""

# PURPOSE: Test Gradient Boosting approach with special handling for the 2.43% ambiguous cases
# HYPOTHESIS: The 2.43% most uncertain predictions should be mostly Extroverts based on pattern analysis
# EXPECTED: Achieve >97.5% accuracy by correctly handling ambiguous cases with modified thresholds
# RESULT: Multiple submission strategies created including ultra-simple rule based on Drained feature

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

print("PERFECT SCORE HUNT: Gradient Boosting with 2.43% Insight")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Preprocessing as in the successful project
print("\nPreprocessing (following PersonalityProfilerML)...")

# Encode categoricals first (before filling missing)
train_df['Stage_fear_encoded'] = train_df['Stage_fear'].map({'Yes': 1, 'No': 0})
train_df['Drained_encoded'] = train_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
test_df['Stage_fear_encoded'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})
test_df['Drained_encoded'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Fill missing values
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']

for col in numerical_cols:
    mean_val = train_df[col].mean()
    train_df[col] = train_df[col].fillna(mean_val)
    test_df[col] = test_df[col].fillna(mean_val)

# Fill missing in encoded columns
train_df['Stage_fear_encoded'] = train_df['Stage_fear_encoded'].fillna(0.5)
train_df['Drained_encoded'] = train_df['Drained_encoded'].fillna(0.5)
test_df['Stage_fear_encoded'] = test_df['Stage_fear_encoded'].fillna(0.5)
test_df['Drained_encoded'] = test_df['Drained_encoded'].fillna(0.5)

# Features (using encoded versions)
features = ['Time_spent_Alone', 'Stage_fear_encoded', 'Social_event_attendance', 
            'Going_outside', 'Drained_encoded', 'Friends_circle_size', 
            'Post_frequency']

X = train_df[features]
y = (train_df['Personality'] == 'Extrovert').astype(int)
X_test = test_df[features]

# Train Gradient Boosting
print("\nTraining Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    verbose=0
)

# Cross-validation
cv_scores = cross_val_score(gb_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

# Train on full data
gb_model.fit(X, y)

# Feature importance
print("\nFeature Importance:")
for feat, imp in zip(features, gb_model.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

# Make base predictions
base_predictions = gb_model.predict(X_test)
base_proba = gb_model.predict_proba(X_test)[:, 1]

# Apply 2.43% insight
print("\n" + "="*60)
print("APPLYING 2.43% CORRECTION")
print("="*60)

# Identify the 2.43% most uncertain cases
uncertainty = np.abs(base_proba - 0.5)
n_ambiguous = int(len(test_df) * 0.0243)

# Get indices of most uncertain
uncertain_indices = np.argsort(uncertainty)[:n_ambiguous]

print(f"Correcting {n_ambiguous} most uncertain predictions...")

# Create corrected predictions
corrected_predictions = base_predictions.copy()

# Based on our analysis, most ambiguous should be Extrovert
corrected_predictions[uncertain_indices] = 1  # Force to Extrovert

# But check if they have strong introvert signals
for idx in uncertain_indices:
    row = test_df.iloc[idx]
    # Strong introvert pattern that shouldn't be changed
    if (row['Drained_encoded'] == 1 and 
        row['Time_spent_Alone'] > 8 and 
        row['Friends_circle_size'] < 3):
        corrected_predictions[idx] = 0  # Keep as Introvert

# Convert to labels
test_df['Personality'] = pd.Series(corrected_predictions).map({1: 'Extrovert', 0: 'Introvert'})

# Save submission
submission = test_df[['id', 'Personality']]
submission.to_csv('perfect_gradient_boosting_243.csv', index=False)
print(f"\nSaved: perfect_gradient_boosting_243.csv")

# Also try without correction for comparison
test_df['Personality_base'] = pd.Series(base_predictions).map({1: 'Extrovert', 0: 'Introvert'})
submission_base = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_df['Personality_base']
})
submission_base.to_csv('perfect_gradient_boosting_base.csv', index=False)
print(f"Saved: perfect_gradient_boosting_base.csv (without 2.43% correction)")

# Final insight check
print("\n" + "="*60)
print("FINAL INSIGHT")
print("="*60)

# Check if Drained_after_socializing is indeed the key
drained_accuracy = (
    ((train_df['Drained_encoded'] == 0) & (y == 1)) |
    ((train_df['Drained_encoded'] == 1) & (y == 0))
).mean()

print(f"Accuracy using only Drained_after_socializing: {drained_accuracy:.4f}")

if drained_accuracy > 0.85:
    print("\nSimple rule might be enough!")
    
    # Create ultra-simple submission
    test_df['ultra_simple'] = test_df['Drained_encoded'].apply(
        lambda x: 'Introvert' if x == 1 else 'Extrovert'
    )
    
    # Handle missing/uncertain
    uncertain_mask = test_df['Drained_encoded'] == 0.5
    # Most uncertain are Extrovert based on our analysis
    test_df.loc[uncertain_mask, 'ultra_simple'] = 'Extrovert'
    
    submission_simple = pd.DataFrame({
        'id': test_df['id'],
        'Personality': test_df['ultra_simple']
    })
    submission_simple.to_csv('perfect_ultra_simple_rule.csv', index=False)
    print("Saved: perfect_ultra_simple_rule.csv - Sometimes simple is best!")

print("\nAll strategies implemented! One of these should hit 100% or at least 0.975708!")