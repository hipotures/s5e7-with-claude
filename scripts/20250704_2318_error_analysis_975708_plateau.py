#!/usr/bin/env python3
"""
PURPOSE: Deep analysis of why 240+ people hit exactly 0.975708 and how to break it
HYPOTHESIS: There's a specific subset of samples everyone misclassifies the same way
EXPECTED: Identify the exact samples causing the plateau and find a pattern
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
import json
from collections import Counter

# Load data
print("Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Prepare features
def prepare_features(df, is_train=True):
    if is_train:
        X = df.drop(['id', 'Personality'], axis=1).copy()
        y = (df['Personality'] == 'Extrovert').astype(int)
    else:
        X = df.drop(['id'], axis=1).copy()
        y = None
    
    # Convert categorical
    for col in ['Stage_fear', 'Drained_after_socializing']:
        X[col] = X[col].map({'Yes': 1, 'No': 0})
        X[col] = X[col].fillna(0.5)
    
    return X, y

X_train, y_train = prepare_features(train_df, is_train=True)
X_test, _ = prepare_features(test_df, is_train=False)

print(f"Dataset size: {len(X_train)} samples")
print(f"Features: {X_train.shape[1]}")

# Calculate theoretical accuracy limit
print("\n=== THEORETICAL ANALYSIS ===")
# If 0.975708 is the ceiling, then error rate is 1 - 0.975708 = 0.024292
error_rate = 1 - 0.975708
misclassified_count = int(error_rate * len(X_train))
print(f"Error rate at 0.975708: {error_rate:.6f} ({error_rate*100:.3f}%)")
print(f"Expected misclassified samples: {misclassified_count} out of {len(X_train)}")

# Train multiple models to find common errors
print("\n=== TRAINING MULTIPLE MODELS ===")

models = {
    'xgboost': xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42),
    'lightgbm': lgb.LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, verbose=-1),
    'catboost': CatBoostClassifier(iterations=200, depth=6, random_seed=42, verbose=False)
}

# Store predictions from each model
all_predictions = {}
all_probabilities = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Get predictions
    predictions = model.predict(X_train)
    probabilities = model.predict_proba(X_train)[:, 1]
    
    all_predictions[name] = predictions
    all_probabilities[name] = probabilities
    
    # Calculate accuracy
    accuracy = (predictions == y_train).mean()
    print(f"{name} accuracy: {accuracy:.6f}")

# Find samples that all models get wrong
print("\n=== ANALYZING COMMON ERRORS ===")

# Convert predictions to DataFrame for easier analysis
pred_df = pd.DataFrame(all_predictions)
pred_df['true_label'] = y_train.values
pred_df['id'] = train_df['id'].values

# Find samples where ALL models are wrong
all_wrong = pred_df[
    (pred_df['xgboost'] != pred_df['true_label']) &
    (pred_df['lightgbm'] != pred_df['true_label']) &
    (pred_df['catboost'] != pred_df['true_label'])
]

print(f"\nSamples misclassified by ALL models: {len(all_wrong)}")
print(f"Percentage: {len(all_wrong)/len(train_df)*100:.3f}%")

# Analyze characteristics of always-wrong samples
if len(all_wrong) > 0:
    wrong_indices = all_wrong.index
    
    print("\n=== CHARACTERISTICS OF ALWAYS-WRONG SAMPLES ===")
    
    # Get features of wrong samples
    wrong_features = X_train.iloc[wrong_indices]
    wrong_labels = y_train.iloc[wrong_indices]
    
    # Analyze each feature
    print("\nFeature distributions in always-wrong samples:")
    for col in X_train.columns:
        wrong_mean = wrong_features[col].mean()
        overall_mean = X_train[col].mean()
        if abs(wrong_mean - overall_mean) > 0.1:
            print(f"{col}: wrong={wrong_mean:.3f}, overall={overall_mean:.3f}")
    
    # Check for specific patterns
    print("\nLabel distribution in always-wrong:")
    print(f"Extroverts: {wrong_labels.sum()} ({wrong_labels.mean():.1%})")
    print(f"Introverts: {len(wrong_labels) - wrong_labels.sum()} ({1-wrong_labels.mean():.1%})")

# Analyze probability distributions for misclassified samples
print("\n=== PROBABILITY ANALYSIS ===")

# Get average probability across models
avg_probs = np.mean([all_probabilities[name] for name in models.keys()], axis=0)

# Find samples with extreme disagreement
prob_std = np.std([all_probabilities[name] for name in models.keys()], axis=0)
high_disagreement = prob_std > 0.1

print(f"\nSamples with high model disagreement: {high_disagreement.sum()}")

# Analyze probability ranges
print("\nMisclassification by probability range:")
prob_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]

for low, high in prob_ranges:
    mask = (avg_probs >= low) & (avg_probs < high)
    if mask.sum() > 0:
        errors = (pred_df['xgboost'][mask] != pred_df['true_label'][mask]).sum()
        error_rate = errors / mask.sum()
        print(f"[{low:.1f}, {high:.1f}): {mask.sum()} samples, {error_rate:.1%} error rate")

# Look for "impossible" samples
print("\n=== SEARCHING FOR IMPOSSIBLE SAMPLES ===")

# Samples that look like clear extroverts but are introverts (or vice versa)
clear_extrovert_features = (
    (X_train['Time_spent_Alone'] <= 1) & 
    (X_train['Drained_after_socializing'] == 0) &
    (X_train['Stage_fear'] == 0) &
    (X_train['Social_event_attendance'] >= 8)
)

mislabeled_extroverts = clear_extrovert_features & (y_train == 0)
print(f"Clear extrovert features but labeled introvert: {mislabeled_extroverts.sum()}")

clear_introvert_features = (
    (X_train['Time_spent_Alone'] >= 8) & 
    (X_train['Drained_after_socializing'] == 1) &
    (X_train['Stage_fear'] == 1) &
    (X_train['Social_event_attendance'] <= 2)
)

mislabeled_introverts = clear_introvert_features & (y_train == 1)
print(f"Clear introvert features but labeled extrovert: {mislabeled_introverts.sum()}")

# Missing value analysis
print("\n=== MISSING VALUE PATTERNS ===")

missing_mask = train_df[['Stage_fear', 'Drained_after_socializing']].isna().any(axis=1)
print(f"Samples with missing values: {missing_mask.sum()}")

if missing_mask.sum() > 0:
    # Error rate for samples with missing values
    missing_errors = (pred_df['xgboost'][missing_mask] != pred_df['true_label'][missing_mask]).sum()
    missing_error_rate = missing_errors / missing_mask.sum()
    print(f"Error rate for missing: {missing_error_rate:.1%}")
    
    # Error rate for complete samples
    complete_errors = (pred_df['xgboost'][~missing_mask] != pred_df['true_label'][~missing_mask]).sum()
    complete_error_rate = complete_errors / (~missing_mask).sum()
    print(f"Error rate for complete: {complete_error_rate:.1%}")

# Find the exact boundary samples
print("\n=== BOUNDARY SAMPLE ANALYSIS ===")

# Samples very close to 0.5 probability
boundary_samples = np.abs(avg_probs - 0.5) < 0.05
print(f"Samples near 0.5 probability: {boundary_samples.sum()}")

if boundary_samples.sum() > 0:
    boundary_error_rate = (pred_df['xgboost'][boundary_samples] != pred_df['true_label'][boundary_samples]).mean()
    print(f"Error rate at boundary: {boundary_error_rate:.1%}")

# Create a detailed error report
error_analysis = {
    'total_samples': len(X_train),
    'theoretical_errors_at_975708': misclassified_count,
    'samples_all_models_wrong': len(all_wrong),
    'high_disagreement_samples': int(high_disagreement.sum()),
    'boundary_samples': int(boundary_samples.sum()),
    'mislabeled_patterns': {
        'clear_extrovert_labeled_introvert': int(mislabeled_extroverts.sum()),
        'clear_introvert_labeled_extrovert': int(mislabeled_introverts.sum())
    },
    'error_by_probability': {}
}

for low, high in prob_ranges:
    mask = (avg_probs >= low) & (avg_probs < high)
    if mask.sum() > 0:
        errors = (pred_df['xgboost'][mask] != pred_df['true_label'][mask]).sum()
        error_analysis['error_by_probability'][f'{low}-{high}'] = {
            'count': int(mask.sum()),
            'errors': int(errors),
            'error_rate': float(errors / mask.sum())
        }

# Save detailed error analysis
with open('output/20250704_2318_error_analysis.json', 'w') as f:
    json.dump(error_analysis, f, indent=2)

# If we found always-wrong samples, save them for inspection
if len(all_wrong) > 0:
    all_wrong_detailed = train_df.iloc[wrong_indices].copy()
    all_wrong_detailed['avg_probability'] = avg_probs[wrong_indices]
    all_wrong_detailed['prob_std'] = prob_std[wrong_indices]
    all_wrong_detailed.to_csv('output/20250704_2318_always_wrong_samples.csv', index=False)
    print(f"\nSaved {len(all_wrong)} always-wrong samples to CSV")

print("\n=== BREAKTHROUGH INSIGHTS ===")
print(f"1. The 0.975708 ceiling means ~{misclassified_count} samples cannot be correctly classified")
print(f"2. {len(all_wrong)} samples are consistently misclassified by all models")
print(f"3. High disagreement on {high_disagreement.sum()} samples suggests uncertainty")
print(f"4. {mislabeled_extroverts.sum() + mislabeled_introverts.sum()} samples appear mislabeled")
print("\nTo break 0.975708, we need to correctly classify some of these 'impossible' samples")
print("Results saved to output/20250704_2318_error_analysis.json")

# RESULT: [To be filled after execution]