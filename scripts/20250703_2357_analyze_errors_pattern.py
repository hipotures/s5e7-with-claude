#!/usr/bin/env python3
"""
PURPOSE: Analyze the ~2.43% misclassified cases to find patterns and understand why 
these specific samples are consistently misclassified

HYPOTHESIS: The ~2.43% error rate is not random but represents a specific subset of 
samples that may have ambiguous characteristics or represent a third personality type

EXPECTED: Find that misclassified cases cluster around decision boundaries or show
contradictory feature patterns suggesting ambivert characteristics

RESULT: Discovered that misclassified cases have lower prediction confidence, cluster
around probability 0.5, and show mixed introvert/extrovert patterns, supporting the
hypothesis of a hidden third class (ambiverts)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

print("ANALYZING MISCLASSIFIED CASES (~2.43%)")
print("="*60)

# Load and prepare data (same as before)
train_df = pd.read_csv("../../train.csv")

# Features
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency']

numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

# Preprocessing
for col in numerical_cols:
    mean_val = train_df[col].mean()
    train_df[col].fillna(mean_val, inplace=True)

for col in categorical_cols:
    train_df[col].fillna('Missing', inplace=True)

# Encode
mapping_yes_no = {'Yes': 1, 'No': 0}
for col in categorical_cols:
    train_df[col] = train_df[col].map(mapping_yes_no)

mapping_personality = {'Extrovert': 1, 'Introvert': 0}
train_df['Personality'] = train_df['Personality'].map(mapping_personality)

X = train_df[features]
y = train_df['Personality']

# Train model with exact parameters
print("Training XGBoost with exact parameters...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    random_state=42,
    n_estimators=1000,
    learning_rate=0.006358,
    max_depth=8,
    subsample=0.8854,
    colsample_bytree=0.6,
    reg_lambda=0.8295,
    reg_alpha=5.5149,
    gamma=0.0395,
    min_child_weight=2,
    use_label_encoder=False,
    verbosity=0
)

# Split for analysis
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
xgb_model.fit(X_train, y_train)

# Get predictions and probabilities
y_pred = xgb_model.predict(X_val)
y_proba = xgb_model.predict_proba(X_val)[:, 1]

# Find misclassified cases
misclassified_mask = y_val != y_pred
correctly_classified_mask = ~misclassified_mask

print(f"\nValidation set size: {len(y_val)}")
print(f"Misclassified: {misclassified_mask.sum()} ({misclassified_mask.sum()/len(y_val)*100:.2f}%)")
print(f"Correctly classified: {correctly_classified_mask.sum()} ({correctly_classified_mask.sum()/len(y_val)*100:.2f}%)")

# Analyze misclassified cases
X_val_df = X_val.copy()
X_val_df['True_Label'] = y_val
X_val_df['Predicted_Label'] = y_pred
X_val_df['Probability'] = y_proba
X_val_df['Misclassified'] = misclassified_mask

misclassified_df = X_val_df[misclassified_mask].copy()
correct_df = X_val_df[correctly_classified_mask].copy()

print("\n" + "="*60)
print("MISCLASSIFIED CASES ANALYSIS")
print("="*60)

# Type of errors
false_positives = misclassified_df[(misclassified_df['True_Label'] == 0) & (misclassified_df['Predicted_Label'] == 1)]
false_negatives = misclassified_df[(misclassified_df['True_Label'] == 1) & (misclassified_df['Predicted_Label'] == 0)]

print(f"\nFalse Positives (Introvert predicted as Extrovert): {len(false_positives)}")
print(f"False Negatives (Extrovert predicted as Introvert): {len(false_negatives)}")

# Compare feature distributions
print("\n" + "-"*60)
print("FEATURE COMPARISON: Misclassified vs Correct")
print("-"*60)

for feature in features:
    if feature in numerical_cols:
        mis_mean = misclassified_df[feature].mean()
        cor_mean = correct_df[feature].mean()
        print(f"\n{feature}:")
        print(f"  Misclassified mean: {mis_mean:.3f}")
        print(f"  Correct mean: {cor_mean:.3f}")
        print(f"  Difference: {mis_mean - cor_mean:.3f}")

# Check for patterns in categorical features
print("\n" + "-"*60)
print("CATEGORICAL PATTERNS IN MISCLASSIFIED")
print("-"*60)

for feature in categorical_cols:
    print(f"\n{feature} distribution:")
    print("Misclassified:")
    print(misclassified_df[feature].value_counts(normalize=True).round(3))
    print("Correct:")
    print(correct_df[feature].value_counts(normalize=True).round(3))

# Probability analysis
print("\n" + "-"*60)
print("PREDICTION CONFIDENCE ANALYSIS")
print("-"*60)

print(f"\nMisclassified cases - probability stats:")
print(f"  Mean probability: {misclassified_df['Probability'].mean():.3f}")
print(f"  Median probability: {misclassified_df['Probability'].median():.3f}")
print(f"  Std probability: {misclassified_df['Probability'].std():.3f}")

print(f"\nCorrect cases - probability stats:")
print(f"  Mean probability: {correct_df['Probability'].mean():.3f}")
print(f"  Median probability: {correct_df['Probability'].median():.3f}")
print(f"  Std probability: {correct_df['Probability'].std():.3f}")

# Find borderline cases
borderline_threshold = 0.1
borderline_cases = X_val_df[np.abs(X_val_df['Probability'] - 0.5) < borderline_threshold]
print(f"\n Borderline cases (prob between 0.4-0.6): {len(borderline_cases)}")
print(f"  Of which misclassified: {borderline_cases['Misclassified'].sum()}")

# Visualizations
plt.figure(figsize=(15, 10))

# 1. Probability distribution
plt.subplot(2, 3, 1)
plt.hist(misclassified_df['Probability'], bins=30, alpha=0.7, label='Misclassified', color='red')
plt.hist(correct_df['Probability'], bins=30, alpha=0.7, label='Correct', color='green')
plt.xlabel('Prediction Probability')
plt.ylabel('Count')
plt.title('Prediction Probability Distribution')
plt.legend()

# 2. Feature importance for errors
plt.subplot(2, 3, 2)
feature_diffs = []
for feature in numerical_cols:
    diff = abs(misclassified_df[feature].mean() - correct_df[feature].mean())
    feature_diffs.append((feature, diff))
feature_diffs.sort(key=lambda x: x[1], reverse=True)
features_sorted, diffs_sorted = zip(*feature_diffs)
plt.barh(features_sorted, diffs_sorted)
plt.xlabel('Mean Absolute Difference')
plt.title('Feature Differences: Misclassified vs Correct')

# 3. Scatter plot of top 2 features
plt.subplot(2, 3, 3)
plt.scatter(correct_df['Drained_after_socializing'], correct_df['Stage_fear'], 
           alpha=0.5, label='Correct', color='green', s=10)
plt.scatter(misclassified_df['Drained_after_socializing'], misclassified_df['Stage_fear'], 
           alpha=0.8, label='Misclassified', color='red', s=30)
plt.xlabel('Drained_after_socializing')
plt.ylabel('Stage_fear')
plt.title('Top 2 Features: Correct vs Misclassified')
plt.legend()

# 4. Time spent alone distribution
plt.subplot(2, 3, 4)
plt.hist(misclassified_df['Time_spent_Alone'], bins=20, alpha=0.7, label='Misclassified', color='red', density=True)
plt.hist(correct_df['Time_spent_Alone'], bins=20, alpha=0.7, label='Correct', color='green', density=True)
plt.xlabel('Time_spent_Alone')
plt.ylabel('Density')
plt.title('Time Spent Alone Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('misclassified_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Check for specific patterns
print("\n" + "="*60)
print("LOOKING FOR SPECIFIC PATTERNS")
print("="*60)

# Pattern 1: Cases where all social indicators suggest one class but labeled as other
introvert_pattern = (X_val_df['Drained_after_socializing'] == 1) & \
                   (X_val_df['Time_spent_Alone'] > X_val_df['Time_spent_Alone'].median()) & \
                   (X_val_df['Social_event_attendance'] < X_val_df['Social_event_attendance'].median())

extrovert_pattern = (X_val_df['Drained_after_socializing'] == 0) & \
                   (X_val_df['Time_spent_Alone'] < X_val_df['Time_spent_Alone'].median()) & \
                   (X_val_df['Social_event_attendance'] > X_val_df['Social_event_attendance'].median())

print("\nCases with strong introvert pattern:")
print(f"  Total: {introvert_pattern.sum()}")
print(f"  Labeled as Extrovert: {(introvert_pattern & (X_val_df['True_Label'] == 1)).sum()}")
print(f"  Misclassified among these: {(introvert_pattern & X_val_df['Misclassified']).sum()}")

print("\nCases with strong extrovert pattern:")
print(f"  Total: {extrovert_pattern.sum()}")
print(f"  Labeled as Introvert: {(extrovert_pattern & (X_val_df['True_Label'] == 0)).sum()}")
print(f"  Misclassified among these: {(extrovert_pattern & X_val_df['Misclassified']).sum()}")

# Save misclassified cases for manual inspection
misclassified_df.to_csv('misclassified_cases.csv', index=False)
print("\nSaved misclassified cases to 'misclassified_cases.csv' for manual inspection")

# Check if there's a pattern in missing values
print("\n" + "-"*60)
print("MISSING VALUE PATTERNS")
print("-"*60)

# Original data with missing
train_df_original = pd.read_csv("../../train.csv")
train_df_original['has_missing'] = train_df_original[features].isnull().any(axis=1)

print(f"Samples with missing values: {train_df_original['has_missing'].sum()} ({train_df_original['has_missing'].sum()/len(train_df_original)*100:.1f}%)")

# Map personality for comparison
train_df_original['Personality_binary'] = train_df_original['Personality'].map(mapping_personality)

# Check if missing values correlate with personality
missing_by_personality = train_df_original.groupby('Personality')['has_missing'].mean()
print("\nProportion with missing values by personality:")
print(missing_by_personality)

print("\n" + "="*60)
print("CONCLUSIONS")
print("="*60)
print("1. Check if misclassified cases have specific patterns")
print("2. Look at borderline cases (probability near 0.5)")
print("3. Analyze if certain feature combinations are always misclassified")
print("4. Consider if these ~2.43% are actually mislabeled in the synthetic data")
print("5. Or if they represent a different underlying pattern")