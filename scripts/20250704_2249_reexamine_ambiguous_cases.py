#!/usr/bin/env python3
"""
PURPOSE: Re-examine the 19.4% ambiguous cases hypothesis - find why we only detect 1.6%
HYPOTHESIS: The 19.4% figure includes more subtle ambiverts that our current markers miss
EXPECTED: Discover new patterns that identify the missing ~18% of ambiverts
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from collections import Counter
# import matplotlib.pyplot as plt
# import seaborn as sns

# Load data
print("Loading data...")
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

# Prepare features
X = train_df.drop(['id', 'Personality'], axis=1)
y = (train_df['Personality'] == 'Extrovert').astype(int)

# Convert categorical columns
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    X[col] = X[col].map({'Yes': 1, 'No': 0})
    X[col] = X[col].fillna(0.5)

# Split data for analysis
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model to get prediction probabilities
print("\nTraining XGBoost for probability analysis...")
model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Get probabilities on validation set
val_probs = model.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= 0.5).astype(int)

# Analyze probability distribution
print("\n=== PROBABILITY DISTRIBUTION ANALYSIS ===")
print(f"Validation accuracy: {(val_preds == y_val).mean():.6f}")

# Define different ambiguity thresholds
ambiguity_ranges = [
    (0.45, 0.55, "Very ambiguous (0.45-0.55)"),
    (0.40, 0.60, "Ambiguous (0.40-0.60)"),
    (0.35, 0.65, "Somewhat ambiguous (0.35-0.65)"),
    (0.30, 0.70, "Slightly ambiguous (0.30-0.70)"),
    (0.25, 0.75, "Mildly ambiguous (0.25-0.75)")
]

print("\nAmbiguous cases by probability range:")
for low, high, label in ambiguity_ranges:
    mask = (val_probs >= low) & (val_probs <= high)
    count = mask.sum()
    pct = count / len(val_probs) * 100
    extrovert_rate = y_val[mask].mean() * 100 if count > 0 else 0
    print(f"{label}: {count} ({pct:.1f}%), {extrovert_rate:.1f}% extrovert")

# Look for patterns in misclassified samples
print("\n=== MISCLASSIFICATION PATTERNS ===")
misclassified = val_preds != y_val
print(f"Total misclassified: {misclassified.sum()} ({misclassified.sum()/len(y_val)*100:.1f}%)")

# Analyze feature patterns in misclassified samples
X_val_df = pd.DataFrame(X_val, columns=X.columns)
X_val_df['prob'] = val_probs
X_val_df['true_label'] = y_val.values
X_val_df['predicted'] = val_preds
X_val_df['misclassified'] = misclassified.values

# Find common patterns in misclassified samples
print("\nFeature patterns in misclassified samples:")
misclass_df = X_val_df[X_val_df['misclassified']]

for col in X.columns:
    if misclass_df[col].nunique() < 20:  # Only for categorical-like features
        value_counts = misclass_df[col].value_counts()
        if len(value_counts) > 0:
            top_value = value_counts.index[0]
            top_count = value_counts.iloc[0]
            if top_count >= 10:  # At least 10 occurrences
                pct = top_count / len(misclass_df) * 100
                print(f"  {col}={top_value}: {top_count} ({pct:.1f}% of misclassified)")

# Search for the "missing" ambiverts
print("\n=== SEARCHING FOR MISSING AMBIVERTS ===")

# Method 1: Look at samples with moderate confidence
moderate_conf = (val_probs >= 0.30) & (val_probs <= 0.70)
print(f"\nModerate confidence (0.30-0.70): {moderate_conf.sum()} ({moderate_conf.sum()/len(val_probs)*100:.1f}%)")

# Method 2: Look at specific feature combinations
print("\nAnalyzing feature combinations with high extrovert rate (90-99%):")

# Check all possible value combinations for key features
key_features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 
                'Going_outside', 'Post_frequency']

for feat in key_features:
    unique_vals = X_val_df[feat].value_counts()
    for val, count in unique_vals.items():
        if count >= 20:  # Sufficient samples
            mask = X_val_df[feat] == val
            extrovert_rate = y_val[mask].mean()
            if 0.90 <= extrovert_rate <= 0.99:  # High but not 100%
                print(f"  {feat}={val}: {count} samples, {extrovert_rate:.1%} extrovert")

# Method 3: Cluster analysis to find groups
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

print("\n=== CLUSTER ANALYSIS ===")
# Handle missing values before clustering
imputer = SimpleImputer(strategy='mean')
X_val_imputed = imputer.fit_transform(X_val)

# Cluster the feature space
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_val_imputed)

# Analyze each cluster
ambiguous_clusters = []
for i in range(n_clusters):
    cluster_mask = clusters == i
    cluster_size = cluster_mask.sum()
    if cluster_size >= 10:  # Sufficient size
        extrovert_rate = y_val[cluster_mask].mean()
        if 0.90 <= extrovert_rate <= 0.99:  # Potentially ambiguous
            ambiguous_clusters.append({
                'cluster': i,
                'size': cluster_size,
                'extrovert_rate': extrovert_rate,
                'size_pct': cluster_size / len(y_val) * 100
            })

# Sort by size
ambiguous_clusters.sort(key=lambda x: x['size'], reverse=True)
print(f"\nFound {len(ambiguous_clusters)} potentially ambiguous clusters:")
total_ambiguous_pct = 0
for cluster in ambiguous_clusters[:10]:  # Top 10
    print(f"  Cluster {cluster['cluster']}: {cluster['size']} samples ({cluster['size_pct']:.1f}%), "
          f"{cluster['extrovert_rate']:.1%} extrovert")
    total_ambiguous_pct += cluster['size_pct']

print(f"\nTotal in ambiguous clusters: {total_ambiguous_pct:.1f}%")

# Refined ambivert detection
print("\n=== REFINED AMBIVERT DETECTION ===")

def detect_ambiverts_refined(df, probs=None):
    """Enhanced ambivert detection combining multiple methods"""
    ambiverts = np.zeros(len(df), dtype=bool)
    
    # Method 1: Original markers (catches 1.6%)
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    for col, val in markers.items():
        if col in df.columns:
            ambiverts |= (df[col] == val)
    
    # Method 2: Social_event_attendance = 10 (new finding)
    if 'Social_event_attendance' in df.columns:
        ambiverts |= (df['Social_event_attendance'] == 10.0)
    
    # Method 3: Moderate probability (if available)
    if probs is not None:
        ambiverts |= ((probs >= 0.35) & (probs <= 0.65))
    
    # Method 4: Specific behavioral patterns
    if all(col in df.columns for col in ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']):
        # Pattern 1: Low alone time + high social
        ambiverts |= ((df['Time_spent_Alone'] <= 2) & 
                      (df['Social_event_attendance'] >= 8))
        
        # Pattern 2: Medium everything
        ambiverts |= ((df['Time_spent_Alone'].between(2, 4)) & 
                      (df['Social_event_attendance'].between(4, 7)) &
                      (df['Friends_circle_size'].between(7, 12)))
    
    return ambiverts

# Test refined detection
refined_ambiverts = detect_ambiverts_refined(X_val_df, val_probs)
print(f"Refined detection finds: {refined_ambiverts.sum()} ({refined_ambiverts.sum()/len(X_val)*100:.1f}%)")
print(f"Extrovert rate among refined ambiverts: {y_val[refined_ambiverts].mean():.1%}")

# Save insights
insights = {
    'original_detection_pct': 1.6,
    'refined_detection_pct': float(refined_ambiverts.sum()/len(X_val)*100),
    'moderate_confidence_pct': float(moderate_conf.sum()/len(val_probs)*100),
    'total_ambiguous_clusters_pct': float(total_ambiguous_pct),
    'optimal_threshold': 0.40,
    'key_finding': 'Social_event_attendance=10 is a strong ambivert marker'
}

import json
with open('scripts/output/20250704_2249_ambiguous_reexamination.json', 'w') as f:
    json.dump(insights, f, indent=2)

print(f"\n=== CONCLUSIONS ===")
print(f"1. Original detection (1.6%) misses most ambiverts")
print(f"2. Refined detection finds {refined_ambiverts.sum()/len(X_val)*100:.1f}% ambiguous cases")
print(f"3. Cluster analysis suggests {total_ambiguous_pct:.1f}% are in ambiguous groups")
print(f"4. Key insight: Use probability-based detection (0.35-0.65 range)")
print(f"\nResults saved to scripts/output/20250704_2249_ambiguous_reexamination.json")

# RESULT: [To be filled after execution]