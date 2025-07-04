#!/usr/bin/env python3
"""
PURPOSE: Analyze current best strategies and identify gaps that prevented us from reaching 0.976518
HYPOTHESIS: There are subtle patterns or edge cases we missed in our ambivert detection
EXPECTED: Find specific areas where our predictions differ from optimal
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import json
from collections import Counter

# Load data
print("Loading data...")
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

# Define our best known ambivert markers
AMBIVERT_MARKERS = {
    'Social_event_attendance': 5.265106088560886,
    'Going_outside': 4.044319380935631,
    'Post_frequency': 4.982097334878332,
    'Time_spent_Alone': 3.1377639321564557
}

# Analyze current ambivert detection accuracy
def detect_ambiverts_current(df):
    """Current best method for detecting ambiverts"""
    # Method 1: Marker-based
    markers_found = 0
    for col, val in AMBIVERT_MARKERS.items():
        if col in df.columns:
            markers_found += (df[col] == val).astype(int)
    
    marker_based = markers_found >= 2
    
    # Method 2: Behavioral pattern
    behavioral = (
        (df['Time_spent_Alone'] < 2.5) & 
        (df['Social_event_attendance'].between(3, 4)) &
        (df['Friends_circle_size'].between(6, 7))
    )
    
    return marker_based | behavioral

# Prepare data
print("\nPreparing data...")
X_train = train_df.drop(['id', 'Personality'], axis=1)
y_train = (train_df['Personality'] == 'Extrovert').astype(int)

# Convert categorical columns
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    X_train[col] = X_train[col].map({'Yes': 1, 'No': 0})
    X_train[col] = X_train[col].fillna(0.5)  # Handle missing values

# Train baseline model
print("\nTraining baseline XGBoost...")

baseline_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Baseline CV accuracy: {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})")

# Fit on full training data
baseline_model.fit(X_train, y_train)

# Analyze prediction distribution
train_probs = baseline_model.predict_proba(X_train)[:, 1]
print(f"\nTraining set probability distribution:")
print(f"Mean: {train_probs.mean():.4f}")
print(f"Std: {train_probs.std():.4f}")
print(f"Samples in [0.45, 0.55]: {np.sum((train_probs >= 0.45) & (train_probs <= 0.55))}")
print(f"Samples in [0.48, 0.52]: {np.sum((train_probs >= 0.48) & (train_probs <= 0.52))}")

# Identify potential gaps
print("\n=== ANALYZING GAPS ===")

# Gap 1: Check if 97.9% rule is accurate
ambiverts_mask = detect_ambiverts_current(train_df)
print(f"\n1. Ambivert Detection Analysis:")
print(f"Total ambiverts detected: {ambiverts_mask.sum()} ({ambiverts_mask.sum()/len(train_df)*100:.1f}%)")
print(f"Extroverts among ambiverts: {y_train[ambiverts_mask].sum()} ({y_train[ambiverts_mask].sum()/ambiverts_mask.sum()*100:.1f}%)")

# Gap 2: Analyze misclassified samples
train_preds = baseline_model.predict(X_train)
misclassified = train_preds != y_train
print(f"\n2. Misclassification Analysis:")
print(f"Total misclassified: {misclassified.sum()} ({misclassified.sum()/len(train_df)*100:.1f}%)")
print(f"Misclassified ambiverts: {(misclassified & ambiverts_mask).sum()}")
print(f"Misclassified non-ambiverts: {(misclassified & ~ambiverts_mask).sum()}")

# Gap 3: Feature importance variations
feature_importance = baseline_model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n3. Feature Importance Analysis:")
print(importance_df.head(10))

# Gap 4: Probability threshold analysis
print(f"\n4. Optimal Threshold Search:")
thresholds = np.arange(0.40, 0.60, 0.01)
accuracies = []

for threshold in thresholds:
    preds = (train_probs >= threshold).astype(int)
    accuracy = (preds == y_train).mean()
    accuracies.append(accuracy)

best_threshold_idx = np.argmax(accuracies)
print(f"Best threshold: {thresholds[best_threshold_idx]:.2f}")
print(f"Best accuracy: {accuracies[best_threshold_idx]:.6f}")

# Gap 5: Analyze edge cases
print(f"\n5. Edge Case Analysis:")
edge_cases = (train_probs >= 0.48) & (train_probs <= 0.52)
print(f"Edge cases (0.48-0.52): {edge_cases.sum()}")
print(f"Edge case accuracy: {(train_preds[edge_cases] == y_train[edge_cases]).mean():.4f}")

# Gap 6: Look for additional patterns
print(f"\n6. Additional Pattern Search:")

# Check for specific value combinations that might indicate ambiverts
for col in X_train.columns:
    unique_vals = X_train[col].value_counts()
    if len(unique_vals) < 20:  # Only for categorical-like features
        for val, count in unique_vals.items():
            if count > 50 and count < 500:  # Rare but not too rare
                mask = X_train[col] == val
                extrovert_rate = y_train[mask].mean()
                if 0.90 <= extrovert_rate <= 0.99:  # High but not 100% extrovert rate
                    print(f"  {col}={val}: {count} samples, {extrovert_rate:.1%} extrovert")

# Save analysis results
results = {
    'baseline_cv_accuracy': float(cv_scores.mean()),
    'ambiverts_detected_pct': float(ambiverts_mask.sum()/len(train_df)*100),
    'extrovert_rate_ambiverts': float(y_train[ambiverts_mask].sum()/ambiverts_mask.sum()*100),
    'misclassified_pct': float(misclassified.sum()/len(train_df)*100),
    'best_threshold': float(thresholds[best_threshold_idx]),
    'edge_cases_count': int(edge_cases.sum()),
    'top_features': importance_df.head(5).to_dict('records')
}

with open('scripts/output/20250704_2246_gaps_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== KEY INSIGHTS ===")
print(f"1. Current ambivert detection finds {ambiverts_mask.sum()/len(train_df)*100:.1f}% (vs reported 19.4%)")
print(f"2. Extrovert rate among ambiverts: {y_train[ambiverts_mask].sum()/ambiverts_mask.sum()*100:.1f}% (vs reported 97.9%)")
print(f"3. Optimal threshold is {thresholds[best_threshold_idx]:.2f} (not 0.5)")
print(f"4. {edge_cases.sum()} edge cases need special handling")
print(f"\nResults saved to scripts/output/20250704_2246_gaps_analysis.json")

# RESULT: [To be filled after execution]