#!/usr/bin/env python3
"""
PURPOSE: Check which features RFECV (Recursive Feature Elimination with Cross-Validation) selected
HYPOTHESIS: RFECV should identify the most important features for personality classification
EXPECTED: Get a subset of features that maximize model performance
RESULT: Identified key features selected by RFECV for optimal model performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV

# Load and prepare data
print("Loading data...")
df = pd.read_csv("../../train.csv")

# Prepare data (same as in feature_selection_comprehensive.py)
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object' and col not in ['Personality', 'id']:
        if set(df_encoded[col].dropna().unique()) <= {'Yes', 'No'}:
            df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})

# Define column types
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 
               'Going_outside', 'Friends_circle_size', 'Post_frequency']

# Simple feature generation (subset of what's in the full script)
print("\nGenerating features...")
df_features = df_encoded.copy()

# Add some basic features
for col in numeric_cols:
    df_features[f'{col}_squared'] = df[col] ** 2
    df_features[f'{col}_log1p'] = np.log1p(df[col])

# Add aggregations
df_features['numeric_mean'] = df[numeric_cols].mean(axis=1)
df_features['numeric_std'] = df[numeric_cols].std(axis=1)

# Add domain-specific
df_features['introvert_score'] = (
    df['Time_spent_Alone'] * 2 - 
    df['Social_event_attendance'] - 
    df['Friends_circle_size'] / 2
)

# Prepare X and y
X = df_features.drop(columns=['Personality', 'id'])
y = LabelEncoder().fit_transform(df['Personality'])

print(f"Total features: {X.shape[1]}")

# Run RFECV
print("\nRunning RFECV...")
estimator = xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)
selector = RFECV(estimator, step=5, cv=3, scoring='accuracy', n_jobs=-1, min_features_to_select=5)
selector.fit(X, y)

# Get selected features
selected_features = X.columns[selector.support_].tolist()
print(f"\nRFECV selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feat}")

# Check if Drained_after_socializing is in the list
if 'Drained_after_socializing' in selected_features:
    print("\n✅ 'Drained_after_socializing' is included (the dominant feature)")
else:
    print("\n❌ 'Drained_after_socializing' is NOT included (surprising!)")

# Train model with selected features and check importance
print("\nTraining model with selected features...")
model = xgb.XGBClassifier(n_estimators=300, random_state=42)
model.fit(X[selected_features], y)

# Get feature importance
importance = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importance within selected features:")
for _, row in importance.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"{row['feature']:30} {row['importance']:.4f} {bar}")