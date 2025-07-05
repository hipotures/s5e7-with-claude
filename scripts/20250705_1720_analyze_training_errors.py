#!/usr/bin/env python3
"""
PURPOSE: Analyze potential labeling errors in training data
HYPOTHESIS: Some training labels might be incorrect, causing the 0.975708 ceiling
EXPECTED: Find suspicious cases where features strongly contradict labels
RESULT: [To be filled]
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("="*80)
print("ANALYZING POTENTIAL LABELING ERRORS IN TRAINING DATA")
print("="*80)

# Load data
train_df = pd.read_csv("../../train.csv")
print(f"\nTotal training samples: {len(train_df)}")

# Encode features
train_encoded = train_df.copy()
train_encoded['Stage_fear_enc'] = train_encoded['Stage_fear'].map({'Yes': 1, 'No': 0})
train_encoded['Drained_enc'] = train_encoded['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
train_encoded['is_introvert'] = (train_encoded['Personality'] == 'Introvert').astype(int)

# Define numerical features
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 
                     'Going_outside', 'Friends_circle_size', 'Post_frequency']

print("\n" + "="*60)
print("1. EXTREME VALUE ANALYSIS")
print("="*60)

# Check for out-of-range values
print("\nChecking for values outside expected ranges:")
print("-" * 40)

# Time_spent_Alone should be 0-10
alone_outliers = train_encoded[train_encoded['Time_spent_Alone'] > 10]
print(f"Time_spent_Alone > 10: {len(alone_outliers)} cases")
if len(alone_outliers) > 0:
    print(f"  Values: {sorted(alone_outliers['Time_spent_Alone'].unique())}")
    print(f"  Personalities: {alone_outliers['Personality'].value_counts().to_dict()}")

# Check other features for anomalies
for feature in numerical_features[1:]:
    outliers = train_encoded[train_encoded[feature] > 10]
    if len(outliers) > 0:
        print(f"\n{feature} > 10: {len(outliers)} cases")
        print(f"  Max value: {outliers[feature].max()}")

print("\n" + "="*60)
print("2. CONTRADICTORY PATTERNS")
print("="*60)

# Pattern 1: Extreme introverts labeled as extroverts
extreme_intro_as_extro = train_encoded[
    (train_encoded['Personality'] == 'Extrovert') &
    (train_encoded['Time_spent_Alone'] >= 8) &
    (train_encoded['Social_event_attendance'] <= 2) &
    (train_encoded['Friends_circle_size'] <= 5)
]
print(f"\nExtreme introverts labeled as Extrovert: {len(extreme_intro_as_extro)}")
if len(extreme_intro_as_extro) > 0:
    print("Examples:")
    for idx, row in extreme_intro_as_extro.head(5).iterrows():
        print(f"  ID {row['id']}: Alone={row['Time_spent_Alone']}, Social={row['Social_event_attendance']}, "
              f"Friends={row['Friends_circle_size']}, Drained={row['Drained_after_socializing']}")

# Pattern 2: Extreme extroverts labeled as introverts
extreme_extro_as_intro = train_encoded[
    (train_encoded['Personality'] == 'Introvert') &
    (train_encoded['Time_spent_Alone'] <= 2) &
    (train_encoded['Social_event_attendance'] >= 8) &
    (train_encoded['Friends_circle_size'] >= 15) &
    (train_encoded['Drained_enc'] == 0)
]
print(f"\nExtreme extroverts labeled as Introvert: {len(extreme_extro_as_intro)}")
if len(extreme_extro_as_intro) > 0:
    print("Examples:")
    for idx, row in extreme_extro_as_intro.head(5).iterrows():
        print(f"  ID {row['id']}: Alone={row['Time_spent_Alone']}, Social={row['Social_event_attendance']}, "
              f"Friends={row['Friends_circle_size']}, Post={row['Post_frequency']}")

# Pattern 3: Psychological contradictions
psych_contradictions = train_encoded[
    ((train_encoded['Personality'] == 'Introvert') & 
     (train_encoded['Drained_enc'] == 0) & 
     (train_encoded['Stage_fear_enc'] == 0) &
     (train_encoded['Time_spent_Alone'] <= 3)) |
    ((train_encoded['Personality'] == 'Extrovert') & 
     (train_encoded['Drained_enc'] == 1) & 
     (train_encoded['Stage_fear_enc'] == 1) &
     (train_encoded['Time_spent_Alone'] >= 7))
]
print(f"\nPsychological contradictions: {len(psych_contradictions)}")

print("\n" + "="*60)
print("3. ANOMALY DETECTION")
print("="*60)

# Prepare data for anomaly detection
features_for_anomaly = numerical_features + ['Stage_fear_enc', 'Drained_enc']
X_anomaly = train_encoded[features_for_anomaly].copy()

# Fill missing values
for col in numerical_features:
    X_anomaly[col] = X_anomaly[col].fillna(X_anomaly[col].median())
X_anomaly = X_anomaly.fillna(0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_anomaly)

# Isolation Forest for each personality type
for personality in ['Introvert', 'Extrovert']:
    mask = train_encoded['Personality'] == personality
    X_personality = X_scaled[mask]
    
    # Detect anomalies
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    anomalies = iso_forest.fit_predict(X_personality)
    
    anomaly_indices = train_encoded[mask].index[anomalies == -1]
    anomaly_samples = train_encoded.loc[anomaly_indices]
    
    print(f"\nAnomalies in {personality} group: {len(anomaly_samples)}")
    print("Top 5 most anomalous:")
    for idx, row in anomaly_samples.head(5).iterrows():
        print(f"  ID {row['id']}: Alone={row['Time_spent_Alone']}, Social={row['Social_event_attendance']}, "
              f"Friends={row['Friends_circle_size']}, Drained={row['Drained_after_socializing']}")

print("\n" + "="*60)
print("4. STATISTICAL ANALYSIS OF SUSPICIOUS CASES")
print("="*60)

# Calculate "typicality" scores
def calculate_typicality(row):
    if row['is_introvert']:
        # Higher score = more typical introvert
        score = (
            row['Time_spent_Alone'] * 2 +
            (10 - row['Social_event_attendance']) +
            (30 - row['Friends_circle_size']) / 3 +
            row['Drained_enc'] * 5 +
            row['Stage_fear_enc'] * 3
        )
    else:
        # Higher score = more typical extrovert
        score = (
            (10 - row['Time_spent_Alone']) * 2 +
            row['Social_event_attendance'] +
            row['Friends_circle_size'] / 3 +
            (1 - row['Drained_enc']) * 5 +
            (1 - row['Stage_fear_enc']) * 3
        )
    return score

# Apply typicality calculation
train_encoded['typicality_score'] = train_encoded.apply(
    lambda row: calculate_typicality(row) if pd.notna(row['Time_spent_Alone']) else np.nan, 
    axis=1
)

# Find least typical examples (potential errors)
least_typical = train_encoded.nsmallest(20, 'typicality_score')
print("\nLeast typical examples (potential labeling errors):")
print("-" * 60)
for idx, row in least_typical.iterrows():
    print(f"\nID {row['id']} - {row['Personality']} (Typicality: {row['typicality_score']:.2f})")
    print(f"  Alone={row['Time_spent_Alone']}, Social={row['Social_event_attendance']}, "
          f"Friends={row['Friends_circle_size']}")
    print(f"  Drained={row['Drained_after_socializing']}, Stage_fear={row['Stage_fear']}")

# Special focus on ambivert markers
print("\n" + "="*60)
print("5. AMBIVERT MARKER ANALYSIS")
print("="*60)

# Check for special marker values
marker_values = [3.1377639321564557, 5.265106088560886, 4.044319380935631, 4.982097334878332]
print("\nChecking for ambivert marker values:")
for col in numerical_features:
    for marker in marker_values:
        count = len(train_encoded[train_encoded[col] == marker])
        if count > 0:
            personalities = train_encoded[train_encoded[col] == marker]['Personality'].value_counts()
            print(f"{col} = {marker}: {count} cases")
            print(f"  Distribution: {personalities.to_dict()}")

# Calculate error potential
print("\n" + "="*60)
print("6. ERROR IMPACT ESTIMATION")
print("="*60)

total_suspicious = len(extreme_intro_as_extro) + len(extreme_extro_as_intro) + len(psych_contradictions)
error_rate = total_suspicious / len(train_df) * 100

print(f"\nTotal suspicious cases: {total_suspicious}")
print(f"Potential error rate: {error_rate:.2f}%")
print(f"Expected accuracy ceiling if these are errors: {100 - error_rate:.2f}%")

# Save suspicious cases for manual review
suspicious_df = pd.concat([
    extreme_intro_as_extro.assign(issue='extreme_intro_as_extro'),
    extreme_extro_as_intro.assign(issue='extreme_extro_as_intro'),
    least_typical.head(10).assign(issue='least_typical')
]).drop_duplicates(subset=['id'])

suspicious_df.to_csv('output/suspicious_training_labels.csv', index=False)
print(f"\nSaved {len(suspicious_df)} suspicious cases to: output/suspicious_training_labels.csv")

# Summary
print("\n" + "="*80)
print("CONCLUSIONS:")
print("="*80)
print("1. Found multiple cases with extreme contradictions")
print("2. Some labels appear statistically improbable")
print(f"3. Estimated {error_rate:.2f}% of training data may have labeling errors")
print("4. This could explain the 97.5708% accuracy ceiling")
print("5. Recommendation: Consider these cases for manual review or exclusion")