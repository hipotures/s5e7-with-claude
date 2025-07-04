#!/usr/bin/env python3
"""Find exact duplicates between train and test - classic Kaggle leak.

PURPOSE: Search for data leakage by finding exact or near-duplicate samples
         between training and test sets, which could enable perfect predictions
         for those samples.

HYPOTHESIS: Some test samples might be exact duplicates of training samples
            (a common competition leak), or near-duplicates that can be matched
            with high confidence.

EXPECTED: Create fingerprints for each sample, find exact matches between train
          and test, identify near-duplicates within tolerance, and check for
          public dataset contamination patterns.

RESULT: Created high-precision fingerprints for all samples using 10 decimal places.
        Searched for exact duplicates - would create perfect predictions if found.
        Implemented near-duplicate detection with 1% tolerance for numerical features.
        Checked for public MBTI dataset signatures (specific feature combinations).
        Examined edge cases (first/last samples). While the infrastructure for leak
        detection was comprehensive, no significant data leakage was discovered.
"""

import pandas as pd
import numpy as np

print("PERFECT SCORE HUNT: Duplicate/Leak Detection")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency']

# Create feature strings for comparison
print("\nCreating feature fingerprints...")

def create_fingerprint(row):
    values = []
    for feat in features:
        val = row[feat]
        if pd.isna(val):
            values.append('NA')
        elif isinstance(val, float):
            values.append(f"{val:.10f}")  # High precision
        else:
            values.append(str(val))
    return '|'.join(values)

train_df['fingerprint'] = train_df.apply(create_fingerprint, axis=1)
test_df['fingerprint'] = test_df.apply(create_fingerprint, axis=1)

# Find exact duplicates
print("\n" + "="*60)
print("SEARCHING FOR EXACT DUPLICATES")
print("="*60)

duplicates = []
for idx, test_row in test_df.iterrows():
    matches = train_df[train_df['fingerprint'] == test_row['fingerprint']]
    if len(matches) > 0:
        duplicates.append({
            'test_id': test_row['id'],
            'train_ids': matches['id'].tolist(),
            'personality': matches['Personality'].iloc[0],
            'all_same': matches['Personality'].nunique() == 1
        })

print(f"\nFound {len(duplicates)} exact duplicates!")

if duplicates:
    # Create submission based on duplicates
    dup_df = pd.DataFrame(duplicates)
    print("\nDuplicate analysis:")
    print(f"- All have same personality: {dup_df['all_same'].all()}")
    
    # Map duplicates
    dup_map = {d['test_id']: d['personality'] for d in duplicates if d['all_same']}
    
    # Create submission
    test_df['leak_personality'] = test_df['id'].map(dup_map)
    print(f"- Can predict {len(dup_map)} test samples with 100% confidence")
    
    if len(dup_map) > 0:
        # For non-duplicates, use a simple rule
        test_df['Personality'] = test_df['leak_personality'].fillna('Extrovert')
        
        submission = test_df[['id', 'Personality']]
        submission.to_csv('perfect_duplicate_leak.csv', index=False)
        print("\nSaved: perfect_duplicate_leak.csv")

# Check for near-duplicates
print("\n" + "="*60)
print("SEARCHING FOR NEAR-DUPLICATES")
print("="*60)

# Numerical features only
num_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

# Fill missing values
for feat in num_features:
    mean_val = train_df[feat].mean()
    train_df[feat] = train_df[feat].fillna(mean_val)
    test_df[feat] = test_df[feat].fillna(mean_val)

# Find samples that are very similar
near_duplicates = []
tolerance = 0.01  # 1% tolerance

for idx, test_row in test_df.iterrows():
    # Calculate distance to all train samples
    distances = np.sqrt(sum((train_df[feat] - test_row[feat])**2 for feat in num_features))
    
    # Find very close matches
    close_matches = train_df[distances < tolerance]
    
    if len(close_matches) > 0:
        personalities = close_matches['Personality'].value_counts()
        most_common = personalities.index[0]
        confidence = personalities.iloc[0] / len(close_matches)
        
        if confidence > 0.9:  # 90% of close matches have same personality
            near_duplicates.append({
                'test_id': test_row['id'],
                'predicted': most_common,
                'confidence': confidence,
                'n_matches': len(close_matches)
            })

print(f"\nFound {len(near_duplicates)} near-duplicates with high confidence")

if near_duplicates:
    near_dup_df = pd.DataFrame(near_duplicates)
    print(f"Average confidence: {near_dup_df['confidence'].mean():.2%}")

# Check for public dataset contamination
print("\n" + "="*60)
print("CHECKING FOR PUBLIC DATASET CONTAMINATION")
print("="*60)

# Common MBTI dataset signatures
mbti_signatures = [
    # PersonalityCafe posts usually have these patterns
    (10, 8, 2),  # Common (friends, social, alone) pattern
    (15, 10, 0),  # Max friends, high social, no alone time
    (5, 5, 5),   # All medium values
]

contamination_count = 0
for signature in mbti_signatures:
    friends, social, alone = signature
    matches = test_df[
        (test_df['Friends_circle_size'] == friends) &
        (test_df['Social_event_attendance'] == social) &
        (test_df['Time_spent_Alone'] == alone)
    ]
    if len(matches) > 0:
        contamination_count += len(matches)
        print(f"Pattern {signature}: {len(matches)} matches")

if contamination_count > 100:
    print(f"\nWARNING: Possible public dataset contamination! {contamination_count} suspicious patterns")

# Edge case: First and last samples
print("\n" + "="*60)
print("EDGE CASES CHECK")
print("="*60)

# Sometimes first/last samples in test are from train
first_test = test_df.iloc[0]
last_test = test_df.iloc[-1]

print(f"\nFirst test sample (ID {first_test['id']}):")
print(f"  Features: {first_test[features].values}")

print(f"\nLast test sample (ID {last_test['id']}):")
print(f"  Features: {last_test[features].values}")

# Check if they match any train samples
for test_sample, name in [(first_test, 'First'), (last_test, 'Last')]:
    matches = train_df[train_df['fingerprint'] == create_fingerprint(test_sample)]
    if len(matches) > 0:
        print(f"\n{name} test sample matches train ID {matches['id'].values[0]}: {matches['Personality'].values[0]}")

print("\nLeak detection complete!")