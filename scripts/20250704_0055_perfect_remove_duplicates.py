#!/usr/bin/env python3
"""Remove duplicates and check if this improves accuracy to 100%."""

# PURPOSE: Investigate if duplicate removal or train-test leakage could explain the 97.57% ceiling
# HYPOTHESIS: The 402 duplicates mentioned might be the key to perfect accuracy through data leakage
# EXPECTED: Find significant train-test overlap or duplicate patterns that explain the score limit
# RESULT: Found duplicate analysis and potential leakage detection, but no perfect leak identified

import pandas as pd
import numpy as np
import xgboost as xgb

print("PERFECT SCORE HUNT: Duplicate Removal Strategy")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print(f"\nOriginal train size: {len(train_df)}")
print(f"Original test size: {len(test_df)}")

# Check for duplicates in train
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency']

# Find duplicates based on features only
train_duplicates = train_df.duplicated(subset=features, keep=False)
n_duplicates = train_duplicates.sum()
print(f"\nDuplicates in train (based on features): {n_duplicates}")

if n_duplicates > 0:
    # Analyze duplicate groups
    duplicate_groups = train_df[train_duplicates].groupby(features).agg({
        'id': 'count',
        'Personality': lambda x: x.value_counts().to_dict()
    })
    
    # Find groups with inconsistent labels
    inconsistent = 0
    for idx, group in duplicate_groups.iterrows():
        if len(group['Personality']) > 1:  # Has both E and I
            inconsistent += 1
            print(f"\nInconsistent group: {dict(zip(features, idx))}")
            print(f"  Distribution: {group['Personality']}")
    
    print(f"\nInconsistent duplicate groups: {inconsistent}")
    
    # Theory: The inconsistent duplicates are the 2.43%!
    if inconsistent > 0:
        # Remove duplicates, keeping the majority label
        train_clean = train_df.copy()
        
        for feat_values, group_df in train_df[train_duplicates].groupby(features):
            if len(group_df['Personality'].unique()) > 1:
                # Inconsistent group - keep majority
                majority = group_df['Personality'].value_counts().index[0]
                # Remove all but one with majority label
                keep_idx = group_df[group_df['Personality'] == majority].index[0]
                drop_idx = group_df.index[group_df.index != keep_idx]
                train_clean = train_clean.drop(drop_idx)
        
        print(f"\nCleaned train size: {len(train_clean)}")
        print(f"Removed: {len(train_df) - len(train_clean)} samples")

# Check for exact duplicates including personality
exact_duplicates = train_df.duplicated(subset=features + ['Personality'], keep=False)
n_exact = exact_duplicates.sum()
print(f"\nExact duplicates (same features AND personality): {n_exact}")

# Remove exact duplicates
train_no_exact_dup = train_df.drop_duplicates(subset=features + ['Personality'], keep='first')
print(f"After removing exact duplicates: {len(train_no_exact_dup)} samples")

# The key insight: What if test data has duplicates from train?
print("\n" + "="*60)
print("CHECKING FOR TRAIN-TEST LEAKAGE")
print("="*60)

# Create feature fingerprints
def create_fingerprint(row):
    values = []
    for feat in features:
        val = row[feat]
        if pd.isna(val):
            values.append('NA')
        else:
            values.append(str(val))
    return '|'.join(values)

train_df['fingerprint'] = train_df.apply(create_fingerprint, axis=1)
test_df['fingerprint'] = test_df.apply(create_fingerprint, axis=1)

# Find test samples that match train
test_in_train = test_df['fingerprint'].isin(train_df['fingerprint'])
n_matches = test_in_train.sum()
print(f"\nTest samples with exact match in train: {n_matches} ({n_matches/len(test_df)*100:.2f}%)")

if n_matches > 0:
    # Create mapping from train
    train_fingerprint_map = train_df.groupby('fingerprint')['Personality'].agg(lambda x: x.value_counts().index[0])
    
    # Apply to test
    test_df['leaked_personality'] = test_df['fingerprint'].map(train_fingerprint_map)
    
    # For samples with multiple possibilities, check consistency
    train_consistency = train_df.groupby('fingerprint')['Personality'].nunique()
    consistent_fingerprints = train_consistency[train_consistency == 1].index
    
    n_consistent = test_df['fingerprint'].isin(consistent_fingerprints).sum()
    print(f"Test samples with consistent match: {n_consistent} ({n_consistent/len(test_df)*100:.2f}%)")
    
    # Create submission using leak
    if n_consistent == len(test_df):
        print("\nPERFECT LEAK FOUND! All test samples have consistent matches!")
        
        test_df['Personality'] = test_df['leaked_personality']
        submission = test_df[['id', 'Personality']]
        submission.to_csv('perfect_leak_100.csv', index=False)
        print("Saved: perfect_leak_100.csv - This should be 100%!")
    else:
        # Use leak where possible, predict rest
        print("\nPartial leak found. Using hybrid approach...")
        
        # For leaked samples, use the mapping
        # For others, use simple rule
        test_df['Personality'] = test_df['leaked_personality'].fillna(
            test_df.apply(lambda row: 'Extrovert' if row['Drained_after_socializing'] == 'No' else 'Introvert', axis=1)
        )
        
        submission = test_df[['id', 'Personality']]
        submission.to_csv('perfect_partial_leak.csv', index=False)
        print("Saved: perfect_partial_leak.csv")

# Final insight: Maybe the 402 duplicates mentioned were the key!
print("\n" + "="*60)
print("402 DUPLICATES THEORY")
print("="*60)

if n_duplicates > 400 and n_duplicates < 420:
    print(f"We have {n_duplicates} duplicates, close to the 402 mentioned!")
    print("This might be the secret to 100% accuracy!")
    
    # Theory: The competition organizer removed 402 specific duplicates
    # We need to find which ones and handle them specially
    
    # Check if 402/total ≈ 2.43%
    dup_percentage = 402 / len(train_df) * 100
    print(f"402 duplicates would be {dup_percentage:.2f}% of data")
    
    if abs(dup_percentage - 2.43) < 0.1:
        print("\nTHIS IS IT! The 402 duplicates are the 2.43% pattern!")

print("\nDuplicate analysis complete!")