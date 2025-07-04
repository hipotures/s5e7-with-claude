#!/usr/bin/env python3
"""Check if features hash to personality in a deterministic way.

PURPOSE: Investigate if there's a hidden deterministic formula or hash function
         that maps features directly to personality, which could enable perfect
         classification.

HYPOTHESIS: The competition might have a hidden deterministic mapping where
            specific feature combinations or hash values always result in the
            same personality type.

EXPECTED: Test various mathematical formulas including sum mod 2, weighted sums,
          XOR patterns, cryptographic hashes, and magic formulas to find a
          deterministic mapping that achieves >97.5% accuracy.

RESULT: Tested multiple approaches:
        - Sum mod 2: Low accuracy
        - Weighted sums: Searched 100 random weights
        - XOR patterns: Created binary features and XORed them
        - MD5 hash: Converted features to string and hashed
        - Magic formula: Created 7-bit encoding from threshold comparisons
        Found some magic number groups with perfect classification (all 0 or all 1)
        for samples with 10+ instances. No single deterministic formula found that
        breaks the 0.975708 barrier.
"""

import pandas as pd
import numpy as np
import hashlib

print("PERFECT SCORE HUNT: Feature Hash Analysis")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Preprocess
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency']

for df in [train_df, test_df]:
    # Fill missing values with specific numbers that might matter
    df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(3.14159)  # Pi
    df['Social_event_attendance'] = df['Social_event_attendance'].fillna(2.71828)  # e
    df['Going_outside'] = df['Going_outside'].fillna(1.61803)  # Golden ratio
    df['Friends_circle_size'] = df['Friends_circle_size'].fillna(7)  # Lucky number
    df['Post_frequency'] = df['Post_frequency'].fillna(5)  # Middle value
    
    # Encode categorical
    df['Stage_fear_num'] = df['Stage_fear'].map({'Yes': 1, 'No': 0}).fillna(0.5)
    df['Drained_num'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0}).fillna(0.5)

train_df['is_extrovert'] = (train_df['Personality'] == 'Extrovert').astype(int)

print("\nTesting various hash functions...")

# Test 1: Simple sum mod 2
print("\n" + "="*60)
print("TEST 1: Feature Sum Mod 2")
print("="*60)

numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency', 'Stage_fear_num', 'Drained_num']

train_df['feature_sum'] = train_df[numerical_features].sum(axis=1)
train_df['sum_mod2'] = (train_df['feature_sum'] % 2 > 0.5).astype(int)

accuracy = (train_df['sum_mod2'] == train_df['is_extrovert']).mean()
print(f"Sum mod 2 accuracy: {accuracy:.4f}")

# Test 2: Weighted sum
print("\n" + "="*60)
print("TEST 2: Weighted Feature Sum")
print("="*60)

# Try different weights
best_accuracy = 0
best_weights = None

for seed in range(100):
    np.random.seed(seed)
    weights = np.random.rand(len(numerical_features))
    
    train_df['weighted_sum'] = sum(train_df[feat] * weight 
                                   for feat, weight in zip(numerical_features, weights))
    
    # Try different thresholds
    for threshold in np.percentile(train_df['weighted_sum'], [25, 50, 75]):
        pred = (train_df['weighted_sum'] > threshold).astype(int)
        acc = (pred == train_df['is_extrovert']).mean()
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_weights = weights.copy()
            best_threshold = threshold

print(f"Best weighted sum accuracy: {best_accuracy:.4f}")
if best_accuracy > 0.97:
    print(f"Weights: {best_weights}")
    print(f"Threshold: {best_threshold}")

# Test 3: XOR patterns
print("\n" + "="*60)
print("TEST 3: XOR Patterns")
print("="*60)

# Binary features
binary_features = []
for feat in numerical_features:
    if feat.endswith('_num'):
        binary_features.append(feat)
    else:
        median = train_df[feat].median()
        binary_feat = f'{feat}_binary'
        train_df[binary_feat] = (train_df[feat] > median).astype(int)
        test_df[binary_feat] = (test_df[feat] > median).astype(int)
        binary_features.append(binary_feat)

# XOR all binary features
train_df['xor_result'] = 0
for feat in binary_features:
    train_df['xor_result'] ^= train_df[feat].astype(int)

xor_accuracy = (train_df['xor_result'] == train_df['is_extrovert']).mean()
print(f"XOR pattern accuracy: {xor_accuracy:.4f}")

# Test 4: Hash function
print("\n" + "="*60)
print("TEST 4: Cryptographic Hash")
print("="*60)

def feature_hash(row):
    # Create string from features
    feature_str = '|'.join(str(row[f]) for f in numerical_features)
    # Hash it
    hash_obj = hashlib.md5(feature_str.encode())
    hash_hex = hash_obj.hexdigest()
    # Convert first character to 0/1
    return int(hash_hex[0], 16) % 2

train_df['hash_pred'] = train_df.apply(feature_hash, axis=1)
hash_accuracy = (train_df['hash_pred'] == train_df['is_extrovert']).mean()
print(f"Hash function accuracy: {hash_accuracy:.4f}")

# Test 5: Magic formula (discovered by reverse engineering?)
print("\n" + "="*60)
print("TEST 5: Magic Formula")
print("="*60)

# Maybe the formula is hidden in feature names?
# T_ime, S_tage, S_ocial, G_oing, D_rained, F_riends, P_ost
# T S S G D F P... any pattern?

train_df['magic'] = (
    (train_df['Time_spent_Alone'] < 3.5) * 1 +
    (train_df['Stage_fear_num'] == 0) * 2 +
    (train_df['Social_event_attendance'] > 5) * 4 +
    (train_df['Going_outside'] > 4) * 8 +
    (train_df['Drained_num'] == 0) * 16 +
    (train_df['Friends_circle_size'] > 8) * 32 +
    (train_df['Post_frequency'] > 5) * 64
)

# Check if certain magic numbers always mean Extrovert
magic_groups = train_df.groupby('magic')['is_extrovert'].agg(['mean', 'count'])
perfect_magic = magic_groups[magic_groups['mean'].isin([0.0, 1.0])]

if len(perfect_magic) > 0:
    print("\nFound perfect magic numbers!")
    print(perfect_magic[perfect_magic['count'] > 10])
    
    # Apply to test
    test_df['magic'] = (
        (test_df['Time_spent_Alone'] < 3.5) * 1 +
        (test_df['Stage_fear_num'] == 0) * 2 +
        (test_df['Social_event_attendance'] > 5) * 4 +
        (test_df['Going_outside'] > 4) * 8 +
        (test_df['Drained_num'] == 0) * 16 +
        (test_df['Friends_circle_size'] > 8) * 32 +
        (test_df['Post_frequency'] > 5) * 64
    )
    
    # Create perfect mapping
    extrovert_magics = perfect_magic[perfect_magic['mean'] == 1.0].index
    test_df['Personality'] = 'Introvert'
    test_df.loc[test_df['magic'].isin(extrovert_magics), 'Personality'] = 'Extrovert'
    
    submission = test_df[['id', 'Personality']]
    submission.to_csv('perfect_magic_formula.csv', index=False)
    print("\nSaved: perfect_magic_formula.csv")

# Test 6: Floating point precision trick
print("\n" + "="*60)
print("TEST 6: Floating Point Precision")
print("="*60)

# Check if exact floating point values matter
unique_values = {}
for feat in numerical_features[:5]:  # Just numerical, not binary
    unique_values[feat] = train_df[feat].value_counts().head(10)
    print(f"\nTop values for {feat}:")
    print(unique_values[feat])

print("\nIf you see values like 3.0000000000000004, the generator might use specific floats!")