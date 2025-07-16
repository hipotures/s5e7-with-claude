#!/usr/bin/env python3
"""
Test the metamodel concept with simplified output.
"""

import pandas as pd
import numpy as np
import ydf

# Load data
print("Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Prepare features
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
               'Going_outside', 'Drained_after_socializing', 
               'Friends_circle_size', 'Post_frequency']

# Example: Find samples with high null count and wrong pattern
print("\nFinding potential mislabeled samples...")

candidates = []
for idx in range(min(100, len(train_df))):  # Check first 100 samples
    sample = train_df.iloc[idx]
    
    # Check null pattern
    has_nulls = sample[feature_cols].isna().any()
    is_introvert = sample['Personality'] == 'Introvert'
    
    # Introvert with no nulls or Extrovert with nulls = potential error
    if (is_introvert and not has_nulls) or (not is_introvert and has_nulls):
        candidates.append({
            'idx': idx,
            'label': sample['Personality'],
            'null_count': sample[feature_cols].isna().sum(),
            'pattern_mismatch': True
        })

print(f"Found {len(candidates)} candidates with pattern mismatch")

# Show top candidates
print("\nTop 5 candidates:")
for i, cand in enumerate(candidates[:5]):
    print(f"{i+1}. idx={cand['idx']}, label={cand['label']}, nulls={cand['null_count']}")

# Test flip evaluation on one candidate
if candidates:
    print("\nTesting flip evaluation on first candidate...")
    test_idx = candidates[0]['idx']
    
    # Simple evaluation: train with and without flip
    train_subset = train_df.sample(n=10000, random_state=42).reset_index(drop=True)
    val_subset = train_df.drop(train_df.index[train_subset.index]).sample(n=2000, random_state=42).reset_index(drop=True)
    
    # Prepare data
    train_data = train_subset[feature_cols + ['Personality']].copy()
    val_data = val_subset[feature_cols + ['Personality']].copy()
    
    # Handle missing values
    train_data[feature_cols] = train_data[feature_cols].fillna(-1)
    val_data[feature_cols] = val_data[feature_cols].fillna(-1)
    
    # Train without flip
    print("Training without flip...")
    model1 = ydf.RandomForestLearner(
        label='Personality',
        num_trees=50,
        max_depth=8
    ).train(train_data)
    
    predictions1 = model1.predict(val_data)
    acc1 = (predictions1 == val_data['Personality']).mean()
    print(f"Accuracy without flip: {acc1:.4f}")
    print(f"Sample predictions: {predictions1[:5].tolist()}")
    print(f"Sample actuals: {val_data['Personality'][:5].tolist()}")
    
    # Train with flip
    if test_idx in train_subset.index:
        train_data_flipped = train_data.copy()
        # Find the row in the subset
        subset_idx = train_subset.index.get_loc(test_idx)
        current = train_data_flipped.iloc[subset_idx]['Personality']
        new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
        train_data_flipped.iloc[subset_idx, train_data_flipped.columns.get_loc('Personality')] = new_label
        
        print(f"Training with flip: {current} → {new_label}")
        model2 = ydf.RandomForestLearner(
            label='Personality',
            num_trees=50,
            max_depth=8
        ).train(train_data_flipped)
        
        acc2 = (model2.predict(val_data) == val_data['Personality']).mean()
        print(f"Accuracy with flip: {acc2:.4f}")
        print(f"Impact: {acc2 - acc1:+.4f}")
    else:
        print("Test sample not in training subset")

print("\n" + "="*50)
print("Key insight: The metamodel would learn from many such evaluations")
print("to predict which flips are likely to improve performance.")
print("="*50)