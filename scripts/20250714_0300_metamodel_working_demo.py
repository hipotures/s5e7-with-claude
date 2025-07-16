#!/usr/bin/env python3
"""
Working metamodel demo with correct YDF prediction handling.
"""

import pandas as pd
import numpy as np
import ydf
from datetime import datetime

print("="*60)
print("METAMODEL FLIP PREDICTOR - WORKING DEMONSTRATION")
print("="*60)

# Load data
print("\n1. Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')
print(f"   Train samples: {len(train_df)}")
print(f"   Test samples: {len(test_df)}")

# Feature columns
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
               'Going_outside', 'Drained_after_socializing', 
               'Friends_circle_size', 'Post_frequency']

print("\n2. Finding pattern violations (potential mislabeling)...")
violations = []
for idx in range(min(1000, len(train_df))):  # Check first 1000
    sample = train_df.iloc[idx]
    has_nulls = sample[feature_cols].isna().any()
    is_introvert = sample['Personality'] == 'Introvert'
    
    # Pattern: Introverts usually have nulls, Extroverts usually don't
    if (is_introvert and not has_nulls) or (not is_introvert and has_nulls):
        violations.append(idx)

print(f"   Found {len(violations)} violations in first 1000 samples")

print("\n3. Testing flip impact on pattern violations...")
# Test on a few violations
test_indices = violations[:3]

for i, idx in enumerate(test_indices):
    print(f"\n   Test #{i+1} - Sample index {idx}:")
    
    # Get sample info
    sample_info = train_df.iloc[idx]
    current_label = sample_info['Personality']
    has_nulls = sample_info[feature_cols].isna().any()
    
    print(f"   Current: {current_label}, Has nulls: {has_nulls}")
    
    # Create small train/val sets
    np.random.seed(42)
    other_indices = [i for i in range(len(train_df)) if i != idx]
    train_idx = np.random.choice(other_indices, 5000, replace=False)
    val_idx = np.random.choice([i for i in other_indices if i not in train_idx], 1000, replace=False)
    
    # Prepare data
    def prepare_data(indices):
        data = train_df.iloc[indices][feature_cols].copy()
        # Handle nulls
        data = data.fillna(-1)
        # Convert categorical
        for col in ['Stage_fear', 'Drained_after_socializing']:
            data[col] = data[col].map({'Yes': 1, 'No': 0, -1: -1})
        data['Personality'] = train_df.iloc[indices]['Personality']
        return data
    
    train_data = prepare_data(train_idx)
    val_data = prepare_data(val_idx)
    
    # Train baseline model
    model1 = ydf.RandomForestLearner(
        label='Personality',
        num_trees=50,
        max_depth=8
    ).train(train_data)
    
    # Get predictions (YDF returns P(Introvert))
    val_probs1 = model1.predict(val_data)
    # Convert to class predictions
    val_preds1 = ['Introvert' if p > 0.5 else 'Extrovert' for p in val_probs1]
    acc1 = np.mean([pred == actual for pred, actual in zip(val_preds1, val_data['Personality'])])
    
    # Now add flipped sample to training
    flip_data = prepare_data([idx])
    flip_data['Personality'] = 'Introvert' if current_label == 'Extrovert' else 'Extrovert'
    train_data_flipped = pd.concat([train_data, flip_data], ignore_index=True)
    
    # Train with flip
    model2 = ydf.RandomForestLearner(
        label='Personality',
        num_trees=50,
        max_depth=8
    ).train(train_data_flipped)
    
    val_probs2 = model2.predict(val_data)
    val_preds2 = ['Introvert' if p > 0.5 else 'Extrovert' for p in val_probs2]
    acc2 = np.mean([pred == actual for pred, actual in zip(val_preds2, val_data['Personality'])])
    
    impact = acc2 - acc1
    print(f"   Flip {current_label} → {flip_data['Personality'].iloc[0]}")
    print(f"   Accuracy: {acc1:.4f} → {acc2:.4f} (impact: {impact:+.4f})")
    
    # Check if this improves pattern consistency
    if impact > 0:
        print(f"   ✓ Positive impact! This flip improves model performance.")
    else:
        print(f"   ✗ No improvement. Original label might be correct.")

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("1. Pattern violations are candidates for mislabeling")
print("2. YDF quickly evaluates flip impact on held-out data")
print("3. Positive impact suggests the flip corrects an error")
print("4. A neural network would learn from many such evaluations")
print("5. Apply to test set to find likely mislabeled samples")
print("="*60)

# Create a simple submission with pattern-based flips
print("\nCreating demonstration submission...")

# Find pattern violations in test set
test_violations = []
for idx in range(len(test_df)):
    sample = test_df.iloc[idx]
    has_nulls = sample[feature_cols].isna().any()
    
    # For demo, assume baseline predicts based on null pattern
    # (In reality, you'd load actual predictions)
    if has_nulls:
        predicted_label = 'Introvert'  # Baseline assumes nulls = Introvert
    else:
        predicted_label = 'Extrovert'  # No nulls = Extrovert
    
    # Check if prediction violates pattern
    if (predicted_label == 'Introvert' and not has_nulls) or \
       (predicted_label == 'Extrovert' and has_nulls):
        test_violations.append(idx)

print(f"Found {len(test_violations)} pattern violations in test set")

if len(test_violations) > 0:
    # Create baseline submission
    baseline = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Introvert' if test_df.iloc[i][feature_cols].isna().any() else 'Extrovert' 
                       for i in range(len(test_df))]
    })
    
    # Create submission with top 5 flips
    submission = baseline.copy()
    flips_to_make = test_violations[:5]
    
    print(f"\nFlipping top {len(flips_to_make)} violations:")
    for idx in flips_to_make:
        test_id = test_df.iloc[idx]['id']
        current = submission.iloc[idx]['Personality']
        new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
        submission.iloc[idx, 1] = new_label
        print(f"  ID {test_id}: {current} → {new_label}")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f'../scores/metamodel_pattern_flips_{timestamp}.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\nSaved to: {submission_file}")

print("\nDemo complete!")