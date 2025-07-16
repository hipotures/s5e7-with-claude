#!/usr/bin/env python3
"""
Final simplified metamodel demo showing the core concept.
"""

import pandas as pd
import numpy as np
import ydf
from datetime import datetime

print("="*60)
print("METAMODEL FLIP PREDICTOR - CONCEPT DEMONSTRATION")
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

print("\n2. Analyzing null patterns in training data...")
# Check null patterns
intro_nulls = train_df[train_df['Personality'] == 'Introvert'][feature_cols].isna().any(axis=1).mean()
extro_nulls = train_df[train_df['Personality'] == 'Extrovert'][feature_cols].isna().any(axis=1).mean()
print(f"   Introverts with nulls: {intro_nulls:.1%}")
print(f"   Extroverts with nulls: {extro_nulls:.1%}")

print("\n3. Finding candidates that violate the pattern...")
violations = []
for idx in range(len(train_df)):
    sample = train_df.iloc[idx]
    has_nulls = sample[feature_cols].isna().any()
    is_introvert = sample['Personality'] == 'Introvert'
    
    # Violation: Introvert without nulls or Extrovert with nulls
    if (is_introvert and not has_nulls) or (not is_introvert and has_nulls):
        violations.append(idx)

print(f"   Found {len(violations)} samples violating the pattern ({len(violations)/len(train_df):.1%})")

print("\n4. Testing flip impact (simplified evaluation)...")
# Take a small subset for quick testing
np.random.seed(42)
test_indices = np.random.choice(violations[:100], size=min(5, len(violations)), replace=False)

flip_results = []
for i, idx in enumerate(test_indices):
    print(f"\n   Testing flip #{i+1} (index {idx})...")
    
    # Create train/val split
    all_indices = list(range(len(train_df)))
    all_indices.remove(idx)
    np.random.shuffle(all_indices)
    
    train_indices = all_indices[:10000]
    val_indices = all_indices[10000:12000]
    
    # Prepare datasets
    X_train = train_df.iloc[train_indices][feature_cols].fillna(-1)
    y_train = train_df.iloc[train_indices]['Personality']
    X_val = train_df.iloc[val_indices][feature_cols].fillna(-1)
    y_val = train_df.iloc[val_indices]['Personality']
    
    # Convert categorical features
    for col in ['Stage_fear', 'Drained_after_socializing']:
        X_train[col] = X_train[col].map({'Yes': 1, 'No': 0, -1: -1})
        X_val[col] = X_val[col].map({'Yes': 1, 'No': 0, -1: -1})
    
    # Create YDF datasets
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    
    # Train without flip
    model1 = ydf.RandomForestLearner(
        label='Personality',
        num_trees=30,
        max_depth=8
    ).train(train_data)
    
    acc1 = (model1.predict(val_data) == y_val).mean()
    
    # Get sample info
    current = train_df.iloc[idx]['Personality']
    new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
    
    # Create version with flipped label
    train_data_flipped = train_data.copy()
    # Add the flipped sample at the end
    flip_sample = train_df.iloc[[idx]]
    flip_features = flip_sample[feature_cols].fillna(-1)
    for col in ['Stage_fear', 'Drained_after_socializing']:
        flip_features[col] = flip_features[col].map({'Yes': 1, 'No': 0, -1: -1})
    flip_features['Personality'] = new_label
    
    train_data_flipped = pd.concat([train_data, flip_features], ignore_index=True)
    
    # Train with flip
    model2 = ydf.RandomForestLearner(
        label='Personality',
        num_trees=30,
        max_depth=8
    ).train(train_data_flipped)
    
    acc2 = (model2.predict(val_data) == y_val).mean()
    
    impact = acc2 - acc1
    print(f"      Original: {current}, Flip to: {new_label}")
    print(f"      Accuracy change: {acc1:.4f} → {acc2:.4f} ({impact:+.4f})")
    
    flip_results.append({
        'idx': idx,
        'original': current,
        'impact': impact
    })

print("\n5. Summary of flip impacts:")
if flip_results:
    positive_flips = [f for f in flip_results if f['impact'] > 0]
    print(f"   Positive impact flips: {len(positive_flips)}/{len(flip_results)}")
    
    if positive_flips:
        print("\n   Best flips found:")
        for flip in sorted(positive_flips, key=lambda x: x['impact'], reverse=True):
            print(f"   - Index {flip['idx']} ({flip['original']}): {flip['impact']:+.4f}")

print("\n6. Concept Summary:")
print("   - The metamodel learns which flips improve performance")
print("   - It uses features like null patterns, KNN disagreement, etc.")
print("   - YDF provides fast evaluation on subsets")
print("   - A neural network learns to predict flip quality")
print("   - This approach could find actual mislabeled data")

print("\n" + "="*60)
print("In production, this would:")
print("1. Generate thousands of flip evaluations")
print("2. Train a neural network on the results")
print("3. Apply to test set to find likely errors")
print("4. Create targeted flip submissions")
print("="*60)