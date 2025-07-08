#!/usr/bin/env python3
"""Quick test of the optimization logic without Optuna"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Load data
print("Loading data...")
train_df = pd.read_csv("output/train_corrected_01.csv")
test_df = pd.read_csv("/mnt/ml/kaggle/playground-series-s5e7/test.csv")

# Preprocess
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                 'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

for col in numerical_cols:
    mean_val = train_df[col].mean()
    train_df[col] = train_df[col].fillna(mean_val)
    test_df[col] = test_df[col].fillna(mean_val)

for col in categorical_cols:
    train_df[col] = train_df[col].fillna('Missing')
    test_df[col] = test_df[col].fillna('Missing')
    train_df[col] = train_df[col].map({'Yes': 1, 'No': 0, 'Missing': 0.5})
    test_df[col] = test_df[col].map({'Yes': 1, 'No': 0, 'Missing': 0.5})

train_df['Personality'] = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

# Create simple features
features = numerical_cols + categorical_cols
X = train_df[features]
y = train_df['Personality']

# Test ambiguity features
print("\nTesting ambiguity feature creation...")
df_feat = X.copy()
df_feat['ambiguous_pattern'] = (
    (X['Time_spent_Alone'] < 3.0) & 
    (X['Social_event_attendance'] < 4.5) &
    (X['Friends_circle_size'] < 7.5)
).astype(float)

df_feat['high_ambiguity'] = (df_feat['ambiguous_pattern'] > 0.5).astype(float)
print(f"High ambiguity samples: {df_feat['high_ambiguity'].sum()}")

# Test CV with ambiguity logic
print("\nTesting CV split...")
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold}:")
    print(f"  Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    
    # Test the problematic indexing
    val_high_ambig = df_feat['high_ambiguity'].iloc[val_idx]
    print(f"  Val high ambig type: {type(val_high_ambig)}, shape: {val_high_ambig.shape}")
    print(f"  Val high ambig values sample: {val_high_ambig.values[:5]}")
    
    # Create dummy predictions
    proba = np.random.rand(len(val_idx))
    pred = (proba > 0.5).astype(int)
    print(f"  Pred type: {type(pred)}, shape: {pred.shape}")
    
    # Test the fix
    val_high_ambig_mask = val_high_ambig.values.astype(bool)
    print(f"  Mask type: {type(val_high_ambig_mask)}, shape: {val_high_ambig_mask.shape}")
    print(f"  Mask sum: {val_high_ambig_mask.sum()}")
    
    # Apply the rule
    try:
        pred[val_high_ambig_mask] = 1
        print("  ✓ Indexing worked!")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    break  # Just test first fold

print("\nTest complete!")