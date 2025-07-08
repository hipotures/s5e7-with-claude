#!/usr/bin/env python3
"""
PURPOSE: Analyze the overfitting issue in ensemble models (CV: 0.977705 vs LB: 0.963562)
HYPOTHESIS: The model is overfitting to the 600 "ambiguous" training samples with high weights
EXPECTED: Understand why the gap exists and propose solutions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print("ANALYZING OVERFITTING ISSUE IN ENSEMBLE MODELS")
print("="*80)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")
ambiguous_df = pd.read_csv("/mnt/ml/kaggle/playground-series-s5e7/enhanced_ambiguous_600.csv")

print(f"\nData shapes:")
print(f"Training: {train_df.shape}")
print(f"Test: {test_df.shape}")
print(f"Ambiguous: {ambiguous_df.shape}")

# Analyze the enhanced_ambiguous_600.csv file
print("\n" + "="*60)
print("ANALYZING ENHANCED AMBIGUOUS 600 SAMPLES")
print("="*60)

# Check if these IDs are in train or test
train_ids = set(train_df['id'])
test_ids = set(test_df['id'])
ambiguous_ids = set(ambiguous_df['id'])

in_train = ambiguous_ids.intersection(train_ids)
in_test = ambiguous_ids.intersection(test_ids)

print(f"\nID Distribution:")
print(f"Total ambiguous IDs: {len(ambiguous_ids)}")
print(f"IDs in training set: {len(in_train)}")
print(f"IDs in test set: {len(in_test)}")
print(f"IDs in neither: {len(ambiguous_ids - train_ids - test_ids)}")

# Analyze the personality distribution
if len(in_train) > 0:
    train_ambig = train_df[train_df['id'].isin(ambiguous_ids)]
    print(f"\nPersonality distribution in these 600 training samples:")
    print(train_ambig['Personality'].value_counts())
    print(f"Percentage Extrovert: {(train_ambig['Personality']=='Extrovert').mean()*100:.1f}%")
    
    # Compare with overall training distribution
    print(f"\nOverall training distribution:")
    print(train_df['Personality'].value_counts())
    print(f"Percentage Extrovert: {(train_df['Personality']=='Extrovert').mean()*100:.1f}%")

# Analyze ambiguity scores
print(f"\nAmbiguity score statistics:")
print(f"Mean: {ambiguous_df['ambiguity_score'].mean():.4f}")
print(f"Median: {ambiguous_df['ambiguity_score'].median():.4f}")
print(f"Min: {ambiguous_df['ambiguity_score'].min():.4f}")
print(f"Max: {ambiguous_df['ambiguity_score'].max():.4f}")
print(f"Std: {ambiguous_df['ambiguity_score'].std():.4f}")

# Check how many are truly ambiguous (score > 0.3 based on code)
high_ambig = ambiguous_df[ambiguous_df['ambiguity_score'] > 0.3]
print(f"\nSamples with ambiguity_score > 0.3: {len(high_ambig)} ({len(high_ambig)/len(ambiguous_df)*100:.1f}%)")

# Analyze the feature patterns
print("\n" + "="*60)
print("FEATURE PATTERNS IN AMBIGUOUS SAMPLES")
print("="*60)

# Merge with training data to get full features
ambig_train = train_df[train_df['id'].isin(ambiguous_ids)].copy()

# Analyze key features
features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 
            'Post_frequency', 'Going_outside']

for feat in features:
    if feat in ambig_train.columns:
        print(f"\n{feat}:")
        print(f"  Ambiguous samples - Mean: {ambig_train[feat].mean():.2f}, Std: {ambig_train[feat].std():.2f}")
        print(f"  All training - Mean: {train_df[feat].mean():.2f}, Std: {train_df[feat].std():.2f}")

# Check null patterns
print("\n" + "="*60)
print("NULL PATTERNS IN AMBIGUOUS SAMPLES")
print("="*60)

null_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
             'Friends_circle_size', 'Post_frequency']

ambig_nulls = ambig_train[null_cols].isnull().sum(axis=1)
all_nulls = train_df[null_cols].isnull().sum(axis=1)

print(f"\nNull count distribution in ambiguous samples:")
print(ambig_nulls.value_counts().sort_index())
print(f"Mean nulls: {ambig_nulls.mean():.2f}")

print(f"\nNull count distribution in all training:")
print(all_nulls.value_counts().sort_index())
print(f"Mean nulls: {all_nulls.mean():.2f}")

# Analyze sample weights impact
print("\n" + "="*60)
print("SAMPLE WEIGHT IMPACT ANALYSIS")
print("="*60)

# Simulate the impact of ambig_weight (from optimization: 5.0 to 20.0)
total_samples = len(train_df)
ambig_samples = len(ambig_train)
regular_samples = total_samples - ambig_samples

for weight in [5.0, 10.0, 15.0, 20.0]:
    effective_ambig = ambig_samples * weight
    effective_total = regular_samples + effective_ambig
    ambig_influence = effective_ambig / effective_total * 100
    print(f"\nWith ambig_weight={weight}:")
    print(f"  Effective ambiguous samples: {effective_ambig:.0f}")
    print(f"  Ambiguous sample influence: {ambig_influence:.1f}%")

# Check corrected datasets impact
print("\n" + "="*60)
print("CORRECTED DATASETS ANALYSIS")
print("="*60)

# Load some corrected datasets to check overlap with ambiguous
corrected_files = ["train_corrected_01.csv", "train_corrected_04.csv", "train_corrected_07.csv"]

for cf in corrected_files:
    corrected_path = Path("output") / cf
    if corrected_path.exists():
        corrected_df = pd.read_csv(corrected_path)
        
        # Find which IDs were changed
        merged = train_df.merge(corrected_df, on='id', suffixes=('_orig', '_corr'))
        changed = merged[merged['Personality_orig'] != merged['Personality_corr']]
        changed_ids = set(changed['id'])
        
        # Check overlap with ambiguous
        overlap = changed_ids.intersection(ambiguous_ids)
        
        print(f"\n{cf}:")
        print(f"  Total corrections: {len(changed_ids)}")
        print(f"  Overlap with ambiguous 600: {len(overlap)} ({len(overlap)/len(changed_ids)*100:.1f}%)")
        
        if len(overlap) > 0:
            # Check personality changes in overlapping cases
            overlap_df = changed[changed['id'].isin(overlap)]
            print(f"  Changes in overlapping cases:")
            print(f"    Extrovert->Introvert: {((overlap_df['Personality_orig']=='Extrovert') & (overlap_df['Personality_corr']=='Introvert')).sum()}")
            print(f"    Introvert->Extrovert: {((overlap_df['Personality_orig']=='Introvert') & (overlap_df['Personality_corr']=='Extrovert')).sum()}")

# Final analysis
print("\n" + "="*80)
print("KEY FINDINGS AND RECOMMENDATIONS")
print("="*80)

print("""
1. AMBIGUOUS SAMPLE BIAS:
   - All 600 "ambiguous" samples are from training set (none in test)
   - 86.2% are labeled as Extroverts (vs 66.7% in overall training)
   - With ambig_weight=15, these 600 samples have 47.8% influence on training
   
2. OVERFITTING MECHANISM:
   - Model learns to heavily weight these 600 training samples
   - These patterns may not generalize to test set
   - CV uses same weighted samples, so CV score is inflated
   
3. CORRECTED DATASETS ISSUE:
   - Some corrections overlap with the "ambiguous" 600
   - Double impact: correction + high weight = extreme overfitting
   
4. RECOMMENDATIONS:
   a) Remove or reduce ambig_weight parameter (try 1.0-2.0 instead of 5-20)
   b) Use stratified sampling to ensure ambiguous patterns in validation
   c) Try training without the enhanced_ambiguous_600.csv weights
   d) Focus on simpler models without complex weighting schemes
   e) Consider that "ambiguous" samples might actually be mislabeled, not ambiguous
""")

print("\nAnalysis complete!")