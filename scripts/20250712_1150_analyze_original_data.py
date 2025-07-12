#!/usr/bin/env python3
"""
Analyze original personality dataset and compare with synthetic
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ORIG_DIR = Path("/mnt/ml/kaggle/original-personality-data/")
SYNTH_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")

def analyze_original_data():
    """Analyze the original personality dataset"""
    
    print("="*60)
    print("ANALYZING ORIGINAL PERSONALITY DATASET")
    print("="*60)
    
    # Load both files (seems like duplicates with typo)
    try:
        df1 = pd.read_csv(ORIG_DIR / "personality_dataset.csv")
        print(f"\npersonality_dataset.csv: {df1.shape}")
        print(f"Columns: {list(df1.columns)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    try:
        df2 = pd.read_csv(ORIG_DIR / "personality_datasert.csv")
        print(f"\npersonality_datasert.csv: {df2.shape}")
        print(f"Columns: {list(df2.columns)}")
    except Exception as e:
        print(f"Error loading datasert: {e}")
    
    # Use the first one
    orig_df = df1
    
    # Load synthetic competition data
    synth_df = pd.read_csv(SYNTH_DIR / "train.csv")
    
    print("\n" + "="*40)
    print("ORIGINAL DATA ANALYSIS")
    print("="*40)
    
    # Basic info
    print(f"\nShape: {orig_df.shape}")
    print(f"\nColumn types:")
    print(orig_df.dtypes)
    
    print("\nFirst 5 rows:")
    print(orig_df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(orig_df.isnull().sum())
    
    # Personality distribution
    if 'Personality' in orig_df.columns:
        print("\nPersonality distribution (original):")
        print(orig_df['Personality'].value_counts(normalize=True))
    
    print("\n" + "="*40)
    print("COMPARISON: ORIGINAL vs SYNTHETIC")
    print("="*40)
    
    # Compare column names
    orig_cols = set(orig_df.columns)
    synth_cols = set(synth_df.columns)
    
    print(f"\nOriginal columns: {sorted(orig_cols)}")
    print(f"\nSynthetic columns: {sorted(synth_cols)}")
    
    print(f"\nColumns in original but not synthetic: {orig_cols - synth_cols}")
    print(f"Columns in synthetic but not original: {synth_cols - orig_cols}")
    
    # Compare common columns
    common_cols = orig_cols & synth_cols
    numeric_cols = []
    
    print("\n" + "="*40)
    print("STATISTICAL COMPARISON")
    print("="*40)
    
    for col in common_cols:
        if col in ['id', 'Personality']:
            continue
            
        if pd.api.types.is_numeric_dtype(orig_df[col]):
            numeric_cols.append(col)
            orig_mean = orig_df[col].mean()
            synth_mean = synth_df[col].mean()
            orig_std = orig_df[col].std()
            synth_std = synth_df[col].std()
            
            print(f"\n{col}:")
            print(f"  Original: mean={orig_mean:.2f}, std={orig_std:.2f}")
            print(f"  Synthetic: mean={synth_mean:.2f}, std={synth_std:.2f}")
            print(f"  Original range: [{orig_df[col].min()}, {orig_df[col].max()}]")
            print(f"  Synthetic range: [{synth_df[col].min()}, {synth_df[col].max()}]")
    
    # Check personality balance
    print("\n" + "="*40)
    print("PERSONALITY BALANCE")
    print("="*40)
    
    if 'Personality' in orig_df.columns:
        orig_intro_pct = (orig_df['Personality'] == 'Introvert').mean() * 100
        synth_intro_pct = (synth_df['Personality'] == 'Introvert').mean() * 100
        
        print(f"Original: {orig_intro_pct:.1f}% Introverts")
        print(f"Synthetic: {synth_intro_pct:.1f}% Introverts")
        print(f"Difference: {abs(orig_intro_pct - synth_intro_pct):.1f}%")
    
    # Look for patterns in original that might explain synthetic
    print("\n" + "="*40)
    print("KEY INSIGHTS")
    print("="*40)
    
    print("""
1. Original dataset characteristics:
   - Size and distribution
   - Feature ranges and types
   - Any obvious patterns or anomalies

2. How synthetic differs from original:
   - Different personality balance?
   - Different feature distributions?
   - Added noise or transformations?

3. Implications for modeling:
   - Should we use original data for training?
   - Can we learn the generation process?
   - Is there information leakage?
""")
    
    return orig_df, synth_df

if __name__ == "__main__":
    orig_df, synth_df = analyze_original_data()