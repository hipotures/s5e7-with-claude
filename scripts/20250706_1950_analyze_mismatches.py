#!/usr/bin/env python3
"""
Analyze why we have 98.98% match instead of 100%
Compare original submission with our recreation
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"
RECREATED_SUBMISSION = WORKSPACE_DIR / "scores/recreated_975708_v2.csv"

def analyze_mismatches():
    """Analyze differences between original and recreated submissions"""
    
    # Load both submissions
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    recreated_df = pd.read_csv(RECREATED_SUBMISSION)
    
    print("="*60)
    print("ANALYZING MISMATCHES")
    print("="*60)
    
    # Basic stats
    print(f"\nOriginal submission: {len(original_df)} records")
    print(f"Recreated submission: {len(recreated_df)} records")
    
    # Compare predictions
    mismatches = []
    for i in range(len(original_df)):
        if original_df.iloc[i]['Personality'] != recreated_df.iloc[i]['Personality']:
            mismatches.append({
                'index': i,
                'id': original_df.iloc[i]['id'],
                'original': original_df.iloc[i]['Personality'],
                'recreated': recreated_df.iloc[i]['Personality']
            })
    
    print(f"\nTotal mismatches: {len(mismatches)} ({len(mismatches)/len(original_df)*100:.2f}%)")
    print(f"Matches: {len(original_df) - len(mismatches)} ({(len(original_df) - len(mismatches))/len(original_df)*100:.2f}%)")
    
    # Analyze mismatch patterns
    mismatch_types = {}
    for m in mismatches:
        key = f"{m['original']} → {m['recreated']}"
        if key not in mismatch_types:
            mismatch_types[key] = 0
        mismatch_types[key] += 1
    
    print("\nMismatch patterns:")
    for pattern, count in mismatch_types.items():
        print(f"  {pattern}: {count} cases")
    
    # Show first 10 mismatches
    print("\nFirst 10 mismatches:")
    print("ID | Original | Recreated")
    print("-"*40)
    for m in mismatches[:10]:
        print(f"{m['id']} | {m['original']} | {m['recreated']}")
    
    # Check if mismatches follow a pattern
    mismatch_ids = [m['id'] for m in mismatches]
    
    # Check for sequential patterns
    print("\nChecking for patterns in mismatch IDs...")
    
    # Are they clustered?
    id_diffs = np.diff(sorted(mismatch_ids))
    print(f"Average gap between mismatch IDs: {np.mean(id_diffs):.1f}")
    print(f"Min gap: {np.min(id_diffs)}, Max gap: {np.max(id_diffs)}")
    
    # Check if specific ID ranges
    print("\nID ranges with mismatches:")
    print(f"Min ID: {min(mismatch_ids)}")
    print(f"Max ID: {max(mismatch_ids)}")
    
    # Save mismatch analysis
    mismatch_df = pd.DataFrame(mismatches)
    mismatch_file = WORKSPACE_DIR / "scores/mismatch_analysis.csv"
    mismatch_df.to_csv(mismatch_file, index=False)
    print(f"\nSaved detailed mismatch analysis to: {mismatch_file}")
    
    return mismatches

def check_original_model_hints():
    """Check for hints about the original model"""
    print("\n" + "="*60)
    print("CHECKING ORIGINAL MODEL HINTS")
    print("="*60)
    
    # The hash in filename: 381482788433
    # This might be from Optuna study
    print("\nOriginal filename components:")
    print("- Score: 0.96950 (CV score)")
    print("- Model: xgb")
    print("- Hash: 381482788433 (likely Optuna study hash)")
    print("- Trial: 148 (Optuna trial number)")
    
    print("\nPossible reasons for 98.98% match instead of 100%:")
    print("1. Different random seed was used")
    print("2. Slightly different preprocessing")
    print("3. Different Optuna hyperparameters we haven't discovered")
    print("4. Post-processing rules were slightly different")
    print("5. The 63 mismatches might be the actual 'ambiverts' handled differently")

if __name__ == "__main__":
    mismatches = analyze_mismatches()
    check_original_model_hints()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("98.98% match is actually very good!")
    print("The 63 mismatches (1.02%) might be:")
    print("- Records where the original used a slightly different threshold")
    print("- True ambiverts that were handled with a specific rule")
    print("- Result of different random seed or hyperparameters")
    print("\nFor flip testing purposes, 98.98% recreation should be sufficient!")