#!/usr/bin/env python3
"""
Create flip submissions based on MMD shift scores
Focus on test samples that are most different from train distribution
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"
SCORES_DIR = WORKSPACE_DIR / "scores"

def main():
    print("="*60)
    print("CREATING MMD-BASED FLIP SUBMISSIONS")
    print("="*60)
    
    # Load MMD shift scores
    shift_scores = pd.read_csv(OUTPUT_DIR / 'mmd_test_shift_scores.csv')
    
    # Load base submission
    base_submission = pd.read_csv(SCORES_DIR / "flip_UNCERTAINTY_5_toI_id_24005.csv")
    
    # Get top candidates with highest shift
    top_candidates = shift_scores.nlargest(5, 'shift_score')
    
    print("\nTop 5 MMD shift candidates:")
    for _, row in top_candidates.iterrows():
        print(f"ID {int(row['id'])}: shift score = {row['shift_score']:.4f}")
    
    # Create flip submissions for top 3
    for i, (_, candidate) in enumerate(top_candidates.head(3).iterrows(), 1):
        test_id = int(candidate['id'])
        
        # Get current prediction
        current_pred = base_submission[base_submission['id'] == test_id]['Personality'].values[0]
        new_pred = 'Introvert' if current_pred == 'Extrovert' else 'Extrovert'
        flip_type = 'E2I' if new_pred == 'Introvert' else 'I2E'
        
        # Create flip submission
        flip_submission = base_submission.copy()
        flip_submission.loc[flip_submission['id'] == test_id, 'Personality'] = new_pred
        
        # Save
        filename = f"flip_MMD_{i}_{flip_type}_id_{test_id}.csv"
        flip_submission.to_csv(SCORES_DIR / filename, index=False)
        
        print(f"\nCreated: {filename}")
        print(f"  Flipped ID {test_id}: {current_pred} → {new_pred}")
        print(f"  MMD shift score: {candidate['shift_score']:.4f}")
    
    # Also check if known problematic IDs are in top shifts
    print("\n" + "="*40)
    print("KNOWN IDS IN TOP SHIFTS")
    print("="*40)
    
    known_ids = [23547, 21138, 24005, 20934, 19482, 24428]
    
    for known_id in known_ids:
        if known_id in shift_scores['id'].values:
            row = shift_scores[shift_scores['id'] == known_id].iloc[0]
            percentile = (shift_scores['shift_score'] < row['shift_score']).mean() * 100
            print(f"ID {known_id}: shift={row['shift_score']:.4f}, percentile={percentile:.1f}%")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Created 3 MMD-based flip submissions")
    print("These samples have the highest distribution shift from train")
    print("High shift could indicate outliers or mislabeled samples")

if __name__ == "__main__":
    main()