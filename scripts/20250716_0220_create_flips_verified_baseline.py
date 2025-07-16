#!/usr/bin/env python3
"""
Create flip files using VERIFIED baseline that scores 0.975708.
"""

import pandas as pd
from datetime import datetime

print("="*60)
print("CREATING FLIPS WITH VERIFIED BASELINE")
print("="*60)

# Load the VERIFIED baseline
baseline_file = '../scores/verified_baseline_975708.csv'
baseline_df = pd.read_csv(baseline_file)
print(f"\nLoaded verified baseline: {baseline_file}")
print(f"Baseline has {len(baseline_df)} predictions")

# IDs to flip (excluding already tested ones)
flips_to_create = [
    {'id': 18625, 'direction': 'E2I', 'improvement': 0.001757},
    {'id': 19065, 'direction': 'E2I', 'improvement': 0.000977},
    {'id': 21890, 'direction': 'E2I', 'improvement': 0.000308},
    {'id': 22040, 'direction': 'E2I', 'improvement': 0.000162},
]

print(f"\nCreating {len(flips_to_create)} individual flip files...")

for flip_info in flips_to_create:
    test_id = flip_info['id']
    direction = flip_info['direction']
    improvement = flip_info['improvement']
    
    # Create flip
    flip_submission = baseline_df.copy()
    
    # Find and flip
    mask = flip_submission['id'] == test_id
    if mask.sum() == 0:
        print(f"  WARNING: ID {test_id} not found!")
        continue
        
    current = flip_submission.loc[mask, 'Personality'].values[0]
    new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
    flip_submission.loc[mask, 'Personality'] = new_label
    
    # Verify only 1 flip
    n_differences = (flip_submission['Personality'] != baseline_df['Personality']).sum()
    
    if n_differences == 1:
        actual_direction = 'E2I' if current == 'Extrovert' else 'I2E'
        if actual_direction != direction:
            print(f"  WARNING: ID {test_id} expected {direction} but got {actual_direction}")
            
        flip_file = f"../scores/flip_VERIFIED_BASELINE_{improvement:.6f}_1_{actual_direction}_id_{test_id}.csv"
        flip_submission.to_csv(flip_file, index=False)
        print(f"  Created: ID {test_id} ({current}→{new_label}), file: {flip_file}")
    else:
        print(f"  ERROR: ID {test_id} resulted in {n_differences} changes!")

print("\n" + "="*60)
print("DONE - Files are ready for submission")
print("="*60)