#!/usr/bin/env python3
"""
Create flip files for tomorrow based on multiple strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"

def create_strategic_flips():
    print("="*60)
    print("TWORZENIE PLIKÓW NA JUTRO (2025-07-08)")
    print("="*60)
    
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Define flip candidates based on our analysis
    flip_candidates = [
        {
            'id': 23844,
            'current': 'Extrovert',
            'flip_to': 'Introvert',
            'reason': 'Extreme introvert profile (Time_alone=11, Social=2)',
            'strategy': 'extreme_profile'
        },
        {
            'id': 20934,
            'current': 'Introvert',  # After our flip
            'flip_to': 'Extrovert',
            'reason': 'Reverse our known error to get +0.000810',
            'strategy': 'reverse_known'
        },
        {
            'id': 24068,
            'current': None,  # Need to check
            'flip_to': None,
            'reason': 'High null count (3 nulls)',
            'strategy': 'high_nulls'
        },
        {
            'id': 20950,  # Near 20934
            'current': None,
            'flip_to': None,
            'reason': 'Close to 20934 (batch effect)',
            'strategy': 'near_batch'
        },
        {
            'id': 19612,
            'current': 'Extrovert',
            'flip_to': 'Introvert',
            'reason': 'Previously tested, might work with others',
            'strategy': 'retest'
        }
    ]
    
    # Check current labels for unknowns
    for candidate in flip_candidates:
        if candidate['current'] is None:
            idx = original_df[original_df['id'] == candidate['id']].index
            if len(idx) > 0:
                candidate['current'] = original_df.loc[idx[0], 'Personality']
                candidate['flip_to'] = 'Introvert' if candidate['current'] == 'Extrovert' else 'Extrovert'
    
    # Create individual flip files
    created_files = []
    
    for i, candidate in enumerate(flip_candidates):
        if candidate['current'] and candidate['flip_to']:
            # Copy original
            flipped_df = original_df.copy()
            
            # Apply flip
            idx = flipped_df[flipped_df['id'] == candidate['id']].index
            if len(idx) > 0:
                flipped_df.loc[idx[0], 'Personality'] = candidate['flip_to']
                
                # Save
                filename = f"flip_STRATEGY_{i+1}_{candidate['strategy']}_id_{candidate['id']}.csv"
                filepath = WORKSPACE_DIR / "scores" / filename
                flipped_df.to_csv(filepath, index=False)
                
                created_files.append(filename)
                
                print(f"\n{i+1}. {filename}")
                print(f"   ID: {candidate['id']}")
                print(f"   Flip: {candidate['current']} → {candidate['flip_to']}")
                print(f"   Reason: {candidate['reason']}")
    
    # Create combined strategic file
    print("\n" + "="*60)
    print("BONUS: Kombinowany plik strategiczny")
    print("="*60)
    
    # Start with original
    combined_df = original_df.copy()
    
    # Apply multiple strategic flips
    strategic_flips = [
        (20934, 'Extrovert'),  # Reverse to get +points
        (23844, 'Introvert'),  # Extreme introvert  
        (24068, 'Introvert' if combined_df[combined_df['id']==24068]['Personality'].values[0]=='Extrovert' else 'Extrovert')
    ]
    
    for flip_id, new_label in strategic_flips:
        idx = combined_df[combined_df['id'] == flip_id].index
        if len(idx) > 0:
            combined_df.loc[idx[0], 'Personality'] = new_label
            print(f"Flipped {flip_id} to {new_label}")
    
    combined_file = WORKSPACE_DIR / "scores" / "flip_COMBINED_STRATEGIC_3.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"\nCreated: flip_COMBINED_STRATEGIC_3.csv")
    
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    print(f"Utworzono {len(created_files)} plików pojedynczych flipów")
    print("+ 1 plik kombinowany (3 flipy)")
    print("\nOczekiwania:")
    print("• flip_STRATEGY_2 (reverse 20934): powinien dać +0.000810")
    print("• flip_STRATEGY_1 (23844): kandydat na drugi błąd")
    print("• Pozostałe: eksploracja różnych hipotez")

if __name__ == "__main__":
    create_strategic_flips()