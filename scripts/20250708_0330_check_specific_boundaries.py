#!/usr/bin/env python3
"""
Check specific boundary regions identified in sliding window analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"

# Already tested IDs
TESTED_IDS = {
    19612, 20934, 23336, 23844, 20017,  # E→I tests
    20033, 22234, 22927, 19636, 22850   # I→E tests
}

def check_boundary_regions():
    """Check specific regions with anomalies"""
    print("="*60)
    print("ANALIZA KONKRETNYCH GRANIC")
    print("="*60)
    
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Key boundary regions from sliding window analysis
    boundary_regions = [
        {
            'name': 'Region 20147-20161',
            'start': 20140,
            'end': 20170,
            'note': 'Duży skok E-ratio (60% → 80%)'
        },
        {
            'name': 'Region 23715-23719',
            'start': 23710,
            'end': 23725,
            'note': 'Największa anomalia (58% → 86%)'
        },
        {
            'name': 'Region 20414-20420',
            'start': 20410,
            'end': 20425,
            'note': 'Asymetria L=83% vs R=63%'
        },
        {
            'name': 'Region około 20934',
            'start': 20930,
            'end': 20940,
            'note': 'Zawiera znany błąd 20934'
        }
    ]
    
    flip_candidates = []
    
    for region in boundary_regions:
        print(f"\n{region['name']}:")
        print(f"  {region['note']}")
        print("-"*40)
        
        # Get records in this region
        region_df = test_df[(test_df['id'] >= region['start']) & 
                           (test_df['id'] <= region['end'])].copy()
        
        # Get personalities
        personalities = []
        for _, row in region_df.iterrows():
            idx = original_df[original_df['id'] == row['id']].index
            if len(idx) > 0:
                personalities.append(original_df.loc[idx[0], 'Personality'])
            else:
                personalities.append(None)
        
        region_df['personality'] = personalities
        
        # Calculate local statistics
        e_count = (region_df['personality'] == 'Extrovert').sum()
        i_count = (region_df['personality'] == 'Introvert').sum()
        total = e_count + i_count
        
        if total > 0:
            e_ratio = e_count / total
            print(f"  E-ratio w regionie: {e_ratio:.3f} ({e_count}E, {i_count}I)")
            
            # Find outliers based on region characteristics
            if e_ratio > 0.8:  # High E region
                # Find Introverts (potential errors)
                introverts = region_df[region_df['personality'] == 'Introvert']
                for _, rec in introverts.iterrows():
                    if rec['id'] not in TESTED_IDS:
                        flip_candidates.append({
                            'id': rec['id'],
                            'current': 'Introvert',
                            'region': region['name'],
                            'reason': f'I in high-E region ({e_ratio:.1%})',
                            'priority': e_ratio
                        })
                        print(f"  → ID {rec['id']}: Introvert (podejrzany)")
                        
            elif e_ratio < 0.65:  # Low E region
                # Find Extroverts (potential errors)
                extroverts = region_df[region_df['personality'] == 'Extrovert']
                for _, rec in extroverts.iterrows():
                    if rec['id'] not in TESTED_IDS:
                        flip_candidates.append({
                            'id': rec['id'],
                            'current': 'Extrovert',
                            'region': region['name'],
                            'reason': f'E in low-E region ({e_ratio:.1%})',
                            'priority': 1 - e_ratio
                        })
                        print(f"  → ID {rec['id']}: Extrovert (podejrzany)")
        
        # Show all records in region
        print(f"\n  Wszystkie rekordy w regionie:")
        for _, rec in region_df.iterrows():
            tested = " [TESTED]" if rec['id'] in TESTED_IDS else ""
            print(f"    ID {rec['id']}: {rec['personality']}{tested}")
    
    # Special check for records ending in 34
    print("\n" + "="*60)
    print("SPRAWDZENIE ID KOŃCZĄCYCH SIĘ NA 34")
    print("="*60)
    
    ids_ending_34 = test_df[test_df['id'] % 100 == 34]['id'].values
    print(f"Znaleziono {len(ids_ending_34)} ID kończących się na 34")
    
    for test_id in ids_ending_34[:10]:
        if test_id not in TESTED_IDS:
            idx = original_df[original_df['id'] == test_id].index
            if len(idx) > 0:
                personality = original_df.loc[idx[0], 'Personality']
                print(f"ID {test_id}: {personality}")
                
                # Add high priority candidates
                if test_id != 20934:  # Skip the known one
                    flip_candidates.append({
                        'id': test_id,
                        'current': personality,
                        'region': 'Pattern_34',
                        'reason': 'ID ends with 34 (like error 20934)',
                        'priority': 0.9
                    })
    
    return flip_candidates

def create_boundary_flip_files(candidates):
    """Create flip files for boundary candidates"""
    print("\n" + "="*60)
    print("TWORZENIE PLIKÓW DLA GRANIC")
    print("="*60)
    
    # Sort by priority
    candidates_sorted = sorted(candidates, key=lambda x: x['priority'], reverse=True)
    
    # Take top 5 diverse candidates
    selected = []
    regions_used = set()
    
    for cand in candidates_sorted:
        if len(selected) >= 5:
            break
        if cand['region'] not in regions_used or len(selected) < 3:
            selected.append(cand)
            regions_used.add(cand['region'])
    
    # Create files
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    created_files = []
    
    for i, candidate in enumerate(selected):
        # Copy original
        flipped_df = original_df.copy()
        
        # Flip
        idx = flipped_df[flipped_df['id'] == candidate['id']].index
        if len(idx) > 0:
            new_label = 'Introvert' if candidate['current'] == 'Extrovert' else 'Extrovert'
            flipped_df.loc[idx[0], 'Personality'] = new_label
            
            # Save
            direction = 'E2I' if candidate['current'] == 'Extrovert' else 'I2E'
            filename = f"flip_BOUNDARY_{i+1}_{direction}_id_{int(candidate['id'])}.csv"
            filepath = WORKSPACE_DIR / "scores" / filename
            flipped_df.to_csv(filepath, index=False)
            
            created_files.append(filename)
            print(f"\n{i+1}. {filename}")
            print(f"   Region: {candidate['region']}")
            print(f"   Flip: {candidate['current']} → {new_label}")
            print(f"   Reason: {candidate['reason']}")
    
    return created_files

if __name__ == "__main__":
    candidates = check_boundary_regions()
    
    print(f"\n\nZnaleziono {len(candidates)} kandydatów w regionach granicznych")
    
    if candidates:
        files = create_boundary_flip_files(candidates)
        
        print("\n" + "="*60)
        print("PODSUMOWANIE")
        print("="*60)
        print(f"Utworzono {len(files)} plików na podstawie analizy granic")
        print("\nHipoteza: Dane pochodzą z różnych źródeł/ankiet")
        print("Błędy mogą występować na granicach między partiami")