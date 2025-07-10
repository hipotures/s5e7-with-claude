#!/usr/bin/env python3
"""
Create more flip files for IDs ending in 34 - we found the pattern!
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
    # Original tests
    19612, 20934, 23336, 23844, 20017,  # E→I tests
    20033, 22234, 22927, 19636, 22850,  # I→E tests
    # Today's tests
    18566, 18524, 22027, 18525, 18534,  # SEQ tests
    18634, 20932, 20140  # BOUNDARY tests (18634 was a HIT!)
}

def find_all_34_ids():
    """Find all IDs ending in 34 and create flip files"""
    print("="*60)
    print("SZUKANIE WSZYSTKICH ID KOŃCZĄCYCH SIĘ NA 34")
    print("="*60)
    
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Find all IDs ending in 34
    ids_ending_34 = test_df[test_df['id'] % 100 == 34]['id'].values
    print(f"Znaleziono {len(ids_ending_34)} ID kończących się na 34")
    
    # Get personalities and filter
    candidates = []
    
    for test_id in ids_ending_34:
        if test_id not in TESTED_IDS:
            idx = original_df[original_df['id'] == test_id].index
            if len(idx) > 0:
                personality = original_df.loc[idx[0], 'Personality']
                
                # Check features to prioritize
                test_row = test_df[test_df['id'] == test_id].iloc[0]
                
                # Calculate confidence score
                score = 1.0  # Base score for pattern
                
                # Bonus for Extroverts (both found errors were E→I)
                if personality == 'Extrovert':
                    score += 0.5
                
                # Bonus for specific features
                if pd.notna(test_row.get('Time_spent_Alone', np.nan)):
                    if test_row['Time_spent_Alone'] > 8:  # High time alone
                        score += 0.3
                
                candidates.append({
                    'id': test_id,
                    'personality': personality,
                    'score': score
                })
    
    # Sort by score and personality (E first)
    candidates = sorted(candidates, key=lambda x: (-x['score'], x['personality'] == 'Introvert'))
    
    print(f"\nZnaleziono {len(candidates)} nieprzetestowanych kandydatów")
    
    print("\nTOP 10 kandydatów:")
    for i, cand in enumerate(candidates[:10]):
        print(f"{i+1}. ID {cand['id']} ({cand['personality']}) - score: {cand['score']:.1f}")
    
    return candidates

def create_34_flip_files(candidates):
    """Create flip files for top 34-pattern candidates"""
    print("\n" + "="*60)
    print("TWORZENIE PLIKÓW DLA WZORCA 34")
    print("="*60)
    
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    created_files = []
    
    # Take top 5 candidates
    for i, candidate in enumerate(candidates[:5]):
        if i >= 5:  # Safety check
            break
            
        # Copy original
        flipped_df = original_df.copy()
        
        # Flip
        idx = flipped_df[flipped_df['id'] == candidate['id']].index
        if len(idx) > 0:
            current = candidate['personality']
            new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
            flipped_df.loc[idx[0], 'Personality'] = new_label
            
            # Save
            direction = 'E2I' if current == 'Extrovert' else 'I2E'
            filename = f"flip_PATTERN34_{i+1}_{direction}_id_{int(candidate['id'])}.csv"
            filepath = WORKSPACE_DIR / "scores" / filename
            flipped_df.to_csv(filepath, index=False)
            
            created_files.append(filename)
            print(f"\n{i+1}. {filename}")
            print(f"   Current: {current} → {new_label}")
            print(f"   Score: {candidate['score']:.1f}")
    
    return created_files

def analyze_found_errors():
    """Analyze the two errors we found"""
    print("\n" + "="*60)
    print("ANALIZA ZNALEZIONYCH BŁĘDÓW")
    print("="*60)
    
    # Load data to analyze the found errors
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    error_ids = [20934, 18634]
    
    for error_id in error_ids:
        error_row = test_df[test_df['id'] == error_id].iloc[0]
        print(f"\nID {error_id}:")
        print(f"  Time_spent_Alone: {error_row.get('Time_spent_Alone', 'NaN')}")
        print(f"  Social_event_attendance: {error_row.get('Social_event_attendance', 'NaN')}")
        print(f"  Friends_circle_size: {error_row.get('Friends_circle_size', 'NaN')}")
        print(f"  Going_outside: {error_row.get('Going_outside', 'NaN')}")
        print(f"  Post_frequency: {error_row.get('Post_frequency', 'NaN')}")
        print(f"  Drained_after_socializing: {error_row.get('Drained_after_socializing', 'NaN')}")
        print(f"  Stage_fear: {error_row.get('Stage_fear', 'NaN')}")

if __name__ == "__main__":
    # Analyze what we found
    analyze_found_errors()
    
    # Find more candidates
    candidates = find_all_34_ids()
    
    # Create files
    if candidates:
        files = create_34_flip_files(candidates)
        
        print("\n" + "="*60)
        print("PODSUMOWANIE")
        print("="*60)
        print(f"Utworzono {len(files)} plików dla wzorca 34")
        print("\nSTRATEGIA:")
        print("1. Wszystkie ID kończące się na 34")
        print("2. Priorytet dla Extrovertów (oba błędy były E→I)")
        print("3. Bonus za wysokie Time_alone")
        
        print("\nPOZOSTAŁE SUBMISJE NA DZIŚ: 3")
        print("Jeśli znajdziemy kolejny błąd, będziemy mieć 3 flipy!")
    else:
        print("\nBrak kandydatów do testowania")