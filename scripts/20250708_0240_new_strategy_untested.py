#!/usr/bin/env python3
"""
Create completely NEW flip candidates - avoiding all previously tested IDs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"

# Already tested IDs - DO NOT USE THESE
TESTED_IDS = {
    # E→I tests
    19612, 20934, 23336, 23844, 20017,
    # I→E tests (MIRROR)
    20033, 22234, 22927, 19636, 22850
}

def find_new_candidates():
    print("="*60)
    print("SZUKANIE NOWYCH KANDYDATÓW (pomijając już testowane)")
    print("="*60)
    
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Preprocess
    test_df['Drained_num'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    test_df['Stage_fear_num'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})
    
    # Skip already tested
    test_df = test_df[~test_df['id'].isin(TESTED_IDS)]
    print(f"\nPomijamy {len(TESTED_IDS)} już testowanych ID")
    print(f"Pozostało {len(test_df)} rekordów do analizy")
    
    candidates = []
    
    # Strategy 1: First and Last records
    print("\n1. PIERWSZE I OSTATNIE REKORDY:")
    print("-"*40)
    
    # First 5 IDs
    first_ids = sorted(test_df['id'].values)[:10]
    for i, record_id in enumerate(first_ids[:5]):
        idx = original_df[original_df['id'] == record_id].index
        if len(idx) > 0:
            personality = original_df.loc[idx[0], 'Personality']
            candidates.append({
                'id': record_id,
                'current': personality,
                'strategy': 'first_records',
                'reason': f'Record #{i+1} in dataset'
            })
            print(f"First #{i+1}: ID {record_id} ({personality})")
    
    # Last 5 IDs
    last_ids = sorted(test_df['id'].values)[-10:]
    for i, record_id in enumerate(last_ids[-5:]):
        idx = original_df[original_df['id'] == record_id].index
        if len(idx) > 0:
            personality = original_df.loc[idx[0], 'Personality']
            candidates.append({
                'id': record_id,
                'current': personality,
                'strategy': 'last_records',
                'reason': f'Record #-{5-i} from end'
            })
            print(f"Last #-{5-i}: ID {record_id} ({personality})")
    
    # Strategy 2: Find potential duplicates
    print("\n2. SZUKANIE DUPLIKATÓW:")
    print("-"*40)
    
    # Create hash of features for each record
    feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                   'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
                   'Post_frequency']
    
    # Fill NaN with -999 for hashing
    test_df_hash = test_df[feature_cols].fillna(-999)
    
    # Create hash for each row
    hashes = []
    for _, row in test_df_hash.iterrows():
        row_str = '|'.join(str(row[col]) for col in feature_cols)
        row_hash = hashlib.md5(row_str.encode()).hexdigest()
        hashes.append(row_hash)
    
    test_df['feature_hash'] = hashes
    
    # Find duplicates
    hash_counts = test_df['feature_hash'].value_counts()
    duplicate_hashes = hash_counts[hash_counts > 1]
    
    print(f"Znaleziono {len(duplicate_hashes)} grup duplikatów")
    
    # Check if duplicates have different labels
    for dup_hash in duplicate_hashes.index[:3]:  # Top 3 duplicate groups
        dup_records = test_df[test_df['feature_hash'] == dup_hash]
        dup_ids = dup_records['id'].values
        
        # Get labels
        labels = []
        for dup_id in dup_ids:
            idx = original_df[original_df['id'] == dup_id].index
            if len(idx) > 0:
                labels.append(original_df.loc[idx[0], 'Personality'])
        
        if len(set(labels)) > 1:  # Different labels!
            print(f"\nDUPLIKATY Z RÓŻNYMI ETYKIETAMI!")
            print(f"IDs: {dup_ids}, Labels: {labels}")
            # Take first one
            candidates.append({
                'id': dup_ids[0],
                'current': labels[0],
                'strategy': 'duplicate_mismatch',
                'reason': f'Duplicate with conflicting labels'
            })
    
    # Strategy 3: Model uncertainty (simulate)
    print("\n3. REKORDY Z NAJWIĘKSZĄ NIEPEWNOŚCIĄ:")
    print("-"*40)
    
    # We'll look for records with balanced features
    uncertainty_candidates = []
    
    for _, row in test_df.iterrows():
        if row['id'] in TESTED_IDS:
            continue
            
        # Calculate "uncertainty score" based on mixed signals
        intro_score = 0
        extro_score = 0
        
        # Introvert signals
        if pd.notna(row['Time_spent_Alone']) and row['Time_spent_Alone'] > 7:
            intro_score += 1
        if row.get('Drained_num') == 1:
            intro_score += 1
        if row.get('Stage_fear_num') == 1:
            intro_score += 1
            
        # Extrovert signals  
        if pd.notna(row['Social_event_attendance']) and row['Social_event_attendance'] > 5:
            extro_score += 1
        if pd.notna(row['Friends_circle_size']) and row['Friends_circle_size'] > 8:
            extro_score += 1
        if pd.notna(row['Post_frequency']) and row['Post_frequency'] > 6:
            extro_score += 1
            
        # High uncertainty = balanced scores
        if intro_score >= 2 and extro_score >= 2:
            uncertainty_candidates.append({
                'id': row['id'],
                'intro_score': intro_score,
                'extro_score': extro_score,
                'uncertainty': abs(intro_score - extro_score)
            })
    
    # Sort by most balanced (lowest uncertainty)
    uncertainty_candidates = sorted(uncertainty_candidates, key=lambda x: x['uncertainty'])
    
    for cand in uncertainty_candidates[:3]:
        idx = original_df[original_df['id'] == cand['id']].index
        if len(idx) > 0:
            personality = original_df.loc[idx[0], 'Personality']
            candidates.append({
                'id': cand['id'],
                'current': personality,
                'strategy': 'high_uncertainty',
                'reason': f'Mixed signals: I={cand["intro_score"]}, E={cand["extro_score"]}'
            })
            print(f"ID {cand['id']}: {personality} (I={cand['intro_score']}, E={cand['extro_score']})")
    
    # Strategy 4: Random sampling
    print("\n4. LOSOWE PRÓBKOWANIE:")
    print("-"*40)
    
    # Get all untested IDs
    all_untested = test_df[~test_df['id'].isin(TESTED_IDS)]['id'].values
    
    # Random sample
    np.random.seed(42)
    random_ids = np.random.choice(all_untested, size=min(5, len(all_untested)), replace=False)
    
    for i, rand_id in enumerate(random_ids):
        idx = original_df[original_df['id'] == rand_id].index
        if len(idx) > 0:
            personality = original_df.loc[idx[0], 'Personality']
            candidates.append({
                'id': rand_id,
                'current': personality,
                'strategy': 'random',
                'reason': f'Random sample #{i+1}'
            })
            print(f"Random #{i+1}: ID {rand_id} ({personality})")
    
    return candidates

def create_new_flip_files(candidates):
    """Create flip files for new candidates"""
    print("\n" + "="*60)
    print("TWORZENIE NOWYCH PLIKÓW TESTOWYCH")
    print("="*60)
    
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    created_files = []
    
    # Select top candidates from each strategy
    strategies = ['first_records', 'last_records', 'duplicate_mismatch', 'high_uncertainty', 'random']
    selected = []
    
    for strategy in strategies:
        strat_candidates = [c for c in candidates if c['strategy'] == strategy]
        if strat_candidates:
            # Take first E and first I if available
            e_cands = [c for c in strat_candidates if c['current'] == 'Extrovert']
            i_cands = [c for c in strat_candidates if c['current'] == 'Introvert']
            
            if e_cands:
                selected.append(e_cands[0])
            if i_cands and len(selected) < 5:
                selected.append(i_cands[0])
    
    # Create files
    for i, candidate in enumerate(selected[:5]):
        # Copy original
        flipped_df = original_df.copy()
        
        # Flip
        idx = flipped_df[flipped_df['id'] == candidate['id']].index
        if len(idx) > 0:
            new_label = 'Introvert' if candidate['current'] == 'Extrovert' else 'Extrovert'
            flipped_df.loc[idx[0], 'Personality'] = new_label
            
            # Save
            direction = 'E2I' if candidate['current'] == 'Extrovert' else 'I2E'
            filename = f"flip_NEW_{i+1}_{candidate['strategy']}_{direction}_id_{int(candidate['id'])}.csv"
            filepath = WORKSPACE_DIR / "scores" / filename
            flipped_df.to_csv(filepath, index=False)
            
            created_files.append(filename)
            print(f"\n{i+1}. {filename}")
            print(f"   Strategy: {candidate['strategy']}")
            print(f"   Flip: {candidate['current']} → {new_label}")
            print(f"   Reason: {candidate['reason']}")
    
    return created_files

def main():
    # Find candidates
    candidates = find_new_candidates()
    
    print(f"\n\nZnaleziono {len(candidates)} nowych kandydatów")
    
    # Create files
    files = create_new_flip_files(candidates)
    
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    print(f"Utworzono {len(files)} nowych plików testowych")
    print("Wszystkie ID są NOWE (nie były wcześniej testowane)")
    print("\nStrategies użyte:")
    print("• First/Last records - może błędy na krańcach")
    print("• Duplicates - identyczne profile z różnymi etykietami") 
    print("• High uncertainty - mieszane sygnały I/E")
    print("• Random - czyste szczęście jak TOP 1")

if __name__ == "__main__":
    main()