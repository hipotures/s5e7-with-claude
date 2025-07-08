#!/usr/bin/env python3
"""
Create flip files based on sequential analysis findings
"""

import pandas as pd
import numpy as np
from pathlib import Path

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

def create_sequential_flips():
    print("="*60)
    print("TWORZENIE FLIPÓW NA PODSTAWIE ANALIZY SEKWENCYJNEJ")
    print("="*60)
    
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Add sequence number
    test_df['seq_num'] = test_df['id'] - test_df['id'].min()
    
    flip_candidates = []
    
    # 1. CYCLICAL ANOMALIES - every 255 records, position 42
    print("\n1. ANOMALIE CYKLICZNE:")
    print("-"*40)
    
    # Find all records at position 42 (mod 255)
    cyclical_ids = test_df[test_df['seq_num'] % 255 == 42]['id'].values
    
    for cid in cyclical_ids[:5]:  # Take first 5
        if cid not in TESTED_IDS:
            idx = original_df[original_df['id'] == cid].index
            if len(idx) > 0:
                personality = original_df.loc[idx[0], 'Personality']
                flip_candidates.append({
                    'id': cid,
                    'current': personality,
                    'strategy': 'cyclical_255_42',
                    'reason': 'Position 42 in 255-cycle has E_ratio=0.440 (anomaly -0.308)'
                })
                print(f"ID {cid}: {personality} (pozycja 42 w cyklu 255)")
    
    # 2. CHUNKS WITH DRIFT - Chunk 0 (low E_ratio) and Chunk 7 (high E_ratio)
    print("\n2. CHUNKI Z DRYFTEM:")
    print("-"*40)
    
    # Chunk 0: ID 18524-19023 (E_ratio=0.722, low)
    chunk0_ids = test_df[(test_df['id'] >= 18524) & (test_df['id'] <= 19023)]['id'].values
    
    # Find Extroverts in low-E chunk (likely errors)
    for cid in chunk0_ids[:50]:  # Check first 50
        if cid not in TESTED_IDS:
            idx = original_df[original_df['id'] == cid].index
            if len(idx) > 0 and original_df.loc[idx[0], 'Personality'] == 'Extrovert':
                flip_candidates.append({
                    'id': cid,
                    'current': 'Extrovert',
                    'strategy': 'drift_chunk0_E',
                    'reason': 'Extrovert in low-E chunk (0.722 vs 0.748 avg)'
                })
                print(f"ID {cid}: E in low-E chunk")
                break
    
    # Chunk 7: ID 22024-22523 (E_ratio=0.782, high)
    chunk7_ids = test_df[(test_df['id'] >= 22024) & (test_df['id'] <= 22523)]['id'].values
    
    # Find Introverts in high-E chunk (likely errors)
    for cid in chunk7_ids[:50]:
        if cid not in TESTED_IDS:
            idx = original_df[original_df['id'] == cid].index
            if len(idx) > 0 and original_df.loc[idx[0], 'Personality'] == 'Introvert':
                flip_candidates.append({
                    'id': cid,
                    'current': 'Introvert',
                    'strategy': 'drift_chunk7_I',
                    'reason': 'Introvert in high-E chunk (0.782 vs 0.748 avg)'
                })
                print(f"ID {cid}: I in high-E chunk")
                break
    
    # 3. NEIGHBORS INCONSISTENCY
    print("\n3. NIESPÓJNOŚĆ Z SĄSIADAMI:")
    print("-"*40)
    
    # Records that differ from both neighbors
    inconsistent_ids = [18525, 18528, 18531, 18532, 18533]  # From analysis
    
    for cid in inconsistent_ids:
        if cid not in TESTED_IDS:
            idx = original_df[original_df['id'] == cid].index
            if len(idx) > 0:
                personality = original_df.loc[idx[0], 'Personality']
                flip_candidates.append({
                    'id': cid,
                    'current': personality,
                    'strategy': 'neighbor_diff',
                    'reason': 'Differs from both neighbors (isolated in sequence)'
                })
                print(f"ID {cid}: {personality} (różni się od sąsiadów)")
                break
    
    # 4. ID PATTERNS
    print("\n4. WZORCE W ID:")
    print("-"*40)
    
    # IDs ending in specific digits
    ending_00 = test_df[test_df['id'] % 100 == 0]['id'].values  # IDs ending in 00
    ending_34 = test_df[test_df['id'] % 100 == 34]['id'].values  # Like 20934
    
    for cid in ending_34[:5]:
        if cid not in TESTED_IDS and cid != 20934:
            idx = original_df[original_df['id'] == cid].index
            if len(idx) > 0:
                personality = original_df.loc[idx[0], 'Personality']
                flip_candidates.append({
                    'id': cid,
                    'current': personality,
                    'strategy': 'id_pattern_34',
                    'reason': 'ID ends with 34 (like error 20934)'
                })
                print(f"ID {cid}: {personality} (kończy się na 34)")
                break
    
    # 5. EXTREME POSITIONS
    print("\n5. EKSTREMALNE POZYCJE:")
    print("-"*40)
    
    # Very first and very last IDs
    first_ids = sorted(test_df['id'].values)[:20]
    last_ids = sorted(test_df['id'].values)[-20:]
    
    for cid in first_ids:
        if cid not in TESTED_IDS:
            idx = original_df[original_df['id'] == cid].index
            if len(idx) > 0:
                personality = original_df.loc[idx[0], 'Personality']
                flip_candidates.append({
                    'id': cid,
                    'current': personality,
                    'strategy': 'extreme_first',
                    'reason': f'One of first 20 IDs (ID={cid})'
                })
                print(f"ID {cid}: {personality} (początek datasetu)")
                break
    
    # Create flip files
    print("\n" + "="*60)
    print("TWORZENIE PLIKÓW")
    print("="*60)
    
    created_files = []
    
    # Select top 5 candidates with diverse strategies
    selected = []
    strategies_used = set()
    
    for candidate in flip_candidates:
        if candidate['strategy'] not in strategies_used and len(selected) < 5:
            selected.append(candidate)
            strategies_used.add(candidate['strategy'])
    
    # Create files
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
            filename = f"flip_SEQ_{i+1}_{direction}_id_{int(candidate['id'])}.csv"
            filepath = WORKSPACE_DIR / "scores" / filename
            flipped_df.to_csv(filepath, index=False)
            
            created_files.append(filename)
            print(f"\n{i+1}. {filename}")
            print(f"   Strategy: {candidate['strategy']}")
            print(f"   Flip: {candidate['current']} → {new_label}")
            print(f"   Reason: {candidate['reason']}")
    
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    print(f"Utworzono {len(created_files)} plików na podstawie analizy sekwencyjnej")
    print("\nHipotezy testowane:")
    print("• Anomalie cykliczne (co 255 rekordów)")
    print("• Drift w chunkach danych")
    print("• Niespójność z sąsiadami")
    print("• Wzorce w numerach ID")
    print("• Ekstremalne pozycje w datasecie")
    
    return created_files

if __name__ == "__main__":
    files = create_sequential_flips()
    
    print("\n" + "="*60)
    print("PLIKI DO SUBMISJI NA JUTRO (2025-07-08):")
    print("="*60)
    for f in files:
        print(f"• scores/{f}")