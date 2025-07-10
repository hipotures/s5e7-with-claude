#!/usr/bin/env python3
"""
Create flip files based on model uncertainty and disagreement
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
    20033, 22234, 22927, 19636, 22850,  # I→E tests
    18566, 18524, 22027, 18525, 18534,  # SEQ tests
    18634, 20932, 20140,                # BOUNDARY tests
    18834, 18934, 19034, 19134, 19234   # PATTERN34 tests
}

def analyze_disagreement_candidates():
    """Analyze model disagreement candidates"""
    print("="*60)
    print("ANALIZA KANDYDATÓW Z ROZBIEŻNOŚCIĄ MODELI")
    print("="*60)
    
    # Load candidates
    candidates_df = pd.read_csv(WORKSPACE_DIR / "scripts/output/uncertainty_flip_candidates.csv")
    
    # Focus on model disagreement candidates
    disagreement_candidates = candidates_df[candidates_df['strategy'] == 'model_disagreement'].copy()
    
    print("\nKandydaci z największą rozbieżnością między modelami:")
    print("-"*40)
    
    for _, row in disagreement_candidates.iterrows():
        tested = " [TESTED]" if row['id'] in TESTED_IDS else " [NEW]"
        print(f"ID {int(row['id'])} ({row['current']}): "
              f"disagreement={row['score']:.3f}, avg_prob={row['prob']:.3f}{tested}")
    
    # Filter out already tested
    new_candidates = disagreement_candidates[~disagreement_candidates['id'].isin(TESTED_IDS)]
    
    print(f"\n{len(new_candidates)} nowych kandydatów (nie testowanych)")
    
    return disagreement_candidates, new_candidates

def create_flip_files(candidates):
    """Create flip files for top candidates"""
    print("\n" + "="*60)
    print("TWORZENIE PLIKÓW FLIP")
    print("="*60)
    
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    created_files = []
    
    # Prioritize:
    # 1. New candidates with high disagreement
    # 2. Different current labels (mix E and I)
    
    selected = []
    
    # Get new candidates first
    new_cands = candidates[~candidates['id'].isin(TESTED_IDS)].copy()
    
    # Add untested high disagreement
    for _, cand in new_cands.iterrows():
        if len(selected) < 5:
            selected.append(cand)
    
    # If not enough, add tested ones with highest disagreement
    if len(selected) < 5:
        tested_cands = candidates[candidates['id'].isin(TESTED_IDS)].copy()
        tested_cands = tested_cands.sort_values('score', ascending=False)
        for _, cand in tested_cands.iterrows():
            if len(selected) < 5:
                selected.append(cand)
    
    # Create files
    for i, candidate in enumerate(selected):
        # Copy original
        flipped_df = original_df.copy()
        
        # Determine flip direction
        current = candidate['current']
        new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
        
        # Apply flip
        idx = flipped_df[flipped_df['id'] == candidate['id']].index
        if len(idx) > 0:
            flipped_df.loc[idx[0], 'Personality'] = new_label
            
            # Save
            direction = 'E2I' if current == 'Extrovert' else 'I2E'
            filename = f"flip_UNCERTAIN_{i+1}_{direction}_id_{int(candidate['id'])}.csv"
            filepath = WORKSPACE_DIR / "scores" / filename
            flipped_df.to_csv(filepath, index=False)
            
            created_files.append(filename)
            
            tested_status = "RETESTED" if candidate['id'] in TESTED_IDS else "NEW"
            print(f"\n{i+1}. {filename}")
            print(f"   Status: {tested_status}")
            print(f"   Flip: {current} → {new_label}")
            print(f"   Disagreement: {candidate['score']:.3f}")
            print(f"   Avg probability: {candidate['prob']:.3f}")
    
    return created_files

def analyze_why_catboost_different():
    """Analyze why CatBoost predictions differ so much"""
    print("\n" + "="*60)
    print("DLACZEGO CATBOOST SIĘ RÓŻNI?")
    print("="*60)
    
    print("\nMożliwe przyczyny:")
    print("1. CatBoost inaczej traktuje missing values")
    print("2. CatBoost może wykrywać bardziej złożone interakcje")
    print("3. CatBoost używa ordered boosting - może być wrażliwy na kolejność")
    
    print("\nCo to może oznaczać:")
    print("• Jeśli CatBoost ma rację → te ID są błędnie oznaczone")
    print("• Jeśli XGB/LGB mają rację → CatBoost overfittuje")
    print("• Rozbieżność wskazuje na 'trudne' przypadki")

def main():
    # Analyze candidates
    all_candidates, new_candidates = analyze_disagreement_candidates()
    
    # Create flip files
    if len(all_candidates) > 0:
        files = create_flip_files(all_candidates)
        
        print("\n" + "="*60)
        print("PODSUMOWANIE")
        print("="*60)
        print(f"Utworzono {len(files)} plików")
        print("\nSTRATEGIA: Model disagreement")
        print("• CatBoost vs XGB/LGB disagreement")
        print("• Może wskazywać na prawdziwe błędy")
        print("• Szczególnie ID 19612 i 20017 (już testowane)")
    
    # Analyze why CatBoost differs
    analyze_why_catboost_different()

if __name__ == "__main__":
    main()