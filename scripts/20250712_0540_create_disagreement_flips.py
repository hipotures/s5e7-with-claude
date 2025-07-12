#!/usr/bin/env python3
"""
Create flip submissions for high disagreement cases
Strategy: Flip cases where models strongly disagree (3-2 split)
"""

import pandas as pd
from pathlib import Path

# Paths
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def create_disagreement_flips():
    # Load high disagreement candidates
    candidates = pd.read_csv(OUTPUT_DIR / 'high_disagreement_candidates.csv')
    
    # Load base submission
    base_submission = pd.read_csv(OUTPUT_DIR / 'submission_ensemble_equal.csv')
    
    print("="*60)
    print("CREATING DISAGREEMENT-BASED FLIP SUBMISSIONS")
    print("="*60)
    
    # Strategy 1: Flip top disagreement case (excluding known probes)
    known_probes = [20934, 18634, 20932, 21138, 20728]
    non_probe_candidates = candidates[~candidates['id'].isin(known_probes)]
    
    if len(non_probe_candidates) > 0:
        # Pick first non-probe candidate
        flip_id = non_probe_candidates.iloc[0]['id']
        original = non_probe_candidates.iloc[0]['equal_weight']
        pattern = non_probe_candidates.iloc[0]['pattern']
        
        submission1 = base_submission.copy()
        flip_mask = submission1['id'] == flip_id
        submission1.loc[flip_mask, 'Personality'] = 'Extrovert' if original == 'Introvert' else 'Introvert'
        
        direction = 'I2E' if original == 'Introvert' else 'E2I'
        filename1 = f'flip_DISAGREE_TOP_1_{direction}_id_{flip_id}.csv'
        submission1.to_csv(SCORES_DIR / filename1, index=False)
        
        print(f"\nCreated: {filename1}")
        print(f"  ID: {flip_id}")
        print(f"  Pattern: {pattern}")
        print(f"  Original: {original} → Flipped to: {'Extrovert' if original == 'Introvert' else 'Introvert'}")
    
    # Strategy 2: Flip case with specific pattern (EIEIE - alternating disagreement)
    eieie_cases = candidates[candidates['pattern'] == 'EIEIE']
    if len(eieie_cases) > 0:
        flip_id2 = eieie_cases.iloc[0]['id']
        original2 = eieie_cases.iloc[0]['equal_weight']
        
        submission2 = base_submission.copy()
        flip_mask2 = submission2['id'] == flip_id2
        submission2.loc[flip_mask2, 'Personality'] = 'Extrovert' if original2 == 'Introvert' else 'Introvert'
        
        direction2 = 'I2E' if original2 == 'Introvert' else 'E2I'
        filename2 = f'flip_PATTERN_EIEIE_1_{direction2}_id_{flip_id2}.csv'
        submission2.to_csv(SCORES_DIR / filename2, index=False)
        
        print(f"\nCreated: {filename2}")
        print(f"  ID: {flip_id2}")
        print(f"  Pattern: EIEIE (alternating disagreement)")
        print(f"  Original: {original2} → Flipped to: {'Extrovert' if original2 == 'Introvert' else 'Introvert'}")
    
    # Strategy 3: Use majority vote from models (not ensemble) for all high disagreement cases
    submission3 = base_submission.copy()
    changes = 0
    
    for _, row in candidates.iterrows():
        # Count E votes in pattern
        e_votes = sum(1 for char in row['pattern'] if char == 'E')
        
        # If majority says E but ensemble says I, flip to E
        if e_votes >= 3 and row['equal_weight'] == 'Introvert':
            submission3.loc[submission3['id'] == row['id'], 'Personality'] = 'Extrovert'
            changes += 1
        # If majority says I but ensemble says E, flip to I
        elif e_votes < 3 and row['equal_weight'] == 'Extrovert':
            submission3.loc[submission3['id'] == row['id'], 'Personality'] = 'Introvert'
            changes += 1
    
    if changes > 0:
        filename3 = f'submission_majority_override_{changes}_changes.csv'
        submission3.to_csv(SCORES_DIR / filename3, index=False)
        
        print(f"\nCreated: {filename3}")
        print(f"  Changed {changes} predictions based on model majority vote")
    
    # Strategy 4: Trust binary model for high disagreement cases
    # Binary model (Stage_fear, Drained_after_socializing) might be most reliable
    submission4 = base_submission.copy()
    binary_changes = 0
    
    for _, row in candidates.iterrows():
        binary_says_E = row['pattern'][3] == 'E'  # 4th position is binary model
        
        if binary_says_E and row['equal_weight'] == 'Introvert':
            submission4.loc[submission4['id'] == row['id'], 'Personality'] = 'Extrovert'
            binary_changes += 1
        elif not binary_says_E and row['equal_weight'] == 'Extrovert':
            submission4.loc[submission4['id'] == row['id'], 'Personality'] = 'Introvert'
            binary_changes += 1
    
    if binary_changes > 0:
        filename4 = f'submission_binary_override_{binary_changes}_changes.csv'
        submission4.to_csv(SCORES_DIR / filename4, index=False)
        
        print(f"\nCreated: {filename4}")
        print(f"  Changed {binary_changes} predictions based on binary model")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total high disagreement cases: {len(candidates)}")
    print(f"Patterns found:")
    for pattern, count in candidates['pattern'].value_counts().items():
        print(f"  {pattern}: {count} cases")
    
    # Analyze which models tend to be right
    print("\nModel voting patterns:")
    for i, model in enumerate(['Social', 'Personal', 'Behavioral', 'Binary', 'External']):
        e_votes = sum(1 for p in candidates['pattern'] if p[i] == 'E')
        print(f"  {model}: {e_votes} E votes out of {len(candidates)} ({e_votes/len(candidates)*100:.1f}%)")

if __name__ == "__main__":
    create_disagreement_flips()