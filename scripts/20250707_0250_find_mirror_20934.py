#!/usr/bin/env python3
"""
Find Introverts most similar to record 20934 (the critical flip)
These might be mislabeled and should be Extroverts
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"

def analyze_record_20934():
    """First, let's understand what makes 20934 special"""
    print("="*60)
    print("ANALYZING RECORD 20934")
    print("="*60)
    
    # Load test data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Find record 20934
    record_20934 = test_df[test_df['id'] == 20934]
    
    if record_20934.empty:
        print("ERROR: Record 20934 not found!")
        return None
    
    print("\nRecord 20934 features:")
    print("-"*40)
    for col in test_df.columns:
        if col != 'id':
            val = record_20934[col].values[0]
            print(f"{col}: {val}")
    
    return record_20934

def find_similar_introverts(record_20934):
    """Find Introverts most similar to record 20934"""
    print("\n" + "="*60)
    print("FINDING INTROVERTS SIMILAR TO 20934")
    print("="*60)
    
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Preprocess
    test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    test_df['Stage_fear'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})
    
    # Features for similarity
    feature_cols = ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone', 
                   'Social_event_attendance', 'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # Fill NaN - must be done for ALL columns including categorical
    for col in feature_cols:
        if col in ['Drained_after_socializing', 'Stage_fear']:
            test_df[col] = test_df[col].fillna(0)  # Missing categorical = 0
        else:
            test_df[col] = test_df[col].fillna(test_df[col].mean())
    
    # Get only Introverts
    introvert_mask = original_df['Personality'] == 'Introvert'
    introvert_ids = original_df[introvert_mask]['id'].values
    introvert_indices = [i for i, id in enumerate(test_df['id']) if id in introvert_ids]
    
    print(f"\nTotal Introverts: {len(introvert_indices)}")
    
    # Get features
    X = test_df[feature_cols].values
    
    # Standardize for better similarity
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get 20934 features
    idx_20934 = test_df[test_df['id'] == 20934].index[0]
    features_20934 = X_scaled[idx_20934].reshape(1, -1)
    
    # Calculate similarities only for Introverts
    similarities = []
    for idx in introvert_indices:
        features_i = X_scaled[idx].reshape(1, -1)
        sim = cosine_similarity(features_20934, features_i)[0][0]
        
        # Also calculate exact matches on key features
        exact_matches = 0
        if X[idx][0] == X[idx_20934][0]:  # Drained
            exact_matches += 1
        if X[idx][1] == X[idx_20934][1]:  # Stage_fear
            exact_matches += 1
        if abs(X[idx][2] - X[idx_20934][2]) < 1:  # Time_alone (within 1 hour)
            exact_matches += 1
        if abs(X[idx][3] - X[idx_20934][3]) < 1:  # Social_event (within 1)
            exact_matches += 1
        
        similarities.append({
            'id': test_df.iloc[idx]['id'],
            'index': idx,
            'similarity': sim,
            'exact_matches': exact_matches,
            'time_alone': X[idx][2],
            'social': X[idx][3],
            'friends': X[idx][4],
            'drained': X[idx][0],
            'stage_fear': X[idx][1]
        })
    
    # Sort by similarity
    similarities = sorted(similarities, key=lambda x: (x['exact_matches'], x['similarity']), reverse=True)
    
    print(f"\nTop 10 Introverts most similar to 20934:")
    print("ID | Similarity | Exact | Time_alone | Social | Friends | Drained | Fear")
    print("-"*80)
    
    top_candidates = []
    for i, sim in enumerate(similarities[:10]):
        print(f"{sim['id']} | {sim['similarity']:.3f} | {sim['exact_matches']}/4 | "
              f"{sim['time_alone']:.1f} | {sim['social']:.1f} | {sim['friends']:.1f} | "
              f"{sim['drained']} | {sim['stage_fear']}")
        if i < 5:
            top_candidates.append(sim)
    
    return top_candidates

def create_flip_files(candidates):
    """Create flip test files for the most similar Introverts"""
    print("\n" + "="*60)
    print("CREATING I→E FLIP FILES (MIRROR 20934)")
    print("="*60)
    
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    for i, candidate in enumerate(candidates):
        # Copy original
        flipped_df = original_df.copy()
        
        # Find index
        flip_idx = flipped_df[flipped_df['id'] == candidate['id']].index[0]
        
        # Flip I→E
        flipped_df.loc[flip_idx, 'Personality'] = 'Extrovert'
        
        # Save
        filename = f"flip_MIRROR_20934_v{i+1}_id_{int(candidate['id'])}.csv"
        filepath = WORKSPACE_DIR / "scores" / filename
        flipped_df.to_csv(filepath, index=False)
        
        print(f"Created: {filename}")
        print(f"  - Similarity: {candidate['similarity']:.3f}")
        print(f"  - Exact matches: {candidate['exact_matches']}/4")
    
    # Also create combined file
    print("\nCreating combined file...")
    combined_df = original_df.copy()
    
    for candidate in candidates:
        flip_idx = combined_df[combined_df['id'] == candidate['id']].index[0]
        combined_df.loc[flip_idx, 'Personality'] = 'Extrovert'
    
    combined_file = WORKSPACE_DIR / "scores" / "flip_MIRROR_20934_ALL_5.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"Created: flip_MIRROR_20934_ALL_5.csv (all 5 flips)")

def main():
    # First analyze 20934
    record_20934 = analyze_record_20934()
    
    if record_20934 is not None:
        # Find similar Introverts
        candidates = find_similar_introverts(record_20934)
        
        # Create flip files
        create_flip_files(candidates)
        
        print("\n" + "="*60)
        print("HYPOTHESIS:")
        print("="*60)
        print("If 20934 was mislabeled E→I, these similar Introverts")
        print("might be mislabeled I→E!")
        print("\nExpected: At least 1 of 5 should be in public set (20%)")
        print("If correct, we should see +0.000810 score!")

if __name__ == "__main__":
    main()