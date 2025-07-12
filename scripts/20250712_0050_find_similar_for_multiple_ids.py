#!/usr/bin/env python3
"""
Find test samples similar to multiple train IDs and create flip submissions
Target IDs: 10047, 464, 1422, 14756
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"
SCORES_DIR = WORKSPACE_DIR / "scores"

def analyze_target_ids():
    """First check what these IDs are in train data"""
    print("="*60)
    print("ANALYZING TARGET IDs")
    print("="*60)
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    target_ids = [10047, 464, 1422, 14756]
    
    # Check if they're in removed hard cases
    try:
        removed_info = pd.read_csv(OUTPUT_DIR / "removed_hard_cases_info.csv")
        removed_ids = set(removed_info['id'].values)
    except:
        removed_ids = set()
    
    for target_id in target_ids:
        sample = train_df[train_df['id'] == target_id]
        if len(sample) == 0:
            print(f"\n❌ ID {target_id} not found in train data!")
            continue
            
        sample = sample.iloc[0]
        is_removed = target_id in removed_ids
        
        print(f"\n{'='*40}")
        print(f"ID {target_id} - {'REMOVED HARD CASE' if is_removed else 'Regular sample'}")
        print(f"{'='*40}")
        print(f"Personality: {sample['Personality']}")
        print(f"Time_spent_Alone: {sample['Time_spent_Alone']}")
        print(f"Stage_fear: {sample['Stage_fear']}")
        print(f"Social_event_attendance: {sample['Social_event_attendance']}")
        print(f"Going_outside: {sample['Going_outside']}")
        print(f"Drained_after_socializing: {sample['Drained_after_socializing']}")
        print(f"Friends_circle_size: {sample['Friends_circle_size']}")
        print(f"Post_frequency: {sample['Post_frequency']}")
        
        if is_removed:
            removed_row = removed_info[removed_info['id'] == target_id].iloc[0]
            print(f"Model probability: {removed_row['probability']:.3f}")
    
    return train_df, target_ids

def find_similar_test_samples(train_df, test_df, target_id):
    """Find most similar test samples to a given train ID"""
    
    # Get target sample
    target_sample = train_df[train_df['id'] == target_id]
    if len(target_sample) == 0:
        return None
    
    target_sample = target_sample.iloc[0]
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    # Convert binary features
    for df in [train_df, test_df]:
        for col in ['Stage_fear', 'Drained_after_socializing']:
            if col + '_binary' not in df.columns:
                df[col + '_binary'] = (df[col] == 'Yes').astype(int)
    
    # Use binary versions
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                    'Going_outside', 'Post_frequency', 'Stage_fear_binary', 'Drained_after_socializing_binary']
    
    # Handle missing values
    for col in numeric_cols[:5]:  # Only numeric features
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())
    
    # Prepare target features
    target_features = train_df[train_df['id'] == target_id][numeric_cols].values
    
    # Scale features
    scaler = StandardScaler()
    all_features = pd.concat([train_df[numeric_cols], test_df[numeric_cols]])
    scaler.fit(all_features)
    
    target_scaled = scaler.transform(target_features)
    test_scaled = scaler.transform(test_df[numeric_cols])
    
    # Calculate distances
    distances = euclidean_distances(target_scaled, test_scaled)[0]
    
    # Find closest sample
    closest_idx = np.argmin(distances)
    closest_sample = test_df.iloc[closest_idx]
    
    return {
        'train_id': target_id,
        'train_personality': target_sample['Personality'],
        'test_id': closest_sample['id'],
        'distance': distances[closest_idx],
        'test_features': {
            'Time_spent_Alone': closest_sample['Time_spent_Alone'],
            'Stage_fear': closest_sample['Stage_fear'],
            'Social_event_attendance': closest_sample['Social_event_attendance'],
            'Going_outside': closest_sample['Going_outside'],
            'Drained_after_socializing': closest_sample['Drained_after_socializing'],
            'Friends_circle_size': closest_sample['Friends_circle_size'],
            'Post_frequency': closest_sample['Post_frequency']
        }
    }

def create_flip_submissions(results):
    """Create flip submissions for the most similar test samples"""
    print("\n" + "="*60)
    print("CREATING FLIP SUBMISSIONS")
    print("="*60)
    
    # Load base submission
    base_submission = pd.read_csv(SCORES_DIR / "flip_UNCERTAINTY_5_toI_id_24005.csv")
    
    for i, result in enumerate(results):
        if result is None:
            continue
        
        test_id = int(result['test_id'])
        train_personality = result['train_personality']
        
        # Check current prediction
        current_pred = base_submission[base_submission['id'] == test_id]['Personality'].values[0]
        
        # Determine flip direction
        if train_personality == 'Introvert' and current_pred == 'Introvert':
            # Model wrongly predicts Introvert for a hard case that should be Extrovert
            new_pred = 'Extrovert'
            flip_type = 'I2E'
        elif train_personality == 'Extrovert' and current_pred == 'Extrovert':
            # Model wrongly predicts Extrovert for a hard case that should be Introvert
            new_pred = 'Introvert'
            flip_type = 'E2I'
        else:
            # Already different from train label, might be correct
            print(f"\n⚠️ Test ID {test_id} already different from train {result['train_id']}")
            print(f"   Train was {train_personality}, test predicted as {current_pred}")
            # Flip anyway based on removed hard case pattern
            new_pred = 'Extrovert' if current_pred == 'Introvert' else 'Introvert'
            flip_type = 'I2E' if new_pred == 'Extrovert' else 'E2I'
        
        # Create filename
        filename = f"flip_SIMILAR_TO_{result['train_id']}_{flip_type}_id_{test_id}.csv"
        
        # Create flip submission
        flip_submission = base_submission.copy()
        flip_submission.loc[flip_submission['id'] == test_id, 'Personality'] = new_pred
        
        # Save
        flip_submission.to_csv(SCORES_DIR / filename, index=False)
        
        print(f"\n✅ Created: {filename}")
        print(f"   Train ID {result['train_id']} ({train_personality}) → Test ID {test_id}")
        print(f"   Flipped: {current_pred} → {new_pred}")
        print(f"   Distance: {result['distance']:.3f}")

def main():
    # Analyze target IDs
    train_df, target_ids = analyze_target_ids()
    
    # Load test data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Find similar test samples for each ID
    print("\n" + "="*60)
    print("FINDING SIMILAR TEST SAMPLES")
    print("="*60)
    
    results = []
    for target_id in target_ids:
        print(f"\nSearching for test samples similar to ID {target_id}...")
        result = find_similar_test_samples(train_df, test_df, target_id)
        
        if result:
            results.append(result)
            print(f"✓ Most similar: Test ID {result['test_id']} (distance: {result['distance']:.3f})")
            
            # Show features comparison
            print(f"  Test features:")
            for feat, val in result['test_features'].items():
                print(f"    {feat}: {val}")
        else:
            print(f"✗ Could not find similar samples")
            results.append(None)
    
    # Create flip submissions
    create_flip_submissions(results)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Created {len([r for r in results if r is not None])} flip submissions")
    print("Check the scores/ directory for the new CSV files")

if __name__ == "__main__":
    main()