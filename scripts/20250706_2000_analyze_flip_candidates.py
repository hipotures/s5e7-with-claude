#!/usr/bin/env python3
"""
Analyze why we can't find good Introvert → Extrovert flip candidates
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")

def main():
    # Load data
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Preprocess
    test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    test_df['Stage_fear'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})
    
    # Fill missing
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 'Going_outside', 'Post_frequency']
    for col in numerical_cols:
        test_df[col] = test_df[col].fillna(test_df[col].mean())
    
    print("ANALYZING INTROVERTS IN ORIGINAL SUBMISSION")
    print("="*60)
    
    introvert_indices = original_df[original_df['Personality'] == 'Introvert'].index
    print(f"\nTotal Introverts: {len(introvert_indices)}")
    
    # Check their features
    extrovert_scores = []
    for idx in introvert_indices:
        features = test_df.iloc[idx]
        
        score = 0
        details = []
        
        if features['Time_spent_Alone'] < 3: 
            score += 1
            details.append(f"alone<3({features['Time_spent_Alone']:.1f})")
        
        if features['Drained_after_socializing'] == 0: 
            score += 1
            details.append("not_drained")
            
        if features['Stage_fear'] == 0: 
            score += 1
            details.append("no_fear")
            
        if features['Social_event_attendance'] > 6: 
            score += 1
            details.append(f"social>6({features['Social_event_attendance']:.1f})")
            
        if features['Friends_circle_size'] > 8: 
            score += 1
            details.append(f"friends>8({features['Friends_circle_size']:.1f})")
        
        extrovert_scores.append({
            'id': original_df.iloc[idx]['id'],
            'index': idx,
            'score': score,
            'details': ', '.join(details) if details else 'no extrovert indicators',
            'time_alone': features['Time_spent_Alone'],
            'social': features['Social_event_attendance'],
            'friends': features['Friends_circle_size']
        })
    
    # Sort by score
    extrovert_scores = sorted(extrovert_scores, key=lambda x: x['score'], reverse=True)
    
    print("\nTop 20 Introverts with Extrovert characteristics:")
    print("ID | Score | Details")
    print("-"*80)
    for item in extrovert_scores[:20]:
        print(f"{item['id']} | {item['score']}/5 | {item['details']}")
    
    # Check distribution of scores
    score_dist = {}
    for item in extrovert_scores:
        score = item['score']
        if score not in score_dist:
            score_dist[score] = 0
        score_dist[score] += 1
    
    print("\nScore distribution for Introverts:")
    for score in sorted(score_dist.keys(), reverse=True):
        print(f"  {score}/5 indicators: {score_dist[score]} people")
    
    # Create simple flip files - just take any 5 Introverts
    print("\n" + "="*60)
    print("CREATING SIMPLE I→E FLIP FILES")
    print("="*60)
    
    # Take first 5 Introverts regardless of score
    for i in range(min(5, len(introvert_indices))):
        idx = introvert_indices[i]
        
        # Copy original
        flipped_df = original_df.copy()
        
        # Flip this one
        flipped_df.loc[idx, 'Personality'] = 'Extrovert'
        
        # Save
        record_id = original_df.iloc[idx]['id']
        filename = f"flip_I2E_simple_{i+1}_id_{int(record_id)}.csv"
        filepath = WORKSPACE_DIR / "scores" / filename
        flipped_df.to_csv(filepath, index=False)
        print(f"Created: {filename}")
    
    print("\nDone! Created 5 simple Introvert→Extrovert flip files")

if __name__ == "__main__":
    main()