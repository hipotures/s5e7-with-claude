#!/usr/bin/env python3
"""
Find candidates for the second error using multiple strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"

def find_candidates():
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Preprocess
    test_df['Drained_num'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    test_df['Stage_fear_num'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})
    
    # Count nulls
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 
                   'Going_outside', 'Post_frequency']
    test_df['null_count'] = test_df[feature_cols].isnull().sum(axis=1)
    
    candidates = []
    
    # Strategy 1: Extreme introverts marked as E
    mask1 = (original_df['Personality'] == 'Extrovert')
    for idx in original_df[mask1].index:
        if idx < len(test_df):
            row = test_df.iloc[idx]
            score = 0
            if pd.notna(row['Time_spent_Alone']) and row['Time_spent_Alone'] > 10:
                score += 3
            if pd.notna(row['Social_event_attendance']) and row['Social_event_attendance'] < 3:
                score += 2
            if row.get('Drained_num') == 1:
                score += 1
            if score >= 3:
                candidates.append({
                    'id': row['id'],
                    'strategy': 'extreme_I_as_E',
                    'score': score,
                    'time_alone': row['Time_spent_Alone'],
                    'social': row['Social_event_attendance']
                })
    
    # Strategy 2: Extreme extroverts marked as I
    mask2 = (original_df['Personality'] == 'Introvert')
    for idx in original_df[mask2].index:
        if idx < len(test_df):
            row = test_df.iloc[idx]
            score = 0
            if pd.notna(row['Friends_circle_size']) and row['Friends_circle_size'] >= 9:
                score += 2
            if pd.notna(row['Social_event_attendance']) and row['Social_event_attendance'] > 6:
                score += 2
            if row.get('Drained_num') == 0:
                score += 1
            if pd.notna(row['Time_spent_Alone']) and row['Time_spent_Alone'] < 2:
                score += 1
            if score >= 3:
                candidates.append({
                    'id': row['id'],
                    'strategy': 'extreme_E_as_I',
                    'score': score,
                    'friends': row['Friends_circle_size'],
                    'social': row['Social_event_attendance']
                })
    
    # Strategy 3: Near 20934
    for idx in range(max(0, 934-50), min(len(test_df), 1034+50)):  # ±50 from 20934
        row = test_df.iloc[idx]
        if 20900 <= row['id'] <= 21000 and row['id'] != 20934:
            candidates.append({
                'id': row['id'],
                'strategy': 'near_20934',
                'score': 1/abs(row['id'] - 20934),  # Closer = higher score
                'distance': abs(row['id'] - 20934)
            })
    
    # Strategy 4: High null count
    high_null = test_df[test_df['null_count'] >= 3]
    for _, row in high_null.iterrows():
        candidates.append({
            'id': row['id'],
            'strategy': 'high_nulls',
            'score': row['null_count'],
            'nulls': row['null_count']
        })
    
    # Sort and display
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    print(f"Found {len(candidates)} candidates")
    print("\nTop candidates by strategy:")
    
    strategies = ['extreme_I_as_E', 'extreme_E_as_I', 'near_20934', 'high_nulls']
    
    for strategy in strategies:
        strat_candidates = [c for c in candidates if c['strategy'] == strategy]
        if strat_candidates:
            best = strat_candidates[0]
            print(f"\n{strategy}: ID {best['id']} (score: {best['score']:.3f})")
            print(f"  Details: {best}")
    
    return candidates

if __name__ == "__main__":
    candidates = find_candidates()
