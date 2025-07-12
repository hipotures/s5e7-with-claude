#!/usr/bin/env python3
"""
Find test samples similar to ID 11798 (removed hard case)
ID 11798 features:
- Time_spent_Alone: 5.0
- Stage_fear: Yes
- Social_event_attendance: 1.0
- Going_outside: 0.0
- Drained_after_socializing: Yes
- Friends_circle_size: 15.0
- Post_frequency: 0.0
- Label: Introvert (but model thinks Extrovert with 94% confidence)
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

def find_similar_test_samples():
    print("="*60)
    print("FINDING TEST SAMPLES SIMILAR TO ID 11798")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Get ID 11798 features
    target_sample = train_df[train_df['id'] == 11798].iloc[0]
    
    print("\nTarget sample (ID 11798):")
    print(f"Time_spent_Alone: {target_sample['Time_spent_Alone']}")
    print(f"Stage_fear: {target_sample['Stage_fear']}")
    print(f"Social_event_attendance: {target_sample['Social_event_attendance']}")
    print(f"Going_outside: {target_sample['Going_outside']}")
    print(f"Drained_after_socializing: {target_sample['Drained_after_socializing']}")
    print(f"Friends_circle_size: {target_sample['Friends_circle_size']}")
    print(f"Post_frequency: {target_sample['Post_frequency']}")
    print(f"Personality: {target_sample['Personality']}")
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    # Convert binary features
    for df in [train_df, test_df]:
        for col in ['Stage_fear', 'Drained_after_socializing']:
            df[col + '_binary'] = (df[col] == 'Yes').astype(int)
    
    # Use binary versions
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                    'Going_outside', 'Post_frequency', 'Stage_fear_binary', 'Drained_after_socializing_binary']
    
    # Handle missing values
    for col in numeric_cols[:5]:  # Only numeric features
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())
    
    # Prepare target features
    target_features = train_df[train_df['id'] == 11798][numeric_cols].values
    
    # Scale features
    scaler = StandardScaler()
    all_features = pd.concat([train_df[numeric_cols], test_df[numeric_cols]])
    scaler.fit(all_features)
    
    target_scaled = scaler.transform(target_features)
    test_scaled = scaler.transform(test_df[numeric_cols])
    
    # Calculate distances
    distances = euclidean_distances(target_scaled, test_scaled)[0]
    
    # Find closest samples
    test_df['distance'] = distances
    test_df['similarity_score'] = 1 / (1 + distances)  # Convert to similarity
    
    # Sort by similarity
    most_similar = test_df.nsmallest(20, 'distance')
    
    print("\n" + "="*60)
    print("MOST SIMILAR TEST SAMPLES")
    print("="*60)
    
    # Load predictions to see what models think
    try:
        pred_df = pd.read_csv(OUTPUT_DIR / 'test_uncertainty_analysis.csv')
        most_similar = most_similar.merge(pred_df[['id', 'mean_pred', 'std_pred']], on='id', how='left')
    except:
        print("Warning: Could not load prediction data")
    
    print(f"\n{'ID':<8} {'Dist':<8} {'Friends':<8} {'Alone':<8} {'Social':<8} {'Out':<8} {'Post':<8} {'Fear':<6} {'Drain':<6}")
    print("-" * 80)
    
    exact_matches = []
    close_matches = []
    
    for _, row in most_similar.iterrows():
        # Check for exact feature match
        exact_match = (
            row['Time_spent_Alone'] == 5.0 and
            row['Stage_fear'] == 'Yes' and
            row['Social_event_attendance'] == 1.0 and
            row['Going_outside'] == 0.0 and
            row['Drained_after_socializing'] == 'Yes' and
            row['Friends_circle_size'] == 15.0 and
            row['Post_frequency'] == 0.0
        )
        
        if exact_match:
            exact_matches.append(row['id'])
        elif row['distance'] < 0.5:
            close_matches.append(row['id'])
        
        print(f"{row['id']:<8} {row['distance']:<8.3f} {row['Friends_circle_size']:<8.0f} "
              f"{row['Time_spent_Alone']:<8.1f} {row['Social_event_attendance']:<8.1f} "
              f"{row['Going_outside']:<8.1f} {row['Post_frequency']:<8.1f} "
              f"{row['Stage_fear']:<6} {row['Drained_after_socializing']:<6}")
    
    print(f"\nExact matches: {len(exact_matches)}")
    if exact_matches:
        print(f"IDs: {exact_matches}")
    
    print(f"\nVery close matches (distance < 0.5): {len(close_matches)}")
    if close_matches:
        print(f"IDs: {close_matches[:10]}")  # First 10
    
    # Also check for high-friend introverts pattern
    print("\n" + "="*60)
    print("HIGH-FRIEND LOW-SOCIAL PATTERN (like 11798)")
    print("="*60)
    
    pattern_matches = test_df[
        (test_df['Friends_circle_size'] >= 12) &
        (test_df['Social_event_attendance'] <= 2) &
        (test_df['Going_outside'] <= 1) &
        (test_df['Stage_fear'] == 'Yes') &
        (test_df['Drained_after_socializing'] == 'Yes')
    ]
    
    print(f"\nFound {len(pattern_matches)} test samples with similar pattern:")
    print("(Many friends but low social activity + stage fear + drained)")
    
    if len(pattern_matches) > 0:
        print(f"\nTop candidates for flip:")
        for _, row in pattern_matches.head(10).iterrows():
            print(f"ID {row['id']}: Friends={row['Friends_circle_size']:.0f}, "
                  f"Social={row['Social_event_attendance']:.1f}, "
                  f"Out={row['Going_outside']:.1f}, "
                  f"Distance={row['distance']:.3f}")
    
    # Save results
    most_similar.to_csv(OUTPUT_DIR / 'similar_to_11798_test_samples.csv', index=False)
    
    return most_similar, exact_matches, pattern_matches

def main():
    most_similar, exact_matches, pattern_matches = find_similar_test_samples()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if len(exact_matches) > 0:
        print(f"\n🎯 EXACT MATCH FOUND! Test ID {exact_matches[0]} has identical features to 11798")
        print("This is the best candidate for flipping!")
    elif len(pattern_matches) > 0:
        best_candidate = pattern_matches.iloc[0]
        print(f"\n📍 Best pattern match: ID {best_candidate['id']}")
        print("High friends + low social activity + stage fear + drained = likely mislabeled")
    else:
        best_candidate = most_similar.iloc[0]
        print(f"\n🔍 Closest match: ID {best_candidate['id']} (distance: {best_candidate['distance']:.3f})")

if __name__ == "__main__":
    main()