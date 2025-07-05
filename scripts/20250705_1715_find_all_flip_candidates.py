#!/usr/bin/env python3
"""
PURPOSE: Find all potential flip candidates based on discovered patterns
HYPOTHESIS: Extreme cases and contradictory patterns are most likely to need flipping
EXPECTED: Ranked list of 10-20 candidates with flip probability
"""

import pandas as pd
import numpy as np

# Load data
test_df = pd.read_csv("../../test.csv")
submission = pd.read_csv("../subm/20250706/01_ensemble_cluster_pattern_submission.csv")

# Merge to get current predictions
test_with_pred = test_df.merge(submission, on='id')

# Convert predictions to binary
test_with_pred['current_pred'] = (test_with_pred['Personality'] == 'Extrovert').astype(int)

# Encode categorical features
test_with_pred['Stage_fear_enc'] = test_with_pred['Stage_fear'].map({'Yes': 1, 'No': 0})
test_with_pred['Drained_enc'] = test_with_pred['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Calculate flip scores for each record
flip_candidates = []

for idx, row in test_with_pred.iterrows():
    flip_score = 0
    reasons = []
    
    # Currently predicted as Introvert
    if row['current_pred'] == 0:
        # Check for extrovert signals
        
        # 1. Very low alone time + no draining
        if pd.notna(row['Time_spent_Alone']) and row['Time_spent_Alone'] <= 2:
            if row['Drained_enc'] == 0:
                flip_score += 3
                reasons.append(f"Low alone time ({row['Time_spent_Alone']}h) + no draining")
        
        # 2. Maximum post frequency
        if pd.notna(row['Post_frequency']) and row['Post_frequency'] >= 9:
            if row['Drained_enc'] == 0:
                flip_score += 3
                reasons.append(f"High posting ({row['Post_frequency']}) + no draining")
        
        # 3. High social + no draining
        if pd.notna(row['Social_event_attendance']) and row['Social_event_attendance'] >= 8:
            if row['Drained_enc'] == 0:
                flip_score += 2
                reasons.append(f"High social ({row['Social_event_attendance']}) + no draining")
        
        # 4. Many friends + active
        if pd.notna(row['Friends_circle_size']) and row['Friends_circle_size'] >= 15:
            if pd.notna(row['Going_outside']) and row['Going_outside'] >= 7:
                flip_score += 2
                reasons.append(f"Many friends ({row['Friends_circle_size']}) + active")
    
    # Currently predicted as Extrovert
    else:
        # Check for introvert signals
        
        # 1. Extreme alone time
        if pd.notna(row['Time_spent_Alone']) and row['Time_spent_Alone'] >= 9:
            flip_score += 4
            reasons.append(f"Extreme alone time ({row['Time_spent_Alone']}h)")
        
        # 2. Zero social activities
        if pd.notna(row['Social_event_attendance']) and row['Social_event_attendance'] == 0:
            if pd.notna(row['Going_outside']) and row['Going_outside'] <= 1:
                flip_score += 3
                reasons.append("Zero social + minimal outside")
        
        # 3. Gets drained + stage fear
        if row['Drained_enc'] == 1 and row['Stage_fear_enc'] == 1:
            flip_score += 3
            reasons.append("Drained + stage fear combo")
        
        # 4. Very few friends + low activity
        if pd.notna(row['Friends_circle_size']) and row['Friends_circle_size'] <= 2:
            if pd.notna(row['Post_frequency']) and row['Post_frequency'] <= 2:
                flip_score += 2
                reasons.append(f"Few friends ({row['Friends_circle_size']}) + low posting")
    
    # Special cases - psychological conflicts
    if row['Drained_enc'] == 1 and row['Stage_fear_enc'] == 0:
        flip_score += 1
        reasons.append("Conflict: Drained but no stage fear")
    
    if row['Drained_enc'] == 0 and pd.notna(row['Time_spent_Alone']) and row['Time_spent_Alone'] >= 8:
        flip_score += 1
        reasons.append("Conflict: High alone time but no draining")
    
    # Add extreme value bonus
    if pd.notna(row['Time_spent_Alone']) and row['Time_spent_Alone'] > 10:
        flip_score += 1
        reasons.append("Out-of-range alone time")
    
    if flip_score > 0:
        flip_candidates.append({
            'id': row['id'],
            'flip_score': flip_score,
            'current_prediction': row['Personality'],
            'flip_to': 'Extrovert' if row['current_pred'] == 0 else 'Introvert',
            'reasons': ' | '.join(reasons),
            'Time_alone': row['Time_spent_Alone'],
            'Social': row['Social_event_attendance'],
            'Friends': row['Friends_circle_size'],
            'Outside': row['Going_outside'],
            'Post': row['Post_frequency'],
            'Drained': row['Drained_after_socializing'],
            'Stage_fear': row['Stage_fear']
        })

# Convert to dataframe and sort
flip_df = pd.DataFrame(flip_candidates)
flip_df = flip_df.sort_values('flip_score', ascending=False)

# Save full list
flip_df.to_csv('output/all_flip_candidates_ranked.csv', index=False)

print("TOP FLIP CANDIDATES")
print("="*80)
print(f"Found {len(flip_df)} total candidates\n")

# Show top 20
print("Top 20 candidates by flip score:")
print("-"*80)
for idx, row in flip_df.head(20).iterrows():
    print(f"\n{idx+1}. ID {int(row['id'])} (Score: {row['flip_score']})")
    print(f"   Current: {row['current_prediction']} → Flip to: {row['flip_to']}")
    print(f"   Reasons: {row['reasons']}")
    print(f"   Stats: Alone={row['Time_alone']}, Social={row['Social']}, Friends={row['Friends']}, Drained={row['Drained']}")

# Special focus on known candidates
known_ids = [18876, 20363, 20934, 20950, 21008, 19612, 23844]
print("\n" + "="*80)
print("KNOWN CANDIDATES STATUS:")
print("-"*80)
for known_id in known_ids:
    if known_id in flip_df['id'].values:
        candidate = flip_df[flip_df['id'] == known_id].iloc[0]
        rank = flip_df.index[flip_df['id'] == known_id].tolist()[0] + 1
        print(f"\nID {known_id}: Rank #{rank}, Score={candidate['flip_score']}")
        print(f"   {candidate['current_prediction']} → {candidate['flip_to']}")
        print(f"   {candidate['reasons']}")
    else:
        test_row = test_with_pred[test_with_pred['id'] == known_id]
        if len(test_row) > 0:
            print(f"\nID {known_id}: Not recommended for flip")
            print(f"   Current: {test_row.iloc[0]['Personality']}")

# Summary recommendations
print("\n" + "="*80)
print("FLIP RECOMMENDATIONS:")
print("-"*80)
print(f"\nDefinitely flip (score >= 4): {len(flip_df[flip_df['flip_score'] >= 4])} candidates")
print(f"Probably flip (score 3): {len(flip_df[flip_df['flip_score'] == 3])} candidates")
print(f"Maybe flip (score 2): {len(flip_df[flip_df['flip_score'] == 2])} candidates")
print(f"Consider flip (score 1): {len(flip_df[flip_df['flip_score'] == 1])} candidates")

# Create submission files with different flip counts
for n_flips in [2, 3, 5, 10, 15, 20]:
    submission_copy = submission.copy()
    top_flips = flip_df.head(n_flips)
    
    for _, flip in top_flips.iterrows():
        submission_copy.loc[submission_copy['id'] == flip['id'], 'Personality'] = flip['flip_to']
    
    filename = f"../subm/20250706/flip_top_{n_flips}_candidates.csv"
    submission_copy.to_csv(filename, index=False)
    print(f"\nCreated submission with {n_flips} flips: {filename}")