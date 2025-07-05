#!/usr/bin/env python3
"""
PURPOSE: Create corrected training datasets by fixing suspected labeling errors
HYPOTHESIS: Correcting mislabeled training data will improve model performance
EXPECTED: Multiple corrected datasets with different correction strategies
"""

import pandas as pd
import numpy as np

print("="*80)
print("CREATING CORRECTED TRAINING DATASETS")
print("="*80)

# Load original training data
train_df = pd.read_csv("../../train.csv")
print(f"Original training samples: {len(train_df)}")

# Encode features for analysis
train_encoded = train_df.copy()
train_encoded['Stage_fear_enc'] = train_encoded['Stage_fear'].map({'Yes': 1, 'No': 0})
train_encoded['Drained_enc'] = train_encoded['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Calculate typicality scores
def calculate_typicality(row):
    if row['Personality'] == 'Introvert':
        score = (
            (row['Time_spent_Alone'] if pd.notna(row['Time_spent_Alone']) else 5) * 2 +
            (10 - (row['Social_event_attendance'] if pd.notna(row['Social_event_attendance']) else 5)) +
            (30 - (row['Friends_circle_size'] if pd.notna(row['Friends_circle_size']) else 15)) / 3 +
            (row['Drained_enc'] if pd.notna(row['Drained_enc']) else 0.5) * 5 +
            (row['Stage_fear_enc'] if pd.notna(row['Stage_fear_enc']) else 0.5) * 3
        )
    else:
        score = (
            (10 - (row['Time_spent_Alone'] if pd.notna(row['Time_spent_Alone']) else 5)) * 2 +
            (row['Social_event_attendance'] if pd.notna(row['Social_event_attendance']) else 5) +
            (row['Friends_circle_size'] if pd.notna(row['Friends_circle_size']) else 15) / 3 +
            (1 - (row['Drained_enc'] if pd.notna(row['Drained_enc']) else 0.5)) * 5 +
            (1 - (row['Stage_fear_enc'] if pd.notna(row['Stage_fear_enc']) else 0.5)) * 3
        )
    return score

train_encoded['typicality_score'] = train_encoded.apply(calculate_typicality, axis=1)

print("\n" + "="*60)
print("IDENTIFYING CORRECTION CANDIDATES")
print("="*60)

# 1. Extreme introverts labeled as extroverts (78 cases)
extreme_intro_as_extro = train_encoded[
    (train_encoded['Personality'] == 'Extrovert') &
    (train_encoded['Time_spent_Alone'] >= 8) &
    (train_encoded['Social_event_attendance'] <= 2) &
    (train_encoded['Friends_circle_size'] <= 5)
].copy()
extreme_intro_as_extro['correction_reason'] = 'extreme_introvert_mislabeled'
print(f"1. Extreme introverts labeled as Extrovert: {len(extreme_intro_as_extro)}")

# 2. Extreme extroverts labeled as introverts (3 cases)
extreme_extro_as_intro = train_encoded[
    (train_encoded['Personality'] == 'Introvert') &
    (train_encoded['Time_spent_Alone'] <= 2) &
    (train_encoded['Social_event_attendance'] >= 8) &
    (train_encoded['Friends_circle_size'] >= 15) &
    (train_encoded['Drained_enc'] == 0)
].copy()
extreme_extro_as_intro['correction_reason'] = 'extreme_extrovert_mislabeled'
print(f"2. Extreme extroverts labeled as Introvert: {len(extreme_extro_as_intro)}")

# 3. Most egregious cases (typicality < 0)
egregious_cases = train_encoded[
    (train_encoded['typicality_score'] < 0) & 
    (train_encoded['Personality'] == 'Extrovert')
].copy()
egregious_cases['correction_reason'] = 'negative_typicality_score'
print(f"3. Egregious cases (typicality < 0): {len(egregious_cases)}")

# 4. Psychological contradictions - more nuanced analysis
psych_contradictions = train_encoded[
    ((train_encoded['Personality'] == 'Introvert') & 
     (train_encoded['Drained_enc'] == 0) & 
     (train_encoded['Stage_fear_enc'] == 0) &
     (train_encoded['Time_spent_Alone'] <= 3) &
     (train_encoded['Social_event_attendance'] >= 7)) |
    ((train_encoded['Personality'] == 'Extrovert') & 
     (train_encoded['Drained_enc'] == 1) & 
     (train_encoded['Stage_fear_enc'] == 1) &
     (train_encoded['Time_spent_Alone'] >= 7) &
     (train_encoded['Social_event_attendance'] <= 3))
].copy()
psych_contradictions['correction_reason'] = 'psychological_contradiction'
print(f"4. Strong psychological contradictions: {len(psych_contradictions)}")

# 5. 11-hour anomalies labeled as extroverts
eleven_hour_extroverts = train_encoded[
    (train_encoded['Time_spent_Alone'] == 11) &
    (train_encoded['Personality'] == 'Extrovert')
].copy()
eleven_hour_extroverts['correction_reason'] = 'eleven_hour_extrovert'
print(f"5. 11-hour extroverts: {len(eleven_hour_extroverts)}")

print("\n" + "="*60)
print("CREATING CORRECTED DATASETS")
print("="*60)

# Function to create corrected dataset
def create_corrected_dataset(train_df, ids_to_flip, dataset_name, description):
    corrected_df = train_df.copy()
    
    # Flip personalities for specified IDs
    flip_mask = corrected_df['id'].isin(ids_to_flip)
    corrected_df.loc[flip_mask & (corrected_df['Personality'] == 'Introvert'), 'Personality'] = 'Extrovert'
    corrected_df.loc[flip_mask & (corrected_df['Personality'] == 'Extrovert'), 'Personality'] = 'Introvert'
    
    # Save dataset
    output_path = f'output/{dataset_name}'
    corrected_df.to_csv(output_path, index=False)
    
    flipped_count = flip_mask.sum()
    print(f"\n{dataset_name}:")
    print(f"  Description: {description}")
    print(f"  Corrections: {flipped_count}")
    print(f"  Saved to: {output_path}")
    
    return corrected_df, flipped_count

# Create datasets
datasets_info = []

# Dataset 1: Only the 78 extreme introverts
ids_78 = extreme_intro_as_extro['id'].tolist()
_, count1 = create_corrected_dataset(
    train_df, ids_78, 
    'train_corrected_01.csv',
    'Corrected 78 extreme introverts mislabeled as extroverts'
)
datasets_info.append({
    'file': 'train_corrected_01.csv',
    'corrections': count1,
    'description': 'Fixed 78 extreme introverts (8+ hours alone, ≤2 social, ≤5 friends) wrongly labeled as Extrovert'
})

# Dataset 2: 78 + 3 extreme extroverts
ids_81 = ids_78 + extreme_extro_as_intro['id'].tolist()
_, count2 = create_corrected_dataset(
    train_df, ids_81,
    'train_corrected_02.csv',
    'Corrected 78 extreme introverts + 3 extreme extroverts'
)
datasets_info.append({
    'file': 'train_corrected_02.csv',
    'corrections': count2,
    'description': 'Fixed 78 extreme introverts + 3 extreme extroverts (≤2 alone, 8+ social, 15+ friends, no draining)'
})

# Dataset 3: Only the most egregious cases (typicality < 0)
ids_egregious = egregious_cases['id'].tolist()
_, count3 = create_corrected_dataset(
    train_df, ids_egregious,
    'train_corrected_03.csv',
    'Corrected most egregious cases with negative typicality'
)
datasets_info.append({
    'file': 'train_corrected_03.csv',
    'corrections': count3,
    'description': 'Fixed cases with negative typicality scores (mathematically impossible classifications)'
})

# Dataset 4: Strong psychological contradictions
ids_psych = psych_contradictions['id'].tolist()
_, count4 = create_corrected_dataset(
    train_df, ids_psych,
    'train_corrected_04.csv',
    'Corrected strong psychological contradictions'
)
datasets_info.append({
    'file': 'train_corrected_04.csv',
    'corrections': count4,
    'description': 'Fixed strong psychological contradictions (intro without typical intro traits, extro with all intro traits)'
})

# Dataset 5: All 11-hour extroverts
ids_eleven = eleven_hour_extroverts['id'].tolist()
_, count5 = create_corrected_dataset(
    train_df, ids_eleven,
    'train_corrected_05.csv',
    'Corrected all 11-hour extroverts'
)
datasets_info.append({
    'file': 'train_corrected_05.csv',
    'corrections': count5,
    'description': 'Fixed all 34 cases with 11 hours alone labeled as Extrovert (out-of-range value)'
})

# Dataset 6: Combined conservative (egregious + psych contradictions)
ids_conservative = list(set(ids_egregious + ids_psych))
_, count6 = create_corrected_dataset(
    train_df, ids_conservative,
    'train_corrected_06.csv',
    'Conservative corrections (egregious + psychological)'
)
datasets_info.append({
    'file': 'train_corrected_06.csv',
    'corrections': count6,
    'description': 'Conservative approach: only the most obvious errors (negative typicality + strong contradictions)'
})

# Dataset 7: Comprehensive (all identified issues)
ids_comprehensive = list(set(ids_78 + extreme_extro_as_intro['id'].tolist() + 
                            ids_egregious + ids_psych + ids_eleven))
_, count7 = create_corrected_dataset(
    train_df, ids_comprehensive,
    'train_corrected_07.csv',
    'Comprehensive corrections (all identified issues)'
)
datasets_info.append({
    'file': 'train_corrected_07.csv',
    'corrections': count7,
    'description': 'Comprehensive: all identified labeling errors combined'
})

# Dataset 8: Ultra-conservative (only IDs 1041, 13225, 18437)
ids_ultra = [1041, 13225, 18437]
_, count8 = create_corrected_dataset(
    train_df, ids_ultra,
    'train_corrected_08.csv',
    'Ultra-conservative: only 3 most obvious errors'
)
datasets_info.append({
    'file': 'train_corrected_08.csv',
    'corrections': count8,
    'description': 'Ultra-conservative: only the 3 most egregious cases (11h alone, 0 social, 0 friends, drained, stage fear)'
})

# Save detailed analysis
print("\n" + "="*60)
print("SAVING DETAILED ANALYSIS")
print("="*60)

# Combine all suspicious cases with details
all_suspicious = pd.concat([
    extreme_intro_as_extro[['id', 'correction_reason']],
    extreme_extro_as_intro[['id', 'correction_reason']],
    egregious_cases[['id', 'correction_reason']],
    psych_contradictions[['id', 'correction_reason']],
    eleven_hour_extroverts[['id', 'correction_reason']]
]).drop_duplicates(subset=['id'])

# Merge with full data
suspicious_details = train_encoded.merge(all_suspicious, on='id', how='inner')
suspicious_details = suspicious_details.sort_values('typicality_score')
suspicious_details.to_csv('output/training_corrections_detailed.csv', index=False)
print(f"Saved detailed analysis: output/training_corrections_detailed.csv")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total unique suspicious cases: {len(all_suspicious)}")
print(f"Percentage of training data: {len(all_suspicious)/len(train_df)*100:.2f}%")
print(f"Created {len(datasets_info)} corrected datasets")

# Return info for report
print("\nDatasets created successfully!")
print("Run create_corrected_datasets_report.py next to generate the report.")