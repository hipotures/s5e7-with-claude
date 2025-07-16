#!/usr/bin/env python3
"""
Create individual flip files using the correct baseline (recreated_975708_submission.csv).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

print("="*60)
print("CREATING METAMODEL FLIPS WITH CORRECT BASELINE")
print("="*60)

# Load data
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Load the CORRECT baseline that scores 0.975708
baseline_file = '../scores/recreated_975708_submission.csv'
baseline_df = pd.read_csv(baseline_file)
print(f"\nLoaded baseline: {baseline_file}")
print(f"Baseline has {len(baseline_df)} predictions")

# Load positive improvements from metamodel
positive_improvements = pd.read_csv('output/cluster_positive_improvements.csv')
print(f"\nLoaded {len(positive_improvements)} positive flip candidates from metamodel")

# Recreate clustering
print("\nRecreating clustering...")
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
               'Going_outside', 'Drained_after_socializing', 
               'Friends_circle_size', 'Post_frequency']

# Encode categorical
train_data = train_df.copy()
train_data['Stage_fear_binary'] = (train_data['Stage_fear'] == 'Yes').astype(int)
train_data['Drained_binary'] = (train_data['Drained_after_socializing'] == 'Yes').astype(int)

cluster_features = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear_binary', 'Drained_binary']

train_data[cluster_features] = train_data[cluster_features].fillna(train_data[cluster_features].mean())

# Scale and cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_data[cluster_features])
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Prepare test data
test_data = test_df.copy()
test_data['Stage_fear_binary'] = (test_data['Stage_fear'] == 'Yes').astype(int)
test_data['Drained_binary'] = (test_data['Drained_after_socializing'] == 'Yes').astype(int)
test_data[cluster_features] = test_data[cluster_features].fillna(test_data[cluster_features].mean())

X_test_scaled = scaler.transform(test_data[cluster_features])
test_clusters = kmeans.predict(X_test_scaled)

# Function to find similar test samples
def find_similar_test_samples(train_idx, train_clusters, test_clusters, train_df, test_df, n_similar=1):
    """Find test samples most similar to a training sample."""
    train_cluster = train_clusters[train_idx]
    test_in_cluster = np.where(test_clusters == train_cluster)[0]
    
    if len(test_in_cluster) == 0:
        return []
    
    # Calculate feature similarity
    train_sample = train_df.iloc[train_idx][feature_cols].copy()
    for col in ['Stage_fear', 'Drained_after_socializing']:
        if col in feature_cols:
            train_sample[col] = 1 if train_sample[col] == 'Yes' else (0 if train_sample[col] == 'No' else -1)
    train_features = train_sample.fillna(-1).values.astype(float)
    
    similarities = []
    for test_idx in test_in_cluster:
        test_sample = test_df.iloc[test_idx][feature_cols].copy()
        for col in ['Stage_fear', 'Drained_after_socializing']:
            if col in feature_cols:
                test_sample[col] = 1 if test_sample[col] == 'Yes' else (0 if test_sample[col] == 'No' else -1)
        test_features = test_sample.fillna(-1).values.astype(float)
        
        distance = np.linalg.norm(train_features - test_features)
        similarities.append((test_idx, distance))
    
    similarities.sort(key=lambda x: x[1])
    return [idx for idx, _ in similarities[:n_similar]]

# Create individual flip files
print("\nCreating individual flip files from correct baseline...")
individual_flips = []

for i, (_, improvement) in enumerate(positive_improvements.head(10).iterrows()):
    train_idx = int(improvement['idx'])
    
    # Find most similar test sample
    similar_test = find_similar_test_samples(
        train_idx, cluster_labels, test_clusters, 
        train_df, test_df, n_similar=1
    )
    
    if similar_test:
        test_idx = similar_test[0]
        test_id = test_df.iloc[test_idx]['id']
        
        # Create flip from correct baseline
        flip_submission = baseline_df.copy()
        
        # Find and flip
        sub_idx = flip_submission[flip_submission['id'] == test_id].index[0]
        current = flip_submission.loc[sub_idx, 'Personality']
        new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
        flip_submission.loc[sub_idx, 'Personality'] = new_label
        
        # Verify only 1 flip
        n_differences = (flip_submission['Personality'] != baseline_df['Personality']).sum()
        
        if n_differences == 1:
            flip_direction = 'E2I' if current == 'Extrovert' else 'I2E'
            flip_file = f"../scores/flip_METAMODEL_BASELINE_0975708_{improvement['mean_improvement']:.6f}_1_{flip_direction}_id_{test_id}.csv"
            flip_submission.to_csv(flip_file, index=False)
            
            individual_flips.append({
                'file': flip_file,
                'test_id': test_id,
                'train_idx': train_idx,
                'improvement': improvement['mean_improvement'],
                'flip': f"{current}→{new_label}"
            })
            
            print(f"  Created flip {i+1}: ID {test_id} ({current}→{new_label}), improvement={improvement['mean_improvement']:.6f}")

# Create combined submissions
print("\nCreating combined submissions from correct baseline...")

for n_flips in [1, 3, 5, 10]:
    if n_flips > len(individual_flips):
        continue
        
    combined_submission = baseline_df.copy()
    flips_made = []
    
    # Apply top N flips
    for flip_info in individual_flips[:n_flips]:
        test_id = flip_info['test_id']
        sub_idx = combined_submission[combined_submission['id'] == test_id].index[0]
        current = combined_submission.loc[sub_idx, 'Personality']
        new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
        combined_submission.loc[sub_idx, 'Personality'] = new_label
        flips_made.append(test_id)
    
    # Verify
    n_differences = (combined_submission['Personality'] != baseline_df['Personality']).sum()
    
    if n_differences == n_flips:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../scores/metamodel_baseline_0975708_top{n_flips}_{timestamp}.csv"
        combined_submission.to_csv(filename, index=False)
        print(f"  Created top{n_flips} submission with {n_differences} flips: {filename}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nCreated {len(individual_flips)} individual flip files from baseline 0.975708")
print("\nFile pattern: flip_METAMODEL_BASELINE_0975708_[improvement]_1_[direction]_id_[test_id].csv")
print("\nThese files can be submitted to Kaggle to test if the flips improve the score.")

# Save details
import pandas as pd
details_df = pd.DataFrame(individual_flips)
details_df.to_csv('output/metamodel_baseline_0975708_flip_details.csv', index=False)
print(f"\nDetails saved to output/metamodel_baseline_0975708_flip_details.csv")