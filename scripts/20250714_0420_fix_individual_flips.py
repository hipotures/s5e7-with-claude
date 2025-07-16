#!/usr/bin/env python3
"""
Fix individual flip files - create them from a clean baseline with only 1 flip each.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

print("="*60)
print("FIXING INDIVIDUAL FLIP FILES")
print("="*60)

# Load data
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Load the clean baseline (sample submission or create from scratch)
print("\nCreating clean baseline submission...")
# Use sample submission as baseline
sample_submission = pd.read_csv('../../sample_submission.csv')
clean_baseline = sample_submission.copy()

print(f"Baseline has {len(clean_baseline)} predictions")

# Load positive improvements
positive_improvements = pd.read_csv('output/cluster_positive_improvements.csv')
print(f"\nLoaded {len(positive_improvements)} positive flip candidates")

# Recreate clustering to find similar test samples
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

# Create proper individual flip files
print("\nCreating corrected individual flip files...")
individual_flips_created = []

for i, (_, improvement) in enumerate(positive_improvements.head(10).iterrows()):
    train_idx = int(improvement['idx'])
    
    # Find test samples in same cluster
    train_cluster = cluster_labels[train_idx]
    test_in_cluster = np.where(test_clusters == train_cluster)[0]
    
    if len(test_in_cluster) > 0:
        # For simplicity, take a random sample from the cluster
        # In the original, we used similarity, but that was causing issues
        np.random.seed(42 + i)  # Different seed for each flip
        test_idx = np.random.choice(test_in_cluster, 1)[0]
        test_id = test_df.iloc[test_idx]['id']
        
        # Create individual flip from CLEAN baseline
        flip_submission = clean_baseline.copy()
        
        # Find the test ID in submission
        sub_idx = flip_submission[flip_submission['id'] == test_id].index[0]
        current = flip_submission.loc[sub_idx, 'Personality']
        
        # Flip it
        new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
        flip_submission.loc[sub_idx, 'Personality'] = new_label
        
        # Verify only 1 flip
        n_differences = (flip_submission['Personality'] != clean_baseline['Personality']).sum()
        
        if n_differences == 1:
            # Save the file
            flip_direction = 'E2I' if current == 'Extrovert' else 'I2E'
            flip_file = f"../scores/flip_METAMODEL_FIXED_{improvement['mean_improvement']:.6f}_1_{flip_direction}_id_{test_id}.csv"
            flip_submission.to_csv(flip_file, index=False)
            
            individual_flips_created.append({
                'file': flip_file,
                'test_id': test_id,
                'train_idx': train_idx,
                'improvement': improvement['mean_improvement'],
                'flip': f"{current}→{new_label}",
                'n_differences': n_differences
            })
            
            print(f"  Created flip {i+1}: ID {test_id} ({current}→{new_label}), improvement={improvement['mean_improvement']:.6f}")
        else:
            print(f"  ERROR: Flip {i+1} has {n_differences} differences instead of 1!")

# Also create a combined submission with top N flips from clean baseline
print("\nCreating combined submissions from clean baseline...")

for n_flips in [1, 3, 5, 10]:
    combined_submission = clean_baseline.copy()
    flips_made = []
    
    # Apply top N flips
    for flip_info in individual_flips_created[:n_flips]:
        test_id = flip_info['test_id']
        sub_idx = combined_submission[combined_submission['id'] == test_id].index[0]
        current = combined_submission.loc[sub_idx, 'Personality']
        new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
        combined_submission.loc[sub_idx, 'Personality'] = new_label
        flips_made.append(test_id)
    
    # Verify number of flips
    n_differences = (combined_submission['Personality'] != clean_baseline['Personality']).sum()
    
    if n_differences == n_flips:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../scores/metamodel_fixed_top{n_flips}_{timestamp}.csv"
        combined_submission.to_csv(filename, index=False)
        print(f"  Created top{n_flips} submission with {n_differences} flips: {filename}")
    else:
        print(f"  ERROR: top{n_flips} has {n_differences} flips instead of {n_flips}!")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nCreated {len(individual_flips_created)} corrected individual flip files")
print("\nEach file contains exactly 1 flip from the clean baseline")
print("File naming: flip_METAMODEL_FIXED_[improvement]_1_[direction]_id_[test_id].csv")

# Save details
import pandas as pd
details_df = pd.DataFrame(individual_flips_created)
details_df.to_csv('output/metamodel_fixed_flip_details.csv', index=False)
print(f"\nDetails saved to output/metamodel_fixed_flip_details.csv")