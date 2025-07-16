#!/usr/bin/env python3
"""
Create submission files based on cluster-based metamodel findings.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CREATING METAMODEL-BASED SUBMISSIONS")
print("="*60)

# Load data
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Load positive improvements from metamodel
positive_improvements = pd.read_csv('output/cluster_positive_improvements.csv')
print(f"\nLoaded {len(positive_improvements)} positive flip candidates from metamodel")

# Recreate clustering (same as before)
print("\nRecreating clustering...")
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
               'Going_outside', 'Drained_after_socializing', 
               'Friends_circle_size', 'Post_frequency']

# Encode categorical
train_data = train_df.copy()
train_data['Stage_fear_binary'] = (train_data['Stage_fear'] == 'Yes').astype(int)
train_data['Drained_binary'] = (train_data['Drained_after_socializing'] == 'Yes').astype(int)

# Features for clustering
cluster_features = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear_binary', 'Drained_binary']

# Handle missing values
train_data[cluster_features] = train_data[cluster_features].fillna(train_data[cluster_features].mean())

# Scale and cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_data[cluster_features])
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Find cluster majorities
cluster_majorities = {}
for cluster_id in range(10):
    cluster_mask = cluster_labels == cluster_id
    cluster_samples = train_df[cluster_mask]
    intro_ratio = (cluster_samples['Personality'] == 'Introvert').mean()
    cluster_majorities[cluster_id] = {
        'ratio': intro_ratio,
        'majority': 'Introvert' if intro_ratio > 0.5 else 'Extrovert'
    }

# Prepare test data
test_data = test_df.copy()
test_data['Stage_fear_binary'] = (test_data['Stage_fear'] == 'Yes').astype(int)
test_data['Drained_binary'] = (test_data['Drained_after_socializing'] == 'Yes').astype(int)
test_data[cluster_features] = test_data[cluster_features].fillna(test_data[cluster_features].mean())

# Predict test clusters
X_test_scaled = scaler.transform(test_data[cluster_features])
test_clusters = kmeans.predict(X_test_scaled)

# Create base submission (cluster majority vote)
base_predictions = []
for cluster in test_clusters:
    base_predictions.append(cluster_majorities[cluster]['majority'])

base_submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': base_predictions
})

print("\nBase submission created using cluster majority vote")

# Strategy 1: Find test samples similar to top positive improvements
print("\n1. Creating submission based on similarity to positive training flips...")

def find_similar_test_samples(train_idx, train_clusters, test_clusters, train_df, test_df, n_similar=10):
    """Find test samples most similar to a training sample."""
    # Get cluster of training sample
    train_cluster = train_clusters[train_idx]
    
    # Find test samples in same cluster
    test_in_cluster = np.where(test_clusters == train_cluster)[0]
    
    if len(test_in_cluster) == 0:
        return []
    
    # Calculate feature similarity
    train_sample = train_df.iloc[train_idx][feature_cols].copy()
    # Convert categorical to numeric
    for col in ['Stage_fear', 'Drained_after_socializing']:
        if col in feature_cols:
            train_sample[col] = 1 if train_sample[col] == 'Yes' else (0 if train_sample[col] == 'No' else -1)
    train_features = train_sample.fillna(-1).values.astype(float)
    
    similarities = []
    
    for test_idx in test_in_cluster:
        test_sample = test_df.iloc[test_idx][feature_cols].copy()
        # Convert categorical to numeric
        for col in ['Stage_fear', 'Drained_after_socializing']:
            if col in feature_cols:
                test_sample[col] = 1 if test_sample[col] == 'Yes' else (0 if test_sample[col] == 'No' else -1)
        test_features = test_sample.fillna(-1).values.astype(float)
        
        # Simple Euclidean distance
        distance = np.linalg.norm(train_features - test_features)
        similarities.append((test_idx, distance))
    
    # Sort by similarity (smallest distance)
    similarities.sort(key=lambda x: x[1])
    
    return [idx for idx, _ in similarities[:n_similar]]

# Create submissions with different numbers of flips
flip_strategies = [
    {'name': 'top1', 'n_flips': 1, 'description': 'Best flip only'},
    {'name': 'top3', 'n_flips': 3, 'description': 'Top 3 flips'},
    {'name': 'top5', 'n_flips': 5, 'description': 'Top 5 flips'},
    {'name': 'top10', 'n_flips': 10, 'description': 'Top 10 flips'},
    {'name': 'all11', 'n_flips': 11, 'description': 'All positive flips'}
]

submissions_created = []

for strategy in flip_strategies:
    print(f"\nCreating {strategy['name']} submission ({strategy['description']})...")
    
    submission = base_submission.copy()
    flips_made = []
    
    # Use top N improvements
    top_improvements = positive_improvements.head(strategy['n_flips'])
    
    for _, improvement in top_improvements.iterrows():
        train_idx = int(improvement['idx'])
        
        # Find similar test samples
        similar_test = find_similar_test_samples(
            train_idx, cluster_labels, test_clusters, 
            train_df, test_df, n_similar=1
        )
        
        if similar_test:
            test_idx = similar_test[0]
            test_id = test_df.iloc[test_idx]['id']
            
            # Apply flip
            sub_idx = submission[submission['id'] == test_id].index[0]
            current = submission.loc[sub_idx, 'Personality']
            new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
            submission.loc[sub_idx, 'Personality'] = new_label
            
            flips_made.append({
                'test_id': test_id,
                'test_idx': test_idx,
                'train_idx': train_idx,
                'improvement': improvement['mean_improvement'],
                'flip': f"{current}→{new_label}"
            })
    
    # Save submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../scores/metamodel_cluster_{strategy['name']}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    
    submissions_created.append({
        'strategy': strategy['name'],
        'filename': filename,
        'n_flips': len(flips_made),
        'flips': flips_made
    })
    
    print(f"  Created {strategy['name']} with {len(flips_made)} flips")
    print(f"  Saved to: {filename}")

# Strategy 2: Flip all test samples in clusters where positive improvements were found
print("\n2. Creating cluster-wide flip submission...")

# Find clusters with positive improvements
positive_train_indices = positive_improvements['idx'].values
positive_clusters = set()
for idx in positive_train_indices:
    positive_clusters.add(cluster_labels[int(idx)])

print(f"\nClusters with positive improvements: {sorted(positive_clusters)}")

# Create submission flipping all minority samples in these clusters
cluster_flip_submission = base_submission.copy()
cluster_flips = 0

for cluster_id in positive_clusters:
    # Get test samples in this cluster
    test_in_cluster = np.where(test_clusters == cluster_id)[0]
    majority = cluster_majorities[cluster_id]['majority']
    
    # Flip samples that match the minority pattern
    for test_idx in test_in_cluster:
        test_id = test_df.iloc[test_idx]['id']
        sub_idx = cluster_flip_submission[cluster_flip_submission['id'] == test_id].index[0]
        current = cluster_flip_submission.loc[sub_idx, 'Personality']
        
        # Only flip if it's currently the minority label (like our training anomalies)
        if current != majority:
            new_label = majority
            cluster_flip_submission.loc[sub_idx, 'Personality'] = new_label
            cluster_flips += 1

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cluster_wide_file = f"../scores/metamodel_cluster_wide_{timestamp}.csv"
cluster_flip_submission.to_csv(cluster_wide_file, index=False)

print(f"\nCluster-wide flip submission:")
print(f"  Flipped {cluster_flips} minority samples in positive clusters")
print(f"  Saved to: {cluster_wide_file}")

# Strategy 3: Create individual flip files for testing
print("\n3. Creating individual flip test files...")

individual_flips = []
for _, improvement in positive_improvements.head(5).iterrows():
    train_idx = int(improvement['idx'])
    
    # Find most similar test sample
    similar_test = find_similar_test_samples(
        train_idx, cluster_labels, test_clusters, 
        train_df, test_df, n_similar=1
    )
    
    if similar_test:
        test_idx = similar_test[0]
        test_id = test_df.iloc[test_idx]['id']
        
        # Determine flip direction
        sub_idx = base_submission[base_submission['id'] == test_id].index[0]
        current = base_submission.loc[sub_idx, 'Personality']
        flip_direction = 'I2E' if current == 'Introvert' else 'E2I'
        
        # Create individual flip file
        flip_submission = base_submission.copy()
        new_label = 'Extrovert' if current == 'Introvert' else 'Introvert'
        flip_submission.loc[sub_idx, 'Personality'] = new_label
        
        flip_file = f"../scores/flip_METAMODEL_CLUSTER_{improvement['mean_improvement']:.6f}_1_{flip_direction}_id_{test_id}.csv"
        flip_submission.to_csv(flip_file, index=False)
        
        individual_flips.append({
            'file': flip_file,
            'test_id': test_id,
            'train_idx': train_idx,
            'improvement': improvement['mean_improvement']
        })

print(f"\nCreated {len(individual_flips)} individual flip test files")

# Summary report
print("\n" + "="*60)
print("SUBMISSION SUMMARY")
print("="*60)

print("\n1. Multi-flip submissions:")
for sub in submissions_created:
    print(f"   - {sub['strategy']}: {sub['n_flips']} flips → {sub['filename']}")

print(f"\n2. Cluster-wide submission:")
print(f"   - {cluster_flips} flips → {cluster_wide_file}")

print(f"\n3. Individual test flips: {len(individual_flips)} files created")

# Save detailed flip information
flip_details = []
for sub in submissions_created:
    for flip in sub['flips']:
        flip_details.append({
            'strategy': sub['strategy'],
            'test_id': flip['test_id'],
            'train_idx': flip['train_idx'],
            'improvement': flip['improvement'],
            'flip': flip['flip']
        })

if flip_details:
    flip_df = pd.DataFrame(flip_details)
    flip_df.to_csv('output/metamodel_flip_details.csv', index=False)
    print(f"\nFlip details saved to output/metamodel_flip_details.csv")

print("\n" + "="*60)
print("All metamodel-based submissions created successfully!")
print("="*60)