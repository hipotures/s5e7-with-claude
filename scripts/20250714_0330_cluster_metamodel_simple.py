#!/usr/bin/env python3
"""
Simplified cluster-based metamodel to test the concept.
"""

import pandas as pd
import numpy as np
import ydf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime

print("="*60)
print("CLUSTER-BASED METAMODEL - SIMPLIFIED VERSION")
print("="*60)

# Load data
print("\n1. Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Prepare for clustering
print("\n2. Preparing data for clustering...")
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

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_data[cluster_features])

# Cluster
print("\n3. Performing KMeans clustering (k=10)...")
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Show cluster distribution
unique, counts = np.unique(cluster_labels, return_counts=True)
print("\nCluster sizes:")
for cluster, count in zip(unique, counts):
    pct = count / len(cluster_labels) * 100
    print(f"  Cluster {cluster}: {count} samples ({pct:.1f}%)")

# Analyze personality distribution per cluster
print("\n4. Analyzing personality distribution per cluster...")
cluster_personalities = {}

for cluster_id in range(10):
    cluster_mask = cluster_labels == cluster_id
    cluster_samples = train_df[cluster_mask]
    intro_count = (cluster_samples['Personality'] == 'Introvert').sum()
    extro_count = (cluster_samples['Personality'] == 'Extrovert').sum()
    intro_ratio = intro_count / (intro_count + extro_count)
    
    cluster_personalities[cluster_id] = {
        'intro_count': intro_count,
        'extro_count': extro_count,
        'intro_ratio': intro_ratio,
        'majority': 'Introvert' if intro_ratio > 0.5 else 'Extrovert'
    }
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Introverts: {intro_count} ({intro_ratio:.1%})")
    print(f"  Extroverts: {extro_count} ({1-intro_ratio:.1%})")
    print(f"  Majority: {cluster_personalities[cluster_id]['majority']}")

# Find anomalies (samples disagreeing with cluster majority)
print("\n5. Finding cluster anomalies...")
anomalies = []

for cluster_id in range(10):
    cluster_mask = cluster_labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    majority = cluster_personalities[cluster_id]['majority']
    
    for idx in cluster_indices:
        if train_df.iloc[idx]['Personality'] != majority:
            anomalies.append({
                'idx': idx,
                'cluster': cluster_id,
                'label': train_df.iloc[idx]['Personality'],
                'majority': majority,
                'cluster_intro_ratio': cluster_personalities[cluster_id]['intro_ratio']
            })

# Sort by cluster unanimity
anomalies.sort(key=lambda x: abs(x['cluster_intro_ratio'] - 0.5), reverse=True)

print(f"\nFound {len(anomalies)} total anomalies")
print("\nTop 10 anomalies (from most unanimous clusters):")
for i, anomaly in enumerate(anomalies[:10]):
    cluster_ratio = anomaly['cluster_intro_ratio']
    unanimity = abs(cluster_ratio - 0.5) * 2  # Convert to 0-1 scale
    print(f"{i+1}. Index {anomaly['idx']}: {anomaly['label']} in Cluster {anomaly['cluster']} "
          f"({anomaly['majority']}-majority, {unanimity:.1%} unanimous)")

# Test a few anomalies with simple evaluation
print("\n6. Testing flip impact for top anomalies...")
test_anomalies = anomalies[:3]

for i, anomaly in enumerate(test_anomalies):
    idx = anomaly['idx']
    print(f"\nTesting anomaly {i+1}: Index {idx}")
    
    # Simple evaluation: train on other clusters, validate on this cluster
    val_cluster = anomaly['cluster']
    train_mask = cluster_labels != val_cluster
    val_mask = cluster_labels == val_cluster
    
    # Prepare data
    train_subset = train_df[train_mask][feature_cols + ['Personality']].copy()
    val_subset = train_df[val_mask][feature_cols + ['Personality']].copy()
    
    # Handle missing values
    train_subset[feature_cols] = train_subset[feature_cols].fillna(-1)
    val_subset[feature_cols] = val_subset[feature_cols].fillna(-1)
    
    # Train baseline
    model1 = ydf.RandomForestLearner(
        label='Personality',
        num_trees=50,
        max_depth=8
    ).train(train_subset)
    
    # Predict
    val_probs1 = model1.predict(val_subset)
    val_preds1 = ['Introvert' if p > 0.5 else 'Extrovert' for p in val_probs1]
    acc1 = np.mean([pred == actual for pred, actual in zip(val_preds1, val_subset['Personality'])])
    
    print(f"  Validation accuracy: {acc1:.4f}")
    print(f"  Cluster {val_cluster} has {len(val_subset)} samples")

# Create submission based on cluster majorities
print("\n7. Creating cluster-based submission...")

# Prepare test data
test_data = test_df.copy()
test_data['Stage_fear_binary'] = (test_data['Stage_fear'] == 'Yes').astype(int)
test_data['Drained_binary'] = (test_data['Drained_after_socializing'] == 'Yes').astype(int)
test_data[cluster_features] = test_data[cluster_features].fillna(test_data[cluster_features].mean())

# Predict clusters for test data
X_test_scaled = scaler.transform(test_data[cluster_features])
test_clusters = kmeans.predict(X_test_scaled)

# Assign labels based on cluster majority
predictions = []
for cluster in test_clusters:
    predictions.append(cluster_personalities[cluster]['majority'])

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': predictions
})

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_file = f'../scores/cluster_based_simple_{timestamp}.csv'
submission.to_csv(submission_file, index=False)
print(f"\nSaved submission to: {submission_file}")

# Show test cluster distribution
print("\nTest set cluster distribution:")
unique, counts = np.unique(test_clusters, return_counts=True)
for cluster, count in zip(unique, counts):
    pct = count / len(test_clusters) * 100
    majority = cluster_personalities[cluster]['majority']
    print(f"  Cluster {cluster} ({majority}): {count} samples ({pct:.1f}%)")

print("\n" + "="*60)
print("SUMMARY: Cluster-based approach assigns labels based on")
print("cluster majority vote. This could work if clusters are")
print("homogeneous and test samples follow same patterns.")
print("="*60)