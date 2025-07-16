#!/usr/bin/env python3
"""
Evaluate flip impacts for cluster anomalies.
"""

import pandas as pd
import numpy as np
import ydf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CLUSTER-BASED FLIP EVALUATION")
print("="*60)

# Load data
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Prepare for clustering
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
        'majority': 'Introvert' if intro_ratio > 0.5 else 'Extrovert',
        'unanimity': abs(intro_ratio - 0.5) * 2
    }

# Find top anomalies from most unanimous clusters
anomalies = []
for cluster_id in sorted(range(10), key=lambda x: cluster_majorities[x]['unanimity'], reverse=True):
    if cluster_majorities[cluster_id]['unanimity'] < 0.9:  # Skip less unanimous clusters
        continue
        
    cluster_mask = cluster_labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    majority = cluster_majorities[cluster_id]['majority']
    
    for idx in cluster_indices:
        if train_df.iloc[idx]['Personality'] != majority:
            anomalies.append({
                'idx': idx,
                'cluster': cluster_id,
                'label': train_df.iloc[idx]['Personality'],
                'majority': majority,
                'unanimity': cluster_majorities[cluster_id]['unanimity']
            })

print(f"\nFound {len(anomalies)} anomalies in highly unanimous clusters (>90%)")

# Evaluate flips using leave-one-cluster-out CV
print("\nEvaluating top anomalies with cluster-based CV...")

def evaluate_flip(train_df, flip_idx, cluster_labels, feature_cols):
    """Evaluate a flip using leave-one-cluster-out CV."""
    flip_cluster = cluster_labels[flip_idx]
    results = []
    
    # Test on each other cluster
    for val_cluster in range(10):
        if val_cluster == flip_cluster:
            continue
            
        # Split data
        train_mask = cluster_labels != val_cluster
        val_mask = cluster_labels == val_cluster
        
        train_subset = train_df[train_mask][feature_cols + ['Personality']].copy()
        val_subset = train_df[val_mask][feature_cols + ['Personality']].copy()
        
        # Handle missing values
        train_subset[feature_cols] = train_subset[feature_cols].fillna(-1)
        val_subset[feature_cols] = val_subset[feature_cols].fillna(-1)
        
        # Train baseline
        model1 = ydf.RandomForestLearner(
            label='Personality',
            num_trees=50,
            max_depth=10
        ).train(train_subset)
        
        # Baseline accuracy
        val_probs1 = model1.predict(val_subset)
        val_preds1 = ['Introvert' if p > 0.5 else 'Extrovert' for p in val_probs1]
        acc1 = np.mean([pred == actual for pred, actual in zip(val_preds1, val_subset['Personality'])])
        
        # If flip sample is in training set, create flipped version
        if flip_idx in np.where(train_mask)[0]:
            train_subset_flipped = train_subset.copy()
            # Find position in subset
            original_indices = np.where(train_mask)[0]
            subset_idx = np.where(original_indices == flip_idx)[0][0]
            
            current = train_subset_flipped.iloc[subset_idx]['Personality']
            new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
            train_subset_flipped.iloc[subset_idx, train_subset_flipped.columns.get_loc('Personality')] = new_label
            
            # Train with flip
            model2 = ydf.RandomForestLearner(
                label='Personality',
                num_trees=50,
                max_depth=10
            ).train(train_subset_flipped)
            
            val_probs2 = model2.predict(val_subset)
            val_preds2 = ['Introvert' if p > 0.5 else 'Extrovert' for p in val_probs2]
            acc2 = np.mean([pred == actual for pred, actual in zip(val_preds2, val_subset['Personality'])])
            
            improvement = acc2 - acc1
            results.append({
                'val_cluster': val_cluster,
                'improvement': improvement,
                'baseline_acc': acc1,
                'flipped_acc': acc2
            })
    
    return results

# Evaluate top anomalies
evaluation_results = []
n_to_test = min(20, len(anomalies))

for i, anomaly in enumerate(anomalies[:n_to_test]):
    print(f"\nEvaluating {i+1}/{n_to_test}: Index {anomaly['idx']}")
    print(f"  {anomaly['label']} in {anomaly['majority']}-majority cluster (unanimity: {anomaly['unanimity']:.1%})")
    
    results = evaluate_flip(train_df, anomaly['idx'], cluster_labels, feature_cols)
    
    if results:
        improvements = [r['improvement'] for r in results]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        positive_count = sum(1 for imp in improvements if imp > 0)
        
        print(f"  Mean improvement: {mean_imp:+.6f} ± {std_imp:.6f}")
        print(f"  Positive in {positive_count}/{len(results)} clusters")
        
        evaluation_results.append({
            'idx': anomaly['idx'],
            'anomaly': anomaly,
            'mean_improvement': mean_imp,
            'std_improvement': std_imp,
            'positive_clusters': positive_count,
            'improvements': improvements
        })

# Find best candidates
print("\n" + "="*60)
print("SUMMARY OF FLIP EVALUATION")
print("="*60)

positive_flips = [r for r in evaluation_results if r['mean_improvement'] > 0]

if positive_flips:
    print(f"\nFound {len(positive_flips)} flips with positive mean improvement:")
    positive_flips.sort(key=lambda x: x['mean_improvement'], reverse=True)
    
    for i, result in enumerate(positive_flips[:10]):
        print(f"\n{i+1}. Index {result['idx']}:")
        print(f"   {result['anomaly']['label']} → {result['anomaly']['majority']}")
        print(f"   Mean improvement: {result['mean_improvement']:+.6f}")
        print(f"   Positive in {result['positive_clusters']}/9 clusters")
        
        # Show cluster-by-cluster results
        sorted_improvements = sorted(enumerate(result['improvements']), 
                                   key=lambda x: x[1]['improvement'], 
                                   reverse=True)
        print("   Best clusters:")
        for j, (_, imp_data) in enumerate(sorted_improvements[:3]):
            print(f"     Cluster {imp_data['val_cluster']}: {imp_data['improvement']:+.6f} "
                  f"({imp_data['baseline_acc']:.4f} → {imp_data['flipped_acc']:.4f})")
else:
    print("\nNo flips with positive mean improvement found.")

# Save results
results_df = pd.DataFrame([{
    'idx': r['idx'],
    'cluster': r['anomaly']['cluster'],
    'original_label': r['anomaly']['label'],
    'cluster_majority': r['anomaly']['majority'],
    'cluster_unanimity': r['anomaly']['unanimity'],
    'mean_improvement': r['mean_improvement'],
    'std_improvement': r['std_improvement'],
    'positive_clusters': r['positive_clusters']
} for r in evaluation_results])

results_df.to_csv('output/cluster_flip_evaluation_results.csv', index=False)
print(f"\nDetailed results saved to output/cluster_flip_evaluation_results.csv")

# Create submission with best flips
if positive_flips:
    print("\nCreating submission with best cluster-based flips...")
    
    # Load baseline submission (cluster-based)
    baseline = pd.read_csv('../scores/cluster_based_simple_20250714_220014.csv')
    submission = baseline.copy()
    
    # Apply top flips
    n_flips = min(5, len(positive_flips))
    flip_ids = []
    
    for i in range(n_flips):
        train_idx = positive_flips[i]['idx']
        train_id = train_df.iloc[train_idx]['id']
        
        # Find corresponding test samples in same cluster
        train_cluster = cluster_labels[train_idx]
        
        # Predict test clusters
        test_data = test_df.copy()
        test_data['Stage_fear_binary'] = (test_data['Stage_fear'] == 'Yes').astype(int)
        test_data['Drained_binary'] = (test_data['Drained_after_socializing'] == 'Yes').astype(int)
        test_data[cluster_features] = test_data[cluster_features].fillna(test_data[cluster_features].mean())
        X_test_scaled = scaler.transform(test_data[cluster_features])
        test_clusters = kmeans.predict(X_test_scaled)
        
        # Find test samples in same cluster
        test_in_cluster = np.where(test_clusters == train_cluster)[0]
        
        if len(test_in_cluster) > 0:
            # Pick one randomly
            test_idx = np.random.choice(test_in_cluster, 1)[0]
            test_id = test_df.iloc[test_idx]['id']
            
            # Apply flip
            sub_idx = submission[submission['id'] == test_id].index[0]
            current = submission.loc[sub_idx, 'Personality']
            new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
            submission.loc[sub_idx, 'Personality'] = new_label
            
            flip_ids.append(test_id)
            print(f"  Flipped test ID {test_id}: {current} → {new_label}")
    
    if flip_ids:
        # Save submission
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = f'../scores/cluster_flip_best_{n_flips}_{timestamp}.csv'
        submission.to_csv(submission_file, index=False)
        print(f"\nSaved submission with {len(flip_ids)} flips to: {submission_file}")

print("\nAnalysis complete!")