#!/usr/bin/env python3
"""
MetaModel with Cluster-based Cross-Validation
Uses KMeans clustering (k=10) instead of random CV splits.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import ydf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
from tqdm import tqdm
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ClusterBasedEvaluator:
    """Evaluate flips using cluster-based cross-validation."""
    
    def __init__(self, train_data, n_clusters=10):
        self.train_data = train_data
        self.n_clusters = n_clusters
        self.feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                            'Going_outside', 'Drained_after_socializing', 
                            'Friends_circle_size', 'Post_frequency']
        
        # Prepare data for clustering
        self._prepare_clustering()
        
    def _prepare_clustering(self):
        """Prepare data and perform clustering."""
        logging.info(f"Performing KMeans clustering with {self.n_clusters} clusters...")
        
        # Prepare features
        data = self.train_data.copy()
        
        # Encode categorical features
        data['Stage_fear_binary'] = (data['Stage_fear'] == 'Yes').astype(int)
        data['Drained_binary'] = (data['Drained_after_socializing'] == 'Yes').astype(int)
        
        # Select features for clustering
        cluster_features = ['Time_spent_Alone', 'Social_event_attendance', 
                           'Friends_circle_size', 'Going_outside', 'Post_frequency',
                           'Stage_fear_binary', 'Drained_binary']
        
        # Handle missing values
        data[cluster_features] = data[cluster_features].fillna(data[cluster_features].mean())
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(data[cluster_features])
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, self.cluster_labels)
        logging.info(f"Clustering silhouette score: {sil_score:.3f}")
        
        # Analyze cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        logging.info("Cluster sizes:")
        for cluster, count in zip(unique, counts):
            pct = count / len(self.cluster_labels) * 100
            logging.info(f"  Cluster {cluster}: {count} samples ({pct:.1f}%)")
    
    def evaluate_flip_cluster_based(self, flip_idx):
        """Evaluate a flip using leave-one-cluster-out cross-validation."""
        results = []
        
        # Get the cluster of the flip sample
        flip_cluster = self.cluster_labels[flip_idx]
        
        # Prepare data
        feature_cols = self.feature_cols
        
        # For each cluster as validation set
        for val_cluster in range(self.n_clusters):
            # Skip if this is the flip sample's cluster
            if val_cluster == flip_cluster:
                continue
                
            # Create train/val split based on clusters
            train_mask = (self.cluster_labels != val_cluster)
            val_mask = (self.cluster_labels == val_cluster)
            
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            
            # Prepare datasets
            train_subset = self.train_data.iloc[train_indices].copy()
            val_subset = self.train_data.iloc[val_indices].copy()
            
            # Prepare data for YDF
            train_data = train_subset[feature_cols + ['Personality']].copy()
            val_data = val_subset[feature_cols + ['Personality']].copy()
            
            # Handle missing values
            train_data[feature_cols] = train_data[feature_cols].fillna(-1)
            val_data[feature_cols] = val_data[feature_cols].fillna(-1)
            
            # Train baseline model
            model1 = ydf.RandomForestLearner(
                label='Personality',
                num_trees=30,
                max_depth=8
            ).train(train_data)
            
            # Get baseline accuracy
            val_probs1 = model1.predict(val_data)
            val_preds1 = ['Introvert' if p > 0.5 else 'Extrovert' for p in val_probs1]
            acc1 = np.mean([pred == actual for pred, actual in zip(val_preds1, val_data['Personality'])])
            
            # Apply flip if sample is in training set
            if flip_idx in train_indices:
                train_data_flipped = train_data.copy()
                # Find position in subset
                subset_idx = np.where(train_indices == flip_idx)[0][0]
                current = train_data_flipped.iloc[subset_idx]['Personality']
                new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
                train_data_flipped.iloc[subset_idx, -1] = new_label
                
                # Train with flip
                model2 = ydf.RandomForestLearner(
                    label='Personality',
                    num_trees=30,
                    max_depth=8
                ).train(train_data_flipped)
                
                val_probs2 = model2.predict(val_data)
                val_preds2 = ['Introvert' if p > 0.5 else 'Extrovert' for p in val_probs2]
                acc2 = np.mean([pred == actual for pred, actual in zip(val_preds2, val_data['Personality'])])
                
                improvement = acc2 - acc1
                results.append({
                    'val_cluster': val_cluster,
                    'val_size': len(val_indices),
                    'baseline_acc': acc1,
                    'flipped_acc': acc2,
                    'improvement': improvement
                })
        
        return results
    
    def get_cluster_info(self, idx):
        """Get cluster information for a sample."""
        return {
            'cluster': self.cluster_labels[idx],
            'cluster_size': np.sum(self.cluster_labels == self.cluster_labels[idx]),
            'n_clusters': self.n_clusters
        }


class ClusterAwareFeatureExtractor:
    """Extract features including cluster-based information."""
    
    def __init__(self, train_data, cluster_evaluator):
        self.train_data = train_data
        self.cluster_evaluator = cluster_evaluator
        self.feature_cols = cluster_evaluator.feature_cols
        
    def extract_features(self, sample_idx):
        """Extract features including cluster information."""
        sample = self.train_data.iloc[sample_idx]
        cluster_info = self.cluster_evaluator.get_cluster_info(sample_idx)
        
        features = {}
        
        # Basic features
        features['null_count'] = sample[self.feature_cols].isna().sum()
        features['has_nulls'] = float(features['null_count'] > 0)
        
        # Cluster features
        features['cluster_id'] = cluster_info['cluster']
        features['cluster_size'] = cluster_info['cluster_size']
        features['cluster_size_ratio'] = cluster_info['cluster_size'] / len(self.train_data)
        
        # Label features
        is_introvert = sample['Personality'] == 'Introvert'
        features['is_introvert'] = float(is_introvert)
        
        # Pattern features
        features['null_introvert_pattern'] = float(features['has_nulls'] and is_introvert)
        features['null_extrovert_pattern'] = float(features['has_nulls'] and not is_introvert)
        
        # Cluster personality distribution
        cluster_mask = self.cluster_evaluator.cluster_labels == cluster_info['cluster']
        cluster_samples = self.train_data[cluster_mask]
        intro_ratio = (cluster_samples['Personality'] == 'Introvert').mean()
        features['cluster_intro_ratio'] = intro_ratio
        features['cluster_extro_ratio'] = 1 - intro_ratio
        
        # Disagreement with cluster majority
        cluster_majority = 'Introvert' if intro_ratio > 0.5 else 'Extrovert'
        features['disagrees_with_cluster'] = float(sample['Personality'] != cluster_majority)
        
        return list(features.values())
    
    def get_feature_names(self):
        """Return feature names."""
        return [
            'null_count', 'has_nulls', 'cluster_id', 'cluster_size', 
            'cluster_size_ratio', 'is_introvert', 'null_introvert_pattern',
            'null_extrovert_pattern', 'cluster_intro_ratio', 'cluster_extro_ratio',
            'disagrees_with_cluster'
        ]


def find_cluster_anomalies(train_data, cluster_evaluator, n_candidates=100):
    """Find samples that disagree with their cluster's majority."""
    anomalies = []
    
    for cluster_id in range(cluster_evaluator.n_clusters):
        # Get samples in this cluster
        cluster_mask = cluster_evaluator.cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_samples = train_data.iloc[cluster_indices]
        
        # Find majority label
        intro_count = (cluster_samples['Personality'] == 'Introvert').sum()
        extro_count = (cluster_samples['Personality'] == 'Extrovert').sum()
        majority_label = 'Introvert' if intro_count > extro_count else 'Extrovert'
        
        # Find samples that disagree with majority
        for idx in cluster_indices:
            if train_data.iloc[idx]['Personality'] != majority_label:
                anomalies.append({
                    'idx': idx,
                    'cluster': cluster_id,
                    'label': train_data.iloc[idx]['Personality'],
                    'majority_label': majority_label,
                    'cluster_intro_ratio': intro_count / len(cluster_samples)
                })
    
    # Sort by cluster imbalance (most unanimous clusters first)
    anomalies.sort(key=lambda x: abs(x['cluster_intro_ratio'] - 0.5), reverse=True)
    
    return anomalies[:n_candidates]


def main():
    """Run cluster-based metamodel analysis."""
    # Load data
    logging.info("Loading data...")
    train_df = pd.read_csv('../../train.csv')
    test_df = pd.read_csv('../../test.csv')
    
    # Initialize cluster-based evaluator
    cluster_evaluator = ClusterBasedEvaluator(train_df, n_clusters=10)
    
    # Find cluster anomalies
    logging.info("\nFinding cluster anomalies...")
    anomalies = find_cluster_anomalies(train_df, cluster_evaluator, n_candidates=50)
    
    logging.info(f"Found {len(anomalies)} samples disagreeing with cluster majority")
    
    # Evaluate top anomalies
    logging.info("\nEvaluating top cluster anomalies...")
    evaluation_results = []
    
    for i, anomaly in enumerate(anomalies[:10]):  # Test top 10
        idx = anomaly['idx']
        logging.info(f"\nEvaluating anomaly {i+1}/{10}:")
        logging.info(f"  Index: {idx}, Cluster: {anomaly['cluster']}")
        logging.info(f"  Label: {anomaly['label']}, Majority: {anomaly['majority_label']}")
        
        # Evaluate flip using cluster-based CV
        results = cluster_evaluator.evaluate_flip_cluster_based(idx)
        
        if results:
            mean_improvement = np.mean([r['improvement'] for r in results])
            std_improvement = np.std([r['improvement'] for r in results])
            positive_clusters = sum(1 for r in results if r['improvement'] > 0)
            
            logging.info(f"  Mean improvement: {mean_improvement:.4f} ± {std_improvement:.4f}")
            logging.info(f"  Positive in {positive_clusters}/{len(results)} clusters")
            
            evaluation_results.append({
                'idx': idx,
                'anomaly': anomaly,
                'mean_improvement': mean_improvement,
                'std_improvement': std_improvement,
                'positive_clusters': positive_clusters,
                'total_clusters': len(results)
            })
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("CLUSTER-BASED METAMODEL SUMMARY")
    logging.info("="*60)
    
    # Find best candidates
    positive_flips = [r for r in evaluation_results if r['mean_improvement'] > 0]
    
    if positive_flips:
        logging.info(f"\nFound {len(positive_flips)} flips with positive impact:")
        positive_flips.sort(key=lambda x: x['mean_improvement'], reverse=True)
        
        for i, result in enumerate(positive_flips[:5]):
            logging.info(f"\n{i+1}. Index {result['idx']}:")
            logging.info(f"   Cluster {result['anomaly']['cluster']}: "
                        f"{result['anomaly']['label']} in {result['anomaly']['majority_label']}-majority cluster")
            logging.info(f"   Impact: {result['mean_improvement']:+.4f} ± {result['std_improvement']:.4f}")
            logging.info(f"   Positive in {result['positive_clusters']}/{result['total_clusters']} clusters")
    else:
        logging.info("\nNo flips with positive impact found")
    
    # Save detailed results
    results_df = pd.DataFrame(evaluation_results)
    results_df.to_csv('output/metamodel_cluster_based_results.csv', index=False)
    logging.info(f"\nDetailed results saved to output/metamodel_cluster_based_results.csv")
    
    # Create test submission based on cluster patterns
    logging.info("\nCreating cluster-based submission...")
    create_cluster_based_submission(test_df, cluster_evaluator, train_df, top_k=5)


def create_cluster_based_submission(test_df, cluster_evaluator, train_df, top_k=5):
    """Create submission based on cluster patterns."""
    # Prepare test data for clustering
    test_data = test_df.copy()
    
    # Encode categorical features
    test_data['Stage_fear_binary'] = (test_data['Stage_fear'] == 'Yes').astype(int)
    test_data['Drained_binary'] = (test_data['Drained_after_socializing'] == 'Yes').astype(int)
    
    # Select features
    cluster_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency',
                       'Stage_fear_binary', 'Drained_binary']
    
    # Handle missing values
    test_data[cluster_features] = test_data[cluster_features].fillna(test_data[cluster_features].mean())
    
    # Scale and predict clusters
    X_test_scaled = cluster_evaluator.scaler.transform(test_data[cluster_features])
    test_clusters = cluster_evaluator.kmeans.predict(X_test_scaled)
    
    # Find cluster majorities from training data
    cluster_majorities = {}
    for cluster_id in range(cluster_evaluator.n_clusters):
        cluster_mask = cluster_evaluator.cluster_labels == cluster_id
        cluster_samples = train_df[cluster_mask]
        intro_ratio = (cluster_samples['Personality'] == 'Introvert').mean()
        cluster_majorities[cluster_id] = 'Introvert' if intro_ratio > 0.5 else 'Extrovert'
    
    # Create predictions based on cluster majority
    predictions = []
    for i, cluster in enumerate(test_clusters):
        predictions.append(cluster_majorities[cluster])
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': predictions
    })
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f'../scores/metamodel_cluster_based_{timestamp}.csv'
    submission.to_csv(submission_file, index=False)
    logging.info(f"Saved cluster-based submission to {submission_file}")
    
    # Show cluster distribution in test
    unique, counts = np.unique(test_clusters, return_counts=True)
    logging.info("\nTest set cluster distribution:")
    for cluster, count in zip(unique, counts):
        pct = count / len(test_clusters) * 100
        majority = cluster_majorities[cluster]
        logging.info(f"  Cluster {cluster} ({majority}-majority): {count} samples ({pct:.1f}%)")


if __name__ == "__main__":
    main()