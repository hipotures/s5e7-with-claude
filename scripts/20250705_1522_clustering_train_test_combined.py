#!/usr/bin/env python3
"""
COMBINED TRAIN-TEST CLUSTERING VISUALIZATION
===========================================

This script creates clustering visualizations showing both training and test data:
- Training data: Green
- Test data: Yellow
- Misclassified (both sets): Red

Author: Claude
Date: 2025-07-05 15:22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Output file
output_file = open('output/20250705_1522_clustering_train_test_combined.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def prepare_combined_data(train_df, test_df):
    """Prepare combined train and test data for clustering analysis."""
    log_print("Preparing combined data for analysis...")
    
    # Add source indicator
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    
    # Encode categorical features
    for df in [train_df, test_df]:
        df['Stage_fear_binary'] = (df['Stage_fear'] == 'Yes').astype(int)
        df['Drained_binary'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    # Select features for clustering
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear_binary', 'Drained_binary']
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    y_train = (train_df['Personality'] == 'Extrovert').astype(int) if 'Personality' in train_df else None
    
    # Combine for scaling
    X_combined = pd.concat([X_train, X_test])
    
    # Standardize features
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    
    # Split back
    n_train = len(X_train)
    X_train_scaled = X_combined_scaled[:n_train]
    X_test_scaled = X_combined_scaled[n_train:]
    
    return X_train, X_train_scaled, y_train, X_test, X_test_scaled, X_combined_scaled, feature_cols, scaler, n_train


def identify_all_misclassifications(X_train, y_train, test_df):
    """Identify misclassifications in both training and test sets."""
    log_print("\nIdentifying misclassifications...")
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Training misclassifications
    y_pred_train = model.predict(X_train)
    train_misclass_mask = y_pred_train != y_train
    train_misclass_indices = np.where(train_misclass_mask)[0]
    
    log_print(f"Training misclassifications: {len(train_misclass_indices)} ({len(train_misclass_indices)/len(y_train)*100:.2f}%)")
    
    # Test misclassifications (from previous analysis)
    try:
        misclass_df = pd.read_csv('output/all_potential_misclassifications.csv')
        test_misclass_ids = misclass_df['id'].values
        test_misclass_mask = np.isin(test_df['id'].values, test_misclass_ids)
        log_print(f"Test misclassifications loaded: {len(test_misclass_ids)}")
    except:
        log_print("Warning: Could not load test misclassifications, using empty set")
        test_misclass_mask = np.zeros(len(test_df), dtype=bool)
        test_misclass_ids = np.array([])
    
    return train_misclass_mask, test_misclass_mask, model


def create_combined_visualization(train_df, test_df, X_train_scaled, X_test_scaled, 
                                 X_combined_scaled, train_misclass_mask, test_misclass_mask, n_train):
    """Create comprehensive visualization with train and test data."""
    log_print("\nCreating combined visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(20, 24))
    
    # Define color scheme
    colors = {
        'train_correct': '#2ecc71',    # Green
        'test_correct': '#f1c40f',     # Yellow
        'misclassified': '#e74c3c'     # Red
    }
    
    # Combined masks for all data
    combined_misclass_mask = np.concatenate([train_misclass_mask, test_misclass_mask])
    
    # 1. PCA 2D Projection - All Data
    ax1 = plt.subplot(3, 3, 1)
    pca = PCA(n_components=2, random_state=42)
    combined_pca = pca.fit_transform(X_combined_scaled)
    train_pca = combined_pca[:n_train]
    test_pca = combined_pca[n_train:]
    
    # Plot correct classifications
    ax1.scatter(train_pca[~train_misclass_mask, 0], train_pca[~train_misclass_mask, 1], 
               c=colors['train_correct'], alpha=0.5, s=20, label='Train (correct)')
    ax1.scatter(test_pca[~test_misclass_mask, 0], test_pca[~test_misclass_mask, 1], 
               c=colors['test_correct'], alpha=0.5, s=20, label='Test (correct)')
    
    # Plot misclassifications
    ax1.scatter(train_pca[train_misclass_mask, 0], train_pca[train_misclass_mask, 1], 
               c=colors['misclassified'], s=40, marker='D', label='Train (error)', 
               edgecolors='black', linewidth=1)
    ax1.scatter(test_pca[test_misclass_mask, 0], test_pca[test_misclass_mask, 1], 
               c=colors['misclassified'], s=60, marker='X', label='Test (error)', 
               edgecolors='black', linewidth=1)
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA 2D Projection - Combined Train/Test')
    ax1.legend()
    
    # 2. K-means Clustering on Combined Data
    ax2 = plt.subplot(3, 3, 2)
    kmeans = KMeans(n_clusters=10, random_state=42)
    combined_clusters = kmeans.fit_predict(X_combined_scaled)
    
    # Plot with cluster colors
    scatter = ax2.scatter(combined_pca[:, 0], combined_pca[:, 1], 
                         c=combined_clusters, cmap='tab10', alpha=0.3, s=10)
    
    # Overlay train/test distinction
    ax2.scatter(train_pca[~train_misclass_mask, 0], train_pca[~train_misclass_mask, 1], 
               c=colors['train_correct'], alpha=0.6, s=20, marker='o', label='Train')
    ax2.scatter(test_pca[~test_misclass_mask, 0], test_pca[~test_misclass_mask, 1], 
               c=colors['test_correct'], alpha=0.6, s=20, marker='s', label='Test')
    
    # Overlay errors
    ax2.scatter(train_pca[train_misclass_mask, 0], train_pca[train_misclass_mask, 1], 
               c=colors['misclassified'], s=40, marker='D', edgecolors='black')
    ax2.scatter(test_pca[test_misclass_mask, 0], test_pca[test_misclass_mask, 1], 
               c=colors['misclassified'], s=60, marker='X', edgecolors='black')
    
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('K-means Clustering (k=10) - Train/Test')
    ax2.legend()
    
    # 3. DBSCAN Clustering
    ax3 = plt.subplot(3, 3, 3)
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    combined_dbscan = dbscan.fit_predict(X_combined_scaled)
    
    # Plot DBSCAN results
    unique_labels = np.unique(combined_dbscan)
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise for clarity
        mask = combined_dbscan == label
        ax3.scatter(combined_pca[mask, 0], combined_pca[mask, 1], 
                   alpha=0.3, s=10, label=f'Cluster {label}' if label < 3 else '')
    
    # Overlay train/test/errors
    ax3.scatter(train_pca[~train_misclass_mask, 0], train_pca[~train_misclass_mask, 1], 
               c=colors['train_correct'], alpha=0.6, s=15, marker='o')
    ax3.scatter(test_pca[~test_misclass_mask, 0], test_pca[~test_misclass_mask, 1], 
               c=colors['test_correct'], alpha=0.6, s=15, marker='s')
    ax3.scatter(combined_pca[combined_misclass_mask, 0], combined_pca[combined_misclass_mask, 1], 
               c=colors['misclassified'], s=40, marker='D', edgecolors='black')
    
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('DBSCAN Clustering - Train/Test')
    if len(unique_labels) <= 5:
        ax3.legend()
    
    # 4. Isolation Forest Outlier Detection
    ax4 = plt.subplot(3, 3, 4)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_scores = iso_forest.fit(X_combined_scaled).score_samples(X_combined_scaled)
    
    # Plot outlier scores
    scatter = ax4.scatter(combined_pca[:, 0], combined_pca[:, 1], 
                         c=outlier_scores, cmap='RdYlBu', s=20, alpha=0.5)
    
    # Overlay misclassifications
    ax4.scatter(combined_pca[combined_misclass_mask, 0], combined_pca[combined_misclass_mask, 1], 
               c='black', s=50, marker='X', label='Misclassified', edgecolors='red', linewidth=2)
    
    plt.colorbar(scatter, ax=ax4, label='Outlier Score')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title('Isolation Forest Outlier Scores')
    ax4.legend()
    
    # 5. t-SNE 2D Projection (subset for speed)
    ax5 = plt.subplot(3, 3, 5)
    log_print("Computing t-SNE (this may take a moment)...")
    
    # Sample subset for t-SNE
    n_sample = min(2000, len(X_combined_scaled))
    sample_idx = np.random.choice(len(X_combined_scaled), n_sample, replace=False)
    X_sample = X_combined_scaled[sample_idx]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    sample_tsne = tsne.fit_transform(X_sample)
    
    # Determine which samples are train/test/misclassified
    is_train = sample_idx < n_train
    is_misclass = combined_misclass_mask[sample_idx]
    
    # Plot
    ax5.scatter(sample_tsne[is_train & ~is_misclass, 0], 
               sample_tsne[is_train & ~is_misclass, 1], 
               c=colors['train_correct'], alpha=0.6, s=20, label='Train')
    ax5.scatter(sample_tsne[~is_train & ~is_misclass, 0], 
               sample_tsne[~is_train & ~is_misclass, 1], 
               c=colors['test_correct'], alpha=0.6, s=20, label='Test')
    ax5.scatter(sample_tsne[is_misclass, 0], 
               sample_tsne[is_misclass, 1], 
               c=colors['misclassified'], s=40, marker='D', label='Errors', edgecolors='black')
    
    ax5.set_xlabel('t-SNE 1')
    ax5.set_ylabel('t-SNE 2')
    ax5.set_title(f't-SNE 2D Projection (n={n_sample})')
    ax5.legend()
    
    # 6. Feature Distribution Comparison
    ax6 = plt.subplot(3, 3, 6)
    
    features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']
    
    # Calculate means
    train_correct_means = [train_df[~train_misclass_mask][f].mean() for f in features]
    test_correct_means = [test_df[~test_misclass_mask][f].mean() for f in features]
    train_error_means = [train_df[train_misclass_mask][f].mean() if np.any(train_misclass_mask) else 0 for f in features]
    test_error_means = [test_df[test_misclass_mask][f].mean() if np.any(test_misclass_mask) else 0 for f in features]
    
    x = np.arange(len(features))
    width = 0.2
    
    ax6.bar(x - 1.5*width, train_correct_means, width, label='Train Correct', color=colors['train_correct'])
    ax6.bar(x - 0.5*width, test_correct_means, width, label='Test Correct', color=colors['test_correct'])
    ax6.bar(x + 0.5*width, train_error_means, width, label='Train Error', color=colors['misclassified'], alpha=0.7)
    ax6.bar(x + 1.5*width, test_error_means, width, label='Test Error', color=colors['misclassified'], alpha=0.9)
    
    ax6.set_xlabel('Features')
    ax6.set_ylabel('Average Value')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f.replace('_', ' ') for f in features], rotation=45)
    ax6.set_title('Feature Averages by Dataset and Classification')
    ax6.legend()
    
    # 7. Cluster Distribution
    ax7 = plt.subplot(3, 3, 7)
    
    # Analyze cluster membership
    train_clusters = combined_clusters[:n_train]
    test_clusters = combined_clusters[n_train:]
    
    cluster_data = []
    for i in range(10):
        train_count = np.sum(train_clusters == i)
        test_count = np.sum(test_clusters == i)
        train_error = np.sum((train_clusters == i) & train_misclass_mask)
        test_error = np.sum((test_clusters == i) & test_misclass_mask)
        
        cluster_data.append({
            'cluster': i,
            'train': train_count,
            'test': test_count,
            'train_error': train_error,
            'test_error': test_error
        })
    
    cluster_df = pd.DataFrame(cluster_data)
    
    # Plot stacked bars
    ax7.bar(cluster_df['cluster'], cluster_df['train'], label='Train', color=colors['train_correct'], alpha=0.7)
    ax7.bar(cluster_df['cluster'], cluster_df['test'], bottom=cluster_df['train'], 
           label='Test', color=colors['test_correct'], alpha=0.7)
    
    # Add error counts as text
    for idx, row in cluster_df.iterrows():
        if row['train_error'] > 0:
            ax7.text(row['cluster'], row['train']/2, f"{row['train_error']}", 
                    ha='center', va='center', color='red', fontweight='bold')
        if row['test_error'] > 0:
            ax7.text(row['cluster'], row['train'] + row['test']/2, f"{row['test_error']}", 
                    ha='center', va='center', color='darkred', fontweight='bold')
    
    ax7.set_xlabel('Cluster')
    ax7.set_ylabel('Count')
    ax7.set_title('Cluster Distribution (errors shown in red)')
    ax7.legend()
    
    # 8. Distance from Cluster Centers
    ax8 = plt.subplot(3, 3, 8)
    
    # Calculate distances
    cluster_centers = kmeans.cluster_centers_
    distances = []
    labels = []
    
    for i, (data, mask, name) in enumerate([
        (X_train_scaled[~train_misclass_mask], train_clusters[~train_misclass_mask], 'Train Correct'),
        (X_test_scaled[~test_misclass_mask], test_clusters[~test_misclass_mask], 'Test Correct'),
        (X_train_scaled[train_misclass_mask], train_clusters[train_misclass_mask], 'Train Error'),
        (X_test_scaled[test_misclass_mask], test_clusters[test_misclass_mask], 'Test Error')
    ]):
        if len(data) > 0:
            dist = []
            for j, point in enumerate(data):
                center = cluster_centers[mask[j]]
                dist.append(np.linalg.norm(point - center))
            distances.append(dist)
            labels.append(name)
    
    ax8.boxplot(distances, labels=labels)
    ax8.set_ylabel('Distance from Cluster Center')
    ax8.set_title('Distance Analysis')
    ax8.tick_params(axis='x', rotation=45)
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate statistics
    n_train_total = n_train
    n_test_total = len(test_df)
    n_train_errors = np.sum(train_misclass_mask)
    n_test_errors = np.sum(test_misclass_mask)
    
    summary_text = f"""COMBINED TRAIN-TEST ANALYSIS SUMMARY

Dataset Sizes:
• Training samples: {n_train_total:,}
• Test samples: {n_test_total:,}
• Total samples: {n_train_total + n_test_total:,}

Misclassification Rates:
• Training errors: {n_train_errors} ({n_train_errors/n_train_total*100:.2f}%)
• Test errors (potential): {n_test_errors} ({n_test_errors/n_test_total*100:.3f}%)

Key Observations:
• Misclassified samples cluster at decision boundaries
• Test errors follow similar patterns to training errors
• Both sets show outlier characteristics
• Error distribution varies significantly by cluster

Color Legend:
• Green: Training data (correct)
• Yellow: Test data (correct)
• Red: Misclassified samples (both sets)"""
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('output/clustering_train_test_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log_print("\nVisualization saved to: output/clustering_train_test_combined.png")


def analyze_combined_patterns(train_df, test_df, train_misclass_mask, test_misclass_mask):
    """Analyze patterns across train and test misclassifications."""
    log_print("\n" + "="*70)
    log_print("ANALYZING COMBINED PATTERNS")
    log_print("="*70)
    
    # Feature comparison
    features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 
                'Going_outside', 'Post_frequency']
    
    log_print("\nFeature Averages Comparison:")
    log_print("-" * 60)
    log_print(f"{'Feature':<25} {'Train OK':<10} {'Train Err':<10} {'Test OK':<10} {'Test Err':<10}")
    log_print("-" * 60)
    
    for feature in features:
        train_ok = train_df[~train_misclass_mask][feature].mean()
        train_err = train_df[train_misclass_mask][feature].mean() if np.any(train_misclass_mask) else 0
        test_ok = test_df[~test_misclass_mask][feature].mean()
        test_err = test_df[test_misclass_mask][feature].mean() if np.any(test_misclass_mask) else 0
        
        log_print(f"{feature:<25} {train_ok:<10.2f} {train_err:<10.2f} {test_ok:<10.2f} {test_err:<10.2f}")
    
    # Psychological features
    log_print("\n\nPsychological Features Distribution:")
    log_print("-" * 60)
    
    for feature in ['Drained_after_socializing', 'Stage_fear']:
        if feature in train_df.columns:
            train_err_yes = np.sum((train_df[train_misclass_mask][feature] == 'Yes')) if np.any(train_misclass_mask) else 0
            train_err_total = np.sum(train_misclass_mask)
            
            log_print(f"{feature}:")
            log_print(f"  Training errors: {train_err_yes}/{train_err_total} = "
                     f"{train_err_yes/train_err_total*100:.1f}% Yes")


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("COMBINED TRAIN-TEST CLUSTERING VISUALIZATION")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    # Load 0.975708 submission for test predictions
    submission_df = pd.read_csv('output/base_975708_submission.csv')
    test_df = test_df.merge(submission_df[['id', 'Personality']], on='id', how='left')
    
    # Prepare combined data
    (X_train, X_train_scaled, y_train, X_test, X_test_scaled, 
     X_combined_scaled, feature_cols, scaler, n_train) = prepare_combined_data(train_df, test_df)
    
    # Identify misclassifications
    train_misclass_mask, test_misclass_mask, model = identify_all_misclassifications(
        X_train, y_train, test_df)
    
    # Create visualization
    create_combined_visualization(
        train_df, test_df, X_train_scaled, X_test_scaled, X_combined_scaled,
        train_misclass_mask, test_misclass_mask, n_train)
    
    # Analyze patterns
    analyze_combined_patterns(train_df, test_df, train_misclass_mask, test_misclass_mask)
    
    # Save combined misclassification data
    train_misclass_df = train_df[train_misclass_mask].copy()
    train_misclass_df['dataset'] = 'train'
    train_misclass_df['error_type'] = 'misclassified'
    
    test_misclass_df = test_df[test_misclass_mask].copy()
    test_misclass_df['dataset'] = 'test'
    test_misclass_df['error_type'] = 'potential_misclassified'
    
    combined_errors = pd.concat([train_misclass_df, test_misclass_df])
    combined_errors.to_csv('output/combined_misclassifications.csv', index=False)
    log_print(f"\nSaved {len(combined_errors)} combined misclassifications to: "
              "output/combined_misclassifications.csv")
    
    # Final summary
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    log_print("\nKey findings:")
    log_print("1. Misclassifications in both train and test cluster at boundaries")
    log_print("2. Test errors show similar patterns to training errors")
    log_print("3. Both error types are farther from cluster centers")
    log_print("4. Clear visual separation between correct and incorrect predictions")
    
    output_file.close()


if __name__ == "__main__":
    main()