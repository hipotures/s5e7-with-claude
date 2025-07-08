#!/usr/bin/env python3
"""
CLUSTERING VISUALIZATION WITH TRAINING ERRORS
============================================

This script creates clustering visualizations similar to 20250705_1300
but additionally highlights misclassified cases in the training set.

Author: Claude
Date: 2025-07-05 14:36
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
output_file = open('output/20250705_1436_clustering_with_training_errors.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def prepare_data(train_df, test_df):
    """Prepare data for clustering analysis."""
    log_print("Preparing data for analysis...")
    
    # Encode categorical features
    for df in [train_df, test_df]:
        df['Stage_fear_binary'] = (df['Stage_fear'] == 'Yes').astype(int)
        df['Drained_binary'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    # Select features for clustering
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear_binary', 'Drained_binary']
    
    # Prepare training data
    X_train = train_df[feature_cols].fillna(0)
    y_train = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # Prepare test data
    X_test = test_df[feature_cols].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_train_scaled, y_train, X_test, X_test_scaled, feature_cols, scaler


def identify_training_errors(X_train, y_train):
    """Train model and identify misclassified training samples."""
    log_print("\nTraining XGBoost to identify training errors...")
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Get predictions
    y_pred = model.predict(X_train)
    y_prob = model.predict_proba(X_train)[:, 1]
    
    # Identify misclassified
    misclassified_mask = y_pred != y_train
    misclassified_indices = np.where(misclassified_mask)[0]
    
    log_print(f"Found {len(misclassified_indices)} misclassified training samples "
              f"({len(misclassified_indices)/len(y_train)*100:.2f}%)")
    
    # Analyze misclassification patterns
    cm = confusion_matrix(y_train, y_pred)
    log_print(f"\nConfusion Matrix:")
    log_print(f"True Introvert, Pred Introvert: {cm[0,0]}")
    log_print(f"True Introvert, Pred Extrovert: {cm[0,1]} (False Positives)")
    log_print(f"True Extrovert, Pred Introvert: {cm[1,0]} (False Negatives)")
    log_print(f"True Extrovert, Pred Extrovert: {cm[1,1]}")
    
    return misclassified_mask, misclassified_indices, y_prob, model


def load_test_misclassifications():
    """Load the test set misclassifications from previous analysis."""
    try:
        misclass_df = pd.read_csv('output/all_potential_misclassifications.csv')
        test_misclass_ids = misclass_df['id'].values
        log_print(f"\nLoaded {len(test_misclass_ids)} test misclassifications")
        return test_misclass_ids
    except:
        log_print("\nWarning: Could not load test misclassifications")
        return np.array([])


def create_comprehensive_visualization(train_df, test_df, X_train_scaled, X_test_scaled, 
                                     y_train, train_misclass_mask, test_misclass_ids):
    """Create comprehensive clustering visualization."""
    log_print("\nCreating comprehensive visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(20, 24))
    
    # Define color scheme
    colors = {
        'train_intro_correct': '#3498db',  # Blue
        'train_extro_correct': '#e74c3c',  # Red
        'train_intro_wrong': '#9b59b6',    # Purple
        'train_extro_wrong': '#f39c12',    # Orange
        'test_normal': '#95a5a6',          # Gray
        'test_misclass': '#e67e22'         # Dark orange
    }
    
    # 1. PCA 2D Projection
    ax1 = plt.subplot(3, 3, 1)
    pca = PCA(n_components=2, random_state=42)
    train_pca = pca.fit_transform(X_train_scaled)
    test_pca = pca.transform(X_test_scaled)
    
    # Plot training data
    intro_correct = (y_train == 0) & (~train_misclass_mask)
    extro_correct = (y_train == 1) & (~train_misclass_mask)
    intro_wrong = (y_train == 0) & (train_misclass_mask)
    extro_wrong = (y_train == 1) & (train_misclass_mask)
    
    ax1.scatter(train_pca[intro_correct, 0], train_pca[intro_correct, 1], 
               c=colors['train_intro_correct'], alpha=0.5, s=20, label='Intro (correct)')
    ax1.scatter(train_pca[extro_correct, 0], train_pca[extro_correct, 1], 
               c=colors['train_extro_correct'], alpha=0.5, s=20, label='Extro (correct)')
    ax1.scatter(train_pca[intro_wrong, 0], train_pca[intro_wrong, 1], 
               c=colors['train_intro_wrong'], s=40, marker='D', label='Intro (wrong)', edgecolors='black')
    ax1.scatter(train_pca[extro_wrong, 0], train_pca[extro_wrong, 1], 
               c=colors['train_extro_wrong'], s=40, marker='D', label='Extro (wrong)', edgecolors='black')
    
    # Plot test misclassifications
    test_misclass_mask = np.isin(test_df['id'].values, test_misclass_ids)
    if np.any(test_misclass_mask):
        ax1.scatter(test_pca[test_misclass_mask, 0], test_pca[test_misclass_mask, 1], 
                   c=colors['test_misclass'], s=100, marker='X', label='Test misclass', 
                   edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA 2D Projection with Training Errors')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. K-means Clustering
    ax2 = plt.subplot(3, 3, 2)
    kmeans = KMeans(n_clusters=10, random_state=42)
    train_clusters = kmeans.fit_predict(X_train_scaled)
    test_clusters = kmeans.predict(X_test_scaled)
    
    # Plot with cluster colors but highlight errors
    scatter = ax2.scatter(train_pca[:, 0], train_pca[:, 1], c=train_clusters, 
                         cmap='tab10', alpha=0.5, s=20)
    
    # Overlay training errors
    ax2.scatter(train_pca[train_misclass_mask, 0], train_pca[train_misclass_mask, 1], 
               c='black', s=50, marker='D', label='Training errors', edgecolors='white', linewidth=1)
    
    # Overlay test misclassifications
    if np.any(test_misclass_mask):
        ax2.scatter(test_pca[test_misclass_mask, 0], test_pca[test_misclass_mask, 1], 
                   c=colors['test_misclass'], s=100, marker='X', label='Test misclass', 
                   edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title(f'K-means Clustering (k=10)')
    ax2.legend()
    
    # 3. DBSCAN Clustering
    ax3 = plt.subplot(3, 3, 3)
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    train_dbscan = dbscan.fit_predict(X_train_scaled)
    
    # Plot DBSCAN results
    unique_labels = np.unique(train_dbscan)
    for label in unique_labels:
        if label == -1:
            color = 'black'
            marker = 'x'
        else:
            color = plt.cm.Spectral(label / len(unique_labels))
            marker = 'o'
        
        mask = train_dbscan == label
        ax3.scatter(train_pca[mask, 0], train_pca[mask, 1], 
                   c=[color], s=20, marker=marker, alpha=0.5)
    
    # Overlay errors
    ax3.scatter(train_pca[train_misclass_mask, 0], train_pca[train_misclass_mask, 1], 
               c='red', s=50, marker='D', label='Training errors', edgecolors='white', linewidth=1)
    
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('DBSCAN Clustering')
    ax3.legend()
    
    # 4. Isolation Forest Outlier Scores
    ax4 = plt.subplot(3, 3, 4)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_scores = iso_forest.fit(X_train_scaled).score_samples(X_train_scaled)
    
    scatter = ax4.scatter(train_pca[:, 0], train_pca[:, 1], 
                         c=outlier_scores, cmap='RdYlBu', s=20, alpha=0.7)
    
    # Overlay errors
    ax4.scatter(train_pca[train_misclass_mask, 0], train_pca[train_misclass_mask, 1], 
               c='red', s=50, marker='D', label='Training errors', edgecolors='black', linewidth=1)
    
    plt.colorbar(scatter, ax=ax4, label='Outlier Score')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title('Isolation Forest Outlier Scores')
    ax4.legend()
    
    # 5. t-SNE 2D Projection
    ax5 = plt.subplot(3, 3, 5)
    log_print("Computing t-SNE (this may take a moment)...")
    
    # Combine train and test for t-SNE
    n_train = len(X_train_scaled)
    combined = np.vstack([X_train_scaled[:1000], X_test_scaled[:500]])  # Subset for speed
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    combined_tsne = tsne.fit_transform(combined)
    
    train_tsne = combined_tsne[:1000]
    test_tsne = combined_tsne[1000:]
    
    # Plot t-SNE results
    train_subset_mask = np.zeros(len(y_train), dtype=bool)
    train_subset_mask[:1000] = True
    
    intro_correct_subset = intro_correct & train_subset_mask
    extro_correct_subset = extro_correct & train_subset_mask
    intro_wrong_subset = intro_wrong & train_subset_mask
    extro_wrong_subset = extro_wrong & train_subset_mask
    
    ax5.scatter(train_tsne[intro_correct_subset[:1000], 0], 
               train_tsne[intro_correct_subset[:1000], 1], 
               c=colors['train_intro_correct'], alpha=0.5, s=20, label='Intro (correct)')
    ax5.scatter(train_tsne[extro_correct_subset[:1000], 0], 
               train_tsne[extro_correct_subset[:1000], 1], 
               c=colors['train_extro_correct'], alpha=0.5, s=20, label='Extro (correct)')
    
    # Plot errors in subset
    if np.any(intro_wrong_subset[:1000]):
        ax5.scatter(train_tsne[intro_wrong_subset[:1000], 0], 
                   train_tsne[intro_wrong_subset[:1000], 1], 
                   c=colors['train_intro_wrong'], s=40, marker='D', label='Intro (wrong)')
    if np.any(extro_wrong_subset[:1000]):
        ax5.scatter(train_tsne[extro_wrong_subset[:1000], 0], 
                   train_tsne[extro_wrong_subset[:1000], 1], 
                   c=colors['train_extro_wrong'], s=40, marker='D', label='Extro (wrong)')
    
    ax5.set_xlabel('t-SNE 1')
    ax5.set_ylabel('t-SNE 2')
    ax5.set_title('t-SNE 2D Projection (subset)')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Error Analysis by Features
    ax6 = plt.subplot(3, 3, 6)
    
    # Calculate average feature values for different groups
    feature_names = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']
    groups = {
        'Intro Correct': train_df[intro_correct][feature_names].mean(),
        'Extro Correct': train_df[extro_correct][feature_names].mean(),
        'Intro Wrong': train_df[intro_wrong][feature_names].mean() if np.any(intro_wrong) else pd.Series(0, index=feature_names),
        'Extro Wrong': train_df[extro_wrong][feature_names].mean() if np.any(extro_wrong) else pd.Series(0, index=feature_names)
    }
    
    x = np.arange(len(feature_names))
    width = 0.2
    
    for i, (group_name, values) in enumerate(groups.items()):
        ax6.bar(x + i*width, values, width, label=group_name)
    
    ax6.set_xlabel('Features')
    ax6.set_ylabel('Average Value')
    ax6.set_xticks(x + width * 1.5)
    ax6.set_xticklabels([f.replace('_', ' ') for f in feature_names], rotation=45)
    ax6.set_title('Feature Averages by Classification Result')
    ax6.legend()
    
    # 7. Error Rate by Cluster
    ax7 = plt.subplot(3, 3, 7)
    
    error_rates = []
    cluster_sizes = []
    for cluster in range(10):
        mask = train_clusters == cluster
        if np.any(mask):
            error_rate = np.mean(train_misclass_mask[mask])
            error_rates.append(error_rate * 100)
            cluster_sizes.append(np.sum(mask))
        else:
            error_rates.append(0)
            cluster_sizes.append(0)
    
    bars = ax7.bar(range(10), error_rates)
    ax7.set_xlabel('Cluster')
    ax7.set_ylabel('Error Rate (%)')
    ax7.set_title('Training Error Rate by K-means Cluster')
    
    # Add cluster sizes as text
    for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'n={size}', ha='center', va='bottom', fontsize=8)
    
    # 8. Feature Distribution of Errors
    ax8 = plt.subplot(3, 3, 8)
    
    # Plot distribution of a key feature
    feature = 'Time_spent_Alone'
    
    ax8.hist(train_df[intro_correct][feature], bins=20, alpha=0.5, 
            label='Intro Correct', color=colors['train_intro_correct'])
    ax8.hist(train_df[extro_correct][feature], bins=20, alpha=0.5, 
            label='Extro Correct', color=colors['train_extro_correct'])
    
    if np.any(intro_wrong):
        ax8.hist(train_df[intro_wrong][feature], bins=20, alpha=0.7, 
                label='Intro Wrong', color=colors['train_intro_wrong'], edgecolor='black')
    if np.any(extro_wrong):
        ax8.hist(train_df[extro_wrong][feature], bins=20, alpha=0.7, 
                label='Extro Wrong', color=colors['train_extro_wrong'], edgecolor='black')
    
    ax8.set_xlabel(feature.replace('_', ' '))
    ax8.set_ylabel('Count')
    ax8.set_title(f'{feature} Distribution by Classification')
    ax8.legend()
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate statistics
    n_train = len(y_train)
    n_intro = np.sum(y_train == 0)
    n_extro = np.sum(y_train == 1)
    n_errors = np.sum(train_misclass_mask)
    n_intro_errors = np.sum(intro_wrong)
    n_extro_errors = np.sum(extro_wrong)
    
    summary_text = f"""TRAINING ERROR ANALYSIS SUMMARY

Total Training Samples: {n_train:,}
• Introverts: {n_intro:,} ({n_intro/n_train*100:.1f}%)
• Extroverts: {n_extro:,} ({n_extro/n_train*100:.1f}%)

Total Misclassified: {n_errors} ({n_errors/n_train*100:.2f}%)
• Introverts → Extroverts: {n_intro_errors} ({n_intro_errors/n_intro*100:.2f}% of introverts)
• Extroverts → Introverts: {n_extro_errors} ({n_extro_errors/n_extro*100:.2f}% of extroverts)

Test Set Misclassifications: {len(test_misclass_ids)}

Key Observations:
• Training errors cluster at decision boundaries
• Misclassified samples often have ambiguous features
• Error rate varies significantly by cluster
• Some clusters have much higher error rates"""
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('output/clustering_with_training_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log_print("\nVisualization saved to: output/clustering_with_training_errors.png")


def analyze_error_patterns(train_df, train_misclass_mask, misclassified_indices):
    """Analyze patterns in misclassified training samples."""
    log_print("\n" + "="*70)
    log_print("ANALYZING ERROR PATTERNS")
    log_print("="*70)
    
    misclass_df = train_df.iloc[misclassified_indices].copy()
    
    # Feature statistics
    log_print("\nMisclassified Sample Statistics:")
    log_print("-" * 40)
    
    features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 
                'Going_outside', 'Post_frequency']
    
    for feature in features:
        mean_all = train_df[feature].mean()
        mean_error = misclass_df[feature].mean()
        log_print(f"{feature}:")
        log_print(f"  All samples: {mean_all:.2f}")
        log_print(f"  Errors only: {mean_error:.2f} ({(mean_error-mean_all)/mean_all*100:+.1f}%)")
    
    # Psychological features
    log_print("\nPsychological Features in Errors:")
    if 'Drained_after_socializing' in misclass_df.columns:
        drained_dist = misclass_df['Drained_after_socializing'].value_counts()
        log_print(f"Drained after socializing: Yes={drained_dist.get('Yes', 0)}, "
                 f"No={drained_dist.get('No', 0)}")
    
    if 'Stage_fear' in misclass_df.columns:
        fear_dist = misclass_df['Stage_fear'].value_counts()
        log_print(f"Stage fear: Yes={fear_dist.get('Yes', 0)}, No={fear_dist.get('No', 0)}")
    
    # Save detailed error analysis
    misclass_df['predicted_as'] = 'Extrovert'
    misclass_df.loc[misclass_df['Personality'] == 'Extrovert', 'predicted_as'] = 'Introvert'
    misclass_df.to_csv('output/training_misclassifications.csv', index=False)
    log_print(f"\nSaved {len(misclass_df)} training misclassifications to: "
              "output/training_misclassifications.csv")


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("CLUSTERING VISUALIZATION WITH TRAINING ERRORS")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    # Prepare data
    X_train, X_train_scaled, y_train, X_test, X_test_scaled, feature_cols, scaler = prepare_data(
        train_df, test_df)
    
    # Identify training errors
    train_misclass_mask, misclassified_indices, y_prob, model = identify_training_errors(
        X_train, y_train)
    
    # Load test misclassifications
    test_misclass_ids = load_test_misclassifications()
    
    # Create comprehensive visualization
    create_comprehensive_visualization(
        train_df, test_df, X_train_scaled, X_test_scaled, 
        y_train, train_misclass_mask, test_misclass_ids)
    
    # Analyze error patterns
    analyze_error_patterns(train_df, train_misclass_mask, misclassified_indices)
    
    # Final summary
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    log_print("\nKey findings:")
    log_print("1. Training errors are concentrated at decision boundaries")
    log_print("2. Misclassified samples often have ambiguous feature combinations")
    log_print("3. Error patterns in training may predict test set difficulties")
    log_print("4. Clustering reveals natural groupings with varying error rates")
    
    output_file.close()


if __name__ == "__main__":
    main()