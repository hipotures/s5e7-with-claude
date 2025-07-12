#!/usr/bin/env python3
"""
Visualize clustering of kept, removed, and test samples
Multiple clustering/dimensionality reduction methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
# from umap import UMAP  # Skip if not installed
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"

def load_and_prepare_data():
    """Load all data and prepare features"""
    print("="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)
    
    # Load original data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Load removed samples info
    removed_info = pd.read_csv(OUTPUT_DIR / "removed_hard_cases_info.csv")
    removed_ids = set(removed_info['id'].values)
    
    # Create masks
    train_df['is_removed'] = train_df['id'].isin(removed_ids)
    
    # Split train into removed and kept
    removed_df = train_df[train_df['is_removed']].copy()
    kept_df = train_df[~train_df['is_removed']].copy()
    
    print(f"Kept (train-removed): {len(kept_df)} samples")
    print(f"Removed: {len(removed_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    def prepare_features(df):
        X = df[feature_cols].copy()
        # Convert binary features
        for col in ['Stage_fear', 'Drained_after_socializing']:
            X[col] = (X[col] == 'Yes').astype(int)
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
        return X
    
    X_kept = prepare_features(kept_df)
    X_removed = prepare_features(removed_df)
    X_test = prepare_features(test_df)
    
    # Combine and scale
    X_all = pd.concat([X_kept, X_removed, X_test])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    # Split back
    n_kept = len(X_kept)
    n_removed = len(X_removed)
    
    X_kept_scaled = X_scaled[:n_kept]
    X_removed_scaled = X_scaled[n_kept:n_kept+n_removed]
    X_test_scaled = X_scaled[n_kept+n_removed:]
    
    # Create labels for coloring
    labels = np.array(['Kept']*n_kept + ['Removed']*n_removed + ['Test']*len(X_test))
    
    return X_scaled, labels, n_kept, n_removed

def create_clustering_visualizations(X_scaled, labels):
    """Create multiple clustering visualizations"""
    print("\n" + "="*60)
    print("CREATING CLUSTERING VISUALIZATIONS")
    print("="*60)
    
    # Set up colors
    color_map = {'Kept': '#3498db', 'Removed': '#e74c3c', 'Test': '#f39c12'}  # Changed test to orange
    colors = [color_map[label] for label in labels]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. PCA
    print("1. Running PCA...")
    ax = plt.subplot(4, 2, 1)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    for label in ['Kept', 'Removed', 'Test']:
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=color_map[label], label=label, alpha=0.6, s=20)
    
    ax.set_title('PCA Visualization', fontsize=16)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. t-SNE
    print("2. Running t-SNE...")
    ax = plt.subplot(4, 2, 2)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    for label in ['Kept', 'Removed', 'Test']:
        mask = labels == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                  c=color_map[label], label=label, alpha=0.6, s=20)
    
    ax.set_title('t-SNE Visualization', fontsize=16)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. PCA with 3 components projected to 2D
    print("3. Running PCA-3D...")
    ax = plt.subplot(4, 2, 3)
    pca3 = PCA(n_components=3, random_state=42)
    X_pca3 = pca3.fit_transform(X_scaled)
    
    for label in ['Kept', 'Removed', 'Test']:
        mask = labels == label
        ax.scatter(X_pca3[mask, 0], X_pca3[mask, 2], 
                  c=color_map[label], label=label, alpha=0.6, s=20)
    
    ax.set_title('PCA Visualization (PC1 vs PC3)', fontsize=16)
    ax.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC3 ({pca3.explained_variance_ratio_[2]:.1%} variance)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. K-Means (k=3)
    print("4. Running K-Means...")
    ax = plt.subplot(4, 2, 4)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Use PCA for visualization
    for label in ['Kept', 'Removed', 'Test']:
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=color_map[label], label=label, alpha=0.6, s=20,
                  edgecolors='black', linewidth=0.5)
    
    # Add cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
              c='black', marker='x', s=200, linewidths=3)
    
    ax.set_title('K-Means Clustering (k=3)', fontsize=16)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. DBSCAN
    print("5. Running DBSCAN...")
    ax = plt.subplot(4, 2, 5)
    dbscan = DBSCAN(eps=1.5, min_samples=50)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    for label in ['Kept', 'Removed', 'Test']:
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=color_map[label], label=label, alpha=0.6, s=20)
    
    # Highlight outliers
    outliers = dbscan_labels == -1
    if outliers.sum() > 0:
        ax.scatter(X_pca[outliers, 0], X_pca[outliers, 1], 
                  c='black', marker='x', s=50, label=f'Outliers ({outliers.sum()})')
    
    ax.set_title('DBSCAN Clustering', fontsize=16)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Gaussian Mixture Model
    print("6. Running GMM...")
    ax = plt.subplot(4, 2, 6)
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm_probs = gmm.fit_predict(X_scaled)
    
    for label in ['Kept', 'Removed', 'Test']:
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=color_map[label], label=label, alpha=0.6, s=20)
    
    ax.set_title('Gaussian Mixture Model', fontsize=16)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Hierarchical Clustering
    print("7. Running Hierarchical Clustering...")
    ax = plt.subplot(4, 2, 7)
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hier_labels = hierarchical.fit_predict(X_scaled)
    
    for label in ['Kept', 'Removed', 'Test']:
        mask = labels == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                  c=color_map[label], label=label, alpha=0.6, s=20)
    
    ax.set_title('Hierarchical Clustering (on t-SNE)', fontsize=16)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Summary statistics
    ax = plt.subplot(4, 2, 8)
    ax.axis('off')
    
    # Calculate overlap statistics
    kept_mask = labels == 'Kept'
    removed_mask = labels == 'Removed'
    test_mask = labels == 'Test'
    
    # Count samples in each cluster for K-means
    cluster_dist = pd.DataFrame({
        'Cluster': range(3),
        'Kept': [((labels[kmeans_labels == i] == 'Kept').sum()) for i in range(3)],
        'Removed': [((labels[kmeans_labels == i] == 'Removed').sum()) for i in range(3)],
        'Test': [((labels[kmeans_labels == i] == 'Test').sum()) for i in range(3)]
    })
    
    summary_text = f"""
    Summary Statistics:
    
    Dataset Sizes:
    - Kept (train-removed): {kept_mask.sum():,}
    - Removed: {removed_mask.sum():,}
    - Test: {test_mask.sum():,}
    
    K-Means Cluster Distribution:
    {cluster_dist.to_string(index=False)}
    
    DBSCAN Results:
    - Number of clusters: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}
    - Outliers detected: {(dbscan_labels == -1).sum()}
    - Outliers by group:
      * Kept: {(dbscan_labels[kept_mask] == -1).sum()}
      * Removed: {(dbscan_labels[removed_mask] == -1).sum()}
      * Test: {(dbscan_labels[test_mask] == -1).sum()}
    
    PCA Variance Explained:
    - PC1: {pca.explained_variance_ratio_[0]:.1%}
    - PC2: {pca.explained_variance_ratio_[1]:.1%}
    - Total: {sum(pca.explained_variance_ratio_):.1%}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Clustering Analysis: Kept vs Removed vs Test', fontsize=20, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'clustering_visualizations_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {OUTPUT_DIR / 'clustering_visualizations_all_methods.png'}")
    
    return X_pca, X_tsne, kmeans_labels, dbscan_labels

def create_individual_plots(X_scaled, labels):
    """Create individual high-resolution plots for each method"""
    print("\n" + "="*60)
    print("CREATING INDIVIDUAL HIGH-RESOLUTION PLOTS")
    print("="*60)
    
    color_map = {'Kept': '#3498db', 'Removed': '#e74c3c', 'Test': '#f39c12'}  # Changed test to orange
    
    methods = [
        ('PCA', PCA(n_components=2, random_state=42)),
        ('t-SNE', TSNE(n_components=2, perplexity=30, random_state=42))
    ]
    
    for method_name, method in methods:
        print(f"\nCreating {method_name} plot...")
        
        # Transform data
        if method_name == 'PCA':
            X_transformed = method.fit_transform(X_scaled)
            xlabel = f'{method_name} 1 ({method.explained_variance_ratio_[0]:.1%} var)'
            ylabel = f'{method_name} 2 ({method.explained_variance_ratio_[1]:.1%} var)'
        else:
            X_transformed = method.fit_transform(X_scaled)
            xlabel = f'{method_name} 1'
            ylabel = f'{method_name} 2'
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot each group
        for label in ['Test', 'Kept', 'Removed']:  # Order matters for layering
            mask = labels == label
            plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                       c=color_map[label], label=label, 
                       alpha=0.6 if label != 'Removed' else 0.8,
                       s=30 if label != 'Removed' else 40,
                       edgecolors='black' if label == 'Removed' else 'none',
                       linewidth=0.5 if label == 'Removed' else 0)
        
        plt.title(f'{method_name} Visualization', fontsize=20)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add density contours
        from scipy.stats import gaussian_kde
        
        for label, color in [('Kept', 'blue'), ('Test', 'orange')]:
            mask = labels == label
            if mask.sum() > 100:  # Need enough points for KDE
                xy = X_transformed[mask]
                try:
                    kde = gaussian_kde(xy.T)
                    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
                    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    f = np.reshape(kde(positions).T, xx.shape)
                    plt.contour(xx, yy, f, colors=color, alpha=0.3, linewidths=1)
                except:
                    pass
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'clustering_{method_name.lower()}_detailed.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: clustering_{method_name.lower()}_detailed.png")

def analyze_separation():
    """Analyze how well separated the groups are"""
    print("\n" + "="*60)
    print("SEPARATION ANALYSIS")
    print("="*60)
    
    # This would include metrics like silhouette score, etc.
    # But for now, visual inspection is most important
    
    print("\nKey observations from visualizations:")
    print("1. Check if removed samples form distinct clusters")
    print("2. Check if test samples overlap more with kept or removed")
    print("3. Look for clear boundaries between groups")
    print("4. Identify which method best separates the groups")

def main():
    # Load and prepare data
    X_scaled, labels, n_kept, n_removed = load_and_prepare_data()
    
    # Create main visualization with all methods
    X_pca, X_tsne, kmeans_labels, dbscan_labels = create_clustering_visualizations(X_scaled, labels)
    
    # Create individual high-res plots
    create_individual_plots(X_scaled, labels)
    
    # Analyze separation
    analyze_separation()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll visualizations saved to {OUTPUT_DIR}")
    print("\nLook for:")
    print("- Whether removed samples cluster separately")
    print("- Which group (kept/removed) test samples are closer to")
    print("- Any clear patterns in the clustering")

if __name__ == "__main__":
    main()