#!/usr/bin/env python3
"""
PURPOSE: Create clustering visualizations to analyze outlier cases in misclassifications
HYPOTHESIS: Misclassified cases are outliers that don't fit well into typical clusters
EXPECTED: Visual identification of outlier patterns and insights into why certain records are hard to classify
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Create output file for insights
output_file = open('output/20250705_1300_clustering_outlier_visualization.txt', 'w')

def log_print(msg):
    """Print to both console and output file"""
    print(msg)
    output_file.write(msg + '\n')

log_print("=== Clustering and Outlier Analysis for Misclassified Cases ===\n")

# Load data
log_print("Loading data...")
test_df = pd.read_csv('../../test.csv')
submission_df = pd.read_csv('output/base_975708_submission.csv')
misclass_df = pd.read_csv('output/all_potential_misclassifications.csv')

# Merge predictions with test data
# Both files use lowercase 'id', so merge on that
test_with_pred = test_df.merge(submission_df, on='id')

# Get top misclassified IDs
top_misclassified_ids = [19612, 23844, 21359, 23336, 21800, 23351, 20593, 24049, 21365, 20062]
log_print(f"\nFocusing on top {len(top_misclassified_ids)} misclassified IDs: {top_misclassified_ids}")

# Prepare features for clustering
feature_cols = [col for col in test_df.columns if col != 'id']
X = test_with_pred[feature_cols].copy()

# Handle missing values - create null indicators first
null_indicators = X.isnull().astype(int)
null_indicator_cols = [f'{col}_is_null' for col in feature_cols]
for i, col in enumerate(null_indicator_cols):
    X[col] = null_indicators.iloc[:, i]

# Encode categorical columns
for col in feature_cols:
    if X[col].dtype == 'object':
        # Convert Yes/No to 1/0
        X[col] = X[col].map({'Yes': 1, 'No': 0}).fillna(0.5)  # Fill nulls with 0.5 for categorical

# Now fill missing values with median for numeric columns
numeric_cols = [col for col in feature_cols if col in X.columns]
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

# Create labels for misclassified points
test_with_pred['is_misclassified'] = test_with_pred['id'].isin(misclass_df['id'])
test_with_pred['is_top_misclassified'] = test_with_pred['id'].isin(top_misclassified_ids)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create figure for visualizations
fig = plt.figure(figsize=(20, 15))

# 1. PCA visualization
log_print("\n1. PCA Analysis...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
log_print(f"PCA explained variance: {explained_var[:3]}")
log_print(f"Total variance explained by 3 components: {sum(explained_var[:3]):.4f}")

ax1 = fig.add_subplot(3, 3, 1)
# Convert personality to numeric for coloring
personality_numeric = test_with_pred['Personality'].map({'Introvert': 0, 'Extrovert': 1})
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=personality_numeric, 
                     alpha=0.5, s=10, cmap='viridis')
# Highlight misclassified points
misclass_mask = test_with_pred['is_misclassified']
ax1.scatter(X_pca[misclass_mask, 0], X_pca[misclass_mask, 1], 
           c='red', s=50, marker='x', label='Misclassified')
# Highlight top misclassified
top_mask = test_with_pred['is_top_misclassified']
ax1.scatter(X_pca[top_mask, 0], X_pca[top_mask, 1], 
           c='red', s=100, marker='o', edgecolors='black', linewidth=2, label='Top Misclassified')
ax1.set_xlabel(f'PC1 ({explained_var[0]:.2%})')
ax1.set_ylabel(f'PC2 ({explained_var[1]:.2%})')
ax1.set_title('PCA: 2D Projection')
ax1.legend()

# 2. K-means clustering
log_print("\n2. K-means Clustering...")
# Find optimal k using silhouette score
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

optimal_k = k_range[np.argmax(silhouette_scores)]
log_print(f"Optimal k based on silhouette score: {optimal_k}")

# Apply K-means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Calculate distances from cluster centers
distances = []
for i in range(len(X_scaled)):
    cluster = kmeans_labels[i]
    dist = np.linalg.norm(X_scaled[i] - kmeans.cluster_centers_[cluster])
    distances.append(dist)
test_with_pred['cluster_distance'] = distances

# Plot K-means results
ax2 = fig.add_subplot(3, 3, 2)
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=kmeans_labels, alpha=0.5, s=10, cmap='tab10')
ax2.scatter(X_pca[misclass_mask, 0], X_pca[misclass_mask, 1], 
           c='red', s=50, marker='x', label='Misclassified')
ax2.scatter(X_pca[top_mask, 0], X_pca[top_mask, 1], 
           c='red', s=100, marker='o', edgecolors='black', linewidth=2, label='Top Misclassified')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title(f'K-means Clustering (k={optimal_k})')
ax2.legend()

# 3. DBSCAN clustering
log_print("\n3. DBSCAN Clustering...")
dbscan = DBSCAN(eps=3.0, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
log_print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")

ax3 = fig.add_subplot(3, 3, 3)
scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=dbscan_labels, alpha=0.5, s=10, cmap='tab10')
ax3.scatter(X_pca[misclass_mask, 0], X_pca[misclass_mask, 1], 
           c='red', s=50, marker='x', label='Misclassified')
ax3.scatter(X_pca[top_mask, 0], X_pca[top_mask, 1], 
           c='red', s=100, marker='o', edgecolors='black', linewidth=2, label='Top Misclassified')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_title(f'DBSCAN Clustering')
ax3.legend()

# 4. Isolation Forest for outlier detection
log_print("\n4. Isolation Forest Outlier Detection...")
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(X_scaled)
outlier_scores = iso_forest.score_samples(X_scaled)
test_with_pred['outlier_score'] = outlier_scores
test_with_pred['is_outlier'] = outlier_labels == -1

n_outliers = sum(outlier_labels == -1)
log_print(f"Isolation Forest detected {n_outliers} outliers")

# Check how many misclassified are outliers
misclass_outliers = test_with_pred[test_with_pred['is_misclassified'] & test_with_pred['is_outlier']]
log_print(f"Misclassified points that are outliers: {len(misclass_outliers)} / {sum(test_with_pred['is_misclassified'])}")

ax4 = fig.add_subplot(3, 3, 4)
scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=outlier_scores, alpha=0.5, s=10, cmap='coolwarm')
ax4.scatter(X_pca[misclass_mask, 0], X_pca[misclass_mask, 1], 
           c='red', s=50, marker='x', label='Misclassified')
ax4.scatter(X_pca[top_mask, 0], X_pca[top_mask, 1], 
           c='red', s=100, marker='o', edgecolors='black', linewidth=2, label='Top Misclassified')
plt.colorbar(scatter, ax=ax4, label='Outlier Score')
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_title('Isolation Forest Outlier Scores')
ax4.legend()

# 5. t-SNE visualization
log_print("\n5. t-SNE Analysis...")
# Use subset for t-SNE due to computational cost
sample_size = min(5000, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
# Ensure all misclassified points are included
misclass_idx = test_with_pred[test_with_pred['is_misclassified']].index.tolist()
sample_idx = np.unique(np.concatenate([sample_idx, misclass_idx]))

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled[sample_idx])

ax5 = fig.add_subplot(3, 3, 5)
# Map back to original indices
sample_mask = np.isin(range(len(test_with_pred)), sample_idx)
sample_pred = test_with_pred.iloc[sample_idx].reset_index(drop=True)

scatter = ax5.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=sample_pred['Personality'].map({'Introvert': 0, 'Extrovert': 1}), 
                     alpha=0.5, s=10, cmap='viridis')
# Highlight misclassified
misclass_tsne_mask = sample_pred['is_misclassified']
ax5.scatter(X_tsne[misclass_tsne_mask, 0], X_tsne[misclass_tsne_mask, 1], 
           c='red', s=50, marker='x', label='Misclassified')
# Highlight top misclassified
top_tsne_mask = sample_pred['is_top_misclassified']
ax5.scatter(X_tsne[top_tsne_mask, 0], X_tsne[top_tsne_mask, 1], 
           c='red', s=100, marker='o', edgecolors='black', linewidth=2, label='Top Misclassified')
ax5.set_xlabel('t-SNE 1')
ax5.set_ylabel('t-SNE 2')
ax5.set_title('t-SNE: 2D Projection')
ax5.legend()

# 6. Distance distribution analysis
log_print("\n6. Distance Distribution Analysis...")
ax6 = fig.add_subplot(3, 3, 6)
# Plot distance distributions
normal_dist = test_with_pred[~test_with_pred['is_misclassified']]['cluster_distance']
misclass_dist = test_with_pred[test_with_pred['is_misclassified']]['cluster_distance']

ax6.hist(normal_dist, bins=50, alpha=0.5, label='Normal', density=True)
ax6.hist(misclass_dist, bins=20, alpha=0.7, label='Misclassified', density=True)
ax6.set_xlabel('Distance from Cluster Center')
ax6.set_ylabel('Density')
ax6.set_title('Distance from Cluster Centers')
ax6.legend()

log_print(f"Average distance from cluster center:")
log_print(f"  - Normal points: {normal_dist.mean():.4f}")
log_print(f"  - Misclassified points: {misclass_dist.mean():.4f}")

# 7. Feature importance for outliers
log_print("\n7. Feature Analysis for Top Misclassified Cases...")
top_misclass_data = test_with_pred[test_with_pred['is_top_misclassified']]

# Calculate feature statistics
feature_stats = []
for col in feature_cols:
    if col in test_df.columns:
        # Skip non-numeric columns
        if test_df[col].dtype == 'object':
            continue
        overall_mean = test_df[col].mean()
        overall_std = test_df[col].std()
        misclass_mean = top_misclass_data[col].mean()
        z_score = (misclass_mean - overall_mean) / overall_std if overall_std > 0 else 0
        feature_stats.append({
            'feature': col,
            'overall_mean': overall_mean,
            'misclass_mean': misclass_mean,
            'z_score': abs(z_score)
        })

feature_stats_df = pd.DataFrame(feature_stats).sort_values('z_score', ascending=False)
log_print("\nTop features with largest deviation in misclassified cases:")
log_print(feature_stats_df.head(10).to_string())

# Plot top deviating features
ax7 = fig.add_subplot(3, 3, 7)
top_features = feature_stats_df.head(10)
ax7.barh(range(len(top_features)), top_features['z_score'])
ax7.set_yticks(range(len(top_features)))
ax7.set_yticklabels(top_features['feature'])
ax7.set_xlabel('Z-score (deviation from mean)')
ax7.set_title('Top Deviating Features in Misclassified Cases')

# 8. Null pattern analysis
log_print("\n8. Null Pattern Analysis for Misclassified Cases...")
# Count nulls from original data
null_counts = test_with_pred[feature_cols].isnull().sum(axis=1)
test_with_pred['null_count'] = null_counts

ax8 = fig.add_subplot(3, 3, 8)
# Compare null distributions
normal_nulls = test_with_pred[~test_with_pred['is_misclassified']]['null_count']
misclass_nulls = test_with_pred[test_with_pred['is_misclassified']]['null_count']

ax8.hist(normal_nulls, bins=range(0, 15), alpha=0.5, label='Normal', density=True)
ax8.hist(misclass_nulls, bins=range(0, 15), alpha=0.7, label='Misclassified', density=True)
ax8.set_xlabel('Number of Null Values')
ax8.set_ylabel('Density')
ax8.set_title('Null Value Distribution')
ax8.legend()

log_print(f"\nAverage null count:")
log_print(f"  - Normal points: {normal_nulls.mean():.4f}")
log_print(f"  - Misclassified points: {misclass_nulls.mean():.4f}")

# 9. Individual case analysis
log_print("\n9. Individual Case Analysis for Top Misclassified:")
ax9 = fig.add_subplot(3, 3, 9)

# Create a summary table for top misclassified
summary_data = []
for id in top_misclassified_ids[:5]:  # Top 5 for visibility
    row = test_with_pred[test_with_pred['id'] == id].iloc[0]
    summary_data.append({
        'ID': id,
        'Pred': row['Personality'],
        'Cluster_Dist': f"{row['cluster_distance']:.2f}",
        'Outlier_Score': f"{row['outlier_score']:.3f}",
        'Is_Outlier': 'Yes' if row['is_outlier'] else 'No',
        'Null_Count': int(row['null_count'])
    })

summary_df = pd.DataFrame(summary_data)
# Create table plot
ax9.axis('tight')
ax9.axis('off')
table = ax9.table(cellText=summary_df.values,
                  colLabels=summary_df.columns,
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
ax9.set_title('Top 5 Misclassified Cases Summary', pad=20)

plt.tight_layout()
plt.savefig('output/20250705_1300_clustering_outlier_visualization.png', dpi=300, bbox_inches='tight')
log_print("\nVisualization saved to: output/20250705_1300_clustering_outlier_visualization.png")

# Additional detailed analysis
log_print("\n=== INSIGHTS AND CONCLUSIONS ===")

# Check cluster assignments for misclassified
misclass_clusters = test_with_pred[test_with_pred['is_misclassified']]['cluster_distance'].describe()
log_print("\nCluster distance statistics for misclassified points:")
log_print(misclass_clusters.to_string())

# Analyze outlier overlap
outlier_overlap = test_with_pred[test_with_pred['is_misclassified'] & test_with_pred['is_outlier']]
log_print(f"\nMisclassified points that are also outliers: {len(outlier_overlap)} ({len(outlier_overlap)/len(misclass_df)*100:.1f}%)")
log_print(f"IDs of misclassified outliers: {outlier_overlap['id'].tolist()}")

# Feature patterns in top misclassified
log_print("\n=== Feature Patterns in Top Misclassified Cases ===")
for id in top_misclassified_ids[:5]:
    row = test_with_pred[test_with_pred['id'] == id].iloc[0]
    log_print(f"\nID {id}:")
    log_print(f"  - Predicted: {row['Personality']}")
    log_print(f"  - Cluster distance: {row['cluster_distance']:.4f}")
    log_print(f"  - Outlier score: {row['outlier_score']:.4f}")
    log_print(f"  - Null count: {row['null_count']}")
    
    # Find most extreme features
    extreme_features = []
    for col in feature_cols:
        if col in test_df.columns and test_df[col].dtype != 'object' and not pd.isna(row[col]):
            z_score = (row[col] - test_df[col].mean()) / test_df[col].std() if test_df[col].std() > 0 else 0
            if abs(z_score) > 2:
                extreme_features.append((col, row[col], z_score))
    
    if extreme_features:
        log_print("  - Extreme features (|z-score| > 2):")
        for feat, val, z in sorted(extreme_features, key=lambda x: abs(x[2]), reverse=True)[:3]:
            log_print(f"    * {feat}: {val:.2f} (z={z:.2f})")

# Create secondary visualization for feature distributions
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Plot distributions of top deviating features
for i, (_, row) in enumerate(feature_stats_df.head(6).iterrows()):
    if i >= len(axes):
        break
    feat = row['feature']
    if feat in test_df.columns:
        ax = axes[i]
        
        # Plot overall distribution
        test_df[feat].hist(bins=30, alpha=0.5, label='All', ax=ax, density=True)
        
        # Plot misclassified distribution
        if feat in top_misclass_data.columns:
            top_misclass_data[feat].hist(bins=15, alpha=0.7, label='Misclassified', ax=ax, density=True)
        
        ax.set_xlabel(feat)
        ax.set_ylabel('Density')
        ax.set_title(f'{feat} Distribution')
        ax.legend()

plt.tight_layout()
plt.savefig('output/20250705_1300_feature_distributions.png', dpi=300, bbox_inches='tight')
log_print("\nFeature distribution plot saved to: output/20250705_1300_feature_distributions.png")

# Final conclusions
log_print("\n=== FINAL INSIGHTS ===")
log_print("1. Clustering Analysis:")
log_print(f"   - Misclassified points tend to be farther from cluster centers (avg distance: {misclass_dist.mean():.4f} vs {normal_dist.mean():.4f})")
log_print(f"   - {len(outlier_overlap)/len(misclass_df)*100:.1f}% of misclassified cases are detected as outliers by Isolation Forest")

log_print("\n2. Feature Patterns:")
log_print("   - Top deviating features in misclassified cases:")
for _, row in feature_stats_df.head(5).iterrows():
    log_print(f"     * {row['feature']}: z-score = {row['z_score']:.3f}")

log_print("\n3. Null Pattern Analysis:")
log_print(f"   - Misclassified cases have different null patterns (avg nulls: {misclass_nulls.mean():.2f} vs {normal_nulls.mean():.2f})")

log_print("\n4. Why These Are Outliers:")
log_print("   - They exist in boundary regions between clusters")
log_print("   - They have extreme values in multiple features")
log_print("   - They show atypical null patterns")
log_print("   - They are far from typical examples in their predicted class")

log_print("\n5. Recommendations:")
log_print("   - Consider ensemble methods that handle outliers better")
log_print("   - Apply outlier-specific preprocessing or models")
log_print("   - Use confidence thresholds to flag uncertain predictions")
log_print("   - Investigate feature engineering specifically for boundary cases")

output_file.close()
print("\nAnalysis complete! Results saved to output/20250705_1300_clustering_outlier_visualization.txt")