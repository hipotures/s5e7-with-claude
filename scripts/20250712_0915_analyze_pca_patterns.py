#!/usr/bin/env python3
"""
Analyze the diagonal stripe patterns in PCA visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def analyze_pca_patterns():
    """Analyze the diagonal stripe patterns in PCA"""
    
    print("="*60)
    print("ANALYZING PCA DIAGONAL STRIPE PATTERNS")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    
    # Features for analysis
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # Check data types and unique values
    print("\nFeature Analysis:")
    for feat in numeric_features:
        unique_vals = train_df[feat].dropna().unique()
        n_unique = len(unique_vals)
        print(f"\n{feat}:")
        print(f"  - Unique values: {n_unique}")
        print(f"  - Min: {train_df[feat].min():.2f}, Max: {train_df[feat].max():.2f}")
        print(f"  - Data type: {train_df[feat].dtype}")
        
        # Check if values are integers disguised as floats
        if train_df[feat].dropna().apply(lambda x: x == int(x)).all():
            print(f"  - All values are integers!")
        
        # Show first 10 unique values if not too many
        if n_unique <= 20:
            print(f"  - Values: {sorted(unique_vals)[:20]}")
    
    # Prepare data for PCA
    X = train_df[numeric_features].fillna(train_df[numeric_features].median())
    
    # Add categorical features
    train_df['Stage_fear_encoded'] = (train_df['Stage_fear'] == 'Yes').astype(int)
    train_df['Drained_encoded'] = (train_df['Drained_after_socializing'] == 'Yes').astype(int)
    
    X_with_cat = train_df[numeric_features + ['Stage_fear_encoded', 'Drained_encoded']].fillna(train_df[numeric_features].median())
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_cat)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Create visualization showing the patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 1. Full PCA scatter
    ax = axes[0, 0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                        c=train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1}),
                        alpha=0.5, s=10, cmap='RdBu')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('PCA Visualization - Full Data\n(Red=Extrovert, Blue=Introvert)')
    plt.colorbar(scatter, ax=ax)
    
    # 2. Zoom on stripe pattern region
    ax = axes[0, 1]
    # Focus on the region with stripes (adjust based on your observation)
    mask = (X_pca[:, 0] > -2) & (X_pca[:, 0] < 2) & (X_pca[:, 1] > -2) & (X_pca[:, 1] < 2)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
              c=train_df[mask]['Personality'].map({'Introvert': 0, 'Extrovert': 1}),
              alpha=0.5, s=20, cmap='RdBu')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Zoomed View of Stripe Pattern Region')
    ax.grid(True, alpha=0.3)
    
    # 3. Component loadings
    ax = axes[1, 0]
    feature_names = numeric_features + ['Stage_fear', 'Drained']
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, feature, 
               fontsize=10, ha='center', va='center')
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('PC1 Loading')
    ax.set_ylabel('PC2 Loading')
    ax.set_title('PCA Feature Loadings')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 4. Correlation heatmap
    ax = axes[1, 1]
    corr_matrix = X_with_cat.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pca_stripe_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze specific patterns
    print("\n" + "="*40)
    print("STRIPE PATTERN ANALYSIS")
    print("="*40)
    
    # Find points in stripe regions
    # Calculate angle from origin
    angles = np.arctan2(X_pca[:, 1], X_pca[:, 0])
    distances = np.sqrt(X_pca[:, 0]**2 + X_pca[:, 1]**2)
    
    # Bin by angle to find stripes
    angle_bins = np.linspace(-np.pi, np.pi, 37)  # 36 bins of 10 degrees each
    angle_binned = np.digitize(angles, angle_bins)
    
    # Count points in each angular bin
    bin_counts = np.bincount(angle_binned)
    
    # Find bins with unusually high counts (potential stripes)
    mean_count = np.mean(bin_counts[1:-1])  # Exclude edge bins
    std_count = np.std(bin_counts[1:-1])
    stripe_bins = np.where(bin_counts > mean_count + 2*std_count)[0]
    
    print(f"\nFound {len(stripe_bins)} potential stripe directions")
    
    # Analyze a specific stripe
    if len(stripe_bins) > 0:
        stripe_idx = angle_binned == stripe_bins[0]
        stripe_data = train_df[stripe_idx]
        
        print(f"\nAnalyzing stripe at angle {np.degrees(angles[stripe_idx].mean()):.1f} degrees:")
        print(f"Number of points: {stripe_idx.sum()}")
        
        # Check if certain feature combinations are overrepresented
        for feat in numeric_features:
            unique_in_stripe = stripe_data[feat].nunique()
            print(f"{feat}: {unique_in_stripe} unique values in stripe")
    
    # Create a detailed view of feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(numeric_features):
        ax = axes[i]
        ax.hist(train_df[feat].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(feat)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {feat}')
        
        # Add text with stats
        stats_text = f"Unique: {train_df[feat].nunique()}\n"
        stats_text += f"Integer: {'Yes' if train_df[feat].dropna().apply(lambda x: x == int(x)).all() else 'No'}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Look for specific patterns in the data
    print("\n" + "="*40)
    print("CHECKING FOR DATA PATTERNS")
    print("="*40)
    
    # Check for common combinations
    feature_combos = train_df[numeric_features].value_counts().head(20)
    print("\nTop 20 most common feature combinations:")
    print(feature_combos)
    
    # Check if certain combinations appear in grid pattern
    X_rounded = np.round(X_with_cat, 1)  # Round to 1 decimal
    unique_combos = np.unique(X_rounded, axis=0)
    print(f"\nNumber of unique combinations (rounded to 1 decimal): {len(unique_combos)}")
    
    # Save findings
    with open(OUTPUT_DIR / 'pca_stripe_findings.txt', 'w') as f:
        f.write("PCA STRIPE PATTERN ANALYSIS\n")
        f.write("="*40 + "\n\n")
        
        f.write("FINDINGS:\n")
        f.write("1. The diagonal stripes in PCA are likely caused by:\n")
        f.write("   - Integer-valued features creating discrete combinations\n")
        f.write("   - Strong correlations between features\n")
        f.write("   - Limited number of unique feature combinations\n\n")
        
        f.write("2. Feature characteristics:\n")
        for feat in numeric_features:
            is_int = train_df[feat].dropna().apply(lambda x: x == int(x)).all()
            n_unique = train_df[feat].nunique()
            f.write(f"   - {feat}: {n_unique} unique values, Integer: {is_int}\n")
        
        f.write(f"\n3. Total unique feature combinations: {len(train_df[numeric_features].drop_duplicates())}\n")
        f.write(f"   Out of {len(train_df)} total samples\n")
    
    print("\nAnalysis complete! Check output files:")
    print("- pca_stripe_analysis.png")
    print("- feature_distributions.png")
    print("- pca_stripe_findings.txt")

if __name__ == "__main__":
    analyze_pca_patterns()