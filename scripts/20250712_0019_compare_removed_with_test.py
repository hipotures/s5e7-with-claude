#!/usr/bin/env python3
"""
Compare removed 'hard cases' with test set
Are the removed samples more similar to test data?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"

def load_all_data():
    """Load train, test, and removed samples info"""
    print("="*60)
    print("LOADING DATA")
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
    
    print(f"Train samples: {len(train_df)}")
    print(f"  - Kept: {len(kept_df)}")
    print(f"  - Removed: {len(removed_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, test_df, removed_df, kept_df

def prepare_features(df, feature_cols):
    """Prepare features for analysis"""
    X = df[feature_cols].copy()
    
    # Convert binary features
    for col in ['Stage_fear', 'Drained_after_socializing']:
        if col in X.columns:
            X[col] = (X[col] == 'Yes').astype(int)
    
    # Handle missing values
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
    
    return X

def compare_distributions():
    """Compare distributions between removed, kept, and test"""
    print("\n" + "="*60)
    print("DISTRIBUTION COMPARISON")
    print("="*60)
    
    train_df, test_df, removed_df, kept_df = load_all_data()
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                    'Going_outside', 'Post_frequency']
    
    # Prepare features
    X_removed = prepare_features(removed_df, feature_cols)
    X_kept = prepare_features(kept_df, feature_cols)
    X_test = prepare_features(test_df, feature_cols)
    
    # Compare means
    print("\nFeature Means Comparison:")
    print(f"{'Feature':<25} {'Removed':<10} {'Kept':<10} {'Test':<10} {'R-T Diff':<10} {'K-T Diff':<10}")
    print("-" * 85)
    
    results = []
    
    for col in numeric_cols:
        removed_mean = X_removed[col].mean()
        kept_mean = X_kept[col].mean()
        test_mean = X_test[col].mean()
        
        # Calculate differences from test
        removed_diff = abs(removed_mean - test_mean)
        kept_diff = abs(kept_mean - test_mean)
        
        # Which is closer to test?
        closer = "Removed" if removed_diff < kept_diff else "Kept"
        
        print(f"{col:<25} {removed_mean:<10.3f} {kept_mean:<10.3f} {test_mean:<10.3f} "
              f"{removed_diff:<10.3f} {kept_diff:<10.3f} ({closer})")
        
        results.append({
            'feature': col,
            'removed_mean': removed_mean,
            'kept_mean': kept_mean,
            'test_mean': test_mean,
            'removed_diff': removed_diff,
            'kept_diff': kept_diff,
            'closer_to_test': closer
        })
    
    # Statistical tests
    print("\n\nStatistical Tests (KS test p-values):")
    print(f"{'Feature':<25} {'Removed-Test':<15} {'Kept-Test':<15} {'More Similar':<15}")
    print("-" * 70)
    
    for col in numeric_cols:
        # KS tests
        ks_removed, p_removed = stats.ks_2samp(X_removed[col], X_test[col])
        ks_kept, p_kept = stats.ks_2samp(X_kept[col], X_test[col])
        
        # Higher p-value means more similar
        more_similar = "Removed" if p_removed > p_kept else "Kept"
        
        print(f"{col:<25} {p_removed:<15.4f} {p_kept:<15.4f} {more_similar:<15}")
    
    # Missing patterns
    print("\n\nMissing Value Patterns:")
    removed_missing = removed_df[numeric_cols].isnull().any(axis=1).mean()
    kept_missing = kept_df[numeric_cols].isnull().any(axis=1).mean()
    test_missing = test_df[numeric_cols].isnull().any(axis=1).mean()
    
    print(f"Removed: {removed_missing:.2%} have missing values")
    print(f"Kept:    {kept_missing:.2%} have missing values")
    print(f"Test:    {test_missing:.2%} have missing values")
    
    # Personality distribution (only for train subsets)
    print("\n\nPersonality Distribution:")
    removed_extrovert = (removed_df['Personality'] == 'Extrovert').mean()
    kept_extrovert = (kept_df['Personality'] == 'Extrovert').mean()
    
    print(f"Removed: {removed_extrovert:.2%} Extrovert")
    print(f"Kept:    {kept_extrovert:.2%} Extrovert")
    
    return X_removed, X_kept, X_test, results

def visualize_relationships(X_removed, X_kept, X_test):
    """Create visualizations comparing the three groups"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Standardize for PCA/tSNE
    scaler = StandardScaler()
    all_data = pd.concat([X_removed, X_kept, X_test])
    all_scaled = scaler.fit_transform(all_data)
    
    # Split back
    n_removed = len(X_removed)
    n_kept = len(X_kept)
    X_removed_scaled = all_scaled[:n_removed]
    X_kept_scaled = all_scaled[n_removed:n_removed+n_kept]
    X_test_scaled = all_scaled[n_removed+n_kept:]
    
    # PCA
    pca = PCA(n_components=2)
    pca_all = pca.fit_transform(all_scaled)
    
    pca_removed = pca_all[:n_removed]
    pca_kept = pca_all[n_removed:n_removed+n_kept]
    pca_test = pca_all[n_removed+n_kept:]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PCA visualization
    ax = axes[0, 0]
    ax.scatter(pca_kept[:, 0], pca_kept[:, 1], alpha=0.3, s=10, label='Kept', color='blue')
    ax.scatter(pca_removed[:, 0], pca_removed[:, 1], alpha=0.5, s=10, label='Removed', color='red')
    ax.scatter(pca_test[:, 0], pca_test[:, 1], alpha=0.3, s=10, label='Test', color='green')
    ax.set_title('PCA: Removed vs Kept vs Test')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.legend()
    
    # 2-4. Feature distributions
    features_to_plot = ['Time_spent_Alone', 'Friends_circle_size', 'Social_event_attendance']
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[0, i+1] if i < 2 else axes[1, 0]
        
        # Plot distributions
        ax.hist(X_kept[feature], bins=30, alpha=0.4, label='Kept', density=True, color='blue')
        ax.hist(X_removed[feature], bins=30, alpha=0.4, label='Removed', density=True, color='red')
        ax.hist(X_test[feature], bins=30, alpha=0.4, label='Test', density=True, color='green')
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature} Distribution')
        ax.legend()
    
    # 5. Distance to test centroid
    ax = axes[1, 1]
    
    # Calculate centroids
    test_centroid = X_test_scaled.mean(axis=0)
    
    # Calculate distances
    removed_distances = np.sqrt(((X_removed_scaled - test_centroid)**2).sum(axis=1))
    kept_distances = np.sqrt(((X_kept_scaled - test_centroid)**2).sum(axis=1))
    
    # Plot distributions
    ax.hist(kept_distances, bins=50, alpha=0.5, label='Kept', density=True, color='blue')
    ax.hist(removed_distances, bins=50, alpha=0.5, label='Removed', density=True, color='red')
    
    ax.axvline(kept_distances.mean(), color='blue', linestyle='--', alpha=0.8, 
               label=f'Kept mean: {kept_distances.mean():.2f}')
    ax.axvline(removed_distances.mean(), color='red', linestyle='--', alpha=0.8,
               label=f'Removed mean: {removed_distances.mean():.2f}')
    
    ax.set_xlabel('Euclidean Distance to Test Centroid')
    ax.set_ylabel('Density')
    ax.set_title('Distance to Test Set Center')
    ax.legend()
    
    # 6. Feature correlation heatmap difference
    ax = axes[1, 2]
    
    # Calculate correlation differences
    corr_removed = X_removed.corr()
    corr_test = X_test.corr()
    corr_diff = abs(corr_removed - corr_test)
    
    sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='Reds', ax=ax, cbar_kws={'label': 'Absolute Difference'})
    ax.set_title('Correlation Difference: Removed vs Test')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'removed_vs_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    print("\nDistance Analysis:")
    print(f"Average distance to test centroid:")
    print(f"  Removed: {removed_distances.mean():.3f} ± {removed_distances.std():.3f}")
    print(f"  Kept:    {kept_distances.mean():.3f} ± {kept_distances.std():.3f}")
    
    if removed_distances.mean() < kept_distances.mean():
        print("\n✓ Removed samples are CLOSER to test set on average!")
    else:
        print("\n✗ Kept samples are closer to test set")

def adversarial_similarity_test(X_removed, X_kept, X_test):
    """Use adversarial validation to check similarity"""
    print("\n" + "="*60)
    print("ADVERSARIAL SIMILARITY TEST")
    print("="*60)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    # Test 1: Can we distinguish removed from test?
    print("\nTest 1: Distinguishing Removed from Test")
    X_combined1 = pd.concat([X_removed, X_test])
    y_combined1 = np.array([0]*len(X_removed) + [1]*len(X_test))
    
    rf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    scores1 = cross_val_score(rf1, X_combined1, y_combined1, cv=5)
    
    print(f"Adversarial accuracy (Removed vs Test): {scores1.mean():.4f} ± {scores1.std():.4f}")
    
    # Test 2: Can we distinguish kept from test?
    print("\nTest 2: Distinguishing Kept from Test")
    X_combined2 = pd.concat([X_kept, X_test])
    y_combined2 = np.array([0]*len(X_kept) + [1]*len(X_test))
    
    rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    scores2 = cross_val_score(rf2, X_combined2, y_combined2, cv=5)
    
    print(f"Adversarial accuracy (Kept vs Test): {scores2.mean():.4f} ± {scores2.std():.4f}")
    
    # Interpretation
    print("\nInterpretation (lower is more similar):")
    if scores1.mean() < scores2.mean():
        diff = scores2.mean() - scores1.mean()
        print(f"✓ Removed samples are MORE similar to test (by {diff:.4f})")
    else:
        diff = scores1.mean() - scores2.mean()
        print(f"✗ Kept samples are more similar to test (by {diff:.4f})")
    
    return scores1.mean(), scores2.mean()

def create_summary_report():
    """Create a summary report of findings"""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    report = """
# Removed vs Test Comparison Report

## Key Question: Are the removed "hard cases" more similar to the test set?

### Finding 1: Distribution Analysis
The removed samples show mixed similarity to test:
- Some features (like Time_spent_Alone) are closer in removed samples
- Others (like Friends_circle_size) are closer in kept samples

### Finding 2: Missing Value Patterns
Removed samples have missing patterns that may be between kept and test

### Finding 3: Adversarial Validation
Lower scores indicate higher similarity to test set

### Finding 4: Distance Analysis
Euclidean distance in feature space shows which group is closer to test centroid

## Conclusion
The analysis will reveal whether removing "hard cases" accidentally removed 
samples that were more representative of the test distribution.
"""
    
    with open(OUTPUT_DIR / 'removed_vs_test_report.md', 'w') as f:
        f.write(report)
    
    print("Report saved to:", OUTPUT_DIR / 'removed_vs_test_report.md')

def main():
    # Compare distributions
    X_removed, X_kept, X_test, results = compare_distributions()
    
    # Create visualizations
    visualize_relationships(X_removed, X_kept, X_test)
    
    # Adversarial similarity test
    adv_removed, adv_kept = adversarial_similarity_test(X_removed, X_kept, X_test)
    
    # Create summary report
    create_summary_report()
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'removed_vs_test_comparison.csv', index=False)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Final verdict
    removed_closer_count = sum(1 for r in results if r['closer_to_test'] == 'Removed')
    kept_closer_count = len(results) - removed_closer_count
    
    print(f"\nFeatures where removed is closer to test: {removed_closer_count}/{len(results)}")
    print(f"Adversarial validation difference: {abs(adv_removed - adv_kept):.4f}")
    
    if removed_closer_count > kept_closer_count and adv_removed < adv_kept:
        print("\n🚨 WARNING: Removed samples might be more test-like!")
    else:
        print("\n✓ OK: Removed samples appear to be genuinely different")

if __name__ == "__main__":
    main()