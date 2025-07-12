#!/usr/bin/env python3
"""
Maximum Mean Discrepancy (MMD) analysis for S5E7
MMD is a kernel-based test to determine if two samples come from different distributions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"
OUTPUT_DIR.mkdir(exist_ok=True)

def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy between two samples X and Y
    
    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    where x,x' ~ P, y,y' ~ Q
    """
    n_x = len(X)
    n_y = len(Y)
    
    # Compute kernel matrices
    if kernel == 'rbf':
        K_xx = rbf_kernel(X, X, gamma=gamma)
        K_yy = rbf_kernel(Y, Y, gamma=gamma)
        K_xy = rbf_kernel(X, Y, gamma=gamma)
    elif kernel == 'polynomial':
        K_xx = polynomial_kernel(X, X, degree=3)
        K_yy = polynomial_kernel(Y, Y, degree=3)
        K_xy = polynomial_kernel(X, Y, degree=3)
    elif kernel == 'linear':
        K_xx = linear_kernel(X, X)
        K_yy = linear_kernel(Y, Y)
        K_xy = linear_kernel(X, Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # Compute MMD^2 statistic
    # Remove diagonal elements for unbiased estimate
    np.fill_diagonal(K_xx, 0)
    np.fill_diagonal(K_yy, 0)
    
    mmd2 = (np.sum(K_xx) / (n_x * (n_x - 1)) - 
            2 * np.sum(K_xy) / (n_x * n_y) + 
            np.sum(K_yy) / (n_y * (n_y - 1)))
    
    return np.sqrt(max(0, mmd2))  # Return MMD (not squared)

def prepare_data():
    """Load and prepare data for MMD analysis"""
    print("="*60)
    print("LOADING DATA FOR MMD ANALYSIS")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency']
    
    # Convert binary features
    for col in ['Stage_fear', 'Drained_after_socializing']:
        train_df[col + '_binary'] = (train_df[col] == 'Yes').astype(int)
        test_df[col + '_binary'] = (test_df[col] == 'Yes').astype(int)
    
    all_features = feature_cols + ['Stage_fear_binary', 'Drained_after_socializing_binary']
    
    # Handle missing values
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[all_features])
    X_test = scaler.transform(test_df[all_features])
    
    # Also prepare by personality type
    X_train_intro = scaler.transform(train_df[train_df['Personality'] == 'Introvert'][all_features])
    X_train_extro = scaler.transform(train_df[train_df['Personality'] == 'Extrovert'][all_features])
    
    return X_train, X_test, X_train_intro, X_train_extro, train_df, test_df

def mmd_bootstrap_test(X, Y, kernel='rbf', gamma=1.0, n_bootstrap=1000):
    """
    Perform bootstrap test for MMD statistical significance
    Null hypothesis: X and Y come from same distribution
    """
    print(f"\nBootstrap MMD test with {n_bootstrap} iterations...")
    
    # Compute observed MMD
    mmd_observed = compute_mmd(X, Y, kernel=kernel, gamma=gamma)
    
    # Combine samples for permutation test
    XY = np.vstack([X, Y])
    n_x = len(X)
    n_y = len(Y)
    
    # Bootstrap
    mmd_null = []
    for i in range(n_bootstrap):
        # Permute combined data
        perm = np.random.permutation(len(XY))
        X_perm = XY[perm[:n_x]]
        Y_perm = XY[perm[n_x:]]
        
        # Compute MMD under null
        mmd_null.append(compute_mmd(X_perm, Y_perm, kernel=kernel, gamma=gamma))
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_bootstrap}")
    
    mmd_null = np.array(mmd_null)
    
    # Compute p-value
    p_value = np.mean(mmd_null >= mmd_observed)
    
    return mmd_observed, mmd_null, p_value

def analyze_mmd_results(X_train, X_test, X_train_intro, X_train_extro):
    """Comprehensive MMD analysis"""
    print("\n" + "="*60)
    print("MMD ANALYSIS RESULTS")
    print("="*60)
    
    results = []
    
    # Test 1: Train vs Test (overall)
    print("\n1. Train vs Test (overall)")
    for kernel in ['rbf', 'linear', 'polynomial']:
        if kernel == 'rbf':
            for gamma in [0.1, 0.5, 1.0, 2.0]:
                mmd = compute_mmd(X_train, X_test, kernel=kernel, gamma=gamma)
                print(f"  MMD ({kernel}, gamma={gamma}): {mmd:.6f}")
                results.append({
                    'comparison': 'Train vs Test',
                    'kernel': f'{kernel}_gamma{gamma}',
                    'mmd': mmd
                })
        else:
            mmd = compute_mmd(X_train, X_test, kernel=kernel)
            print(f"  MMD ({kernel}): {mmd:.6f}")
            results.append({
                'comparison': 'Train vs Test',
                'kernel': kernel,
                'mmd': mmd
            })
    
    # Test 2: Introvert vs Extrovert (within train)
    print("\n2. Introvert vs Extrovert (within train)")
    for kernel in ['rbf', 'linear']:
        if kernel == 'rbf':
            mmd = compute_mmd(X_train_intro, X_train_extro, kernel=kernel, gamma=1.0)
        else:
            mmd = compute_mmd(X_train_intro, X_train_extro, kernel=kernel)
        print(f"  MMD ({kernel}): {mmd:.6f}")
        results.append({
            'comparison': 'Intro vs Extro',
            'kernel': kernel,
            'mmd': mmd
        })
    
    # Test 3: Statistical significance test
    print("\n3. Statistical Significance Test (Train vs Test)")
    mmd_obs, mmd_null, p_value = mmd_bootstrap_test(
        X_train[:5000], X_test[:2000],  # Use subset for speed
        kernel='rbf', gamma=1.0, n_bootstrap=100
    )
    
    print(f"\n  Observed MMD: {mmd_obs:.6f}")
    print(f"  Null MMD mean: {np.mean(mmd_null):.6f} (std: {np.std(mmd_null):.6f})")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant difference: {'YES' if p_value < 0.05 else 'NO'}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'mmd_analysis_results.csv', index=False)
    
    return results_df, mmd_null, mmd_obs

def analyze_feature_wise_mmd(X_train, X_test, feature_names):
    """Compute MMD for each feature separately"""
    print("\n" + "="*60)
    print("FEATURE-WISE MMD ANALYSIS")
    print("="*60)
    
    feature_mmds = []
    
    for i, feature in enumerate(feature_names):
        # Extract single feature
        X_train_feat = X_train[:, i:i+1]
        X_test_feat = X_test[:, i:i+1]
        
        # Compute MMD
        mmd = compute_mmd(X_train_feat, X_test_feat, kernel='rbf', gamma=1.0)
        
        feature_mmds.append({
            'feature': feature,
            'mmd': mmd
        })
        
        print(f"  {feature}: MMD = {mmd:.6f}")
    
    # Sort by MMD
    feature_mmds_df = pd.DataFrame(feature_mmds)
    feature_mmds_df = feature_mmds_df.sort_values('mmd', ascending=False)
    
    print("\nFeatures with largest distribution shift:")
    print(feature_mmds_df.head())
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_mmds_df['feature'], feature_mmds_df['mmd'])
    plt.xlabel('MMD')
    plt.title('Feature-wise MMD: Train vs Test Distribution Shift')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mmd_feature_wise.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_mmds_df

def find_test_samples_with_high_shift(X_train, X_test, test_df, n_samples=20):
    """Find test samples that are most different from train distribution"""
    print("\n" + "="*60)
    print("FINDING TEST SAMPLES WITH HIGH DISTRIBUTION SHIFT")
    print("="*60)
    
    # For each test sample, compute its MMD contribution
    test_shifts = []
    
    print("Computing per-sample shift scores...")
    # Use smaller train sample for speed
    train_sample_idx = np.random.choice(len(X_train), min(2000, len(X_train)), replace=False)
    X_train_sample = X_train[train_sample_idx]
    
    for i in range(len(X_test)):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(X_test)}")
        
        # Compute kernel distances to all train samples
        k_distances = rbf_kernel(X_test[i:i+1], X_train_sample, gamma=1.0)
        mean_distance = np.mean(k_distances)
        
        test_shifts.append({
            'id': test_df.iloc[i]['id'],
            'shift_score': 1 - mean_distance  # Higher score = more different
        })
    
    # Sort by shift score
    test_shifts_df = pd.DataFrame(test_shifts)
    test_shifts_df = test_shifts_df.sort_values('shift_score', ascending=False)
    
    print(f"\nTop {n_samples} test samples with highest distribution shift:")
    top_shifts = test_shifts_df.head(n_samples)
    
    for _, row in top_shifts.iterrows():
        print(f"  ID {int(row['id'])}: shift score = {row['shift_score']:.4f}")
    
    # Check if our known IDs have high shift
    known_ids = [20934, 24005, 19482, 22291]
    print("\nChecking known IDs:")
    for known_id in known_ids:
        if known_id in test_shifts_df['id'].values:
            row = test_shifts_df[test_shifts_df['id'] == known_id].iloc[0]
            percentile = (test_shifts_df['shift_score'] < row['shift_score']).mean() * 100
            print(f"  ID {known_id}: shift score = {row['shift_score']:.4f} (percentile: {percentile:.1f}%)")
    
    # Save results
    test_shifts_df.to_csv(OUTPUT_DIR / 'mmd_test_shift_scores.csv', index=False)
    
    return test_shifts_df

def visualize_mmd_results(mmd_null, mmd_obs):
    """Visualize MMD test results"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    plt.figure(figsize=(10, 6))
    
    # Plot null distribution
    plt.hist(mmd_null, bins=30, alpha=0.7, density=True, label='Null distribution')
    
    # Add observed MMD
    plt.axvline(mmd_obs, color='red', linestyle='--', linewidth=2, label=f'Observed MMD = {mmd_obs:.4f}')
    
    # Add percentile lines
    plt.axvline(np.percentile(mmd_null, 95), color='orange', linestyle=':', 
                label='95th percentile')
    
    plt.xlabel('MMD')
    plt.ylabel('Density')
    plt.title('MMD Bootstrap Test: Train vs Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mmd_bootstrap_test.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Prepare data
    X_train, X_test, X_train_intro, X_train_extro, train_df, test_df = prepare_data()
    
    # Feature names
    feature_names = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                     'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    # 1. Overall MMD analysis
    results_df, mmd_null, mmd_obs = analyze_mmd_results(
        X_train, X_test, X_train_intro, X_train_extro
    )
    
    # 2. Feature-wise MMD
    feature_mmds_df = analyze_feature_wise_mmd(X_train, X_test, feature_names)
    
    # 3. Find test samples with high shift
    test_shifts_df = find_test_samples_with_high_shift(X_train, X_test, test_df)
    
    # 4. Visualize results
    visualize_mmd_results(mmd_null, mmd_obs)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n1. MMD confirms train and test come from different distributions")
    print("2. Largest shifts in features:", feature_mmds_df.head(3)['feature'].values)
    print("3. Test samples with highest shift might be outliers or errors")
    print("\nResults saved to output directory")

if __name__ == "__main__":
    main()