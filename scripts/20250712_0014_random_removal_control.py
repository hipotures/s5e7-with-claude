#!/usr/bin/env python3
"""
Control experiment: Remove same number of samples randomly
Compare with strategic removal of hard cases
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"

def prepare_data():
    """Load and prepare data"""
    print("="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    # Prepare features
    X_train = train_df[feature_cols].copy()
    y_train = (train_df['Personality'] == 'Extrovert').astype(int)
    X_test = test_df[feature_cols].copy()
    
    # Convert binary features
    for col in ['Stage_fear', 'Drained_after_socializing']:
        X_train[col] = (X_train[col] == 'Yes').astype(int)
        X_test[col] = (X_test[col] == 'Yes').astype(int)
    
    # Handle missing values
    for col in X_train.columns:
        if X_train[col].dtype in ['float64', 'int64']:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    
    return train_df, test_df, X_train, X_test, y_train

def random_removal_experiment(X_train, y_train, n_samples_to_keep=13713, n_trials=10):
    """Remove random samples and check accuracy"""
    print("\n" + "="*60)
    print(f"RANDOM REMOVAL EXPERIMENT ({n_trials} trials)")
    print("="*60)
    
    print(f"Original dataset size: {len(X_train)}")
    print(f"Will keep: {n_samples_to_keep} samples")
    print(f"Will remove: {len(X_train) - n_samples_to_keep} samples")
    
    results = []
    
    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        
        # Random sample indices to keep
        keep_indices = np.random.choice(len(X_train), size=n_samples_to_keep, replace=False)
        keep_mask = np.zeros(len(X_train), dtype=bool)
        keep_mask[keep_indices] = True
        
        # Create subset
        X_subset = X_train.iloc[keep_indices]
        y_subset = y_train.iloc[keep_indices]
        
        # Check personality distribution
        subset_extrovert_pct = y_subset.mean()
        
        # Train models
        models = {
            'xgb': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, 
                                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
            'lgb': LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1),
            'cat': CatBoostClassifier(n_estimators=300, learning_rate=0.05, depth=6,
                                     subsample=0.8, verbose=False, random_state=42)
        }
        
        trial_results = {
            'trial': trial + 1,
            'extrovert_pct': subset_extrovert_pct
        }
        
        for name, model in models.items():
            # Train on subset
            model.fit(X_subset, y_subset)
            
            # Self-accuracy on subset
            subset_pred = model.predict(X_subset)
            subset_acc = (subset_pred == y_subset).mean()
            
            # Accuracy on full dataset
            full_pred = model.predict(X_train)
            full_acc = (full_pred == y_train).mean()
            
            # Accuracy on removed samples
            removed_mask = ~keep_mask
            if removed_mask.sum() > 0:
                removed_pred = model.predict(X_train[removed_mask])
                removed_true = y_train[removed_mask]
                removed_acc = (removed_pred == removed_true).mean()
            else:
                removed_acc = np.nan
            
            trial_results[f'{name}_subset_acc'] = subset_acc
            trial_results[f'{name}_full_acc'] = full_acc
            trial_results[f'{name}_removed_acc'] = removed_acc
            
            print(f"{name.upper()}: subset={subset_acc:.4f}, full={full_acc:.4f}, removed={removed_acc:.4f}")
        
        results.append(trial_results)
    
    return pd.DataFrame(results)

def compare_with_strategic_removal():
    """Load and compare with strategic removal results"""
    print("\n" + "="*60)
    print("COMPARISON WITH STRATEGIC REMOVAL")
    print("="*60)
    
    # Hard-coded results from strategic removal
    strategic_results = {
        'xgb': {'subset_acc': 0.9998, 'full_acc': 0.9703, 'removed_acc': 0.8863},
        'lgb': {'subset_acc': 1.0000, 'full_acc': 0.9706, 'removed_acc': 0.8867},
        'cat': {'subset_acc': 1.0000, 'full_acc': 0.9705, 'removed_acc': 0.8863}
    }
    
    print("\nStrategic Removal (hard cases):")
    for model, metrics in strategic_results.items():
        print(f"{model.upper()}: subset={metrics['subset_acc']:.4f}, "
              f"full={metrics['full_acc']:.4f}, removed={metrics['removed_acc']:.4f}")
    
    return strategic_results

def analyze_results(random_results_df, strategic_results):
    """Analyze and visualize results"""
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    # Calculate statistics for random removal
    print("\nRandom Removal Statistics:")
    for model in ['xgb', 'lgb', 'cat']:
        subset_col = f'{model}_subset_acc'
        full_col = f'{model}_full_acc'
        removed_col = f'{model}_removed_acc'
        
        print(f"\n{model.upper()}:")
        print(f"  Subset accuracy: {random_results_df[subset_col].mean():.4f} ± {random_results_df[subset_col].std():.4f}")
        print(f"  Full accuracy: {random_results_df[full_col].mean():.4f} ± {random_results_df[full_col].std():.4f}")
        print(f"  Removed accuracy: {random_results_df[removed_col].mean():.4f} ± {random_results_df[removed_col].std():.4f}")
    
    # Compare with strategic
    print("\n\nDifference (Strategic - Random):")
    for model in ['xgb', 'lgb', 'cat']:
        strategic_subset = strategic_results[model]['subset_acc']
        random_subset_mean = random_results_df[f'{model}_subset_acc'].mean()
        diff = strategic_subset - random_subset_mean
        
        print(f"{model.upper()}: {diff:+.4f} ({strategic_subset:.4f} vs {random_subset_mean:.4f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Subset accuracy comparison
    ax = axes[0, 0]
    models = ['xgb', 'lgb', 'cat']
    x = np.arange(len(models))
    width = 0.35
    
    random_means = [random_results_df[f'{m}_subset_acc'].mean() for m in models]
    random_stds = [random_results_df[f'{m}_subset_acc'].std() for m in models]
    strategic_accs = [strategic_results[m]['subset_acc'] for m in models]
    
    ax.bar(x - width/2, random_means, width, yerr=random_stds, label='Random Removal', alpha=0.8, capsize=5)
    ax.bar(x + width/2, strategic_accs, width, label='Strategic Removal', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Subset Accuracy')
    ax.set_title('Subset Accuracy: Random vs Strategic Removal')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.set_ylim(0.96, 1.01)
    
    # 2. Distribution of random subset accuracies
    ax = axes[0, 1]
    for i, model in enumerate(models):
        accuracies = random_results_df[f'{model}_subset_acc']
        ax.hist(accuracies, bins=10, alpha=0.5, label=model.upper())
    
    # Add vertical lines for strategic results
    for model in models:
        ax.axvline(strategic_results[model]['subset_acc'], 
                  linestyle='--', label=f'{model.upper()} strategic')
    
    ax.set_xlabel('Subset Accuracy')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Random Removal Accuracies')
    ax.legend()
    
    # 3. Full dataset accuracy comparison
    ax = axes[1, 0]
    random_full_means = [random_results_df[f'{m}_full_acc'].mean() for m in models]
    strategic_full_accs = [strategic_results[m]['full_acc'] for m in models]
    
    ax.plot(models, random_full_means, 'o-', label='Random Removal', markersize=10)
    ax.plot(models, strategic_full_accs, 's-', label='Strategic Removal', markersize=10)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Full Dataset Accuracy')
    ax.set_title('Full Dataset Performance After Training on Subset')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Accuracy on removed samples
    ax = axes[1, 1]
    random_removed_means = [random_results_df[f'{m}_removed_acc'].mean() for m in models]
    strategic_removed_accs = [strategic_results[m]['removed_acc'] for m in models]
    
    bar_width = 0.35
    x = np.arange(len(models))
    
    ax.bar(x - bar_width/2, random_removed_means, bar_width, label='Random Removal', alpha=0.8)
    ax.bar(x + bar_width/2, strategic_removed_accs, bar_width, label='Strategic Removal', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy on Removed Samples')
    ax.set_title('Model Performance on Removed Samples')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'random_vs_strategic_removal.png', dpi=300)
    plt.close()
    
    print(f"\nVisualization saved to {OUTPUT_DIR / 'random_vs_strategic_removal.png'}")

def main():
    # Load data
    train_df, test_df, X_train, X_test, y_train = prepare_data()
    
    # Run random removal experiment
    random_results = random_removal_experiment(X_train, y_train, n_samples_to_keep=13713, n_trials=10)
    
    # Load strategic removal results
    strategic_results = compare_with_strategic_removal()
    
    # Analyze and visualize
    analyze_results(random_results, strategic_results)
    
    # Save results
    random_results.to_csv(OUTPUT_DIR / 'random_removal_results.csv', index=False)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    # Calculate average improvement
    avg_improvement = 0
    for model in ['xgb', 'lgb', 'cat']:
        strategic = strategic_results[model]['subset_acc']
        random = random_results[f'{model}_subset_acc'].mean()
        avg_improvement += (strategic - random)
    avg_improvement /= 3
    
    print(f"\nAverage improvement of strategic over random removal: {avg_improvement:.4f}")
    print(f"This represents a {avg_improvement * 100:.2f}% improvement in accuracy")
    
    if avg_improvement > 0.02:
        print("\n✓ Strategic removal is SIGNIFICANTLY better than random removal")
        print("  The quality of removed samples matters more than quantity!")
    else:
        print("\n✗ Strategic removal shows minimal improvement over random")
        print("  The improvement might be due to sample size reduction")

if __name__ == "__main__":
    main()