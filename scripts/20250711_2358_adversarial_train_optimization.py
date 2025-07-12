#!/usr/bin/env python3
"""
Use adversarial validation to find optimal training subset
Goal: Achieve 100% accuracy by training on samples most similar to test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"

def prepare_data():
    """Load and prepare data"""
    print("="*60)
    print("LOADING DATA")
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

def adversarial_validation(X_train, X_test):
    """Get probability of each train sample being from test distribution"""
    print("\n" + "="*60)
    print("ADVERSARIAL VALIDATION")
    print("="*60)
    
    # Create labels
    train_adv = X_train.copy()
    test_adv = X_test.copy()
    train_adv['is_test'] = 0
    test_adv['is_test'] = 1
    
    # Combine
    combined = pd.concat([train_adv, test_adv], ignore_index=True)
    X_adv = combined.drop('is_test', axis=1)
    y_adv = combined['is_test']
    
    # Train adversarial model
    rf_adv = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_adv.fit(X_adv, y_adv)
    
    # Get probabilities for train samples
    train_probs = rf_adv.predict_proba(X_train)[:, 1]
    
    print(f"Adversarial model accuracy: {rf_adv.score(X_adv, y_adv):.4f}")
    print(f"Train samples probability of being test-like:")
    print(f"  Min: {train_probs.min():.4f}")
    print(f"  Max: {train_probs.max():.4f}")
    print(f"  Mean: {train_probs.mean():.4f}")
    
    return train_probs, rf_adv

def select_optimal_train_subset(train_df, X_train, y_train, train_probs, percentile=80):
    """Select train samples most similar to test"""
    print("\n" + "="*60)
    print(f"SELECTING TOP {percentile}% TEST-LIKE SAMPLES")
    print("="*60)
    
    # Get threshold
    threshold = np.percentile(train_probs, percentile)
    mask = train_probs >= threshold
    
    print(f"Threshold: {threshold:.4f}")
    print(f"Selected samples: {mask.sum()} out of {len(train_df)}")
    
    # Create subset
    train_subset = train_df[mask].copy()
    X_train_subset = X_train[mask].copy()
    y_train_subset = y_train[mask].copy()
    
    # Analyze personality distribution
    orig_extrovert_pct = (train_df['Personality'] == 'Extrovert').mean()
    subset_extrovert_pct = (train_subset['Personality'] == 'Extrovert').mean()
    
    print(f"\nPersonality distribution:")
    print(f"  Original: {orig_extrovert_pct:.2%} Extrovert")
    print(f"  Subset:   {subset_extrovert_pct:.2%} Extrovert")
    
    return train_subset, X_train_subset, y_train_subset, mask

def analyze_subset_characteristics(train_df, mask):
    """Analyze what makes the subset special"""
    print("\n" + "="*60)
    print("ANALYZING SUBSET CHARACTERISTICS")
    print("="*60)
    
    selected = train_df[mask]
    not_selected = train_df[~mask]
    
    # Missing value patterns
    selected_missing = selected.isnull().any(axis=1).mean()
    not_selected_missing = not_selected.isnull().any(axis=1).mean()
    
    print(f"\nMissing value patterns:")
    print(f"  Selected:     {selected_missing:.2%} have missing values")
    print(f"  Not selected: {not_selected_missing:.2%} have missing values")
    
    # ID patterns
    selected_ids = selected['id'].values
    not_selected_ids = not_selected['id'].values
    
    print(f"\nID patterns:")
    print(f"  Selected:     {(selected_ids % 3 == 0).mean():.2%} divisible by 3")
    print(f"  Not selected: {(not_selected_ids % 3 == 0).mean():.2%} divisible by 3")
    
    # Feature means
    print("\nFeature differences (selected vs not selected):")
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                    'Going_outside', 'Post_frequency']
    
    for col in numeric_cols:
        sel_mean = selected[col].mean()
        not_sel_mean = not_selected[col].mean()
        diff = sel_mean - not_sel_mean
        print(f"  {col:25} {diff:+.3f} (sel: {sel_mean:.3f}, not: {not_sel_mean:.3f})")

def train_models_and_evaluate(X_train_full, y_train_full, X_train_subset, y_train_subset):
    """Train models on full and subset data, compare performance"""
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    models = {
        'xgb': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
        'lgb': LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1),
        'rf': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n{name.upper()}:")
        
        # Train on full data
        model_full = model.__class__(**model.get_params())
        model_full.fit(X_train_full, y_train_full)
        
        # Self-accuracy on full train
        train_pred_full = model_full.predict(X_train_full)
        train_acc_full = (train_pred_full == y_train_full).mean()
        
        # Train on subset
        model_subset = model.__class__(**model.get_params())
        model_subset.fit(X_train_subset, y_train_subset)
        
        # Self-accuracy on subset
        train_pred_subset = model_subset.predict(X_train_subset)
        train_acc_subset = (train_pred_subset == y_train_subset).mean()
        
        # Cross-accuracy: subset model on full data
        cross_pred = model_subset.predict(X_train_full)
        cross_acc = (cross_pred == y_train_full).mean()
        
        print(f"  Full model on full data:     {train_acc_full:.4f}")
        print(f"  Subset model on subset data: {train_acc_subset:.4f}")
        print(f"  Subset model on full data:   {cross_acc:.4f}")
        
        results.append({
            'model': name,
            'full_self_acc': train_acc_full,
            'subset_self_acc': train_acc_subset,
            'cross_acc': cross_acc,
            'model_full': model_full,
            'model_subset': model_subset
        })
    
    return results

def find_perfect_subset(train_df, X_train, y_train, train_probs):
    """Try different percentiles to find subset that gives 100% accuracy"""
    print("\n" + "="*60)
    print("SEARCHING FOR PERFECT SUBSET")
    print("="*60)
    
    percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
    best_results = []
    
    for pct in percentiles:
        print(f"\n--- Testing percentile {pct}% ---")
        
        # Select subset
        threshold = np.percentile(train_probs, pct)
        mask = train_probs >= threshold
        
        if mask.sum() < 100:  # Too few samples
            print(f"Skipping - only {mask.sum()} samples")
            continue
        
        X_subset = X_train[mask]
        y_subset = y_train[mask]
        
        # Quick test with XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        xgb.fit(X_subset, y_subset)
        
        # Self accuracy
        acc = xgb.score(X_subset, y_subset)
        
        # Check errors
        pred = xgb.predict(X_subset)
        errors = pred != y_subset
        error_rate = errors.mean()
        
        print(f"Samples: {mask.sum()}, Accuracy: {acc:.4f}, Errors: {errors.sum()}")
        
        best_results.append({
            'percentile': pct,
            'n_samples': mask.sum(),
            'accuracy': acc,
            'error_rate': error_rate,
            'mask': mask
        })
        
        if acc >= 0.999:  # Nearly perfect
            print("*** NEAR PERFECT ACCURACY FOUND! ***")
            # Analyze the errors
            if errors.sum() > 0:
                error_indices = train_df[mask].index[errors]
                print(f"Error indices: {list(error_indices)[:10]}")
    
    return best_results

def analyze_mislabeled_candidates(train_df, X_train, y_train):
    """Find training samples that might be mislabeled"""
    print("\n" + "="*60)
    print("FINDING MISLABELED CANDIDATES")
    print("="*60)
    
    # Train multiple models
    models = [
        XGBClassifier(n_estimators=200, max_depth=5, random_state=42),
        LGBMClassifier(n_estimators=200, num_leaves=31, random_state=42, verbosity=-1),
        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    ]
    
    predictions = []
    
    # Get out-of-fold predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model in models:
        oof_pred = np.zeros(len(X_train))
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            oof_pred[val_idx] = model.predict(X_train.iloc[val_idx])
        
        predictions.append(oof_pred)
    
    # Find consensus errors
    predictions = np.array(predictions)
    consensus = (predictions.mean(axis=0) > 0.5).astype(int)
    
    # Find disagreements with labels
    errors = consensus != y_train
    error_indices = train_df.index[errors]
    
    print(f"Found {errors.sum()} potential mislabeled samples ({errors.mean():.2%})")
    
    # Analyze error characteristics
    error_df = train_df[errors]
    print("\nError sample characteristics:")
    print(f"Personality distribution: {error_df['Personality'].value_counts()}")
    
    return error_indices, errors

def create_cleaned_dataset(train_df, error_indices):
    """Create dataset with suspected errors removed or corrected"""
    print("\n" + "="*60)
    print("CREATING CLEANED DATASET")
    print("="*60)
    
    # Option 1: Remove errors
    train_clean_removed = train_df.drop(error_indices)
    
    # Option 2: Flip labels
    train_clean_flipped = train_df.copy()
    for idx in error_indices:
        current = train_clean_flipped.loc[idx, 'Personality']
        train_clean_flipped.loc[idx, 'Personality'] = 'Introvert' if current == 'Extrovert' else 'Extrovert'
    
    print(f"Original size: {len(train_df)}")
    print(f"After removing errors: {len(train_clean_removed)}")
    print(f"Flipped labels: {len(error_indices)}")
    
    return train_clean_removed, train_clean_flipped

def main():
    # Load data
    train_df, test_df, X_train, X_test, y_train = prepare_data()
    
    # Adversarial validation
    train_probs, rf_adv = adversarial_validation(X_train, X_test)
    
    # Save adversarial probabilities
    train_df['adv_prob'] = train_probs
    train_df[['id', 'adv_prob', 'Personality']].to_csv(
        OUTPUT_DIR / 'train_adversarial_probabilities.csv', index=False
    )
    
    # Select optimal subset (80th percentile)
    train_subset, X_train_subset, y_train_subset, mask = select_optimal_train_subset(
        train_df, X_train, y_train, train_probs, percentile=80
    )
    
    # Analyze characteristics
    analyze_subset_characteristics(train_df, mask)
    
    # Train and evaluate models
    results = train_models_and_evaluate(X_train, y_train, X_train_subset, y_train_subset)
    
    # Search for perfect subset
    best_results = find_perfect_subset(train_df, X_train, y_train, train_probs)
    
    # Find mislabeled candidates
    error_indices, errors = analyze_mislabeled_candidates(train_df, X_train, y_train)
    
    # Create cleaned datasets
    train_clean_removed, train_clean_flipped = create_cleaned_dataset(train_df, error_indices)
    
    # Save cleaned datasets
    train_clean_removed.to_csv(OUTPUT_DIR / 'train_cleaned_removed.csv', index=False)
    train_clean_flipped.to_csv(OUTPUT_DIR / 'train_cleaned_flipped.csv', index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Adversarial probability distribution
    ax = axes[0, 0]
    ax.hist(train_probs, bins=50, alpha=0.7, color='blue')
    ax.axvline(np.percentile(train_probs, 80), color='red', linestyle='--', label='80th percentile')
    ax.set_xlabel('Probability of being test-like')
    ax.set_ylabel('Count')
    ax.set_title('Adversarial Validation Probabilities')
    ax.legend()
    
    # 2. Accuracy vs percentile
    ax = axes[0, 1]
    pcts = [r['percentile'] for r in best_results]
    accs = [r['accuracy'] for r in best_results]
    ax.plot(pcts, accs, 'o-', markersize=8)
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect accuracy')
    ax.set_xlabel('Percentile threshold')
    ax.set_ylabel('Self-accuracy')
    ax.set_title('Accuracy vs Subset Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Sample size vs accuracy
    ax = axes[1, 0]
    sizes = [r['n_samples'] for r in best_results]
    ax.scatter(sizes, accs, s=100, alpha=0.7)
    ax.set_xlabel('Number of samples in subset')
    ax.set_ylabel('Self-accuracy')
    ax.set_title('Sample Size vs Accuracy')
    ax.grid(True, alpha=0.3)
    
    # 4. Error analysis
    ax = axes[1, 1]
    personality_error_rate = train_df.groupby('Personality').apply(
        lambda x: errors[x.index].mean()
    )
    personality_error_rate.plot(kind='bar', ax=ax)
    ax.set_title('Error Rate by Personality Type')
    ax.set_ylabel('Error rate')
    ax.set_xlabel('Personality')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'adversarial_optimization_analysis.png', dpi=300)
    plt.close()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nKey findings:")
    print(f"1. Adversarial validation accuracy: {rf_adv.score(X_train, np.zeros(len(X_train))):.4f}")
    print(f"2. Found {len(error_indices)} potential mislabeled samples")
    print(f"3. Best subset accuracy: {max(r['accuracy'] for r in best_results):.4f}")
    print(f"\nCleaned datasets saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()