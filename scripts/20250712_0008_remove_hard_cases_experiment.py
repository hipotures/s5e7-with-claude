#!/usr/bin/env python3
"""
Remove hard/confusing cases from training to improve model performance
Strategy: Train -> Find errors -> Remove errors and similar cases -> Retrain
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
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

def train_initial_model(X_train, y_train):
    """Train initial model and find misclassified samples"""
    print("\n" + "="*60)
    print("TRAINING INITIAL MODEL")
    print("="*60)
    
    # Use XGBoost as primary model
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    # Get predictions
    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)[:, 1]
    
    # Find errors
    errors = train_pred != y_train
    error_indices = np.where(errors)[0]
    
    # Calculate metrics
    accuracy = (train_pred == y_train).mean()
    
    print(f"Initial model accuracy: {accuracy:.4f}")
    print(f"Number of errors: {errors.sum()} ({errors.mean():.2%})")
    
    # Analyze error confidence
    error_probas = train_proba[errors]
    print(f"\nError confidence distribution:")
    print(f"  Mean probability: {error_probas.mean():.3f}")
    print(f"  Min probability: {error_probas.min():.3f}")
    print(f"  Max probability: {error_probas.max():.3f}")
    
    # High confidence errors (model was very sure but wrong)
    high_conf_errors = errors & ((train_proba > 0.8) | (train_proba < 0.2))
    print(f"  High confidence errors: {high_conf_errors.sum()}")
    
    return model, errors, error_indices, train_proba

def find_similar_samples(X_train, error_indices, n_neighbors=10):
    """Find samples similar to the errors"""
    print("\n" + "="*60)
    print("FINDING SIMILAR SAMPLES TO ERRORS")
    print("="*60)
    
    # Standardize features for distance calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 because sample includes itself
    nn.fit(X_scaled)
    
    # Find neighbors for each error sample
    all_similar_indices = set()
    
    for error_idx in error_indices:
        distances, indices = nn.kneighbors([X_scaled[error_idx]])
        # Skip first index (self)
        similar_indices = indices[0][1:]
        all_similar_indices.update(similar_indices)
    
    # Add original error indices
    all_similar_indices.update(error_indices)
    
    print(f"Error samples: {len(error_indices)}")
    print(f"Similar samples found: {len(all_similar_indices) - len(error_indices)}")
    print(f"Total samples to remove: {len(all_similar_indices)}")
    
    return list(all_similar_indices)

def analyze_removed_samples(train_df, X_train, y_train, remove_indices):
    """Analyze characteristics of removed samples"""
    print("\n" + "="*60)
    print("ANALYZING REMOVED SAMPLES")
    print("="*60)
    
    # Create masks
    remove_mask = np.zeros(len(train_df), dtype=bool)
    remove_mask[remove_indices] = True
    
    removed_df = train_df[remove_mask]
    kept_df = train_df[~remove_mask]
    
    print(f"\nPersonality distribution:")
    print(f"Original dataset:")
    print(train_df['Personality'].value_counts(normalize=True))
    print(f"\nRemoved samples:")
    print(removed_df['Personality'].value_counts(normalize=True))
    print(f"\nKept samples:")
    print(kept_df['Personality'].value_counts(normalize=True))
    
    # Analyze features
    print("\nFeature differences (removed vs kept):")
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                    'Going_outside', 'Post_frequency']
    
    for col in numeric_cols:
        removed_mean = removed_df[col].mean()
        kept_mean = kept_df[col].mean()
        diff = removed_mean - kept_mean
        print(f"  {col:25} {diff:+.3f} (removed: {removed_mean:.3f}, kept: {kept_mean:.3f})")
    
    # Missing values
    removed_missing = removed_df[numeric_cols].isnull().any(axis=1).mean()
    kept_missing = kept_df[numeric_cols].isnull().any(axis=1).mean()
    print(f"\nMissing values:")
    print(f"  Removed samples: {removed_missing:.2%}")
    print(f"  Kept samples: {kept_missing:.2%}")
    
    return removed_df, kept_df

def train_cleaned_model(X_train, y_train, remove_indices):
    """Train model on cleaned dataset"""
    print("\n" + "="*60)
    print("TRAINING MODEL ON CLEANED DATA")
    print("="*60)
    
    # Create clean dataset
    keep_mask = np.ones(len(X_train), dtype=bool)
    keep_mask[remove_indices] = False
    
    X_clean = X_train[keep_mask]
    y_clean = y_train[keep_mask]
    
    print(f"Original training size: {len(X_train)}")
    print(f"Cleaned training size: {len(X_clean)} ({len(X_clean)/len(X_train):.1%})")
    
    # Train multiple models
    models = {
        'xgb': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, 
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        'lgb': LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1),
        'cat': CatBoostClassifier(n_estimators=300, learning_rate=0.05, depth=6,
                                 subsample=0.8, verbose=False, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name.upper()}...")
        
        # Train on cleaned data
        model.fit(X_clean, y_clean)
        
        # Evaluate on cleaned data
        clean_pred = model.predict(X_clean)
        clean_acc = (clean_pred == y_clean).mean()
        
        # Evaluate on full data
        full_pred = model.predict(X_train)
        full_acc = (full_pred == y_train).mean()
        
        # Check if removed samples would still be errors
        removed_pred = model.predict(X_train.iloc[remove_indices])
        removed_true = y_train.iloc[remove_indices]
        removed_acc = (removed_pred == removed_true).mean()
        
        print(f"  Accuracy on cleaned data: {clean_acc:.4f}")
        print(f"  Accuracy on full data: {full_acc:.4f}")
        print(f"  Accuracy on removed samples: {removed_acc:.4f}")
        
        results[name] = {
            'model': model,
            'clean_acc': clean_acc,
            'full_acc': full_acc,
            'removed_acc': removed_acc
        }
    
    return results

def iterative_cleaning(X_train, y_train, max_iterations=5, min_improvement=0.0001):
    """Iteratively remove hard cases until no improvement"""
    print("\n" + "="*60)
    print("ITERATIVE CLEANING PROCESS")
    print("="*60)
    
    current_X = X_train.copy()
    current_y = y_train.copy()
    removed_indices_total = []
    iteration_results = []
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Train model
        model = XGBClassifier(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
        model.fit(current_X, current_y)
        
        # Find errors
        pred = model.predict(current_X)
        errors = pred != current_y
        error_count = errors.sum()
        accuracy = 1 - errors.mean()
        
        print(f"Current accuracy: {accuracy:.4f}")
        print(f"Errors: {error_count}")
        
        if error_count == 0:
            print("Perfect accuracy achieved!")
            break
        
        # Find indices to remove (errors + similar)
        error_indices = np.where(errors)[0]
        similar_indices = find_similar_samples(current_X, error_indices, n_neighbors=5)
        
        # Remove samples
        keep_mask = np.ones(len(current_X), dtype=bool)
        keep_mask[similar_indices] = False
        
        current_X = current_X[keep_mask]
        current_y = current_y[keep_mask]
        
        # Track removed indices
        removed_indices_total.extend(similar_indices)
        
        iteration_results.append({
            'iteration': iteration + 1,
            'accuracy': accuracy,
            'errors': error_count,
            'removed': len(similar_indices),
            'remaining': len(current_X)
        })
        
        # Check improvement
        if iteration > 0:
            improvement = accuracy - iteration_results[-2]['accuracy']
            if improvement < min_improvement:
                print(f"Improvement too small ({improvement:.6f}), stopping")
                break
    
    return iteration_results, removed_indices_total

def create_visualizations(train_df, remove_indices, iteration_results):
    """Create comprehensive visualizations"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy improvement over iterations
    if iteration_results:
        ax = axes[0, 0]
        iterations = [r['iteration'] for r in iteration_results]
        accuracies = [r['accuracy'] for r in iteration_results]
        ax.plot(iterations, accuracies, 'o-', markersize=8, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Improvement Through Iterative Cleaning')
        ax.grid(True, alpha=0.3)
    
    # 2. Samples remaining
    if iteration_results:
        ax = axes[0, 1]
        remaining = [r['remaining'] for r in iteration_results]
        ax.plot(iterations, remaining, 'o-', markersize=8, linewidth=2, color='orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Training Samples Remaining')
        ax.set_title('Dataset Size Reduction')
        ax.grid(True, alpha=0.3)
    
    # 3. Personality distribution in removed samples
    ax = axes[1, 0]
    remove_mask = np.zeros(len(train_df), dtype=bool)
    remove_mask[remove_indices[:len(train_df)]] = True  # Handle index overflow
    
    removed_personality = train_df[remove_mask]['Personality'].value_counts()
    kept_personality = train_df[~remove_mask]['Personality'].value_counts()
    
    x = np.arange(2)
    width = 0.35
    
    ax.bar(x - width/2, [removed_personality.get('Introvert', 0), removed_personality.get('Extrovert', 0)], 
           width, label='Removed', alpha=0.8)
    ax.bar(x + width/2, [kept_personality.get('Introvert', 0), kept_personality.get('Extrovert', 0)], 
           width, label='Kept', alpha=0.8)
    
    ax.set_xlabel('Personality')
    ax.set_ylabel('Count')
    ax.set_title('Personality Distribution: Removed vs Kept')
    ax.set_xticks(x)
    ax.set_xticklabels(['Introvert', 'Extrovert'])
    ax.legend()
    
    # 4. Feature analysis of removed samples
    ax = axes[1, 1]
    remove_mask_safe = remove_mask[:len(train_df)]
    if remove_mask_safe.sum() > 0:
        feature_diffs = []
        feature_names = []
        
        numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                        'Going_outside', 'Post_frequency']
        
        for col in numeric_cols:
            removed_mean = train_df[remove_mask_safe][col].mean()
            kept_mean = train_df[~remove_mask_safe][col].mean()
            diff_pct = (removed_mean - kept_mean) / kept_mean * 100
            feature_diffs.append(diff_pct)
            feature_names.append(col.replace('_', ' '))
        
        colors = ['red' if d < 0 else 'green' for d in feature_diffs]
        ax.barh(feature_names, feature_diffs, color=colors, alpha=0.7)
        ax.set_xlabel('Difference from Kept Samples (%)')
        ax.set_title('Feature Characteristics of Removed Samples')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hard_cases_removal_analysis.png', dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {OUTPUT_DIR / 'hard_cases_removal_analysis.png'}")

def main():
    # Load data
    train_df, test_df, X_train, X_test, y_train = prepare_data()
    
    # Train initial model and find errors
    initial_model, errors, error_indices, train_proba = train_initial_model(X_train, y_train)
    
    # Find similar samples to errors
    similar_indices = find_similar_samples(X_train, error_indices, n_neighbors=10)
    
    # Analyze removed samples
    removed_df, kept_df = analyze_removed_samples(train_df, X_train, y_train, similar_indices)
    
    # Train on cleaned data
    cleaned_results = train_cleaned_model(X_train, y_train, similar_indices)
    
    # Try iterative cleaning
    iteration_results, all_removed = iterative_cleaning(X_train, y_train)
    
    # Create visualizations
    create_visualizations(train_df, similar_indices, iteration_results)
    
    # Save cleaned datasets
    keep_mask = np.ones(len(train_df), dtype=bool)
    keep_mask[similar_indices[:len(train_df)]] = False
    
    train_cleaned = train_df[keep_mask]
    train_cleaned.to_csv(OUTPUT_DIR / 'train_hard_cases_removed.csv', index=False)
    
    # Save list of removed indices
    removed_info = pd.DataFrame({
        'id': train_df.iloc[similar_indices[:len(train_df)]]['id'],
        'personality': train_df.iloc[similar_indices[:len(train_df)]]['Personality'],
        'probability': train_proba[similar_indices[:len(train_df)]]
    })
    removed_info.to_csv(OUTPUT_DIR / 'removed_hard_cases_info.csv', index=False)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nKey findings:")
    print(f"1. Initial accuracy: {1 - errors.mean():.4f}")
    print(f"2. Samples removed: {len(similar_indices)} ({len(similar_indices)/len(train_df):.1%})")
    print(f"3. Best cleaned accuracy: {max(r['clean_acc'] for r in cleaned_results.values()):.4f}")
    
    if iteration_results:
        print(f"4. Iterative cleaning achieved: {iteration_results[-1]['accuracy']:.4f}")
        print(f"   Total samples removed: {len(all_removed)}")

if __name__ == "__main__":
    main()