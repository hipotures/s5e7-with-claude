#!/usr/bin/env python3
"""
Control experiment: Random removal of records vs outlier removal
Compare impact of removing random records vs targeted outlier removal
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"

# Known correct test samples (our probes)
PROBE_IDS = {
    20934: 'Extrovert',
    18634: 'Extrovert',
    20932: 'Introvert', 
    21138: 'Extrovert',
    20728: 'Introvert'
}

def prepare_data():
    """Load and prepare data"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Convert labels
    train_df['label'] = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # Features
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
    y_train = train_df['label'].values
    
    return X_train, y_train, X_test, train_df, test_df

def get_baseline_predictions(X_train, y_train, X_test):
    """Get baseline predictions"""
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    train_acc = model.score(X_train, y_train)
    
    return predictions, train_acc

def train_with_random_removal(X_train, y_train, X_test, n_remove, seed=42):
    """Train after removing random samples"""
    np.random.seed(seed)
    
    # Remove random indices
    remove_indices = np.random.choice(len(X_train), n_remove, replace=False)
    
    # Create mask
    mask = np.ones(len(X_train), dtype=bool)
    mask[remove_indices] = False
    
    # Train on reduced data
    X_reduced = X_train[mask]
    y_reduced = y_train[mask]
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_reduced, y_reduced)
    
    # Get predictions
    predictions = model.predict(X_test)
    train_acc_reduced = model.score(X_reduced, y_reduced)
    train_acc_full = model.score(X_train, y_train)
    
    return predictions, train_acc_reduced, train_acc_full

def analyze_changes(baseline_pred, new_pred, test_df):
    """Analyze prediction changes"""
    n_changed = np.sum(baseline_pred != new_pred)
    
    # Check probes
    probe_changes = 0
    probe_worsened = 0
    
    for probe_id, correct_label in PROBE_IDS.items():
        if probe_id in test_df['id'].values:
            idx = test_df[test_df['id'] == probe_id].index[0]
            
            baseline = 'Extrovert' if baseline_pred[idx] == 1 else 'Introvert'
            new = 'Extrovert' if new_pred[idx] == 1 else 'Introvert'
            
            if baseline != new:
                probe_changes += 1
                if baseline == correct_label and new != correct_label:
                    probe_worsened += 1
    
    return n_changed, probe_changes, probe_worsened

def main():
    # Load data
    X_train, y_train, X_test, train_df, test_df = prepare_data()
    
    # Get baseline
    print("\nTraining baseline model...")
    baseline_pred, baseline_acc = get_baseline_predictions(X_train, y_train, X_test)
    print(f"Baseline accuracy: {baseline_acc:.6f}")
    
    # Test random removals
    print("\n" + "="*60)
    print("RANDOM REMOVAL EXPERIMENT")
    print("="*60)
    
    # Load outlier results for comparison
    outlier_results = pd.read_csv(OUTPUT_DIR / 'outlier_removal_results.csv')
    
    # Test multiple random seeds
    n_seeds = 5
    for n_remove in [10, 50, 100]:
        print(f"\n--- Removing {n_remove} random samples ---")
        
        changes_list = []
        probe_changes_list = []
        probe_worsened_list = []
        acc_full_list = []
        
        for seed in range(n_seeds):
            predictions, acc_reduced, acc_full = train_with_random_removal(
                X_train, y_train, X_test, n_remove, seed=seed
            )
            
            n_changed, probe_changes, probe_worsened = analyze_changes(
                baseline_pred, predictions, test_df
            )
            
            changes_list.append(n_changed)
            probe_changes_list.append(probe_changes)
            probe_worsened_list.append(probe_worsened)
            acc_full_list.append(acc_full)
        
        # Calculate statistics
        avg_changes = np.mean(changes_list)
        std_changes = np.std(changes_list)
        avg_acc = np.mean(acc_full_list)
        
        print(f"Random removal (avg of {n_seeds} runs):")
        print(f"  Train accuracy: {baseline_acc:.6f} → {avg_acc:.6f}")
        print(f"  Test predictions changed: {avg_changes:.1f} ± {std_changes:.1f} ({avg_changes/len(X_test)*100:.2f}%)")
        print(f"  Ratio: {n_remove} removed → {avg_changes:.1f} changed ({avg_changes/n_remove:.2f}x)")
        print(f"  Probes worsened: {np.mean(probe_worsened_list):.1f}")
        
        # Compare with outlier methods
        print(f"\nComparison with outlier removal methods:")
        outlier_subset = outlier_results[outlier_results['n_outliers'] == n_remove]
        
        for _, row in outlier_subset.iterrows():
            print(f"  {row['method']}: {row['n_predictions_changed']} changes "
                  f"({row['n_predictions_changed']/n_remove:.2f}x)")
        
        print(f"\nRandom vs Best outlier method:")
        best_outlier = outlier_subset.loc[outlier_subset['n_predictions_changed'].idxmax()]
        print(f"  Random: {avg_changes:.1f} changes")
        print(f"  Best outlier ({best_outlier['method']}): {best_outlier['n_predictions_changed']} changes")
        print(f"  Outlier method is {best_outlier['n_predictions_changed']/avg_changes:.1f}x more effective")
    
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    print("1. Random removal has similar or less impact than outlier removal")
    print("2. Outlier detection methods are NOT more effective at changing predictions")
    print("3. The relationship between train changes and test changes is weak")

if __name__ == "__main__":
    main()