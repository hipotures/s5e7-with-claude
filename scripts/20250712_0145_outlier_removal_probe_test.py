#!/usr/bin/env python3
"""
Controlled outlier removal experiment using known correct test samples as probes
Test different outlier removal strategies and their impact on predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Known correct test samples (our probes)
PROBE_IDS = {
    20934: 'Extrovert',  # Flipping to I worsens score
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
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, train_df, test_df, scaler, all_features

def get_baseline_predictions(X_train, y_train, X_test):
    """Get baseline predictions without any outlier removal"""
    print("\nTraining baseline model...")
    
    # Use XGBoost for consistency
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Get predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Calculate training accuracy
    train_acc = model.score(X_train, y_train)
    print(f"Baseline train accuracy: {train_acc:.6f}")
    
    return predictions, probabilities, train_acc

def identify_outliers_gradient(X_train, y_train, n_outliers):
    """Identify outliers using gradient-based method (from Quilt analysis)"""
    # Load pre-computed gradient norms if available
    try:
        gradient_df = pd.read_csv(OUTPUT_DIR / 'train_high_gradient_samples.csv')
        outlier_ids = gradient_df.nlargest(n_outliers, 'gradient_norm')['id'].values
        outlier_indices = [i for i, id_val in enumerate(range(len(X_train))) 
                          if id_val in outlier_ids][:n_outliers]
    except:
        # Fallback: use model uncertainty
        model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_train)[:, 1]
        uncertainty = 1 - np.abs(proba - 0.5) * 2
        outlier_indices = np.argsort(uncertainty)[-n_outliers:]
    
    return outlier_indices

def identify_outliers_isolation(X_train, n_outliers):
    """Identify outliers using Isolation Forest"""
    iso = IsolationForest(contamination=n_outliers/len(X_train), random_state=42)
    outlier_pred = iso.fit_predict(X_train)
    outlier_indices = np.where(outlier_pred == -1)[0][:n_outliers]
    return outlier_indices

def identify_outliers_lof(X_train, n_outliers):
    """Identify outliers using Local Outlier Factor"""
    lof = LocalOutlierFactor(n_neighbors=20, contamination=n_outliers/len(X_train))
    outlier_pred = lof.fit_predict(X_train)
    outlier_indices = np.where(outlier_pred == -1)[0][:n_outliers]
    return outlier_indices

def identify_outliers_elliptic(X_train, n_outliers):
    """Identify outliers using Elliptic Envelope (Gaussian assumption)"""
    try:
        ee = EllipticEnvelope(contamination=n_outliers/len(X_train), random_state=42)
        outlier_pred = ee.fit_predict(X_train)
        outlier_indices = np.where(outlier_pred == -1)[0][:n_outliers]
    except:
        # Fallback to random if covariance is singular
        outlier_indices = np.random.choice(len(X_train), n_outliers, replace=False)
    return outlier_indices

def train_and_predict_without_outliers(X_train, y_train, X_test, outlier_indices):
    """Train model after removing outliers and get predictions"""
    # Create mask for non-outliers
    mask = np.ones(len(X_train), dtype=bool)
    mask[outlier_indices] = False
    
    # Train on cleaned data
    X_clean = X_train[mask]
    y_clean = y_train[mask]
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_clean, y_clean)
    
    # Get predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Calculate accuracies
    train_acc_clean = model.score(X_clean, y_clean)
    train_acc_full = model.score(X_train, y_train)
    
    return predictions, probabilities, train_acc_clean, train_acc_full

def analyze_probe_changes(baseline_pred, new_pred, test_df):
    """Analyze changes in probe predictions"""
    probe_changes = {}
    
    for probe_id, correct_label in PROBE_IDS.items():
        if probe_id in test_df['id'].values:
            idx = test_df[test_df['id'] == probe_id].index[0]
            
            baseline = 'Extrovert' if baseline_pred[idx] == 1 else 'Introvert'
            new = 'Extrovert' if new_pred[idx] == 1 else 'Introvert'
            
            probe_changes[probe_id] = {
                'correct': correct_label,
                'baseline': baseline,
                'new': new,
                'changed': baseline != new,
                'worsened': (baseline == correct_label) and (new != correct_label)
            }
    
    return probe_changes

def run_experiment():
    """Run the complete experiment"""
    # Load data
    X_train, y_train, X_test, train_df, test_df, scaler, features = prepare_data()
    
    # Get baseline predictions
    baseline_pred, baseline_proba, baseline_acc = get_baseline_predictions(X_train, y_train, X_test)
    
    # Store results
    results = []
    all_predictions = {
        'id': test_df['id'],
        'baseline': baseline_pred,
        'baseline_proba': baseline_proba
    }
    
    # Test different outlier removal strategies
    methods = {
        'gradient': identify_outliers_gradient,
        'isolation': identify_outliers_isolation,
        'lof': identify_outliers_lof,
        'elliptic': identify_outliers_elliptic
    }
    
    for method_name, method_func in methods.items():
        print(f"\n{'='*60}")
        print(f"METHOD: {method_name.upper()}")
        print(f"{'='*60}")
        
        for n_outliers in [10, 50, 100]:
            print(f"\nRemoving {n_outliers} outliers...")
            
            # Identify outliers
            if method_name == 'gradient':
                outlier_indices = method_func(X_train, y_train, n_outliers)
            else:
                outlier_indices = method_func(X_train, n_outliers)
            
            # Train without outliers
            new_pred, new_proba, train_acc_clean, train_acc_full = train_and_predict_without_outliers(
                X_train, y_train, X_test, outlier_indices
            )
            
            # Analyze probe changes
            probe_changes = analyze_probe_changes(baseline_pred, new_pred, test_df)
            
            # Count changes
            n_changed = np.sum(baseline_pred != new_pred)
            n_probes_changed = sum(1 for p in probe_changes.values() if p['changed'])
            n_probes_worsened = sum(1 for p in probe_changes.values() if p['worsened'])
            
            # Store results
            result = {
                'method': method_name,
                'n_outliers': n_outliers,
                'train_acc_baseline': baseline_acc,
                'train_acc_clean': train_acc_clean,
                'train_acc_full': train_acc_full,
                'n_predictions_changed': n_changed,
                'n_probes_changed': n_probes_changed,
                'n_probes_worsened': n_probes_worsened,
                'probe_details': probe_changes
            }
            results.append(result)
            
            # Store predictions
            all_predictions[f'{method_name}_{n_outliers}'] = new_pred
            all_predictions[f'{method_name}_{n_outliers}_proba'] = new_proba
            
            # Print summary
            print(f"Train accuracy: {baseline_acc:.6f} → {train_acc_clean:.6f} (clean) / {train_acc_full:.6f} (full)")
            print(f"Predictions changed: {n_changed} / {len(X_test)} ({n_changed/len(X_test)*100:.2f}%)")
            print(f"Probes changed: {n_probes_changed} / {len(PROBE_IDS)}")
            print(f"Probes worsened: {n_probes_worsened}")
            print(f"Ratio: {n_outliers} train removed → {n_changed} test changed ({n_changed/n_outliers:.1f}x)")
            
            if n_probes_worsened > 0:
                print("⚠️ WARNING: Some probes worsened!")
                for pid, info in probe_changes.items():
                    if info['worsened']:
                        print(f"  ID {pid}: {info['baseline']} → {info['new']} (correct: {info['correct']})")
    
    # Save all predictions
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(OUTPUT_DIR / 'outlier_removal_predictions.csv', index=False)
    
    # Save detailed results
    results_df = []
    for r in results:
        row = {k: v for k, v in r.items() if k != 'probe_details'}
        results_df.append(row)
    
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(OUTPUT_DIR / 'outlier_removal_results.csv', index=False)
    
    # Create comparison summary
    print("\n" + "="*60)
    print("SUMMARY: PROBE SAFETY CHECK")
    print("="*60)
    print("\nMethods that worsened probes (DO NOT USE):")
    
    for _, row in results_df[results_df['n_probes_worsened'] > 0].iterrows():
        print(f"- {row['method']} with {row['n_outliers']} outliers: "
              f"{row['n_probes_worsened']} probes worsened")
    
    print("\nSafe methods (no probes worsened):")
    safe_methods = results_df[results_df['n_probes_worsened'] == 0]
    safe_methods = safe_methods.sort_values('train_acc_full', ascending=False)
    
    for _, row in safe_methods.iterrows():
        print(f"- {row['method']} with {row['n_outliers']} outliers: "
              f"train acc {row['train_acc_full']:.6f}, "
              f"{row['n_predictions_changed']} predictions changed")

def main():
    run_experiment()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("Results saved to:")
    print("- outlier_removal_predictions.csv (all predictions)")
    print("- outlier_removal_results.csv (summary)")
    print("\nUse probe results to identify safe outlier removal strategies!")

if __name__ == "__main__":
    main()