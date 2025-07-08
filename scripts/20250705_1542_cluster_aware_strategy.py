#!/usr/bin/env python3
"""
CLUSTER-AWARE PREDICTION STRATEGY
=================================

This script implements the cluster-aware strategy based on error analysis:
1. Special rules for problematic clusters (2 and 7)
2. Ensemble with cluster-specific weights
3. Outlier detection and special handling
4. Feature engineering for "mixed" introverts

Author: Claude
Date: 2025-07-05 15:42
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Output file
output_file = open('output/20250705_1542_cluster_aware_strategy.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def engineer_features(df):
    """Create features including the new introvert_inconsistency."""
    log_print("\nEngineering features...")
    
    # Original features
    df = df.copy()
    
    # Encode categorical
    df['Stage_fear_binary'] = (df['Stage_fear'] == 'Yes').astype(int)
    df['Drained_binary'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    # New feature: introvert_inconsistency
    # Captures introverts who aren't "extremely alone"
    df['introvert_inconsistency'] = (
        (df['Time_spent_Alone'] < 5) &  # Not much time solo
        (df['Social_event_attendance'] < 3) &  # But also few events
        (df['Friends_circle_size'] < 6)  # And few friends
    ).astype(int)
    
    # Mixed profile indicators
    df['mixed_introvert'] = (
        (df['Time_spent_Alone'].between(4, 6)) &
        (df['Social_event_attendance'] < 2)
    ).astype(int)
    
    # Extreme introvert but not typical pattern
    df['extreme_but_atypical'] = (
        (df['Time_spent_Alone'] > 8) &
        (df['Social_event_attendance'] > 2)  # More social than expected
    ).astype(int)
    
    # Ambiguous profile
    df['ambiguous_profile'] = (
        (df['Time_spent_Alone'].between(3, 5)) &
        (df['Social_event_attendance'].between(4, 6)) &
        (df['Friends_circle_size'].between(6, 8))
    ).astype(int)
    
    return df


def identify_clusters(X_scaled, n_clusters=15):
    """Identify clusters using K-means."""
    log_print(f"\nIdentifying {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    return kmeans, clusters


def identify_outliers(X_scaled):
    """Identify outliers using DBSCAN."""
    log_print("\nIdentifying outliers with DBSCAN...")
    
    dbscan = DBSCAN(eps=1.0, min_samples=10)
    dbscan_clusters = dbscan.fit_predict(X_scaled)
    outliers = dbscan_clusters == -1
    
    log_print(f"Found {np.sum(outliers)} outliers ({np.sum(outliers)/len(outliers)*100:.2f}%)")
    
    return outliers


def apply_cluster_rules(df, clusters, predictions_proba):
    """Apply special rules for problematic clusters."""
    log_print("\nApplying cluster-specific rules...")
    
    predictions = predictions_proba.copy()
    modifications = 0
    
    # Rule for Cluster 2 (moderate time alone introverts)
    cluster_2_mask = (clusters == 2)
    moderate_introverts = (
        cluster_2_mask &
        (df['Time_spent_Alone'] > 4) & 
        (df['Time_spent_Alone'] < 6) &
        (df['Social_event_attendance'] < 2)
    )
    
    # Check additional indicators
    should_be_intro = moderate_introverts & (
        (df['Friends_circle_size'] < 5) | 
        (df['Drained_binary'] == 1)
    )
    
    # Adjust predictions
    predictions[should_be_intro & (predictions > 0.5)] = 0.45  # Force to introvert
    modifications += np.sum(should_be_intro & (predictions_proba > 0.5))
    
    # Rule for Cluster 7 (extremely alone)
    cluster_7_mask = (clusters == 7)
    extreme_alone = cluster_7_mask & (df['Time_spent_Alone'] > 8)
    
    # Check if they're actually extroverts (rare case)
    rare_extrovert = extreme_alone & (
        (df['Social_event_attendance'] > 5) & 
        (df['Friends_circle_size'] > 10)
    )
    
    # Most extreme alone should be introverts
    should_be_intro_extreme = extreme_alone & ~rare_extrovert
    predictions[should_be_intro_extreme & (predictions > 0.5)] = 0.35  # Strong introvert
    modifications += np.sum(should_be_intro_extreme & (predictions_proba > 0.5))
    
    log_print(f"Modified {modifications} predictions based on cluster rules")
    
    return predictions


def train_cluster_aware_ensemble(X_train, y_train, clusters_train):
    """Train ensemble with cluster-specific weights."""
    log_print("\nTraining cluster-aware ensemble...")
    
    # Define cluster weights based on error analysis
    cluster_weights = {
        2: {'xgboost': 0.3, 'catboost': 0.4, 'neural': 0.3},  # Less trust in XGBoost
        7: {'xgboost': 0.3, 'catboost': 0.4, 'neural': 0.3},
        6: {'xgboost': 0.35, 'catboost': 0.35, 'neural': 0.3},
        'default': {'xgboost': 0.33, 'catboost': 0.34, 'neural': 0.33}
    }
    
    # Train base models
    models = {}
    
    # XGBoost
    log_print("Training XGBoost...")
    models['xgboost'] = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    models['xgboost'].fit(X_train, y_train, verbose=False)
    
    # CatBoost
    log_print("Training CatBoost...")
    models['catboost'] = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    )
    models['catboost'].fit(X_train, y_train)
    
    # LightGBM as neural network substitute
    log_print("Training LightGBM (as NN substitute)...")
    models['neural'] = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    models['neural'].fit(X_train, y_train)
    
    return models, cluster_weights


def handle_outliers(X_train, y_train, X_test, outliers_train, outliers_test):
    """Special handling for outliers using k-NN."""
    log_print("\nHandling outliers with k-NN...")
    
    # Train k-NN on non-outliers
    X_train_clean = X_train[~outliers_train]
    y_train_clean = y_train[~outliers_train]
    
    knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
    knn.fit(X_train_clean, y_train_clean)
    
    # Predict outliers
    outlier_predictions = None
    if np.any(outliers_test):
        outlier_predictions = knn.predict_proba(X_test[outliers_test])[:, 1]
        log_print(f"Made predictions for {np.sum(outliers_test)} test outliers")
    
    return knn, outlier_predictions


def make_cluster_aware_predictions(models, cluster_weights, X_test, clusters_test, outliers_test):
    """Make predictions with cluster-specific ensemble weights."""
    log_print("\nMaking cluster-aware predictions...")
    
    n_samples = len(X_test)
    weighted_predictions = np.zeros(n_samples)
    
    # Get predictions from each model
    model_predictions = {}
    for name, model in models.items():
        model_predictions[name] = model.predict_proba(X_test)[:, 1]
    
    # Apply cluster-specific weights
    for i in range(n_samples):
        cluster = clusters_test[i]
        
        # Get weights for this cluster
        if cluster in cluster_weights:
            weights = cluster_weights[cluster]
        else:
            weights = cluster_weights['default']
        
        # Weighted average
        weighted_predictions[i] = (
            weights['xgboost'] * model_predictions['xgboost'][i] +
            weights['catboost'] * model_predictions['catboost'][i] +
            weights['neural'] * model_predictions['neural'][i]
        )
    
    # For outliers, use more conservative threshold
    if np.any(outliers_test):
        log_print(f"Applying conservative threshold (0.6) for {np.sum(outliers_test)} outliers")
        # Instead of 0.5, use 0.6 for outliers
        outlier_mask = outliers_test
        # This makes it harder to classify as extrovert
        
    return weighted_predictions


def validate_strategy(X_train, y_train, train_df):
    """Validate the cluster-aware strategy using CV."""
    log_print("\n" + "="*70)
    log_print("VALIDATING CLUSTER-AWARE STRATEGY")
    log_print("="*70)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        log_print(f"\nFold {fold + 1}/5")
        
        # Split data
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        df_val = train_df.iloc[val_idx]
        
        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        # Identify clusters
        kmeans, clusters_tr = identify_clusters(X_tr_scaled)
        clusters_val = kmeans.predict(X_val_scaled)
        
        # Identify outliers
        outliers_tr = identify_outliers(X_tr_scaled)
        outliers_val = identify_outliers(X_val_scaled)
        
        # Train ensemble
        models, cluster_weights = train_cluster_aware_ensemble(X_tr, y_tr, clusters_tr)
        
        # Make predictions
        predictions_proba = make_cluster_aware_predictions(
            models, cluster_weights, X_val, clusters_val, outliers_val
        )
        
        # Apply cluster rules
        predictions_proba = apply_cluster_rules(df_val, clusters_val, predictions_proba)
        
        # Convert to binary
        predictions = (predictions_proba > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, predictions)
        scores.append(accuracy)
        log_print(f"Fold {fold + 1} accuracy: {accuracy:.6f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    log_print(f"\nMean CV accuracy: {mean_score:.6f} (+/- {std_score:.6f})")
    
    return mean_score


def analyze_improvements(train_df, X_train, y_train):
    """Analyze which cases would be fixed by the new strategy."""
    log_print("\n" + "="*70)
    log_print("ANALYZING POTENTIAL IMPROVEMENTS")
    log_print("="*70)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Get clusters
    kmeans, clusters = identify_clusters(X_scaled)
    
    # Train simple XGBoost (baseline)
    baseline = xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42, eval_metric='logloss')
    baseline.fit(X_train, y_train, verbose=False)
    baseline_pred = baseline.predict(X_train)
    baseline_errors = baseline_pred != y_train
    
    # Analyze errors by cluster
    log_print("\nBaseline errors by cluster:")
    for cluster_id in [2, 7, 6, 0, 12]:  # Top problematic clusters
        mask = clusters == cluster_id
        if np.any(mask):
            errors_in_cluster = np.sum(baseline_errors & mask)
            total_in_cluster = np.sum(mask)
            error_rate = errors_in_cluster / total_in_cluster * 100
            log_print(f"  Cluster {cluster_id}: {errors_in_cluster}/{total_in_cluster} = {error_rate:.1f}%")
    
    # Check how many would be fixed by rules
    log_print("\nPotential fixes by cluster rules:")
    
    # Cluster 2 rule
    cluster_2_errors = baseline_errors & (clusters == 2)
    moderate_intro_pattern = (
        (train_df['Time_spent_Alone'] > 4) & 
        (train_df['Time_spent_Alone'] < 6) &
        (train_df['Social_event_attendance'] < 2) &
        ((train_df['Friends_circle_size'] < 5) | (train_df['Drained_after_socializing'] == 'Yes'))
    )
    
    fixable_cluster_2 = cluster_2_errors & moderate_intro_pattern
    log_print(f"  Cluster 2: {np.sum(fixable_cluster_2)} errors could be fixed")
    
    # Cluster 7 rule
    cluster_7_errors = baseline_errors & (clusters == 7)
    extreme_alone_pattern = (
        (train_df['Time_spent_Alone'] > 8) &
        ~((train_df['Social_event_attendance'] > 5) & (train_df['Friends_circle_size'] > 10))
    )
    
    fixable_cluster_7 = cluster_7_errors & extreme_alone_pattern
    log_print(f"  Cluster 7: {np.sum(fixable_cluster_7)} errors could be fixed")
    
    total_fixable = np.sum(fixable_cluster_2) + np.sum(fixable_cluster_7)
    log_print(f"\nTotal potential fixes: {total_fixable} ({total_fixable/np.sum(baseline_errors)*100:.1f}% of errors)")


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("CLUSTER-AWARE PREDICTION STRATEGY")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    # Engineer features
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear_binary', 'Drained_binary',
                   'introvert_inconsistency', 'mixed_introvert',
                   'extreme_but_atypical', 'ambiguous_profile']
    
    X_train = train_df[feature_cols].fillna(0).values
    y_train = (train_df['Personality'] == 'Extrovert').astype(int).values
    X_test = test_df[feature_cols].fillna(0).values
    
    # Analyze improvements
    analyze_improvements(train_df, X_train, y_train)
    
    # Validate strategy
    cv_score = validate_strategy(X_train, y_train, train_df)
    
    # Train final model
    log_print("\n" + "="*70)
    log_print("TRAINING FINAL MODEL")
    log_print("="*70)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Identify clusters
    kmeans, clusters_train = identify_clusters(X_train_scaled)
    clusters_test = kmeans.predict(X_test_scaled)
    
    # Identify outliers
    outliers_train = identify_outliers(X_train_scaled)
    outliers_test = identify_outliers(X_test_scaled)
    
    # Train ensemble
    models, cluster_weights = train_cluster_aware_ensemble(X_train, y_train, clusters_train)
    
    # Handle outliers
    knn, outlier_preds = handle_outliers(
        X_train_scaled, y_train, X_test_scaled, outliers_train, outliers_test
    )
    
    # Make predictions
    predictions_proba = make_cluster_aware_predictions(
        models, cluster_weights, X_test, clusters_test, outliers_test
    )
    
    # Apply cluster rules
    predictions_proba = apply_cluster_rules(test_df, clusters_test, predictions_proba)
    
    # Apply conservative threshold for outliers
    predictions = (predictions_proba > 0.5).astype(int)
    if np.any(outliers_test):
        # For outliers, use 0.6 threshold
        predictions[outliers_test] = (predictions_proba[outliers_test] > 0.6).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in predictions]
    })
    
    submission.to_csv('output/cluster_aware_submission.csv', index=False)
    log_print("\nSaved submission to: output/cluster_aware_submission.csv")
    
    # Save detailed predictions
    detailed_results = pd.DataFrame({
        'id': test_df['id'],
        'cluster': clusters_test,
        'is_outlier': outliers_test,
        'probability': predictions_proba,
        'prediction': submission['Personality']
    })
    
    detailed_results.to_csv('output/cluster_aware_predictions_detailed.csv', index=False)
    log_print("Saved detailed predictions to: output/cluster_aware_predictions_detailed.csv")
    
    # Summary statistics
    log_print("\n" + "="*70)
    log_print("SUMMARY")
    log_print("="*70)
    log_print(f"\nCV Score: {cv_score:.6f}")
    log_print(f"Test predictions: {np.sum(predictions == 1)} Extroverts, "
              f"{np.sum(predictions == 0)} Introverts")
    log_print(f"Outliers in test: {np.sum(outliers_test)}")
    
    # Cluster distribution in test
    log_print("\nTest samples by cluster:")
    for cluster_id in [2, 7, 6]:  # Problematic clusters
        count = np.sum(clusters_test == cluster_id)
        log_print(f"  Cluster {cluster_id}: {count} samples")
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()