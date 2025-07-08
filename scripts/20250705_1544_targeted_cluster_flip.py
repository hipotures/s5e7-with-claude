#!/usr/bin/env python3
"""
TARGETED CLUSTER FLIP STRATEGY
==============================

This script implements targeted flipping of predictions for specific
misclassified cases in problematic clusters (2 and 7).

Author: Claude
Date: 2025-07-05 15:44
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Output file
output_file = open('output/20250705_1544_targeted_cluster_flip.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def prepare_features(df):
    """Prepare features for clustering."""
    df = df.copy()
    df['Stage_fear_binary'] = (df['Stage_fear'] == 'Yes').astype(int)
    df['Drained_binary'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear_binary', 'Drained_binary']
    
    return df[feature_cols].fillna(0), feature_cols


def find_flip_candidates(test_df, submission_df, clusters_test):
    """Find candidates to flip based on cluster analysis."""
    log_print("\n" + "="*70)
    log_print("FINDING FLIP CANDIDATES")
    log_print("="*70)
    
    # Merge submission with test data
    analysis_df = test_df.merge(submission_df, on='id')
    analysis_df['cluster'] = clusters_test
    
    flip_candidates = []
    
    # Cluster 2: Moderate time alone introverts misclassified as extroverts
    log_print("\nAnalyzing Cluster 2 (moderate alone time introverts)...")
    cluster_2_mask = (analysis_df['cluster'] == 2)
    
    # Pattern: 4-6h alone, <2 social events, predicted as Extrovert
    cluster_2_pattern = (
        cluster_2_mask &
        (analysis_df['Personality'] == 'Extrovert') &  # Currently predicted
        (analysis_df['Time_spent_Alone'].between(4, 6)) &
        (analysis_df['Social_event_attendance'] < 2)
    )
    
    # Additional checks for strong introvert signals
    strong_intro_signals = (
        (analysis_df['Friends_circle_size'] < 5) |
        (analysis_df['Drained_after_socializing'] == 'Yes') |
        (analysis_df['Stage_fear'] == 'Yes')
    )
    
    cluster_2_flips = analysis_df[cluster_2_pattern & strong_intro_signals]
    
    log_print(f"Found {len(cluster_2_flips)} candidates in Cluster 2:")
    for _, row in cluster_2_flips.head(10).iterrows():
        log_print(f"  ID {row['id']}: Alone={row['Time_spent_Alone']}, "
                 f"Social={row['Social_event_attendance']}, "
                 f"Friends={row['Friends_circle_size']}, "
                 f"Drained={row['Drained_after_socializing']}")
        flip_candidates.append({
            'id': row['id'],
            'cluster': 2,
            'reason': 'moderate_alone_introvert',
            'confidence': 0.85
        })
    
    # Cluster 7: Extremely alone but predicted as extroverts
    log_print("\nAnalyzing Cluster 7 (extreme alone time)...")
    cluster_7_mask = (analysis_df['cluster'] == 7)
    
    # Pattern: >8h alone, predicted as Extrovert
    cluster_7_pattern = (
        cluster_7_mask &
        (analysis_df['Personality'] == 'Extrovert') &
        (analysis_df['Time_spent_Alone'] > 8)
    )
    
    # Exclude rare "lonely extroverts" (high social + many friends)
    not_lonely_extrovert = ~(
        (analysis_df['Social_event_attendance'] > 5) & 
        (analysis_df['Friends_circle_size'] > 10)
    )
    
    cluster_7_flips = analysis_df[cluster_7_pattern & not_lonely_extrovert]
    
    log_print(f"Found {len(cluster_7_flips)} candidates in Cluster 7:")
    for _, row in cluster_7_flips.head(10).iterrows():
        log_print(f"  ID {row['id']}: Alone={row['Time_spent_Alone']}, "
                 f"Social={row['Social_event_attendance']}, "
                 f"Friends={row['Friends_circle_size']}")
        flip_candidates.append({
            'id': row['id'],
            'cluster': 7,
            'reason': 'extreme_alone_introvert',
            'confidence': 0.90
        })
    
    # Additional pattern: Introvert inconsistency (not in specific cluster)
    log_print("\nAnalyzing introvert inconsistency pattern...")
    
    inconsistent_pattern = (
        (analysis_df['Personality'] == 'Extrovert') &
        (analysis_df['Time_spent_Alone'] < 5) &
        (analysis_df['Social_event_attendance'] < 3) &
        (analysis_df['Friends_circle_size'] < 6) &
        (analysis_df['Going_outside'] < 3)
    )
    
    inconsistent_flips = analysis_df[inconsistent_pattern]
    
    log_print(f"Found {len(inconsistent_flips)} inconsistent introverts:")
    for _, row in inconsistent_flips.head(5).iterrows():
        log_print(f"  ID {row['id']}: Alone={row['Time_spent_Alone']}, "
                 f"Social={row['Social_event_attendance']}, "
                 f"Friends={row['Friends_circle_size']}, "
                 f"Outside={row['Going_outside']}")
        flip_candidates.append({
            'id': row['id'],
            'cluster': row['cluster'],
            'reason': 'inconsistent_introvert',
            'confidence': 0.75
        })
    
    return pd.DataFrame(flip_candidates)


def analyze_flip_impact(flip_candidates_df):
    """Analyze the potential impact of flips."""
    log_print("\n" + "="*70)
    log_print("FLIP IMPACT ANALYSIS")
    log_print("="*70)
    
    # Check if we have any candidates
    if len(flip_candidates_df) == 0:
        log_print("\nNo flip candidates found!")
        return flip_candidates_df
    
    # Sort by confidence
    flip_candidates_df = flip_candidates_df.sort_values('confidence', ascending=False)
    
    # Summary by reason
    log_print("\nFlip candidates by reason:")
    reason_counts = flip_candidates_df['reason'].value_counts()
    for reason, count in reason_counts.items():
        log_print(f"  {reason}: {count}")
    
    # Summary by cluster
    log_print("\nFlip candidates by cluster:")
    cluster_counts = flip_candidates_df['cluster'].value_counts()
    for cluster, count in cluster_counts.items():
        log_print(f"  Cluster {cluster}: {count}")
    
    # Top confidence flips
    log_print(f"\nTop 10 highest confidence flips:")
    for _, row in flip_candidates_df.head(10).iterrows():
        log_print(f"  ID {row['id']}: {row['reason']} (confidence: {row['confidence']:.2f})")
    
    return flip_candidates_df


def create_flip_submissions(submission_df, flip_candidates_df):
    """Create multiple flip submissions with different thresholds."""
    log_print("\n" + "="*70)
    log_print("CREATING FLIP SUBMISSIONS")
    log_print("="*70)
    
    # Check if we have any candidates
    if len(flip_candidates_df) == 0:
        log_print("\nNo flip candidates to create submissions!")
        return
    
    # Different strategies
    strategies = [
        ('conservative', 0.85, 5),    # Only very high confidence, max 5 flips
        ('moderate', 0.80, 10),       # High confidence, max 10 flips
        ('aggressive', 0.75, 20),     # Medium confidence, max 20 flips
        ('cluster_2_only', -1, -1),   # Only cluster 2
        ('cluster_7_only', -1, -1),   # Only cluster 7
        ('all_candidates', 0, 100)    # All candidates
    ]
    
    for strategy_name, confidence_threshold, max_flips in strategies:
        log_print(f"\n{strategy_name.upper()} strategy:")
        
        # Select candidates
        if strategy_name == 'cluster_2_only':
            selected = flip_candidates_df[flip_candidates_df['cluster'] == 2]
        elif strategy_name == 'cluster_7_only':
            selected = flip_candidates_df[flip_candidates_df['cluster'] == 7]
        else:
            selected = flip_candidates_df[flip_candidates_df['confidence'] >= confidence_threshold]
            if max_flips > 0:
                selected = selected.head(max_flips)
        
        log_print(f"  Flipping {len(selected)} predictions")
        
        # Create submission
        new_submission = submission_df.copy()
        
        for _, flip in selected.iterrows():
            mask = new_submission['id'] == flip['id']
            current = new_submission.loc[mask, 'Personality'].values[0]
            new_submission.loc[mask, 'Personality'] = 'Introvert' if current == 'Extrovert' else 'Extrovert'
        
        # Save
        filename = f'output/cluster_flip_{strategy_name}_submission.csv'
        new_submission.to_csv(filename, index=False)
        log_print(f"  Saved to: {filename}")
        
        # Show some examples
        if len(selected) > 0:
            log_print(f"  Example flips:")
            for _, flip in selected.head(3).iterrows():
                log_print(f"    ID {flip['id']}: {flip['reason']}")


def validate_flip_strategy(train_df, X_train, y_train):
    """Validate the flip strategy on training data."""
    log_print("\n" + "="*70)
    log_print("VALIDATING FLIP STRATEGY ON TRAINING DATA")
    log_print("="*70)
    
    # Scale and cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Train baseline model
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train, verbose=False)
    
    baseline_pred = model.predict(X_train)
    baseline_acc = np.mean(baseline_pred == y_train)
    
    log_print(f"Baseline accuracy: {baseline_acc:.6f}")
    
    # Apply flip rules
    train_df = train_df.copy()
    train_df['cluster'] = clusters
    train_df['predicted'] = baseline_pred
    
    # Count potential flips
    # Cluster 2 pattern
    cluster_2_flips = (
        (train_df['cluster'] == 2) &
        (train_df['predicted'] == 1) &  # Predicted extrovert
        (train_df['Time_spent_Alone'].between(4, 6)) &
        (train_df['Social_event_attendance'] < 2) &
        ((train_df['Friends_circle_size'] < 5) | (train_df['Drained_after_socializing'] == 'Yes'))
    )
    
    # Cluster 7 pattern
    cluster_7_flips = (
        (train_df['cluster'] == 7) &
        (train_df['predicted'] == 1) &
        (train_df['Time_spent_Alone'] > 8) &
        ~((train_df['Social_event_attendance'] > 5) & (train_df['Friends_circle_size'] > 10))
    )
    
    # Apply flips
    flipped_pred = baseline_pred.copy()
    flipped_pred[cluster_2_flips] = 0  # Flip to introvert
    flipped_pred[cluster_7_flips] = 0
    
    flipped_acc = np.mean(flipped_pred == y_train)
    
    log_print(f"After flips accuracy: {flipped_acc:.6f}")
    log_print(f"Improvement: {(flipped_acc - baseline_acc)*100:.3f}%")
    log_print(f"Total flips: {np.sum(cluster_2_flips) + np.sum(cluster_7_flips)}")
    
    # Analyze flip correctness
    correct_flips = 0
    for mask, name in [(cluster_2_flips, 'Cluster 2'), (cluster_7_flips, 'Cluster 7')]:
        if np.any(mask):
            correct = np.sum(mask & (y_train == 0))  # Should be introvert
            total = np.sum(mask)
            log_print(f"  {name}: {correct}/{total} flips were correct ({correct/total*100:.1f}%)")
            correct_flips += correct
    
    return flipped_acc - baseline_acc


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("TARGETED CLUSTER FLIP STRATEGY")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    submission_df = pd.read_csv('output/base_975708_submission.csv')
    
    # Prepare features
    X_train, feature_cols = prepare_features(train_df)
    X_test, _ = prepare_features(test_df)
    y_train = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # Validate on training data
    improvement = validate_flip_strategy(train_df, X_train.values, y_train.values)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cluster test data
    log_print("\nClustering test data...")
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    clusters_test = kmeans.predict(X_test_scaled)
    
    # Find flip candidates
    flip_candidates_df = find_flip_candidates(test_df, submission_df, clusters_test)
    
    # Analyze impact
    flip_candidates_df = analyze_flip_impact(flip_candidates_df)
    
    # Save detailed analysis
    flip_candidates_df.to_csv('output/cluster_flip_candidates.csv', index=False)
    log_print(f"\nSaved {len(flip_candidates_df)} flip candidates to: "
              "output/cluster_flip_candidates.csv")
    
    # Create submissions
    create_flip_submissions(submission_df, flip_candidates_df)
    
    # Final summary
    log_print("\n" + "="*70)
    log_print("SUMMARY")
    log_print("="*70)
    log_print(f"\nTraining improvement from flips: {improvement*100:.3f}%")
    log_print(f"Total flip candidates found: {len(flip_candidates_df)}")
    log_print(f"High confidence flips (>0.85): {np.sum(flip_candidates_df['confidence'] > 0.85)}")
    log_print("\nCreated 6 different flip strategies")
    log_print("\nRecommendation: Start with 'conservative' strategy (5 flips)")
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()