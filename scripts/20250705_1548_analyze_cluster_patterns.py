#!/usr/bin/env python3
"""
ANALYZE CLUSTER PATTERNS AND MISCLASSIFICATIONS
==============================================

This script analyzes patterns in clusters 2 and 7 to understand
why the model struggles with these cases.

Author: Claude
Date: 2025-07-05 15:48
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Output file
output_file = open('output/20250705_1548_analyze_cluster_patterns.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def analyze_cluster_patterns(test_df, submission_df):
    """Analyze patterns in problematic clusters."""
    log_print("="*70)
    log_print("ANALYZING CLUSTER PATTERNS")
    log_print("="*70)
    
    # Prepare features
    test_df = test_df.copy()
    test_df['Stage_fear_binary'] = (test_df['Stage_fear'] == 'Yes').astype(int)
    test_df['Drained_binary'] = (test_df['Drained_after_socializing'] == 'Yes').astype(int)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear_binary', 'Drained_binary']
    
    X_test = test_df[feature_cols].fillna(0)
    
    # Scale
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Cluster
    log_print("\nClustering test data...")
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_test_scaled)
    
    # Merge with predictions
    analysis_df = test_df.copy()
    analysis_df['cluster'] = clusters
    analysis_df = analysis_df.merge(submission_df, on='id')
    
    # Analyze each problematic cluster
    for cluster_id in [2, 7, 6]:
        log_print(f"\n\nCLUSTER {cluster_id} ANALYSIS:")
        log_print("-" * 50)
        
        cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]
        n_total = len(cluster_data)
        n_intro = np.sum(cluster_data['Personality'] == 'Introvert')
        n_extro = np.sum(cluster_data['Personality'] == 'Extrovert')
        
        log_print(f"Total samples: {n_total}")
        log_print(f"Predicted as Introvert: {n_intro} ({n_intro/n_total*100:.1f}%)")
        log_print(f"Predicted as Extrovert: {n_extro} ({n_extro/n_total*100:.1f}%)")
        
        # Feature statistics
        log_print(f"\nFeature averages:")
        for feature in ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']:
            avg = cluster_data[feature].mean()
            std = cluster_data[feature].std()
            log_print(f"  {feature}: {avg:.2f} (±{std:.2f})")
        
        # Look for specific patterns
        if cluster_id == 2:
            # Moderate alone time introverts
            pattern = (
                (cluster_data['Time_spent_Alone'].between(4, 6)) &
                (cluster_data['Social_event_attendance'] < 2) &
                (cluster_data['Personality'] == 'Extrovert')
            )
            log_print(f"\nModerate alone + low social predicted as Extrovert: {np.sum(pattern)}")
            
            if np.sum(pattern) > 0:
                log_print("Examples:")
                for _, row in cluster_data[pattern].head(5).iterrows():
                    log_print(f"  ID {row['id']}: Alone={row['Time_spent_Alone']}, "
                             f"Social={row['Social_event_attendance']}, "
                             f"Friends={row['Friends_circle_size']}")
        
        elif cluster_id == 7:
            # Extreme alone time
            pattern = (
                (cluster_data['Time_spent_Alone'] > 8) &
                (cluster_data['Personality'] == 'Extrovert')
            )
            log_print(f"\nExtreme alone time predicted as Extrovert: {np.sum(pattern)}")
            
            if np.sum(pattern) > 0:
                log_print("Examples:")
                for _, row in cluster_data[pattern].head(5).iterrows():
                    log_print(f"  ID {row['id']}: Alone={row['Time_spent_Alone']}, "
                             f"Social={row['Social_event_attendance']}, "
                             f"Friends={row['Friends_circle_size']}")
    
    # Look for introvert inconsistency pattern
    log_print("\n\nINTROVERT INCONSISTENCY PATTERN:")
    log_print("-" * 50)
    
    inconsistent = (
        (analysis_df['Time_spent_Alone'] < 5) &
        (analysis_df['Social_event_attendance'] < 3) &
        (analysis_df['Friends_circle_size'] < 6) &
        (analysis_df['Personality'] == 'Extrovert')
    )
    
    log_print(f"Found {np.sum(inconsistent)} cases with low everything but predicted as Extrovert")
    
    if np.sum(inconsistent) > 0:
        log_print("\nExamples:")
        for _, row in analysis_df[inconsistent].head(10).iterrows():
            log_print(f"  ID {row['id']} (Cluster {row['cluster']}): "
                     f"Alone={row['Time_spent_Alone']}, "
                     f"Social={row['Social_event_attendance']}, "
                     f"Friends={row['Friends_circle_size']}, "
                     f"Outside={row['Going_outside']}")
    
    return analysis_df


def create_visualization(analysis_df):
    """Create visualization of problematic patterns."""
    log_print("\nCreating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cluster distribution
    ax1 = axes[0, 0]
    cluster_counts = analysis_df['cluster'].value_counts().sort_index()
    personality_by_cluster = analysis_df.groupby(['cluster', 'Personality']).size().unstack(fill_value=0)
    
    personality_by_cluster.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Count')
    ax1.set_title('Personality Distribution by Cluster')
    ax1.legend(title='Predicted')
    
    # 2. Time alone vs Social for clusters 2 and 7
    ax2 = axes[0, 1]
    
    for cluster_id, color in [(2, 'red'), (7, 'blue'), (6, 'green')]:
        cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]
        
        introverts = cluster_data[cluster_data['Personality'] == 'Introvert']
        extroverts = cluster_data[cluster_data['Personality'] == 'Extrovert']
        
        ax2.scatter(introverts['Time_spent_Alone'], introverts['Social_event_attendance'],
                   alpha=0.5, s=30, color=color, marker='o', label=f'C{cluster_id} Intro')
        ax2.scatter(extroverts['Time_spent_Alone'], extroverts['Social_event_attendance'],
                   alpha=0.5, s=30, color=color, marker='x', label=f'C{cluster_id} Extro')
    
    ax2.set_xlabel('Time Spent Alone')
    ax2.set_ylabel('Social Event Attendance')
    ax2.set_title('Problematic Clusters: Time Alone vs Social')
    ax2.legend()
    
    # 3. Feature distributions for cluster 2
    ax3 = axes[1, 0]
    cluster_2 = analysis_df[analysis_df['cluster'] == 2]
    
    features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']
    positions = np.arange(len(features))
    width = 0.35
    
    intro_means = [cluster_2[cluster_2['Personality'] == 'Introvert'][f].mean() for f in features]
    extro_means = [cluster_2[cluster_2['Personality'] == 'Extrovert'][f].mean() for f in features]
    
    ax3.bar(positions - width/2, intro_means, width, label='Predicted Intro', alpha=0.7)
    ax3.bar(positions + width/2, extro_means, width, label='Predicted Extro', alpha=0.7)
    
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Average Value')
    ax3.set_xticks(positions)
    ax3.set_xticklabels([f.replace('_', ' ') for f in features], rotation=45)
    ax3.set_title('Cluster 2: Feature Averages by Prediction')
    ax3.legend()
    
    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Count patterns
    moderate_alone_extro = np.sum(
        (analysis_df['cluster'] == 2) &
        (analysis_df['Time_spent_Alone'].between(4, 6)) &
        (analysis_df['Social_event_attendance'] < 2) &
        (analysis_df['Personality'] == 'Extrovert')
    )
    
    extreme_alone_extro = np.sum(
        (analysis_df['cluster'] == 7) &
        (analysis_df['Time_spent_Alone'] > 8) &
        (analysis_df['Personality'] == 'Extrovert')
    )
    
    inconsistent_introverts = np.sum(
        (analysis_df['Time_spent_Alone'] < 5) &
        (analysis_df['Social_event_attendance'] < 3) &
        (analysis_df['Friends_circle_size'] < 6) &
        (analysis_df['Personality'] == 'Extrovert')
    )
    
    summary_text = f"""PATTERN SUMMARY

Cluster 2 (Moderate Alone):
• Total: {np.sum(analysis_df['cluster'] == 2)}
• Moderate alone + low social → Extro: {moderate_alone_extro}

Cluster 7 (Extreme Alone):
• Total: {np.sum(analysis_df['cluster'] == 7)}
• Extreme alone → Extro: {extreme_alone_extro}

Inconsistent Introverts:
• Low everything → Extro: {inconsistent_introverts}

Total potential flips: {moderate_alone_extro + extreme_alone_extro + inconsistent_introverts}

These are the cases that match the
patterns from training error analysis."""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('output/cluster_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log_print("Saved visualization to: output/cluster_pattern_analysis.png")


def compare_submissions():
    """Compare base and cluster-aware submissions."""
    log_print("\n" + "="*70)
    log_print("COMPARING SUBMISSIONS")
    log_print("="*70)
    
    base_df = pd.read_csv('output/base_975708_submission.csv')
    cluster_df = pd.read_csv('output/cluster_aware_submission.csv')
    
    # Find differences
    merged = base_df.merge(cluster_df, on='id', suffixes=('_base', '_cluster'))
    differences = merged[merged['Personality_base'] != merged['Personality_cluster']]
    
    log_print(f"\nFound {len(differences)} differences between submissions")
    
    if len(differences) > 0:
        log_print("\nFlip summary:")
        log_print(f"Base Intro → Cluster Extro: {np.sum((differences['Personality_base'] == 'Introvert') & (differences['Personality_cluster'] == 'Extrovert'))}")
        log_print(f"Base Extro → Cluster Intro: {np.sum((differences['Personality_base'] == 'Extrovert') & (differences['Personality_cluster'] == 'Introvert'))}")
        
        log_print("\nFirst 10 differences:")
        for _, row in differences.head(10).iterrows():
            log_print(f"  ID {row['id']}: {row['Personality_base']} → {row['Personality_cluster']}")
    
    return differences


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("CLUSTER PATTERN ANALYSIS")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    test_df = pd.read_csv("../../test.csv")
    submission_df = pd.read_csv('output/base_975708_submission.csv')
    
    # Analyze patterns
    analysis_df = analyze_cluster_patterns(test_df, submission_df)
    
    # Create visualization
    create_visualization(analysis_df)
    
    # Compare submissions
    differences = compare_submissions()
    
    # Save potential flip candidates based on patterns
    log_print("\n" + "="*70)
    log_print("CREATING MANUAL FLIP LIST")
    log_print("="*70)
    
    # Find all cases matching our patterns
    flip_candidates = []
    
    # Pattern 1: Cluster 2, moderate alone + low social
    pattern1 = (
        (analysis_df['cluster'] == 2) &
        (analysis_df['Time_spent_Alone'].between(4, 6)) &
        (analysis_df['Social_event_attendance'] < 2) &
        (analysis_df['Personality'] == 'Extrovert')
    )
    
    for _, row in analysis_df[pattern1].iterrows():
        flip_candidates.append({
            'id': row['id'],
            'pattern': 'cluster2_moderate_alone',
            'current': 'Extrovert',
            'should_be': 'Introvert',
            'alone': row['Time_spent_Alone'],
            'social': row['Social_event_attendance'],
            'friends': row['Friends_circle_size']
        })
    
    # Pattern 2: Cluster 7, extreme alone
    pattern2 = (
        (analysis_df['cluster'] == 7) &
        (analysis_df['Time_spent_Alone'] > 8) &
        (analysis_df['Personality'] == 'Extrovert')
    )
    
    for _, row in analysis_df[pattern2].iterrows():
        flip_candidates.append({
            'id': row['id'],
            'pattern': 'cluster7_extreme_alone',
            'current': 'Extrovert',
            'should_be': 'Introvert',
            'alone': row['Time_spent_Alone'],
            'social': row['Social_event_attendance'],
            'friends': row['Friends_circle_size']
        })
    
    # Pattern 3: Inconsistent introverts
    pattern3 = (
        (analysis_df['Time_spent_Alone'] < 5) &
        (analysis_df['Social_event_attendance'] < 3) &
        (analysis_df['Friends_circle_size'] < 6) &
        (analysis_df['Personality'] == 'Extrovert')
    )
    
    for _, row in analysis_df[pattern3].iterrows():
        flip_candidates.append({
            'id': row['id'],
            'pattern': 'inconsistent_introvert',
            'current': 'Extrovert',
            'should_be': 'Introvert',
            'alone': row['Time_spent_Alone'],
            'social': row['Social_event_attendance'],
            'friends': row['Friends_circle_size']
        })
    
    # Save candidates
    if flip_candidates:
        flip_df = pd.DataFrame(flip_candidates)
        flip_df.to_csv('output/manual_flip_candidates.csv', index=False)
        log_print(f"\nSaved {len(flip_df)} flip candidates to: output/manual_flip_candidates.csv")
        
        # Create flip submission
        flip_submission = submission_df.copy()
        for _, candidate in flip_df.iterrows():
            mask = flip_submission['id'] == candidate['id']
            flip_submission.loc[mask, 'Personality'] = candidate['should_be']
        
        flip_submission.to_csv('output/pattern_based_flip_submission.csv', index=False)
        log_print("Created pattern-based flip submission: output/pattern_based_flip_submission.csv")
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()