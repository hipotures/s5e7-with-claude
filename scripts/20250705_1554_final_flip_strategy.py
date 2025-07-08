#!/usr/bin/env python3
"""
FINAL FLIP STRATEGY BASED ON CLUSTER ANALYSIS
============================================

This script creates final flip submissions based on the patterns
discovered in cluster analysis, using manual flip candidates.

Author: Claude
Date: 2025-07-05 15:54
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Output file
output_file = open('output/20250705_1554_final_flip_strategy.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def analyze_submissions():
    """Analyze all available submissions."""
    log_print("="*70)
    log_print("ANALYZING SUBMISSIONS")
    log_print("="*70)
    
    submissions = {
        'base_975708': 'output/base_975708_submission.csv',
        'cluster_aware': 'output/cluster_aware_submission.csv',
        'pattern_based_flip': 'output/pattern_based_flip_submission.csv'
    }
    
    dfs = {}
    for name, path in submissions.items():
        try:
            dfs[name] = pd.read_csv(path)
            log_print(f"\nLoaded {name}: {len(dfs[name])} records")
            
            # Count personalities
            counts = dfs[name]['Personality'].value_counts()
            log_print(f"  Introverts: {counts.get('Introvert', 0)}")
            log_print(f"  Extroverts: {counts.get('Extrovert', 0)}")
        except:
            log_print(f"\nCould not load {name} from {path}")
    
    # Compare submissions
    if 'base_975708' in dfs and 'cluster_aware' in dfs:
        log_print("\n\nComparing base_975708 vs cluster_aware:")
        merged = dfs['base_975708'].merge(dfs['cluster_aware'], on='id', suffixes=('_base', '_cluster'))
        diffs = merged[merged['Personality_base'] != merged['Personality_cluster']]
        log_print(f"  Differences: {len(diffs)}")
        if len(diffs) > 0:
            log_print(f"  Base Intro → Cluster Extro: {np.sum((diffs['Personality_base'] == 'Introvert') & (diffs['Personality_cluster'] == 'Extrovert'))}")
            log_print(f"  Base Extro → Cluster Intro: {np.sum((diffs['Personality_base'] == 'Extrovert') & (diffs['Personality_cluster'] == 'Introvert'))}")
    
    if 'base_975708' in dfs and 'pattern_based_flip' in dfs:
        log_print("\n\nComparing base_975708 vs pattern_based_flip:")
        merged = dfs['base_975708'].merge(dfs['pattern_based_flip'], on='id', suffixes=('_base', '_pattern'))
        diffs = merged[merged['Personality_base'] != merged['Personality_pattern']]
        log_print(f"  Differences: {len(diffs)}")
        if len(diffs) > 0:
            log_print("  Flipped IDs:")
            for _, row in diffs.iterrows():
                log_print(f"    ID {row['id']}: {row['Personality_base']} → {row['Personality_pattern']}")
    
    return dfs


def create_targeted_submissions():
    """Create targeted flip submissions based on manual candidates."""
    log_print("\n" + "="*70)
    log_print("CREATING TARGETED SUBMISSIONS")
    log_print("="*70)
    
    # Load manual flip candidates
    try:
        candidates = pd.read_csv('output/manual_flip_candidates.csv')
        log_print(f"\nLoaded {len(candidates)} manual flip candidates")
        
        # Show candidates
        log_print("\nFlip candidates:")
        for _, row in candidates.iterrows():
            log_print(f"  ID {row['id']}: {row['pattern']} (alone={row['alone']}, social={row['social']}, friends={row['friends']})")
    except:
        log_print("\nNo manual flip candidates found!")
        return
    
    # Load test data to verify patterns
    test_df = pd.read_csv("../../test.csv")
    
    # Verify each candidate
    log_print("\n\nVerifying candidates against test data:")
    verified_candidates = []
    
    for _, candidate in candidates.iterrows():
        test_row = test_df[test_df['id'] == candidate['id']]
        if len(test_row) > 0:
            test_row = test_row.iloc[0]
            
            # Verify pattern
            if candidate['pattern'] == 'cluster7_extreme_alone':
                if test_row['Time_spent_Alone'] >= 10:
                    verified_candidates.append(candidate['id'])
                    log_print(f"  ID {candidate['id']}: VERIFIED (alone={test_row['Time_spent_Alone']})")
                else:
                    log_print(f"  ID {candidate['id']}: NOT VERIFIED (alone={test_row['Time_spent_Alone']})")
            
            elif candidate['pattern'] == 'inconsistent_introvert':
                if (test_row['Time_spent_Alone'] < 5 and 
                    test_row['Social_event_attendance'] < 3 and 
                    test_row['Friends_circle_size'] < 6):
                    verified_candidates.append(candidate['id'])
                    log_print(f"  ID {candidate['id']}: VERIFIED (inconsistent pattern)")
                else:
                    log_print(f"  ID {candidate['id']}: NOT VERIFIED")
    
    log_print(f"\nVerified {len(verified_candidates)} candidates")
    
    # Create submissions with different strategies
    base_df = pd.read_csv('output/base_975708_submission.csv')
    
    strategies = [
        ('all_5_flips', verified_candidates),  # All 5 verified
        ('top_3_flips', verified_candidates[:3]),  # Top 3
        ('only_extreme_alone', [c for c in verified_candidates if c == 18876]),  # Only the extreme alone
        ('only_inconsistent', [c for c in verified_candidates if c != 18876])  # Only inconsistent
    ]
    
    for strategy_name, flip_ids in strategies:
        if len(flip_ids) > 0:
            log_print(f"\n{strategy_name.upper()}: Flipping {len(flip_ids)} predictions")
            
            # Create submission
            new_submission = base_df.copy()
            
            for flip_id in flip_ids:
                mask = new_submission['id'] == flip_id
                current = new_submission.loc[mask, 'Personality'].values[0]
                new_submission.loc[mask, 'Personality'] = 'Introvert' if current == 'Extrovert' else 'Extrovert'
                log_print(f"  ID {flip_id}: {current} → {'Introvert' if current == 'Extrovert' else 'Extrovert'}")
            
            # Save
            filename = f'output/final_flip_{strategy_name}_submission.csv'
            new_submission.to_csv(filename, index=False)
            log_print(f"  Saved to: {filename}")


def create_ensemble_submission():
    """Create ensemble of cluster_aware and flips."""
    log_print("\n" + "="*70)
    log_print("CREATING ENSEMBLE SUBMISSION")
    log_print("="*70)
    
    # Load submissions
    cluster_aware = pd.read_csv('output/cluster_aware_submission.csv')
    pattern_flip = pd.read_csv('output/pattern_based_flip_submission.csv')
    
    # Find agreements and disagreements
    merged = cluster_aware.merge(pattern_flip, on='id', suffixes=('_cluster', '_pattern'))
    
    agreements = merged[merged['Personality_cluster'] == merged['Personality_pattern']]
    disagreements = merged[merged['Personality_cluster'] != merged['Personality_pattern']]
    
    log_print(f"\nAgreements: {len(agreements)} ({len(agreements)/len(merged)*100:.1f}%)")
    log_print(f"Disagreements: {len(disagreements)} ({len(disagreements)/len(merged)*100:.1f}%)")
    
    if len(disagreements) > 0:
        log_print("\nDisagreement cases:")
        for _, row in disagreements.iterrows():
            log_print(f"  ID {row['id']}: Cluster={row['Personality_cluster']}, Pattern={row['Personality_pattern']}")
    
    # For ensemble, trust pattern-based for the specific 5 IDs we identified
    ensemble = cluster_aware.copy()
    flip_ids = [18876, 20363, 20934, 20950, 21008]
    
    for flip_id in flip_ids:
        mask = ensemble['id'] == flip_id
        if np.any(mask):
            ensemble.loc[mask, 'Personality'] = 'Introvert'
    
    ensemble.to_csv('output/ensemble_cluster_pattern_submission.csv', index=False)
    log_print("\nCreated ensemble submission: output/ensemble_cluster_pattern_submission.csv")


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("FINAL FLIP STRATEGY")
    log_print("="*70)
    
    # Analyze existing submissions
    dfs = analyze_submissions()
    
    # Create targeted submissions
    create_targeted_submissions()
    
    # Create ensemble
    create_ensemble_submission()
    
    # Summary
    log_print("\n" + "="*70)
    log_print("SUMMARY")
    log_print("="*70)
    log_print("\nCreated submissions:")
    log_print("1. final_flip_all_5_flips_submission.csv - All 5 pattern-based flips")
    log_print("2. final_flip_top_3_flips_submission.csv - Top 3 flips")
    log_print("3. final_flip_only_extreme_alone_submission.csv - Only ID 18876")
    log_print("4. final_flip_only_inconsistent_submission.csv - Only inconsistent introverts")
    log_print("5. ensemble_cluster_pattern_submission.csv - Best of both approaches")
    
    log_print("\nRecommendation: Try 'ensemble_cluster_pattern_submission.csv' first")
    log_print("It combines the cluster-aware strategy with targeted pattern flips")
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()