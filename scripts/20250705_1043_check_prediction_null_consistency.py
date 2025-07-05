#!/usr/bin/env python3
"""
CHECK PREDICTION CONSISTENCY WITH NULL PATTERNS
==============================================

This script analyzes whether our predictions align with the discovered null patterns:
- Introverts have 2-4x more missing values
- Especially in psychological features (Drained_after_socializing, Stage_fear)

We'll check if the model correctly predicts more introverts for records with many nulls.

Author: Claude
Date: 2025-07-05 10:43
"""

# PURPOSE: Verify if model predictions align with null pattern discovery
# HYPOTHESIS: If nulls encode personality, predictions should show more introverts among high-null records
# EXPECTED: Records with many nulls (especially psychological) should be predicted as introverts more often
# RESULT: [To be determined after execution]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Output file
output_file = open('output/20250705_1043_check_prediction_null_consistency.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def analyze_null_prediction_consistency():
    """Main analysis function."""
    log_print("="*70)
    log_print("NULL PATTERN - PREDICTION CONSISTENCY ANALYSIS")
    log_print("="*70)
    
    # Load test data to analyze null patterns
    log_print("\nLoading test data...")
    test_df = pd.read_csv("../../test.csv")
    log_print(f"Test shape: {test_df.shape}")
    
    # Load a recent submission to analyze predictions
    # Try to find the most recent submission
    import os
    import glob
    
    # Look for recent submissions
    submission_patterns = [
        "output/submission_return_to_ambivert.csv",
        "../subm/DATE_*/submission_*.csv",
        "output/submission_*.csv"
    ]
    
    submission_file = None
    for pattern in submission_patterns:
        files = glob.glob(pattern)
        if files:
            # Use the most recent one
            submission_file = sorted(files)[-1]
            break
    
    if not submission_file:
        log_print("ERROR: No submission file found!")
        return
    
    log_print(f"Using submission: {submission_file}")
    submission_df = pd.read_csv(submission_file)
    
    # Map predictions to binary (0=Introvert, 1=Extrovert)
    if 'Personality' in submission_df.columns:
        submission_df['prediction'] = (submission_df['Personality'] == 'Extrovert').astype(int)
    elif 'Target' in submission_df.columns:
        submission_df['prediction'] = submission_df['Target']
    else:
        log_print("ERROR: Cannot find prediction column!")
        return
    
    # Merge predictions with test data
    test_with_pred = test_df.merge(submission_df[['id', 'prediction']], on='id')
    
    # Analyze null patterns in test data
    log_print("\n" + "="*60)
    log_print("NULL PATTERNS IN TEST DATA")
    log_print("="*60)
    
    # Count nulls per record
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    test_with_pred['null_count'] = test_with_pred[feature_cols].isnull().sum(axis=1)
    
    # Specific null indicators
    test_with_pred['has_drained_null'] = test_with_pred['Drained_after_socializing'].isnull()
    test_with_pred['has_stage_null'] = test_with_pred['Stage_fear'].isnull()
    test_with_pred['has_psych_null'] = test_with_pred['has_drained_null'] | test_with_pred['has_stage_null']
    
    # Overall statistics
    log_print(f"\nTest data null statistics:")
    log_print(f"Records with no nulls: {(test_with_pred['null_count'] == 0).sum()} ({(test_with_pred['null_count'] == 0).mean():.1%})")
    log_print(f"Records with 1+ nulls: {(test_with_pred['null_count'] > 0).sum()} ({(test_with_pred['null_count'] > 0).mean():.1%})")
    log_print(f"Average null count: {test_with_pred['null_count'].mean():.2f}")
    
    # Null distribution by predicted personality
    log_print("\n" + "="*60)
    log_print("PREDICTION ANALYSIS BY NULL COUNT")
    log_print("="*60)
    
    # Group by null count and analyze predictions
    null_analysis = test_with_pred.groupby('null_count').agg({
        'prediction': ['count', 'mean', 'sum']
    }).round(3)
    
    null_analysis.columns = ['total_records', 'extrovert_rate', 'extrovert_count']
    null_analysis['introvert_count'] = null_analysis['total_records'] - null_analysis['extrovert_count']
    null_analysis['introvert_rate'] = 1 - null_analysis['extrovert_rate']
    
    log_print("\nPredictions by null count:")
    log_print(f"{'Null Count':<12} {'Records':<10} {'Intro Rate':<12} {'Extro Rate':<12}")
    log_print("-" * 50)
    
    for null_count, row in null_analysis.iterrows():
        if row['total_records'] > 10:  # Only show if enough samples
            log_print(f"{null_count:<12} {row['total_records']:<10} {row['introvert_rate']:<12.1%} {row['extrovert_rate']:<12.1%}")
    
    # Analysis of psychological nulls
    log_print("\n" + "="*60)
    log_print("PSYCHOLOGICAL FEATURES NULL ANALYSIS")
    log_print("="*60)
    
    # Predictions for records with psychological nulls
    psych_null_pred = test_with_pred.groupby('has_psych_null')['prediction'].agg(['count', 'mean'])
    
    log_print("\nPredictions by psychological null presence:")
    log_print(f"No psych nulls: {psych_null_pred.loc[False, 'count']} records, "
             f"{(1 - psych_null_pred.loc[False, 'mean']):.1%} predicted as Introvert")
    log_print(f"Has psych nulls: {psych_null_pred.loc[True, 'count']} records, "
             f"{(1 - psych_null_pred.loc[True, 'mean']):.1%} predicted as Introvert")
    
    # Detailed analysis for specific nulls
    log_print("\nDetailed null analysis:")
    for col in ['Drained_after_socializing', 'Stage_fear']:
        has_null = test_with_pred[col].isnull()
        null_pred = test_with_pred[has_null]['prediction'].mean()
        no_null_pred = test_with_pred[~has_null]['prediction'].mean()
        
        log_print(f"\n{col}:")
        log_print(f"  With null: {has_null.sum()} records, {(1 - null_pred):.1%} predicted as Introvert")
        log_print(f"  No null: {(~has_null).sum()} records, {(1 - no_null_pred):.1%} predicted as Introvert")
        log_print(f"  Ratio: {(1 - null_pred) / (1 - no_null_pred):.2f}x more likely to be Introvert with null")
    
    # Expected vs Actual patterns
    log_print("\n" + "="*60)
    log_print("CONSISTENCY CHECK: EXPECTED VS ACTUAL")
    log_print("="*60)
    
    # Based on training data, we expect:
    # - Records with Drained null: ~4.2x more likely to be Introvert
    # - Records with Stage_fear null: ~1.8x more likely to be Introvert
    
    log_print("\nExpected patterns (from training data):")
    log_print("- Drained_after_socializing null: 4.2x more likely to be Introvert")
    log_print("- Stage_fear null: 1.8x more likely to be Introvert")
    log_print("- Overall: Introverts should have 2-4x more nulls")
    
    # Check if predictions follow this pattern
    log_print("\nActual prediction patterns:")
    
    # Compare high null vs low null predictions
    high_null = test_with_pred['null_count'] >= 2
    low_null = test_with_pred['null_count'] == 0
    
    high_null_intro_rate = 1 - test_with_pred[high_null]['prediction'].mean()
    low_null_intro_rate = 1 - test_with_pred[low_null]['prediction'].mean()
    
    log_print(f"\nHigh null (2+): {high_null_intro_rate:.1%} predicted as Introvert")
    log_print(f"No nulls: {low_null_intro_rate:.1%} predicted as Introvert")
    log_print(f"Ratio: {high_null_intro_rate / (low_null_intro_rate + 0.001):.2f}x")
    
    # Visualization
    log_print("\n" + "="*60)
    log_print("CREATING VISUALIZATIONS")
    log_print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction rate by null count
    ax1 = axes[0, 0]
    null_counts = sorted(test_with_pred['null_count'].unique())
    intro_rates = []
    for nc in null_counts:
        mask = test_with_pred['null_count'] == nc
        if mask.sum() > 10:
            intro_rates.append(1 - test_with_pred[mask]['prediction'].mean())
        else:
            intro_rates.append(None)
    
    ax1.plot(null_counts, intro_rates, 'o-', markersize=8, linewidth=2)
    ax1.set_xlabel('Number of Null Values')
    ax1.set_ylabel('Introvert Prediction Rate')
    ax1.set_title('Prediction Pattern by Null Count')
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of null counts by predicted personality
    ax2 = axes[0, 1]
    intro_nulls = test_with_pred[test_with_pred['prediction'] == 0]['null_count']
    extro_nulls = test_with_pred[test_with_pred['prediction'] == 1]['null_count']
    
    ax2.hist([intro_nulls, extro_nulls], bins=8, label=['Predicted Introvert', 'Predicted Extrovert'], 
             alpha=0.7, density=True)
    ax2.set_xlabel('Number of Null Values')
    ax2.set_ylabel('Density')
    ax2.set_title('Null Count Distribution by Predicted Personality')
    ax2.legend()
    
    # 3. Psychological nulls impact
    ax3 = axes[1, 0]
    psych_categories = ['No Psych Nulls', 'Has Drained Null', 'Has Stage Null', 'Has Both']
    
    no_psych = ~test_with_pred['has_drained_null'] & ~test_with_pred['has_stage_null']
    only_drained = test_with_pred['has_drained_null'] & ~test_with_pred['has_stage_null']
    only_stage = ~test_with_pred['has_drained_null'] & test_with_pred['has_stage_null']
    both_psych = test_with_pred['has_drained_null'] & test_with_pred['has_stage_null']
    
    intro_rates_psych = [
        1 - test_with_pred[no_psych]['prediction'].mean(),
        1 - test_with_pred[only_drained]['prediction'].mean() if only_drained.sum() > 0 else 0,
        1 - test_with_pred[only_stage]['prediction'].mean() if only_stage.sum() > 0 else 0,
        1 - test_with_pred[both_psych]['prediction'].mean() if both_psych.sum() > 0 else 0
    ]
    
    bars = ax3.bar(psych_categories, intro_rates_psych)
    ax3.set_ylabel('Introvert Prediction Rate')
    ax3.set_title('Impact of Psychological Feature Nulls')
    ax3.set_xticklabels(psych_categories, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, rate in zip(bars, intro_rates_psych):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # 4. Consistency score
    ax4 = axes[1, 1]
    
    # Calculate consistency scores
    consistency_scores = []
    labels = []
    
    # For each feature, check if null → introvert prediction pattern holds
    for col in feature_cols:
        has_null = test_with_pred[col].isnull()
        if has_null.sum() > 10:
            null_intro_rate = 1 - test_with_pred[has_null]['prediction'].mean()
            no_null_intro_rate = 1 - test_with_pred[~has_null]['prediction'].mean()
            consistency = null_intro_rate - no_null_intro_rate
            consistency_scores.append(consistency)
            labels.append(col.replace('_', ' '))
    
    # Sort by consistency
    sorted_idx = np.argsort(consistency_scores)[::-1]
    consistency_scores = [consistency_scores[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]
    
    bars = ax4.barh(labels, consistency_scores)
    ax4.set_xlabel('Consistency Score (Null→Introvert Bias)')
    ax4.set_title('Prediction Consistency with Null Pattern')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Color bars
    for bar, score in zip(bars, consistency_scores):
        if score > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig('output/null_prediction_consistency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log_print("\nVisualization saved to: output/null_prediction_consistency_analysis.png")
    
    # Summary and recommendations
    log_print("\n" + "="*60)
    log_print("SUMMARY AND RECOMMENDATIONS")
    log_print("="*60)
    
    # Calculate overall consistency
    overall_consistency = high_null_intro_rate > low_null_intro_rate
    
    if overall_consistency:
        log_print("\n✓ CONSISTENT: Model predictions align with null patterns!")
        log_print(f"  - High null records are {high_null_intro_rate / (low_null_intro_rate + 0.001):.2f}x "
                 f"more likely to be predicted as Introvert")
    else:
        log_print("\n✗ INCONSISTENT: Model predictions do NOT align with null patterns!")
        log_print("  - This suggests the model may not be fully utilizing the null pattern information")
    
    # Specific recommendations
    log_print("\nRecommendations:")
    
    if high_null_intro_rate / (low_null_intro_rate + 0.001) < 2.0:
        log_print("1. Model is under-utilizing null patterns - consider:")
        log_print("   - Adding stronger null indicator features")
        log_print("   - Using null-aware ensemble weights")
        log_print("   - Post-processing predictions based on null count")
    
    # Check specific psychological nulls
    drained_null_mask = test_with_pred['Drained_after_socializing'].isnull()
    if drained_null_mask.sum() > 0:
        drained_consistency = (1 - test_with_pred[drained_null_mask]['prediction'].mean()) / \
                            (1 - test_with_pred[~drained_null_mask]['prediction'].mean() + 0.001)
        
        if drained_consistency < 3.0:
            log_print("2. Drained_after_socializing null pattern is under-utilized")
            log_print(f"   - Expected: 4.2x more introverts, Actual: {drained_consistency:.2f}x")
            log_print("   - Consider special handling for this critical null")
    
    # Potential for improvement
    log_print("\n3. Potential accuracy improvement:")
    
    # Estimate misclassifications that could be fixed
    high_null_extro = test_with_pred[high_null & (test_with_pred['prediction'] == 1)]
    potential_fixes = len(high_null_extro)
    potential_improvement = potential_fixes / len(test_with_pred) * 100
    
    log_print(f"   - {potential_fixes} high-null records predicted as Extrovert")
    log_print(f"   - Potential improvement: up to {potential_improvement:.2f}%")
    
    # Save detailed results
    detailed_results = {
        'null_count_analysis': null_analysis.to_dict(),
        'psychological_null_impact': {
            'no_psych_nulls_intro_rate': float(intro_rates_psych[0]),
            'has_drained_null_intro_rate': float(intro_rates_psych[1]),
            'has_stage_null_intro_rate': float(intro_rates_psych[2]),
            'has_both_intro_rate': float(intro_rates_psych[3])
        },
        'consistency_metrics': {
            'high_null_intro_rate': float(high_null_intro_rate),
            'low_null_intro_rate': float(low_null_intro_rate),
            'ratio': float(high_null_intro_rate / (low_null_intro_rate + 0.001)),
            'is_consistent': bool(overall_consistency)
        }
    }
    
    import json
    with open('output/null_prediction_consistency_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    log_print("\nDetailed results saved to: output/null_prediction_consistency_results.json")


def main():
    """Main execution."""
    analyze_null_prediction_consistency()
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()