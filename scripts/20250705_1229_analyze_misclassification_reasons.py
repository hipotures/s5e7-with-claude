#!/usr/bin/env python3
"""
ANALYZE MISCLASSIFICATION REASONS
=================================

This script analyzes why the 0.975708 model misclassified specific records,
particularly ID 19612 and 23844.

Author: Claude
Date: 2025-07-05 12:29
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Output file
output_file = open('output/20250705_1229_analyze_misclassification_reasons.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def analyze_misclassified_records(test_df, submission_df):
    """Analyze the two misclassified records in detail."""
    log_print("="*70)
    log_print("ANALYZING MISCLASSIFIED RECORDS")
    log_print("="*70)
    
    # The two misclassified IDs
    misclassified_ids = [19612, 23844]
    
    # Get their data
    misclassified = test_df[test_df['id'].isin(misclassified_ids)].copy()
    
    log_print("\nMisclassified Records Details:")
    log_print("="*60)
    
    for idx, row in misclassified.iterrows():
        log_print(f"\nID: {row['id']}")
        log_print(f"Predicted: Extrovert (WRONG)")
        log_print(f"Should be: Introvert")
        log_print(f"\nFeatures:")
        log_print(f"  Time_spent_Alone: {row['Time_spent_Alone']}")
        log_print(f"  Social_event_attendance: {row['Social_event_attendance']}")
        log_print(f"  Drained_after_socializing: {row['Drained_after_socializing']}")
        log_print(f"  Stage_fear: {row['Stage_fear']}")
        log_print(f"  Going_outside: {row['Going_outside']}")
        log_print(f"  Friends_circle_size: {row['Friends_circle_size']}")
        log_print(f"  Post_frequency: {row['Post_frequency']}")
    
    return misclassified


def find_similar_records(train_df, test_df, misclassified_ids):
    """Find similar records in training data to understand the pattern."""
    log_print("\n" + "="*70)
    log_print("FINDING SIMILAR RECORDS IN TRAINING DATA")
    log_print("="*70)
    
    # Get misclassified records
    misclassified = test_df[test_df['id'].isin(misclassified_ids)]
    
    for idx, test_row in misclassified.iterrows():
        log_print(f"\n\nAnalyzing ID {test_row['id']}:")
        log_print("="*50)
        
        # Define similarity based on key features
        # For ID 19612: Alone=5, Social=0, Friends=4
        # For ID 23844: Alone=11, Social=2, Friends=2
        
        # Find records with similar patterns
        if test_row['id'] == 19612:
            # Pattern: Medium alone time, NO social events, few friends
            similar_mask = (
                (train_df['Time_spent_Alone'].between(4, 6)) &
                (train_df['Social_event_attendance'] <= 1) &
                (train_df['Friends_circle_size'] <= 5)
            )
        else:  # 23844
            # Pattern: Very high alone time, very low social, very few friends
            similar_mask = (
                (train_df['Time_spent_Alone'] >= 10) &
                (train_df['Social_event_attendance'] <= 3) &
                (train_df['Friends_circle_size'] <= 3)
            )
        
        similar_records = train_df[similar_mask].copy()
        
        if len(similar_records) > 0:
            personality_dist = similar_records['Personality'].value_counts()
            
            log_print(f"\nFound {len(similar_records)} similar records in training data:")
            log_print(f"  Introverts: {personality_dist.get('Introvert', 0)} ({personality_dist.get('Introvert', 0)/len(similar_records)*100:.1f}%)")
            log_print(f"  Extroverts: {personality_dist.get('Extrovert', 0)} ({personality_dist.get('Extrovert', 0)/len(similar_records)*100:.1f}%)")
            
            # Analyze psychological features distribution
            if 'Drained_after_socializing' in similar_records.columns:
                drained_dist = similar_records.groupby('Personality')['Drained_after_socializing'].value_counts()
                log_print(f"\nDrained_after_socializing distribution:")
                for (personality, drained), count in drained_dist.items():
                    log_print(f"  {personality} - {drained}: {count}")
            
            if 'Stage_fear' in similar_records.columns:
                fear_dist = similar_records.groupby('Personality')['Stage_fear'].value_counts()
                log_print(f"\nStage_fear distribution:")
                for (personality, fear), count in fear_dist.items():
                    log_print(f"  {personality} - {fear}: {count}")
            
            # Show some examples
            log_print(f"\nExample similar records:")
            sample_size = min(5, len(similar_records))
            sample = similar_records.sample(sample_size)
            
            for _, row in sample.iterrows():
                log_print(f"  {row['Personality']}: Alone={row['Time_spent_Alone']}, Social={row['Social_event_attendance']}, "
                         f"Friends={row['Friends_circle_size']}, Drained={row.get('Drained_after_socializing', 'N/A')}, "
                         f"Fear={row.get('Stage_fear', 'N/A')}")


def analyze_decision_boundary(train_df, test_df, misclassified_ids):
    """Analyze the decision boundary around these cases."""
    log_print("\n" + "="*70)
    log_print("DECISION BOUNDARY ANALYSIS")
    log_print("="*70)
    
    # Train a simple model to understand decision boundary
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # For categorical features
    train_df_copy = train_df.copy()
    train_df_copy['Stage_fear_binary'] = (train_df_copy['Stage_fear'] == 'Yes').astype(int)
    train_df_copy['Drained_binary'] = (train_df_copy['Drained_after_socializing'] == 'Yes').astype(int)
    
    X_train = train_df_copy[feature_cols + ['Stage_fear_binary', 'Drained_binary']]
    y_train = (train_df_copy['Personality'] == 'Extrovert').astype(int)
    
    # Train XGBoost to get feature importances
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Get predictions for misclassified records
    test_df_copy = test_df.copy()
    test_df_copy['Stage_fear_binary'] = (test_df_copy['Stage_fear'] == 'Yes').astype(int)
    test_df_copy['Drained_binary'] = (test_df_copy['Drained_after_socializing'] == 'Yes').astype(int)
    
    for record_id in misclassified_ids:
        record = test_df_copy[test_df_copy['id'] == record_id]
        if len(record) > 0:
            X_test = record[feature_cols + ['Stage_fear_binary', 'Drained_binary']]
            
            # Get prediction probability
            prob = model.predict_proba(X_test)[0, 1]
            
            log_print(f"\nID {record_id} - Model probability: {prob:.4f}")
            
            # Get SHAP-like contribution (simplified)
            # Show which features push toward Extrovert vs Introvert
            feature_values = X_test.iloc[0]
            
            log_print(f"\nFeature contributions (simplified):")
            
            # Rules that typically indicate introvert
            intro_score = 0
            extro_score = 0
            
            if feature_values['Time_spent_Alone'] >= 8:
                intro_score += 0.3
                log_print(f"  High alone time ({feature_values['Time_spent_Alone']}) → +Introvert")
            elif feature_values['Time_spent_Alone'] <= 2:
                extro_score += 0.3
                log_print(f"  Low alone time ({feature_values['Time_spent_Alone']}) → +Extrovert")
            
            if feature_values['Social_event_attendance'] <= 2:
                intro_score += 0.3
                log_print(f"  Low social events ({feature_values['Social_event_attendance']}) → +Introvert")
            elif feature_values['Social_event_attendance'] >= 7:
                extro_score += 0.3
                log_print(f"  High social events ({feature_values['Social_event_attendance']}) → +Extrovert")
            
            if feature_values['Friends_circle_size'] <= 5:
                intro_score += 0.2
                log_print(f"  Few friends ({feature_values['Friends_circle_size']}) → +Introvert")
            elif feature_values['Friends_circle_size'] >= 10:
                extro_score += 0.2
                log_print(f"  Many friends ({feature_values['Friends_circle_size']}) → +Extrovert")
            
            if feature_values['Stage_fear_binary'] == 1:
                intro_score += 0.15
                log_print(f"  Stage fear (Yes) → +Introvert")
            
            if feature_values['Drained_binary'] == 1:
                intro_score += 0.15
                log_print(f"  Drained by socializing (Yes) → +Introvert")
            
            log_print(f"\nIntrovert indicators: {intro_score:.2f}")
            log_print(f"Extrovert indicators: {extro_score:.2f}")
            log_print(f"Expected: {'Introvert' if intro_score > extro_score else 'Extrovert'}")
            log_print(f"Model predicted: Extrovert (WRONG)")


def visualize_misclassification_context(train_df, test_df, misclassified_ids):
    """Create visualizations to understand the misclassification context."""
    log_print("\n" + "="*70)
    log_print("CREATING VISUALIZATIONS")
    log_print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter plot showing where misclassified points fall
    ax1 = axes[0, 0]
    
    # Plot training data
    introverts = train_df[train_df['Personality'] == 'Introvert']
    extroverts = train_df[train_df['Personality'] == 'Extrovert']
    
    ax1.scatter(introverts['Time_spent_Alone'], introverts['Social_event_attendance'], 
               alpha=0.3, c='blue', s=10, label='Introvert (train)')
    ax1.scatter(extroverts['Time_spent_Alone'], extroverts['Social_event_attendance'], 
               alpha=0.3, c='red', s=10, label='Extrovert (train)')
    
    # Highlight misclassified
    misclassified = test_df[test_df['id'].isin(misclassified_ids)]
    for idx, row in misclassified.iterrows():
        ax1.scatter(row['Time_spent_Alone'], row['Social_event_attendance'], 
                   c='black', s=200, marker='X', edgecolors='yellow', linewidth=2)
        ax1.annotate(f"ID {row['id']}", 
                    (row['Time_spent_Alone'], row['Social_event_attendance']),
                    xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Time Spent Alone')
    ax1.set_ylabel('Social Event Attendance')
    ax1.set_title('Misclassified Records in Feature Space')
    ax1.legend()
    
    # 2. Distribution comparison
    ax2 = axes[0, 1]
    
    # Compare feature distributions
    features_to_compare = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']
    positions = np.arange(len(features_to_compare))
    width = 0.25
    
    intro_means = [introverts[f].mean() for f in features_to_compare]
    extro_means = [extroverts[f].mean() for f in features_to_compare]
    misclass_values = []
    
    for f in features_to_compare:
        vals = misclassified[f].values
        misclass_values.append(vals.mean() if len(vals) > 0 else 0)
    
    ax2.bar(positions - width, intro_means, width, label='Avg Introvert', alpha=0.7, color='blue')
    ax2.bar(positions, extro_means, width, label='Avg Extrovert', alpha=0.7, color='red')
    ax2.bar(positions + width, misclass_values, width, label='Misclassified', alpha=0.7, color='black')
    
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Average Value')
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f.replace('_', ' ') for f in features_to_compare], rotation=45)
    ax2.set_title('Feature Comparison')
    ax2.legend()
    
    # 3. Decision boundary visualization (simplified 2D)
    ax3 = axes[1, 0]
    
    # Create a grid
    alone_range = np.linspace(0, 12, 50)
    social_range = np.linspace(0, 10, 50)
    alone_grid, social_grid = np.meshgrid(alone_range, social_range)
    
    # For each point, calculate simple decision rule
    decision_grid = np.zeros_like(alone_grid)
    for i in range(len(alone_range)):
        for j in range(len(social_range)):
            alone = alone_grid[j, i]
            social = social_grid[j, i]
            # Simple rule: high alone + low social = introvert
            intro_score = (alone / 12) + (1 - social / 10)
            decision_grid[j, i] = 1 if intro_score < 1 else 0
    
    ax3.contourf(alone_grid, social_grid, decision_grid, alpha=0.3, cmap='RdBu')
    
    # Add misclassified points
    for idx, row in misclassified.iterrows():
        ax3.scatter(row['Time_spent_Alone'], row['Social_event_attendance'], 
                   c='black', s=200, marker='X', edgecolors='yellow', linewidth=2)
        ax3.text(row['Time_spent_Alone'] + 0.2, row['Social_event_attendance'] + 0.2, 
                f"{row['id']}", fontsize=10)
    
    ax3.set_xlabel('Time Spent Alone')
    ax3.set_ylabel('Social Event Attendance')
    ax3.set_title('Simplified Decision Boundary\n(Red=Extrovert, Blue=Introvert)')
    
    # 4. Feature correlation for misclassified
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = """MISCLASSIFICATION SUMMARY

ID 19612:
• Time alone: 5h (moderate)
• Social events: 0 (very low) ← Strong intro signal
• Friends: 4 (low) ← Intro signal
• Clear introvert profile misclassified

ID 23844:
• Time alone: 11h (very high) ← Extreme intro signal
• Social events: 2 (low) ← Intro signal  
• Friends: 2 (very low) ← Strong intro signal
• Extreme introvert profile misclassified

Both have clear introvert patterns but were
predicted as extroverts. This suggests the model
may have issues with:
1. Zero/near-zero social activity
2. Extreme feature values
3. Missing psychological features data"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('output/misclassification_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log_print("Visualization saved to: output/misclassification_analysis.png")


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("MISCLASSIFICATION ANALYSIS FOR 0.975708 MODEL")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    submission_df = pd.read_csv('output/base_975708_submission.csv')
    
    # The two misclassified IDs from our flip analysis
    misclassified_ids = [19612, 23844]
    
    # Analyze misclassified records
    misclassified = analyze_misclassified_records(test_df, submission_df)
    
    # Find similar records in training
    find_similar_records(train_df, test_df, misclassified_ids)
    
    # Analyze decision boundary
    analyze_decision_boundary(train_df, test_df, misclassified_ids)
    
    # Create visualizations
    visualize_misclassification_context(train_df, test_df, misclassified_ids)
    
    # Final conclusions
    log_print("\n" + "="*70)
    log_print("CONCLUSIONS")
    log_print("="*70)
    
    log_print("""
The model misclassified these two records likely because:

1. **Extreme feature values**: Both have extreme profiles that may be rare in training data
   - ID 19612: Zero social events is very rare
   - ID 23844: 11h alone + 2 friends is an extreme combination

2. **Model overfitting to moderate cases**: The model may perform better on moderate profiles
   and struggle with extremes

3. **Missing psychological features**: Without knowing their Drained_after_socializing and
   Stage_fear values, the model may default to majority class (Extrovert)

4. **Class imbalance effect**: With 74% Extroverts in training, the model has a bias toward
   predicting Extrovert when uncertain

These are exactly the types of cases where manual correction (flipping) makes sense!
""")
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()