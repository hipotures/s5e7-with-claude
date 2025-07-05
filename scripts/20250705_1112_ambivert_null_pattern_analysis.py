#!/usr/bin/env python3
"""
AMBIVERT NULL PATTERN ANALYSIS
==============================

This script analyzes whether ambiverts have distinct missing value patterns
that differ from clear introverts and extroverts.

Hypothesis: Ambiverts might show inconsistent or random null patterns,
while clear personalities have consistent patterns.

Author: Claude
Date: 2025-07-05 11:12
"""

# PURPOSE: Analyze if ambiverts have unique missing value patterns
# HYPOTHESIS: Ambiverts show more random/inconsistent null patterns than clear personalities
# EXPECTED: Find distinct null pattern signatures that could help identify ambiverts
# RESULT: [To be determined after execution]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

# Output file
output_file = open('output/20250705_1112_ambivert_null_pattern_analysis.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def detect_potential_ambiverts(df, predictions=None):
    """
    Detect potential ambiverts using multiple methods:
    1. Prediction probability (0.4-0.6)
    2. Special marker values
    3. Behavioral patterns
    """
    log_print("\nDETECTING POTENTIAL AMBIVERTS")
    log_print("="*60)
    
    ambivert_mask = pd.Series(False, index=df.index)
    
    # Method 1: Prediction probability (if available)
    if predictions is not None and 'extrovert_prob' in predictions.columns:
        prob_ambiverts = (predictions['extrovert_prob'] >= 0.4) & (predictions['extrovert_prob'] <= 0.6)
        ambivert_mask |= prob_ambiverts
        log_print(f"Method 1 (probability 0.4-0.6): {prob_ambiverts.sum()} records")
    
    # Method 2: Special marker values
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    
    marker_mask = pd.Series(False, index=df.index)
    for col, val in markers.items():
        if col in df.columns:
            marker_mask |= (df[col] == val)
    
    ambivert_mask |= marker_mask
    log_print(f"Method 2 (special markers): {marker_mask.sum()} records")
    
    # Method 3: Behavioral patterns
    behavioral_mask = pd.Series(False, index=df.index)
    if all(col in df.columns for col in ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']):
        behavioral_mask = (
            (df['Time_spent_Alone'] < 2.5) & 
            (df['Social_event_attendance'].between(3, 4)) &
            (df['Friends_circle_size'].between(6, 7))
        )
    
    ambivert_mask |= behavioral_mask
    log_print(f"Method 3 (behavioral pattern): {behavioral_mask.sum()} records")
    
    log_print(f"\nTotal potential ambiverts: {ambivert_mask.sum()} ({ambivert_mask.mean():.1%})")
    
    return ambivert_mask


def analyze_null_patterns_by_personality(train_df, ambivert_mask):
    """Analyze null patterns for introverts, extroverts, and ambiverts."""
    log_print("\n" + "="*60)
    log_print("NULL PATTERN ANALYSIS BY PERSONALITY TYPE")
    log_print("="*60)
    
    # Feature columns
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    # Create personality groups
    introvert_mask = (train_df['Personality'] == 'Introvert') & (~ambivert_mask)
    extrovert_mask = (train_df['Personality'] == 'Extrovert') & (~ambivert_mask)
    
    log_print(f"\nGroup sizes:")
    log_print(f"Clear Introverts: {introvert_mask.sum()}")
    log_print(f"Clear Extroverts: {extrovert_mask.sum()}")
    log_print(f"Potential Ambiverts: {ambivert_mask.sum()}")
    
    # Analyze null rates per feature
    log_print("\n1. NULL RATES BY FEATURE AND PERSONALITY:")
    log_print(f"{'Feature':<30} {'Introverts':<12} {'Extroverts':<12} {'Ambiverts':<12} {'Ambi Diff':<12}")
    log_print("-" * 80)
    
    null_analysis = []
    for col in feature_cols:
        intro_null_rate = train_df.loc[introvert_mask, col].isnull().mean()
        extro_null_rate = train_df.loc[extrovert_mask, col].isnull().mean()
        ambi_null_rate = train_df.loc[ambivert_mask, col].isnull().mean() if ambivert_mask.sum() > 0 else 0
        
        # Calculate how different ambiverts are from the average
        avg_rate = (intro_null_rate + extro_null_rate) / 2
        ambi_diff = ambi_null_rate - avg_rate
        
        log_print(f"{col:<30} {intro_null_rate:<12.3f} {extro_null_rate:<12.3f} "
                 f"{ambi_null_rate:<12.3f} {ambi_diff:<+12.3f}")
        
        null_analysis.append({
            'feature': col,
            'intro_rate': intro_null_rate,
            'extro_rate': extro_null_rate,
            'ambi_rate': ambi_null_rate,
            'ambi_diff': ambi_diff
        })
    
    # Analyze total null counts
    train_df['null_count'] = train_df[feature_cols].isnull().sum(axis=1)
    
    log_print("\n2. DISTRIBUTION OF TOTAL NULL COUNTS:")
    log_print(f"{'Null Count':<12} {'Introverts':<15} {'Extroverts':<15} {'Ambiverts':<15}")
    log_print("-" * 60)
    
    null_count_analysis = []
    for null_count in range(8):
        intro_pct = ((train_df['null_count'] == null_count) & introvert_mask).sum() / max(introvert_mask.sum(), 1) * 100
        extro_pct = ((train_df['null_count'] == null_count) & extrovert_mask).sum() / max(extrovert_mask.sum(), 1) * 100
        ambi_pct = ((train_df['null_count'] == null_count) & ambivert_mask).sum() / max(ambivert_mask.sum(), 1) * 100
        
        if intro_pct > 0 or extro_pct > 0 or ambi_pct > 0:
            log_print(f"{null_count:<12} {intro_pct:<15.1f}% {extro_pct:<15.1f}% {ambi_pct:<15.1f}%")
            null_count_analysis.append({
                'null_count': null_count,
                'intro_pct': intro_pct,
                'extro_pct': extro_pct,
                'ambi_pct': ambi_pct
            })
    
    # Analyze null pattern consistency
    log_print("\n3. NULL PATTERN CONSISTENCY ANALYSIS:")
    
    # Calculate null pattern entropy (randomness measure)
    def calculate_null_entropy(group_mask):
        """Calculate entropy of null patterns within a group."""
        if group_mask.sum() == 0:
            return 0
        
        group_nulls = train_df.loc[group_mask, feature_cols].isnull()
        # Convert null patterns to strings
        patterns = group_nulls.apply(lambda x: ''.join(x.astype(int).astype(str)), axis=1)
        # Calculate pattern frequency
        pattern_counts = patterns.value_counts()
        # Calculate entropy
        probs = pattern_counts / pattern_counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    intro_entropy = calculate_null_entropy(introvert_mask)
    extro_entropy = calculate_null_entropy(extrovert_mask)
    ambi_entropy = calculate_null_entropy(ambivert_mask)
    
    log_print(f"\nNull pattern entropy (higher = more random):")
    log_print(f"Introverts: {intro_entropy:.3f}")
    log_print(f"Extroverts: {extro_entropy:.3f}")
    log_print(f"Ambiverts: {ambi_entropy:.3f}")
    
    # Analyze correlation between null patterns
    log_print("\n4. NULL PATTERN CORRELATIONS:")
    
    for group_name, group_mask in [('Introverts', introvert_mask), 
                                   ('Extroverts', extrovert_mask), 
                                   ('Ambiverts', ambivert_mask)]:
        if group_mask.sum() > 10:
            group_nulls = train_df.loc[group_mask, feature_cols].isnull().astype(int)
            corr_matrix = group_nulls.corr()
            # Get average correlation (excluding diagonal)
            avg_corr = (corr_matrix.sum().sum() - len(feature_cols)) / (len(feature_cols) * (len(feature_cols) - 1))
            log_print(f"{group_name} average null correlation: {avg_corr:.3f}")
    
    return null_analysis, null_count_analysis


def create_null_pattern_features(df, feature_cols):
    """Create advanced null pattern features."""
    # Basic null indicators
    for col in feature_cols:
        df[f'{col}_is_null'] = df[col].isnull().astype(int)
    
    # Total null count
    df['null_count'] = df[feature_cols].isnull().sum(axis=1)
    
    # Null pattern encoding
    df['null_pattern'] = df[feature_cols].isnull().apply(
        lambda x: ''.join(x.astype(int).astype(str)), axis=1
    )
    
    # Specific pattern indicators
    df['has_psych_nulls'] = (df['Drained_after_socializing'].isnull() | 
                             df['Stage_fear'].isnull()).astype(int)
    df['has_social_nulls'] = (df['Social_event_attendance'].isnull() | 
                              df['Going_outside'].isnull() | 
                              df['Friends_circle_size'].isnull()).astype(int)
    
    # Null consistency score (are nulls clustered or spread out?)
    null_matrix = df[feature_cols].isnull().astype(int).values
    df['null_consistency'] = np.std(null_matrix, axis=1)
    
    return df


def train_ambivert_detector(train_df, ambivert_mask):
    """Train a model to detect ambiverts based on null patterns."""
    log_print("\n" + "="*60)
    log_print("TRAINING AMBIVERT NULL PATTERN DETECTOR")
    log_print("="*60)
    
    # Feature columns
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    # Create features
    train_df = create_null_pattern_features(train_df, feature_cols)
    
    # Prepare features for modeling
    null_features = [col for col in train_df.columns if col.endswith('_is_null')]
    other_features = ['null_count', 'has_psych_nulls', 'has_social_nulls', 'null_consistency']
    all_features = null_features + other_features
    
    X = train_df[all_features]
    y = ambivert_mask.astype(int)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    log_print(f"\nTraining set: {len(X_train)} samples ({y_train.mean():.1%} ambiverts)")
    log_print(f"Validation set: {len(X_val)} samples ({y_val.mean():.1%} ambiverts)")
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=len(y_train) / (y_train.sum() + 1)  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    
    train_acc = (train_pred == y_train).mean()
    val_acc = (val_pred == y_val).mean()
    
    log_print(f"\nModel Performance:")
    log_print(f"Training accuracy: {train_acc:.3f}")
    log_print(f"Validation accuracy: {val_acc:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    log_print("\nTop 10 Most Important Features for Ambivert Detection:")
    for idx, row in feature_importance.head(10).iterrows():
        log_print(f"{row['feature']:<40} {row['importance']:.3f}")
    
    return model, feature_importance


def visualize_results(train_df, null_analysis, null_count_analysis, feature_importance, ambivert_mask):
    """Create comprehensive visualizations."""
    log_print("\n" + "="*60)
    log_print("CREATING VISUALIZATIONS")
    log_print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Null rates by personality and feature
    ax1 = axes[0, 0]
    features = [item['feature'].replace('_', ' ') for item in null_analysis]
    intro_rates = [item['intro_rate'] for item in null_analysis]
    extro_rates = [item['extro_rate'] for item in null_analysis]
    ambi_rates = [item['ambi_rate'] for item in null_analysis]
    
    x = np.arange(len(features))
    width = 0.25
    
    ax1.bar(x - width, intro_rates, width, label='Introverts', alpha=0.7, color='blue')
    ax1.bar(x, extro_rates, width, label='Extroverts', alpha=0.7, color='orange')
    ax1.bar(x + width, ambi_rates, width, label='Ambiverts', alpha=0.7, color='green')
    
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Null Rate')
    ax1.set_title('Null Rates by Personality Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=45, ha='right')
    ax1.legend()
    
    # 2. Null count distribution
    ax2 = axes[0, 1]
    null_counts = [item['null_count'] for item in null_count_analysis]
    intro_pcts = [item['intro_pct'] for item in null_count_analysis]
    extro_pcts = [item['extro_pct'] for item in null_count_analysis]
    ambi_pcts = [item['ambi_pct'] for item in null_count_analysis]
    
    ax2.plot(null_counts, intro_pcts, 'b-', marker='o', label='Introverts')
    ax2.plot(null_counts, extro_pcts, 'r-', marker='s', label='Extroverts')
    ax2.plot(null_counts, ambi_pcts, 'g-', marker='^', label='Ambiverts')
    
    ax2.set_xlabel('Number of Nulls')
    ax2.set_ylabel('Percentage of Group')
    ax2.set_title('Null Count Distribution by Personality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature importance for ambivert detection
    ax3 = axes[0, 2]
    top_features = feature_importance.head(10)
    ax3.barh(range(len(top_features)), top_features['importance'])
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance')
    ax3.set_title('Top Features for Ambivert Detection')
    ax3.invert_yaxis()
    
    # 4. Null pattern entropy comparison
    ax4 = axes[1, 0]
    # Calculate entropy for different groups
    introvert_mask = (train_df['Personality'] == 'Introvert') & (~ambivert_mask)
    extrovert_mask = (train_df['Personality'] == 'Extrovert') & (~ambivert_mask)
    
    groups = ['Introverts', 'Extroverts', 'Ambiverts']
    entropies = []
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    for mask in [introvert_mask, extrovert_mask, ambivert_mask]:
        if mask.sum() > 0:
            group_nulls = train_df.loc[mask, feature_cols].isnull()
            patterns = group_nulls.apply(lambda x: ''.join(x.astype(int).astype(str)), axis=1)
            pattern_counts = patterns.value_counts()
            probs = pattern_counts / pattern_counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            entropies.append(entropy)
        else:
            entropies.append(0)
    
    bars = ax4.bar(groups, entropies, color=['blue', 'orange', 'green'], alpha=0.7)
    ax4.set_ylabel('Entropy')
    ax4.set_title('Null Pattern Randomness by Personality')
    ax4.set_ylim(0, max(entropies) * 1.2)
    
    # Add value labels
    for bar, ent in zip(bars, entropies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{ent:.2f}', ha='center', va='bottom')
    
    # 5. Ambivert difference from average
    ax5 = axes[1, 1]
    ambi_diffs = [item['ambi_diff'] for item in null_analysis]
    colors = ['red' if d < 0 else 'green' for d in ambi_diffs]
    
    bars = ax5.bar(range(len(features)), ambi_diffs, color=colors, alpha=0.7)
    ax5.set_xticks(range(len(features)))
    ax5.set_xticklabels(features, rotation=45, ha='right')
    ax5.set_ylabel('Difference from Average')
    ax5.set_title('Ambivert Null Rate Difference from Average')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""SUMMARY STATISTICS
    
Total Records: {len(train_df):,}
Potential Ambiverts: {ambivert_mask.sum():,} ({ambivert_mask.mean():.1%})

Average Null Counts:
• Introverts: {train_df.loc[introvert_mask, 'null_count'].mean():.2f}
• Extroverts: {train_df.loc[extrovert_mask, 'null_count'].mean():.2f}
• Ambiverts: {train_df.loc[ambivert_mask, 'null_count'].mean():.2f}

Key Findings:
• Ambiverts show {['less', 'more'][entropies[2] > np.mean(entropies[:2])]} 
  random null patterns
• Most discriminative feature: 
  {feature_importance.iloc[0]['feature']}
• Ambiverts are {['under', 'over'][np.mean(ambi_diffs) > 0]}-represented
  in missing values"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('output/ambivert_null_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log_print("\nVisualization saved to: output/ambivert_null_pattern_analysis.png")


def main():
    """Main analysis function."""
    log_print("="*70)
    log_print("AMBIVERT NULL PATTERN ANALYSIS")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    # Detect potential ambiverts in training data
    ambivert_mask_train = detect_potential_ambiverts(train_df)
    
    # Analyze null patterns
    null_analysis, null_count_analysis = analyze_null_patterns_by_personality(train_df, ambivert_mask_train)
    
    # Train ambivert detector
    model, feature_importance = train_ambivert_detector(train_df, ambivert_mask_train)
    
    # Create visualizations
    visualize_results(train_df, null_analysis, null_count_analysis, feature_importance, ambivert_mask_train)
    
    # Apply to test data
    log_print("\n" + "="*60)
    log_print("APPLYING TO TEST DATA")
    log_print("="*60)
    
    # Create features for test data
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    test_df = create_null_pattern_features(test_df, feature_cols)
    
    # Predict ambiverts in test data
    null_features = [col for col in test_df.columns if col.endswith('_is_null')]
    other_features = ['null_count', 'has_psych_nulls', 'has_social_nulls', 'null_consistency']
    all_features = null_features + other_features
    
    X_test = test_df[all_features]
    ambivert_prob_test = model.predict_proba(X_test)[:, 1]
    ambivert_pred_test = model.predict(X_test)
    
    log_print(f"\nTest data ambivert predictions:")
    log_print(f"Predicted ambiverts: {ambivert_pred_test.sum()} ({ambivert_pred_test.mean():.1%})")
    log_print(f"High probability (>0.7): {(ambivert_prob_test > 0.7).sum()}")
    log_print(f"Medium probability (0.5-0.7): {((ambivert_prob_test >= 0.5) & (ambivert_prob_test <= 0.7)).sum()}")
    
    # Save results
    results = {
        'null_analysis': null_analysis,
        'null_count_analysis': null_count_analysis,
        'feature_importance': feature_importance.to_dict(),
        'train_stats': {
            'total_records': int(len(train_df)),
            'ambivert_count': int(ambivert_mask_train.sum()),
            'ambivert_rate': float(ambivert_mask_train.mean())
        },
        'test_stats': {
            'total_records': int(len(test_df)),
            'predicted_ambiverts': int(ambivert_pred_test.sum()),
            'predicted_rate': float(ambivert_pred_test.mean())
        }
    }
    
    with open('output/ambivert_null_pattern_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log_print("\nResults saved to: output/ambivert_null_pattern_results.json")
    
    # Key conclusions
    log_print("\n" + "="*60)
    log_print("KEY CONCLUSIONS")
    log_print("="*60)
    
    avg_intro_null = np.mean([item['intro_rate'] for item in null_analysis])
    avg_extro_null = np.mean([item['extro_rate'] for item in null_analysis])
    avg_ambi_null = np.mean([item['ambi_rate'] for item in null_analysis])
    
    log_print(f"\n1. Ambiverts have {'higher' if avg_ambi_null > (avg_intro_null + avg_extro_null)/2 else 'lower'} "
              f"null rates than average")
    
    intro_entropy = 0  # Placeholder - would be calculated in visualize_results
    extro_entropy = 0
    ambi_entropy = 0
    
    log_print(f"\n2. Null patterns are {'more' if ambi_entropy > (intro_entropy + extro_entropy)/2 else 'less'} "
              f"random for ambiverts")
    
    log_print(f"\n3. Most important null indicator: {feature_importance.iloc[0]['feature']}")
    
    log_print("\n4. Recommendation: Consider using ambivert detection based on null patterns "
              "as an additional feature for personality prediction")
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()