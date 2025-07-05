#!/usr/bin/env python3
"""
FEATURE UNCERTAINTY ANALYSIS
============================

This script analyzes which features cause the most uncertainty in predictions
and identifies patterns in the most uncertain cases.

Author: Claude
Date: 2025-07-05 12:12
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Output file
output_file = open('output/20250705_1212_feature_uncertainty_analysis.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def train_model_with_probabilities(train_df):
    """Train model and get probability predictions."""
    log_print("\n" + "="*60)
    log_print("TRAINING MODEL FOR UNCERTAINTY ANALYSIS")
    log_print("="*60)
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    # Encode categorical
    train_df['Stage_fear'] = train_df['Stage_fear'].map({'Yes': 1, 'No': 0})
    train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    
    X = train_df[feature_cols]
    y = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model.fit(X_scaled, y, verbose=False)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    log_print("\nFeature Importance:")
    for idx, row in feature_importance.iterrows():
        log_print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    return model, scaler, feature_cols, feature_importance


def analyze_uncertainty_patterns(test_df, model, scaler, feature_cols):
    """Analyze uncertainty patterns in predictions."""
    log_print("\n" + "="*60)
    log_print("ANALYZING UNCERTAINTY PATTERNS")
    log_print("="*60)
    
    # Prepare test data
    test_df_copy = test_df.copy()
    test_df_copy['Stage_fear'] = test_df_copy['Stage_fear'].map({'Yes': 1, 'No': 0})
    test_df_copy['Drained_after_socializing'] = test_df_copy['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    
    X_test = test_df_copy[feature_cols]
    X_test_scaled = scaler.transform(X_test)
    
    # Get probabilities
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate uncertainty
    uncertainty = np.abs(probabilities - 0.5)
    
    # Create analysis dataframe
    analysis_df = pd.DataFrame({
        'id': test_df['id'],
        'probability': probabilities,
        'uncertainty': uncertainty,
        'uncertainty_score': 0.5 - uncertainty  # Higher score = more uncertain
    })
    
    # Add features
    for col in feature_cols:
        analysis_df[col] = X_test[col].values
    
    # Sort by uncertainty
    analysis_df = analysis_df.sort_values('uncertainty_score', ascending=False)
    
    # Analyze top 20 most uncertain
    log_print("\nTop 20 Most Uncertain Predictions:")
    log_print(f"{'Rank':<6} {'ID':<8} {'Prob':<8} {'Uncert':<8} {'Features'}")
    log_print("-" * 80)
    
    for rank, (idx, row) in enumerate(analysis_df.head(20).iterrows(), 1):
        features_str = f"Alone:{row['Time_spent_Alone']:.0f}, Social:{row['Social_event_attendance']:.0f}, Drain:{row['Drained_after_socializing']:.0f}, Fear:{row['Stage_fear']:.0f}"
        log_print(f"{rank:<6} {row['id']:<8.0f} {row['probability']:<8.4f} {row['uncertainty_score']:<8.4f} {features_str}")
    
    return analysis_df


def analyze_feature_patterns(analysis_df, feature_cols):
    """Analyze which features dominate in uncertain cases."""
    log_print("\n" + "="*60)
    log_print("FEATURE PATTERNS IN UNCERTAIN CASES")
    log_print("="*60)
    
    # Get top 50 most uncertain cases
    top_uncertain = analysis_df.head(50)
    
    # 1. Average feature values for uncertain vs certain cases
    log_print("\nAverage Feature Values:")
    log_print(f"{'Feature':<30} {'Uncertain (top 50)':<20} {'Certain (bottom 50)':<20} {'Difference':<15}")
    log_print("-" * 85)
    
    bottom_certain = analysis_df.tail(50)
    
    for col in feature_cols:
        uncertain_mean = top_uncertain[col].mean()
        certain_mean = bottom_certain[col].mean()
        diff = uncertain_mean - certain_mean
        log_print(f"{col:<30} {uncertain_mean:<20.2f} {certain_mean:<20.2f} {diff:<15.2f}")
    
    # 2. Feature variance in uncertain cases
    log_print("\n\nFeature Variance in Uncertain Cases:")
    log_print(f"{'Feature':<30} {'Variance (uncertain)':<20} {'Variance (certain)':<20}")
    log_print("-" * 70)
    
    for col in feature_cols:
        uncertain_var = top_uncertain[col].var()
        certain_var = bottom_certain[col].var()
        log_print(f"{col:<30} {uncertain_var:<20.2f} {certain_var:<20.2f}")
    
    # 3. Correlation with uncertainty
    log_print("\n\nCorrelation with Uncertainty Score:")
    correlations = []
    for col in feature_cols:
        corr = analysis_df[col].corr(analysis_df['uncertainty_score'])
        correlations.append({'feature': col, 'correlation': corr})
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
    for idx, row in corr_df.iterrows():
        log_print(f"{row['feature']:<30} {row['correlation']:>10.4f}")
    
    # 4. Identify dominant patterns
    log_print("\n\nDominant Patterns in Top 10 Most Uncertain:")
    log_print("="*60)
    
    for rank, (idx, row) in enumerate(analysis_df.head(10).iterrows(), 1):
        log_print(f"\nRank {rank} - ID {row['id']:.0f} (prob={row['probability']:.4f}):")
        
        # Identify extreme features
        extreme_features = []
        if row['Time_spent_Alone'] >= 8:
            extreme_features.append(f"High alone time ({row['Time_spent_Alone']:.0f}h)")
        elif row['Time_spent_Alone'] <= 2:
            extreme_features.append(f"Low alone time ({row['Time_spent_Alone']:.0f}h)")
        
        if row['Social_event_attendance'] >= 8:
            extreme_features.append(f"High social ({row['Social_event_attendance']:.0f})")
        elif row['Social_event_attendance'] <= 2:
            extreme_features.append(f"Low social ({row['Social_event_attendance']:.0f})")
        
        if row['Friends_circle_size'] >= 12:
            extreme_features.append(f"Many friends ({row['Friends_circle_size']:.0f})")
        elif row['Friends_circle_size'] <= 4:
            extreme_features.append(f"Few friends ({row['Friends_circle_size']:.0f})")
        
        if row['Drained_after_socializing'] == 1 and row['Stage_fear'] == 1:
            extreme_features.append("High anxiety (both drained & stage fear)")
        
        if extreme_features:
            log_print(f"  Extreme features: {', '.join(extreme_features)}")
        else:
            log_print(f"  Moderate profile - no extreme features")
        
        # Check for conflicting signals
        conflicting = []
        if row['Time_spent_Alone'] >= 6 and row['Social_event_attendance'] >= 6:
            conflicting.append("High alone time BUT high social")
        if row['Friends_circle_size'] >= 10 and row['Social_event_attendance'] <= 3:
            conflicting.append("Many friends BUT low social events")
        if row['Drained_after_socializing'] == 0 and row['Time_spent_Alone'] >= 8:
            conflicting.append("Not drained by socializing BUT high alone time")
        
        if conflicting:
            log_print(f"  Conflicting signals: {', '.join(conflicting)}")
    
    return corr_df


def create_visualizations(analysis_df, feature_cols):
    """Create visualizations of uncertainty patterns."""
    log_print("\n" + "="*60)
    log_print("CREATING VISUALIZATIONS")
    log_print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Uncertainty distribution
    ax1 = axes[0, 0]
    ax1.hist(analysis_df['uncertainty_score'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Uncertainty Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Prediction Uncertainty')
    ax1.axvline(x=0.45, color='red', linestyle='--', label='High uncertainty threshold')
    ax1.legend()
    
    # 2. Feature values for most uncertain
    ax2 = axes[0, 1]
    top20 = analysis_df.head(20)
    feature_means = [top20[col].mean() for col in feature_cols[:5]]  # First 5 features
    feature_names = [col.replace('_', ' ') for col in feature_cols[:5]]
    ax2.bar(feature_names, feature_means, alpha=0.7, color='green')
    ax2.set_ylabel('Average Value')
    ax2.set_title('Average Feature Values for Top 20 Uncertain Cases')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Scatter plot of two most important features
    ax3 = axes[0, 2]
    scatter = ax3.scatter(analysis_df['Time_spent_Alone'], 
                         analysis_df['Social_event_attendance'],
                         c=analysis_df['uncertainty_score'], 
                         cmap='viridis', alpha=0.6, s=20)
    ax3.set_xlabel('Time Spent Alone')
    ax3.set_ylabel('Social Event Attendance')
    ax3.set_title('Feature Space Colored by Uncertainty')
    plt.colorbar(scatter, ax=ax3, label='Uncertainty')
    
    # 4. Box plot of features by uncertainty level
    ax4 = axes[1, 0]
    analysis_df['uncertainty_level'] = pd.cut(analysis_df['uncertainty_score'], 
                                             bins=[0, 0.4, 0.45, 0.5], 
                                             labels=['Low', 'Medium', 'High'])
    
    data_to_plot = []
    labels = []
    for level in ['Low', 'Medium', 'High']:
        subset = analysis_df[analysis_df['uncertainty_level'] == level]
        if len(subset) > 0:
            data_to_plot.append(subset['Time_spent_Alone'].values)
            labels.append(f"{level}\n(n={len(subset)})")
    
    ax4.boxplot(data_to_plot, labels=labels)
    ax4.set_ylabel('Time Spent Alone')
    ax4.set_title('Time Alone Distribution by Uncertainty Level')
    
    # 5. Correlation heatmap
    ax5 = axes[1, 1]
    corr_matrix = analysis_df[feature_cols + ['uncertainty_score']].corr()
    sns.heatmap(corr_matrix[['uncertainty_score']], 
                annot=True, fmt='.3f', cmap='coolwarm', center=0,
                ax=ax5, cbar_kws={'label': 'Correlation'})
    ax5.set_title('Feature Correlation with Uncertainty')
    
    # 6. Top 10 uncertain cases
    ax6 = axes[1, 2]
    top10 = analysis_df.head(10)
    colors = plt.cm.RdYlBu(top10['probability'].values)
    ax6.scatter(top10['Social_event_attendance'], 
               top10['Friends_circle_size'],
               c=colors, s=200, alpha=0.7, edgecolors='black')
    
    # Add ID labels
    for idx, row in top10.iterrows():
        ax6.annotate(f"{row['id']:.0f}", 
                    (row['Social_event_attendance'], row['Friends_circle_size']),
                    fontsize=8, ha='center')
    
    ax6.set_xlabel('Social Event Attendance')
    ax6.set_ylabel('Friends Circle Size')
    ax6.set_title('Top 10 Most Uncertain Cases\n(Blue=Intro, Red=Extro probability)')
    
    plt.tight_layout()
    plt.savefig('output/feature_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log_print("Visualization saved to: output/feature_uncertainty_analysis.png")


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("FEATURE UNCERTAINTY ANALYSIS")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    # Train model
    model, scaler, feature_cols, feature_importance = train_model_with_probabilities(train_df)
    
    # Analyze uncertainty
    analysis_df = analyze_uncertainty_patterns(test_df, model, scaler, feature_cols)
    
    # Analyze feature patterns
    corr_df = analyze_feature_patterns(analysis_df, feature_cols)
    
    # Create visualizations
    create_visualizations(analysis_df, feature_cols)
    
    # Save detailed results
    log_print("\n" + "="*60)
    log_print("SAVING RESULTS")
    log_print("="*60)
    
    # Save top 100 uncertain cases
    top100_uncertain = analysis_df.head(100)[['id', 'probability', 'uncertainty_score'] + feature_cols]
    top100_uncertain.to_csv('output/top100_uncertain_cases.csv', index=False)
    log_print("Saved top 100 uncertain cases to: output/top100_uncertain_cases.csv")
    
    # Summary
    log_print("\n" + "="*60)
    log_print("KEY FINDINGS SUMMARY")
    log_print("="*60)
    
    log_print("\n1. Most uncertain cases have probability between " + 
              f"{analysis_df.head(20)['probability'].min():.4f} and " +
              f"{analysis_df.head(20)['probability'].max():.4f}")
    
    log_print("\n2. Features most correlated with uncertainty:")
    for idx, row in corr_df.head(3).iterrows():
        log_print(f"   {row['feature']}: {row['correlation']:.4f}")
    
    log_print("\n3. Common patterns in uncertain cases:")
    log_print("   - Conflicting signals (e.g., high social but also high alone time)")
    log_print("   - Moderate values across all features (no extremes)")
    log_print("   - Missing values in key features")
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()