#!/usr/bin/env python3
"""
RETURN TO PROVEN STRATEGY - NO OVERFITTING
==========================================

This script returns to the proven ambivert detection strategy without the
null pattern overfitting that caused poor leaderboard performance.

Key principles:
1. Simple is better - avoid complex feature engineering
2. Focus on ambivert detection using behavioral patterns
3. Use conservative thresholds to avoid overfitting
4. Trust the 0.975708 ceiling as a real constraint

Author: Claude
Date: 2025-07-05 11:40
"""

# PURPOSE: Return to simple, proven strategy without overfitting
# HYPOTHESIS: The 0.975708 ceiling is real; we should aim for stable ~0.975 performance
# EXPECTED: Achieve consistent 0.975+ on both CV and leaderboard
# RESULT: [To be determined after execution]

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

# Output file
output_file = open('output/20250705_1140_return_to_proven_strategy.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def create_simple_features(df):
    """Create only the most basic, proven features."""
    # No complex null pattern features that caused overfitting
    # Just simple behavioral scores
    
    # Basic extroversion indicators
    df['extroversion_score'] = (
        df['Social_event_attendance'] + 
        df['Going_outside'] + 
        (10 - df['Time_spent_Alone']) +
        df['Friends_circle_size'] / 2 +
        df['Post_frequency'] / 2
    ) / 5
    
    # Basic introversion indicators
    df['introversion_score'] = (
        df['Time_spent_Alone'] +
        (10 - df['Social_event_attendance']) +
        (10 - df['Going_outside']) +
        (10 - df['Friends_circle_size']) / 2 +
        (10 - df['Post_frequency']) / 2
    ) / 5
    
    # Balance indicator
    df['balance_score'] = abs(df['extroversion_score'] - df['introversion_score'])
    
    return df


def detect_ambiverts_conservative(df):
    """Conservative ambivert detection to avoid overfitting."""
    ambivert_mask = pd.Series(False, index=df.index)
    
    # Only the most clear behavioral patterns
    balanced_behavior = (
        (df['Social_event_attendance'].between(4, 6)) &
        (df['Time_spent_Alone'].between(3, 5)) &
        (df['Friends_circle_size'].between(5, 7)) &
        (df['balance_score'] < 2)  # Low difference between intro/extro scores
    )
    
    ambivert_mask |= balanced_behavior
    
    # Special marker detection (but conservative)
    marker_social = abs(df['Social_event_attendance'] - 5.265106) < 0.001
    marker_alone = abs(df['Time_spent_Alone'] - 3.137764) < 0.001
    
    # Require multiple markers to avoid false positives
    has_multiple_markers = (marker_social.astype(int) + marker_alone.astype(int)) >= 2
    ambivert_mask |= has_multiple_markers
    
    return ambivert_mask


def train_simple_model(train_df):
    """Train a simple, robust model."""
    log_print("\n" + "="*60)
    log_print("TRAINING SIMPLE ROBUST MODEL")
    log_print("="*60)
    
    # Create simple features
    train_df = create_simple_features(train_df)
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency',
                   'extroversion_score', 'introversion_score', 'balance_score']
    
    # Encode categorical
    train_df['Stage_fear'] = train_df['Stage_fear'].map({'Yes': 1, 'No': 0})
    train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    
    X = train_df[feature_cols]
    y = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # Detect ambiverts
    ambivert_mask = detect_ambiverts_conservative(train_df)
    log_print(f"\nDetected {ambivert_mask.sum()} potential ambiverts ({ambivert_mask.mean():.1%})")
    
    # Simple cross-validation
    cv_scores = []
    models = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        log_print(f"\nFold {fold+1}/5")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Simple Random Forest - no fancy hyperparameters
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=20,  # Prevent overfitting
            min_samples_leaf=10,   # Prevent overfitting
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train_fold)
        
        # Get predictions
        val_pred = model.predict(X_val_scaled)
        val_prob = model.predict_proba(X_val_scaled)[:, 1]
        
        # Conservative ambivert handling
        val_ambiverts = ambivert_mask.iloc[val_idx]
        ambivert_indices = val_ambiverts[val_ambiverts].index
        
        # For ambiverts with uncertain predictions, slightly favor Extrovert
        uncertain_mask = (val_prob > 0.45) & (val_prob < 0.55)
        ambivert_uncertain = val_idx[uncertain_mask & val_ambiverts.values]
        
        if len(ambivert_uncertain) > 0:
            # 96% of ambiverts are labeled as Extrovert
            val_pred[uncertain_mask & val_ambiverts.values] = 1
        
        accuracy = accuracy_score(y_val_fold, val_pred)
        cv_scores.append(accuracy)
        log_print(f"  Accuracy: {accuracy:.5f}")
        
        models.append({
            'model': model,
            'scaler': scaler
        })
    
    log_print(f"\nMean CV Score: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})")
    
    return models, cv_scores


def apply_to_test(test_df, models):
    """Apply the simple model to test data."""
    log_print("\n" + "="*60)
    log_print("APPLYING TO TEST DATA")
    log_print("="*60)
    
    # Create features
    test_df = create_simple_features(test_df)
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency',
                   'extroversion_score', 'introversion_score', 'balance_score']
    
    # Encode categorical
    test_df['Stage_fear'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})
    test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    
    X_test = test_df[feature_cols]
    
    # Detect ambiverts
    ambivert_mask = detect_ambiverts_conservative(test_df)
    log_print(f"\nDetected {ambivert_mask.sum()} potential ambiverts in test ({ambivert_mask.mean():.1%})")
    
    # Ensemble predictions
    all_probs = []
    
    for model_dict in models:
        model = model_dict['model']
        scaler = model_dict['scaler']
        
        X_test_scaled = scaler.transform(X_test)
        prob = model.predict_proba(X_test_scaled)[:, 1]
        all_probs.append(prob)
    
    # Average probabilities
    final_prob = np.mean(all_probs, axis=0)
    
    # Standard predictions
    predictions = (final_prob > 0.5).astype(int)
    
    # Conservative ambivert handling
    uncertain_mask = (final_prob > 0.45) & (final_prob < 0.55)
    ambivert_uncertain = uncertain_mask & ambivert_mask
    
    if ambivert_uncertain.sum() > 0:
        # For uncertain ambiverts, use 0.48 threshold (slight Extrovert bias)
        predictions[ambivert_uncertain] = (final_prob[ambivert_uncertain] > 0.48).astype(int)
        log_print(f"\nApplied special handling to {ambivert_uncertain.sum()} uncertain ambiverts")
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': np.where(predictions == 1, 'Extrovert', 'Introvert')
    })
    
    # Summary
    log_print(f"\nPrediction Summary:")
    log_print(f"Introverts: {(submission['Personality'] == 'Introvert').sum()} "
              f"({(submission['Personality'] == 'Introvert').mean():.1%})")
    log_print(f"Extroverts: {(submission['Personality'] == 'Extrovert').sum()} "
              f"({(submission['Personality'] == 'Extrovert').mean():.1%})")
    
    return submission


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("RETURN TO PROVEN STRATEGY - NO OVERFITTING")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    log_print(f"Train shape: {train_df.shape}")
    log_print(f"Test shape: {test_df.shape}")
    
    # Check class distribution
    intro_rate = (train_df['Personality'] == 'Introvert').mean()
    log_print(f"\nTraining data distribution:")
    log_print(f"Introverts: {intro_rate:.1%}")
    log_print(f"Extroverts: {1-intro_rate:.1%}")
    
    # Train model
    models, cv_scores = train_simple_model(train_df)
    
    # Apply to test
    submission = apply_to_test(test_df, models)
    
    # Save submission
    submission_path = "output/proven_strategy_submission_20250705_1140.csv"
    submission.to_csv(submission_path, index=False)
    log_print(f"\nSubmission saved to: {submission_path}")
    
    # Save results
    results = {
        'cv_scores': cv_scores,
        'mean_cv': float(np.mean(cv_scores)),
        'std_cv': float(np.std(cv_scores)),
        'expected_lb': float(np.mean(cv_scores) * 0.998),  # Conservative estimate
        'prediction_distribution': {
            'introverts': int((submission['Personality'] == 'Introvert').sum()),
            'extroverts': int((submission['Personality'] == 'Extrovert').sum()),
            'intro_rate': float((submission['Personality'] == 'Introvert').mean())
        }
    }
    
    with open('output/proven_strategy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log_print("\nResults saved to: output/proven_strategy_results.json")
    
    # Final assessment
    log_print("\n" + "="*70)
    log_print("FINAL ASSESSMENT")
    log_print("="*70)
    log_print(f"\nMean CV: {np.mean(cv_scores):.5f}")
    log_print(f"Expected LB: ~{np.mean(cv_scores) * 0.998:.5f}")
    
    if np.mean(cv_scores) >= 0.975:
        log_print("\n✓ This should achieve stable performance near the 0.975708 ceiling")
        log_print("  without the overfitting issues of the null pattern approach.")
    else:
        log_print("\n⚠ Performance below expected. May need parameter tuning.")
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()