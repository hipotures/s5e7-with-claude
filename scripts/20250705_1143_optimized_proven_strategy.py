#!/usr/bin/env python3
"""
OPTIMIZED PROVEN STRATEGY - TARGET 0.975
========================================

This script optimizes the proven strategy to get closer to 0.975 accuracy
by using ensemble of different algorithms and careful threshold tuning.

Key improvements:
1. Ensemble of RF, XGBoost, and LightGBM
2. Better ambivert detection using probability patterns
3. Optimized probability thresholds
4. Careful handling of edge cases

Author: Claude
Date: 2025-07-05 11:43
"""

# PURPOSE: Optimize proven strategy to reach ~0.975 accuracy
# HYPOTHESIS: Ensemble and threshold optimization can push us closer to ceiling
# EXPECTED: Achieve 0.974-0.975 on CV and similar on leaderboard
# RESULT: [To be determined after execution]

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

# Output file
output_file = open('output/20250705_1143_optimized_proven_strategy.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def create_balanced_features(df):
    """Create balanced behavioral features."""
    # Basic scores
    df['social_score'] = (df['Social_event_attendance'] + df['Going_outside']) / 2
    df['alone_score'] = df['Time_spent_Alone']
    df['online_presence'] = (df['Post_frequency'] + df['Friends_circle_size']) / 2
    
    # Psychological indicators
    df['anxiety_indicator'] = 0
    if 'Stage_fear' in df.columns and df['Stage_fear'].dtype == 'object':
        df['anxiety_indicator'] += (df['Stage_fear'] == 'Yes').astype(int)
    if 'Drained_after_socializing' in df.columns and df['Drained_after_socializing'].dtype == 'object':
        df['anxiety_indicator'] += (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    # Balance indicators
    df['social_balance'] = df['social_score'] - df['alone_score']
    df['activity_variance'] = np.var([df['Social_event_attendance'], 
                                     df['Going_outside'], 
                                     df['Post_frequency']], axis=0)
    
    # Interaction features
    df['social_anxiety_interaction'] = df['social_score'] * (2 - df['anxiety_indicator'])
    df['online_offline_ratio'] = df['Post_frequency'] / (df['Going_outside'] + 1)
    
    # Fill any NaN values that might have been created
    df = df.fillna(0)
    
    return df


def detect_edge_cases(df, probabilities):
    """Detect edge cases including ambiverts and extreme personalities."""
    edge_cases = pd.DataFrame(index=df.index)
    
    # Uncertainty based on probability
    edge_cases['uncertain'] = (probabilities > 0.4) & (probabilities < 0.6)
    edge_cases['very_uncertain'] = (probabilities > 0.45) & (probabilities < 0.55)
    
    # Behavioral ambiverts
    edge_cases['behavioral_ambivert'] = (
        (df['Social_event_attendance'].between(4, 6)) &
        (df['Time_spent_Alone'].between(3, 5)) &
        (df['Friends_circle_size'].between(5, 8))
    )
    
    # Extreme personalities (very clear cases)
    edge_cases['extreme_introvert'] = (
        (df['Time_spent_Alone'] >= 8) &
        (df['Social_event_attendance'] <= 2) &
        (df['anxiety_indicator'] >= 1)
    )
    
    edge_cases['extreme_extrovert'] = (
        (df['Time_spent_Alone'] <= 2) &
        (df['Social_event_attendance'] >= 8) &
        (df['Friends_circle_size'] >= 8)
    )
    
    # Special markers (conservative)
    edge_cases['has_special_marker'] = (
        (abs(df['Social_event_attendance'] - 5.265106) < 0.001) |
        (abs(df['Going_outside'] - 4.044319) < 0.001) |
        (abs(df['Post_frequency'] - 4.982097) < 0.001) |
        (abs(df['Time_spent_Alone'] - 3.137764) < 0.001)
    )
    
    return edge_cases


def train_ensemble_model(train_df):
    """Train an ensemble of models."""
    log_print("\n" + "="*60)
    log_print("TRAINING OPTIMIZED ENSEMBLE MODEL")
    log_print("="*60)
    
    # Create features
    train_df = create_balanced_features(train_df)
    
    # Prepare features
    base_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency']
    engineered_features = ['social_score', 'alone_score', 'online_presence', 
                          'anxiety_indicator', 'social_balance', 'activity_variance',
                          'social_anxiety_interaction', 'online_offline_ratio']
    
    # Handle categorical features
    train_df['Stage_fear_binary'] = (train_df['Stage_fear'] == 'Yes').astype(int)
    train_df['Drained_binary'] = (train_df['Drained_after_socializing'] == 'Yes').astype(int)
    
    feature_cols = base_features + engineered_features + ['Stage_fear_binary', 'Drained_binary']
    
    X = train_df[feature_cols]
    y = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # Cross-validation
    cv_scores = []
    models_collection = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        log_print(f"\nFold {fold+1}/5")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Train multiple models
        models = {}
        
        # 1. Random Forest
        models['rf'] = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=8,
            max_features='sqrt',
            random_state=42
        )
        models['rf'].fit(X_train_scaled, y_train_fold)
        
        # 2. Extra Trees
        models['et'] = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=25,
            min_samples_leaf=10,
            random_state=42
        )
        models['et'].fit(X_train_scaled, y_train_fold)
        
        # 3. XGBoost
        models['xgb'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        models['xgb'].fit(X_train_scaled, y_train_fold, verbose=False)
        
        # 4. LightGBM
        models['lgb'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        models['lgb'].fit(X_train_scaled, y_train_fold)
        
        # 5. Logistic Regression (for stability)
        models['lr'] = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        models['lr'].fit(X_train_scaled, y_train_fold)
        
        # Get predictions from each model
        predictions = {}
        probabilities = {}
        
        for name, model in models.items():
            pred = model.predict(X_val_scaled)
            prob = model.predict_proba(X_val_scaled)[:, 1]
            predictions[name] = pred
            probabilities[name] = prob
            acc = accuracy_score(y_val_fold, pred)
            log_print(f"  {name}: {acc:.5f}")
        
        # Ensemble predictions
        # Weighted average based on individual performance
        weights = {
            'rf': 0.25,
            'et': 0.20,
            'xgb': 0.30,
            'lgb': 0.20,
            'lr': 0.05
        }
        
        ensemble_prob = sum(probabilities[name] * weight 
                           for name, weight in weights.items())
        
        # Detect edge cases
        val_df = train_df.iloc[val_idx].copy()
        edge_cases = detect_edge_cases(val_df, ensemble_prob)
        
        # Standard predictions
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        # Apply special handling
        # 1. For very uncertain behavioral ambiverts
        ambivert_uncertain = edge_cases['behavioral_ambivert'] & edge_cases['very_uncertain']
        if ambivert_uncertain.sum() > 0:
            # Use threshold of 0.48 (slight extrovert bias as 96% of ambiverts are extroverts)
            ensemble_pred[ambivert_uncertain] = (ensemble_prob[ambivert_uncertain] > 0.48).astype(int)
        
        # 2. For extreme personalities, trust the model more
        extreme_mask = edge_cases['extreme_introvert'] | edge_cases['extreme_extrovert']
        # No adjustment needed - these should be clear
        
        # 3. For special markers with uncertainty
        marker_uncertain = edge_cases['has_special_marker'] & edge_cases['uncertain']
        if marker_uncertain.sum() > 0:
            # These are likely ambiverts - use 0.47 threshold
            ensemble_pred[marker_uncertain] = (ensemble_prob[marker_uncertain] > 0.47).astype(int)
        
        accuracy = accuracy_score(y_val_fold, ensemble_pred)
        cv_scores.append(accuracy)
        log_print(f"  Ensemble: {accuracy:.5f}")
        
        models_collection.append({
            'models': models,
            'scaler': scaler,
            'weights': weights
        })
    
    log_print(f"\nMean CV Score: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})")
    
    return models_collection, cv_scores


def apply_to_test(test_df, models_collection):
    """Apply the ensemble to test data."""
    log_print("\n" + "="*60)
    log_print("APPLYING TO TEST DATA")
    log_print("="*60)
    
    # Create features
    test_df = create_balanced_features(test_df)
    
    # Prepare features
    base_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency']
    engineered_features = ['social_score', 'alone_score', 'online_presence', 
                          'anxiety_indicator', 'social_balance', 'activity_variance',
                          'social_anxiety_interaction', 'online_offline_ratio']
    
    test_df['Stage_fear_binary'] = (test_df['Stage_fear'] == 'Yes').astype(int)
    test_df['Drained_binary'] = (test_df['Drained_after_socializing'] == 'Yes').astype(int)
    
    feature_cols = base_features + engineered_features + ['Stage_fear_binary', 'Drained_binary']
    X_test = test_df[feature_cols]
    
    # Get predictions from each fold
    all_probs = []
    
    for fold_dict in models_collection:
        models = fold_dict['models']
        scaler = fold_dict['scaler']
        weights = fold_dict['weights']
        
        X_test_scaled = scaler.transform(X_test)
        
        # Get weighted ensemble probability for this fold
        fold_prob = sum(
            models[name].predict_proba(X_test_scaled)[:, 1] * weight
            for name, weight in weights.items()
        )
        all_probs.append(fold_prob)
    
    # Average across folds
    final_prob = np.mean(all_probs, axis=0)
    
    # Detect edge cases
    edge_cases = detect_edge_cases(test_df, final_prob)
    
    # Standard predictions
    predictions = (final_prob > 0.5).astype(int)
    
    # Apply special handling
    # Count how many samples get special handling
    special_handled = 0
    
    # 1. Behavioral ambiverts with uncertainty
    ambivert_uncertain = edge_cases['behavioral_ambivert'] & edge_cases['very_uncertain']
    if ambivert_uncertain.sum() > 0:
        predictions[ambivert_uncertain] = (final_prob[ambivert_uncertain] > 0.48).astype(int)
        special_handled += ambivert_uncertain.sum()
    
    # 2. Special markers with uncertainty
    marker_uncertain = edge_cases['has_special_marker'] & edge_cases['uncertain']
    if marker_uncertain.sum() > 0:
        predictions[marker_uncertain] = (final_prob[marker_uncertain] > 0.47).astype(int)
        special_handled += marker_uncertain.sum()
    
    log_print(f"\nApplied special handling to {special_handled} samples")
    log_print(f"Behavioral ambiverts: {edge_cases['behavioral_ambivert'].sum()}")
    log_print(f"With special markers: {edge_cases['has_special_marker'].sum()}")
    log_print(f"Uncertain predictions: {edge_cases['uncertain'].sum()}")
    
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
    
    return submission, final_prob


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("OPTIMIZED PROVEN STRATEGY - TARGET 0.975")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    log_print(f"Train shape: {train_df.shape}")
    log_print(f"Test shape: {test_df.shape}")
    
    # Train ensemble
    models_collection, cv_scores = train_ensemble_model(train_df)
    
    # Apply to test
    submission, probabilities = apply_to_test(test_df, models_collection)
    
    # Save submission
    submission_path = "output/optimized_strategy_submission_20250705_1143.csv"
    submission.to_csv(submission_path, index=False)
    log_print(f"\nSubmission saved to: {submission_path}")
    
    # Also save probabilities for analysis
    prob_df = pd.DataFrame({
        'id': test_df['id'],
        'probability': probabilities,
        'prediction': submission['Personality']
    })
    prob_df.to_csv("output/optimized_strategy_probabilities.csv", index=False)
    
    # Save results
    results = {
        'cv_scores': cv_scores,
        'mean_cv': float(np.mean(cv_scores)),
        'std_cv': float(np.std(cv_scores)),
        'expected_lb': float(np.mean(cv_scores) * 0.999),  # Very conservative
        'prediction_distribution': {
            'introverts': int((submission['Personality'] == 'Introvert').sum()),
            'extroverts': int((submission['Personality'] == 'Extrovert').sum()),
            'intro_rate': float((submission['Personality'] == 'Introvert').mean())
        }
    }
    
    with open('output/optimized_strategy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log_print("\nResults saved to: output/optimized_strategy_results.json")
    
    # Final assessment
    log_print("\n" + "="*70)
    log_print("FINAL ASSESSMENT")
    log_print("="*70)
    log_print(f"\nMean CV: {np.mean(cv_scores):.5f}")
    log_print(f"Expected LB: ~{np.mean(cv_scores) * 0.999:.5f}")
    
    if np.mean(cv_scores) >= 0.974:
        log_print("\n✓ This optimized ensemble should achieve performance very close")
        log_print("  to the 0.975708 ceiling with good generalization.")
    else:
        log_print("\n⚠ Performance still below target. The 0.975708 ceiling")
        log_print("  may indeed be a hard mathematical limit.")
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()