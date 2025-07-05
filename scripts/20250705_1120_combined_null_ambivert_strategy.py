#!/usr/bin/env python3
"""
COMBINED NULL-AMBIVERT STRATEGY MODEL
=====================================

This script combines two key discoveries:
1. Missing values encode personality (introverts have 2-4x more nulls)
2. Ambiverts have complete data (74.3% have zero nulls)

Strategy:
- Use null patterns to identify introverts
- Use data completeness to identify potential ambiverts
- Apply special rules for ambiguous cases

Author: Claude
Date: 2025-07-05 11:20
"""

# PURPOSE: Combine null patterns and ambivert detection for breakthrough performance
# HYPOTHESIS: Ambiverts are hidden in complete data records, while introverts hide in nulls
# EXPECTED: Better identification of edge cases leading to accuracy improvement
# RESULT: [To be determined after execution]

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import warnings
warnings.filterwarnings('ignore')

# Output file
output_file = open('output/20250705_1120_combined_null_ambivert_strategy.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


def create_null_features(df):
    """Create comprehensive null-based features."""
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    # Basic null indicators
    for col in feature_cols:
        df[f'{col}_null'] = df[col].isnull().astype(int)
    
    # Total null count
    df['null_count'] = df[feature_cols].isnull().sum(axis=1)
    
    # Null pattern groups
    df['has_psych_nulls'] = (df['Drained_after_socializing'].isnull() | 
                             df['Stage_fear'].isnull()).astype(int)
    df['has_social_nulls'] = (df['Social_event_attendance'].isnull() | 
                              df['Going_outside'].isnull()).astype(int)
    
    # Special patterns
    df['has_drained_null'] = df['Drained_after_socializing'].isnull().astype(int)
    df['has_stage_fear_null'] = df['Stage_fear'].isnull().astype(int)
    
    # Data completeness indicators
    df['no_nulls'] = (df['null_count'] == 0).astype(int)
    df['complete_social_data'] = ((~df['Social_event_attendance'].isnull()) & 
                                  (~df['Friends_circle_size'].isnull()) & 
                                  (~df['Going_outside'].isnull())).astype(int)
    
    # Weighted null score (psychological nulls weighted higher)
    df['weighted_null_score'] = (
        df['Drained_after_socializing_null'] * 2.0 +
        df['Stage_fear_null'] * 1.5 +
        df['Social_event_attendance_null'] * 1.0 +
        df['Going_outside_null'] * 0.8 +
        df['Friends_circle_size_null'] * 0.7 +
        df['Post_frequency_null'] * 0.6 +
        df['Time_spent_Alone_null'] * 0.5
    )
    
    # Null consistency (are nulls clustered?)
    null_matrix = df[[col + '_null' for col in feature_cols]].values
    df['null_variance'] = np.var(null_matrix, axis=1)
    
    return df


def detect_ambiverts(df, predictions=None):
    """Enhanced ambivert detection using multiple signals."""
    ambivert_scores = pd.Series(0.0, index=df.index)
    
    # Signal 1: Complete data (strongest signal from analysis)
    if 'no_nulls' in df.columns:
        ambivert_scores += df['no_nulls'] * 0.4
    
    # Signal 2: Moderate feature values
    if all(col in df.columns for col in ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']):
        moderate_alone = df['Time_spent_Alone'].between(2, 4)
        moderate_social = df['Social_event_attendance'].between(3, 5)
        moderate_friends = df['Friends_circle_size'].between(5, 8)
        ambivert_scores += (moderate_alone & moderate_social & moderate_friends).astype(float) * 0.3
    
    # Signal 3: Prediction uncertainty (if available)
    if predictions is not None and 'prob' in predictions.columns:
        uncertain = predictions['prob'].between(0.4, 0.6)
        ambivert_scores += uncertain.astype(float) * 0.2
    
    # Signal 4: Special behavioral patterns
    if 'Post_frequency' in df.columns and 'Going_outside' in df.columns:
        balanced_behavior = (
            df['Post_frequency'].between(3, 6) & 
            df['Going_outside'].between(3, 5)
        )
        ambivert_scores += balanced_behavior.astype(float) * 0.1
    
    return ambivert_scores


def impute_personality_aware(X_train, y_train, X_val):
    """Impute missing values using personality information during training."""
    from sklearn.impute import SimpleImputer
    
    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()
    
    # For training: impute based on personality class
    for col in X_train.columns:
        if X_train[col].isnull().any():
            # Introverts
            intro_mask = y_train == 0
            if intro_mask.sum() > 0:
                intro_median = X_train.loc[intro_mask, col].median()
                X_train_imputed.loc[intro_mask & X_train[col].isnull(), col] = intro_median
            
            # Extroverts
            extro_mask = y_train == 1
            if extro_mask.sum() > 0:
                extro_median = X_train.loc[extro_mask, col].median()
                X_train_imputed.loc[extro_mask & X_train[col].isnull(), col] = extro_median
    
    # For validation: use overall median
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train_imputed)
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    return X_train_imputed, X_val_imputed


def train_combined_model(train_df):
    """Train model with combined null-ambivert strategy."""
    log_print("\n" + "="*60)
    log_print("TRAINING COMBINED NULL-AMBIVERT MODEL")
    log_print("="*60)
    
    # Create features
    train_df = create_null_features(train_df)
    
    # Prepare data
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    # Add null features
    null_features = [col + '_null' for col in feature_cols]
    other_features = ['null_count', 'has_psych_nulls', 'has_social_nulls', 
                     'has_drained_null', 'has_stage_fear_null', 'no_nulls',
                     'complete_social_data', 'weighted_null_score', 'null_variance']
    
    all_features = feature_cols + null_features + other_features
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(train_df['Personality'])
    
    # Encode categorical features
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})
    
    X = train_df[all_features]
    
    # Cross-validation
    cv_scores = []
    models = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        log_print(f"\nFold {fold+1}/5")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Impute with personality awareness
        X_train_imputed, X_val_imputed = impute_personality_aware(
            X_train_fold[feature_cols], y_train_fold, X_val_fold[feature_cols]
        )
        
        # Combine imputed features with null indicators
        X_train_combined = pd.concat([
            X_train_imputed,
            X_train_fold[null_features + other_features]
        ], axis=1)
        
        X_val_combined = pd.concat([
            X_val_imputed,
            X_val_fold[null_features + other_features]
        ], axis=1)
        
        # Add ambivert scores
        ambivert_train = detect_ambiverts(X_train_combined)
        ambivert_val = detect_ambiverts(X_val_combined)
        
        X_train_combined['ambivert_score'] = ambivert_train
        X_val_combined['ambivert_score'] = ambivert_val
        
        # Train ensemble
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_combined, y_train_fold)
        xgb_pred = xgb_model.predict(X_val_combined)
        xgb_prob = xgb_model.predict_proba(X_val_combined)[:, 1]
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_combined, y_train_fold)
        lgb_pred = lgb_model.predict(X_val_combined)
        lgb_prob = lgb_model.predict_proba(X_val_combined)[:, 1]
        
        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train_combined, y_train_fold)
        cat_pred = cat_model.predict(X_val_combined)
        cat_prob = cat_model.predict_proba(X_val_combined)[:, 1]
        
        # Ensemble probabilities
        ensemble_prob = 0.4 * xgb_prob + 0.3 * lgb_prob + 0.3 * cat_prob
        
        # Apply special rules
        predictions_df = pd.DataFrame({
            'prob': ensemble_prob,
            'pred': (ensemble_prob > 0.5).astype(int)
        }, index=X_val_combined.index)
        
        # Rule 1: Strong null pattern → Introvert
        strong_null_mask = (
            (X_val_combined['has_drained_null'] == 1) & 
            (X_val_combined['weighted_null_score'] > 2) &
            (ensemble_prob > 0.3) & (ensemble_prob < 0.6)
        )
        predictions_df.loc[strong_null_mask, 'pred'] = 0  # Introvert
        
        # Rule 2: Complete data + high ambivert score + uncertain → check carefully
        ambivert_mask = (
            (X_val_combined['no_nulls'] == 1) &
            (X_val_combined['ambivert_score'] > 0.6) &
            (ensemble_prob > 0.45) & (ensemble_prob < 0.55)
        )
        # For ambiverts, use a calibrated threshold
        predictions_df.loc[ambivert_mask, 'pred'] = (ensemble_prob[ambivert_mask] > 0.475).astype(int)
        
        # Rule 3: No nulls + extreme probability → trust the model
        confident_mask = (
            (X_val_combined['no_nulls'] == 1) &
            ((ensemble_prob < 0.2) | (ensemble_prob > 0.8))
        )
        # Keep original predictions for confident cases
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val_fold, predictions_df['pred'])
        cv_scores.append(accuracy)
        log_print(f"  Accuracy: {accuracy:.5f}")
        
        # Save models
        models.append({
            'xgb': xgb_model,
            'lgb': lgb_model,
            'cat': cat_model,
            'features': X_train_combined.columns.tolist()
        })
    
    log_print(f"\nMean CV Score: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})")
    
    # Analyze feature importance
    log_print("\n" + "="*60)
    log_print("FEATURE IMPORTANCE ANALYSIS")
    log_print("="*60)
    
    # Average feature importance across folds
    all_features_list = models[0]['features']
    avg_importance = np.zeros(len(all_features_list))
    
    for model_dict in models:
        xgb_imp = model_dict['xgb'].feature_importances_
        lgb_imp = model_dict['lgb'].feature_importances_
        cat_imp = model_dict['cat'].feature_importances_
        avg_importance += (xgb_imp + lgb_imp + cat_imp) / 3
    
    avg_importance /= len(models)
    
    # Sort and display
    importance_df = pd.DataFrame({
        'feature': all_features_list,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    log_print("\nTop 20 Most Important Features:")
    for idx, row in importance_df.head(20).iterrows():
        log_print(f"{row['feature']:<40} {row['importance']:.4f}")
    
    return models, cv_scores, importance_df


def apply_to_test(test_df, models):
    """Apply the combined strategy to test data."""
    log_print("\n" + "="*60)
    log_print("APPLYING TO TEST DATA")
    log_print("="*60)
    
    # Create features
    test_df = create_null_features(test_df)
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                   'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    # Encode categorical
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in categorical_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})
    
    # Get all features
    all_features = models[0]['features'][:-1]  # Exclude ambivert_score, will add later
    X_test = test_df[all_features[:-1]]  # Temporary, without ambivert score
    
    # Impute test data
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_test_imputed = pd.DataFrame(
        imputer.fit_transform(X_test[feature_cols]),
        columns=feature_cols,
        index=X_test.index
    )
    
    # Combine features
    null_features = [col + '_null' for col in feature_cols]
    other_features = ['null_count', 'has_psych_nulls', 'has_social_nulls', 
                     'has_drained_null', 'has_stage_fear_null', 'no_nulls',
                     'complete_social_data', 'weighted_null_score', 'null_variance']
    
    X_test_combined = pd.concat([
        X_test_imputed,
        test_df[null_features + other_features]
    ], axis=1)
    
    # Add ambivert scores
    X_test_combined['ambivert_score'] = detect_ambiverts(X_test_combined)
    
    # Ensemble predictions
    all_probs = []
    
    for model_dict in models:
        xgb_prob = model_dict['xgb'].predict_proba(X_test_combined)[:, 1]
        lgb_prob = model_dict['lgb'].predict_proba(X_test_combined)[:, 1]
        cat_prob = model_dict['cat'].predict_proba(X_test_combined)[:, 1]
        
        ensemble_prob = 0.4 * xgb_prob + 0.3 * lgb_prob + 0.3 * cat_prob
        all_probs.append(ensemble_prob)
    
    # Average across folds
    final_prob = np.mean(all_probs, axis=0)
    
    # Create predictions
    predictions_df = pd.DataFrame({
        'id': test_df['id'],
        'prob': final_prob,
        'Personality': 'Extrovert'  # Default
    })
    
    # Apply rules
    # Rule 1: Strong null pattern
    strong_null_mask = (
        (test_df['has_drained_null'] == 1) & 
        (test_df['weighted_null_score'] > 2) &
        (final_prob > 0.3) & (final_prob < 0.6)
    )
    predictions_df.loc[strong_null_mask, 'Personality'] = 'Introvert'
    
    # Rule 2: Ambivert handling
    ambivert_mask = (
        (test_df['no_nulls'] == 1) &
        (X_test_combined['ambivert_score'] > 0.6) &
        (final_prob > 0.45) & (final_prob < 0.55)
    )
    # Use calibrated threshold for ambiverts
    predictions_df.loc[ambivert_mask, 'Personality'] = np.where(
        final_prob[ambivert_mask] > 0.475, 'Extrovert', 'Introvert'
    )
    
    # Rule 3: Standard threshold for others
    standard_mask = ~(strong_null_mask | ambivert_mask)
    predictions_df.loc[standard_mask, 'Personality'] = np.where(
        final_prob[standard_mask] > 0.5, 'Extrovert', 'Introvert'
    )
    
    # Summary statistics
    log_print(f"\nPrediction Summary:")
    log_print(f"Total predictions: {len(predictions_df)}")
    log_print(f"Introverts: {(predictions_df['Personality'] == 'Introvert').sum()} "
              f"({(predictions_df['Personality'] == 'Introvert').mean():.1%})")
    log_print(f"Extroverts: {(predictions_df['Personality'] == 'Extrovert').sum()} "
              f"({(predictions_df['Personality'] == 'Extrovert').mean():.1%})")
    
    log_print(f"\nRule Applications:")
    log_print(f"Strong null pattern rules: {strong_null_mask.sum()}")
    log_print(f"Ambivert handling rules: {ambivert_mask.sum()}")
    log_print(f"Standard predictions: {standard_mask.sum()}")
    
    # Analyze by null count
    log_print(f"\nPredictions by Null Count:")
    test_df['prediction'] = predictions_df['Personality']
    for null_count in range(5):
        mask = test_df['null_count'] == null_count
        if mask.sum() > 0:
            intro_rate = (test_df.loc[mask, 'prediction'] == 'Introvert').mean()
            log_print(f"  {null_count} nulls: {mask.sum()} records, {intro_rate:.1%} introverts")
    
    return predictions_df


def main():
    """Main execution function."""
    log_print("="*70)
    log_print("COMBINED NULL-AMBIVERT STRATEGY MODEL")
    log_print("="*70)
    
    # Load data
    log_print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    log_print(f"Train shape: {train_df.shape}")
    log_print(f"Test shape: {test_df.shape}")
    
    # Train model
    models, cv_scores, importance_df = train_combined_model(train_df)
    
    # Apply to test
    predictions = apply_to_test(test_df, models)
    
    # Save predictions
    timestamp = "20250705_1120"
    submission_path = f"output/combined_null_ambivert_submission_{timestamp}.csv"
    predictions[['id', 'Personality']].to_csv(submission_path, index=False)
    log_print(f"\nSubmission saved to: {submission_path}")
    
    # Save detailed results
    results = {
        'cv_scores': cv_scores,
        'mean_cv': float(np.mean(cv_scores)),
        'std_cv': float(np.std(cv_scores)),
        'feature_importance': importance_df.head(30).to_dict('records'),
        'prediction_distribution': {
            'introverts': int((predictions['Personality'] == 'Introvert').sum()),
            'extroverts': int((predictions['Personality'] == 'Extrovert').sum()),
            'intro_rate': float((predictions['Personality'] == 'Introvert').mean())
        }
    }
    
    with open('output/combined_null_ambivert_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log_print("\nResults saved to: output/combined_null_ambivert_results.json")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Feature importance
    ax1 = axes[0]
    top_features = importance_df.head(15)
    ax1.barh(range(len(top_features)), top_features['importance'])
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.set_xlabel('Importance')
    ax1.set_title('Top 15 Most Important Features')
    ax1.invert_yaxis()
    
    # CV scores
    ax2 = axes[1]
    ax2.plot(range(1, 6), cv_scores, 'bo-', markersize=10)
    ax2.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                label=f'Mean: {np.mean(cv_scores):.5f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Cross-Validation Scores')
    ax2.set_xticks(range(1, 6))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/combined_null_ambivert_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log_print("\nVisualization saved to: output/combined_null_ambivert_results.png")
    
    # Final summary
    log_print("\n" + "="*70)
    log_print("FINAL SUMMARY")
    log_print("="*70)
    log_print(f"\nMean CV Accuracy: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})")
    log_print(f"Expected Public LB: ~{np.mean(cv_scores)*0.995:.5f}")  # Conservative estimate
    
    if np.mean(cv_scores) > 0.975708:
        log_print("\n🎉 BREAKTHROUGH! We've exceeded the 0.975708 ceiling!")
    else:
        log_print(f"\nStill below ceiling. Gap: {0.975708 - np.mean(cv_scores):.5f}")
    
    log_print("\n" + "="*70)
    log_print("ANALYSIS COMPLETE")
    log_print("="*70)
    
    output_file.close()


if __name__ == "__main__":
    main()