#!/usr/bin/env python3
"""
SIMPLE BREAKTHROUGH TEST - FOCUSED ON KEY INSIGHT
================================================

Simplified implementation focusing on the core breakthrough:
properly handling the 2.43% ambiguous cases.

Author: Claude
Date: 2025-01-04
"""

# PURPOSE: Simplified implementation focusing solely on handling the 2.43% ambiguous cases
# HYPOTHESIS: A simple approach with proper ambiguous case handling can achieve breakthrough
# EXPECTED: Dynamic thresholds and the 96.2% rule will push accuracy above 97.57%
# RESULT: Three methods tested with ensemble approach and special case handling

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)

def preprocess_data(df):
    """Preprocess data - handle categorical columns."""
    df_processed = df.copy()
    
    # Handle categorical columns
    categorical_mappings = {
        'Stage_fear': {'No': 0, 'Yes': 1},
        'Drained_after_socializing': {'No': 0, 'Yes': 1}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)
    
    # Handle Target column if exists
    if 'Personality' in df_processed.columns:
        df_processed['Target'] = (df_processed['Personality'] == 'Extrovert').astype(int)
    
    return df_processed

def identify_ambiguous_simple(df):
    """Simple ambiguous case identification."""
    # Known marker values
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    
    # Check each marker
    has_marker = pd.Series([False] * len(df))
    for col, val in markers.items():
        if col in df.columns:
            has_marker |= (np.abs(df[col] - val) < 1e-6)
    
    # Additional pattern
    pattern1 = (
        (df['Time_spent_Alone'] < 2.5) & 
        (df['Social_event_attendance'] >= 3) & 
        (df['Social_event_attendance'] <= 4)
    )
    
    pattern2 = (
        (df['Friends_circle_size'] >= 6) & 
        (df['Friends_circle_size'] <= 7) &
        (df['Drained_after_socializing'] == 0)
    )
    
    return has_marker | pattern1 | pattern2

def create_simple_features(df):
    """Create basic features."""
    features = df.copy()
    
    # Basic interaction features
    features['social_vs_alone'] = (
        features['Social_event_attendance'] + features['Going_outside'] - 
        features['Time_spent_Alone']
    )
    
    features['introvert_score'] = (
        features['Time_spent_Alone'] * features['Drained_after_socializing'] + 
        features['Stage_fear']
    )
    
    features['extrovert_score'] = (
        features['Social_event_attendance'] + features['Going_outside'] + 
        features['Friends_circle_size'] / 3
    )
    
    # Marker count
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    
    features['marker_count'] = 0
    for col, val in markers.items():
        if col in features.columns:
            features['marker_count'] += (np.abs(features[col] - val) < 1e-6).astype(int)
    
    return features

def main():
    """Run simple breakthrough test."""
    print("="*60)
    print("SIMPLE BREAKTHROUGH TEST")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    # Preprocess
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    test_ids = test_df['id']
    
    # Identify ambiguous cases
    print("\nIdentifying ambiguous cases...")
    train_ambiguous = identify_ambiguous_simple(train_df)
    test_ambiguous = identify_ambiguous_simple(test_df)
    
    print(f"Training ambiguous: {train_ambiguous.sum()} ({train_ambiguous.mean():.2%})")
    print(f"Test ambiguous: {test_ambiguous.sum()} ({test_ambiguous.mean():.2%})")
    
    # Check ambiguous distribution in training
    if train_ambiguous.sum() > 0:
        ambig_extrovert_rate = train_df[train_ambiguous]['Target'].mean()
        print(f"Ambiguous cases: {ambig_extrovert_rate:.1%} are Extrovert")
    
    # Create features
    print("\nCreating features...")
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                    'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    train_features = create_simple_features(train_df[feature_cols])
    test_features = create_simple_features(test_df[feature_cols])
    
    X = train_features.values
    y = train_df['Target'].values
    X_test = test_features.values
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # METHOD 1: Standard XGBoost
    print("\n" + "-"*40)
    print("METHOD 1: Standard XGBoost")
    print("-"*40)
    
    model1 = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        tree_method='gpu_hist'
    )
    
    model1.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              early_stopping_rounds=50, 
              verbose=False)
    
    pred1 = model1.predict(X_val)
    acc1 = accuracy_score(y_val, pred1)
    print(f"Validation accuracy: {acc1:.6f}")
    
    # METHOD 2: Weighted training on ambiguous
    print("\n" + "-"*40)
    print("METHOD 2: Weighted Training (10x on ambiguous)")
    print("-"*40)
    
    # Create weights
    train_ambiguous_mask = identify_ambiguous_simple(
        pd.DataFrame(X_train, columns=train_features.columns)
    )
    weights = np.ones(len(y_train))
    weights[train_ambiguous_mask] = 10.0
    
    model2 = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        tree_method='gpu_hist'
    )
    
    model2.fit(X_train, y_train, 
              sample_weight=weights,
              eval_set=[(X_val, y_val)], 
              early_stopping_rounds=50, 
              verbose=False)
    
    pred2 = model2.predict(X_val)
    acc2 = accuracy_score(y_val, pred2)
    print(f"Validation accuracy: {acc2:.6f}")
    
    # METHOD 3: Post-processing rules
    print("\n" + "-"*40)
    print("METHOD 3: Dynamic Thresholds + 96.2% Rule")
    print("-"*40)
    
    # Get probabilities
    proba1 = model1.predict_proba(X_val)[:, 1]
    proba2 = model2.predict_proba(X_val)[:, 1]
    ensemble_proba = (proba1 + proba2) / 2
    
    # Apply dynamic thresholds
    val_ambiguous = identify_ambiguous_simple(
        pd.DataFrame(X_val, columns=train_features.columns)
    )
    
    pred3 = np.zeros(len(y_val))
    for i in range(len(y_val)):
        if val_ambiguous.iloc[i]:
            # Ambiguous case
            if abs(ensemble_proba[i] - 0.5) < 0.1:
                # Very uncertain - apply 96.2% rule
                pred3[i] = 1
            else:
                # Use lower threshold
                pred3[i] = int(ensemble_proba[i] > 0.45)
        else:
            # Clear case
            pred3[i] = int(ensemble_proba[i] > 0.5)
    
    acc3 = accuracy_score(y_val, pred3)
    print(f"Validation accuracy: {acc3:.6f}")
    
    # Make final predictions on test set
    print("\n" + "="*40)
    print("FINAL TEST PREDICTIONS")
    print("="*40)
    
    # Get test probabilities
    test_proba1 = model1.predict_proba(X_test)[:, 1]
    test_proba2 = model2.predict_proba(X_test)[:, 1]
    test_ensemble_proba = (test_proba1 + test_proba2) / 2
    
    # Apply best method
    final_predictions = np.zeros(len(X_test))
    
    special_cases = 0
    for i in range(len(X_test)):
        if test_ambiguous.iloc[i]:
            # Ambiguous case
            if abs(test_ensemble_proba[i] - 0.5) < 0.1:
                # Very uncertain - apply 96.2% rule
                final_predictions[i] = 1
                special_cases += 1
            elif test_ensemble_proba[i] < 0.25 and test_features.iloc[i]['marker_count'] >= 2:
                # Exception: multiple markers + very low prob = rare introvert
                final_predictions[i] = 0
            else:
                # Use lower threshold
                final_predictions[i] = int(test_ensemble_proba[i] > 0.45)
        else:
            # Clear case
            final_predictions[i] = int(test_ensemble_proba[i] > 0.5)
    
    print(f"\nApplied special handling to {special_cases} cases")
    print(f"Prediction distribution:")
    print(f"  Introverts: {(final_predictions == 0).sum()} ({(final_predictions == 0).mean():.2%})")
    print(f"  Extroverts: {(final_predictions == 1).sum()} ({(final_predictions == 1).mean():.2%})")
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_ids,
        'Target': final_predictions.astype(int)
    })
    
    submission.to_csv('submission_simple_breakthrough.csv', index=False)
    print(f"\nSubmission saved to: submission_simple_breakthrough.csv")
    
    # Analysis of ambiguous handling
    print("\n" + "="*40)
    print("AMBIGUOUS CASE ANALYSIS")
    print("="*40)
    
    if test_ambiguous.sum() > 0:
        ambig_indices = test_ambiguous[test_ambiguous].index[:10]
        print("\nFirst 10 ambiguous test cases:")
        for idx in ambig_indices:
            i = idx  # Direct index
            prob = test_ensemble_proba[i]
            pred = final_predictions[i]
            markers = test_features.iloc[i]['marker_count']
            print(f"  Index {i}: prob={prob:.4f}, pred={pred}, markers={markers}")

if __name__ == "__main__":
    main()