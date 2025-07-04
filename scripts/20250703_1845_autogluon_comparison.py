#!/usr/bin/env python3
"""
Compare AutoGluon performance with numeric vs categorical features for S5E7 dataset.
AutoGluon automatically treats columns with <=25 unique values as categorical,
so we need to force numeric columns to have more unique values for fair comparison.

PURPOSE: Test AutoGluon's automatic feature type detection vs explicit type specification
HYPOTHESIS: AutoGluon's automatic detection might outperform forced numeric treatment
EXPECTED: Better performance when letting AutoGluon auto-detect categorical features
RESULT: Categorical treatment improved by ~0.2%, confirming AutoGluon's heuristics are effective
"""

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import time
import tempfile
import shutil
import warnings
warnings.filterwarnings('ignore')


def prepare_data_numeric(df, target_col, force_numeric_cols):
    """Prepare data forcing specified columns to be treated as numeric."""
    X = df.drop(columns=[target_col, 'id']).copy()
    y = df[target_col].copy()
    
    # For columns we want to force as numeric, add small random noise
    # to increase unique values above AutoGluon's threshold (25)
    for col in force_numeric_cols:
        if col in X.columns:
            # Add small random noise to make values continuous
            # This prevents AutoGluon from auto-converting to categorical
            noise = np.random.normal(0, 0.001, size=len(X))
            X[col] = X[col].astype(float) + noise
            print(f"  {col}: {X[col].nunique()} unique values (after adding noise)")
    
    # Combine X and y for AutoGluon
    data = X.copy()
    data[target_col] = y
    
    return data, force_numeric_cols


def prepare_data_categorical(df, target_col, categorical_cols):
    """Prepare data with specified columns as categorical."""
    X = df.drop(columns=[target_col, 'id']).copy()
    y = df[target_col].copy()
    
    # Convert specified columns to string to ensure categorical treatment
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
            print(f"  {col}: {X[col].nunique()} unique values (as string)")
    
    # Combine X and y for AutoGluon
    data = X.copy()
    data[target_col] = y
    
    return data, categorical_cols


def evaluate_autogluon(data, target_col, label, time_limit=300):
    """Evaluate AutoGluon with specified time limit."""
    
    # Create temporary directory for AutoGluon
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nTraining AutoGluon ({label})...")
        print(f"Time limit: {time_limit} seconds")
        
        start_time = time.time()
        
        # Train AutoGluon
        predictor = TabularPredictor(
            label=target_col,
            eval_metric='accuracy',
            path=tmpdir,
            verbosity=1  # Reduced verbosity
        )
        
        predictor.fit(
            data,
            time_limit=time_limit,
            presets='medium_quality',  # Faster preset
            num_bag_folds=3,  # 3-fold CV for speed
            num_bag_sets=1,
            num_stack_levels=0,  # No stacking for speed
            auto_stack=False,
            holdout_frac=0.2  # Use holdout instead of CV for speed
        )
        
        training_time = time.time() - start_time
        
        # Get leaderboard
        leaderboard = predictor.leaderboard(silent=True)
        
        # Get best model score
        best_score = leaderboard.iloc[0]['score_val']
        best_model = leaderboard.iloc[0]['model']
        
        print(f"\nTraining completed in {training_time:.1f}s")
        print(f"Best model: {best_model}")
        print(f"Validation accuracy: {best_score:.4f}")
        
        # Show top 5 models
        print("\nTop 5 models:")
        for i, row in leaderboard.head(5).iterrows():
            print(f"  {row['model']}: {row['score_val']:.4f}")
        
        # Get feature importance if available
        try:
            feature_importance = predictor.feature_importance(data)
            print(f"\nTop 5 important features:")
            for feat, imp in feature_importance.head(5).items():
                print(f"  {feat}: {imp:.3f}")
        except:
            pass
        
        return best_score, training_time, leaderboard


def main():
    # Load dataset
    print("Loading S5E7 dataset...")
    df = pd.read_csv("../../train.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Define columns with low cardinality
    low_cardinality_columns = [
        'Time_spent_Alone',
        'Social_event_attendance', 
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency'
    ]
    
    print(f"\nLow cardinality columns: {low_cardinality_columns}")
    
    # Check unique values
    print("\nOriginal unique values per column:")
    for col in low_cardinality_columns:
        if col in df.columns:
            n_unique = df[col].nunique()
            print(f"  {col}: {n_unique} unique values")
    
    # Test 1: Force as numeric (add noise to prevent auto-categorization)
    print("\n" + "="*60)
    print("TEST 1: Force low-cardinality features as NUMERIC")
    print("="*60)
    print("Adding noise to prevent AutoGluon's auto-categorization...")
    data_numeric, _ = prepare_data_numeric(df, 'Personality', low_cardinality_columns)
    score_numeric, time_numeric, leaderboard_numeric = evaluate_autogluon(
        data_numeric, 'Personality', 'numeric features', time_limit=120
    )
    
    # Test 2: Explicitly categorical
    print("\n" + "="*60)
    print("TEST 2: Low-cardinality features as CATEGORICAL")
    print("="*60)
    print("Converting to string to ensure categorical treatment...")
    data_categorical, _ = prepare_data_categorical(df, 'Personality', low_cardinality_columns)
    score_categorical, time_categorical, leaderboard_categorical = evaluate_autogluon(
        data_categorical, 'Personality', 'categorical features', time_limit=120
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Numeric features (forced):     {score_numeric:.4f} (time: {time_numeric:.1f}s)")
    print(f"Categorical features:          {score_categorical:.4f} (time: {time_categorical:.1f}s)")
    
    improvement = (score_categorical - score_numeric) / score_numeric * 100
    print(f"\nDifference: {improvement:+.2f}%")
    
    if improvement > 0.1:
        print("✅ Categorical encoding improved performance!")
    elif improvement < -0.1:
        print("❌ Numeric encoding performed better")
    else:
        print("➖ No significant difference")
    
    # Compare model selections
    print("\n" + "="*60)
    print("MODEL SELECTION COMPARISON")
    print("="*60)
    print("\nNumeric version - models used:")
    for model in leaderboard_numeric.head(3)['model'].values:
        print(f"  - {model}")
    
    print("\nCategorical version - models used:")
    for model in leaderboard_categorical.head(3)['model'].values:
        print(f"  - {model}")


if __name__ == "__main__":
    main()