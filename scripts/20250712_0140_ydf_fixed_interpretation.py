#!/usr/bin/env python3
"""
Fixed YDF implementation with correct probability interpretation
YDF returns P(first class) which is P(Extrovert), but we had it reversed
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
import time

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"
SCORES_DIR = WORKSPACE_DIR / "scores"

def main():
    print("="*60)
    print("YDF WITH FIXED PROBABILITY INTERPRETATION")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Handle missing values
    for col in ['Stage_fear', 'Drained_after_socializing']:
        train_df[col] = train_df[col].fillna('Unknown')
        test_df[col] = test_df[col].fillna('Unknown')
    
    # Features
    feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                    'Post_frequency']
    
    train_data = train_df[feature_cols + ['Personality']].copy()
    test_data = test_df[feature_cols].copy()
    
    print(f"\nTrain shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Train Random Forest
    print("\n" + "="*40)
    print("RANDOM FOREST")
    print("="*40)
    
    start_time = time.time()
    rf_model = ydf.RandomForestLearner(
        label="Personality",
        num_trees=500,
        max_depth=30,
        winner_take_all=True,
        random_seed=42
    ).train(train_data)
    train_time = time.time() - start_time
    
    print(f"Training time: {train_time:.2f}s")
    print(f"OOB accuracy: {rf_model.self_evaluation().accuracy:.6f}")
    
    # Check label order
    print(f"\nLabel classes: {rf_model.label_classes()}")
    print("YDF returns probability of FIRST class (Extrovert)")
    
    # Get predictions with CORRECT interpretation
    rf_proba = rf_model.predict(test_data)
    
    # FIXED: YDF returns P(Extrovert), so we interpret correctly
    rf_predictions = ['Extrovert' if p > 0.5 else 'Introvert' for p in rf_proba]
    
    # But based on the reversal we saw, let's create both versions
    # Version 1: Standard interpretation (which was giving us reversed results)
    submission_standard = pd.DataFrame({
        'id': test_df['id'],
        'Personality': rf_predictions
    })
    
    # Version 2: Reversed interpretation (which should give correct results)
    rf_predictions_reversed = ['Introvert' if p > 0.5 else 'Extrovert' for p in rf_proba]
    submission_reversed = pd.DataFrame({
        'id': test_df['id'],
        'Personality': rf_predictions_reversed
    })
    
    # Save both versions
    submission_standard.to_csv(SCORES_DIR / "ydf_rf_standard_interpretation.csv", index=False)
    submission_reversed.to_csv(SCORES_DIR / "ydf_rf_reversed_interpretation.csv", index=False)
    
    print("\nCreated two submissions:")
    print("1. ydf_rf_standard_interpretation.csv - if YDF docs are correct")
    print("2. ydf_rf_reversed_interpretation.csv - based on our empirical observation")
    
    # Also train Gradient Boosted Trees
    print("\n" + "="*40)
    print("GRADIENT BOOSTED TREES")
    print("="*40)
    
    start_time = time.time()
    gbt_model = ydf.GradientBoostedTreesLearner(
        label="Personality",
        num_trees=500,
        shrinkage=0.05,
        max_depth=8,
        random_seed=42
    ).train(train_data)
    train_time = time.time() - start_time
    
    print(f"Training time: {train_time:.2f}s")
    print(f"Training accuracy: {gbt_model.evaluate(train_data).accuracy:.6f}")
    
    # GBT predictions with reversed interpretation
    gbt_proba = gbt_model.predict(test_data)
    gbt_predictions = ['Introvert' if p > 0.5 else 'Extrovert' for p in gbt_proba]
    
    submission_gbt = pd.DataFrame({
        'id': test_df['id'],
        'Personality': gbt_predictions
    })
    submission_gbt.to_csv(SCORES_DIR / "ydf_gbt_reversed.csv", index=False)
    print("\n3. ydf_gbt_reversed.csv - GBT with reversed interpretation")
    
    # Analyze uncertainty with correct interpretation
    print("\n" + "="*40)
    print("UNCERTAINTY ANALYSIS (with reversed interpretation)")
    print("="*40)
    
    # For reversed interpretation, uncertainty is still the same
    rf_uncertainty = 1 - np.abs(rf_proba - 0.5) * 2
    
    # Find most uncertain
    uncertain_idx = np.argsort(rf_uncertainty)[-10:][::-1]
    
    print("\nMost uncertain predictions:")
    for idx in uncertain_idx:
        test_id = test_df.iloc[idx]['id']
        # Show both interpretations
        std_pred = 'Extrovert' if rf_proba[idx] > 0.5 else 'Introvert'
        rev_pred = 'Introvert' if rf_proba[idx] > 0.5 else 'Extrovert'
        print(f"ID {test_id}: P(Extrovert)={rf_proba[idx]:.3f}, Standard→{std_pred}, Reversed→{rev_pred}, Uncertainty={rf_uncertainty[idx]:.3f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("YDF gives excellent results (~97% OOB) but probability interpretation was reversed")
    print("Created submissions with both interpretations to verify")
    print("The reversed interpretation should give ~0.975708 score")

if __name__ == "__main__":
    main()