#!/usr/bin/env python3
"""
Simplified test of Yggdrasil Decision Forests on S5E7 dataset
Focus on getting basic results without complex feature importance analysis
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
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("="*60)
    print("YGGDRASIL DECISION FORESTS - SIMPLE TEST")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Handle missing values in categorical columns
    for col in ['Stage_fear', 'Drained_after_socializing']:
        train_df[col] = train_df[col].fillna('Unknown')
        test_df[col] = test_df[col].fillna('Unknown')
    
    # Keep only features and target
    feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                    'Post_frequency']
    
    train_data = train_df[feature_cols + ['Personality']].copy()
    test_data = test_df[feature_cols].copy()
    
    print(f"\nTrain shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Test 1: Random Forest
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
    
    # Evaluate
    evaluation = rf_model.evaluate(train_data)
    print(f"Training time: {train_time:.2f}s")
    print(f"Training accuracy: {evaluation.accuracy:.6f}")
    print(f"OOB accuracy: {rf_model.self_evaluation().accuracy:.6f}")
    
    # Get predictions - YDF returns probabilities for first class (Extrovert)
    rf_proba = rf_model.predict(test_data)
    rf_predictions = ['Extrovert' if p > 0.5 else 'Introvert' for p in rf_proba]
    
    # Test 2: Gradient Boosted Trees
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
    
    evaluation = gbt_model.evaluate(train_data)
    print(f"Training time: {train_time:.2f}s")
    print(f"Training accuracy: {evaluation.accuracy:.6f}")
    
    # Get predictions - YDF returns probabilities for first class (Extrovert)
    gbt_proba = gbt_model.predict(test_data)
    gbt_predictions = ['Extrovert' if p > 0.5 else 'Introvert' for p in gbt_proba]
    
    # Compare predictions
    print("\n" + "="*40)
    print("PREDICTION COMPARISON")
    print("="*40)
    
    agreement = sum(rf == gbt for rf, gbt in zip(rf_predictions, gbt_predictions)) / len(rf_predictions)
    print(f"Model agreement: {agreement:.2%}")
    
    # Find disagreements between models
    disagreements = [rf != gbt for rf, gbt in zip(rf_predictions, gbt_predictions)]
    disagreement_indices = np.where(disagreements)[0]
    
    print(f"\nNumber of disagreements: {len(disagreement_indices)}")
    if len(disagreement_indices) > 0:
        print("\nFirst 20 disagreements:")
        for idx in disagreement_indices[:20]:
            test_id = test_df.iloc[idx]['id']
            print(f"ID {test_id}: RF={rf_predictions[idx]}, GBT={gbt_predictions[idx]}")
    
    # Create submissions
    print("\n" + "="*40)
    print("CREATING SUBMISSIONS")
    print("="*40)
    
    # RF submission
    rf_submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': rf_predictions
    })
    rf_submission.to_csv(WORKSPACE_DIR / "scores" / "ydf_random_forest_500trees.csv", index=False)
    print("Created: ydf_random_forest_500trees.csv")
    
    # GBT submission
    gbt_submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': gbt_predictions
    })
    gbt_submission.to_csv(WORKSPACE_DIR / "scores" / "ydf_gradient_boosted_500trees.csv", index=False)
    print("Created: ydf_gradient_boosted_500trees.csv")
    
    # Find most uncertain predictions (close to 0.5)
    rf_uncertainty = 1 - np.abs(rf_proba - 0.5) * 2
    gbt_uncertainty = 1 - np.abs(gbt_proba - 0.5) * 2
    avg_uncertainty = (rf_uncertainty + gbt_uncertainty) / 2
    
    # Print most uncertain
    uncertain_idx = np.argsort(avg_uncertainty)[-10:][::-1]
    print("\n" + "="*40)
    print("MOST UNCERTAIN PREDICTIONS")
    print("="*40)
    for idx in uncertain_idx:
        test_id = test_df.iloc[idx]['id']
        print(f"ID {test_id}: RF={rf_proba[idx]:.3f}, GBT={gbt_proba[idx]:.3f}, Avg uncertainty={avg_uncertainty[idx]:.3f}")
    
    # Save analysis
    analysis_df = pd.DataFrame({
        'id': test_df['id'],
        'rf_pred': rf_predictions,
        'gbt_pred': gbt_predictions,
        'disagreement': disagreements
    })
    analysis_df.to_csv(OUTPUT_DIR / 'ydf_predictions_analysis.csv', index=False)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Random Forest OOB Accuracy: {rf_model.self_evaluation().accuracy:.6f}")
    print(f"GBT Training Accuracy: {gbt_model.evaluate(train_data).accuracy:.6f}")
    print(f"Model Agreement: {agreement:.2%}")
    print("\nYDF Advantages:")
    print("- Very fast training (< 1s for both models)")
    print("- Native missing value handling")
    print("- Built-in OOB evaluation")
    print("- No preprocessing needed")

if __name__ == "__main__":
    main()