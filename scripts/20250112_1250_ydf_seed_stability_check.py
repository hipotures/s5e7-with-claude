#!/usr/bin/env python3
"""
Quick check of YDF seed stability
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def check_ydf_stability():
    """Check if YDF gives same predictions with different seeds"""
    
    print("="*60)
    print("YDF SEED STABILITY CHECK")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    train_ydf = train_df[feature_cols + ['Personality']]
    test_ydf = test_df[feature_cols]
    
    # Test with different seeds
    seeds = [42, 123, 456, 789, 999]
    predictions = []
    
    for seed in seeds:
        print(f"\nTesting seed {seed}...")
        
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=300,
            max_depth=6,
            random_seed=seed
        )
        
        model = learner.train(train_ydf)
        preds = model.predict(test_ydf)
        
        # Convert to binary
        binary_preds = ['Introvert' if float(str(p)) > 0.5 else 'Extrovert' for p in preds]
        predictions.append(binary_preds)
    
    # Check agreement
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    
    # Compare all predictions
    all_same = True
    for i in range(1, len(predictions)):
        diff = sum(1 for a, b in zip(predictions[0], predictions[i]) if a != b)
        print(f"Seed {seeds[0]} vs Seed {seeds[i]}: {diff} differences")
        if diff > 0:
            all_same = False
    
    if all_same:
        print("\n✅ YDF is PERFECTLY STABLE - same predictions regardless of seed!")
        print("This means:")
        print("1. No randomness in predictions")
        print("2. Multi-seed ensemble won't help")
        print("3. Focus on other strategies")
    else:
        print("\n❌ YDF predictions vary with seed")
        print("Multi-seed ensemble might help!")
    
    # Check if probabilities are also identical
    print("\n" + "="*40)
    print("CHECKING PROBABILITY STABILITY")
    print("="*40)
    
    prob_predictions = []
    for seed in seeds[:3]:  # Check first 3
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=300,
            max_depth=6,
            random_seed=seed
        )
        
        model = learner.train(train_ydf)
        preds = model.predict(test_ydf)
        probs = [float(str(p)) for p in preds]
        prob_predictions.append(probs)
    
    # Check probability differences
    max_diff = 0
    for i in range(len(prob_predictions[0])):
        probs_at_i = [p[i] for p in prob_predictions]
        diff = max(probs_at_i) - min(probs_at_i)
        max_diff = max(max_diff, diff)
    
    print(f"Maximum probability difference across seeds: {max_diff:.6f}")
    
    if max_diff < 0.0001:
        print("✅ Even probabilities are nearly identical!")
    else:
        print("❌ Probabilities vary between seeds")

if __name__ == "__main__":
    check_ydf_stability()