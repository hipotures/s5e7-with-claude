#!/usr/bin/env python3
"""
Find cases where YDF predictions vary with seed
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")

def find_uncertain_cases():
    """Find test cases where YDF predictions vary"""
    
    print("="*60)
    print("FINDING YDF UNCERTAIN CASES")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    train_ydf = train_df[feature_cols + ['Personality']]
    test_ydf = test_df[feature_cols]
    
    # Collect predictions from many seeds
    n_seeds = 30
    all_probs = []
    
    print(f"\nCollecting predictions from {n_seeds} seeds...")
    for seed in range(42, 42 + n_seeds):
        if seed % 5 == 2:
            print(f"  Progress: {seed-42+1}/{n_seeds}")
        
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=300,
            max_depth=6,
            random_seed=seed
        )
        
        model = learner.train(train_ydf)
        preds = model.predict(test_ydf)
        probs = [float(str(p)) for p in preds]
        all_probs.append(probs)
    
    # Convert to numpy array
    prob_matrix = np.array(all_probs)  # shape: (n_seeds, n_samples)
    
    # Calculate statistics
    mean_probs = np.mean(prob_matrix, axis=0)
    std_probs = np.std(prob_matrix, axis=0)
    min_probs = np.min(prob_matrix, axis=0)
    max_probs = np.max(prob_matrix, axis=0)
    range_probs = max_probs - min_probs
    
    # Find cases with variation
    print("\n" + "="*40)
    print("ANALYSIS")
    print("="*40)
    
    # Cases with any variation
    varied_mask = range_probs > 0.0001
    print(f"\nCases with probability variation: {varied_mask.sum()} ({varied_mask.sum()/len(varied_mask)*100:.2f}%)")
    
    # Cases that flip between classes
    flips_mask = (min_probs < 0.5) & (max_probs > 0.5)
    print(f"Cases that flip between classes: {flips_mask.sum()}")
    
    if flips_mask.sum() > 0:
        print("\n🎯 CASES THAT FLIP (POTENTIAL AMBIVERTS):")
        flip_indices = np.where(flips_mask)[0]
        
        for idx in flip_indices:
            test_id = test_df.iloc[idx]['id']
            print(f"\nID {test_id}:")
            print(f"  Mean probability: {mean_probs[idx]:.4f}")
            print(f"  Std deviation: {std_probs[idx]:.4f}")
            print(f"  Range: [{min_probs[idx]:.4f}, {max_probs[idx]:.4f}]")
            print(f"  Features:")
            for col in feature_cols:
                val = test_df.iloc[idx][col]
                print(f"    {col}: {val}")
    
    # Top uncertain cases (high std)
    print("\n" + "="*40)
    print("TOP 10 MOST UNCERTAIN CASES")
    print("="*40)
    
    top_uncertain = np.argsort(std_probs)[-10:][::-1]
    
    for rank, idx in enumerate(top_uncertain):
        test_id = test_df.iloc[idx]['id']
        print(f"\n{rank+1}. ID {test_id}:")
        print(f"   Mean: {mean_probs[idx]:.4f}, Std: {std_probs[idx]:.4f}")
        print(f"   Range: [{min_probs[idx]:.4f}, {max_probs[idx]:.4f}]")
        
        # Count predictions
        binary_preds = (prob_matrix[:, idx] > 0.5).astype(int)
        intro_count = binary_preds.sum()
        extro_count = len(binary_preds) - intro_count
        print(f"   Votes: {intro_count} Introvert, {extro_count} Extrovert")
    
    # Save uncertain IDs
    uncertain_ids = test_df.iloc[top_uncertain]['id'].values
    print(f"\n\nMost uncertain IDs: {uncertain_ids}")
    
    return uncertain_ids

if __name__ == "__main__":
    uncertain_ids = find_uncertain_cases()