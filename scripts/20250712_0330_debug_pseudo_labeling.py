#!/usr/bin/env python3
"""
Debug pseudo-labeling to understand why small changes cause huge prediction shifts
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")

def debug_pseudo_labeling():
    """Debug what happens when we add pseudo-labels"""
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print("="*60)
    print("DEBUG: PSEUDO-LABELING IMPACT")
    print("="*60)
    
    # Train baseline model
    print("\n1. BASELINE MODEL")
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        compute_oob_performances=True,
        random_seed=42
    )
    
    model1 = learner.train(train_df)
    
    # Get baseline predictions
    pred1 = model1.predict(test_df)
    pred1_classes = []
    pred1_proba = []
    
    for p in pred1:
        prob = float(str(p))
        pred1_proba.append(prob)
        pred1_classes.append('Extrovert' if prob > 0.5 else 'Introvert')
    
    # Count baseline distribution
    baseline_E = sum(1 for p in pred1_classes if p == 'Extrovert')
    baseline_I = len(pred1_classes) - baseline_E
    print(f"Baseline predictions: E={baseline_E}, I={baseline_I}")
    
    # Find high confidence predictions
    confidence = np.abs(np.array(pred1_proba) - 0.5) * 2
    high_conf_mask = confidence >= 0.995
    high_conf_indices = np.where(high_conf_mask)[0]
    
    print(f"High confidence (>=0.995): {len(high_conf_indices)} samples")
    
    # Take first 50 high confidence Introverts
    introvert_indices = [i for i in high_conf_indices if pred1_classes[i] == 'Introvert'][:50]
    
    print(f"\nAdding {len(introvert_indices)} high-confidence Introverts")
    
    # Create pseudo-labeled data
    pseudo_samples = []
    for idx in introvert_indices:
        sample = test_df.iloc[idx].to_dict()
        sample['Personality'] = 'Introvert'
        pseudo_samples.append(sample)
        print(f"  ID {sample['id']}: prob={pred1_proba[idx]:.4f}")
        if len(pseudo_samples) >= 5:
            print("  ...")
            break
    
    # Actually create all 50
    pseudo_samples = []
    for idx in introvert_indices:
        sample = test_df.iloc[idx].to_dict()
        sample['Personality'] = 'Introvert'
        pseudo_samples.append(sample)
    
    pseudo_df = pd.DataFrame(pseudo_samples)
    
    # Add to training
    train_with_pseudo = pd.concat([train_df, pseudo_df], ignore_index=True)
    
    print(f"\n2. MODEL WITH PSEUDO-LABELS")
    print(f"New training size: {len(train_with_pseudo)} (+{len(pseudo_df)})")
    
    # Train new model
    model2 = learner.train(train_with_pseudo)
    
    # Get new predictions
    pred2 = model2.predict(test_df)
    pred2_classes = []
    pred2_proba = []
    
    for p in pred2:
        prob = float(str(p))
        pred2_proba.append(prob)
        pred2_classes.append('Extrovert' if prob > 0.5 else 'Introvert')
    
    # Count new distribution
    new_E = sum(1 for p in pred2_classes if p == 'Extrovert')
    new_I = len(pred2_classes) - new_E
    print(f"New predictions: E={new_E}, I={new_I}")
    
    # Analyze changes
    n_changed = sum(1 for i in range(len(pred1_classes)) if pred1_classes[i] != pred2_classes[i])
    print(f"\n3. CHANGES")
    print(f"Predictions changed: {n_changed}/{len(test_df)} ({n_changed/len(test_df)*100:.1f}%)")
    
    # Which direction?
    E_to_I = sum(1 for i in range(len(pred1_classes)) 
                 if pred1_classes[i] == 'Extrovert' and pred2_classes[i] == 'Introvert')
    I_to_E = sum(1 for i in range(len(pred1_classes)) 
                 if pred1_classes[i] == 'Introvert' and pred2_classes[i] == 'Extrovert')
    
    print(f"  E→I: {E_to_I}")
    print(f"  I→E: {I_to_E}")
    
    # Check if pseudo-labeled samples changed their own predictions
    print(f"\n4. SELF-CONSISTENCY CHECK")
    pseudo_ids = set(pseudo_df['id'])
    self_consistent = 0
    self_changed = 0
    
    for i, test_id in enumerate(test_df['id']):
        if test_id in pseudo_ids:
            if pred2_classes[i] == 'Introvert':
                self_consistent += 1
            else:
                self_changed += 1
                print(f"  WARNING: ID {test_id} was pseudo-labeled as I but now predicts E!")
    
    print(f"Pseudo-labeled samples: {self_consistent} consistent, {self_changed} changed")
    
    # Analyze probability shifts
    print(f"\n5. PROBABILITY SHIFTS")
    prob_diffs = np.array(pred2_proba) - np.array(pred1_proba)
    
    print(f"Average probability shift: {np.mean(prob_diffs):.4f}")
    print(f"Samples shifted toward E (diff > 0.1): {sum(prob_diffs > 0.1)}")
    print(f"Samples shifted toward I (diff < -0.1): {sum(prob_diffs < -0.1)}")
    
    # Show biggest shifts
    biggest_shifts = np.argsort(np.abs(prob_diffs))[-10:]
    print("\nBiggest probability shifts:")
    for idx in biggest_shifts[::-1]:
        print(f"  ID {test_df.iloc[idx]['id']}: {pred1_proba[idx]:.3f} → {pred2_proba[idx]:.3f} "
              f"(diff={prob_diffs[idx]:+.3f}) [{pred1_classes[idx]}→{pred2_classes[idx]}]")

if __name__ == "__main__":
    debug_pseudo_labeling()