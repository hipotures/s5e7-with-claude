#!/usr/bin/env python3
"""
Balanced pseudo-labeling that maintains class proportions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def balanced_pseudo_labeling(train_df, test_df, n_iterations=10):
    """Pseudo-labeling that maintains train set class proportions"""
    
    # Calculate target proportions from train set
    train_proportions = train_df['Personality'].value_counts(normalize=True)
    target_E_ratio = train_proportions['Extrovert']
    target_I_ratio = train_proportions['Introvert']
    
    print("="*60)
    print("BALANCED PSEUDO-LABELING")
    print(f"Target proportions: E={target_E_ratio:.1%}, I={target_I_ratio:.1%}")
    print("="*60)
    
    results = []
    current_train = train_df.copy()
    all_pseudo_ids = set()
    
    # Lower thresholds for each class separately
    E_threshold = 0.99  # Lower threshold for E since we have many
    I_threshold = 0.90  # Much lower threshold for I since we have few
    
    SAMPLES_PER_ITER = 50
    
    for iteration in range(n_iterations):
        print(f"\n{'='*40}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*40}")
        
        # Train model
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=300,
            random_seed=42
        )
        model = learner.train(current_train)
        
        # Get predictions
        predictions = model.predict(test_df)
        
        proba_list = []
        pred_classes = []
        
        for pred in predictions:
            prob_I = float(str(pred))
            proba_list.append(prob_I)
            pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
        
        proba_array = np.array(proba_list)
        confidence = np.abs(proba_array - 0.5) * 2
        
        # Find high confidence samples for each class
        E_candidates = []
        I_candidates = []
        
        for i in range(len(test_df)):
            if test_df.iloc[i]['id'] not in all_pseudo_ids:
                if pred_classes[i] == 'Extrovert' and confidence[i] >= E_threshold:
                    E_candidates.append((i, confidence[i]))
                elif pred_classes[i] == 'Introvert' and confidence[i] >= I_threshold:
                    I_candidates.append((i, confidence[i]))
        
        # Sort by confidence
        E_candidates.sort(key=lambda x: x[1], reverse=True)
        I_candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Candidates: {len(E_candidates)} E (>={E_threshold}), {len(I_candidates)} I (>={I_threshold})")
        
        # Calculate how many of each to add
        n_E_to_add = int(SAMPLES_PER_ITER * target_E_ratio)
        n_I_to_add = SAMPLES_PER_ITER - n_E_to_add
        
        # But we might not have enough candidates
        n_E_actual = min(n_E_to_add, len(E_candidates))
        n_I_actual = min(n_I_to_add, len(I_candidates))
        
        if n_E_actual + n_I_actual == 0:
            # Try lowering thresholds
            E_threshold *= 0.95
            I_threshold *= 0.95
            print(f"No candidates found. Lowering thresholds to E>={E_threshold:.3f}, I>={I_threshold:.3f}")
            
            if E_threshold < 0.5 or I_threshold < 0.5:
                print("Thresholds too low. Stopping.")
                break
            continue
        
        # Add pseudo-labels
        pseudo_samples = []
        
        # Add E samples
        for idx, conf in E_candidates[:n_E_actual]:
            sample = test_df.iloc[idx].to_dict()
            sample['Personality'] = 'Extrovert'
            pseudo_samples.append(sample)
            all_pseudo_ids.add(sample['id'])
        
        # Add I samples  
        for idx, conf in I_candidates[:n_I_actual]:
            sample = test_df.iloc[idx].to_dict()
            sample['Personality'] = 'Introvert'
            pseudo_samples.append(sample)
            all_pseudo_ids.add(sample['id'])
        
        if len(pseudo_samples) > 0:
            pseudo_df = pd.DataFrame(pseudo_samples)
            print(f"\nAdding {len(pseudo_df)} samples:")
            print(f"  Extrovert: {n_E_actual} (target was {n_E_to_add})")
            print(f"  Introvert: {n_I_actual} (target was {n_I_to_add})")
            
            # Show confidence ranges
            if n_E_actual > 0:
                E_confs = [conf for idx, conf in E_candidates[:n_E_actual]]
                print(f"  E confidence: {min(E_confs):.3f} - {max(E_confs):.3f}")
            if n_I_actual > 0:
                I_confs = [conf for idx, conf in I_candidates[:n_I_actual]]
                print(f"  I confidence: {min(I_confs):.3f} - {max(I_confs):.3f}")
            
            current_train = pd.concat([current_train, pseudo_df], ignore_index=True)
            print(f"\nTotal training size: {len(current_train)} (+{len(all_pseudo_ids)})")
            
            # Check if we're maintaining proportions
            current_proportions = current_train['Personality'].value_counts(normalize=True)
            print(f"Current proportions: E={current_proportions['Extrovert']:.1%}, I={current_proportions['Introvert']:.1%}")
            
            # Store results
            results.append({
                'iteration': iteration + 1,
                'n_added_E': n_E_actual,
                'n_added_I': n_I_actual,
                'n_total': len(all_pseudo_ids),
                'predictions': pred_classes.copy(),
                'E_threshold': E_threshold,
                'I_threshold': I_threshold
            })
        
        if len(all_pseudo_ids) >= 500:
            print(f"\nReached 500 pseudo-labels. Stopping.")
            break
    
    return results, current_train

def main():
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"Original train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Run balanced pseudo-labeling
    results, final_train = balanced_pseudo_labeling(train_df, test_df)
    
    # Create submission
    if len(results) > 0:
        final_result = results[-1]
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': final_result['predictions']
        })
        
        filename = f'submission_balanced_pseudo_{final_result["n_total"]}_samples.csv'
        submission.to_csv(SCORES_DIR / filename, index=False)
        print(f"\nCreated: {filename}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        total_E = sum(r['n_added_E'] for r in results)
        total_I = sum(r['n_added_I'] for r in results)
        print(f"Total pseudo-labels added: {total_E + total_I}")
        print(f"  Extrovert: {total_E} ({total_E/(total_E+total_I)*100:.1f}%)")
        print(f"  Introvert: {total_I} ({total_I/(total_E+total_I)*100:.1f}%)")

if __name__ == "__main__":
    main()