#!/usr/bin/env python3
"""
Conservative Pseudo-Labeling with more iterations and higher thresholds
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def load_and_prepare_data():
    """Load and prepare train/test data"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"Original train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df

def train_model_and_predict(train_df, test_df):
    """Train YDF model and get predictions with confidence"""
    # Train YDF model
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        compute_oob_performances=True,
        random_seed=42
    )
    
    model = learner.train(train_df)
    
    # Make predictions on test set
    predictions = model.predict(test_df)
    
    # Convert to probabilities and classes
    test_proba = []
    test_pred = []
    
    for pred in predictions:
        prob = float(str(pred))
        test_proba.append(prob)
        # YDF returns P(Introvert), not P(Extrovert)!
        test_pred.append('Introvert' if prob > 0.5 else 'Extrovert')
    
    return model, test_pred, np.array(test_proba)

def conservative_pseudo_labeling(train_df, test_df, n_iterations=20):
    """Conservative pseudo-labeling with fixed high threshold"""
    
    results = []
    CONFIDENCE_THRESHOLD = 0.995  # Very high fixed threshold
    MAX_SAMPLES_PER_ITER = 50    # Smaller batches to see finer oscillations
    
    print("\n" + "="*60)
    print("CONSERVATIVE PSEUDO-LABELING")
    print(f"Fixed threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Max samples per iteration: {MAX_SAMPLES_PER_ITER}")
    print("="*60)
    
    # Keep track of all added IDs
    original_train_size = len(train_df)
    current_train = train_df.copy()
    all_pseudo_ids = set()
    
    for iteration in range(n_iterations):
        print(f"\n{'='*40}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*40}")
        
        # Train model
        model, test_pred, test_proba = train_model_and_predict(current_train, test_df)
        
        # Calculate confidence
        confidence = np.abs(test_proba - 0.5) * 2
        
        # Find high confidence samples not yet added
        high_conf_indices = []
        for i in range(len(test_df)):
            if confidence[i] >= CONFIDENCE_THRESHOLD and test_df.iloc[i]['id'] not in all_pseudo_ids:
                high_conf_indices.append(i)
        
        print(f"\nFound {len(high_conf_indices)} new samples with confidence >= {CONFIDENCE_THRESHOLD}")
        
        # Limit number of samples
        if len(high_conf_indices) > MAX_SAMPLES_PER_ITER:
            # Sort by confidence and take top ones
            conf_values = [(i, confidence[i]) for i in high_conf_indices]
            conf_values.sort(key=lambda x: x[1], reverse=True)
            high_conf_indices = [x[0] for x in conf_values[:MAX_SAMPLES_PER_ITER]]
            print(f"Limited to top {MAX_SAMPLES_PER_ITER} samples")
        
        if len(high_conf_indices) == 0:
            print("No new high-confidence samples found. Stopping.")
            break
        
        # Create pseudo-labeled samples
        pseudo_samples = []
        for idx in high_conf_indices:
            sample = test_df.iloc[idx].to_dict()
            sample['Personality'] = test_pred[idx]
            pseudo_samples.append(sample)
            all_pseudo_ids.add(sample['id'])
        
        pseudo_df = pd.DataFrame(pseudo_samples)
        
        # Analyze what we're adding
        pseudo_dist = pseudo_df['Personality'].value_counts()
        print(f"\nAdding {len(pseudo_df)} samples:")
        print(f"  Extrovert: {pseudo_dist.get('Extrovert', 0)}")
        print(f"  Introvert: {pseudo_dist.get('Introvert', 0)}")
        
        # Show some examples
        print("\nSample additions (first 5):")
        for i, (idx, row) in enumerate(pseudo_df.head().iterrows()):
            conf_val = confidence[high_conf_indices[i]]
            prob_val = test_proba[high_conf_indices[i]]
            print(f"  ID {row['id']}: {row['Personality']} (prob={prob_val:.4f}, conf={conf_val:.4f})")
        
        # Add to training set
        current_train = pd.concat([current_train, pseudo_df], ignore_index=True)
        print(f"\nTotal training size: {len(current_train)} (+{len(current_train) - original_train_size})")
        
        # Store results
        result = {
            'iteration': iteration + 1,
            'n_added': len(pseudo_df),
            'n_total': len(all_pseudo_ids),
            'predictions': test_pred.copy(),
            'probabilities': test_proba.copy(),
            'confidence': confidence.copy(),
            'added_ids': list(pseudo_df['id'])
        }
        results.append(result)
        
        # Stop if we've added enough
        if len(all_pseudo_ids) >= 1000:
            print(f"\nReached 1000 pseudo-labels. Stopping.")
            break
    
    return results, current_train, all_pseudo_ids

def visualize_oscillations(results):
    """Visualize the oscillation pattern"""
    iterations = []
    extrovert_counts = []
    introvert_counts = []
    total_changes = []
    
    for i, result in enumerate(results):
        iterations.append(i + 1)
        # Count E/I in added samples
        added_ids = result['added_ids']
        if len(added_ids) > 0:
            # Get predictions for added samples
            test_df = pd.read_csv(DATA_DIR / "test.csv")
            id_to_pred = dict(zip(test_df['id'], result['predictions']))
            
            e_count = sum(1 for id in added_ids if id_to_pred[id] == 'Extrovert')
            i_count = len(added_ids) - e_count
            
            extrovert_counts.append(e_count)
            introvert_counts.append(i_count)
        else:
            extrovert_counts.append(0)
            introvert_counts.append(0)
        
        # Calculate changes
        if i > 0:
            prev_pred = results[i-1]['predictions']
            curr_pred = result['predictions']
            n_changed = sum([1 for j in range(len(prev_pred)) if prev_pred[j] != curr_pred[j]])
            total_changes.append(n_changed)
        else:
            total_changes.append(0)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot E/I additions
    ax1.plot(iterations, extrovert_counts, 'r-o', label='Extroverts added')
    ax1.plot(iterations, introvert_counts, 'b-o', label='Introverts added')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Count')
    ax1.set_title('Pseudo-labels Added per Iteration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot prediction changes
    ax2.plot(iterations[1:], total_changes[1:], 'g-o')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Predictions Changed')
    ax2.set_title('Prediction Changes Between Iterations')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pseudo_labeling_oscillations.png', dpi=300)
    plt.close()
    
    print("\nSaved oscillation visualization")

def analyze_results(results, test_df):
    """Analyze the pseudo-labeling results"""
    print("\n" + "="*60)
    print("ANALYSIS OF PSEUDO-LABELING")
    print("="*60)
    
    # Track prediction changes
    if len(results) >= 2:
        total_changes = 0
        for i in range(1, len(results)):
            prev_pred = results[i-1]['predictions']
            curr_pred = results[i]['predictions']
            n_changed = sum([1 for j in range(len(prev_pred)) if prev_pred[j] != curr_pred[j]])
            total_changes += n_changed
            
            print(f"\nIteration {i} → {i+1}:")
            print(f"  Added: {results[i]['n_added']} samples")
            print(f"  Predictions changed: {n_changed} ({n_changed/len(prev_pred)*100:.1f}%)")
    
    # Show final statistics
    final_result = results[-1]
    print(f"\nFinal statistics:")
    print(f"  Total iterations: {len(results)}")
    print(f"  Total pseudo-labels: {final_result['n_total']}")
    print(f"  Average per iteration: {final_result['n_total']/len(results):.1f}")
    
    # Confidence distribution in final iteration
    final_conf = final_result['confidence']
    print(f"\nFinal confidence distribution:")
    for threshold in [0.99, 0.95, 0.90, 0.85, 0.80]:
        count = sum(final_conf >= threshold)
        print(f"  >= {threshold}: {count} ({count/len(final_conf)*100:.1f}%)")
    
    # Save all added IDs for inspection
    all_added_ids = []
    for result in results:
        all_added_ids.extend(result['added_ids'])
    
    pseudo_summary = pd.DataFrame({
        'id': all_added_ids,
        'iteration': [i+1 for i, r in enumerate(results) for _ in r['added_ids']]
    })
    pseudo_summary.to_csv(OUTPUT_DIR / 'pseudo_labeling_added_ids.csv', index=False)
    print(f"\nSaved list of all {len(all_added_ids)} added IDs")

def create_submissions(results, test_df):
    """Create submission files"""
    print("\n" + "="*60)
    print("CREATING SUBMISSIONS")
    print("="*60)
    
    # Final iteration submission
    final_pred = results[-1]['predictions']
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': final_pred
    })
    
    filename = f'submission_pseudo_conservative_{results[-1]["n_total"]}_samples.csv'
    submission.to_csv(SCORES_DIR / filename, index=False)
    print(f"Created: {filename}")
    
    # Also create intermediate submissions for testing
    if len(results) > 3:
        # Submission after ~half the iterations
        mid_idx = len(results) // 2
        mid_submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': results[mid_idx]['predictions']
        })
        mid_filename = f'submission_pseudo_conservative_iter{mid_idx+1}_{results[mid_idx]["n_total"]}_samples.csv'
        mid_submission.to_csv(SCORES_DIR / mid_filename, index=False)
        print(f"Created: {mid_filename} (intermediate)")

def main():
    # Load data
    train_df, test_df = load_and_prepare_data()
    
    # Perform conservative pseudo-labeling
    results, final_train, all_pseudo_ids = conservative_pseudo_labeling(
        train_df, test_df, n_iterations=20
    )
    
    # Analyze results
    analyze_results(results, test_df)
    
    # Visualize oscillations
    visualize_oscillations(results)
    
    # Create submissions
    create_submissions(results, test_df)
    
    print("\n" + "="*60)
    print("CONSERVATIVE PSEUDO-LABELING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()