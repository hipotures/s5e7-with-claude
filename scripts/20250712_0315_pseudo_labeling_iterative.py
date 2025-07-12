#!/usr/bin/env python3
"""
Pseudo-Labeling Iterative Approach
Use high-confidence predictions on test set as pseudo-labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
from sklearn.preprocessing import StandardScaler
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

def train_initial_model(train_df, test_df):
    """Train initial YDF model and get predictions with confidence"""
    print("\n" + "="*60)
    print("TRAINING INITIAL MODEL")
    print("="*60)
    
    # Train YDF model
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        compute_oob_performances=True,
        random_seed=42
    )
    
    model = learner.train(train_df)
    
    # Get OOB performance
    if hasattr(model, 'out_of_bag_evaluation'):
        oob_eval = model.out_of_bag_evaluation()
        if oob_eval and hasattr(oob_eval, 'accuracy'):
            print(f"Initial OOB accuracy: {oob_eval.accuracy:.6f}")
    
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
    
    return model, test_pred, test_proba

def select_high_confidence_samples(test_df, test_pred, test_proba, confidence_threshold=0.95):
    """Select test samples with high confidence predictions"""
    
    # Calculate confidence (distance from 0.5)
    confidence = np.abs(np.array(test_proba) - 0.5) * 2
    
    # Select high confidence samples
    high_conf_mask = confidence >= confidence_threshold
    n_selected = sum(high_conf_mask)
    
    print(f"\nSelecting samples with confidence >= {confidence_threshold}")
    print(f"Selected: {n_selected}/{len(test_df)} ({n_selected/len(test_df)*100:.1f}%)")
    
    # Debug: show confidence distribution
    print(f"Confidence distribution:")
    print(f"  >= 0.99: {sum(confidence >= 0.99)}")
    print(f"  >= 0.95: {sum(confidence >= 0.95)}")
    print(f"  >= 0.90: {sum(confidence >= 0.90)}")
    print(f"  >= 0.85: {sum(confidence >= 0.85)}")
    print(f"  >= 0.80: {sum(confidence >= 0.80)}")
    
    # Create pseudo-labeled dataset
    pseudo_df = test_df[high_conf_mask].copy()
    pseudo_df['Personality'] = [test_pred[i] for i in range(len(test_pred)) if high_conf_mask[i]]
    
    # Store confidence separately for analysis but don't include in training data
    pseudo_confidence = confidence[high_conf_mask]
    
    # Analyze distribution
    print(f"\nPseudo-label distribution:")
    print(pseudo_df['Personality'].value_counts())
    
    return pseudo_df, confidence, pseudo_confidence

def iterative_pseudo_labeling(train_df, test_df, n_iterations=5):
    """Perform iterative pseudo-labeling"""
    
    results = []
    confidence_thresholds = [0.99, 0.95, 0.90, 0.85, 0.80]
    
    print("\n" + "="*60)
    print("ITERATIVE PSEUDO-LABELING")
    print("="*60)
    
    # Keep track of original train size
    original_train_size = len(train_df)
    current_train = train_df.copy()
    
    for iteration in range(n_iterations):
        print(f"\n{'='*40}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*40}")
        
        # Train model
        model, test_pred, test_proba = train_initial_model(current_train, test_df)
        
        # Select confidence threshold for this iteration
        conf_threshold = confidence_thresholds[min(iteration, len(confidence_thresholds)-1)]
        
        # Select high confidence samples
        pseudo_df, confidence, pseudo_confidence = select_high_confidence_samples(
            test_df, test_pred, test_proba, conf_threshold
        )
        
        # Check if we have new samples to add
        if 'id' in current_train.columns and 'id' in pseudo_df.columns:
            # Remove already added samples
            existing_ids = set(current_train['id'])
            new_pseudo = pseudo_df[~pseudo_df['id'].isin(existing_ids)]
            n_new = len(new_pseudo)
        else:
            new_pseudo = pseudo_df
            n_new = len(new_pseudo)
        
        if n_new == 0:
            print("No new high-confidence samples to add. Stopping.")
            break
        
        # Add pseudo-labeled data to training set
        print(f"\nAdding {n_new} new pseudo-labeled samples")
        current_train = pd.concat([current_train, new_pseudo], ignore_index=True)
        print(f"New training size: {len(current_train)} (+{len(current_train) - original_train_size})")
        
        # Store results
        result = {
            'iteration': iteration + 1,
            'confidence_threshold': conf_threshold,
            'n_pseudo_total': len(current_train) - original_train_size,
            'n_pseudo_new': n_new,
            'train_size': len(current_train),
            'predictions': test_pred.copy(),
            'probabilities': test_proba.copy(),
            'confidence': confidence.copy()
        }
        results.append(result)
        
        # Early stopping if we've added too many
        if len(current_train) > original_train_size * 1.5:
            print("\nReached 50% increase in training size. Stopping.")
            break
    
    return results, current_train

def analyze_prediction_changes(results):
    """Analyze how predictions change across iterations"""
    print("\n" + "="*60)
    print("PREDICTION STABILITY ANALYSIS")
    print("="*60)
    
    if len(results) < 2:
        print("Need at least 2 iterations for comparison")
        return
    
    # Compare consecutive iterations
    for i in range(1, len(results)):
        prev_pred = results[i-1]['predictions']
        curr_pred = results[i]['predictions']
        
        n_changed = sum([1 for j in range(len(prev_pred)) if prev_pred[j] != curr_pred[j]])
        
        print(f"\nIteration {i} → {i+1}:")
        print(f"  Predictions changed: {n_changed} ({n_changed/len(prev_pred)*100:.1f}%)")
        print(f"  Pseudo-labels added: {results[i]['n_pseudo_new']}")
        print(f"  Total pseudo-labels: {results[i]['n_pseudo_total']}")
    
    # Find most unstable predictions
    n_samples = len(results[0]['predictions'])
    stability_count = np.zeros(n_samples)
    
    for i in range(1, len(results)):
        for j in range(n_samples):
            if results[i-1]['predictions'][j] != results[i]['predictions'][j]:
                stability_count[j] += 1
    
    # Most unstable samples
    unstable_indices = np.where(stability_count > 0)[0]
    print(f"\n{len(unstable_indices)} samples changed prediction at least once")
    
    if len(unstable_indices) > 0:
        # Save unstable samples for analysis
        test_df = pd.read_csv(DATA_DIR / "test.csv")
        unstable_df = test_df.iloc[unstable_indices].copy()
        unstable_df['n_changes'] = stability_count[unstable_indices]
        unstable_df = unstable_df.sort_values('n_changes', ascending=False)
        
        unstable_df.to_csv(OUTPUT_DIR / 'pseudo_labeling_unstable_samples.csv', index=False)
        print(f"\nTop 10 most unstable IDs:")
        for _, row in unstable_df.head(10).iterrows():
            print(f"  ID {row['id']}: changed {int(row['n_changes'])} times")

def create_submissions(results, test_df):
    """Create submissions from different iterations"""
    print("\n" + "="*60)
    print("CREATING SUBMISSIONS")
    print("="*60)
    
    # Best iteration (last one)
    best_result = results[-1]
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': best_result['predictions']
    })
    
    filename = f'submission_pseudo_label_iter{len(results)}.csv'
    submission.to_csv(SCORES_DIR / filename, index=False)
    print(f"\nCreated: {filename}")
    print(f"  Used {best_result['n_pseudo_total']} pseudo-labeled samples")
    
    # Also save first iteration for comparison
    if len(results) > 1:
        first_submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': results[0]['predictions']
        })
        first_filename = 'submission_pseudo_label_iter1.csv'
        first_submission.to_csv(SCORES_DIR / first_filename, index=False)
        print(f"\nCreated: {first_filename} (for comparison)")
    
    # Save high confidence predictions (confidence > 0.99)
    final_conf = results[-1]['confidence']
    very_high_conf = final_conf > 0.99
    n_very_high = sum(very_high_conf)
    
    print(f"\n{n_very_high} predictions with confidence > 0.99 ({n_very_high/len(test_df)*100:.1f}%)")

def visualize_confidence_distribution(results):
    """Visualize confidence distribution across iterations"""
    plt.figure(figsize=(12, 6))
    
    # Plot confidence distribution for each iteration
    for i, result in enumerate(results):
        plt.subplot(1, 2, 1)
        confidence = result['confidence']
        plt.hist(confidence, bins=50, alpha=0.5, label=f'Iter {i+1}', density=True)
    
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution Across Iterations')
    plt.legend()
    
    # Plot number of high confidence samples
    plt.subplot(1, 2, 2)
    conf_thresholds = np.arange(0.5, 1.01, 0.05)
    
    for i, result in enumerate(results):
        confidence = result['confidence']
        counts = [sum(confidence >= t) for t in conf_thresholds]
        plt.plot(conf_thresholds, counts, marker='o', label=f'Iter {i+1}')
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Number of Samples')
    plt.title('High Confidence Sample Counts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pseudo_labeling_confidence.png', dpi=300)
    plt.close()
    
    print("\nSaved confidence distribution plot")

def main():
    # Load data
    train_df, test_df = load_and_prepare_data()
    
    # Perform iterative pseudo-labeling
    results, final_train = iterative_pseudo_labeling(train_df, test_df, n_iterations=5)
    
    # Analyze changes
    analyze_prediction_changes(results)
    
    # Create submissions
    create_submissions(results, test_df)
    
    # Visualize results
    visualize_confidence_distribution(results)
    
    # Summary
    print("\n" + "="*60)
    print("PSEUDO-LABELING SUMMARY")
    print("="*60)
    print(f"Original training samples: {len(train_df)}")
    print(f"Final training samples: {len(final_train)}")
    print(f"Pseudo-labeled samples added: {len(final_train) - len(train_df)}")
    print(f"Iterations completed: {len(results)}")
    
    # Check final distribution
    if len(final_train) > len(train_df):
        pseudo_only = final_train[len(train_df):]
        print(f"\nPseudo-label distribution:")
        print(pseudo_only['Personality'].value_counts())

if __name__ == "__main__":
    main()