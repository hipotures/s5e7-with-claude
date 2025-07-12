#!/usr/bin/env python3
"""
Monitor accuracy improvement during pseudo-labeling iterations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def evaluate_model_accuracy(train_df, n_splits=5):
    """Evaluate model using cross-validation"""
    accuracies = []
    
    for i in range(n_splits):
        # Split data
        train_split, val_split = train_test_split(
            train_df, test_size=0.2, random_state=42+i, stratify=train_df['Personality']
        )
        
        # Train model
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=300,
            random_seed=42
        )
        model = learner.train(train_split)
        
        # Evaluate
        predictions = model.predict(val_split)
        pred_classes = []
        
        for pred in predictions:
            prob_I = float(str(pred))
            pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
        
        accuracy = sum(p == t for p, t in zip(pred_classes, val_split['Personality'])) / len(val_split)
        accuracies.append(accuracy)
    
    return np.mean(accuracies), np.std(accuracies)

def monitor_pseudo_labeling(train_df, test_df, n_iterations=10):
    """Pseudo-labeling with accuracy monitoring"""
    
    # Calculate target proportions
    train_proportions = train_df['Personality'].value_counts(normalize=True)
    target_E_ratio = train_proportions['Extrovert']
    target_I_ratio = train_proportions['Introvert']
    
    print("="*60)
    print("PSEUDO-LABELING WITH ACCURACY MONITORING")
    print(f"Target proportions: E={target_E_ratio:.1%}, I={target_I_ratio:.1%}")
    print("="*60)
    
    results = []
    current_train = train_df.copy()
    all_pseudo_ids = set()
    
    # Evaluate initial accuracy
    print("\nEvaluating initial model...")
    initial_acc, initial_std = evaluate_model_accuracy(current_train)
    print(f"Initial accuracy: {initial_acc:.6f} ± {initial_std:.6f}")
    
    results.append({
        'iteration': 0,
        'n_samples': len(current_train),
        'n_pseudo': 0,
        'accuracy': initial_acc,
        'std': initial_std
    })
    
    # Thresholds
    E_threshold = 0.99
    I_threshold = 0.90
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
        
        # Find high confidence samples
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
        
        print(f"Candidates: {len(E_candidates)} E, {len(I_candidates)} I")
        
        # Calculate how many to add
        n_E_to_add = int(SAMPLES_PER_ITER * target_E_ratio)
        n_I_to_add = SAMPLES_PER_ITER - n_E_to_add
        
        n_E_actual = min(n_E_to_add, len(E_candidates))
        n_I_actual = min(n_I_to_add, len(I_candidates))
        
        if n_E_actual + n_I_actual == 0:
            print("No more candidates found. Stopping.")
            break
        
        # Add pseudo-labels
        pseudo_samples = []
        
        for idx, conf in E_candidates[:n_E_actual]:
            sample = test_df.iloc[idx].to_dict()
            sample['Personality'] = 'Extrovert'
            pseudo_samples.append(sample)
            all_pseudo_ids.add(sample['id'])
        
        for idx, conf in I_candidates[:n_I_actual]:
            sample = test_df.iloc[idx].to_dict()
            sample['Personality'] = 'Introvert'
            pseudo_samples.append(sample)
            all_pseudo_ids.add(sample['id'])
        
        if len(pseudo_samples) > 0:
            pseudo_df = pd.DataFrame(pseudo_samples)
            print(f"Adding {len(pseudo_df)} samples: {n_E_actual} E, {n_I_actual} I")
            
            current_train = pd.concat([current_train, pseudo_df], ignore_index=True)
            
            # Evaluate new accuracy
            print("Evaluating model with pseudo-labels...")
            new_acc, new_std = evaluate_model_accuracy(current_train, n_splits=3)
            print(f"New accuracy: {new_acc:.6f} ± {new_std:.6f}")
            
            # Calculate improvement
            improvement = new_acc - results[-1]['accuracy']
            print(f"Improvement: {improvement:+.6f}")
            
            results.append({
                'iteration': iteration + 1,
                'n_samples': len(current_train),
                'n_pseudo': len(all_pseudo_ids),
                'accuracy': new_acc,
                'std': new_std,
                'n_added_E': n_E_actual,
                'n_added_I': n_I_actual
            })
    
    return results, current_train

def plot_accuracy_progression(results):
    """Plot accuracy vs pseudo-labeling iterations"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Accuracy plot
    iterations = [r['iteration'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    stds = [r['std'] for r in results]
    
    ax1.errorbar(iterations, accuracies, yerr=stds, marker='o', linewidth=2, capsize=5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cross-Validation Accuracy')
    ax1.set_title('Model Accuracy vs Pseudo-Labeling Iterations')
    ax1.grid(True, alpha=0.3)
    
    # Annotate improvement
    for i in range(1, len(results)):
        improvement = results[i]['accuracy'] - results[i-1]['accuracy']
        ax1.annotate(f'{improvement:+.4f}', 
                    xy=(results[i]['iteration'], results[i]['accuracy']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # Samples added
    ax2.bar(iterations[1:], [r['n_added_E'] for r in results[1:]], 
            label='Extrovert', color='red', alpha=0.7)
    ax2.bar(iterations[1:], [r['n_added_I'] for r in results[1:]], 
            bottom=[r['n_added_E'] for r in results[1:]], 
            label='Introvert', color='blue', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Samples Added')
    ax2.set_title('Pseudo-Labels Added per Iteration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pseudo_labeling_accuracy_monitor.png', dpi=300)
    plt.close()

def main():
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"Original train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Run pseudo-labeling with monitoring
    results, final_train = monitor_pseudo_labeling(train_df, test_df, n_iterations=10)
    
    # Plot results
    plot_accuracy_progression(results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nAccuracy progression:")
    for r in results:
        if r['iteration'] == 0:
            print(f"Initial: {r['accuracy']:.6f} ± {r['std']:.6f}")
        else:
            improvement = r['accuracy'] - results[0]['accuracy']
            print(f"Iter {r['iteration']:2d}: {r['accuracy']:.6f} ± {r['std']:.6f} "
                  f"(total improvement: {improvement:+.6f})")
    
    total_improvement = results[-1]['accuracy'] - results[0]['accuracy']
    print(f"\nTotal improvement: {total_improvement:+.6f}")
    print(f"Final train size: {len(final_train)} (+{len(final_train) - len(train_df)} pseudo-labels)")
    
    # Create final submission
    if len(results) > 1:
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=300,
            random_seed=42
        )
        final_model = learner.train(final_train)
        
        predictions = final_model.predict(test_df)
        pred_classes = []
        
        for pred in predictions:
            prob_I = float(str(pred))
            pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': pred_classes
        })
        
        filename = f'submission_pseudo_monitored_{len(final_train) - len(train_df)}_samples.csv'
        submission.to_csv(SCORES_DIR / filename, index=False)
        print(f"\nCreated: {filename}")

if __name__ == "__main__":
    main()