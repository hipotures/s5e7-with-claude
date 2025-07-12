#!/usr/bin/env python3
"""
Aggressive pseudo-labeling with lower confidence thresholds
Test more aggressive strategies to add more pseudo-labels
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

def aggressive_pseudo_labeling(train_df, test_df, strategy='very_aggressive'):
    """Aggressive pseudo-labeling with different strategies"""
    
    print("="*60)
    print(f"AGGRESSIVE PSEUDO-LABELING - {strategy.upper()}")
    print("="*60)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Define strategies
    strategies = {
        'moderate': {
            'E_threshold': 0.85,
            'I_threshold': 0.75,
            'samples_per_iter': 100,
            'max_samples': 1000
        },
        'aggressive': {
            'E_threshold': 0.80,
            'I_threshold': 0.70,
            'samples_per_iter': 200,
            'max_samples': 2000
        },
        'very_aggressive': {
            'E_threshold': 0.75,
            'I_threshold': 0.65,
            'samples_per_iter': 300,
            'max_samples': 3000
        },
        'extreme': {
            'E_threshold': 0.70,
            'I_threshold': 0.60,
            'samples_per_iter': 500,
            'max_samples': 4000
        }
    }
    
    config = strategies[strategy]
    
    print(f"\nStrategy configuration:")
    print(f"  E threshold: {config['E_threshold']}")
    print(f"  I threshold: {config['I_threshold']}")
    print(f"  Samples per iteration: {config['samples_per_iter']}")
    print(f"  Max total samples: {config['max_samples']}")
    
    # Calculate target proportions
    train_proportions = train_df['Personality'].value_counts(normalize=True)
    target_E_ratio = train_proportions['Extrovert']
    target_I_ratio = train_proportions['Introvert']
    
    print(f"\nTarget proportions: E={target_E_ratio:.1%}, I={target_I_ratio:.1%}")
    
    # Initialize
    current_train = train_df.copy()
    all_pseudo_ids = set()
    iteration_results = []
    
    # Validation split for monitoring
    X_train, X_val, y_train, y_val = train_test_split(
        train_df[feature_cols + ['id', 'Personality']], 
        train_df['Personality'],
        test_size=0.2, 
        random_state=42, 
        stratify=train_df['Personality']
    )
    
    print("\n" + "-"*60)
    
    iteration = 0
    while len(all_pseudo_ids) < config['max_samples']:
        iteration += 1
        print(f"\nIteration {iteration}:")
        
        # Train model
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=300,
            random_seed=42
        )
        
        model = learner.train(current_train[feature_cols + ['Personality']])
        
        # Validate performance
        val_predictions = model.predict(X_val[feature_cols])
        val_pred_classes = []
        for pred in val_predictions:
            prob_I = float(str(pred))
            val_pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
        
        val_accuracy = sum(p == t for p, t in zip(val_pred_classes, y_val)) / len(y_val)
        
        # Get predictions on test set
        predictions = model.predict(test_df[feature_cols])
        
        proba_list = []
        pred_classes = []
        
        for pred in predictions:
            prob_I = float(str(pred))
            proba_list.append(prob_I)
            pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
        
        proba_array = np.array(proba_list)
        confidence = np.abs(proba_array - 0.5) * 2
        
        # Find candidates with lower thresholds
        E_candidates = []
        I_candidates = []
        
        for i in range(len(test_df)):
            if test_df.iloc[i]['id'] not in all_pseudo_ids:
                if pred_classes[i] == 'Extrovert' and confidence[i] >= config['E_threshold']:
                    E_candidates.append((i, confidence[i], proba_array[i]))
                elif pred_classes[i] == 'Introvert' and confidence[i] >= config['I_threshold']:
                    I_candidates.append((i, confidence[i], proba_array[i]))
        
        # Sort by confidence
        E_candidates.sort(key=lambda x: x[1], reverse=True)
        I_candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Candidates: {len(E_candidates)} E, {len(I_candidates)} I")
        print(f"  Validation accuracy: {val_accuracy:.4f}")
        
        if len(E_candidates) == 0 and len(I_candidates) == 0:
            print("  No more candidates found. Stopping.")
            break
        
        # Calculate how many to add
        n_to_add = min(config['samples_per_iter'], 
                      config['max_samples'] - len(all_pseudo_ids))
        n_E_to_add = int(n_to_add * target_E_ratio)
        n_I_to_add = n_to_add - n_E_to_add
        
        # Actual numbers based on availability
        n_E_actual = min(n_E_to_add, len(E_candidates))
        n_I_actual = min(n_I_to_add, len(I_candidates))
        
        # If we can't maintain ratio, adjust
        if n_E_actual < n_E_to_add and len(I_candidates) > n_I_actual:
            n_I_actual = min(n_to_add - n_E_actual, len(I_candidates))
        elif n_I_actual < n_I_to_add and len(E_candidates) > n_E_actual:
            n_E_actual = min(n_to_add - n_I_actual, len(E_candidates))
        
        # Add pseudo-labels
        pseudo_samples = []
        
        # Add E samples
        for idx, conf, prob in E_candidates[:n_E_actual]:
            sample = test_df.iloc[idx].to_dict()
            sample['Personality'] = 'Extrovert'
            pseudo_samples.append(sample)
            all_pseudo_ids.add(sample['id'])
        
        # Add I samples  
        for idx, conf, prob in I_candidates[:n_I_actual]:
            sample = test_df.iloc[idx].to_dict()
            sample['Personality'] = 'Introvert'
            pseudo_samples.append(sample)
            all_pseudo_ids.add(sample['id'])
        
        if len(pseudo_samples) > 0:
            pseudo_df = pd.DataFrame(pseudo_samples)
            current_train = pd.concat([current_train, pseudo_df], ignore_index=True)
            
            # Calculate confidence stats
            E_confs = [conf for _, conf, _ in E_candidates[:n_E_actual]]
            I_confs = [conf for _, conf, _ in I_candidates[:n_I_actual]]
            
            iteration_results.append({
                'iteration': iteration,
                'n_added': len(pseudo_samples),
                'n_added_E': n_E_actual,
                'n_added_I': n_I_actual,
                'total_pseudo': len(all_pseudo_ids),
                'val_accuracy': val_accuracy,
                'E_conf_min': min(E_confs) if E_confs else 0,
                'E_conf_mean': np.mean(E_confs) if E_confs else 0,
                'I_conf_min': min(I_confs) if I_confs else 0,
                'I_conf_mean': np.mean(I_confs) if I_confs else 0
            })
            
            print(f"  Added: {n_E_actual} E (conf: {min(E_confs) if E_confs else 0:.3f}-1.000), "
                  f"{n_I_actual} I (conf: {min(I_confs) if I_confs else 0:.3f}-1.000)")
            print(f"  Total pseudo-labels: {len(all_pseudo_ids)}")
    
    return current_train, iteration_results, model

def visualize_aggressive_results(all_results):
    """Visualize results from different aggressive strategies"""
    
    print("\n" + "="*60)
    print("VISUALIZING AGGRESSIVE PSEUDO-LABELING RESULTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Pseudo-labels added over iterations
    ax1 = axes[0, 0]
    for strategy, results in all_results.items():
        if results:
            iterations = [r['iteration'] for r in results]
            total_pseudo = [r['total_pseudo'] for r in results]
            ax1.plot(iterations, total_pseudo, 'o-', label=strategy, linewidth=2)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Pseudo-labels')
    ax1.set_title('Pseudo-labels Accumulation by Strategy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy over iterations
    ax2 = axes[0, 1]
    for strategy, results in all_results.items():
        if results:
            iterations = [r['iteration'] for r in results]
            val_acc = [r['val_accuracy'] for r in results]
            ax2.plot(iterations, val_acc, 'o-', label=strategy, linewidth=2)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Model Performance During Pseudo-labeling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confidence evolution
    ax3 = axes[1, 0]
    for strategy, results in all_results.items():
        if results:
            iterations = [r['iteration'] for r in results]
            e_conf = [r['E_conf_min'] for r in results]
            i_conf = [r['I_conf_min'] for r in results]
            
            ax3.plot(iterations, e_conf, '-', label=f'{strategy} (E)', linewidth=2)
            ax3.plot(iterations, i_conf, '--', label=f'{strategy} (I)', linewidth=1)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Minimum Confidence')
    ax3.set_title('Minimum Confidence Thresholds Used')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Strategy comparison summary
    ax4 = axes[1, 1]
    strategy_names = list(all_results.keys())
    total_samples = [results[-1]['total_pseudo'] if results else 0 for results in all_results.values()]
    final_accuracy = [results[-1]['val_accuracy'] if results else 0 for results in all_results.values()]
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar(x - width/2, total_samples, width, label='Total Samples', color='skyblue')
    bars2 = ax4_twin.bar(x + width/2, final_accuracy, width, label='Final Accuracy', color='orange')
    
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Total Pseudo-labels', color='skyblue')
    ax4_twin.set_ylabel('Final Validation Accuracy', color='orange')
    ax4.set_title('Strategy Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategy_names, rotation=45)
    ax4.tick_params(axis='y', labelcolor='skyblue')
    ax4_twin.tick_params(axis='y', labelcolor='orange')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'aggressive_pseudo_labeling_comparison.png', dpi=300)
    plt.close()
    
    print("   Saved: aggressive_pseudo_labeling_comparison.png")

def create_aggressive_submissions(train_df, test_df):
    """Create submissions using aggressive pseudo-labeling"""
    
    print("\n" + "="*60)
    print("CREATING AGGRESSIVE PSEUDO-LABELING SUBMISSIONS")
    print("="*60)
    
    strategies = ['moderate', 'aggressive', 'very_aggressive', 'extreme']
    all_results = {}
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    for strategy in strategies:
        print(f"\n\nTesting {strategy} strategy...")
        print("="*60)
        
        # Run aggressive pseudo-labeling
        final_train, iteration_results, final_model = aggressive_pseudo_labeling(
            train_df, test_df, strategy
        )
        
        all_results[strategy] = iteration_results
        
        # Make final predictions
        if iteration_results:
            final_predictions = final_model.predict(test_df[feature_cols])
            pred_classes = []
            
            for pred in final_predictions:
                prob_I = float(str(pred))
                pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
            
            # Create submission
            submission = pd.DataFrame({
                'id': test_df['id'],
                'Personality': pred_classes
            })
            
            total_pseudo = iteration_results[-1]['total_pseudo']
            filename = f'submission_aggressive_{strategy}_{total_pseudo}_samples.csv'
            submission.to_csv(SCORES_DIR / filename, index=False)
            print(f"\nCreated: {filename}")
            
            # Analyze final distribution
            final_E = sum(1 for p in pred_classes if p == 'Extrovert')
            print(f"Final predictions: {final_E} E ({final_E/len(pred_classes)*100:.1f}%), "
                  f"{len(pred_classes)-final_E} I ({(len(pred_classes)-final_E)/len(pred_classes)*100:.1f}%)")
    
    return all_results

def analyze_pseudo_label_quality(train_df, test_df):
    """Analyze quality of pseudo-labels at different confidence levels"""
    
    print("\n" + "="*60)
    print("PSEUDO-LABEL QUALITY ANALYSIS")
    print("="*60)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Train base model
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        random_seed=42
    )
    
    model = learner.train(train_df[feature_cols + ['Personality']])
    
    # Get test predictions
    predictions = model.predict(test_df[feature_cols])
    
    probas = []
    for pred in predictions:
        probas.append(float(str(pred)))
    
    probas = np.array(probas)
    confidence = np.abs(probas - 0.5) * 2
    
    # Analyze confidence distribution
    print("\nConfidence distribution in test set:")
    conf_ranges = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    
    for low, high in conf_ranges:
        mask = (confidence >= low) & (confidence < high)
        count = mask.sum()
        print(f"  [{low:.1f}, {high:.1f}): {count} samples ({count/len(test_df)*100:.1f}%)")
    
    # Estimate potential pseudo-labels at different thresholds
    print("\nPotential pseudo-labels by threshold:")
    
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for thresh in thresholds:
        n_above = (confidence >= thresh).sum()
        print(f"  Threshold {thresh}: {n_above} samples ({n_above/len(test_df)*100:.1f}%)")

def main():
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Analyze pseudo-label quality first
    analyze_pseudo_label_quality(train_df, test_df)
    
    # Create aggressive submissions
    all_results = create_aggressive_submissions(train_df, test_df)
    
    # Visualize results
    visualize_aggressive_results(all_results)
    
    # Save detailed results
    for strategy, results in all_results.items():
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(OUTPUT_DIR / f'aggressive_pseudo_{strategy}_details.csv', index=False)
    
    print("\n" + "="*60)
    print("AGGRESSIVE PSEUDO-LABELING COMPLETE")
    print("="*60)
    
    # Summary
    print("\nStrategy summary:")
    for strategy, results in all_results.items():
        if results:
            final = results[-1]
            print(f"  {strategy}: {final['total_pseudo']} pseudo-labels, "
                  f"final accuracy: {final['val_accuracy']:.4f}")

if __name__ == "__main__":
    main()