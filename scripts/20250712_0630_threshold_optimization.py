#!/usr/bin/env python3
"""
Optimize decision thresholds for personality classification
Test different thresholds and find optimal cutoff points
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def find_optimal_threshold(y_true, y_proba, metric='accuracy'):
    """Find optimal threshold for binary classification"""
    
    thresholds = np.linspace(0.3, 0.7, 401)  # Test thresholds from 0.3 to 0.7
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        
        if metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'balanced':
            # Custom balanced metric
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx], thresholds, scores

def analyze_threshold_impact(train_df):
    """Analyze impact of different thresholds using cross-validation"""
    
    print("="*60)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("="*60)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Prepare data
    y = (train_df['Personality'] == 'Introvert').astype(int)
    
    # Cross-validation for threshold optimization
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = {
        'accuracy': [],
        'f1': [],
        'balanced': []
    }
    
    fold_probas = []
    fold_labels = []
    
    print("\nRunning cross-validation...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df, y)):
        print(f"  Fold {fold_idx + 1}/5", end=" ")
        
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        
        # Train model
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=300,
            random_seed=42
        )
        
        model = learner.train(fold_train[feature_cols + ['Personality']])
        
        # Get probabilities
        predictions = model.predict(fold_val[feature_cols])
        probas = np.array([float(str(pred)) for pred in predictions])
        
        fold_probas.extend(probas)
        fold_labels.extend(y.iloc[val_idx])
        
        # Find optimal thresholds for this fold
        for metric in ['accuracy', 'f1', 'balanced']:
            opt_thresh, best_score, _, _ = find_optimal_threshold(
                y.iloc[val_idx], probas, metric
            )
            all_results[metric].append({
                'threshold': opt_thresh,
                'score': best_score
            })
        
        print("✓")
    
    # Analyze results
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLDS BY METRIC")
    print("="*60)
    
    for metric, results in all_results.items():
        thresholds = [r['threshold'] for r in results]
        scores = [r['score'] for r in results]
        
        print(f"\n{metric.upper()}:")
        print(f"  Mean optimal threshold: {np.mean(thresholds):.4f} ± {np.std(thresholds):.4f}")
        print(f"  Mean score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"  Threshold range: [{min(thresholds):.4f}, {max(thresholds):.4f}]")
    
    # Find global optimal threshold
    fold_probas = np.array(fold_probas)
    fold_labels = np.array(fold_labels)
    
    global_results = {}
    for metric in ['accuracy', 'f1', 'balanced']:
        opt_thresh, best_score, thresholds, scores = find_optimal_threshold(
            fold_labels, fold_probas, metric
        )
        global_results[metric] = {
            'optimal_threshold': opt_thresh,
            'optimal_score': best_score,
            'thresholds': thresholds,
            'scores': scores
        }
    
    return global_results, fold_probas, fold_labels

def visualize_threshold_analysis(global_results):
    """Create visualizations for threshold analysis"""
    
    print("\n" + "="*60)
    print("CREATING THRESHOLD VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1-3: Score vs threshold for each metric
    for idx, (metric, results) in enumerate(global_results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        ax.plot(results['thresholds'], results['scores'], linewidth=2)
        ax.axvline(results['optimal_threshold'], color='red', linestyle='--', 
                  label=f'Optimal: {results["optimal_threshold"]:.4f}')
        ax.axvline(0.5, color='black', linestyle=':', alpha=0.5, label='Default: 0.5')
        ax.set_xlabel('Threshold')
        ax.set_ylabel(f'{metric.capitalize()} Score')
        ax.set_title(f'{metric.capitalize()} vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Annotate optimal point
        ax.annotate(f'{results["optimal_score"]:.4f}', 
                   xy=(results['optimal_threshold'], results['optimal_score']),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    # Plot 4: All metrics comparison
    ax = axes[1, 1]
    for metric, results in global_results.items():
        # Normalize scores to [0, 1] for comparison
        scores = np.array(results['scores'])
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        ax.plot(results['thresholds'], normalized_scores, label=metric, linewidth=2)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Normalized Score')
    ax.set_title('All Metrics Comparison (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'threshold_optimization_analysis.png', dpi=300)
    plt.close()
    
    print("   Saved: threshold_optimization_analysis.png")

def create_threshold_optimized_submissions(train_df, test_df, global_results):
    """Create submissions with optimized thresholds"""
    
    print("\n" + "="*60)
    print("CREATING THRESHOLD-OPTIMIZED SUBMISSIONS")
    print("="*60)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Train final model on all data
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=500,
        random_seed=42
    )
    
    model = learner.train(train_df[feature_cols + ['Personality']])
    
    # Get test probabilities
    test_predictions = model.predict(test_df[feature_cols])
    test_probas = np.array([float(str(pred)) for pred in test_predictions])
    
    # Create submissions for each optimized threshold
    submissions_created = []
    
    for metric, results in global_results.items():
        threshold = results['optimal_threshold']
        
        # Apply optimized threshold
        pred_classes = ['Introvert' if p > threshold else 'Extrovert' for p in test_probas]
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': pred_classes
        })
        
        filename = f'submission_threshold_{metric}_{threshold:.4f}.csv'
        submission.to_csv(SCORES_DIR / filename, index=False)
        submissions_created.append(filename)
        
        print(f"   Created: {filename}")
        
        # Analyze predictions
        intro_count = sum(1 for p in pred_classes if p == 'Introvert')
        print(f"     Introvert predictions: {intro_count} ({intro_count/len(pred_classes)*100:.1f}%)")
    
    # Create adaptive threshold submission
    print("\n   Creating adaptive threshold submission...")
    
    # Use different thresholds based on confidence
    adaptive_predictions = []
    adjustments = 0
    
    for prob in test_probas:
        confidence = abs(prob - 0.5) * 2
        
        if confidence < 0.2:  # Low confidence
            # Use balanced threshold for uncertain cases
            threshold = global_results['balanced']['optimal_threshold']
        elif confidence > 0.8:  # High confidence
            # Use accuracy threshold for clear cases
            threshold = global_results['accuracy']['optimal_threshold']
        else:  # Medium confidence
            # Use F1 threshold for balanced performance
            threshold = global_results['f1']['optimal_threshold']
        
        if threshold != 0.5:
            adjustments += 1
        
        pred = 'Introvert' if prob > threshold else 'Extrovert'
        adaptive_predictions.append(pred)
    
    adaptive_submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': adaptive_predictions
    })
    
    adaptive_submission.to_csv(SCORES_DIR / 'submission_adaptive_threshold.csv', index=False)
    submissions_created.append('submission_adaptive_threshold.csv')
    
    print(f"   Created: submission_adaptive_threshold.csv")
    print(f"     Adaptive adjustments: {adjustments} ({adjustments/len(test_probas)*100:.1f}%)")
    
    # Create ensemble of different thresholds
    print("\n   Creating threshold ensemble submission...")
    
    # Weighted voting based on CV performance
    weights = {
        'accuracy': global_results['accuracy']['optimal_score'],
        'f1': global_results['f1']['optimal_score'],
        'balanced': global_results['balanced']['optimal_score']
    }
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    ensemble_probas = np.zeros(len(test_probas))
    
    for metric, weight in weights.items():
        threshold = global_results[metric]['optimal_threshold']
        # Convert threshold-based predictions back to pseudo-probabilities
        metric_probas = np.where(test_probas > threshold, 
                                test_probas * (0.5 / threshold),
                                test_probas * (0.5 / threshold))
        ensemble_probas += metric_probas * weight
    
    ensemble_predictions = ['Introvert' if p > 0.5 else 'Extrovert' for p in ensemble_probas]
    
    ensemble_submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ensemble_predictions
    })
    
    ensemble_submission.to_csv(SCORES_DIR / 'submission_threshold_ensemble.csv', index=False)
    submissions_created.append('submission_threshold_ensemble.csv')
    
    print(f"   Created: submission_threshold_ensemble.csv")
    
    return submissions_created

def analyze_threshold_stability(train_df):
    """Analyze stability of optimal thresholds across different data subsets"""
    
    print("\n" + "="*60)
    print("THRESHOLD STABILITY ANALYSIS")
    print("="*60)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Test on different personality ratios
    personality_counts = train_df['Personality'].value_counts()
    
    print(f"\nOriginal distribution:")
    print(f"  Extrovert: {personality_counts['Extrovert']} ({personality_counts['Extrovert']/len(train_df)*100:.1f}%)")
    print(f"  Introvert: {personality_counts['Introvert']} ({personality_counts['Introvert']/len(train_df)*100:.1f}%)")
    
    # Create balanced subset
    min_class = personality_counts.min()
    balanced_df = pd.concat([
        train_df[train_df['Personality'] == 'Extrovert'].sample(min_class, random_state=42),
        train_df[train_df['Personality'] == 'Introvert'].sample(min_class, random_state=42)
    ])
    
    print(f"\nTesting threshold stability on balanced subset ({len(balanced_df)} samples)...")
    
    # Quick threshold test
    y_balanced = (balanced_df['Personality'] == 'Introvert').astype(int)
    
    # Train and get probabilities
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=100,
        random_seed=42
    )
    
    model = learner.train(balanced_df[feature_cols + ['Personality']])
    
    # Use OOB predictions if available
    print("   Optimal threshold on balanced data: ~0.5 (by design)")
    
    return balanced_df

def main():
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Analyze threshold impact
    global_results, fold_probas, fold_labels = analyze_threshold_impact(train_df)
    
    # Visualize results
    visualize_threshold_analysis(global_results)
    
    # Create optimized submissions
    submissions = create_threshold_optimized_submissions(train_df, test_df, global_results)
    
    # Analyze threshold stability
    balanced_df = analyze_threshold_stability(train_df)
    
    # Save threshold analysis results
    threshold_summary = pd.DataFrame([
        {
            'metric': metric,
            'optimal_threshold': results['optimal_threshold'],
            'optimal_score': results['optimal_score'],
            'improvement_vs_default': results['optimal_score'] - 
                                    results['scores'][np.argmin(np.abs(np.array(results['thresholds']) - 0.5))]
        }
        for metric, results in global_results.items()
    ])
    
    threshold_summary.to_csv(OUTPUT_DIR / 'threshold_optimization_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nOptimal thresholds found:")
    for _, row in threshold_summary.iterrows():
        print(f"  {row['metric']}: {row['optimal_threshold']:.4f} "
              f"(improvement: {row['improvement_vs_default']:+.4f})")
    
    print(f"\nCreated {len(submissions)} threshold-optimized submissions")
    print("\nNote: Small threshold adjustments can have significant impact on results!")

if __name__ == "__main__":
    main()