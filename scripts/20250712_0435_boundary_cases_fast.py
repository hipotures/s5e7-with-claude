#!/usr/bin/env python3
"""
Fast boundary cases analysis - simplified version
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

def main():
    print("="*60)
    print("FAST BOUNDARY CASES ANALYSIS")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"\nData shapes:")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    
    # Features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_df[feature_cols + ['id']], 
        train_df['Personality'],
        test_size=0.2, 
        random_state=42, 
        stratify=train_df['Personality']
    )
    
    # Train model
    print("\nTraining model for boundary detection...")
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=200,
        random_seed=42
    )
    
    train_data = X_train.copy()
    train_data['Personality'] = y_train
    model = learner.train(train_data[feature_cols + ['Personality']])
    
    # Get predictions on validation set
    val_predictions = model.predict(X_val[feature_cols])
    
    # Extract probabilities and find boundary cases
    probabilities = []
    pred_classes = []
    
    for pred in val_predictions:
        prob_I = float(str(pred))
        probabilities.append(prob_I)
        pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
    
    probabilities = np.array(probabilities)
    confidence = np.abs(probabilities - 0.5) * 2
    
    # Create results dataframe
    val_results = pd.DataFrame({
        'id': X_val['id'].values,
        'actual': y_val.values,
        'predicted': pred_classes,
        'probability': probabilities,
        'confidence': confidence,
        'is_correct': y_val.values == pred_classes
    })
    
    # Find boundary cases
    boundary_threshold = 0.1
    boundary_cases = val_results[val_results['confidence'] < boundary_threshold]
    
    print(f"\n1. BOUNDARY CASES (confidence < {boundary_threshold}):")
    print(f"   Found: {len(boundary_cases)} ({len(boundary_cases)/len(val_results)*100:.1f}%)")
    print(f"   Accuracy on boundary: {boundary_cases['is_correct'].mean():.3f}")
    print(f"   Overall validation accuracy: {val_results['is_correct'].mean():.3f}")
    
    # Misclassified analysis
    misclassified = val_results[~val_results['is_correct']]
    high_conf_misclass = misclassified[misclassified['confidence'] > 0.8]
    
    print(f"\n2. MISCLASSIFICATION ANALYSIS:")
    print(f"   Total misclassified: {len(misclassified)} ({len(misclassified)/len(val_results)*100:.1f}%)")
    print(f"   High confidence errors (>0.8): {len(high_conf_misclass)}")
    print(f"   Mean confidence of errors: {misclassified['confidence'].mean():.3f}")
    
    # Feature analysis for boundary cases
    boundary_ids = set(boundary_cases['id'])
    train_boundary = train_df[train_df['id'].isin(boundary_ids)]
    
    print(f"\n3. FEATURE PATTERNS IN BOUNDARY CASES:")
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    for feature in numeric_features:
        all_mean = train_df[feature].mean()
        boundary_mean = train_boundary[feature].mean()
        diff = (boundary_mean - all_mean) / all_mean * 100 if all_mean != 0 else 0
        print(f"   {feature}: {diff:+.1f}% vs average")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Confidence distribution
    axes[0, 0].hist(val_results['confidence'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(boundary_threshold, color='red', linestyle='--', label=f'Boundary ({boundary_threshold})')
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Model Confidence Distribution')
    axes[0, 0].legend()
    
    # Plot 2: Probability distribution
    axes[0, 1].hist(val_results['probability'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0.5, color='red', linestyle='--', label='Decision boundary')
    axes[0, 1].set_xlabel('P(Introvert)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Probability Distribution')
    axes[0, 1].legend()
    
    # Plot 3: Confidence vs Accuracy
    conf_bins = np.linspace(0, 1, 11)
    bin_acc = []
    bin_centers = []
    
    for i in range(len(conf_bins)-1):
        mask = (val_results['confidence'] >= conf_bins[i]) & (val_results['confidence'] < conf_bins[i+1])
        if mask.sum() > 0:
            bin_acc.append(val_results[mask]['is_correct'].mean())
            bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
    
    axes[1, 0].plot(bin_centers, bin_acc, 'o-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy vs Confidence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Class distribution in boundary cases
    boundary_dist = boundary_cases['actual'].value_counts()
    axes[1, 1].bar(boundary_dist.index, boundary_dist.values)
    axes[1, 1].set_xlabel('Personality')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Boundary Cases Distribution')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boundary_analysis_fast.png', dpi=300)
    plt.close()
    
    # Create boundary-aware submission
    print("\n4. CREATING BOUNDARY-AWARE SUBMISSION")
    
    # Train on full data with boundary awareness
    # Give less weight to very confident predictions (they might be overconfident)
    full_model = learner.train(train_df[feature_cols + ['Personality']])
    
    # Predict test
    test_predictions = full_model.predict(test_df[feature_cols])
    
    test_pred_classes = []
    test_confidences = []
    
    for pred in test_predictions:
        prob_I = float(str(pred))
        conf = abs(prob_I - 0.5) * 2
        
        # For very low confidence cases, use a slight bias toward majority class
        if conf < 0.05:  # Very uncertain
            # Slight bias toward Extrovert (majority class)
            pred_class = 'Introvert' if prob_I > 0.52 else 'Extrovert'
        else:
            pred_class = 'Introvert' if prob_I > 0.5 else 'Extrovert'
        
        test_pred_classes.append(pred_class)
        test_confidences.append(conf)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': test_pred_classes
    })
    
    submission.to_csv(SCORES_DIR / 'submission_boundary_aware.csv', index=False)
    print("   Created: submission_boundary_aware.csv")
    
    # Test statistics
    test_confidences = np.array(test_confidences)
    print(f"\n5. TEST SET STATISTICS:")
    print(f"   Mean confidence: {test_confidences.mean():.3f}")
    print(f"   Low confidence (<0.1): {(test_confidences < 0.1).sum()} samples")
    print(f"   High confidence (>0.9): {(test_confidences > 0.9).sum()} samples")
    
    # Find potential ambiverts in test set
    very_low_conf = test_confidences < 0.05
    potential_ambiverts = test_df.iloc[very_low_conf][['id']].copy()
    potential_ambiverts['confidence'] = test_confidences[very_low_conf]
    potential_ambiverts['prediction'] = np.array(test_pred_classes)[very_low_conf]
    
    print(f"\n6. POTENTIAL AMBIVERTS IN TEST SET:")
    print(f"   Found {len(potential_ambiverts)} samples with confidence < 0.05")
    
    if len(potential_ambiverts) > 0:
        print("\n   Top 10 most uncertain predictions:")
        top_uncertain = potential_ambiverts.nsmallest(10, 'confidence')
        for _, row in top_uncertain.iterrows():
            print(f"   ID {row['id']}: conf={row['confidence']:.4f}, pred={row['prediction']}")
    
    # Save detailed results
    val_results.to_csv(OUTPUT_DIR / 'boundary_validation_results.csv', index=False)
    potential_ambiverts.to_csv(OUTPUT_DIR / 'test_potential_ambiverts.csv', index=False)
    
    print("\n" + "="*60)
    print("BOUNDARY ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()