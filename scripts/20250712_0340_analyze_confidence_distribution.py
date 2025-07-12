#!/usr/bin/env python3
"""
Analyze confidence distribution for both classes
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

def analyze_confidence_by_class():
    """Analyze confidence distribution for each predicted class"""
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print("="*60)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Check train distribution
    print(f"\nTrain distribution:")
    print(train_df['Personality'].value_counts())
    print(f"Ratio: {train_df['Personality'].value_counts(normalize=True)}")
    
    # Train model
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        random_seed=42
    )
    
    model = learner.train(train_df)
    
    # Get predictions on test set
    predictions = model.predict(test_df)
    
    # Extract probabilities and classes
    proba_list = []
    pred_classes = []
    
    for pred in predictions:
        # YDF returns P(Introvert)
        prob_introvert = float(str(pred))
        proba_list.append(prob_introvert)
        pred_classes.append('Introvert' if prob_introvert > 0.5 else 'Extrovert')
    
    proba_array = np.array(proba_list)
    
    # Calculate confidence (distance from 0.5)
    confidence = np.abs(proba_array - 0.5) * 2
    
    # Separate by predicted class
    extrovert_mask = np.array(pred_classes) == 'Extrovert'
    introvert_mask = np.array(pred_classes) == 'Introvert'
    
    conf_extrovert = confidence[extrovert_mask]
    conf_introvert = confidence[introvert_mask]
    
    print(f"\nTest predictions:")
    print(f"Extrovert: {sum(extrovert_mask)} ({sum(extrovert_mask)/len(test_df)*100:.1f}%)")
    print(f"Introvert: {sum(introvert_mask)} ({sum(introvert_mask)/len(test_df)*100:.1f}%)")
    
    # Analyze high confidence samples
    for threshold in [0.99, 0.995, 0.999]:
        n_high_conf_E = sum(conf_extrovert >= threshold)
        n_high_conf_I = sum(conf_introvert >= threshold)
        
        print(f"\nConfidence >= {threshold}:")
        print(f"  Extrovert: {n_high_conf_E} ({n_high_conf_E/sum(extrovert_mask)*100:.1f}% of E predictions)")
        print(f"  Introvert: {n_high_conf_I} ({n_high_conf_I/sum(introvert_mask)*100:.1f}% of I predictions)")
        print(f"  Total: {n_high_conf_E + n_high_conf_I}")
        print(f"  Ratio E:I = {n_high_conf_E}:{n_high_conf_I}")
    
    # Visualize
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Distribution of probabilities
    ax1.hist(proba_array[extrovert_mask], bins=50, alpha=0.5, label='Predicted E', color='red', density=True)
    ax1.hist(proba_array[introvert_mask], bins=50, alpha=0.5, label='Predicted I', color='blue', density=True)
    ax1.set_xlabel('P(Introvert)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Probabilities')
    ax1.legend()
    ax1.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    
    # Distribution of confidence
    ax2.hist(conf_extrovert, bins=50, alpha=0.5, label='Predicted E', color='red', density=True)
    ax2.hist(conf_introvert, bins=50, alpha=0.5, label='Predicted I', color='blue', density=True)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Confidence')
    ax2.legend()
    
    # Cumulative high confidence counts
    thresholds = np.arange(0.5, 1.01, 0.01)
    e_counts = [sum(conf_extrovert >= t) for t in thresholds]
    i_counts = [sum(conf_introvert >= t) for t in thresholds]
    
    ax3.plot(thresholds, e_counts, 'r-', label='Extrovert', linewidth=2)
    ax3.plot(thresholds, i_counts, 'b-', label='Introvert', linewidth=2)
    ax3.set_xlabel('Confidence Threshold')
    ax3.set_ylabel('Number of Samples')
    ax3.set_title('High Confidence Sample Counts')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0.995, color='black', linestyle='--', alpha=0.5, label='Threshold=0.995')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confidence_distribution_by_class.png', dpi=300)
    plt.close()
    
    # Analyze which features lead to high confidence
    print("\n" + "="*60)
    print("HIGH CONFIDENCE SAMPLE CHARACTERISTICS")
    print("="*60)
    
    # Get indices of highest confidence predictions
    high_conf_E_indices = np.where(extrovert_mask & (confidence >= 0.995))[0]
    high_conf_I_indices = np.where(introvert_mask & (confidence >= 0.995))[0]
    
    print(f"\nTop 5 highest confidence Extroverts:")
    for idx in high_conf_E_indices[:5]:
        print(f"  ID {test_df.iloc[idx]['id']}: conf={confidence[idx]:.4f}, prob_I={proba_array[idx]:.4f}")
        
    print(f"\nTop 5 highest confidence Introverts (if any):")
    if len(high_conf_I_indices) > 0:
        for idx in high_conf_I_indices[:5]:
            print(f"  ID {test_df.iloc[idx]['id']}: conf={confidence[idx]:.4f}, prob_I={proba_array[idx]:.4f}")
    else:
        print("  None found with confidence >= 0.995")
    
    # Check actual probability values
    print("\n" + "="*60)
    print("PROBABILITY EXTREMES")
    print("="*60)
    
    print(f"\nFor predicted Extroverts (P(I) should be < 0.5):")
    print(f"  Min P(I): {proba_array[extrovert_mask].min():.6f}")
    print(f"  Max P(I): {proba_array[extrovert_mask].max():.6f}")
    print(f"  Mean P(I): {proba_array[extrovert_mask].mean():.6f}")
    
    print(f"\nFor predicted Introverts (P(I) should be > 0.5):")
    print(f"  Min P(I): {proba_array[introvert_mask].min():.6f}")
    print(f"  Max P(I): {proba_array[introvert_mask].max():.6f}")
    print(f"  Mean P(I): {proba_array[introvert_mask].mean():.6f}")
    
    # Count extreme probabilities
    print(f"\nExtreme probabilities (0.0 or 1.0):")
    n_prob_0 = sum(proba_array == 0.0)
    n_prob_1 = sum(proba_array == 1.0)
    print(f"  P(I) = 0.0: {n_prob_0} samples (all predict E)")
    print(f"  P(I) = 1.0: {n_prob_1} samples (all predict I)")

if __name__ == "__main__":
    analyze_confidence_by_class()