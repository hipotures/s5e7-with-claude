#!/usr/bin/env python3
"""
Verify predictions are not reversed by comparing with YDF baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def create_ydf_baseline():
    """Create baseline YDF predictions for comparison"""
    print("Creating YDF baseline predictions...")
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Train YDF model (this should give ~0.975708 score)
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        random_seed=42
    )
    
    model = learner.train(train_df)
    
    # Make predictions
    predictions = model.predict(test_df)
    
    # Convert to class labels
    # YDF returns probabilities for Extrovert
    pred_list = []
    for p in predictions:
        # Convert to float if needed
        prob = float(str(p))
        # YDF returns P(Introvert), not P(Extrovert)!
        pred_list.append('Introvert' if prob > 0.5 else 'Extrovert')
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': pred_list
    })
    
    submission.to_csv(OUTPUT_DIR / 'ydf_baseline_submission.csv', index=False)
    
    return submission

def compare_submissions():
    """Compare different submissions to check for E-I reversal"""
    
    # Create YDF baseline
    ydf_baseline = create_ydf_baseline()
    
    # Load ensemble submission
    ensemble_sub = pd.read_csv(OUTPUT_DIR / 'submission_ensemble_equal.csv')
    
    # Load sample submission
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    
    print("\n" + "="*60)
    print("SUBMISSION COMPARISON")
    print("="*60)
    
    # Compare distributions
    print("\nPersonality distributions:")
    print(f"YDF baseline: {ydf_baseline['Personality'].value_counts().to_dict()}")
    print(f"Ensemble: {ensemble_sub['Personality'].value_counts().to_dict()}")
    print(f"Sample (all E): {sample_sub['Personality'].value_counts().to_dict()}")
    
    # Check agreement
    ydf_ensemble_agree = sum(ydf_baseline['Personality'] == ensemble_sub['Personality'])
    print(f"\nYDF vs Ensemble agreement: {ydf_ensemble_agree}/{len(ydf_baseline)} ({ydf_ensemble_agree/len(ydf_baseline)*100:.1f}%)")
    
    # If agreement is very low, likely reversed
    if ydf_ensemble_agree < len(ydf_baseline) * 0.2:
        print("\n⚠️ WARNING: Very low agreement suggests possible E-I reversal!")
    
    # Sample first 20 predictions
    print("\nFirst 20 predictions comparison:")
    print(f"{'ID':>6} {'YDF':>10} {'Ensemble':>10} {'Match':>7}")
    print("-"*40)
    
    for i in range(20):
        match = '✓' if ydf_baseline.iloc[i]['Personality'] == ensemble_sub.iloc[i]['Personality'] else '✗'
        print(f"{ydf_baseline.iloc[i]['id']:>6} {ydf_baseline.iloc[i]['Personality']:>10} "
              f"{ensemble_sub.iloc[i]['Personality']:>10} {match:>7}")
    
    # Check specific IDs we know
    print("\n" + "="*60)
    print("KNOWN IDS CHECK")
    print("="*60)
    
    known_ids = [20934, 18634, 20932, 21138, 20728]
    print(f"{'ID':>6} {'YDF':>10} {'Ensemble':>10}")
    print("-"*30)
    
    for known_id in known_ids:
        ydf_pred = ydf_baseline[ydf_baseline['id'] == known_id]['Personality'].values[0]
        ens_pred = ensemble_sub[ensemble_sub['id'] == known_id]['Personality'].values[0]
        print(f"{known_id:>6} {ydf_pred:>10} {ens_pred:>10}")
    
    # Test if YDF gives expected score
    print("\n" + "="*60)
    print("EXPECTED SCORES")
    print("="*60)
    print("YDF baseline should score ~0.975708")
    print("If significantly different, check for errors")
    
    return ydf_baseline, ensemble_sub

def check_all_submissions():
    """Check all created submissions for consistency"""
    print("\n" + "="*60)
    print("CHECKING ALL SUBMISSIONS")
    print("="*60)
    
    submission_files = list(SCORES_DIR.glob('*.csv'))
    
    if len(submission_files) == 0:
        print("No submissions found in scores/")
        return
    
    # Load first submission as reference
    ref_sub = pd.read_csv(submission_files[0])
    
    for sub_file in submission_files[1:]:
        sub = pd.read_csv(sub_file)
        
        # Check distribution
        e_count = sum(sub['Personality'] == 'Extrovert')
        i_count = sum(sub['Personality'] == 'Introvert')
        
        print(f"\n{sub_file.name}:")
        print(f"  Extrovert: {e_count} ({e_count/len(sub)*100:.1f}%)")
        print(f"  Introvert: {i_count} ({i_count/len(sub)*100:.1f}%)")
        
        # Check differences from reference
        n_diff = sum(sub['Personality'] != ref_sub['Personality'])
        print(f"  Differences from reference: {n_diff}")

if __name__ == "__main__":
    # Compare main submissions
    ydf_baseline, ensemble_sub = compare_submissions()
    
    # Check all submissions
    check_all_submissions()
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)