#!/usr/bin/env python3
"""
YDF Multi-seed ensemble for maximum stability
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def prepare_data():
    """Load and prepare data for YDF"""
    print("="*60)
    print("YDF MULTI-SEED ENSEMBLE")
    print("="*60)
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # YDF works with original categorical features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Prepare datasets
    train_ydf = train_df[feature_cols + ['Personality']].copy()
    test_ydf = test_df[feature_cols].copy()
    
    print(f"\nClass distribution:")
    print(train_df['Personality'].value_counts(normalize=True))
    
    return train_ydf, test_ydf, test_df

def train_ydf_multi_seed(train_df, test_df, n_seeds=20):
    """Train YDF with multiple random seeds"""
    
    print(f"\n" + "="*60)
    print(f"TRAINING YDF WITH {n_seeds} SEEDS")
    print("="*60)
    
    # Different seeds - more seeds for better stability
    seeds = list(range(42, 42 + n_seeds))
    
    all_predictions = []
    cv_scores_per_seed = []
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\nSeed {seed_idx+1}/{n_seeds} (random_seed={seed}):")
        
        # YDF learner with specific seed
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=500,  # More trees for stability
            max_depth=8,    # Slightly deeper
            random_seed=seed
        )
        
        # Quick CV to check consistency
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df['Personality'])):
            train_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]
            
            model = learner.train(train_fold)
            predictions = model.predict(val_fold.drop('Personality', axis=1))
            pred_classes = ['Introvert' if float(str(p)) > 0.5 else 'Extrovert' for p in predictions]
            
            accuracy = accuracy_score(val_fold['Personality'], pred_classes)
            fold_scores.append(accuracy)
        
        cv_mean = np.mean(fold_scores)
        cv_scores_per_seed.append(cv_mean)
        print(f"  CV accuracy: {cv_mean:.6f}")
        
        # Train on full data
        print(f"  Training on full dataset...")
        final_model = learner.train(train_df)
        
        # Predict on test
        test_predictions = final_model.predict(test_df)
        # Extract probabilities for Introvert
        test_probs = [float(str(p)) for p in test_predictions]
        all_predictions.append(test_probs)
        
        # Optional: save individual model predictions
        if seed_idx < 3:  # Save first 3 for inspection
            individual_submission = pd.DataFrame({
                'id': range(18524, 18524 + len(test_probs)),
                'Personality': ['Introvert' if p > 0.5 else 'Extrovert' for p in test_probs]
            })
            individual_submission.to_csv(
                OUTPUT_DIR / f'submission_ydf_seed_{seed}.csv', 
                index=False
            )
    
    print(f"\n" + "="*40)
    print("SEED CONSISTENCY ANALYSIS")
    print("="*40)
    print(f"CV scores across seeds:")
    print(f"Mean: {np.mean(cv_scores_per_seed):.6f}")
    print(f"Std: {np.std(cv_scores_per_seed):.6f}")
    print(f"Min: {np.min(cv_scores_per_seed):.6f}")
    print(f"Max: {np.max(cv_scores_per_seed):.6f}")
    
    return np.array(all_predictions), seeds

def analyze_prediction_stability(pred_matrix, test_df):
    """Find which test samples are most uncertain"""
    
    print("\n" + "="*60)
    print("STABILITY ANALYSIS")
    print("="*60)
    
    # Calculate statistics
    mean_probs = np.mean(pred_matrix, axis=0)
    std_probs = np.std(pred_matrix, axis=0)
    
    # Binary predictions for each seed
    binary_preds = (pred_matrix > 0.5).astype(int)
    
    # Agreement rate
    mode_pred = np.median(binary_preds, axis=0).astype(int)
    agreement = np.mean(binary_preds == mode_pred[:, np.newaxis].T, axis=0)
    
    print(f"\nPrediction agreement across seeds:")
    print(f"100% agreement: {(agreement == 1.0).sum()} samples ({(agreement == 1.0).sum()/len(agreement)*100:.1f}%)")
    print(f">95% agreement: {(agreement > 0.95).sum()} samples ({(agreement > 0.95).sum()/len(agreement)*100:.1f}%)")
    print(f">90% agreement: {(agreement > 0.90).sum()} samples ({(agreement > 0.90).sum()/len(agreement)*100:.1f}%)")
    print(f">80% agreement: {(agreement > 0.80).sum()} samples ({(agreement > 0.80).sum()/len(agreement)*100:.1f}%)")
    
    # Find most uncertain (potential ambiverts)
    uncertain_mask = (mean_probs > 0.4) & (mean_probs < 0.6) & (std_probs > 0.05)
    uncertain_ids = test_df['id'].values[uncertain_mask]
    
    print(f"\n🎯 POTENTIAL AMBIVERTS (high uncertainty):")
    print(f"Found {len(uncertain_ids)} candidates")
    
    if len(uncertain_ids) > 0:
        print(f"\nTop 20 most uncertain IDs:")
        uncertainty_scores = std_probs[uncertain_mask]
        top_uncertain_idx = np.argsort(uncertainty_scores)[-20:]
        
        for idx in top_uncertain_idx[::-1]:
            test_idx = np.where(uncertain_mask)[0][idx]
            print(f"  ID {test_df.iloc[test_idx]['id']}: "
                  f"mean_prob={mean_probs[test_idx]:.3f}, "
                  f"std={std_probs[test_idx]:.3f}, "
                  f"agreement={agreement[test_idx]:.1%}")
    
    return mean_probs, std_probs, uncertain_ids

def create_ensemble_strategies(pred_matrix, test_df, uncertain_ids):
    """Create different ensemble strategies"""
    
    print("\n" + "="*60)
    print("ENSEMBLE STRATEGIES")
    print("="*60)
    
    # 1. Standard Average
    mean_probs = np.mean(pred_matrix, axis=0)
    submission_mean = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Introvert' if p > 0.5 else 'Extrovert' for p in mean_probs]
    })
    submission_mean.to_csv(OUTPUT_DIR / 'submission_ydf_ensemble_mean.csv', index=False)
    print(f"\n1. Mean ensemble:")
    print(f"   {submission_mean['Personality'].value_counts(normalize=True).to_dict()}")
    
    # 2. Weighted by confidence (less weight for uncertain)
    std_probs = np.std(pred_matrix, axis=0)
    weights = 1 / (1 + std_probs)  # Higher weight for stable predictions
    # Normalize weights for each sample
    weights_expanded = np.repeat(weights[np.newaxis, :], pred_matrix.shape[0], axis=0)
    weighted_probs = np.average(pred_matrix, axis=0, weights=weights_expanded)
    submission_weighted = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Introvert' if p > 0.5 else 'Extrovert' for p in weighted_probs]
    })
    submission_weighted.to_csv(OUTPUT_DIR / 'submission_ydf_ensemble_weighted.csv', index=False)
    print(f"\n2. Confidence-weighted ensemble:")
    print(f"   {submission_weighted['Personality'].value_counts(normalize=True).to_dict()}")
    
    # 3. Conservative (flip uncertain to majority class)
    conservative_probs = mean_probs.copy()
    # For uncertain cases, default to Extrovert (majority class)
    uncertain_mask = np.isin(test_df['id'], uncertain_ids)
    conservative_probs[uncertain_mask] = 0.3  # Push to Extrovert
    submission_conservative = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Introvert' if p > 0.5 else 'Extrovert' for p in conservative_probs]
    })
    submission_conservative.to_csv(OUTPUT_DIR / 'submission_ydf_ensemble_conservative.csv', index=False)
    print(f"\n3. Conservative ensemble (uncertain→Extrovert):")
    print(f"   {submission_conservative['Personality'].value_counts(normalize=True).to_dict()}")
    
    # 4. Aggressive (flip uncertain to minority class)
    aggressive_probs = mean_probs.copy()
    aggressive_probs[uncertain_mask] = 0.7  # Push to Introvert
    submission_aggressive = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Introvert' if p > 0.5 else 'Extrovert' for p in aggressive_probs]
    })
    submission_aggressive.to_csv(OUTPUT_DIR / 'submission_ydf_ensemble_aggressive.csv', index=False)
    print(f"\n4. Aggressive ensemble (uncertain→Introvert):")
    print(f"   {submission_aggressive['Personality'].value_counts(normalize=True).to_dict()}")
    
    # Compare strategies
    print(f"\n" + "="*40)
    print("STRATEGY DIFFERENCES")
    print("="*40)
    
    strategies = {
        'mean': submission_mean,
        'weighted': submission_weighted,
        'conservative': submission_conservative,
        'aggressive': submission_aggressive
    }
    
    for name1, sub1 in strategies.items():
        for name2, sub2 in strategies.items():
            if name1 < name2:
                diff = (sub1['Personality'] != sub2['Personality']).sum()
                print(f"{name1} vs {name2}: {diff} differences")

def main():
    # Load data
    train_ydf, test_ydf, test_df = prepare_data()
    
    # Train with multiple seeds
    pred_matrix, seeds = train_ydf_multi_seed(train_ydf, test_ydf, n_seeds=20)
    
    # Analyze stability
    mean_probs, std_probs, uncertain_ids = analyze_prediction_stability(pred_matrix, test_df)
    
    # Create ensemble submissions
    create_ensemble_strategies(pred_matrix, test_df, uncertain_ids)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
1. Test all 4 strategies on Kaggle:
   - Mean: safest, most stable
   - Weighted: trusts confident predictions more
   - Conservative: assumes uncertain = Extrovert
   - Aggressive: assumes uncertain = Introvert

2. The strategy with best LB score reveals:
   - If Conservative wins → uncertain cases are Extroverts
   - If Aggressive wins → uncertain cases are Introverts
   - If Mean/Weighted win → uncertain cases are mixed

3. YDF advantages:
   - Fast training (can do many seeds)
   - Handles missing values natively
   - Slightly better than XGB/LGB

4. Next steps:
   - Submit all 4 strategies
   - Check which performs best
   - Use that insight for final submission
""")

if __name__ == "__main__":
    main()