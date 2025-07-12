#!/usr/bin/env python3
"""
Multi-seed ensemble strategy for stable predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import ydf
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def prepare_data():
    """Load and prepare data"""
    print("="*60)
    print("MULTI-SEED ENSEMBLE STRATEGY")
    print("="*60)
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Encode categorical
    for df in [train_df, test_df]:
        df['Stage_fear_encoded'] = (df['Stage_fear'] == 'Yes').astype(int)
        df['Drained_encoded'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    # Features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear_encoded', 'Drained_encoded']
    
    # Handle missing values
    for col in feature_cols[:5]:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())
    
    X_train = train_df[feature_cols].values
    y_train = (train_df['Personality'] == 'Introvert').astype(int).values
    X_test = test_df[feature_cols].values
    
    return X_train, y_train, X_test, test_df

def train_multi_seed_models(X_train, y_train, X_test, n_seeds=10):
    """Train same model with different random seeds"""
    
    print(f"\nTraining models with {n_seeds} different seeds...")
    
    all_predictions = []
    
    # Different seeds
    seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021][:n_seeds]
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\nSeed {seed_idx+1}/{n_seeds} (seed={seed}):")
        
        # Train multiple models with this seed
        models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                device='cuda:0',
                tree_method='hist',
                random_state=seed
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                device='gpu',
                random_state=seed,
                verbose=-1
            ),
            'cat': CatBoostClassifier(
                iterations=300,
                depth=5,
                learning_rate=0.05,
                task_type='GPU',
                random_seed=seed,
                verbose=False
            )
        }
        
        seed_predictions = []
        
        for name, model in models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Predict probabilities
            pred_proba = model.predict_proba(X_test)[:, 1]
            seed_predictions.append(pred_proba)
            
            print(f"  {name}: trained")
        
        # Average predictions for this seed
        seed_avg = np.mean(seed_predictions, axis=0)
        all_predictions.append(seed_avg)
    
    return all_predictions, seeds

def analyze_prediction_stability(all_predictions, X_test):
    """Analyze which predictions are stable across seeds"""
    
    print("\n" + "="*60)
    print("PREDICTION STABILITY ANALYSIS")
    print("="*60)
    
    # Convert to array
    pred_matrix = np.array(all_predictions)  # shape: (n_seeds, n_samples)
    
    # Calculate statistics
    mean_preds = np.mean(pred_matrix, axis=0)
    std_preds = np.std(pred_matrix, axis=0)
    
    # Binary predictions for each seed
    binary_preds = (pred_matrix > 0.5).astype(int)
    
    # How often each sample gets same prediction
    consistency = np.mean(binary_preds == binary_preds[0], axis=0)
    
    print(f"\nPrediction consistency across seeds:")
    print(f"100% consistent: {(consistency == 1.0).sum()} samples ({(consistency == 1.0).sum()/len(consistency)*100:.1f}%)")
    print(f">90% consistent: {(consistency > 0.9).sum()} samples ({(consistency > 0.9).sum()/len(consistency)*100:.1f}%)")
    print(f">80% consistent: {(consistency > 0.8).sum()} samples ({(consistency > 0.8).sum()/len(consistency)*100:.1f}%)")
    
    # Find most uncertain predictions (high std)
    uncertain_idx = np.argsort(std_preds)[-20:]
    
    print(f"\nMost uncertain predictions (high variance across seeds):")
    print(f"Sample indices: {uncertain_idx}")
    print(f"Their std: {std_preds[uncertain_idx]}")
    print(f"Their mean pred: {mean_preds[uncertain_idx]}")
    
    # These are potential ambiverts!
    potential_ambiverts = np.where((mean_preds > 0.45) & (mean_preds < 0.55) & (std_preds > 0.05))[0]
    print(f"\nPotential ambiverts (near 50-50 with high variance): {len(potential_ambiverts)} samples")
    
    return mean_preds, std_preds, consistency

def create_ensemble_submissions(all_predictions, test_df, seeds):
    """Create different ensemble strategies"""
    
    print("\n" + "="*60)
    print("CREATING ENSEMBLE SUBMISSIONS")
    print("="*60)
    
    pred_matrix = np.array(all_predictions)
    
    # Strategy 1: Simple average
    mean_preds = np.mean(pred_matrix, axis=0)
    submission_mean = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Introvert' if p > 0.5 else 'Extrovert' for p in mean_preds]
    })
    submission_mean.to_csv(OUTPUT_DIR / 'submission_ensemble_mean.csv', index=False)
    print(f"\n1. Mean ensemble saved")
    print(f"   Distribution: {submission_mean['Personality'].value_counts(normalize=True).to_dict()}")
    
    # Strategy 2: Majority voting
    binary_preds = (pred_matrix > 0.5).astype(int)
    majority_vote = np.mean(binary_preds, axis=0) > 0.5
    submission_vote = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Introvert' if p else 'Extrovert' for p in majority_vote]
    })
    submission_vote.to_csv(OUTPUT_DIR / 'submission_ensemble_vote.csv', index=False)
    print(f"\n2. Majority vote ensemble saved")
    print(f"   Distribution: {submission_vote['Personality'].value_counts(normalize=True).to_dict()}")
    
    # Strategy 3: Conservative (high confidence only)
    std_preds = np.std(pred_matrix, axis=0)
    conservative_preds = mean_preds.copy()
    # For high uncertainty cases, default to majority class (Extrovert)
    conservative_preds[std_preds > 0.1] = 0.3  # Push towards Extrovert
    submission_conservative = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Introvert' if p > 0.5 else 'Extrovert' for p in conservative_preds]
    })
    submission_conservative.to_csv(OUTPUT_DIR / 'submission_ensemble_conservative.csv', index=False)
    print(f"\n3. Conservative ensemble saved")
    print(f"   Distribution: {submission_conservative['Personality'].value_counts(normalize=True).to_dict()}")
    
    # Compare differences
    diff_mean_vote = (submission_mean['Personality'] != submission_vote['Personality']).sum()
    diff_mean_cons = (submission_mean['Personality'] != submission_conservative['Personality']).sum()
    
    print(f"\nDifferences between strategies:")
    print(f"Mean vs Vote: {diff_mean_vote} different predictions")
    print(f"Mean vs Conservative: {diff_mean_cons} different predictions")

def main():
    # Prepare data
    X_train, y_train, X_test, test_df = prepare_data()
    
    # Train with multiple seeds
    all_predictions, seeds = train_multi_seed_models(X_train, y_train, X_test, n_seeds=10)
    
    # Analyze stability
    mean_preds, std_preds, consistency = analyze_prediction_stability(all_predictions, X_test)
    
    # Create ensemble submissions
    create_ensemble_submissions(all_predictions, test_df, seeds)
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("""
1. Stable predictions (same across seeds) = confident cases
2. Unstable predictions (vary across seeds) = potential ambiverts
3. Ensemble reduces variance and improves generalization
4. Different strategies for risk management:
   - Mean: balanced approach
   - Vote: discrete decisions
   - Conservative: avoid risky predictions

This approach maximizes stability for private test set!
""")

if __name__ == "__main__":
    main()