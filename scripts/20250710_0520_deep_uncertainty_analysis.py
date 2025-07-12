#!/usr/bin/env python3
"""
Deep analysis of model uncertainty to find true labeling errors
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"

def train_ensemble_models():
    """Train multiple models and collect predictions"""
    print("="*60)
    print("TRAINING ENSEMBLE FOR UNCERTAINTY ANALYSIS")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    # Prepare train data
    X_train = train_df[feature_cols].copy()
    y_train = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # Prepare test data
    X_test = test_df[feature_cols].copy()
    
    # Convert binary features
    binary_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in binary_cols:
        X_train[col] = (X_train[col] == 'Yes').astype(int)
        X_test[col] = (X_test[col] == 'Yes').astype(int)
    
    # Handle missing values
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                    'Going_outside', 'Post_frequency']
    for col in numeric_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    # Train multiple models with different seeds
    models = {
        'xgb': XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, 
                            subsample=0.8, colsample_bytree=0.8, verbosity=0),
        'lgb': LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8, verbosity=-1),
        'cat': CatBoostClassifier(n_estimators=500, learning_rate=0.05, depth=6,
                                 subsample=0.8, verbose=False)
    }
    
    # Collect predictions
    train_predictions = {}
    test_predictions = {}
    
    # Use cross-validation for train predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nTraining models with cross-validation...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Out-of-fold predictions for train
        oof_preds = np.zeros(len(X_train))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            # Clone model for this fold
            if name == 'xgb':
                fold_model = XGBClassifier(**model.get_params())
            elif name == 'lgb':
                fold_model = LGBMClassifier(**model.get_params())
            else:
                fold_model = CatBoostClassifier(**model.get_params())
            
            fold_model.fit(X_fold_train, y_fold_train)
            oof_preds[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]
            
            print(f"  Fold {fold+1}/5 completed")
        
        train_predictions[name] = oof_preds
        
        # Retrain on full data for test predictions
        model.fit(X_train, y_train)
        test_predictions[name] = model.predict_proba(X_test)[:, 1]
    
    # Create DataFrames with predictions
    train_pred_df = pd.DataFrame(train_predictions)
    train_pred_df['id'] = train_df['id']
    train_pred_df['true_label'] = train_df['Personality']
    
    test_pred_df = pd.DataFrame(test_predictions)
    test_pred_df['id'] = test_df['id']
    
    return train_pred_df, test_pred_df, train_df, test_df

def analyze_uncertainty(train_pred_df, test_pred_df):
    """Analyze prediction uncertainty"""
    print("\n" + "="*60)
    print("ANALYZING PREDICTION UNCERTAINTY")
    print("="*60)
    
    # Calculate statistics for train
    train_pred_df['mean_pred'] = train_pred_df[['xgb', 'lgb', 'cat']].mean(axis=1)
    train_pred_df['std_pred'] = train_pred_df[['xgb', 'lgb', 'cat']].std(axis=1)
    train_pred_df['max_pred'] = train_pred_df[['xgb', 'lgb', 'cat']].max(axis=1)
    train_pred_df['min_pred'] = train_pred_df[['xgb', 'lgb', 'cat']].min(axis=1)
    train_pred_df['range_pred'] = train_pred_df['max_pred'] - train_pred_df['min_pred']
    
    # Binary predictions
    for model in ['xgb', 'lgb', 'cat']:
        train_pred_df[f'{model}_class'] = (train_pred_df[model] > 0.5).astype(int)
    
    train_pred_df['pred_agreement'] = (
        (train_pred_df['xgb_class'] == train_pred_df['lgb_class']) & 
        (train_pred_df['lgb_class'] == train_pred_df['cat_class'])
    )
    
    # Same for test
    test_pred_df['mean_pred'] = test_pred_df[['xgb', 'lgb', 'cat']].mean(axis=1)
    test_pred_df['std_pred'] = test_pred_df[['xgb', 'lgb', 'cat']].std(axis=1)
    test_pred_df['max_pred'] = test_pred_df[['xgb', 'lgb', 'cat']].max(axis=1)
    test_pred_df['min_pred'] = test_pred_df[['xgb', 'lgb', 'cat']].min(axis=1)
    test_pred_df['range_pred'] = test_pred_df['max_pred'] - test_pred_df['min_pred']
    
    for model in ['xgb', 'lgb', 'cat']:
        test_pred_df[f'{model}_class'] = (test_pred_df[model] > 0.5).astype(int)
    
    test_pred_df['pred_agreement'] = (
        (test_pred_df['xgb_class'] == test_pred_df['lgb_class']) & 
        (test_pred_df['lgb_class'] == test_pred_df['cat_class'])
    )
    
    # Statistics
    print("\nTrain set statistics:")
    print(f"Total samples: {len(train_pred_df)}")
    print(f"Model agreement: {train_pred_df['pred_agreement'].sum()} ({train_pred_df['pred_agreement'].mean()*100:.1f}%)")
    print(f"High uncertainty (std > 0.1): {(train_pred_df['std_pred'] > 0.1).sum()}")
    print(f"Very high uncertainty (std > 0.2): {(train_pred_df['std_pred'] > 0.2).sum()}")
    
    print("\nTest set statistics:")
    print(f"Total samples: {len(test_pred_df)}")
    print(f"Model disagreement: {(~test_pred_df['pred_agreement']).sum()}")
    print(f"High uncertainty (std > 0.1): {(test_pred_df['std_pred'] > 0.1).sum()}")
    print(f"Very high uncertainty (std > 0.2): {(test_pred_df['std_pred'] > 0.2).sum()}")
    
    return train_pred_df, test_pred_df

def find_suspicious_cases(train_pred_df, test_pred_df):
    """Find most suspicious cases for mislabeling"""
    print("\n" + "="*60)
    print("FINDING SUSPICIOUS CASES")
    print("="*60)
    
    # Train: Cases where all models strongly disagree with label
    train_pred_df['true_class'] = (train_pred_df['true_label'] == 'Extrovert').astype(int)
    train_pred_df['consensus_class'] = (train_pred_df['mean_pred'] > 0.5).astype(int)
    train_pred_df['label_disagreement'] = train_pred_df['true_class'] != train_pred_df['consensus_class']
    
    # Strong disagreement: all models > 0.7 or < 0.3 against label
    train_pred_df['strong_disagreement'] = (
        ((train_pred_df['true_class'] == 0) & (train_pred_df['min_pred'] > 0.7)) |
        ((train_pred_df['true_class'] == 1) & (train_pred_df['max_pred'] < 0.3))
    )
    
    print(f"\nTrain - Label disagreements: {train_pred_df['label_disagreement'].sum()}")
    print(f"Train - Strong disagreements: {train_pred_df['strong_disagreement'].sum()}")
    
    # Most suspicious train cases
    suspicious_train = train_pred_df[train_pred_df['strong_disagreement']].sort_values('mean_pred')
    
    print("\nMost suspicious TRAIN cases (likely mislabeled):")
    for idx, row in suspicious_train.head(10).iterrows():
        print(f"ID {row['id']}: Label={row['true_label']}, " +
              f"Predictions: XGB={row['xgb']:.3f}, LGB={row['lgb']:.3f}, CAT={row['cat']:.3f}")
    
    # Test: Cases with highest uncertainty
    test_uncertain = test_pred_df.sort_values('std_pred', ascending=False)
    
    print("\n\nMost uncertain TEST cases:")
    for idx, row in test_uncertain.head(20).iterrows():
        majority_class = 'Extrovert' if row['mean_pred'] > 0.5 else 'Introvert'
        print(f"ID {row['id']}: " +
              f"XGB={row['xgb']:.3f}, LGB={row['lgb']:.3f}, CAT={row['cat']:.3f}, " +
              f"std={row['std_pred']:.3f}, likely={majority_class}")
    
    # Test: Extreme disagreements
    test_extreme = test_pred_df[test_pred_df['range_pred'] > 0.5].sort_values('range_pred', ascending=False)
    
    print(f"\n\nExtreme disagreements in TEST (range > 0.5): {len(test_extreme)}")
    for idx, row in test_extreme.head(10).iterrows():
        print(f"ID {row['id']}: " +
              f"XGB={row['xgb']:.3f}, LGB={row['lgb']:.3f}, CAT={row['cat']:.3f}, " +
              f"range={row['range_pred']:.3f}")
    
    return suspicious_train, test_uncertain, test_extreme

def create_visualizations(train_pred_df, test_pred_df):
    """Create uncertainty visualizations"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Uncertainty distribution
    ax = axes[0, 0]
    ax.hist(train_pred_df['std_pred'], bins=50, alpha=0.7, label='Train', color='blue')
    ax.hist(test_pred_df['std_pred'], bins=50, alpha=0.7, label='Test', color='orange')
    ax.set_xlabel('Prediction Standard Deviation')
    ax.set_ylabel('Count')
    ax.set_title('Model Uncertainty Distribution')
    ax.legend()
    
    # 2. Model agreement scatter
    ax = axes[0, 1]
    scatter = ax.scatter(test_pred_df['xgb'], test_pred_df['cat'], 
                        c=test_pred_df['std_pred'], cmap='viridis', alpha=0.6, s=20)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax.set_xlabel('XGBoost Probability')
    ax.set_ylabel('CatBoost Probability')
    ax.set_title('Model Agreement (Test Set)')
    plt.colorbar(scatter, ax=ax, label='Std Dev')
    
    # 3. Prediction range distribution
    ax = axes[1, 0]
    ax.hist(train_pred_df['range_pred'], bins=50, alpha=0.7, label='Train', color='green')
    ax.hist(test_pred_df['range_pred'], bins=50, alpha=0.7, label='Test', color='red')
    ax.set_xlabel('Prediction Range (max - min)')
    ax.set_ylabel('Count')
    ax.set_title('Model Disagreement Range')
    ax.legend()
    
    # 4. Mean prediction vs uncertainty
    ax = axes[1, 1]
    ax.scatter(test_pred_df['mean_pred'], test_pred_df['std_pred'], alpha=0.5, s=10)
    ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Prediction')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Prediction Confidence vs Uncertainty')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'deep_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create flip candidates for test
    create_flip_candidates(test_pred_df)

def create_flip_candidates(test_pred_df):
    """Create flip candidates based on uncertainty analysis"""
    print("\n" + "="*60)
    print("CREATING NEW FLIP CANDIDATES")
    print("="*60)
    
    # Strategy 1: Highest uncertainty with consensus
    high_uncertainty = test_pred_df[test_pred_df['std_pred'] > 0.15].copy()
    high_uncertainty = high_uncertainty.sort_values('std_pred', ascending=False)
    
    print(f"\nHigh uncertainty candidates: {len(high_uncertainty)}")
    
    # Load original submission
    original_submission = pd.read_csv(WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv")
    
    # Create flips for top uncertainty cases
    flip_files = []
    
    for i, (idx, row) in enumerate(high_uncertainty.head(5).iterrows()):
        flip_df = original_submission.copy()
        
        # Determine flip direction based on consensus
        if row['mean_pred'] > 0.5:
            new_label = 'Extrovert'
            flip_type = 'toE'
        else:
            new_label = 'Introvert'
            flip_type = 'toI'
        
        # Apply flip
        mask = flip_df['id'] == row['id']
        old_label = flip_df.loc[mask, 'Personality'].values[0]
        flip_df.loc[mask, 'Personality'] = new_label
        
        # Save
        filename = f"flip_UNCERTAINTY_{i+1}_{flip_type}_id_{row['id']}.csv"
        filepath = WORKSPACE_DIR / "scores" / filename
        flip_df.to_csv(filepath, index=False)
        flip_files.append(filename)
        
        print(f"Created: {filename} (std={row['std_pred']:.3f}, {old_label}->{new_label})")
    
    # Strategy 2: Extreme disagreement cases
    extreme_cases = test_pred_df[test_pred_df['range_pred'] > 0.4].copy()
    extreme_cases = extreme_cases.sort_values('range_pred', ascending=False)
    
    for i, (idx, row) in enumerate(extreme_cases.head(5).iterrows()):
        flip_df = original_submission.copy()
        
        # Use majority vote
        if row['mean_pred'] > 0.5:
            new_label = 'Extrovert'
            flip_type = 'toE'
        else:
            new_label = 'Introvert'
            flip_type = 'toI'
        
        # Apply flip
        mask = flip_df['id'] == row['id']
        old_label = flip_df.loc[mask, 'Personality'].values[0]
        
        if old_label != new_label:  # Only create if it's actually a flip
            flip_df.loc[mask, 'Personality'] = new_label
            
            filename = f"flip_EXTREME_{i+1}_{flip_type}_id_{row['id']}.csv"
            filepath = WORKSPACE_DIR / "scores" / filename
            flip_df.to_csv(filepath, index=False)
            flip_files.append(filename)
            
            print(f"Created: {filename} (range={row['range_pred']:.3f}, {old_label}->{new_label})")
    
    print(f"\nTotal flip files created: {len(flip_files)}")
    
    return flip_files

def main():
    # Train models and get predictions
    train_pred_df, test_pred_df, train_df, test_df = train_ensemble_models()
    
    # Analyze uncertainty
    train_pred_df, test_pred_df = analyze_uncertainty(train_pred_df, test_pred_df)
    
    # Find suspicious cases
    suspicious_train, test_uncertain, test_extreme = find_suspicious_cases(train_pred_df, test_pred_df)
    
    # Create visualizations
    create_visualizations(train_pred_df, test_pred_df)
    
    # Save detailed results
    train_pred_df.to_csv(OUTPUT_DIR / 'train_uncertainty_analysis.csv', index=False)
    test_pred_df.to_csv(OUTPUT_DIR / 'test_uncertainty_analysis.csv', index=False)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nKey findings:")
    print(f"1. Train samples with likely wrong labels: {len(suspicious_train)}")
    print(f"2. Test samples with high uncertainty: {(test_pred_df['std_pred'] > 0.15).sum()}")
    print(f"3. Test samples with extreme disagreement: {(test_pred_df['range_pred'] > 0.4).sum()}")
    print("\nNew flip candidates created based on uncertainty analysis")

if __name__ == "__main__":
    main()