#!/usr/bin/env python3
"""
Train models using original + synthetic + VAE-generated data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Paths
ORIG_DIR = Path("/mnt/ml/kaggle/original-personality-data/")
SYNTH_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def load_all_datasets():
    """Load original, synthetic, and VAE-generated data"""
    
    print("="*60)
    print("LOADING ALL DATASETS")
    print("="*60)
    
    # 1. Original data (with missing values)
    orig_df = pd.read_csv(ORIG_DIR / "personality_dataset.csv")
    orig_df['source'] = 'original'
    print(f"\nOriginal data: {orig_df.shape}")
    
    # 2. Original data with imputation
    orig_imputed_df = pd.read_csv(ORIG_DIR / "personality_datasert.csv")
    orig_imputed_df['source'] = 'original_imputed'
    print(f"Original (imputed): {orig_imputed_df.shape}")
    
    # 3. Competition synthetic data
    synth_df = pd.read_csv(SYNTH_DIR / "train.csv")
    synth_df['source'] = 'synthetic_competition'
    print(f"Synthetic (competition): {synth_df.shape}")
    
    # 4. VAE-generated data
    vae_df = pd.read_csv(OUTPUT_DIR / "synthetic_vae_generated.csv")
    vae_df['source'] = 'vae_generated'
    print(f"VAE-generated: {vae_df.shape}")
    
    # Combine all
    all_data = pd.concat([
        orig_df,
        orig_imputed_df,
        synth_df,
        vae_df
    ], ignore_index=True)
    
    print(f"\nCombined dataset: {all_data.shape}")
    print("\nSource distribution:")
    print(all_data['source'].value_counts())
    
    return all_data, synth_df

def prepare_features(df, is_test=False):
    """Prepare features for training"""
    
    # Feature columns
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Encode categorical
    df['Stage_fear_encoded'] = (df['Stage_fear'] == 'Yes').astype(int)
    df['Drained_encoded'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    # Numeric features
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency',
                       'Stage_fear_encoded', 'Drained_encoded']
    
    # Handle missing values
    for col in numeric_features[:5]:  # Only numeric columns might have NaN
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    X = df[numeric_features].values
    
    if is_test:
        return X, None, numeric_features
    else:
        y = (df['Personality'] == 'Introvert').astype(int).values
        return X, y, numeric_features

def train_models_on_combined_data(all_data, test_df):
    """Train different models on combined dataset"""
    
    print("\n" + "="*60)
    print("TRAINING MODELS ON COMBINED DATA")
    print("="*60)
    
    # Prepare data
    X_train, y_train, feature_names = prepare_features(all_data)
    X_test, _, _ = prepare_features(test_df, is_test=True)
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Models to train
    models = {
        'xgb_combined': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            device='cuda:0',
            tree_method='hist',
            random_state=42
        ),
        'lgb_combined': lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            device='gpu',
            random_state=42
        ),
        'cat_combined': CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            task_type='GPU',
            random_seed=42,
            verbose=False
        )
    }
    
    predictions = {}
    
    # Cross-validation on synthetic data only (for fair comparison)
    synth_mask = all_data['source'] == 'synthetic_competition'
    X_synth = X_train[synth_mask]
    y_synth = y_train[synth_mask]
    
    print("\nCross-validation on synthetic data only:")
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # CV on synthetic
        cv_scores = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_synth, y_synth)):
            X_fold_train = X_synth[train_idx]
            y_fold_train = y_synth[train_idx]
            X_fold_val = X_synth[val_idx]
            y_fold_val = y_synth[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            val_pred = model.predict(X_fold_val)
            val_score = accuracy_score(y_fold_val, val_pred)
            cv_scores.append(val_score)
            print(f"  Fold {fold+1}: {val_score:.6f}")
        
        print(f"  Mean CV: {np.mean(cv_scores):.6f}")
        
        # Train on full combined dataset
        print(f"  Training on full combined dataset...")
        model.fit(X_train, y_train)
        
        # Predict on test
        test_pred = model.predict(X_test)
        predictions[name] = test_pred
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for feat, imp in zip(feature_names, importances):
                print(f"    {feat}: {imp:.4f}")
    
    return predictions

def create_ensemble_submission(predictions, test_df):
    """Create ensemble submission"""
    
    print("\n" + "="*60)
    print("CREATING ENSEMBLE SUBMISSION")
    print("="*60)
    
    # Majority voting
    pred_matrix = np.column_stack(list(predictions.values()))
    ensemble_pred = (pred_matrix.mean(axis=1) > 0.5).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ensemble_pred
    })
    
    submission['Personality'] = submission['Personality'].map({
        0: 'Extrovert',
        1: 'Introvert'
    })
    
    # Save
    output_file = OUTPUT_DIR / "submission_combined_data_ensemble.csv"
    submission.to_csv(output_file, index=False)
    print(f"\nSaved submission to: {output_file}")
    
    # Check distribution
    print(f"\nPrediction distribution:")
    print(submission['Personality'].value_counts(normalize=True))
    
    return submission

def analyze_data_impact():
    """Analyze how different data sources impact predictions"""
    
    print("\n" + "="*60)
    print("DATA SOURCE IMPACT ANALYSIS")
    print("="*60)
    
    print("""
Key questions:
1. Does original data help? (only 2,900 samples)
2. Does VAE-generated data improve generalization?
3. Is the class imbalance in synthetic data intentional?
4. Should we weight samples by source?

Insights:
- Original data is more balanced (48.6% vs 26% introverts)
- Synthetic shifted distributions (more extraverted features)
- VAE created 50-50 balanced data
- Combined training might reduce overfitting to synthetic patterns
""")

def main():
    # Load all datasets
    all_data, synth_df = load_all_datasets()
    
    # Load test data
    test_df = pd.read_csv(SYNTH_DIR / "test.csv")
    
    # Train models
    predictions = train_models_on_combined_data(all_data, test_df)
    
    # Create submission
    submission = create_ensemble_submission(predictions, test_df)
    
    # Analyze impact
    analyze_data_impact()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nHypothesis: Training on diverse data sources might")
    print("improve generalization and break the 0.975708 barrier.")

if __name__ == "__main__":
    main()