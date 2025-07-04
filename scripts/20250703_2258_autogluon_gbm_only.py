#!/usr/bin/env python3
"""
PURPOSE: AutoGluon with only GBM (LightGBM) model
HYPOTHESIS: Focusing on a single high-performance model type might achieve target score
EXPECTED: Test GBM-only configurations with various hyperparameters
RESULT: Trained LightGBM models through AutoGluon with custom configurations
"""

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import time

print("AUTOGLUON - GBM ONLY")
print("="*60)

# Load data
print("\nLoading data...")
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

target = 'Personality'

# OPTION 1: Only GBM (LightGBM)
print("\n" + "="*60)
print("OPTION 1: ONLY GBM MODEL")
print("="*60)

predictor_gbm = TabularPredictor(
    label=target,
    eval_metric='accuracy',
    path='autogluon_gbm_only',
    verbosity=2
).fit(
    train_data=train_df,
    time_limit=600,  # 10 minutes
    
    # Specify only GBM
    hyperparameters={'GBM': {}},  # Empty dict = default GBM parameters
    
    # Optional: with CV
    num_bag_folds=5,  # 5-fold CV
    
    # No ensemble since we have only one model type
    num_stack_levels=0,
)

# OPTION 2: GBM with custom parameters
print("\n" + "="*60)
print("OPTION 2: GBM WITH CUSTOM PARAMETERS")
print("="*60)

predictor_gbm_custom = TabularPredictor(
    label=target,
    eval_metric='accuracy',
    path='autogluon_gbm_custom',
    verbosity=2
).fit(
    train_data=train_df,
    time_limit=600,
    
    # GBM with specific parameters
    hyperparameters={
        'GBM': {
            'num_boost_round': 1000,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0,
            'lambda_l2': 0,
        }
    },
    
    num_bag_folds=5,
)

# OPTION 3: Multiple specific models (for comparison)
print("\n" + "="*60)
print("OPTION 3: SPECIFIC MODEL SUBSET")
print("="*60)

predictor_subset = TabularPredictor(
    label=target,
    eval_metric='accuracy',
    path='autogluon_model_subset',
    verbosity=2
).fit(
    train_data=train_df,
    time_limit=1200,  # 20 minutes
    
    # Multiple specific models
    hyperparameters={
        'GBM': {},      # LightGBM
        'XGB': {},      # XGBoost  
        'CAT': {},      # CatBoost
        # 'RF': {},     # Random Forest (commented out)
        # 'XT': {},     # Extra Trees (commented out)
        # 'NN_TORCH': {},  # Neural Network (commented out)
    },
    
    num_bag_folds=5,
)

# Make predictions with GBM only model
print("\n" + "="*60)
print("MAKING PREDICTIONS")
print("="*60)

test_features = test_df.drop(columns=['id'] if 'id' in test_df.columns else [])
predictions = predictor_gbm.predict(test_features)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': predictions
})

filename = 'submission_AUTOGLUON_GBM_only.csv'
submission.to_csv(filename, index=False)
print(f"\nSaved: {filename}")

# Show results
print("\n" + "="*60)
print("GBM PERFORMANCE")
print("="*60)

leaderboard = predictor_gbm.leaderboard(train_df, silent=False)
print(leaderboard)

# Get specific model info
print("\n" + "="*60)
print("GBM MODEL DETAILS")
print("="*60)

model_info = predictor_gbm.info()
print(f"Number of models: {model_info['num_models']}")
print(f"Model names: {predictor_gbm.get_model_names()}")

# Feature importance from GBM
print("\n" + "="*60)
print("FEATURE IMPORTANCE (GBM)")
print("="*60)

importance = predictor_gbm.feature_importance(train_df)
print(importance)

print("\n" + "="*60)
print("AVAILABLE MODELS IN AUTOGLUON:")
print("="*60)
print("- GBM: LightGBM (Gradient Boosting Machine)")
print("- XGB: XGBoost") 
print("- CAT: CatBoost")
print("- RF: Random Forest")
print("- XT: Extra Trees")
print("- KNN: K-Nearest Neighbors")
print("- NN_TORCH: Neural Network (PyTorch)")
print("- FASTAI: Neural Network (FastAI)")
print("- LR: Linear/Logistic Regression")

print("\nTo use only specific models, use hyperparameters parameter:")
print("hyperparameters={'GBM': {}, 'XGB': {}}")  
print("\nTo exclude models, set them to None:")
print("hyperparameters={'GBM': {}, 'XGB': {}, 'RF': None}")