#!/usr/bin/env python3
"""
PURPOSE: Extended AutoGluon training with 2+ hours to find optimal ensemble for 0.976518
HYPOTHESIS: AutoGluon with extensive time and all models can break through the plateau
EXPECTED: Achieve breakthrough performance with sophisticated auto-ML
RESULT: [To be filled after execution]

NOTE: Designed for GPU server - uses AutoGluon with GPU support
Expected runtime: 2+ hours
"""

import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Prepare features - keep it simple for AutoGluon
train_df['label'] = (train_df['Personality'] == 'Extrovert').astype(int)
train_data = train_df.drop(['Personality'], axis=1)
test_data = test_df.copy()

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Load the always-wrong samples from error analysis
try:
    always_wrong_df = pd.read_csv('output/20250704_2318_always_wrong_samples.csv')
    always_wrong_ids = set(always_wrong_df['id'].values)
    print(f"\nLoaded {len(always_wrong_ids)} always-wrong sample IDs")
    # Create index mask for always-wrong samples
    always_wrong_indices = train_df[train_df['id'].isin(always_wrong_ids)].index
except:
    always_wrong_ids = set()
    always_wrong_indices = []
    print("\nNo always-wrong samples file found")

# Custom metric that penalizes errors on known difficult samples
def custom_accuracy(y_true, y_pred, sample_weight=None):
    """Custom metric that heavily weights the always-wrong samples"""
    if sample_weight is None:
        return (y_true == y_pred).mean()
    else:
        return np.average(y_true == y_pred, weights=sample_weight)

# Create sample weights (will be passed separately, not as a column)
if len(always_wrong_indices) > 0:
    # Give 10x weight to always-wrong samples
    sample_weights = np.ones(len(train_data))
    sample_weights[always_wrong_indices] = 10.0
    print(f"Applied 10x weight to {len(always_wrong_indices)} difficult samples")
else:
    sample_weights = None

# AutoGluon configuration for extended training
print("\n=== CONFIGURING AUTOGLUON ===")

# Hyperparameters for each model type
hyperparameters = {
    'GBM': [
        {'num_boost_round': 1000, 'learning_rate': 0.03, 'num_leaves': 127},
        {'num_boost_round': 500, 'learning_rate': 0.05, 'num_leaves': 63},
        {'num_boost_round': 300, 'learning_rate': 0.1, 'num_leaves': 31},
    ],
    'XGB': [
        {'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 8},
        {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6},
        {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 4},
    ],
    'CAT': [
        {'iterations': 1000, 'learning_rate': 0.03, 'depth': 8},
        {'iterations': 500, 'learning_rate': 0.05, 'depth': 6},
    ],
    'NN_TORCH': [
        {'num_epochs': 100, 'learning_rate': 0.001, 'dropout_prob': 0.3},
        {'num_epochs': 50, 'learning_rate': 0.005, 'dropout_prob': 0.2},
    ],
    'RF': [
        {'n_estimators': 300, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_features': 0.8},
    ],
    'XT': [
        {'n_estimators': 300, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_features': 0.8},
    ],
}

# Train with extended time limit
print("\n=== STARTING EXTENDED AUTOGLUON TRAINING ===")
print("This will take 2+ hours. Progress will be displayed below.")

start_time = time.time()

predictor = TabularPredictor(
    label='label',
    problem_type='binary',
    eval_metric='accuracy',
    path='AutoGluon_extended_models',
    verbosity=2
)

# Fit with extended configuration
predictor.fit(
    train_data=train_data,
    time_limit=7200,  # 2 hours
    presets='best_quality',
    hyperparameters=hyperparameters,
    num_bag_folds=5,
    num_bag_sets=2,
    num_stack_levels=2,
    excluded_model_types=['KNN'],  # Exclude KNN as it's slow for large datasets
    ag_args_fit={
        #'num_gpus': 1,  # Use GPU when available
        'sample_weight': sample_weights,  # Pass weights array, not column name
    },
    ag_args_ensemble={
        'fold_fitting_strategy': 'sequential_local',  # Better for difficult samples
    }
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time/3600:.2f} hours")

# Get model information
model_info = predictor.info()
print("\n=== MODEL INFORMATION ===")
print(f"Best model: {model_info['best_model']}")
print(f"Number of models trained: {len(predictor.model_names())}")

# Leaderboard
leaderboard = predictor.leaderboard(silent=False)
print("\n=== LEADERBOARD ===")
print(leaderboard.head(10))

# Feature importance
feature_importance = predictor.feature_importance(train_data)
print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance)

# Analyze predictions on always-wrong samples
if len(always_wrong_indices) > 0:
    print("\n=== ANALYZING DIFFICULT SAMPLES ===")
    wrong_samples = train_data.iloc[always_wrong_indices].copy()
    # No need to clean - sample_weight is not a column anymore
    wrong_preds = predictor.predict_proba(wrong_samples)
    wrong_predictions = predictor.predict(wrong_samples)
    wrong_accuracy = (wrong_predictions == wrong_samples['label']).mean()
    print(f"Accuracy on always-wrong samples: {wrong_accuracy:.4f}")
    print(f"Average probability on always-wrong: {wrong_preds[1].mean():.4f}")

# Generate test predictions
print("\n=== GENERATING TEST PREDICTIONS ===")

# Get probabilities
test_probs = predictor.predict_proba(test_data, as_multiclass=False)

# Apply custom threshold based on analysis
optimal_threshold = 0.35  # From error analysis

# Make predictions with rules for edge cases
test_predictions = np.zeros(len(test_data))

for i in range(len(test_data)):
    prob = test_probs.iloc[i]
    
    # Check for missing values (strong indicator of extrovert)
    if pd.isna(test_df.iloc[i]['Drained_after_socializing']):
        test_predictions[i] = 1  # Extrovert
    # Check for very high probability boundaries
    elif 0.96 <= prob <= 0.98:
        test_predictions[i] = 1  # Force to extrovert
    # Apply optimal threshold
    else:
        test_predictions[i] = int(prob >= optimal_threshold)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in test_predictions.astype(int)]
})

# Save submission
from datetime import datetime
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)

submission_path = f'subm/DATE_{date_str}/20250704_2319_autogluon_extended.csv'
submission.to_csv(submission_path, index=False)

# Save detailed results
results = {
    'training_time_hours': training_time / 3600,
    'best_model': model_info['best_model'],
    'num_models': len(predictor.model_names()),
    'leaderboard_top5': leaderboard.head(5).to_dict('records'),
    'feature_importance': feature_importance.to_dict(),
    'difficult_samples_accuracy': float(wrong_accuracy) if always_wrong_ids else None,
    'optimal_threshold': optimal_threshold,
    'prediction_distribution': {
        'introverts': int((test_predictions == 0).sum()),
        'extroverts': int((test_predictions == 1).sum())
    }
}

with open('output/20250704_2319_autogluon_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Feature analysis for breakthrough insights
print("\n=== BREAKTHROUGH ANALYSIS ===")

# Check if AutoGluon found any special patterns
predictor_internals = predictor.fit_summary()
print("\nFit summary:")
print(predictor_internals)

print(f"\n=== SUMMARY ===")
print(f"Training time: {training_time/3600:.2f} hours")
print(f"Models trained: {len(predictor.model_names())}")
print(f"Best model score: {leaderboard.iloc[0]['score_val']:.6f}")
print(f"Prediction distribution: {(test_predictions == 1).sum()}/{len(test_predictions)} extroverts")
print(f"Submission saved to: {submission_path}")

# RESULT: [To be filled after execution]
