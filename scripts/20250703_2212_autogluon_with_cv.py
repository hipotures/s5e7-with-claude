#!/usr/bin/env python3
"""
PURPOSE: AutoGluon with Cross-Validation - multiple options
HYPOTHESIS: Cross-validation in AutoGluon can improve model generalization
EXPECTED: Test different CV strategies including bagging and holdout validation
RESULT: Implemented multiple CV approaches with AutoGluon for robust predictions
"""

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import time

print("AUTOGLUON WITH CROSS-VALIDATION")
print("="*60)

# Load data
print("\nLoading data...")
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

target = 'Personality'

# OPTION 1: Bagging with K-Fold (most common)
print("\n" + "="*60)
print("OPTION 1: BAGGING WITH K-FOLD CV")
print("="*60)

predictor_cv = TabularPredictor(
    label=target,
    eval_metric='accuracy',
    path='autogluon_cv_models',
    verbosity=2
).fit(
    train_data=train_df,
    time_limit=3600,
    presets='best_quality',
    
    # CV Configuration
    num_bag_folds=5,  # 5-fold CV
    num_bag_sets=1,   # Number of times to repeat k-fold (1 = standard k-fold)
    num_stack_levels=1,  # Stacking levels (0 = no stacking)
    
    # This ensures out-of-fold predictions for better generalization
    ag_args_fit={
        'num_gpus': 1,  # Use GPU if available
    }
)

print("\nCV Leaderboard:")
leaderboard_cv = predictor_cv.leaderboard(train_df, silent=False)
print(leaderboard_cv)

# OPTION 2: Hold-out validation (if you want specific split)
print("\n" + "="*60)
print("OPTION 2: HOLD-OUT VALIDATION")
print("="*60)

# Create validation split
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df[target]
)

predictor_holdout = TabularPredictor(
    label=target,
    eval_metric='accuracy',
    path='autogluon_holdout_models',
    verbosity=2
).fit(
    train_data=train_data,
    tuning_data=val_data,  # Explicit validation set
    time_limit=1800,  # 30 minutes for this example
    presets='best_quality'
)

# OPTION 3: Custom CV with fit_extra
print("\n" + "="*60)
print("OPTION 3: REFIT WITH DIFFERENT CV FOLDS")
print("="*60)

# You can refit models with different CV configurations
predictor_cv.fit_extra(
    hyperparameters='auto',
    time_limit=600,  # Additional 10 minutes
    num_bag_folds=10,  # Try 10-fold CV
)

# Make predictions with CV model
print("\n" + "="*60)
print("MAKING PREDICTIONS WITH CV MODEL")
print("="*60)

test_features = test_df.drop(columns=['id'] if 'id' in test_df.columns else [])
predictions_cv = predictor_cv.predict(test_features)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': predictions_cv
})

filename = 'submission_AUTOGLUON_CV_5fold.csv'
submission.to_csv(filename, index=False)
print(f"\nSaved: {filename}")

# Show detailed results
print("\n" + "="*60)
print("CV PERFORMANCE DETAILS")
print("="*60)

# Get out-of-fold predictions (OOF)
print("\nModel performance (with CV):")
model_performances = predictor_cv.leaderboard(silent=True)
print(model_performances[['model', 'score_val', 'pred_time_val', 'fit_time']].head(10))

# Feature importance with CV
print("\n" + "="*60)
print("FEATURE IMPORTANCE (from CV)")
print("="*60)
importance_cv = predictor_cv.feature_importance(train_df)
print(importance_cv)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Key CV parameters in AutoGluon:")
print("- num_bag_folds: Number of CV folds (default=0, no bagging)")
print("- num_bag_sets: Number of times to repeat k-fold")
print("- num_stack_levels: Stacking levels (0=no stacking)")
print("- ag_args_ensemble: Control ensemble strategies")
print("\nWith 5-fold CV, each model is trained 5 times")
print("Final prediction is typically the average of all folds")