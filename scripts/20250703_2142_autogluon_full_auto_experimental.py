#!/usr/bin/env python3
"""
PURPOSE: AutoGluon full automatic mode - 3600 seconds, no limits
HYPOTHESIS: Given enough time, AutoGluon can find the optimal model configuration
EXPECTED: AutoGluon will explore all model types and find best configuration
RESULT: Ran AutoGluon in experimental quality mode for comprehensive model search
"""

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import time

print("AUTOGLUON FULL AUTOMATIC MODE")
print("="*60)
print("Time limit: 3600 seconds (1 hour)")
print("No model limits - AutoGluon will decide everything")
print("="*60)

# Load data
print("\nLoading data...")
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# AutoGluon will handle everything - no preprocessing needed!
# Just specify the target column
target = 'Personality'

print(f"\nTarget column: {target}")
print(f"Features: AutoGluon will decide")
print(f"Preprocessing: AutoGluon will handle")

# Create predictor with best quality preset
print("\n" + "-"*60)
print("Starting AutoGluon training...")
print("-"*60)

start_time = time.time()

predictor = TabularPredictor(
    label=target,
    eval_metric='accuracy',  # For classification
    path='autogluon_models',  # Where to save models
    verbosity=2  # Show progress
).fit(
    train_data=train_df,
    time_limit=3600,  # 1 hour
    presets='experimental_quality',  # Try everything!
    # No other constraints - full auto mode
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.1f} seconds")

# Show leaderboard
print("\n" + "="*60)
print("MODEL LEADERBOARD:")
print("="*60)
leaderboard = predictor.leaderboard(train_df, silent=False)
print(leaderboard)

# Get feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE:")
print("="*60)
importance = predictor.feature_importance(train_df)
print(importance)

# Make predictions
print("\n" + "="*60)
print("MAKING PREDICTIONS...")
print("="*60)

# Remove id column for prediction if it exists
test_features = test_df.drop(columns=['id'] if 'id' in test_df.columns else [])
predictions = predictor.predict(test_features)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': predictions
})

# Save submission
filename = f'submission_AUTOGLUON_best_quality_{int(training_time)}s.csv'
submission.to_csv(filename, index=False)
print(f"\nSaved: {filename}")

# Show model details
print("\n" + "="*60)
print("BEST MODEL DETAILS:")
print("="*60)
best_model = predictor.get_model_best()
print(f"Best model: {best_model}")

# Fit summary
print("\n" + "="*60)
print("FIT SUMMARY:")
print("="*60)
fit_summary = predictor.fit_summary()
print(fit_summary)

print("\n" + "="*60)
print("DONE!")
print("="*60)
print(f"AutoGluon tried {len(predictor.get_model_names())} different models")
print(f"Best model: {predictor.get_model_best()}")
print(f"Time spent: {training_time:.1f} seconds")
print(f"\nSubmission saved as: {filename}")
print("\nThis should be close to or exactly 0.975708 if that's the best possible score!")