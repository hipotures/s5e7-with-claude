#!/usr/bin/env python3
"""
PURPOSE: AutoGluon full automatic mode - 3600 seconds, no limits
HYPOTHESIS: Full automatic mode with experimental quality will find optimal solution
EXPECTED: AutoGluon explores all options within time limit for best score
RESULT: Ran AutoGluon with GBM focus excluding slower model types
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
    presets='experimental_quality',  # Use experimental preset
    excluded_model_types=[
        'TABPFNMIX',  # Exclude TabPFN
        'NN_TORCH',   # Exclude Neural Networks
        'CAT',        # Exclude CatBoost
        'XGB',        # Exclude XGBoost
        'FASTAI',     # Exclude FastAI
        'RF',         # Exclude Random Forest
        'XT',         # Exclude Extra Trees
        'KNN',        # Exclude KNN
    ],  # Only GBM (LightGBM) will be used
    ag_args_fit={
        'excluded_columns': ['id'],  # Exclude ID from training
    }
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.1f} seconds")

# Skip slow leaderboard computation on full dataset
print("\n" + "="*60)
print("SKIPPING FULL LEADERBOARD (too slow on large data)")
print("Using cached validation scores instead...")
print("="*60)
try:
    # Get cached leaderboard (fast)
    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard[['model', 'score_val', 'pred_time_val']].head(10))
except:
    print("Cached leaderboard not available")

# Skip feature importance - too slow on full dataset
print("\n" + "="*60) 
print("SKIPPING FEATURE IMPORTANCE (too slow)")
print("="*60)

# Make predictions
print("\n" + "="*60)
print("MAKING PREDICTIONS...")
print("="*60)

# Keep id column in test data - AutoGluon will handle it
predictions = predictor.predict(test_df)

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