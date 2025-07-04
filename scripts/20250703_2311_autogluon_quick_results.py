#!/usr/bin/env python3
"""
PURPOSE: Quick way to get AutoGluon results without slow leaderboard
HYPOTHESIS: Skip expensive leaderboard computation to get predictions faster
EXPECTED: Extract model results and predictions without full evaluation
RESULT: Implemented fast methods to get predictions from trained AutoGluon models
"""

from autogluon.tabular import TabularPredictor
import pandas as pd

print("QUICK AUTOGLUON RESULTS")
print("="*60)

# Load predictor
predictor = TabularPredictor.load('autogluon_models')

# Option 1: Get leaderboard WITHOUT recomputing (cached results)
print("\nOption 1: Cached leaderboard (fast)")
print("-"*40)
try:
    # This uses validation scores from training, no recomputation
    leaderboard_cached = predictor.leaderboard(silent=True)
    print(leaderboard_cached[['model', 'score_val', 'pred_time_val', 'fit_time']].head(10))
except:
    print("No cached leaderboard available")

# Option 2: Get only best model info
print("\n\nOption 2: Best model only")
print("-"*40)
best_model = predictor.get_model_best()
print(f"Best model: {best_model}")

# Get info without full leaderboard
model_info = predictor.info()
print(f"Number of models: {model_info['num_models']}")
print(f"Best model score: {model_info.get('best_model_score_val', 'N/A')}")

# Option 3: Skip leaderboard, go straight to predictions
print("\n\nOption 3: Skip to predictions")
print("-"*40)

# Load test data
test_df = pd.read_csv("../../test.csv")
test_features = test_df.drop(columns=['id'])

# Make predictions
print("Making predictions...")
predictions = predictor.predict(test_features)

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': predictions
})

filename = 'submission_AUTOGLUON_quick.csv'
submission.to_csv(filename, index=False)
print(f"Saved: {filename}")

# Option 4: Get feature importance (usually faster than leaderboard)
print("\n\nOption 4: Feature importance (if needed)")
print("-"*40)
try:
    # Use data='test' for faster computation on smaller dataset
    importance = predictor.feature_importance(data='test', num_shuffle_sets=1)
    print(importance.head())
except:
    print("Feature importance computation failed or not available")

print("\n\nDONE! Got results without slow leaderboard computation.")