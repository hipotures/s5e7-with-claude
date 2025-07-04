#!/usr/bin/env python3
"""
PURPOSE: AutoGluon with multiple GBM variants
HYPOTHESIS: Different GBM configurations might achieve better performance
EXPECTED: Train multiple GBM variants with different hyperparameters
RESULT: Created ensemble of GBM models with Extra Trees, default, and large variants
"""

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

print("AUTOGLUON - MULTIPLE GBM VARIANTS")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

target = 'Personality'

# Your configuration - 3 GBM variants
print("\nTraining 3 GBM variants:")
print("1. GBM_XT - Extra Trees mode")
print("2. GBM - Default parameters")
print("3. GBM_Large - Large model with more leaves")

predictor = TabularPredictor(
    label=target,
    eval_metric='accuracy',
    path='autogluon_gbm_variants',
    verbosity=2
).fit(
    train_data=train_df,
    time_limit=1800,  # 30 minutes
    
    hyperparameters={
        'GBM': [
            # Variant 1: Extra Trees mode (more randomness)
            {
                'extra_trees': True, 
                'ag_args': {'name_suffix': 'XT'}
            }, 
            
            # Variant 2: Default GBM
            {}, 
            
            # Variant 3: Large model
            {
                'learning_rate': 0.03,
                'num_leaves': 128,
                'feature_fraction': 0.9,
                'min_data_in_leaf': 3,
                'ag_args': {
                    'name_suffix': 'Large',
                    'priority': 0,  # Higher priority in ensemble
                    'hyperparameter_tune_kwargs': None
                }
            }
        ]
    },
    
    # Optional: with CV
    num_bag_folds=5,
    
    # Can use stacking since we have multiple models
    num_stack_levels=1,
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Show leaderboard
leaderboard = predictor.leaderboard(train_df, silent=False)
print(leaderboard)

# Model names will be:
# - LightGBM_XT
# - LightGBM  
# - LightGBM_Large
# Plus their bagged versions if using CV

print("\n" + "="*60)
print("MODEL DETAILS")
print("="*60)

model_names = predictor.get_model_names()
print(f"Total models trained: {len(model_names)}")
print("Model names:")
for name in model_names:
    print(f"  - {name}")

# Make predictions
test_features = test_df.drop(columns=['id'] if 'id' in test_df.columns else [])
predictions = predictor.predict(test_features)

submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': predictions
})

filename = 'submission_AUTOGLUON_GBM_3variants.csv'
submission.to_csv(filename, index=False)
print(f"\nSaved: {filename}")

print("\n" + "="*60)
print("EXPLANATION OF VARIANTS:")
print("="*60)
print("1. GBM_XT (Extra Trees):")
print("   - extra_trees=True makes it similar to Random Forest")
print("   - More randomness, less overfitting")
print("   - Faster training")
print("\n2. GBM (Default):")
print("   - Standard LightGBM with AutoGluon's defaults")
print("   - Balanced performance")
print("\n3. GBM_Large:")
print("   - More leaves (128 vs default 31)")
print("   - Lower learning rate (0.03)")
print("   - Can capture more complex patterns")
print("   - Higher risk of overfitting")

# You can also add more variants
print("\n" + "="*60)
print("OTHER USEFUL VARIANTS YOU COULD ADD:")
print("="*60)

other_variants = """
'GBM': [
    # Small/Fast variant
    {
        'num_leaves': 16,
        'learning_rate': 0.1,
        'num_boost_round': 100,
        'ag_args': {'name_suffix': 'Small'}
    },
    
    # Regularized variant
    {
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'min_data_in_leaf': 20,
        'ag_args': {'name_suffix': 'Regularized'}
    },
    
    # Deep variant
    {
        'num_leaves': 31,
        'max_depth': 12,  # Deeper trees
        'ag_args': {'name_suffix': 'Deep'}
    }
]
"""
print(other_variants)