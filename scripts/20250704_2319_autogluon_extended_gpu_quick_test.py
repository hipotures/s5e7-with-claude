#!/usr/bin/env python3
"""
Quick test to verify AutoGluon setup before 2-hour training
"""

import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import time

# Load data
print("Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Prepare features
train_df['label'] = (train_df['Personality'] == 'Extrovert').astype(int)
train_data = train_df.drop(['Personality'], axis=1)
test_data = test_df.copy()

print("\n=== DATA VERIFICATION ===")
print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")
print(f"Train columns: {list(train_data.columns)}")
print(f"Test columns: {list(test_data.columns)}")
print(f"Label distribution: {train_data['label'].value_counts().to_dict()}")

# Quick 30-second test
print("\n=== QUICK 30-SECOND TEST ===")
predictor = TabularPredictor(
    label='label',
    problem_type='binary',
    eval_metric='accuracy',
    path='AutoGluon_quick_test',
    verbosity=2
)

start = time.time()
predictor.fit(
    train_data=train_data.head(1000),  # Only 1000 samples
    time_limit=30,  # Only 30 seconds
    presets='medium_quality',
    excluded_model_types=['NN_TORCH', 'CAT', 'XGB'],  # Only fast models
)

print(f"\nTraining time: {time.time() - start:.1f}s")

# Test prediction on a few samples
test_sample = test_data.head(5)
predictions = predictor.predict(test_sample)
probs = predictor.predict_proba(test_sample)

print("\n=== TEST PREDICTIONS ===")
print("Test IDs:", test_sample['id'].values)
print("Predictions:", predictions.values)
print("Probabilities:")
print(probs)

# Cleanup
import shutil
shutil.rmtree('AutoGluon_quick_test', ignore_errors=True)

print("\n✓ Quick test completed successfully!")
print("Ready for full 2-hour training.")