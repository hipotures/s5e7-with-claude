#!/usr/bin/env python3
"""Test what study names will be generated"""

import hashlib

def get_study_name(model_name, dataset_name, cv_folds=5):
    """Generate unique study name for Optuna"""
    base = f"{model_name}_{dataset_name}_{cv_folds}fold"
    return hashlib.md5(base.encode()).hexdigest()[:12]

# Test all combinations
models = ['xgb', 'gbm', 'cat']
datasets = [
    "train_corrected_01.csv",
    "train_corrected_02.csv", 
    "train_corrected_03.csv",
    "train_corrected_04.csv",
    "train_corrected_05.csv",
    "train_corrected_06.csv",
    "train_corrected_07.csv",
    "train_corrected_08.csv",
]

print("Study names that will be generated:")
print("="*60)

for dataset in datasets:
    print(f"\n{dataset}:")
    for model in models:
        study_name = get_study_name(model, dataset)
        print(f"  {model}: {study_name}.db")