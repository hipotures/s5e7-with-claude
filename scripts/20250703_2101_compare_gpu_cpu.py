#!/usr/bin/env python3
"""
PURPOSE: Compare XGBoost results between GPU and CPU to find differences in predictions
HYPOTHESIS: GPU and CPU implementations might produce slightly different results due to numerical precision
EXPECTED: Identify if GPU/CPU differences could explain the exact 0.975708 score
RESULT: Compared GPU vs CPU predictions to understand potential differences
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print("GPU vs CPU COMPARISON")
print("="*60)

# Prepare data (best configuration)
features = ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone']
train_df[features] = train_df[features].fillna(0)
test_df[features] = test_df[features].fillna(0)

for col in features:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})
        test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})

X_train = train_df[features]
X_test = test_df[features]
le = LabelEncoder()
y_train = le.fit_transform(train_df['Personality'])

print(f"Features: {features}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Test configurations - focusing on simpler models that might achieve exact score
configs = [
    {
        'name': 'Minimal_v1',
        'base_params': {
            'n_estimators': 10,
            'max_depth': 2,
            'learning_rate': 0.3,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    },
    {
        'name': 'Minimal_v2',
        'base_params': {
            'n_estimators': 20,
            'max_depth': 2,
            'learning_rate': 0.3,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    },
    {
        'name': 'Simple_v1',
        'base_params': {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.3,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    },
    {
        'name': 'Simple_v2',
        'base_params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.3,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    },
    {
        'name': 'Standard',
        'base_params': {
            'n_estimators': 300,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    }
]

results = []

for config in configs:
    print(f"\n\nTesting configuration: {config['name']}")
    print("-"*60)
    
    # CPU version
    print("\nCPU Version:")
    cpu_params = config['base_params'].copy()
    cpu_params.update({
        'tree_method': 'hist',
        'device': 'cpu',
        'n_jobs': -1
    })
    
    model_cpu = xgb.XGBClassifier(**cpu_params)
    
    # Time CPU training
    start_time = time.time()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cpu_scores = cross_val_score(model_cpu, X_train, y_train, cv=skf, scoring='accuracy')
    cpu_time = time.time() - start_time
    
    cpu_mean = cpu_scores.mean()
    cpu_std = cpu_scores.std()
    
    print(f"  CV Score: {cpu_mean:.8f} ± {cpu_std:.8f}")
    print(f"  Time: {cpu_time:.2f}s")
    
    # Train on full data for predictions
    model_cpu.fit(X_train, y_train)
    cpu_predictions = model_cpu.predict(X_test)
    cpu_pred_labels = le.inverse_transform(cpu_predictions)
    
    # GPU version
    print("\nGPU Version:")
    gpu_params = config['base_params'].copy()
    gpu_params.update({
        'tree_method': 'hist',
        'device': 'cuda'
    })
    
    model_gpu = xgb.XGBClassifier(**gpu_params)
    
    # Time GPU training
    start_time = time.time()
    gpu_scores = cross_val_score(model_gpu, X_train, y_train, cv=skf, scoring='accuracy')
    gpu_time = time.time() - start_time
    
    gpu_mean = gpu_scores.mean()
    gpu_std = gpu_scores.std()
    
    print(f"  CV Score: {gpu_mean:.8f} ± {gpu_std:.8f}")
    print(f"  Time: {gpu_time:.2f}s")
    
    # Train on full data for predictions
    model_gpu.fit(X_train, y_train)
    gpu_predictions = model_gpu.predict(X_test)
    gpu_pred_labels = le.inverse_transform(gpu_predictions)
    
    # Compare results
    score_diff = cpu_mean - gpu_mean
    print(f"\nDifference (CPU - GPU): {score_diff:+.8f}")
    
    # Check if CPU gives us the target score
    if abs(cpu_mean - 0.975708) < 0.0001:
        print("⭐ CPU GIVES EXACT TARGET SCORE!")
    elif cpu_mean > 0.975:
        print(f"✓ CPU gives high score: {cpu_mean:.6f}")
    
    # Compare predictions
    pred_diff = np.sum(cpu_predictions != gpu_predictions)
    print(f"Prediction differences: {pred_diff} out of {len(cpu_predictions)}")
    
    # Save if CPU score is very good
    if cpu_mean > 0.970:
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': cpu_pred_labels
        })
        filename = f'submission_CPU_{config["name"]}_{cpu_mean:.6f}.csv'
        submission.to_csv(filename, index=False)
        print(f"Saved CPU submission: {filename}")
    
    results.append({
        'config': config['name'],
        'cpu_score': cpu_mean,
        'gpu_score': gpu_mean,
        'difference': score_diff,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'pred_diff': pred_diff
    })

# Summary
print("\n\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Config':<10} {'CPU Score':<12} {'GPU Score':<12} {'Difference':<12} {'Pred Diff':<10}")
print("-"*60)

for r in results:
    print(f"{r['config']:<10} {r['cpu_score']:<12.8f} {r['gpu_score']:<12.8f} "
          f"{r['difference']:+12.8f} {r['pred_diff']:<10}")

print(f"\nTarget score: 0.975708")
print(f"Best CPU score: {max(r['cpu_score'] for r in results):.8f}")
print(f"Best GPU score: {max(r['gpu_score'] for r in results):.8f}")

# Try more parameter combinations that might give 0.975708
print("\n\nTrying extensive parameter combinations on CPU:")
print("-"*60)

# More comprehensive search
for n_est in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
    for max_d in [1, 2, 3]:
        for lr in [0.1, 0.2, 0.3, 0.5]:
            model = xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                learning_rate=lr,
                random_state=42,
                tree_method='hist',
                device='cpu',
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
            mean_score = scores.mean()
            
            if mean_score > 0.975:
                print(f"n_est={n_est:3d}, depth={max_d}, lr={lr:.1f} -> CV: {mean_score:.8f}")
                
                if abs(mean_score - 0.975708) < 0.0001:
                    print("⭐⭐⭐ EXACT MATCH FOUND! ⭐⭐⭐")
                    # Save this configuration
                    model.fit(X_train, y_train)
                    exact_predictions = model.predict(X_test)
                    exact_labels = le.inverse_transform(exact_predictions)
                    
                    submission_exact = pd.DataFrame({
                        'id': test_df['id'],
                        'Personality': exact_labels
                    })
                    submission_exact.to_csv(f'submission_EXACT_MATCH_{mean_score:.8f}.csv', index=False)
                    print(f"Saved exact match submission!")
                    break