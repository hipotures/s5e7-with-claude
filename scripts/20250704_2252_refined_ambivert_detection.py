#!/usr/bin/env python3
"""
PURPOSE: Test refined ambivert detection methods combining all insights
HYPOTHESIS: Better ambivert detection + optimal thresholds = breakthrough to 0.976518
EXPECTED: Achieve validation accuracy approaching or exceeding 0.975708
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

# Prepare features
def prepare_features(df, is_train=True):
    if is_train:
        X = df.drop(['id', 'Personality'], axis=1).copy()
        y = (df['Personality'] == 'Extrovert').astype(int)
    else:
        X = df.drop(['id'], axis=1).copy()
        y = None
    
    # Convert categorical
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in categorical_cols:
        X[col] = X[col].map({'Yes': 1, 'No': 0})
        X[col] = X[col].fillna(0.5)
    
    # Key feature engineering based on discoveries
    X['social_to_alone_ratio'] = (X['Social_event_attendance'] + 1) / (X['Time_spent_Alone'] + 1)
    X['social_intensity'] = X['Social_event_attendance'] * (1 - X['Drained_after_socializing'])
    X['true_introvert_score'] = X['Time_spent_Alone'] * X['Drained_after_socializing']
    
    # Ambivert indicators from our analysis
    X['low_alone_time'] = (X['Time_spent_Alone'] <= 3).astype(int)
    X['high_social'] = (X['Social_event_attendance'] >= 4).astype(int)
    X['moderate_friends'] = ((X['Friends_circle_size'] >= 6) & (X['Friends_circle_size'] <= 12)).astype(int)
    
    # Special markers
    X['has_marker_social_10'] = (X['Social_event_attendance'] == 10.0).astype(int)
    X['has_exact_markers'] = 0
    
    # Check for exact marker values
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    
    for col, val in markers.items():
        X['has_exact_markers'] += (np.abs(X[col] - val) < 1e-6).astype(int)
    
    return X, y

# Prepare data
X_train, y_train = prepare_features(train_df, is_train=True)
X_test, _ = prepare_features(test_df, is_train=False)

print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Advanced ambivert detection
def detect_ambiverts_advanced(X, probs=None):
    """Multi-method ambivert detection"""
    ambiverts = np.zeros(len(X), dtype=bool)
    confidence = np.zeros(len(X))
    
    # Method 1: Exact markers (highest confidence)
    has_markers = X['has_exact_markers'] >= 2
    ambiverts |= has_markers
    confidence[has_markers] = 0.9
    
    # Method 2: Social_event_attendance = 10
    social_10 = X['has_marker_social_10'] == 1
    ambiverts |= social_10
    confidence[social_10] = 0.8
    
    # Method 3: Behavioral pattern
    behavioral = (
        (X['low_alone_time'] == 1) & 
        (X['high_social'] == 1) & 
        (X['moderate_friends'] == 1)
    )
    ambiverts |= behavioral
    confidence[behavioral] = 0.7
    
    # Method 4: Probability-based (if available)
    if probs is not None:
        prob_ambiguous = (probs >= 0.35) & (probs <= 0.65)
        ambiverts |= prob_ambiguous
        confidence[prob_ambiguous] = np.maximum(confidence[prob_ambiguous], 0.6)
    
    # Method 5: High extrovert features pattern
    extrovert_pattern = (
        (X['Time_spent_Alone'] <= 3) & 
        (X['Social_event_attendance'] >= 7) &
        (X['Friends_circle_size'] >= 10)
    )
    ambiverts |= extrovert_pattern
    confidence[extrovert_pattern] = 0.5
    
    return ambiverts, confidence

# Cross-validation with refined strategy
print("\n=== CROSS-VALIDATION WITH REFINED STRATEGY ===")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
ambivert_stats = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"\nFold {fold + 1}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    
    # Get probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    
    # Detect ambiverts
    ambiverts, confidence = detect_ambiverts_advanced(X_val, val_probs)
    n_ambiverts = ambiverts.sum()
    
    # Apply refined prediction strategy
    predictions = np.zeros(len(val_probs))
    
    # Convert to numpy arrays to avoid pandas indexing issues
    ambiverts_arr = ambiverts.values if hasattr(ambiverts, 'values') else ambiverts
    confidence_arr = confidence.values if hasattr(confidence, 'values') else confidence
    
    for i in range(len(val_probs)):
        if ambiverts_arr[i]:
            # Dynamic threshold based on confidence
            if confidence_arr[i] >= 0.8:  # High confidence ambivert
                threshold = 0.42
            elif confidence_arr[i] >= 0.6:  # Medium confidence
                threshold = 0.45
            else:  # Low confidence
                threshold = 0.48
            
            # Apply 97.9% rule for very uncertain cases
            if val_probs[i] >= 0.45 and val_probs[i] <= 0.55:
                predictions[i] = 1  # Force to Extrovert
            else:
                predictions[i] = int(val_probs[i] >= threshold)
        else:
            # Non-ambiverts use optimal threshold
            predictions[i] = int(val_probs[i] >= 0.35)
    
    # Calculate accuracy
    accuracy = (predictions == y_val).mean()
    cv_scores.append(accuracy)
    
    # Stats
    ambivert_accuracy = (predictions[ambiverts_arr] == y_val.values[ambiverts_arr]).mean() if n_ambiverts > 0 else 0
    ambivert_stats.append({
        'n_ambiverts': n_ambiverts,
        'pct_ambiverts': n_ambiverts / len(y_val) * 100,
        'ambivert_accuracy': ambivert_accuracy,
        'extrovert_rate': y_val.values[ambiverts_arr].mean() if n_ambiverts > 0 else 0
    })
    
    print(f"Ambiverts detected: {n_ambiverts} ({n_ambiverts/len(y_val)*100:.1f}%)")
    print(f"Fold accuracy: {accuracy:.6f}")

print(f"\n=== CROSS-VALIDATION RESULTS ===")
print(f"Mean CV accuracy: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
print(f"Average ambiverts detected: {np.mean([s['pct_ambiverts'] for s in ambivert_stats]):.1f}%")

# Train final ensemble
print("\n=== TRAINING FINAL ENSEMBLE ===")

# Model 1: XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Model 2: LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

# Model 3: CatBoost
cat_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=False
)

# Train all models
print("Training XGBoost...")
xgb_model.fit(X_train, y_train)

print("Training LightGBM...")
lgb_model.fit(X_train, y_train)

print("Training CatBoost...")
cat_model.fit(X_train, y_train)

# Generate test predictions
print("\n=== GENERATING TEST PREDICTIONS ===")

# Get probabilities from each model
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
cat_probs = cat_model.predict_proba(X_test)[:, 1]

# Ensemble probabilities
ensemble_probs = (xgb_probs + lgb_probs + cat_probs) / 3

# Detect ambiverts in test set
test_ambiverts, test_confidence = detect_ambiverts_advanced(X_test, ensemble_probs)
print(f"Test ambiverts detected: {test_ambiverts.sum()} ({test_ambiverts.sum()/len(X_test)*100:.1f}%)")

# Apply refined strategy to test
test_predictions = np.zeros(len(ensemble_probs))

for i in range(len(ensemble_probs)):
    if test_ambiverts[i]:
        # Dynamic threshold based on confidence
        if test_confidence[i] >= 0.8:
            threshold = 0.42
        elif test_confidence[i] >= 0.6:
            threshold = 0.45
        else:
            threshold = 0.48
        
        # Apply 97.9% rule for very uncertain cases
        if ensemble_probs[i] >= 0.45 and ensemble_probs[i] <= 0.55:
            test_predictions[i] = 1  # Force to Extrovert
        else:
            test_predictions[i] = int(ensemble_probs[i] >= threshold)
    else:
        # Non-ambiverts use optimal threshold
        test_predictions[i] = int(ensemble_probs[i] >= 0.35)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in test_predictions.astype(int)]
})

# Save submission
import os
from datetime import datetime
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)

submission_path = f'subm/DATE_{date_str}/20250704_2252_refined_ambivert_detection.csv'
submission.to_csv(submission_path, index=False)

# Save results
results = {
    'cv_mean': float(np.mean(cv_scores)),
    'cv_std': float(np.std(cv_scores)),
    'avg_ambiverts_pct': float(np.mean([s['pct_ambiverts'] for s in ambivert_stats])),
    'test_ambiverts_pct': float(test_ambiverts.sum()/len(X_test)*100),
    'ensemble_models': ['XGBoost', 'LightGBM', 'CatBoost'],
    'submission_path': submission_path
}

with open('scripts/output/20250704_2252_refined_detection_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== FINAL RESULTS ===")
print(f"CV Score: {np.mean(cv_scores):.6f}")
print(f"Test ambiverts: {test_ambiverts.sum()/len(X_test)*100:.1f}%")
print(f"Submission saved to: {submission_path}")

# RESULT: CV Score 0.9736, detected ~6% ambiverts, submission ready