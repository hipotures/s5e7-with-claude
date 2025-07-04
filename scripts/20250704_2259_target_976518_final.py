#!/usr/bin/env python3
"""
PURPOSE: Final attempt to achieve 0.976518 with simplified approach and precise rules
HYPOTHESIS: The winner used simpler features but more precise decision rules
EXPECTED: Achieve score > 0.975708, targeting 0.976518
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

# Simple but effective feature engineering
def prepare_features_simple(df, is_train=True):
    if is_train:
        X = df.drop(['id', 'Personality'], axis=1).copy()
        y = (df['Personality'] == 'Extrovert').astype(int)
    else:
        X = df.drop(['id'], axis=1).copy()
        y = None
    
    # Convert categorical - key insight: handle missing values specially
    for col in ['Stage_fear', 'Drained_after_socializing']:
        # First capture missing
        X[f'{col}_missing'] = X[col].isna().astype(int)
        # Then convert
        X[col] = X[col].map({'Yes': 1, 'No': 0})
        # Fill with discovered optimal values
        if col == 'Drained_after_socializing':
            X[col] = X[col].fillna(0)  # Missing means likely NOT drained (extrovert)
        else:
            X[col] = X[col].fillna(0)  # Missing means likely NO fear (extrovert)
    
    # Critical features only
    X['is_extrovert_pattern'] = (
        (X['Time_spent_Alone'] <= 3) & 
        (X['Drained_after_socializing'] == 0)
    ).astype(int)
    
    X['is_introvert_pattern'] = (
        (X['Time_spent_Alone'] >= 7) & 
        (X['Drained_after_socializing'] == 1)
    ).astype(int)
    
    # The key insight: Friends circle 4-5 is the uncertain zone
    X['friends_uncertain'] = X['Friends_circle_size'].isin([4, 5]).astype(int)
    
    # Social 10 is special
    X['social_10'] = (X['Social_event_attendance'] == 10).astype(int)
    
    return X, y

# Prepare data
X_train, y_train = prepare_features_simple(train_df, is_train=True)
X_test, _ = prepare_features_simple(test_df, is_train=False)

print(f"Features: {X_train.shape[1]}")

# The breakthrough decision function
def make_predictions_breakthrough(model, X, is_val=False):
    """Apply discovered rules with high precision"""
    probs = model.predict_proba(X)[:, 1]
    predictions = np.zeros(len(probs))
    
    for i in range(len(probs)):
        prob = probs[i]
        row = X.iloc[i]
        
        # Rule 1: Clear extrovert pattern overrides everything
        if row['is_extrovert_pattern'] == 1:
            predictions[i] = 1
            continue
            
        # Rule 2: Clear introvert pattern overrides everything
        if row['is_introvert_pattern'] == 1:
            predictions[i] = 0
            continue
        
        # Rule 3: Missing values indicate extrovert (key insight!)
        if row['Drained_after_socializing_missing'] == 1:
            predictions[i] = 1
            continue
            
        # Rule 4: Friends 4-5 is uncertain - use probability
        if row['friends_uncertain'] == 1:
            # These are true 50/50 cases - use exact probability
            predictions[i] = int(prob >= 0.5)
            continue
            
        # Rule 5: Social 10 is likely extrovert
        if row['social_10'] == 1:
            predictions[i] = int(prob >= 0.4)
            continue
        
        # Rule 6: For high probability boundaries
        if prob >= 0.96 and prob <= 0.98:
            predictions[i] = 1  # Force to extrovert
            continue
            
        # Default: Use optimal threshold from analysis
        predictions[i] = int(prob >= 0.35)
    
    return predictions.astype(int)

# Cross-validation with precise scoring
print("\n=== CROSS-VALIDATION ===")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train simple XGBoost
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,  # Slightly less depth
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        random_state=42
    )
    
    model.fit(X_tr, y_tr)
    
    # Make predictions with breakthrough rules
    predictions = make_predictions_breakthrough(model, X_val, is_val=True)
    
    accuracy = (predictions == y_val).mean()
    cv_scores.append(accuracy)
    print(f"Fold {fold + 1}: {accuracy:.6f}")

print(f"\nMean CV: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")

# Train final model with more estimators
print("\n=== TRAINING FINAL MODEL ===")
final_model = xgb.XGBClassifier(
    n_estimators=1000,  # More trees for final
    max_depth=5,
    learning_rate=0.05,  # Lower learning rate
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    random_state=42
)

print("Training on full dataset...")
final_model.fit(X_train, y_train)

# Analyze feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop features:")
print(feature_importance.head(10))

# Generate test predictions
print("\n=== GENERATING TEST PREDICTIONS ===")
test_predictions = make_predictions_breakthrough(final_model, X_test, is_val=False)

# Additional post-processing based on discoveries
print("\nApplying final adjustments...")

# Find potential ambiverts in test set using multiple models
models = []
for seed in [42, 123, 456]:
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        random_state=seed
    )
    m.fit(X_train, y_train)
    models.append(m)

# Get probability consensus
all_probs = []
for m in models:
    all_probs.append(m.predict_proba(X_test)[:, 1])

prob_std = np.std(all_probs, axis=0)
prob_mean = np.mean(all_probs, axis=0)

# High uncertainty cases (high std between models)
high_uncertainty = prob_std > 0.1
print(f"High uncertainty cases: {high_uncertainty.sum()}")

# For high uncertainty cases with probability near 0.5, apply 97% rule
for i in np.where(high_uncertainty)[0]:
    if 0.45 <= prob_mean[i] <= 0.55:
        test_predictions[i] = 1  # Force to extrovert

# Create submission
print("\n=== CREATING SUBMISSION ===")
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in test_predictions]
})

# Save
import os
from datetime import datetime
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)

submission_path = f'subm/DATE_{date_str}/20250704_2259_target_976518_final.csv'
submission.to_csv(submission_path, index=False)

print(f"\nPrediction distribution:")
print(f"Introverts: {(test_predictions == 0).sum()}")
print(f"Extroverts: {(test_predictions == 1).sum()}")
print(f"Ratio: {(test_predictions == 1).sum() / len(test_predictions):.1%}")

print(f"\nSubmission saved to: {submission_path}")
print("\nKey strategies used:")
print("1. Missing values = Extrovert")
print("2. Friends 4-5 = True uncertainty")
print("3. Clear patterns override probability")
print("4. Multi-model uncertainty detection")
print("5. 97% rule for high uncertainty ambiverts")

# RESULT: [To be filled after execution]