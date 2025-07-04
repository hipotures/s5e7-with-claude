#!/usr/bin/env python3
"""
PURPOSE: Implement final ensemble combining ALL insights to achieve 0.976518+
HYPOTHESIS: Combining all discoveries with precise rules will break the barrier
EXPECTED: Achieve validation score > 0.975708 and potentially reach 0.976518
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

# Advanced feature engineering incorporating all insights
def create_advanced_features(df, is_train=True):
    if is_train:
        X = df.drop(['id', 'Personality'], axis=1).copy()
        y = (df['Personality'] == 'Extrovert').astype(int)
    else:
        X = df.drop(['id'], axis=1).copy()
        y = None
    
    # Convert categorical with special handling for missing
    X['Stage_fear'] = X['Stage_fear'].map({'Yes': 1, 'No': 0})
    X['Drained_after_socializing'] = X['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    
    # Special indicators for missing values (important discovery)
    X['Stage_fear_missing'] = X['Stage_fear'].isna().astype(int)
    X['Drained_missing'] = X['Drained_after_socializing'].isna().astype(int)
    
    # Fill missing with discovered optimal values
    X['Stage_fear'] = X['Stage_fear'].fillna(0.5)
    X['Drained_after_socializing'] = X['Drained_after_socializing'].fillna(0.5)
    
    # Critical boundary indicators from analysis
    X['friends_boundary_4_5'] = X['Friends_circle_size'].isin([4, 5]).astype(int)
    X['friends_high'] = (X['Friends_circle_size'] >= 6).astype(int)
    X['friends_very_low'] = (X['Friends_circle_size'] <= 3).astype(int)
    
    # Time alone critical thresholds
    X['alone_very_low'] = (X['Time_spent_Alone'] <= 3).astype(int)
    X['alone_high'] = (X['Time_spent_Alone'] >= 7).astype(int)
    
    # Social event special values
    X['social_10'] = (X['Social_event_attendance'] == 10).astype(int)
    X['social_high'] = (X['Social_event_attendance'] >= 7).astype(int)
    X['social_moderate'] = X['Social_event_attendance'].between(4, 7).astype(int)
    
    # Exact marker detection
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    
    X['exact_markers'] = 0
    for col, val in markers.items():
        X[f'marker_{col}'] = (np.abs(X[col] - val) < 1e-6).astype(int)
        X['exact_markers'] += X[f'marker_{col}']
    
    # Interaction features based on discoveries
    X['social_intensity'] = X['Social_event_attendance'] * (1 - X['Drained_after_socializing'])
    X['true_introvert'] = X['Time_spent_Alone'] * X['Drained_after_socializing'] * X['Stage_fear']
    X['social_ratio'] = (X['Social_event_attendance'] + 1) / (X['Time_spent_Alone'] + 1)
    
    # Ambivert scores
    X['ambivert_behavioral'] = (
        X['alone_very_low'] * 0.3 +
        X['social_high'] * 0.3 +
        X['friends_high'] * 0.2 +
        (1 - X['Drained_after_socializing']) * 0.2
    )
    
    # Pattern indicators
    X['extreme_extrovert'] = (
        (X['Time_spent_Alone'] == 0) & 
        (X['Drained_after_socializing'] == 0) &
        (X['Stage_fear'] == 0)
    ).astype(int)
    
    X['extreme_introvert'] = (
        (X['Time_spent_Alone'] >= 8) & 
        (X['Drained_after_socializing'] == 1) &
        (X['Stage_fear'] == 1)
    ).astype(int)
    
    # Missing pattern indicator
    X['has_missing'] = (X['Stage_fear_missing'] | X['Drained_missing']).astype(int)
    
    return X, y

# Prepare data
print("Creating advanced features...")
X_train, y_train = create_advanced_features(train_df, is_train=True)
X_test, _ = create_advanced_features(test_df, is_train=False)

print(f"Feature count: {X_train.shape[1]}")

# Advanced prediction strategy
def apply_breakthrough_strategy(probs, features_df):
    """Apply all discovered rules and insights"""
    predictions = np.zeros(len(probs))
    
    for i in range(len(probs)):
        row = features_df.iloc[i]
        prob = probs[i]
        
        # Rule 1: Exact markers -> use special threshold
        if row['exact_markers'] >= 2:
            threshold = 0.42
        
        # Rule 2: Social_event_attendance = 10
        elif row['social_10'] == 1:
            threshold = 0.40
        
        # Rule 3: Friends circle boundary (4-5)
        elif row['friends_boundary_4_5'] == 1:
            threshold = 0.50  # Most uncertain range
        
        # Rule 4: Missing values
        elif row['has_missing'] == 1:
            if row['Drained_missing'] == 1:
                threshold = 0.60  # 40.2% extrovert when missing
            else:
                threshold = 0.40  # 60.7% extrovert when missing
        
        # Rule 5: Extreme patterns
        elif row['extreme_extrovert'] == 1:
            predictions[i] = 1  # Force extrovert
            continue
        elif row['extreme_introvert'] == 1:
            predictions[i] = 0  # Force introvert
            continue
        
        # Rule 6: High ambivert score
        elif row['ambivert_behavioral'] >= 0.7:
            # Apply 97.9% rule for high confidence ambiverts
            if 0.45 <= prob <= 0.55:
                predictions[i] = 1  # Force extrovert
                continue
            else:
                threshold = 0.42
        
        # Rule 7: Probability boundaries
        elif 0.96 <= prob <= 0.98:
            predictions[i] = 1  # Force high boundary to extrovert
            continue
        
        # Default: Use discovered optimal threshold
        else:
            threshold = 0.35
        
        predictions[i] = int(prob >= threshold)
    
    return predictions.astype(int)

# Cross-validation
print("\n=== CROSS-VALIDATION ===")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"\nFold {fold + 1}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train multiple models
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    
    # Train models
    xgb_model.fit(X_tr, y_tr)
    lgb_model.fit(X_tr, y_tr)
    
    # Get probabilities
    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
    ensemble_probs = (xgb_probs + lgb_probs) / 2
    
    # Apply breakthrough strategy
    predictions = apply_breakthrough_strategy(ensemble_probs, X_val)
    
    # Calculate accuracy
    accuracy = (predictions == y_val).mean()
    cv_scores.append(accuracy)
    print(f"Fold accuracy: {accuracy:.6f}")

print(f"\n=== CV RESULTS ===")
print(f"Mean: {np.mean(cv_scores):.6f}")
print(f"Std: {np.std(cv_scores):.6f}")

# Train final models on full data
print("\n=== TRAINING FINAL ENSEMBLE ===")

models = {
    'xgb': xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ),
    'lgb': lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    ),
    'cat': CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.1,
        random_seed=42, verbose=False
    )
}

# Train all models
test_probs = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    test_probs[name] = model.predict_proba(X_test)[:, 1]

# Ensemble predictions
ensemble_test_probs = np.mean([test_probs[name] for name in models.keys()], axis=0)

# Apply breakthrough strategy
print("\n=== GENERATING FINAL PREDICTIONS ===")
final_predictions = apply_breakthrough_strategy(ensemble_test_probs, X_test)

# Analyze predictions
print(f"\nPrediction distribution:")
print(f"Introverts: {(final_predictions == 0).sum()}")
print(f"Extroverts: {(final_predictions == 1).sum()}")

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in final_predictions]
})

# Save submission
import os
from datetime import datetime
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)

submission_path = f'subm/DATE_{date_str}/20250704_2257_breakthrough_ensemble_final.csv'
submission.to_csv(submission_path, index=False)

print(f"\n=== FINAL SUMMARY ===")
print(f"CV Score: {np.mean(cv_scores):.6f}")
print(f"Features used: {X_train.shape[1]}")
print(f"Models in ensemble: {len(models)}")
print(f"Submission saved to: {submission_path}")
print("\nThis submission incorporates:")
print("- Exact marker detection")
print("- Missing value handling")
print("- Friends circle boundaries")
print("- Probability boundary forcing")
print("- Dynamic thresholds")
print("- 97.9% ambivert rule")

# RESULT: [To be filled after execution]