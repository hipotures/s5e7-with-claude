#!/usr/bin/env python3
"""
PURPOSE: Map personality features to MBTI types and use this knowledge to detect and
handle ambiverts more effectively

HYPOTHESIS: The original data was based on 16 MBTI types reduced to binary E/I. By
reconstructing other MBTI dimensions (T/F, J/P, S/N) from behavioral features, we can
better identify ambiverts and adjust predictions accordingly

EXPECTED: Create MBTI-inspired features that help identify ~2-3% ambiverts and adjust
their classification thresholds to exceed 0.975708 accuracy

RESULT: Successfully engineered MBTI dimension scores (E/I, T/F, J/P, S/N) and behavioral
contradiction features. Identified ambiverts using multiple criteria including MBTI
likelihood scores and adjusted decision thresholds based on E/I tendency
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

print("MBTI-BASED AMBIVERT DETECTION STRATEGY")
print("="*60)

# MBTI knowledge: Some types are more ambivert than others
# Ambivert types: ISFJ, ISFP, INFJ, ESTP, ESFP, ENFJ
# Clear introverts: INTJ, INTP, ISTJ, ISTP
# Clear extroverts: ENTJ, ENTP, ESTJ, ESFJ, ENFP, ESTP

print("\nMBTI Background:")
print("- Original data likely had 16 MBTI types")
print("- Reduced to E/I for competition")
print("- Some types are naturally ambiverted")

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Features that might indicate other MBTI dimensions
print("\nMapping features to MBTI dimensions...")

# Standard preprocessing first
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

for col in numerical_cols:
    mean_val = train_df[col].mean()
    train_df[col] = train_df[col].fillna(mean_val)
    test_df[col] = test_df[col].fillna(mean_val)

for col in categorical_cols:
    train_df[col] = train_df[col].fillna('Missing')
    test_df[col] = test_df[col].fillna('Missing')
    train_df[col] = train_df[col].map({'Yes': 1, 'No': 0, 'Missing': 0.5})
    test_df[col] = test_df[col].map({'Yes': 1, 'No': 0, 'Missing': 0.5})

train_df['Personality'] = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

# Create MBTI-inspired features
print("\nCreating MBTI-inspired features...")

# 1. Introversion/Extraversion score (we already have this)
for df in [train_df, test_df]:
    df['E_I_score'] = (
        df['Time_spent_Alone'] * -1 + 
        df['Social_event_attendance'] + 
        df['Going_outside'] + 
        df['Friends_circle_size'] / 1.5 + 
        df['Post_frequency'] - 
        df['Drained_after_socializing'] * 5
    ) / 30

# 2. Thinking/Feeling indicators (analytical vs emotional)
# Stage fear might indicate Feeling, systematic behavior indicates Thinking
for df in [train_df, test_df]:
    df['T_F_score'] = (
        df['Stage_fear'] * -1 +  # Fear = Feeling
        (df['Post_frequency'] > 7).astype(int) * -1 +  # High posting = Feeling
        (df['Friends_circle_size'] == 10).astype(int) * -1  # Exact 10 = systematic = Thinking
    ) / 3

# 3. Judging/Perceiving (structured vs flexible)
# Regular patterns might indicate Judging
for df in [train_df, test_df]:
    # Check for "round" numbers (structured behavior)
    df['round_numbers'] = (
        (df['Social_event_attendance'] % 5 == 0).astype(int) +
        (df['Going_outside'] % 5 == 0).astype(int) +
        (df['Friends_circle_size'] % 5 == 0).astype(int)
    ) / 3
    
    df['J_P_score'] = df['round_numbers']

# 4. Sensing/Intuition (practical vs abstract)
# This is harder to infer, but extreme values might indicate Intuition
for df in [train_df, test_df]:
    df['S_N_score'] = (
        (df['Time_spent_Alone'] > 8).astype(int) +
        (df['Social_event_attendance'] < 2).astype(int) +
        (df['Post_frequency'] > 8).astype(int)
    ) / 3

# 5. Ambivert likelihood based on MBTI patterns
for df in [train_df, test_df]:
    # Types that are often ambiverts have:
    # - Moderate E/I score
    # - High F score (emotional, people-oriented)
    # - Flexible behavior (low J)
    
    df['ambivert_likelihood'] = (
        (np.abs(df['E_I_score']) < 0.3).astype(int) * 0.4 +  # Moderate E/I
        (df['T_F_score'] < -0.2).astype(int) * 0.3 +  # Feeling preference
        (df['J_P_score'] < 0.3).astype(int) * 0.3  # Flexible behavior
    )

# 6. Behavioral contradiction score
for df in [train_df, test_df]:
    df['contradiction_score'] = (
        # High social activity but gets drained
        ((df['Social_event_attendance'] > 7) & (df['Drained_after_socializing'] == 1)).astype(int) +
        # Low alone time but has stage fear
        ((df['Time_spent_Alone'] < 3) & (df['Stage_fear'] == 1)).astype(int) +
        # Many friends but low social events
        ((df['Friends_circle_size'] > 10) & (df['Social_event_attendance'] < 3)).astype(int)
    ) / 3

# All features
base_features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                 'Post_frequency']

mbti_features = ['E_I_score', 'T_F_score', 'J_P_score', 'S_N_score', 
                 'ambivert_likelihood', 'contradiction_score', 'round_numbers']

all_features = base_features + mbti_features

X = train_df[all_features]
y = train_df['Personality']
X_test = test_df[all_features]

# Train model with special attention to ambiverts
print("\nTraining XGBoost with MBTI features...")

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    random_state=42,
    n_estimators=1500,
    learning_rate=0.005,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    reg_alpha=6.0,
    gamma=0.05,
    min_child_weight=3,
    use_label_encoder=False,
    verbosity=0
)

# Cross-validation with ambivert analysis
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
ambivert_adjustments = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    
    # Identify potential ambiverts
    val_df = X_val.copy()
    val_df['proba'] = proba
    
    # Multiple criteria for ambiverts
    ambivert_mask = (
        (val_df['ambivert_likelihood'] > 0.6) |  # High MBTI ambivert score
        (val_df['contradiction_score'] > 0.5) |  # Contradictory behavior
        ((np.abs(val_df['E_I_score']) < 0.2) & (np.abs(proba - 0.5) < 0.15))  # Moderate + uncertain
    )
    
    # Adjusted predictions
    pred = (proba > 0.5).astype(int)
    pred_adjusted = pred.copy()
    
    # For ambiverts, use probability closer to their E/I tendency
    ambivert_indices = np.where(ambivert_mask)[0]
    for idx in ambivert_indices:
        ei_score = val_df.iloc[idx]['E_I_score']
        if ei_score > 0:  # Leans extrovert
            pred_adjusted[idx] = (proba[idx] > 0.45).astype(int)
        else:  # Leans introvert
            pred_adjusted[idx] = (proba[idx] > 0.55).astype(int)
    
    accuracy = (pred_adjusted == y_val.values).mean()
    cv_scores.append(accuracy)
    ambivert_adjustments.append(len(ambivert_indices))
    
    print(f"  Fold {fold+1}: {accuracy:.6f} (adjusted {len(ambivert_indices)} ambiverts)")

print(f"\nMean CV Score: {np.mean(cv_scores):.6f}")
print(f"Average ambiverts per fold: {np.mean(ambivert_adjustments):.1f}")

# Train on full data
print("\nTraining on full dataset...")
model.fit(X, y)

# Predictions with MBTI-based ambivert detection
proba_test = model.predict_proba(X_test)[:, 1]

# Identify ambiverts in test set
test_ambiverts = (
    (test_df['ambivert_likelihood'] > 0.6) |
    (test_df['contradiction_score'] > 0.5) |
    ((np.abs(test_df['E_I_score']) < 0.2) & (np.abs(proba_test - 0.5) < 0.15))
)

print(f"\nIdentified {test_ambiverts.sum()} potential ambiverts in test ({test_ambiverts.sum()/len(test_df)*100:.1f}%)")

# Make adjusted predictions
predictions = (proba_test > 0.5).astype(int)

# Adjust for ambiverts
ambivert_indices = np.where(test_ambiverts)[0]
for idx in ambivert_indices:
    ei_score = test_df.iloc[idx]['E_I_score']
    if ei_score > 0:
        predictions[idx] = (proba_test[idx] > 0.45).astype(int)
    else:
        predictions[idx] = (proba_test[idx] > 0.55).astype(int)

# Save predictions
mapping_inverse = {1: 'Extrovert', 0: 'Introvert'}
pred_labels = [mapping_inverse[p] for p in predictions]

submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': pred_labels
})

filename = f'submission_MBTI_STRATEGY_{np.mean(cv_scores):.6f}.csv'
submission.to_csv(filename, index=False)
print(f"\nSaved: {filename}")

# Feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE:")
print("="*60)
importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))

print("\n" + "="*60)
print("STRATEGY INSIGHTS:")
print("="*60)
print("1. Created MBTI-inspired features to detect personality nuances")
print("2. Identified ambiverts using behavioral contradictions")
print("3. Adjusted decision thresholds based on E/I tendency")
print("4. This approach acknowledges the 16→2 type reduction")
print("\nSubmit this to see if we can break 0.975708!")