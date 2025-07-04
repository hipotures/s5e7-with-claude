#!/usr/bin/env python3
"""Pseudo-labeling strategy for ambiverts using iterative refinement.

PURPOSE: Use pseudo-labeling and ensemble methods to better classify ambiverts by
         leveraging high-confidence predictions to augment training data and using
         adaptive thresholds based on prediction confidence.

HYPOTHESIS: Ambiverts can be better classified using: (1) ensemble of models with
            different class weights, (2) calibrated probabilities for better uncertainty
            estimates, (3) pseudo-labeling with high-confidence test samples, and
            (4) adaptive thresholds based on confidence levels.

EXPECTED: Identify high, medium, and low confidence predictions. Use pseudo-labeling
          to add high-confidence test samples to training. Apply different thresholds
          (0.5, 0.49, 0.48, 0.47) based on confidence levels and ambivert similarity.

RESULT: Created ensemble of 5 models with different class weights. Implemented
        calibrated probabilities using isotonic regression. Successfully applied
        pseudo-labeling for high-confidence predictions (>0.9 or <0.1). Used
        adaptive thresholds: 0.5 for high confidence, 0.49 for medium, 0.48 for
        low, and 0.47 for ambivert-like behavioral patterns.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

print("PSEUDO-LABELING STRATEGY FOR AMBIVERTS")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Load potential ambiverts from previous analysis
ambiverts_df = pd.read_csv("potential_ambiverts.csv")
ambivert_ids = set(ambiverts_df['id'].values)

print(f"\nLoaded {len(ambivert_ids)} potential ambiverts from previous analysis")

# Standard preprocessing
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency']

numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

# Fill missing
for col in numerical_cols:
    mean_val = train_df[col].mean()
    train_df[col] = train_df[col].fillna(mean_val)
    test_df[col] = test_df[col].fillna(mean_val)

for col in categorical_cols:
    train_df[col] = train_df[col].fillna('Missing')
    test_df[col] = test_df[col].fillna('Missing')

# Encode
mapping_yes_no = {'Yes': 1, 'No': 0}
for col in categorical_cols:
    train_df[col] = train_df[col].map(mapping_yes_no)
    test_df[col] = test_df[col].map(mapping_yes_no)

mapping_personality = {'Extrovert': 1, 'Introvert': 0}
train_df['Personality'] = train_df['Personality'].map(mapping_personality)

# Mark potential ambiverts in training data
train_df['is_ambivert'] = train_df['id'].isin(ambivert_ids).astype(int)
print(f"Marked {train_df['is_ambivert'].sum()} ambiverts in training data")

# Create feature engineering for ambiverts
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# 1. Behavioral consistency score
train_df['behavior_consistency'] = (
    (train_df['Drained_after_socializing'] == 1).astype(int) +
    (train_df['Stage_fear'] == 1).astype(int) +
    (train_df['Time_spent_Alone'] > 5).astype(int) +
    (train_df['Social_event_attendance'] < 5).astype(int) +
    (train_df['Friends_circle_size'] < 8).astype(int)
) / 5

test_df['behavior_consistency'] = (
    (test_df['Drained_after_socializing'] == 1).astype(int) +
    (test_df['Stage_fear'] == 1).astype(int) +
    (test_df['Time_spent_Alone'] > 5).astype(int) +
    (test_df['Social_event_attendance'] < 5).astype(int) +
    (test_df['Friends_circle_size'] < 8).astype(int)
) / 5

# 2. Social balance score
train_df['social_balance'] = 1 - abs(train_df['Social_event_attendance'] - 5) / 5
test_df['social_balance'] = 1 - abs(test_df['Social_event_attendance'] - 5) / 5

# 3. Activity variance
activity_features = ['Social_event_attendance', 'Going_outside', 'Post_frequency']
train_df['activity_variance'] = train_df[activity_features].var(axis=1)
test_df['activity_variance'] = test_df[activity_features].var(axis=1)

extended_features = features + ['behavior_consistency', 'social_balance', 'activity_variance']

X = train_df[extended_features]
y = train_df['Personality']
X_test = test_df[extended_features]

# Strategy 1: Train multiple models with different class weights
print("\nTraining ensemble with different class weights...")

models = []
weights = [
    {0: 1.0, 1: 1.0},    # Balanced
    {0: 1.2, 1: 0.8},    # Favor introverts
    {0: 0.8, 1: 1.2},    # Favor extroverts
    {0: 1.1, 1: 0.9},    # Slight favor introverts
    {0: 0.9, 1: 1.1},    # Slight favor extroverts
]

for i, weight in enumerate(weights):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        random_state=42 + i,
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=weight[1]/weight[0],
        use_label_encoder=False,
        verbosity=0
    )
    
    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated.fit(X, y)
    models.append(calibrated)
    print(f"  Model {i+1} trained with weights {weight}")

# Get ensemble predictions
print("\nGetting ensemble predictions...")
probas = []
for model in models:
    proba = model.predict_proba(X_test)[:, 1]
    probas.append(proba)

probas = np.array(probas)
mean_proba = probas.mean(axis=0)
std_proba = probas.std(axis=0)

# Identify high-uncertainty predictions (potential ambiverts)
uncertainty = np.minimum(mean_proba, 1 - mean_proba)
high_uncertainty = uncertainty < 0.1  # Very confident predictions
medium_uncertainty = (uncertainty >= 0.1) & (uncertainty < 0.2)
low_uncertainty = uncertainty >= 0.2  # Least confident

print(f"\nPrediction confidence distribution:")
print(f"  High confidence: {high_uncertainty.sum()} ({high_uncertainty.sum()/len(mean_proba)*100:.1f}%)")
print(f"  Medium confidence: {medium_uncertainty.sum()} ({medium_uncertainty.sum()/len(mean_proba)*100:.1f}%)")
print(f"  Low confidence: {low_uncertainty.sum()} ({low_uncertainty.sum()/len(mean_proba)*100:.1f}%)")

# Strategy 2: Pseudo-labeling with confidence threshold
print("\n" + "="*60)
print("PSEUDO-LABELING ITERATION")
print("="*60)

# Use high-confidence predictions to augment training data
high_conf_mask = (mean_proba > 0.9) | (mean_proba < 0.1)
pseudo_labels = (mean_proba > 0.5).astype(int)

if high_conf_mask.sum() > 0:
    # Add high-confidence test samples to training
    pseudo_X = X_test[high_conf_mask].copy()
    pseudo_y = pseudo_labels[high_conf_mask]
    
    X_augmented = pd.concat([X, pseudo_X])
    y_augmented = pd.concat([y, pd.Series(pseudo_y, index=pseudo_X.index)])
    
    print(f"Added {len(pseudo_X)} high-confidence test samples to training")
    
    # Retrain with augmented data
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        random_state=42,
        n_estimators=1000,
        learning_rate=0.006358,
        max_depth=8,
        subsample=0.8854,
        colsample_bytree=0.6,
        reg_lambda=0.8295,
        reg_alpha=5.5149,
        gamma=0.0395,
        min_child_weight=2,
        use_label_encoder=False,
        verbosity=0
    )
    
    final_model.fit(X_augmented, y_augmented)
    final_proba = final_model.predict_proba(X_test)[:, 1]
else:
    final_proba = mean_proba

# Strategy 3: Adaptive thresholds based on confidence
print("\nApplying adaptive thresholds...")

predictions = np.zeros(len(final_proba))

# Different thresholds for different confidence levels
predictions[high_uncertainty] = (final_proba[high_uncertainty] > 0.5).astype(int)
predictions[medium_uncertainty] = (final_proba[medium_uncertainty] > 0.49).astype(int)
predictions[low_uncertainty] = (final_proba[low_uncertainty] > 0.48).astype(int)

# For records similar to known ambiverts, use even lower threshold
ambivert_similarity = test_df['behavior_consistency'] < 0.3
predictions[ambivert_similarity] = (final_proba[ambivert_similarity] > 0.47).astype(int)

print(f"Adjusted {ambivert_similarity.sum()} predictions for ambivert-like records")

# Convert to labels
mapping_inverse = {1: 'Extrovert', 0: 'Introvert'}
pred_labels = [mapping_inverse[int(p)] for p in predictions]

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': pred_labels
})

filename = 'submission_PSEUDO_LABELING_ADAPTIVE.csv'
submission.to_csv(filename, index=False)
print(f"\nSaved: {filename}")

# Save detailed analysis
analysis = pd.DataFrame({
    'id': test_df['id'],
    'mean_proba': mean_proba,
    'std_proba': std_proba,
    'uncertainty': uncertainty,
    'final_proba': final_proba,
    'prediction': predictions,
    'behavior_consistency': test_df['behavior_consistency'],
    'social_balance': test_df['social_balance']
})

analysis.to_csv('pseudo_labeling_analysis.csv', index=False)
print("Saved detailed analysis to pseudo_labeling_analysis.csv")

print("\n" + "="*60)
print("STRATEGY SUMMARY:")
print("="*60)
print("1. Ensemble of 5 models with different class weights")
print("2. Calibrated probabilities for better uncertainty estimates")
print("3. Pseudo-labeling with high-confidence predictions")
print("4. Adaptive thresholds based on prediction confidence")
print("5. Special handling for ambivert-like behavioral patterns")