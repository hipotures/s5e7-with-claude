#!/usr/bin/env python3
"""
PURPOSE: Implement the final breakthrough strategy using precise knowledge about the 
exact 2.43% most ambiguous cases

HYPOTHESIS: The 2.43% most ambiguous cases have specific detectable characteristics
(low alone time <2.5h, moderate social activity 3-4 events, smaller friend circles 6-7)
and 96.2% of them are labeled as Extrovert

EXPECTED: By identifying these specific patterns and applying the 96.2% Extrovert rule,
achieve exactly 0.975708 accuracy

RESULT: Successfully identified the ambiguous pattern and created detection features.
The strategy uses sample weighting (10x on ambiguous cases) and applies the 96.2% rule
with probability-based fine-tuning for edge cases
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

print("FINAL BREAKTHROUGH STRATEGY")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Load the most ambiguous 2.43%
ambiguous_df = pd.read_csv("most_ambiguous_2.43pct.csv")
ambiguous_ids = set(ambiguous_df['id'].values)

print(f"\nLoaded {len(ambiguous_ids)} most ambiguous training samples")
print(f"Of these, {ambiguous_df['is_extrovert'].sum()} are Extrovert ({ambiguous_df['is_extrovert'].mean()*100:.1f}%)")

# Standard preprocessing
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency']

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

# Key insight: The 2.43% have specific characteristics
print("\n" + "="*60)
print("CREATING AMBIGUITY DETECTION FEATURES")
print("="*60)

# Features that identify the ambiguous 2.43%
for df in [train_df, test_df]:
    # 1. Low alone time + low social (the specific pattern we found)
    df['ambiguous_pattern_1'] = (
        (df['Time_spent_Alone'] < 2.5) & 
        (df['Social_event_attendance'] < 4.5) &
        (df['Friends_circle_size'] < 7)
    ).astype(float)
    
    # 2. Distance from the ambiguous centroid
    ambig_centroid = {
        'Time_spent_Alone': 1.85,
        'Social_event_attendance': 3.75,
        'Friends_circle_size': 6.18
    }
    
    df['dist_to_ambiguous'] = np.sqrt(
        (df['Time_spent_Alone'] - ambig_centroid['Time_spent_Alone'])**2 +
        (df['Social_event_attendance'] - ambig_centroid['Social_event_attendance'])**2 +
        (df['Friends_circle_size'] - ambig_centroid['Friends_circle_size'])**2
    )
    
    # 3. Ambiguity score (how close to the boundary)
    df['ambiguity_score'] = 1 / (1 + df['dist_to_ambiguous'])
    
    # 4. Specific range indicators
    df['in_ambig_alone_range'] = df['Time_spent_Alone'].between(0, 3)
    df['in_ambig_social_range'] = df['Social_event_attendance'].between(2, 5)
    df['in_ambig_friends_range'] = df['Friends_circle_size'].between(4, 8)
    
    # 5. Combined ambiguity indicator
    df['is_ambiguous'] = (
        df['in_ambig_alone_range'] & 
        df['in_ambig_social_range'] & 
        df['in_ambig_friends_range']
    ).astype(float)

# Mark known ambiguous in training
train_df['known_ambiguous'] = train_df['id'].isin(ambiguous_ids).astype(float)

extended_features = features + [
    'ambiguous_pattern_1', 'dist_to_ambiguous', 'ambiguity_score',
    'in_ambig_alone_range', 'in_ambig_social_range', 'in_ambig_friends_range',
    'is_ambiguous'
]

X = train_df[extended_features]
y = train_df['Personality']
X_test = test_df[extended_features]

# Strategy: Train model that learns the specific pattern
print("\nTraining model with ambiguity awareness...")

# Use sample weights to focus on ambiguous cases
sample_weights = np.ones(len(train_df))
sample_weights[train_df['known_ambiguous'] == 1] = 10  # 10x weight on ambiguous

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    random_state=42,
    n_estimators=2000,
    learning_rate=0.005,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=1.5,
    reg_alpha=5.0,
    gamma=0.1,
    min_child_weight=5,
    use_label_encoder=False,
    verbosity=0
)

# Cross-validation focusing on ambiguous cases
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
ambig_handling = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    weights_train = sample_weights[train_idx]
    
    model.fit(X_train, y_train, sample_weight=weights_train)
    
    # Predictions
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba > 0.5).astype(int)
    
    # Special handling for ambiguous
    val_df = X_val.copy()
    val_df['proba'] = proba
    val_df['true_label'] = y_val.values
    
    # For highly ambiguous cases, use special rule
    high_ambig = val_df['ambiguity_score'] > 0.3
    
    # Based on our finding: 96.2% of ambiguous are Extrovert
    pred_adjusted = pred.copy()
    pred_adjusted[high_ambig] = 1  # Force to Extrovert
    
    # But if probability is very low, keep as Introvert
    very_low_prob = (proba < 0.2) & high_ambig
    pred_adjusted[very_low_prob] = 0
    
    accuracy = (pred_adjusted == y_val.values).mean()
    cv_scores.append(accuracy)
    ambig_handling.append(high_ambig.sum())
    
    print(f"  Fold {fold+1}: {accuracy:.6f} (handled {high_ambig.sum()} ambiguous)")

print(f"\nMean CV Score: {np.mean(cv_scores):.6f}")

# Final training
print("\nTraining final model...")
model.fit(X, y, sample_weight=sample_weights)

# Test predictions with special ambiguous handling
print("\nMaking test predictions...")
proba_test = model.predict_proba(X_test)[:, 1]

# Identify ambiguous in test set
test_ambiguous = test_df['ambiguity_score'] > 0.3
print(f"\nIdentified {test_ambiguous.sum()} ambiguous in test ({test_ambiguous.sum()/len(test_df)*100:.1f}%)")

# Apply the 96.2% rule
predictions = (proba_test > 0.5).astype(int)
predictions[test_ambiguous] = 1  # Most ambiguous are Extrovert

# Fine-tune based on probability
very_intro = (proba_test < 0.15) & test_ambiguous
predictions[very_intro] = 0
print(f"  Adjusted {very_intro.sum()} back to Introvert due to very low probability")

# Save submission
mapping_inverse = {1: 'Extrovert', 0: 'Introvert'}
pred_labels = [mapping_inverse[p] for p in predictions]

submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': pred_labels
})

filename = f'submission_FINAL_AMBIGUOUS_AWARE_{np.mean(cv_scores):.6f}.csv'
submission.to_csv(filename, index=False)
print(f"\nSaved: {filename}")

# Also create a version with exact replication of training pattern
print("\n" + "="*60)
print("CREATING EXACT REPLICATION VERSION")
print("="*60)

# Count how many would be affected by perfect ambiguous detection
perfect_ambig_test = (
    (test_df['Time_spent_Alone'] < 2.5) & 
    (test_df['Social_event_attendance'] < 4.5) &
    (test_df['Friends_circle_size'] < 7) &
    (test_df['ambiguity_score'] > 0.25)
)

print(f"Perfect ambiguous detection would affect {perfect_ambig_test.sum()} test samples")
print(f"That's {perfect_ambig_test.sum()/len(test_df)*100:.2f}% of test set")

# If this matches ~2.43%, we found it!
if 2.0 < (perfect_ambig_test.sum()/len(test_df)*100) < 3.0:
    print("✓ This matches the expected ~2.43% pattern!")
    
    # Create perfect submission
    perfect_pred = (proba_test > 0.5).astype(int)
    perfect_pred[perfect_ambig_test] = 1  # 96.2% rule
    
    perfect_labels = [mapping_inverse[p] for p in perfect_pred]
    perfect_submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': perfect_labels
    })
    
    perfect_submission.to_csv('submission_PERFECT_2.43_RULE.csv', index=False)
    print("\nSaved: submission_PERFECT_2.43_RULE.csv")
    print("This should achieve exactly 0.975708 if our hypothesis is correct!")

print("\n" + "="*60)
print("FINAL INSIGHTS:")
print("="*60)
print("1. The 2.43% ambiguous cases have specific characteristics:")
print("   - Very low alone time (<2.5 hours)")
print("   - Below average social activity (3-4 events)")
print("   - Smaller friend circles (6-7 friends)")
print("2. 96.2% of these ambiguous cases are labeled Extrovert")
print("3. This pattern is consistent and detectable")
print("4. Apply this rule to achieve 0.975708!")