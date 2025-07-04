#!/usr/bin/env python3
"""Fast optimization of ambiguous detection - 30 trials only"""

# PURPOSE: Quick optimization run with reduced trials for faster experimentation
# HYPOTHESIS: Even with fewer trials, we can find good thresholds for ambiguous detection
# EXPECTED: Achieve similar performance to full optimization but in less time
# RESULT: Simplified optimization with 3-fold CV and 30 trials for rapid iteration

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

print("FAST OPTIMIZATION (30 trials)")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Load the most ambiguous 2.43%
ambiguous_df = pd.read_csv("most_ambiguous_2.43pct.csv")
ambiguous_ids = set(ambiguous_df['id'].values)

# Preprocessing
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

# Base features
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency']

X_base = train_df[features]
y = train_df['Personality']
X_test_base = test_df[features]

def objective(trial):
    """Simplified objective for faster execution"""
    
    # Key parameters only
    alone_thresh = trial.suggest_float('alone_thresh', 2.0, 3.0)
    social_thresh = trial.suggest_float('social_thresh', 4.0, 5.0)
    friends_thresh = trial.suggest_int('friends_thresh', 6, 8)
    ambig_score_thresh = trial.suggest_float('ambig_score_thresh', 0.25, 0.35)
    
    # Create features
    train_features = X_base.copy()
    test_features = X_test_base.copy()
    
    for df in [train_features, test_features]:
        # Pattern detection
        df['ambiguous_pattern'] = (
            (df['Time_spent_Alone'] < alone_thresh) & 
            (df['Social_event_attendance'] < social_thresh) &
            (df['Friends_circle_size'] < friends_thresh)
        ).astype(float)
        
        # Distance from centroid
        df['dist_to_ambiguous'] = np.sqrt(
            (df['Time_spent_Alone'] - 1.85)**2 +
            (df['Social_event_attendance'] - 3.75)**2 +
            (df['Friends_circle_size'] - 6.18)**2
        )
        
        df['ambiguity_score'] = 1 / (1 + df['dist_to_ambiguous'])
    
    # Extended features
    extended_features = features + ['ambiguous_pattern', 'dist_to_ambiguous', 'ambiguity_score']
    
    X = train_features[extended_features]
    X_test = test_features[extended_features]
    
    # Sample weights
    sample_weights = np.ones(len(train_df))
    sample_weights[train_df['id'].isin(ambiguous_ids)] = 10
    
    # Simple model
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.006358,
        max_depth=8,
        subsample=0.8854,
        colsample_bytree=0.6,
        tree_method='gpu_hist',
        random_state=42,
        use_label_encoder=False,
        verbosity=0
    )
    
    # 3-fold CV (faster)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        weights_train = sample_weights[train_idx]
        
        model.fit(X_train, y_train, sample_weight=weights_train)
        
        # Predictions
        proba = model.predict_proba(X_val)[:, 1]
        
        # Apply rule
        val_ambig = (train_features.iloc[val_idx]['ambiguity_score'] > ambig_score_thresh)
        
        pred = (proba > 0.5).astype(int)
        pred[val_ambig] = 1  # Force ambiguous to Extrovert
        
        # Keep very low probability as Introvert
        very_intro = (proba < 0.15) & val_ambig
        pred[very_intro] = 0
        
        accuracy = (pred == y_val.values).mean()
        cv_scores.append(accuracy)
    
    # Check test percentage
    test_ambig_pct = (test_features['ambiguity_score'] > ambig_score_thresh).sum() / len(test_df) * 100
    pct_penalty = abs(test_ambig_pct - 2.43) * 0.0001
    
    return np.mean(cv_scores) - pct_penalty

# Optimize
print("\nStarting optimization (30 trials)...")

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\nBest score: {study.best_value:.6f}")
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model
print("\nTraining final model...")

best_params = study.best_params

# Create final features
for df in [train_df, test_df]:
    df['ambiguous_pattern'] = (
        (df['Time_spent_Alone'] < best_params['alone_thresh']) & 
        (df['Social_event_attendance'] < best_params['social_thresh']) &
        (df['Friends_circle_size'] < best_params['friends_thresh'])
    ).astype(float)
    
    df['dist_to_ambiguous'] = np.sqrt(
        (df['Time_spent_Alone'] - 1.85)**2 +
        (df['Social_event_attendance'] - 3.75)**2 +
        (df['Friends_circle_size'] - 6.18)**2
    )
    
    df['ambiguity_score'] = 1 / (1 + df['dist_to_ambiguous'])

extended_features = features + ['ambiguous_pattern', 'dist_to_ambiguous', 'ambiguity_score']

X_final = train_df[extended_features]
X_test_final = test_df[extended_features]

# Sample weights
sample_weights = np.ones(len(train_df))
sample_weights[train_df['id'].isin(ambiguous_ids)] = 10

# Final model
model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.006358,
    max_depth=8,
    subsample=0.8854,
    colsample_bytree=0.6,
    reg_lambda=0.8295,
    reg_alpha=5.5149,
    gamma=0.0395,
    min_child_weight=2,
    tree_method='gpu_hist',
    random_state=42,
    use_label_encoder=False,
    verbosity=0
)

model.fit(X_final, train_df['Personality'], sample_weight=sample_weights)

# Predictions
proba_test = model.predict_proba(X_test_final)[:, 1]
test_ambiguous = test_df['ambiguity_score'] > best_params['ambig_score_thresh']

predictions = (proba_test > 0.5).astype(int)
predictions[test_ambiguous] = 1  # Force ambiguous to Extrovert

# Fine-tune
very_intro = (proba_test < 0.15) & test_ambiguous
predictions[very_intro] = 0

print(f"\nAdjusted {test_ambiguous.sum()} ambiguous cases ({test_ambiguous.sum()/len(test_df)*100:.2f}%)")
print(f"Reverted {very_intro.sum()} back to Introvert")

# Save
mapping_inverse = {1: 'Extrovert', 0: 'Introvert'}
pred_labels = [mapping_inverse[p] for p in predictions]

submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': pred_labels
})

submission.to_csv(f'029_optuna_fast_{study.best_value:.6f}.csv', index=False)
print(f"\nSaved: 029_optuna_fast_{study.best_value:.6f}.csv")