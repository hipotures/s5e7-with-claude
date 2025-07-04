#!/usr/bin/env python3
"""Optimize ambiguous detection thresholds and parameters using Optuna"""

# PURPOSE: Use Optuna to find optimal thresholds for ambiguous case detection and handling
# HYPOTHESIS: Better threshold optimization for ambiguous cases will break the 97.57% barrier
# EXPECTED: Find precise thresholds that correctly classify the 2.43% ambiguous cases
# RESULT: Comprehensive optimization with 100 trials, saving best parameters to JSON

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("OPTIMIZING AMBIGUOUS DETECTION WITH OPTUNA")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Load the most ambiguous 2.43%
ambiguous_df = pd.read_csv("most_ambiguous_2.43pct.csv")
ambiguous_ids = set(ambiguous_df['id'].values)

print(f"\nLoaded {len(ambiguous_ids)} most ambiguous training samples")

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
train_df['known_ambiguous'] = train_df['id'].isin(ambiguous_ids).astype(float)

X_base = train_df[features]
y = train_df['Personality']
X_test_base = test_df[features]

def create_ambiguity_features(df, alone_thresh, social_thresh, friends_thresh, 
                            ambig_score_thresh, proba_low_thresh, proba_high_thresh):
    """Create ambiguity detection features with given thresholds"""
    df_feat = df.copy()
    
    # Pattern detection
    df_feat['ambiguous_pattern'] = (
        (df['Time_spent_Alone'] < alone_thresh) & 
        (df['Social_event_attendance'] < social_thresh) &
        (df['Friends_circle_size'] < friends_thresh)
    ).astype(float)
    
    # Distance from ambiguous centroid
    ambig_centroid = {
        'Time_spent_Alone': 1.85,
        'Social_event_attendance': 3.75,
        'Friends_circle_size': 6.18
    }
    
    df_feat['dist_to_ambiguous'] = np.sqrt(
        (df['Time_spent_Alone'] - ambig_centroid['Time_spent_Alone'])**2 +
        (df['Social_event_attendance'] - ambig_centroid['Social_event_attendance'])**2 +
        (df['Friends_circle_size'] - ambig_centroid['Friends_circle_size'])**2
    )
    
    df_feat['ambiguity_score'] = 1 / (1 + df_feat['dist_to_ambiguous'])
    
    # Range indicators
    df_feat['in_ambig_alone_range'] = df['Time_spent_Alone'].between(0, alone_thresh)
    df_feat['in_ambig_social_range'] = df['Social_event_attendance'].between(2, social_thresh)
    df_feat['in_ambig_friends_range'] = df['Friends_circle_size'].between(4, friends_thresh)
    
    df_feat['is_ambiguous'] = (
        df_feat['in_ambig_alone_range'] & 
        df_feat['in_ambig_social_range'] & 
        df_feat['in_ambig_friends_range']
    ).astype(float)
    
    # High ambiguity flag based on score
    df_feat['high_ambiguity'] = (df_feat['ambiguity_score'] > ambig_score_thresh).astype(float)
    
    return df_feat, df_feat['ambiguity_score'] > ambig_score_thresh

def objective(trial):
    """Optuna objective function"""
    
    # Hyperparameters to optimize
    alone_thresh = trial.suggest_float('alone_thresh', 2.0, 3.5)
    social_thresh = trial.suggest_float('social_thresh', 3.5, 5.5)
    friends_thresh = trial.suggest_int('friends_thresh', 6, 9)
    ambig_score_thresh = trial.suggest_float('ambig_score_thresh', 0.2, 0.4)
    
    # Probability thresholds for adjustment
    proba_low_thresh = trial.suggest_float('proba_low_thresh', 0.1, 0.25)
    proba_high_thresh = trial.suggest_float('proba_high_thresh', 0.45, 0.55)
    
    # Sample weight for ambiguous cases
    ambig_weight = trial.suggest_float('ambig_weight', 5, 20)
    
    # XGBoost parameters
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.01),
        'max_depth': trial.suggest_int('max_depth', 5, 8),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 3.0, 7.0),
        'gamma': trial.suggest_float('gamma', 0.05, 0.2),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 7)
    }
    
    # Create features with current thresholds
    X_train_feat, train_high_ambig = create_ambiguity_features(
        X_base, alone_thresh, social_thresh, friends_thresh, 
        ambig_score_thresh, proba_low_thresh, proba_high_thresh
    )
    
    # Combine with original features
    feature_cols = features + ['ambiguous_pattern', 'dist_to_ambiguous', 'ambiguity_score',
                              'in_ambig_alone_range', 'in_ambig_social_range', 
                              'in_ambig_friends_range', 'is_ambiguous', 'high_ambiguity']
    
    X = X_train_feat[feature_cols]
    
    # Sample weights
    sample_weights = np.ones(len(train_df))
    sample_weights[train_df['known_ambiguous'] == 1] = ambig_weight
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        weights_train = sample_weights[train_idx]
        
        # Train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
            **xgb_params
        )
        
        model.fit(X_train, y_train, sample_weight=weights_train)
        
        # Predictions
        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba > proba_high_thresh).astype(int)
        
        # Apply ambiguous rule
        val_high_ambig = train_high_ambig.iloc[val_idx]
        pred[val_high_ambig] = 1  # Force to Extrovert
        
        # But if probability is very low, keep as Introvert
        very_low_prob = (proba < proba_low_thresh) & val_high_ambig
        pred[very_low_prob] = 0
        
        accuracy = (pred == y_val.values).mean()
        cv_scores.append(accuracy)
    
    mean_score = np.mean(cv_scores)
    
    # Bonus for detecting close to 2.43% as ambiguous
    test_feat, test_high_ambig = create_ambiguity_features(
        X_test_base, alone_thresh, social_thresh, friends_thresh, 
        ambig_score_thresh, proba_low_thresh, proba_high_thresh
    )
    
    test_ambig_pct = test_high_ambig.sum() / len(test_df) * 100
    pct_penalty = abs(test_ambig_pct - 2.43) * 0.0001  # Small penalty for deviation
    
    return mean_score - pct_penalty

# Optimize
print("\nStarting Optuna optimization...")
print("This will take several minutes...\n")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)
print(f"Best CV score: {study.best_value:.6f}")
print("\nBest parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best parameters
print("\n" + "="*60)
print("TRAINING FINAL MODEL")
print("="*60)

best_params = study.best_params

# Create features with best thresholds
X_train_final, train_high_ambig = create_ambiguity_features(
    X_base, 
    best_params['alone_thresh'], 
    best_params['social_thresh'], 
    best_params['friends_thresh'],
    best_params['ambig_score_thresh'],
    best_params['proba_low_thresh'],
    best_params['proba_high_thresh']
)

X_test_final, test_high_ambig = create_ambiguity_features(
    X_test_base,
    best_params['alone_thresh'], 
    best_params['social_thresh'], 
    best_params['friends_thresh'],
    best_params['ambig_score_thresh'],
    best_params['proba_low_thresh'],
    best_params['proba_high_thresh']
)

# Feature columns
feature_cols = features + ['ambiguous_pattern', 'dist_to_ambiguous', 'ambiguity_score',
                          'in_ambig_alone_range', 'in_ambig_social_range', 
                          'in_ambig_friends_range', 'is_ambiguous', 'high_ambiguity']

X_final = X_train_final[feature_cols]
X_test_pred = X_test_final[feature_cols]

# Sample weights
sample_weights = np.ones(len(train_df))
sample_weights[train_df['known_ambiguous'] == 1] = best_params['ambig_weight']

# Extract XGBoost params
xgb_params = {k: v for k, v in best_params.items() 
              if k in ['n_estimators', 'learning_rate', 'max_depth', 'subsample',
                       'colsample_bytree', 'reg_lambda', 'reg_alpha', 'gamma', 'min_child_weight']}

# Train final model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    random_state=42,
    use_label_encoder=False,
    verbosity=0,
    **xgb_params
)

print("Training on full dataset...")
model.fit(X_final, y, sample_weight=sample_weights)

# Make predictions
proba_test = model.predict_proba(X_test_pred)[:, 1]
predictions = (proba_test > best_params['proba_high_thresh']).astype(int)

# Apply ambiguous rule
print(f"\nApplying ambiguous rule to {test_high_ambig.sum()} test samples ({test_high_ambig.sum()/len(test_df)*100:.2f}%)")
predictions[test_high_ambig] = 1  # Force to Extrovert

# Fine-tune based on probability
very_intro = (proba_test < best_params['proba_low_thresh']) & test_high_ambig
predictions[very_intro] = 0
print(f"Adjusted {very_intro.sum()} back to Introvert due to very low probability")

# Save submission
mapping_inverse = {1: 'Extrovert', 0: 'Introvert'}
pred_labels = [mapping_inverse[p] for p in predictions]

submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': pred_labels
})

filename = f'029_optuna_optimized_{study.best_value:.6f}.csv'
submission.to_csv(filename, index=False)
print(f"\nSaved: {filename}")

# Save optimization results
results_dict = {
    'best_score': study.best_value,
    'best_params': study.best_params,
    'test_ambiguous_count': int(test_high_ambig.sum()),
    'test_ambiguous_pct': float(test_high_ambig.sum() / len(test_df) * 100)
}

import json
with open('optuna_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\nOptimization results saved to optuna_results.json")
print("\n" + "="*60)
print("DONE!")