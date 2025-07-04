#!/usr/bin/env python3
"""Use ML-based imputation for missing values - might reveal hidden patterns!"""

# PURPOSE: Test if ML-based imputation can reveal the true values that achieve 100% accuracy
# HYPOTHESIS: Missing values might have been deterministically generated and ML can recover them
# EXPECTED: ML imputation will find patterns in missing data that lead to perfect classification
# RESULT: Two imputation methods tested (custom ML and IterativeImputer) with simple rule applied

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import xgboost as xgb

print("PERFECT SCORE HUNT: ML-Based Missing Value Imputation")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Save original personality before dropping
train_personality = train_df['Personality'].copy()

# Separate features
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency']
categorical_features = ['Stage_fear', 'Drained_after_socializing']

print("\nMissing values analysis:")
for col in numerical_features + categorical_features:
    train_missing = train_df[col].isna().sum()
    test_missing = test_df[col].isna().sum()
    if train_missing > 0 or test_missing > 0:
        print(f"{col}: {train_missing} train, {test_missing} test")

# Method 1: Manual feature-by-feature imputation
print("\n" + "="*60)
print("METHOD 1: Custom ML Imputation")
print("="*60)

def ml_impute_column(df_train, df_test, target_col, feature_cols):
    """Impute missing values in target_col using other features."""
    print(f"\nImputing {target_col}...")
    
    # Combine train and test for consistent imputation
    df_combined = pd.concat([df_train[feature_cols + [target_col]], 
                            df_test[feature_cols + [target_col]]], 
                            ignore_index=True)
    
    # Split into rows with and without target
    mask_not_missing = df_combined[target_col].notna()
    
    if mask_not_missing.sum() == len(df_combined):
        print(f"  No missing values in {target_col}")
        return df_train[target_col], df_test[target_col]
    
    X_train_imp = df_combined.loc[mask_not_missing, feature_cols]
    y_train_imp = df_combined.loc[mask_not_missing, target_col]
    X_predict = df_combined.loc[~mask_not_missing, feature_cols]
    
    # Use appropriate model
    if df_combined[target_col].dtype == 'object' or len(df_combined[target_col].unique()) < 10:
        # Categorical
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        # Numerical
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Handle any missing in features first (simple mean/mode)
    for col in feature_cols:
        if X_train_imp[col].isna().any():
            if X_train_imp[col].dtype == 'object':
                fill_val = X_train_imp[col].mode()[0] if len(X_train_imp[col].mode()) > 0 else 'Unknown'
                X_train_imp[col].fillna(fill_val, inplace=True)
                X_predict[col].fillna(fill_val, inplace=True)
            else:
                fill_val = X_train_imp[col].mean()
                X_train_imp[col].fillna(fill_val, inplace=True)
                X_predict[col].fillna(fill_val, inplace=True)
    
    # Train and predict
    model.fit(X_train_imp, y_train_imp)
    predictions = model.predict(X_predict)
    
    # Fill missing values
    df_combined.loc[~mask_not_missing, target_col] = predictions
    
    # Split back
    train_len = len(df_train)
    return df_combined[target_col][:train_len], df_combined[target_col][train_len:]

# First encode categorical features
for df in [train_df, test_df]:
    df['Stage_fear_enc'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
    df['Drained_enc'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Impute each column using others (but NOT personality!)
all_features = numerical_features + ['Stage_fear_enc', 'Drained_enc']

# Create copies for imputation
train_imputed = train_df.copy()
test_imputed = test_df.copy()

# Impute in order of least to most missing
for target in all_features:
    other_features = [f for f in all_features if f != target]
    
    train_imputed[target], test_imputed[target] = ml_impute_column(
        train_imputed, test_imputed, target, other_features
    )

print("\n" + "="*60)
print("METHOD 2: Scikit-learn IterativeImputer")
print("="*60)

# Prepare data for IterativeImputer
X_train = train_df[all_features].copy()
X_test = test_df[all_features].copy()

# Use IterativeImputer with XGBoost
imputer = IterativeImputer(
    estimator=xgb.XGBRegressor(n_estimators=100, random_state=42),
    max_iter=10,
    random_state=42
)

# Fit on train and transform both
X_train_iter = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=all_features,
    index=X_train.index
)

X_test_iter = pd.DataFrame(
    imputer.transform(X_test),
    columns=all_features,
    index=X_test.index
)

# Compare imputation methods
print("\nComparing imputation results on Drained_after_socializing:")
mask = train_df['Drained_after_socializing'].isna()
if mask.sum() > 0:
    print(f"\nOriginal missing: {mask.sum()}")
    print(f"Custom imputed values: {train_imputed.loc[mask, 'Drained_enc'].value_counts()}")
    print(f"Iterative imputed mean: {X_train_iter.loc[mask, 'Drained_enc'].mean():.3f}")

# Use the imputed data for prediction
print("\n" + "="*60)
print("PREDICTIONS WITH ML-IMPUTED DATA")
print("="*60)

# Simple rule with ML-imputed Drained
test_imputed['Personality'] = test_imputed['Drained_enc'].apply(
    lambda x: 'Introvert' if x > 0.5 else 'Extrovert'
)

submission1 = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_imputed['Personality']
})
submission1.to_csv('001_ml_imputed_simple_rule.csv', index=False)
print("Saved: 001_ml_imputed_simple_rule.csv")

# XGBoost with all ML-imputed features
features_for_model = ['Time_spent_Alone', 'Stage_fear_enc', 'Social_event_attendance',
                      'Going_outside', 'Drained_enc', 'Friends_circle_size', 'Post_frequency']

y_train = (train_personality == 'Extrovert').astype(int)

model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.006358,
    max_depth=8,
    random_state=42,
    use_label_encoder=False,
    verbosity=0
)

model.fit(train_imputed[features_for_model], y_train)
pred = model.predict(test_imputed[features_for_model])

test_imputed['Personality_xgb'] = pd.Series(pred).map({1: 'Extrovert', 0: 'Introvert'})

submission2 = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_imputed['Personality_xgb']
})
submission2.to_csv('001_ml_imputed_xgboost.csv', index=False)
print("Saved: 001_ml_imputed_xgboost.csv")

print("\n" + "="*60)
print("KEY INSIGHT")
print("="*60)
print("ML imputation might reveal the TRUE values that were masked!")
print("If the generator used a deterministic rule, ML might discover it!")
print("\nThese could be your 100% solutions!")