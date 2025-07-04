#!/usr/bin/env python3
"""Compare Method 1 (train+test) vs Method 2 (train only) imputation"""

# PURPOSE: Compare different imputation strategies to see if data leakage helps performance
# HYPOTHESIS: Combined train+test imputation might capture true patterns through mild leakage
# EXPECTED: Method 1 (combined) will outperform Method 2 (train-only) due to pattern capture
# RESULT: Created 4 submissions testing combinations of imputation methods and models

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import xgboost as xgb

print("COMPARING IMPUTATION METHODS")
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

# First encode categorical features
for df in [train_df, test_df]:
    df['Stage_fear_enc'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
    df['Drained_enc'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

all_features = numerical_features + ['Stage_fear_enc', 'Drained_enc']

print("\n" + "="*60)
print("METHOD 1: Train+Test Combined Imputation")
print("="*60)

def ml_impute_combined(df_train, df_test, target_col, feature_cols):
    """Impute using train+test combined (potential leakage)"""
    print(f"\nImputing {target_col} (combined)...")
    
    # Combine train and test
    df_combined = pd.concat([df_train[feature_cols + [target_col]], 
                            df_test[feature_cols + [target_col]]], 
                            ignore_index=True)
    
    mask_not_missing = df_combined[target_col].notna()
    
    if mask_not_missing.sum() == len(df_combined):
        print(f"  No missing values in {target_col}")
        return df_train[target_col], df_test[target_col]
    
    X_train_imp = df_combined.loc[mask_not_missing, feature_cols]
    y_train_imp = df_combined.loc[mask_not_missing, target_col]
    X_predict = df_combined.loc[~mask_not_missing, feature_cols]
    
    # Use appropriate model
    if df_combined[target_col].dtype == 'object' or len(df_combined[target_col].unique()) < 10:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Handle missing in features
    for col in feature_cols:
        if X_train_imp[col].isna().any():
            if X_train_imp[col].dtype == 'object':
                fill_val = X_train_imp[col].mode()[0] if len(X_train_imp[col].mode()) > 0 else 'Unknown'
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

# Method 1: Combined imputation
train_method1 = train_df.copy()
test_method1 = test_df.copy()

for target in all_features:
    other_features = [f for f in all_features if f != target]
    train_method1[target], test_method1[target] = ml_impute_combined(
        train_method1, test_method1, target, other_features
    )

print("\n" + "="*60)
print("METHOD 2: Train-Only Imputation (IterativeImputer)")
print("="*60)

# Method 2: Train-only imputation
X_train = train_df[all_features].copy()
X_test = test_df[all_features].copy()

imputer = IterativeImputer(
    estimator=xgb.XGBRegressor(n_estimators=100, random_state=42),
    max_iter=10,
    random_state=42
)

# Fit on train only
print("Fitting imputer on train data only...")
X_train_method2 = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=all_features,
    index=X_train.index
)

print("Applying to test data...")
X_test_method2 = pd.DataFrame(
    imputer.transform(X_test),
    columns=all_features,
    index=X_test.index
)

# Compare results
print("\n" + "="*60)
print("COMPARISON OF IMPUTATION RESULTS")
print("="*60)

# Check Drained_enc specifically
mask_train = train_df['Drained_after_socializing'].isna()
mask_test = test_df['Drained_after_socializing'].isna()

if mask_train.sum() > 0:
    print(f"\nTrain missing Drained values: {mask_train.sum()}")
    print(f"Method 1 (combined) mean: {train_method1.loc[mask_train, 'Drained_enc'].mean():.3f}")
    print(f"Method 2 (train-only) mean: {X_train_method2.loc[mask_train, 'Drained_enc'].mean():.3f}")

if mask_test.sum() > 0:
    print(f"\nTest missing Drained values: {mask_test.sum()}")
    print(f"Method 1 (combined) mean: {test_method1.loc[mask_test, 'Drained_enc'].mean():.3f}")
    print(f"Method 2 (train-only) mean: {X_test_method2.loc[mask_test, 'Drained_enc'].mean():.3f}")
    
    # Distribution comparison
    print(f"\nMethod 1 distribution: >0.5: {(test_method1.loc[mask_test, 'Drained_enc'] > 0.5).sum()}, <=0.5: {(test_method1.loc[mask_test, 'Drained_enc'] <= 0.5).sum()}")
    print(f"Method 2 distribution: >0.5: {(X_test_method2.loc[mask_test, 'Drained_enc'] > 0.5).sum()}, <=0.5: {(X_test_method2.loc[mask_test, 'Drained_enc'] <= 0.5).sum()}")

print("\n" + "="*60)
print("CREATING SUBMISSIONS")
print("="*60)

# Prepare for predictions
y_train = (train_personality == 'Extrovert').astype(int)

# Model for predictions
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.006358,
    max_depth=8,
    random_state=42,
    use_label_encoder=False,
    verbosity=0
)

# Submission 1: Method 1 (combined) with simple rule
test_method1['Personality'] = test_method1['Drained_enc'].apply(
    lambda x: 'Introvert' if x > 0.5 else 'Extrovert'
)

submission1 = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_method1['Personality']
})
submission1.to_csv('025_method1_combined_simple.csv', index=False)
print("Saved: 025_method1_combined_simple.csv (Method 1: train+test imputation, simple rule)")

# Submission 2: Method 1 (combined) with XGBoost
features_for_model = ['Time_spent_Alone', 'Stage_fear_enc', 'Social_event_attendance',
                      'Going_outside', 'Drained_enc', 'Friends_circle_size', 'Post_frequency']

model.fit(train_method1[features_for_model], y_train)
pred = model.predict(test_method1[features_for_model])
test_method1['Personality_xgb'] = pd.Series(pred).map({1: 'Extrovert', 0: 'Introvert'})

submission2 = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_method1['Personality_xgb']
})
submission2.to_csv('026_method1_combined_xgboost.csv', index=False)
print("Saved: 026_method1_combined_xgboost.csv (Method 1: train+test imputation, XGBoost)")

# Submission 3: Method 2 (train-only) with simple rule
test_method2_df = test_df.copy()
test_method2_df['Drained_enc'] = X_test_method2['Drained_enc']
test_method2_df['Personality'] = test_method2_df['Drained_enc'].apply(
    lambda x: 'Introvert' if x > 0.5 else 'Extrovert'
)

submission3 = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_method2_df['Personality']
})
submission3.to_csv('027_method2_trainonly_simple.csv', index=False)
print("Saved: 027_method2_trainonly_simple.csv (Method 2: train-only imputation, simple rule)")

# Submission 4: Method 2 (train-only) with XGBoost
# Prepare train data with method2 imputation
train_method2_df = train_df.copy()
for col in all_features:
    train_method2_df[col] = X_train_method2[col]

# Prepare test data with method2 imputation
test_method2_full = test_df.copy()
for col in all_features:
    test_method2_full[col] = X_test_method2[col]

# Train new model
model2 = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.006358,
    max_depth=8,
    random_state=42,
    use_label_encoder=False,
    verbosity=0
)

model2.fit(train_method2_df[features_for_model], y_train)
pred2 = model2.predict(test_method2_full[features_for_model])
test_method2_full['Personality'] = pd.Series(pred2).map({1: 'Extrovert', 0: 'Introvert'})

submission4 = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_method2_full['Personality']
})
submission4.to_csv('028_method2_trainonly_xgboost.csv', index=False)
print("Saved: 028_method2_trainonly_xgboost.csv (Method 2: train-only imputation, XGBoost)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Method 1 (combined): Uses train+test data together for imputation")
print("                     Potential data leakage but might capture true pattern")
print("\nMethod 2 (train-only): Uses only train data for imputation")
print("                       No data leakage, more conservative approach")
print("\nFiles created:")
print("025: Method 1 + Simple rule")
print("026: Method 1 + XGBoost")
print("027: Method 2 + Simple rule")
print("028: Method 2 + XGBoost")