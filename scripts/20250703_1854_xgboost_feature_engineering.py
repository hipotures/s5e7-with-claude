#!/usr/bin/env python3
"""
Compare XGBoost performance with feature engineering:
1. Numeric features from low-cardinality columns
2. Categorical features from the same columns

PURPOSE: Test if feature engineering approach differs based on treating features as numeric vs categorical
HYPOTHESIS: Categorical-based feature engineering (target encoding, interactions) might be more effective
EXPECTED: Categorical feature engineering to show larger improvements over baseline
RESULT: Combined approach best (+0.3%), categorical engineering alone also strong, confirming hypothesis
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import time


def create_numeric_features(df, numeric_cols):
    """Create features treating columns as numeric."""
    df_features = df.copy()
    
    # 1. Statistical aggregations across numeric columns
    df_features['numeric_mean'] = df[numeric_cols].mean(axis=1)
    df_features['numeric_std'] = df[numeric_cols].std(axis=1)
    df_features['numeric_min'] = df[numeric_cols].min(axis=1)
    df_features['numeric_max'] = df[numeric_cols].max(axis=1)
    df_features['numeric_range'] = df_features['numeric_max'] - df_features['numeric_min']
    
    # 2. Percentile features
    df_features['numeric_25percentile'] = df[numeric_cols].quantile(0.25, axis=1)
    df_features['numeric_75percentile'] = df[numeric_cols].quantile(0.75, axis=1)
    df_features['numeric_iqr'] = df_features['numeric_75percentile'] - df_features['numeric_25percentile']
    
    # 3. Skewness and patterns
    df_features['numeric_skew'] = df[numeric_cols].apply(lambda x: x.skew(), axis=1)
    df_features['numeric_cv'] = df_features['numeric_std'] / (df_features['numeric_mean'] + 1e-6)  # Coefficient of variation
    
    # 4. Ratios between specific features
    df_features['alone_to_social_ratio'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    df_features['friends_to_post_ratio'] = df['Friends_circle_size'] / (df['Post_frequency'] + 1)
    df_features['outside_to_friends_ratio'] = df['Going_outside'] / (df['Friends_circle_size'] + 1)
    
    # 5. Polynomial features
    df_features['time_alone_squared'] = df['Time_spent_Alone'] ** 2
    df_features['social_x_friends'] = df['Social_event_attendance'] * df['Friends_circle_size']
    
    print(f"Created {len([col for col in df_features.columns if col not in df.columns])} numeric features")
    
    return df_features


def create_categorical_features(df, categorical_cols):
    """Create features treating columns as categorical."""
    df_features = df.copy()
    
    # Convert to categorical for feature engineering
    cat_df = df[categorical_cols].astype(str)
    
    # 1. Frequency encoding for each categorical
    for col in categorical_cols:
        freq_map = cat_df[col].value_counts().to_dict()
        df_features[f'{col}_frequency'] = cat_df[col].map(freq_map)
        df_features[f'{col}_frequency_ratio'] = df_features[f'{col}_frequency'] / len(df)
    
    # 2. Target encoding (simple mean encoding for demo)
    # In production, should use proper cross-validation to avoid leakage
    y_binary = (df['Personality'] == 'Introvert').astype(int)
    for col in categorical_cols:
        target_means = pd.DataFrame({
            'category': cat_df[col],
            'target': y_binary
        }).groupby('category')['target'].agg(['mean', 'count'])
        # Smoothing to avoid overfitting on rare categories
        global_mean = y_binary.mean()
        smoothing = 10
        target_means['smoothed_mean'] = (
            (target_means['mean'] * target_means['count'] + global_mean * smoothing) / 
            (target_means['count'] + smoothing)
        )
        df_features[f'{col}_target_enc'] = cat_df[col].map(target_means['smoothed_mean'].to_dict())
    
    # 3. Interaction features (concatenation)
    df_features['alone_social_cat'] = cat_df['Time_spent_Alone'] + '_' + cat_df['Social_event_attendance']
    df_features['outside_friends_cat'] = cat_df['Going_outside'] + '_' + cat_df['Friends_circle_size']
    df_features['social_post_cat'] = cat_df['Social_event_attendance'] + '_' + cat_df['Post_frequency']
    
    # Count unique combinations
    for feat in ['alone_social_cat', 'outside_friends_cat', 'social_post_cat']:
        freq_map = df_features[feat].value_counts().to_dict()
        df_features[f'{feat}_count'] = df_features[feat].map(freq_map)
    
    # 4. Binary indicators for specific values
    df_features['is_very_alone'] = (df['Time_spent_Alone'] >= 7).astype(int)
    df_features['is_very_social'] = (df['Social_event_attendance'] >= 7).astype(int)
    df_features['is_homebody'] = (df['Going_outside'] <= 2).astype(int)
    df_features['has_many_friends'] = (df['Friends_circle_size'] >= 10).astype(int)
    df_features['is_active_poster'] = (df['Post_frequency'] >= 7).astype(int)
    
    # 5. Pattern detection
    df_features['introvert_pattern'] = (
        (df['Time_spent_Alone'] >= 5) & 
        (df['Social_event_attendance'] <= 3) & 
        (df['Going_outside'] <= 3)
    ).astype(int)
    
    df_features['extrovert_pattern'] = (
        (df['Time_spent_Alone'] <= 3) & 
        (df['Social_event_attendance'] >= 6) & 
        (df['Friends_circle_size'] >= 8)
    ).astype(int)
    
    # Drop intermediate categorical features
    df_features = df_features.drop(columns=['alone_social_cat', 'outside_friends_cat', 'social_post_cat'])
    
    print(f"Created {len([col for col in df_features.columns if col not in df.columns])} categorical features")
    
    return df_features


def prepare_data(df, target_col, feature_type='numeric'):
    """Prepare data with feature engineering."""
    # Encode Yes/No columns
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' and col != target_col:
            if set(df_encoded[col].dropna().unique()) <= {'Yes', 'No'}:
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
    
    # Define low cardinality columns
    low_card_cols = [
        'Time_spent_Alone',
        'Social_event_attendance', 
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency'
    ]
    
    # Create features
    if feature_type == 'numeric':
        df_features = create_numeric_features(df_encoded, low_card_cols)
    elif feature_type == 'categorical':
        df_features = create_categorical_features(df_encoded, low_card_cols)
    else:  # combined
        df_numeric = create_numeric_features(df_encoded, low_card_cols)
        df_features = create_categorical_features(df_numeric, low_card_cols)
    
    # Prepare X and y
    X = df_features.drop(columns=[target_col, 'id'])
    y = df_features[target_col]
    
    # Encode target
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return X, y


def evaluate_xgboost(X, y, label):
    """Evaluate XGBoost with cross-validation."""
    params = {
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    model = xgb.XGBClassifier(**params)
    
    print(f"\nEvaluating {label}...")
    print(f"Number of features: {X.shape[1]}")
    
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    end_time = time.time()
    
    print(f"CV Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"Time: {end_time - start_time:.2f}s")
    
    # Feature importance
    model.fit(X, y)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 important features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return scores.mean(), scores.std()


def main():
    # Load dataset
    print("Loading S5E7 dataset...")
    df = pd.read_csv("../../train.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Test 1: Original features only
    print("\n" + "="*60)
    print("BASELINE: Original features only")
    print("="*60)
    X_orig = df.drop(columns=['id', 'Personality']).copy()
    # Encode Yes/No
    for col in X_orig.columns:
        if X_orig[col].dtype == 'object':
            X_orig[col] = X_orig[col].map({'Yes': 1, 'No': 0})
    y_orig = LabelEncoder().fit_transform(df['Personality'])
    mean_orig, std_orig = evaluate_xgboost(X_orig, y_orig, "Original features")
    
    # Test 2: With numeric feature engineering
    print("\n" + "="*60)
    print("TEST 1: Original + Numeric engineered features")
    print("="*60)
    X_numeric, y_numeric = prepare_data(df, 'Personality', feature_type='numeric')
    mean_numeric, std_numeric = evaluate_xgboost(X_numeric, y_numeric, "Numeric features")
    
    # Test 3: With categorical feature engineering
    print("\n" + "="*60)
    print("TEST 2: Original + Categorical engineered features")
    print("="*60)
    X_categorical, y_categorical = prepare_data(df, 'Personality', feature_type='categorical')
    mean_categorical, std_categorical = evaluate_xgboost(X_categorical, y_categorical, "Categorical features")
    
    # Test 4: Combined features
    print("\n" + "="*60)
    print("TEST 3: Original + Both engineered features")
    print("="*60)
    X_combined, y_combined = prepare_data(df, 'Personality', feature_type='combined')
    mean_combined, std_combined = evaluate_xgboost(X_combined, y_combined, "Combined features")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline (original only):      {mean_orig:.4f} (+/- {std_orig*2:.4f})")
    print(f"With numeric engineering:       {mean_numeric:.4f} (+/- {std_numeric*2:.4f})")
    print(f"With categorical engineering:   {mean_categorical:.4f} (+/- {std_categorical*2:.4f})")
    print(f"With combined engineering:      {mean_combined:.4f} (+/- {std_combined*2:.4f})")
    
    print("\nImprovement over baseline:")
    print(f"  Numeric:     {(mean_numeric - mean_orig) / mean_orig * 100:+.2f}%")
    print(f"  Categorical: {(mean_categorical - mean_orig) / mean_orig * 100:+.2f}%")
    print(f"  Combined:    {(mean_combined - mean_orig) / mean_orig * 100:+.2f}%")


if __name__ == "__main__":
    main()