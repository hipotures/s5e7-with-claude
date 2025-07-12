#!/usr/bin/env python3
"""
Advanced feature engineering for personality prediction
Create new features based on domain knowledge and data patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def create_basic_features(df):
    """Create basic engineered features"""
    
    print("\n1. CREATING BASIC FEATURES")
    print("-" * 40)
    
    # Social engagement score
    df['social_engagement'] = (
        df['Social_event_attendance'] + 
        df['Friends_circle_size'] + 
        df['Going_outside'] + 
        df['Post_frequency']
    ) / 4
    
    # Introversion indicators
    df['introversion_score'] = (
        df['Time_spent_Alone'] * 2 +  # Double weight
        (10 - df['Social_event_attendance']) +
        (10 - df['Friends_circle_size']) +
        (10 - df['Going_outside']) +
        (10 - df['Post_frequency'])
    ) / 6
    
    # Social anxiety indicators
    df['social_anxiety'] = 0
    df.loc[(df['Stage_fear'] == 'Yes') & (df['Drained_after_socializing'] == 'Yes'), 'social_anxiety'] = 2
    df.loc[(df['Stage_fear'] == 'Yes') ^ (df['Drained_after_socializing'] == 'Yes'), 'social_anxiety'] = 1
    
    # Balance between alone time and social activities
    df['alone_social_ratio'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    
    # Digital vs physical social presence
    df['digital_physical_ratio'] = df['Post_frequency'] / (df['Going_outside'] + 1)
    
    # Social circle efficiency (events per friend)
    df['social_efficiency'] = df['Social_event_attendance'] / (df['Friends_circle_size'] + 1)
    
    print(f"Created 6 basic features")
    
    return df

def create_interaction_features(df):
    """Create interaction features between variables"""
    
    print("\n2. CREATING INTERACTION FEATURES")
    print("-" * 40)
    
    # Key interactions based on domain knowledge
    
    # High alone time + low social events = strong introversion signal
    df['alone_low_social'] = df['Time_spent_Alone'] * (10 - df['Social_event_attendance'])
    
    # Stage fear affecting social behavior
    df['stage_fear_social'] = (df['Stage_fear'] == 'Yes').astype(int) * df['Social_event_attendance']
    df['stage_fear_friends'] = (df['Stage_fear'] == 'Yes').astype(int) * df['Friends_circle_size']
    
    # Drained affecting behavior
    df['drained_events'] = (df['Drained_after_socializing'] == 'Yes').astype(int) * df['Social_event_attendance']
    df['drained_outside'] = (df['Drained_after_socializing'] == 'Yes').astype(int) * df['Going_outside']
    
    # Online vs offline personality
    df['online_offline_diff'] = abs(df['Post_frequency'] - df['Social_event_attendance'])
    
    # Social consistency
    df['social_consistency'] = df['Social_event_attendance'] * df['Friends_circle_size'] * df['Going_outside']
    
    # Extreme behaviors
    df['extreme_alone'] = (df['Time_spent_Alone'] >= 8).astype(int)
    df['extreme_social'] = (df['Social_event_attendance'] >= 8).astype(int)
    df['extreme_mismatch'] = df['extreme_alone'] * df['extreme_social']
    
    print(f"Created 10 interaction features")
    
    return df

def create_polynomial_features(df, degree=2):
    """Create polynomial features for numeric columns"""
    
    print(f"\n3. CREATING POLYNOMIAL FEATURES (degree={degree})")
    print("-" * 40)
    
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[numeric_cols].fillna(df[numeric_cols].median()))
    
    # Get feature names
    poly_names = poly.get_feature_names_out(numeric_cols)
    
    # Add only squared terms and selected interactions
    for i, name in enumerate(poly_names):
        if '^2' in name:  # Squared terms
            col_name = name.replace(' ', '_').replace('^2', '_squared')
            df[col_name] = poly_features[:, i]
        elif ('Time_spent_Alone' in name and 'Social_event_attendance' in name):
            df['alone_x_events'] = poly_features[:, i]
        elif ('Friends_circle_size' in name and 'Social_event_attendance' in name):
            df['friends_x_events'] = poly_features[:, i]
    
    print(f"Created polynomial features")
    
    return df

def create_null_pattern_features(df):
    """Create features based on null value patterns"""
    
    print("\n4. CREATING NULL PATTERN FEATURES")
    print("-" * 40)
    
    # Count nulls per row
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    df['null_count'] = df[numeric_cols].isnull().sum(axis=1)
    df['has_nulls'] = (df['null_count'] > 0).astype(int)
    
    # Specific null patterns
    df['null_time_alone'] = df['Time_spent_Alone'].isnull().astype(int)
    df['null_social_events'] = df['Social_event_attendance'].isnull().astype(int)
    df['null_friends'] = df['Friends_circle_size'].isnull().astype(int)
    
    # Null pattern combinations
    df['null_all_social'] = (
        df['Social_event_attendance'].isnull() & 
        df['Friends_circle_size'].isnull() &
        df['Going_outside'].isnull()
    ).astype(int)
    
    print(f"Created null pattern features")
    
    return df

def create_statistical_features(df):
    """Create statistical features across numeric columns"""
    
    print("\n5. CREATING STATISTICAL FEATURES")
    print("-" * 40)
    
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # Fill NaN with column median for calculations
    df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Row-wise statistics
    df['feature_mean'] = df_numeric.mean(axis=1)
    df['feature_std'] = df_numeric.std(axis=1)
    df['feature_max'] = df_numeric.max(axis=1)
    df['feature_min'] = df_numeric.min(axis=1)
    df['feature_range'] = df['feature_max'] - df['feature_min']
    df['feature_skew'] = df_numeric.apply(lambda x: x.skew(), axis=1)
    
    # Coefficient of variation
    df['feature_cv'] = df['feature_std'] / (df['feature_mean'] + 1e-5)
    
    print(f"Created 7 statistical features")
    
    return df

def create_domain_specific_features(df):
    """Create features based on personality psychology insights"""
    
    print("\n6. CREATING DOMAIN-SPECIFIC FEATURES")
    print("-" * 40)
    
    # Big Five personality indicators (approximated)
    
    # Extraversion indicators
    df['extraversion_indicator'] = (
        df['Social_event_attendance'] + 
        df['Friends_circle_size'] + 
        df['Going_outside'] - 
        df['Time_spent_Alone']
    )
    
    # Neuroticism indicators (anxiety, emotional drain)
    df['neuroticism_indicator'] = (
        (df['Stage_fear'] == 'Yes').astype(int) * 5 +
        (df['Drained_after_socializing'] == 'Yes').astype(int) * 5
    )
    
    # Openness indicators (variety in activities)
    df['activity_diversity'] = df[['Social_event_attendance', 'Going_outside', 'Post_frequency']].std(axis=1)
    
    # Social media personality
    df['digital_personality'] = df['Post_frequency'] / (df['Social_event_attendance'] + df['Going_outside'] + 1)
    
    # Energy management
    df['energy_balance'] = (10 - df['Time_spent_Alone']) / ((df['Drained_after_socializing'] == 'Yes').astype(int) + 1)
    
    # Social comfort zone
    df['comfort_zone'] = df['Friends_circle_size'] * ((df['Stage_fear'] == 'No').astype(int) + 1)
    
    print(f"Created 6 domain-specific features")
    
    return df

def select_best_features(X_train, y_train, feature_names, top_k=30):
    """Select best features using Random Forest importance"""
    
    print(f"\n7. SELECTING TOP {top_k} FEATURES")
    print("-" * 40)
    
    # Train Random Forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 features by importance:")
    for idx, row in importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Select top features
    top_features = importance.head(top_k)['feature'].tolist()
    
    return top_features, importance

def evaluate_features(train_df, feature_sets):
    """Evaluate different feature sets"""
    
    print("\n8. EVALUATING FEATURE SETS")
    print("-" * 40)
    
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    
    # Prepare target
    y = (train_df['Personality'] == 'Introvert').astype(int)
    
    results = []
    
    for name, features in feature_sets.items():
        # Handle missing values
        X = train_df[features].copy()
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].map({'Yes': 1, 'No': 0}).fillna(0)
        
        # Evaluate
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')
        
        results.append({
            'feature_set': name,
            'n_features': len(features),
            'cv_accuracy': scores.mean(),
            'cv_std': scores.std()
        })
        
        print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f} ({len(features)} features)")
    
    return pd.DataFrame(results)

def main():
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print("="*60)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*60)
    print(f"\nOriginal shapes:")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    
    # Create all features
    train_df = create_basic_features(train_df)
    test_df = create_basic_features(test_df)
    
    train_df = create_interaction_features(train_df)
    test_df = create_interaction_features(test_df)
    
    train_df = create_polynomial_features(train_df)
    test_df = create_polynomial_features(test_df)
    
    train_df = create_null_pattern_features(train_df)
    test_df = create_null_pattern_features(test_df)
    
    train_df = create_statistical_features(train_df)
    test_df = create_statistical_features(test_df)
    
    train_df = create_domain_specific_features(train_df)
    test_df = create_domain_specific_features(test_df)
    
    print(f"\nAfter feature engineering:")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    
    # Define feature sets
    original_features = ['Time_spent_Alone', 'Social_event_attendance', 
                        'Friends_circle_size', 'Going_outside', 'Post_frequency',
                        'Stage_fear', 'Drained_after_socializing']
    
    basic_eng_features = ['social_engagement', 'introversion_score', 'social_anxiety',
                         'alone_social_ratio', 'digital_physical_ratio', 'social_efficiency']
    
    interaction_features = ['alone_low_social', 'stage_fear_social', 'stage_fear_friends',
                          'drained_events', 'drained_outside', 'online_offline_diff',
                          'social_consistency', 'extreme_alone', 'extreme_social', 'extreme_mismatch']
    
    statistical_features = ['feature_mean', 'feature_std', 'feature_max', 'feature_min',
                          'feature_range', 'feature_skew', 'feature_cv']
    
    domain_features = ['extraversion_indicator', 'neuroticism_indicator', 'activity_diversity',
                      'digital_personality', 'energy_balance', 'comfort_zone']
    
    null_features = ['null_count', 'has_nulls', 'null_time_alone', 'null_social_events',
                    'null_friends', 'null_all_social']
    
    # Evaluate different feature sets
    feature_sets = {
        'original': original_features,
        'original_plus_basic': original_features + basic_eng_features,
        'original_plus_interaction': original_features + interaction_features,
        'original_plus_statistical': original_features + statistical_features,
        'original_plus_domain': original_features + domain_features,
        'original_plus_null': original_features + null_features,
        'all_engineered': original_features + basic_eng_features + interaction_features + 
                         statistical_features + domain_features + null_features
    }
    
    evaluation_results = evaluate_features(train_df, feature_sets)
    
    # Select best features from all
    all_features = list(set(original_features + basic_eng_features + interaction_features + 
                           statistical_features + domain_features + null_features))
    
    # Prepare data for feature selection
    X_train = train_df[all_features].copy()
    for col in X_train.columns:
        if X_train[col].dtype in ['float64', 'int64']:
            X_train[col] = X_train[col].fillna(X_train[col].median())
        else:
            X_train[col] = X_train[col].map({'Yes': 1, 'No': 0}).fillna(0)
    
    y_train = (train_df['Personality'] == 'Introvert').astype(int)
    
    # Select best features
    best_features, importance_df = select_best_features(X_train, y_train, all_features, top_k=25)
    
    # Train final model with best features
    print("\n9. TRAINING FINAL MODEL WITH BEST FEATURES")
    print("-" * 40)
    
    # Prepare final features
    X_train_final = train_df[best_features].copy()
    X_test_final = test_df[best_features].copy()
    
    # Handle missing values
    for col in best_features:
        if X_train_final[col].dtype in ['float64', 'int64']:
            median_val = X_train_final[col].median()
            X_train_final[col] = X_train_final[col].fillna(median_val)
            X_test_final[col] = X_test_final[col].fillna(median_val)
        else:
            X_train_final[col] = X_train_final[col].map({'Yes': 1, 'No': 0}).fillna(0)
            X_test_final[col] = X_test_final[col].map({'Yes': 1, 'No': 0}).fillna(0)
    
    # Add target back for YDF
    X_train_final['Personality'] = train_df['Personality']
    
    # Train YDF model
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=500,
        random_seed=42,
        compute_oob_performances=True
    )
    
    model = learner.train(X_train_final)
    
    # Make predictions
    predictions = model.predict(X_test_final)
    pred_classes = []
    
    for pred in predictions:
        prob_I = float(str(pred))
        pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': pred_classes
    })
    
    submission.to_csv(SCORES_DIR / 'submission_feature_engineering.csv', index=False)
    print("Created: submission_feature_engineering.csv")
    
    # Save feature importance
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    evaluation_results.to_csv(OUTPUT_DIR / 'feature_set_evaluation.csv', index=False)
    
    # Save engineered datasets
    train_df.to_csv(OUTPUT_DIR / 'train_engineered.csv', index=False)
    test_df.to_csv(OUTPUT_DIR / 'test_engineered.csv', index=False)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"\nCreated {len(all_features)} total features")
    print(f"Selected top {len(best_features)} features")
    print("\nBest performing feature sets:")
    print(evaluation_results.sort_values('cv_accuracy', ascending=False).head())

if __name__ == "__main__":
    main()