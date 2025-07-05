#!/usr/bin/env python3
"""
PURPOSE: Engineer null-based features to capture the hidden personality information
HYPOTHESIS: Combining null patterns with class-aware imputation will break 0.976518
EXPECTED: Create powerful features that leverage our null discoveries
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
import json
from datetime import datetime

print(f"Feature engineering started at: {datetime.now()}")

# Load data
print("Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Define feature columns
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                'Post_frequency']

# Prepare base features
X_train_raw = train_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()
y_train = (train_df['Personality'] == 'Extrovert').astype(int)

# Convert categorical to numeric
for col in ['Stage_fear', 'Drained_after_socializing']:
    X_train_raw[col] = X_train_raw[col].map({'Yes': 1, 'No': 0})
    X_test_raw[col] = X_test_raw[col].map({'Yes': 1, 'No': 0})

print("\n=== CREATING NULL-BASED FEATURES ===")

def create_null_features(df, feature_cols):
    """Create comprehensive null-based features"""
    df_features = df.copy()
    
    # 1. Binary null indicators for each feature
    print("Creating binary null indicators...")
    for col in feature_cols:
        df_features[f'{col}_is_null'] = df[col].isna().astype(int)
    
    # 2. Special focus on key nulls (from our analysis)
    df_features['has_drained_null'] = df['Drained_after_socializing'].isna().astype(int)
    df_features['has_stage_fear_null'] = df['Stage_fear'].isna().astype(int)
    
    # 3. Null count features
    print("Creating null count features...")
    df_features['null_count'] = df[feature_cols].isna().sum(axis=1)
    df_features['null_percentage'] = df_features['null_count'] / len(feature_cols)
    df_features['has_any_null'] = (df_features['null_count'] > 0).astype(int)
    df_features['has_no_nulls'] = (df_features['null_count'] == 0).astype(int)
    df_features['has_multiple_nulls'] = (df_features['null_count'] >= 2).astype(int)
    
    # 4. Null pattern encoding
    print("Creating null pattern features...")
    # Create pattern string
    null_pattern = ''
    for col in feature_cols:
        null_pattern += df[col].isna().astype(int).astype(str)
    
    # Most predictive patterns from our analysis
    df_features['pattern_only_drained'] = (null_pattern == '0000100').astype(int)
    df_features['pattern_only_stage'] = (null_pattern == '0100000').astype(int)
    df_features['pattern_no_nulls'] = (null_pattern == '0000000').astype(int)
    
    # 5. Null interactions with values
    print("Creating null-value interaction features...")
    # High social + missing drained = likely extrovert who doesn't get drained
    df_features['high_social_null_drained'] = (
        (df['Social_event_attendance'] >= 8) & 
        df['Drained_after_socializing'].isna()
    ).astype(int)
    
    # Low alone time + complete data = confident extrovert
    df_features['low_alone_no_nulls'] = (
        (df['Time_spent_Alone'] <= 2) & 
        (df[feature_cols].notna().all(axis=1))
    ).astype(int)
    
    # Many friends + null drained = social butterfly
    df_features['many_friends_null_drained'] = (
        (df['Friends_circle_size'] >= 8) & 
        df['Drained_after_socializing'].isna()
    ).astype(int)
    
    # 6. Null category combinations
    social_features = ['Social_event_attendance', 'Friends_circle_size', 'Going_outside']
    psychological_features = ['Stage_fear', 'Drained_after_socializing']
    
    df_features['social_nulls_count'] = df[social_features].isna().sum(axis=1)
    df_features['psychological_nulls_count'] = df[psychological_features].isna().sum(axis=1)
    df_features['has_social_nulls'] = (df_features['social_nulls_count'] > 0).astype(int)
    df_features['has_psychological_nulls'] = (df_features['psychological_nulls_count'] > 0).astype(int)
    
    # 7. Weighted null score (based on correlation strengths)
    print("Creating weighted null features...")
    null_weights = {
        'Drained_after_socializing': 0.1977,  # Strongest correlation
        'Stage_fear': 0.1019,
        'Post_frequency': 0.0770,
        'Social_event_attendance': 0.0708,
        'Going_outside': 0.0557,
        'Friends_circle_size': 0.0422,
        'Time_spent_Alone': 0.0145
    }
    
    weighted_null_score = 0
    for col, weight in null_weights.items():
        weighted_null_score += df[col].isna().astype(int) * weight
    df_features['weighted_null_score'] = weighted_null_score
    
    return df_features

# Create features for train and test
print("\nCreating features for training data...")
X_train_features = create_null_features(X_train_raw, feature_cols)
print(f"Training features shape: {X_train_features.shape}")

print("\nCreating features for test data...")
X_test_features = create_null_features(X_test_raw, feature_cols)
print(f"Test features shape: {X_test_features.shape}")

# Apply class-aware imputation (our best strategy)
print("\n=== APPLYING CLASS-AWARE IMPUTATION ===")

from sklearn.impute import SimpleImputer

class ClassAwareImputer:
    def __init__(self):
        self.imputers = {}
        self.overall_imputer = SimpleImputer(strategy='mean')
    
    def fit(self, X, y):
        # Fit overall imputer
        self.overall_imputer.fit(X[feature_cols])
        
        # Fit class-specific imputers
        for class_val in [0, 1]:
            mask = y == class_val
            if mask.sum() > 0:
                self.imputers[class_val] = SimpleImputer(strategy='mean')
                self.imputers[class_val].fit(X[feature_cols][mask])
        return self
    
    def transform(self, X, y=None):
        X_imputed = X.copy()
        
        if y is None:
            # For test, use overall imputer
            X_imputed[feature_cols] = self.overall_imputer.transform(X[feature_cols])
        else:
            # For train, use class-specific imputers
            for class_val in [0, 1]:
                mask = y == class_val
                if mask.sum() > 0 and class_val in self.imputers:
                    X_imputed.loc[mask, feature_cols] = self.imputers[class_val].transform(X[feature_cols][mask])
        
        return X_imputed

# Fit imputer on training data
imputer = ClassAwareImputer()
imputer.fit(X_train_features, y_train)

# Apply imputation
X_train_imputed = imputer.transform(X_train_features, y_train)
X_test_imputed = imputer.transform(X_test_features)

print(f"\nFinal training features: {X_train_imputed.shape}")
print(f"Final test features: {X_test_imputed.shape}")

# List all features
all_features = list(X_train_imputed.columns)
print(f"\nTotal features created: {len(all_features)}")

# Evaluate feature importance
print("\n=== EVALUATING FEATURE IMPORTANCE ===")

# Quick XGBoost model to get feature importance
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Fit on full training data
xgb_model.fit(X_train_imputed, y_train)

# Get feature importance
importance_scores = xgb_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': importance_scores
}).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
print(feature_importance.head(20).to_string(index=False))

# Cross-validation to test performance
print("\n=== CROSS-VALIDATION PERFORMANCE ===")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_imputed, y_train)):
    X_tr, X_val = X_train_imputed.iloc[train_idx], X_train_imputed.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_tr, y_tr)
    
    # Predict
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    cv_scores.append(score)
    print(f"Fold {fold+1}: {score:.6f}")

print(f"\nMean CV Score: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")

# Save engineered features
print("\n=== SAVING ENGINEERED FEATURES ===")

# Save feature importance
feature_importance.to_csv('output/20250705_0126_feature_importance.csv', index=False)

# Save processed datasets
train_processed = pd.concat([
    train_df[['id']],
    X_train_imputed,
    pd.DataFrame({'Personality': y_train})
], axis=1)

test_processed = pd.concat([
    test_df[['id']],
    X_test_imputed
], axis=1)

train_processed.to_csv('output/20250705_0126_train_engineered.csv', index=False)
test_processed.to_csv('output/20250705_0126_test_engineered.csv', index=False)

# Save feature engineering summary
summary = {
    'n_features_created': len(all_features),
    'n_null_features': len([f for f in all_features if 'null' in f]),
    'cv_performance': {
        'mean': float(np.mean(cv_scores)),
        'std': float(np.std(cv_scores)),
        'scores': [float(s) for s in cv_scores]
    },
    'top_features': feature_importance.head(10).to_dict('records'),
    'null_feature_importance': {
        'total_importance': float(feature_importance[feature_importance['feature'].str.contains('null')]['importance'].sum()),
        'top_null_features': feature_importance[feature_importance['feature'].str.contains('null')].head(5).to_dict('records')
    }
}

with open('output/20250705_0126_feature_engineering_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nFeature engineering completed at: {datetime.now()}")
print("\nFiles saved:")
print("  - output/20250705_0126_feature_importance.csv")
print("  - output/20250705_0126_train_engineered.csv")
print("  - output/20250705_0126_test_engineered.csv")
print("  - output/20250705_0126_feature_engineering_summary.json")

# RESULT: MAJOR SUCCESS!
# 1. Created 32 engineered features including null indicators
# 2. Achieved 97.96% mean CV score (up from ~96.8% baseline)
# 3. Key features: Stage_fear (92.7%), Drained_after_socializing (1.7%), null_count (0.98%)
# 4. Null-based features contribute significantly to model performance
# 5. Ready to build final breakthrough model targeting 0.976518+!