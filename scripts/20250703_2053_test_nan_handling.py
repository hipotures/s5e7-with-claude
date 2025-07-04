#!/usr/bin/env python3
"""
PURPOSE: Test different NaN handling strategies for missing data in the dataset
HYPOTHESIS: Different strategies for handling missing values may impact model performance
EXPECTED: Find the optimal NaN handling method (XGBoost native, fill with -1/0/median, etc.)
RESULT: Determined best approach for handling missing values in the personality dataset
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print("TESTING NaN HANDLING STRATEGIES")
print("="*60)

# Different NaN strategies
strategies = [
    {
        'name': 'XGBoost native (no fill)',
        'fill_method': None
    },
    {
        'name': 'Fill with -1',
        'fill_method': lambda df: df.fillna(-1)
    },
    {
        'name': 'Fill with 0',
        'fill_method': lambda df: df.fillna(0)
    },
    {
        'name': 'Fill with mode',
        'fill_method': lambda df: df.fillna(df.mode().iloc[0])
    },
    {
        'name': 'Fill with median',
        'fill_method': lambda df: df.apply(lambda x: x.fillna(x.median()) if x.dtype in ['float64', 'int64'] else x.fillna(x.mode()[0] if len(x.mode()) > 0 else 'missing'))
    }
]

# Encode target once
le = LabelEncoder()
y_train = le.fit_transform(train_df['Personality'])

best_score = 0
best_strategy = None

for strategy in strategies:
    print(f"\n\nStrategy: {strategy['name']}")
    print("-"*40)
    
    # Copy data
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # Handle NaN based on strategy
    if strategy['fill_method'] is not None:
        # Skip target and id columns
        cols_to_fill = [col for col in train_work.columns if col not in ['Personality', 'id']]
        train_work[cols_to_fill] = strategy['fill_method'](train_work[cols_to_fill])
        test_work[cols_to_fill] = strategy['fill_method'](test_work[cols_to_fill])
    
    # Encode Yes/No (skip if XGBoost native)
    if strategy['name'] != 'XGBoost native (no fill)':
        for col in train_work.columns:
            if train_work[col].dtype == 'object' and col not in ['Personality', 'id']:
                train_work[col] = train_work[col].map({'Yes': 1, 'No': 0})
                test_work[col] = test_work[col].map({'Yes': 1, 'No': 0})
    
    # Test with optimal features
    features = ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone']
    
    try:
        X_train = train_work[features]
        X_test = test_work[features]
        
        # XGBoost can handle categorical directly
        if strategy['name'] == 'XGBoost native (no fill)':
            # Convert Yes/No to category
            for col in features:
                if X_train[col].dtype == 'object':
                    X_train[col] = X_train[col].astype('category')
                    X_test[col] = X_test[col].astype('category')
        
        # Model
        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_score = scores.mean()
        
        print(f"CV Score: {cv_score:.6f}")
        print(f"CV Std: {scores.std():.6f}")
        
        # Train and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        pred_labels = le.inverse_transform(predictions)
        
        # Distribution
        unique, counts = np.unique(pred_labels, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"Prediction distribution: {dist}")
        
        if cv_score > best_score:
            best_score = cv_score
            best_strategy = strategy['name']
            
            # Save if very good
            if cv_score > 0.975:
                submission = pd.DataFrame({
                    'id': test_df['id'],
                    'Personality': pred_labels
                })
                filename = f'submission_nan_{strategy["name"].replace(" ", "_")}_{cv_score:.6f}.csv'
                submission.to_csv(filename, index=False)
                print(f"⭐ HIGH SCORE! Saved: {filename}")
                
    except Exception as e:
        print(f"Error: {e}")

print(f"\n\nBEST STRATEGY: {best_strategy} with CV score: {best_score:.6f}")

# Try one more thing - use ALL features with best NaN handling
print("\n\nTESTING WITH ALL FEATURES:")
print("-"*40)

train_all = train_df.copy()
test_all = test_df.copy()

# Fill NaN with -1 (or best strategy)
cols_to_fill = [col for col in train_all.columns if col not in ['Personality', 'id']]
train_all[cols_to_fill] = train_all[cols_to_fill].fillna(-1)
test_all[cols_to_fill] = test_all[cols_to_fill].fillna(-1)

# Encode Yes/No
for col in train_all.columns:
    if train_all[col].dtype == 'object' and col not in ['Personality', 'id']:
        train_all[col] = train_all[col].map({'Yes': 1, 'No': 0})
        test_all[col] = test_all[col].map({'Yes': 1, 'No': 0})

# Use all features
feature_cols = [col for col in train_all.columns if col not in ['Personality', 'id']]
X_train_all = train_all[feature_cols]
X_test_all = test_all[feature_cols]

# Model with all features
model_all = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

scores_all = cross_val_score(model_all, X_train_all, y_train, cv=5, scoring='accuracy')
print(f"All features CV Score: {scores_all.mean():.6f}")

# Also check feature importance
model_all.fit(X_train_all, y_train)
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_all.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 important features:")
print(importances.head(10))