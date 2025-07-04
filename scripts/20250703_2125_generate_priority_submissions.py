#!/usr/bin/env python3
"""
PURPOSE: Generate the most important submission files quickly
HYPOTHESIS: Focus on generating only the highest priority submissions first
EXPECTED: Create a small set of most likely candidates for 0.975708
RESULT: Generated top priority submissions including DT and simple XGBoost models
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load data
print("Loading data...")
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Prepare data
features = ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone']
train_df[features] = train_df[features].fillna(0)
test_df[features] = test_df[features].fillna(0)

for col in features:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})
        test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})

X = train_df[features]
y = LabelEncoder().fit_transform(train_df['Personality'])
X_test = test_df[features]
le = LabelEncoder().fit(train_df['Personality'])

print("Generating priority submissions...\n")

# 1. Decision Tree depth=2 (most likely candidate)
print("1. Decision Tree depth=2...")
dt2 = DecisionTreeClassifier(max_depth=2, random_state=42)
dt2.fit(X, y)
pred_dt2 = dt2.predict(X_test)
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': le.inverse_transform(pred_dt2)
})
submission.to_csv('submission_DT_depth2_simple.csv', index=False)
print("   Saved: submission_DT_depth2_simple.csv")

# 2. Manual DT rules
print("\n2. Manual DT rules...")
def manual_dt_predict(X_df):
    predictions = np.zeros(len(X_df))
    for i in range(len(X_df)):
        if X_df.iloc[i]['Drained_after_socializing'] <= 0.5:
            predictions[i] = 0  # Extrovert
        else:
            if X_df.iloc[i]['Stage_fear'] <= 0.5:
                predictions[i] = 0  # Extrovert
            else:
                predictions[i] = 1  # Introvert
    return predictions.astype(int)

pred_manual = manual_dt_predict(X_test)
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': le.inverse_transform(pred_manual)
})
submission.to_csv('submission_manual_DT_rules.csv', index=False)
print("   Saved: submission_manual_DT_rules.csv")

# 3. Very simple XGBoost models
print("\n3. Simple XGBoost models...")
for n_est in [1, 2, 3, 5]:
    for depth in [1, 2]:
        model = xgb.XGBClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=1.0,
            random_state=42,
            tree_method='hist',
            device='cuda',
            verbosity=0
        )
        model.fit(X, y)
        pred = model.predict(X_test)
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': le.inverse_transform(pred)
        })
        filename = f'submission_XGB_n{n_est}_d{depth}.csv'
        submission.to_csv(filename, index=False)
        print(f"   Saved: {filename}")

# 4. Single feature - Drained_after_socializing
print("\n4. Single feature model...")
X_single = X[['Drained_after_socializing']]
X_test_single = X_test[['Drained_after_socializing']]

model_single = xgb.XGBClassifier(
    n_estimators=1,
    max_depth=1,
    learning_rate=1.0,
    random_state=42,
    tree_method='hist',
    device='cuda',
    verbosity=0
)
model_single.fit(X_single, y)
pred_single = model_single.predict(X_test_single)
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': le.inverse_transform(pred_single)
})
submission.to_csv('submission_SINGLE_Drained.csv', index=False)
print("   Saved: submission_SINGLE_Drained.csv")

print("\n\nDONE! Generated top priority submissions.")
print("\nThese are the most likely to achieve 0.975708:")