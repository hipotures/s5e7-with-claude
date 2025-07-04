#!/usr/bin/env python3
"""
PURPOSE: Debug predictions to understand label mapping issue in classification models
HYPOTHESIS: The model predictions might have inverted labels (Introvert/Extrovert swapped)
EXPECTED: Identify if label encoding is causing misclassification between personality types
RESULT: Discovered that labels were indeed swapped, requiring a fix in submission files
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Prepare features
X_train = train_df.drop(columns=['Personality', 'id'])
y_train = train_df['Personality']

# Encode Yes/No columns
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].map({'Yes': 1, 'No': 0})

# Encode target
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

print("\nLabel encoding:")
print(f"Classes: {le.classes_}")
print(f"Extrovert -> {le.transform(['Extrovert'])[0]}")
print(f"Introvert -> {le.transform(['Introvert'])[0]}")

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Train simple XGBoost
print("\nTraining XGBoost...")
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)

# Make predictions on validation
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)

print("\nValidation predictions analysis:")
print(f"Unique predicted classes: {np.unique(y_pred)}")
print(f"Prediction distribution: {pd.Series(y_pred).value_counts().to_dict()}")
print(f"True distribution: {pd.Series(y_val).value_counts().to_dict()}")

# Compare predictions
print("\nFirst 20 validation samples:")
print("True | Pred | Proba_0 | Proba_1 | True_label | Pred_label")
print("-" * 60)
for i in range(20):
    true_class = y_val[i]
    pred_class = y_pred[i]
    proba_0 = y_pred_proba[i, 0]
    proba_1 = y_pred_proba[i, 1]
    true_label = le.inverse_transform([true_class])[0]
    pred_label = le.inverse_transform([pred_class])[0]
    match = "✓" if true_class == pred_class else "✗"
    print(f"{true_class:4} | {pred_class:4} | {proba_0:7.4f} | {proba_1:7.4f} | {true_label:10} | {pred_label:10} {match}")

# Now check test predictions
print("\n\nTest set predictions:")
X_test = test_df.drop(columns=['id'])
for col in X_test.columns:
    if X_test[col].dtype == 'object':
        X_test[col] = X_test[col].map({'Yes': 1, 'No': 0})

test_pred = model.predict(X_test)
test_pred_proba = model.predict_proba(X_test)

print(f"Test prediction distribution: {pd.Series(test_pred).value_counts().to_dict()}")
print(f"Test prediction labels: {pd.Series(le.inverse_transform(test_pred)).value_counts().to_dict()}")

# Compare with manual submission
manual_sub = pd.read_csv('/mnt/ml/minotaur/submission-s5e7-20250701_211313-000-0.9637.csv')
print(f"\nManual submission distribution: {manual_sub['personality'].value_counts().to_dict()}")

# Check if predictions are inverted
print("\nFirst 10 test predictions comparison:")
print("ID    | My_pred | My_label   | Manual_label | Match?")
print("-" * 55)
for i in range(10):
    test_id = test_df.iloc[i]['id']
    my_pred_class = test_pred[i]
    my_pred_label = le.inverse_transform([my_pred_class])[0]
    manual_label = manual_sub[manual_sub['id'] == test_id]['personality'].values[0]
    match = "✓" if my_pred_label == manual_label else "✗"
    print(f"{test_id} | {my_pred_class:7} | {my_pred_label:10} | {manual_label:12} | {match}")

# Final check - are all predictions inverted?
my_predictions = pd.Series(le.inverse_transform(test_pred))
manual_predictions = manual_sub['personality']
matches = sum(my_predictions == manual_predictions)
print(f"\nTotal matches: {matches}/{len(test_df)} ({matches/len(test_df)*100:.1f}%)")

# Check if inverting helps
inverted_mapping = {0: 'Introvert', 1: 'Extrovert'}  # Inverted
inverted_predictions = [inverted_mapping[p] for p in test_pred]
inverted_matches = sum(pd.Series(inverted_predictions) == manual_predictions)
print(f"Matches with inverted mapping: {inverted_matches}/{len(test_df)} ({inverted_matches/len(test_df)*100:.1f}%)")