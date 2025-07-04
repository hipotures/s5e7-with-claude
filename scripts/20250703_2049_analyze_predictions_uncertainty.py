#!/usr/bin/env python3
"""
PURPOSE: Analyze prediction uncertainty to find potential improvements in model confidence
HYPOTHESIS: Examining prediction probabilities may reveal patterns for misclassified samples
EXPECTED: Identify samples with low confidence predictions that could be improved
RESULT: Found patterns in uncertain predictions that could guide model improvements
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Simple preprocessing
X_train = train_df.drop(columns=['Personality', 'id'])
y_train = train_df['Personality']
X_test = test_df.drop(columns=['id'])

# Encode
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].map({'Yes': 1, 'No': 0})
        X_test[col] = X_test[col].map({'Yes': 1, 'No': 0})

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Train simple model
model = xgb.XGBClassifier(n_estimators=1000, max_depth=8, random_state=42)
model.fit(X_train, y_train_encoded)

# Get probabilities
test_probas = model.predict_proba(X_test)
test_preds = model.predict(X_test)

# Analyze uncertainty
uncertainty = np.abs(test_probas[:, 1] - 0.5)
most_uncertain_idx = np.argsort(uncertainty)[:100]  # 100 most uncertain

print("PREDICTION UNCERTAINTY ANALYSIS")
print("="*60)
print(f"Total predictions: {len(test_preds)}")
print(f"Class distribution: {pd.Series(test_preds).value_counts().to_dict()}")
print(f"\nMost certain predictions (prob > 0.95 or < 0.05): {np.sum((test_probas[:, 1] > 0.95) | (test_probas[:, 1] < 0.05))}")
print(f"Uncertain predictions (0.4 < prob < 0.6): {np.sum((test_probas[:, 1] > 0.4) & (test_probas[:, 1] < 0.6))}")

# Analyze most uncertain cases
print(f"\n20 MOST UNCERTAIN PREDICTIONS:")
print("-"*80)
print(f"{'Index':>6} {'Test_ID':>8} {'Prob_0':>8} {'Prob_1':>8} {'Pred':>10} {'Uncertainty':>12}")
print("-"*80)

for i in range(20):
    idx = most_uncertain_idx[i]
    test_id = test_df.iloc[idx]['id']
    prob_0 = test_probas[idx, 0]
    prob_1 = test_probas[idx, 1]
    pred = le.inverse_transform([test_preds[idx]])[0]
    uncert = uncertainty[idx]
    print(f"{idx:>6} {test_id:>8} {prob_0:>8.4f} {prob_1:>8.4f} {pred:>10} {uncert:>12.4f}")

# Check feature values for uncertain cases
print("\nFEATURE ANALYSIS FOR UNCERTAIN CASES:")
print("-"*60)
uncertain_features = X_test.iloc[most_uncertain_idx[:20]]

# Check if Drained_after_socializing is the differentiator
if 'Drained_after_socializing' in X_test.columns:
    drain_values = uncertain_features['Drained_after_socializing'].value_counts()
    print(f"Drained_after_socializing distribution in uncertain cases:")
    print(drain_values)
    
# Look for patterns
print("\nMean feature values for uncertain vs certain predictions:")
certain_idx = np.where((test_probas[:, 1] > 0.95) | (test_probas[:, 1] < 0.05))[0]
uncertain_idx = np.where((test_probas[:, 1] > 0.45) & (test_probas[:, 1] < 0.55))[0]

for col in X_test.columns:
    if X_test[col].nunique() <= 20:  # Only for low cardinality
        certain_mean = X_test.iloc[certain_idx][col].mean()
        uncertain_mean = X_test.iloc[uncertain_idx][col].mean()
        diff = uncertain_mean - certain_mean
        if abs(diff) > 0.1:
            print(f"{col:30} Certain: {certain_mean:.3f}, Uncertain: {uncertain_mean:.3f}, Diff: {diff:+.3f}")

# Save uncertain predictions for manual analysis
uncertain_analysis = pd.DataFrame({
    'id': test_df.iloc[most_uncertain_idx[:100]]['id'],
    'predicted': le.inverse_transform(test_preds[most_uncertain_idx[:100]]),
    'prob_extrovert': test_probas[most_uncertain_idx[:100], 0],
    'prob_introvert': test_probas[most_uncertain_idx[:100], 1],
    'uncertainty': uncertainty[most_uncertain_idx[:100]]
})

# Add feature values
for col in X_test.columns:
    uncertain_analysis[col] = X_test.iloc[most_uncertain_idx[:100]][col].values

uncertain_analysis.to_csv('uncertain_predictions_analysis.csv', index=False)
print(f"\nSaved 100 most uncertain predictions to 'uncertain_predictions_analysis.csv'")

# Threshold analysis
print("\nTHRESHOLD ANALYSIS:")
print("-"*40)
for threshold in [0.45, 0.48, 0.50, 0.52, 0.55]:
    preds_at_threshold = (test_probas[:, 1] > threshold).astype(int)
    intro_count = np.sum(preds_at_threshold == 1)
    extro_count = np.sum(preds_at_threshold == 0)
    print(f"Threshold {threshold:.2f}: Introvert={intro_count}, Extrovert={extro_count}")