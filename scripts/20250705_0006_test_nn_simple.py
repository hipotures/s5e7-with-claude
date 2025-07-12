#!/usr/bin/env python3
"""
Quick test to check if neural network can learn with class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

# Load data
print("Loading data...")
train_df = pd.read_csv('../../train.csv')

# Simple preprocessing
X = train_df.drop(['id', 'Personality'], axis=1).copy()
y = (train_df['Personality'] == 'Extrovert').astype(int)

# Handle categorical
for col in ['Stage_fear', 'Drained_after_socializing']:
    X[col] = X[col].map({'Yes': 1.0, 'No': 0.0})

# Fill missing values with mean
X = X.fillna(X.mean())

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print(f"\nClass distribution in training:")
print(f"Class 0 (Introvert): {(y_train==0).sum()} ({(y_train==0).mean():.2%})")
print(f"Class 1 (Extrovert): {(y_train==1).sum()} ({(y_train==1).mean():.2%})")

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights: {class_weight_dict}")

# Create simple model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nTraining WITHOUT class weights:")
history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=1
)

# Check predictions
probs = model.predict(X_val, verbose=0).flatten()
print(f"\nPredictions WITHOUT weights:")
print(f"Min prob: {probs.min():.4f}, Max prob: {probs.max():.4f}, Mean prob: {probs.mean():.4f}")
print(f"Predictions == 0: {(probs < 0.5).sum()}, Predictions == 1: {(probs >= 0.5).sum()}")

# Retrain with class weights
model2 = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n\nTraining WITH class weights:")
history2 = model2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    class_weight=class_weight_dict,
    verbose=1
)

# Check predictions
probs2 = model2.predict(X_val, verbose=0).flatten()
print(f"\nPredictions WITH weights:")
print(f"Min prob: {probs2.min():.4f}, Max prob: {probs2.max():.4f}, Mean prob: {probs2.mean():.4f}")
print(f"Predictions == 0: {(probs2 < 0.5).sum()}, Predictions == 1: {(probs2 >= 0.5).sum()}")

# Try different thresholds
print("\n\nTrying different thresholds:")
for threshold in [0.26, 0.35, 0.5, 0.74]:
    preds = (probs2 >= threshold).astype(int)
    acc = (preds == y_val).mean()
    print(f"Threshold {threshold:.2f}: Accuracy = {acc:.4f}")