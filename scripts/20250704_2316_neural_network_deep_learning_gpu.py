#!/usr/bin/env python3
"""
PURPOSE: Deep learning with neural networks to capture complex patterns for 0.976518
HYPOTHESIS: Neural networks with attention mechanisms can find subtle feature interactions
EXPECTED: Achieve breakthrough performance using deep learning on GPU
RESULT: [To be filled after execution]

NOTE: Designed for GPU server - uses TensorFlow with GPU acceleration
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPUs")
    except RuntimeError as e:
        print(e)

# Load data
print("Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Prepare features with neural network in mind
def prepare_features_nn(df, is_train=True):
    if is_train:
        X = df.drop(['id', 'Personality'], axis=1).copy()
        y = (df['Personality'] == 'Extrovert').astype(int)
    else:
        X = df.drop(['id'], axis=1).copy()
        y = None
    
    # Convert categorical
    for col in ['Stage_fear', 'Drained_after_socializing']:
        # Create embeddings for categorical
        X[f'{col}_missing'] = X[col].isna().astype(float)
        X[col] = X[col].map({'Yes': 1.0, 'No': 0.0})
        X[col] = X[col].fillna(0.5)  # Missing as middle value
    
    # Normalize numerical features
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                     'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    for col in numerical_cols:
        X[f'{col}_normalized'] = (X[col] - X[col].mean()) / (X[col].std() + 1e-8)
        # Keep original for embeddings
    
    return X, y

X_train, y_train = prepare_features_nn(train_df, is_train=True)
X_test, _ = prepare_features_nn(test_df, is_train=False)

print(f"Features: {X_train.shape[1]}")

# Calculate class weights for imbalanced dataset
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights: {class_weight_dict}")
print(f"Class balance: {np.bincount(y_train)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model architectures
def create_attention_model(input_dim):
    """Neural network with self-attention mechanism"""
    inputs = layers.Input(shape=(input_dim,))
    
    # Initial dense layers
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Self-attention mechanism
    attention = layers.Dense(128, activation='tanh')(x)
    attention = layers.Dense(128, activation='softmax')(attention)
    attended = layers.Multiply()([x, attention])
    
    # Combine attended and original
    x = layers.Concatenate()([x, attended])
    
    # Final layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_deep_model(input_dim):
    """Deep neural network with residual connections"""
    inputs = layers.Input(shape=(input_dim,))
    
    # First block
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Residual blocks
    for units in [256, 256, 128, 128]:
        residual = x
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Residual connection
        if residual.shape[-1] != units:
            residual = layers.Dense(units)(residual)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
    
    # Final layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_embedding_model(input_dim):
    """Model with entity embeddings for categorical features"""
    inputs = layers.Input(shape=(input_dim,))
    
    # Split features
    continuous = layers.Lambda(lambda x: x[:, :5])(inputs)  # First 5 are original continuous
    categorical = layers.Lambda(lambda x: x[:, 5:7])(inputs)  # Stage_fear, Drained
    rest = layers.Lambda(lambda x: x[:, 7:])(inputs)  # Rest of features
    
    # Embeddings for pseudo-categorical (discretized continuous)
    embedded_features = []
    
    # Time alone embedding (0-10 hours)
    time_alone = layers.Lambda(lambda x: tf.clip_by_value(x[:, 0:1], 0, 10))(continuous)
    time_alone_int = layers.Lambda(lambda x: tf.cast(x, tf.int32))(time_alone)
    time_alone_emb = layers.Embedding(11, 8, input_length=1)(time_alone_int)
    time_alone_emb = layers.Flatten()(time_alone_emb)
    embedded_features.append(time_alone_emb)
    
    # Combine all features
    all_features = layers.Concatenate()([
        continuous, categorical, rest
    ] + embedded_features)
    
    # Dense network
    x = layers.Dense(256, activation='relu')(all_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Training with cross-validation
print("\n=== NEURAL NETWORK TRAINING ===")

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled, y_train)):
    print(f"\nFold {fold + 1}")
    
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train.iloc[train_idx].values, y_train.iloc[val_idx].values
    
    # Create and compile model
    if fold < 2:
        model = create_attention_model(X_train.shape[1])
        model_type = "attention"
    elif fold < 4:
        model = create_deep_model(X_train.shape[1])
        model_type = "deep"
    else:
        model = create_embedding_model(X_train.shape[1])
        model_type = "embedding"
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with class weights
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=0
    )
    
    # Evaluate with custom threshold
    val_probs = model.predict(X_val, verbose=0).flatten()
    
    # Debug: Check predictions
    print(f"  Val probs - min: {val_probs.min():.4f}, max: {val_probs.max():.4f}, mean: {val_probs.mean():.4f}")
    print(f"  Val labels - 0s: {(y_val==0).sum()}, 1s: {(y_val==1).sum()}")
    unique_probs = np.unique(np.round(val_probs, 4))
    print(f"  Unique prob values: {len(unique_probs)} - first 5: {unique_probs[:5]}")
    
    # Find optimal threshold
    best_threshold = 0.5
    best_accuracy = 0
    for threshold in np.arange(0.3, 0.7, 0.01):
        preds = (val_probs >= threshold).astype(int)
        accuracy = (preds == y_val).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    cv_scores.append(best_accuracy)
    models.append((model, best_threshold, model_type))
    
    print(f"Model type: {model_type}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Validation accuracy: {best_accuracy:.6f}")

print(f"\nMean CV accuracy: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")

# Train final ensemble on full data
print("\n=== TRAINING FINAL ENSEMBLE ===")

final_models = []

# Train each architecture on full data
architectures = [
    (create_attention_model, "attention"),
    (create_deep_model, "deep"),
    (create_embedding_model, "embedding")
]

for create_func, name in architectures:
    print(f"\nTraining {name} model on full data...")
    
    model = create_func(X_train.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with validation split and class weights
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=0
    )
    
    final_models.append((model, name))
    print(f"Final {name} validation accuracy: {max(history.history['val_accuracy']):.6f}")

# Generate predictions
print("\n=== GENERATING PREDICTIONS ===")

# Ensemble predictions
test_predictions = []
for model, name in final_models:
    probs = model.predict(X_test_scaled, verbose=0).flatten()
    test_predictions.append(probs)
    print(f"{name} prediction mean: {probs.mean():.3f}")

# Average ensemble
ensemble_probs = np.mean(test_predictions, axis=0)

# Apply discovered optimal threshold
final_predictions = (ensemble_probs >= 0.35).astype(int)

# Apply hard rules for certain patterns
for i in range(len(final_predictions)):
    # Missing Drained = Extrovert
    if X_test.iloc[i]['Drained_after_socializing_missing'] == 1:
        final_predictions[i] = 1
    # Very low alone time + not drained = Extrovert
    elif (X_test.iloc[i]['Time_spent_Alone'] <= 1 and 
          X_test.iloc[i]['Drained_after_socializing'] == 0):
        final_predictions[i] = 1
    # Very high alone time + drained = Introvert
    elif (X_test.iloc[i]['Time_spent_Alone'] >= 9 and 
          X_test.iloc[i]['Drained_after_socializing'] == 1):
        final_predictions[i] = 0

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in final_predictions]
})

# Save
from datetime import datetime
date_str = datetime.now().strftime('%Y%m%d')
os.makedirs(f'subm/DATE_{date_str}', exist_ok=True)

submission_path = f'subm/DATE_{date_str}/20250704_2316_neural_network_ensemble.csv'
submission.to_csv(submission_path, index=False)

# Save model performance
results = {
    'cv_scores': [float(s) for s in cv_scores],
    'mean_cv': float(np.mean(cv_scores)),
    'std_cv': float(np.std(cv_scores)),
    'models_trained': len(final_models),
    'prediction_distribution': {
        'introverts': int((final_predictions == 0).sum()),
        'extroverts': int((final_predictions == 1).sum())
    }
}

with open('output/20250704_2316_nn_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== SUMMARY ===")
print(f"Mean CV: {np.mean(cv_scores):.6f}")
print(f"Models trained: {len(final_models)}")
print(f"Prediction distribution: {(final_predictions == 1).sum()}/{len(final_predictions)} extroverts")
print(f"Submission saved to: {submission_path}")

# RESULT: [To be filled after execution]