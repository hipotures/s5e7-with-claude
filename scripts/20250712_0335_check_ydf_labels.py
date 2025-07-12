#!/usr/bin/env python3
"""
Check if YDF correctly interprets Personality labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")

def check_ydf_label_interpretation():
    """Check how YDF interprets the Personality labels"""
    
    print("="*60)
    print("YDF LABEL INTERPRETATION CHECK")
    print("="*60)
    
    # Load original data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Check unique values and their order
    print("\n1. ORIGINAL DATA")
    print(f"Unique Personality values: {sorted(train_df['Personality'].unique())}")
    print(f"Value counts in train:")
    print(train_df['Personality'].value_counts())
    
    # Create a small test dataset
    print("\n2. CONTROLLED TEST")
    
    # Create dataset with known labels
    test_data = pd.DataFrame({
        'Time_spent_Alone': [10, 10, 10, 1, 1, 1],
        'Stage_fear': ['Yes', 'Yes', 'Yes', 'No', 'No', 'No'],
        'Social_event_attendance': [0, 0, 0, 5, 5, 5],
        'Going_outside': [0, 0, 0, 5, 5, 5],
        'Drained_after_socializing': ['Yes', 'Yes', 'Yes', 'No', 'No', 'No'],
        'Friends_circle_size': [1, 1, 1, 50, 50, 50],
        'Post_frequency': [0, 0, 0, 10, 10, 10],
        'Personality': ['Introvert', 'Introvert', 'Introvert', 
                       'Extrovert', 'Extrovert', 'Extrovert']
    })
    
    print("Test data:")
    print(test_data)
    
    # Train model
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=100,
        random_seed=42
    )
    
    model = learner.train(test_data)
    
    # Predict on same data
    predictions = model.predict(test_data)
    
    print("\n3. PREDICTIONS ON TRAINING DATA")
    for i, (idx, row) in enumerate(test_data.iterrows()):
        pred = predictions[i]
        prob = float(str(pred))
        print(f"Row {i}: True={row['Personality']}, Pred_prob={prob:.3f}")
    
    # Check what probability means
    print("\n4. PROBABILITY INTERPRETATION")
    print("If prob > 0.5, YDF predicts the second class alphabetically")
    print("Classes sorted: ['Extrovert', 'Introvert']")
    print("So prob is probability of 'Introvert' (second class)")
    
    # Verify on real data
    print("\n5. VERIFICATION ON REAL DATA")
    small_train = train_df.head(100)
    model_real = learner.train(small_train)
    pred_real = model_real.predict(small_train)
    
    # Count predictions
    pred_classes = []
    for p in pred_real:
        prob = float(str(p))
        # If YDF returns prob of second class (Introvert)
        pred_class = 'Introvert' if prob > 0.5 else 'Extrovert'
        pred_classes.append(pred_class)
    
    # Compare with actual
    accuracy = sum(1 for i in range(len(small_train)) 
                   if pred_classes[i] == small_train.iloc[i]['Personality']) / len(small_train)
    
    print(f"Accuracy on training data: {accuracy:.3f}")
    print(f"Predicted distribution: {pd.Series(pred_classes).value_counts()}")
    print(f"Actual distribution: {small_train['Personality'].value_counts()}")
    
    # Test the opposite interpretation
    pred_classes_opposite = []
    for p in pred_real:
        prob = float(str(p))
        # Opposite: prob is for Extrovert
        pred_class = 'Extrovert' if prob > 0.5 else 'Introvert'
        pred_classes_opposite.append(pred_class)
    
    accuracy_opposite = sum(1 for i in range(len(small_train)) 
                           if pred_classes_opposite[i] == small_train.iloc[i]['Personality']) / len(small_train)
    
    print(f"\nWith opposite interpretation:")
    print(f"Accuracy: {accuracy_opposite:.3f}")
    
    print("\n6. CONCLUSION")
    if accuracy > accuracy_opposite:
        print("YDF probability represents P(Introvert)")
    else:
        print("YDF probability represents P(Extrovert)")

if __name__ == "__main__":
    check_ydf_label_interpretation()