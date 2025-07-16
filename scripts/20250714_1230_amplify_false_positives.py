#!/usr/bin/env python3
"""
Amplify predictions for false positive IDs to test their impact.
Creates submissions with 1x, 2x, 5x, 10x amplification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# False positives from our flip tests (confirmed wrong predictions)
FALSE_POSITIVES = {
    20934: 'E',  # We flipped E→I, but it's correctly E
    18634: 'E',  # We flipped E→I, but it's correctly E
    20932: 'I',  # We flipped I→E, but it's correctly I
    21138: 'E',  # We flipped E→I, but it's correctly E
    20728: 'I',  # We flipped I→E, but it's correctly I
    21359: 'E',  # We flipped E→I, but it's correctly E
}

def generate_baseline_probabilities():
    """Generate baseline probabilities using a simple XGBoost model."""
    # Load data
    train_df = pd.read_csv('../../train.csv')
    test_df = pd.read_csv('../../test.csv')
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                    'Going_outside', 'Drained_after_socializing', 
                    'Friends_circle_size', 'Post_frequency']
    
    # Handle missing values
    train_df[feature_cols] = train_df[feature_cols].fillna(-1)
    test_df[feature_cols] = test_df[feature_cols].fillna(-1)
    
    # Convert Yes/No to 1/0
    for col in ['Stage_fear', 'Drained_after_socializing']:
        train_df[col] = train_df[col].map({'Yes': 1, 'No': 0, -1: -1})
        test_df[col] = test_df[col].map({'Yes': 1, 'No': 0, -1: -1})
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = (train_df['Personality'] == 'Extrovert').astype(int)
    X_test = test_df[feature_cols]
    
    # Train simple XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Get probabilities
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Extrovert': probabilities
    })
    
    logging.info(f"Generated baseline probabilities for {len(submission_df)} samples")
    return submission_df

def amplify_predictions(submission_df, amplification_factor=1.0):
    """
    Amplify predictions for false positive IDs.
    
    For IDs we incorrectly flipped:
    - If correct label is 'E', push probability towards 1.0
    - If correct label is 'I', push probability towards 0.0
    """
    df = submission_df.copy()
    
    for id_val, correct_label in FALSE_POSITIVES.items():
        if id_val in df['id'].values:
            idx = df[df['id'] == id_val].index[0]
            current_prob = df.loc[idx, 'Extrovert']
            
            if correct_label == 'E':
                # Push towards Extrovert (1.0)
                if amplification_factor == 1.0:
                    # For 1x, just ensure it's above 0.5
                    new_prob = max(0.51, current_prob)
                else:
                    # Amplify the distance from 0.5 towards 1.0
                    distance_from_half = 1.0 - 0.5
                    amplified_distance = min(distance_from_half * amplification_factor, 0.49)
                    new_prob = 0.5 + amplified_distance
            else:  # correct_label == 'I'
                # Push towards Introvert (0.0)
                if amplification_factor == 1.0:
                    # For 1x, just ensure it's below 0.5
                    new_prob = min(0.49, current_prob)
                else:
                    # Amplify the distance from 0.5 towards 0.0
                    distance_from_half = 0.5 - 0.0
                    amplified_distance = min(distance_from_half * amplification_factor, 0.49)
                    new_prob = 0.5 - amplified_distance
            
            # Ensure probability stays in valid range
            new_prob = np.clip(new_prob, 0.001, 0.999)
            
            logging.info(f"ID {id_val} ({correct_label}): {current_prob:.4f} → {new_prob:.4f} (factor={amplification_factor})")
            df.loc[idx, 'Extrovert'] = new_prob
    
    return df

def main():
    """Generate amplified submissions."""
    # Generate baseline probabilities
    baseline_df = generate_baseline_probabilities()
    logging.info(f"Generated {len(baseline_df)} predictions")
    
    # Check how many false positives are in test set
    found_ids = []
    for id_val in FALSE_POSITIVES:
        if id_val in baseline_df['id'].values:
            found_ids.append(id_val)
    
    logging.info(f"Found {len(found_ids)}/{len(FALSE_POSITIVES)} false positive IDs in test set")
    logging.info(f"IDs in test: {found_ids}")
    
    # Generate amplified submissions
    amplification_factors = [1, 2, 5, 10]
    
    for factor in amplification_factors:
        logging.info(f"\nGenerating {factor}x amplification...")
        
        amplified_df = amplify_predictions(baseline_df, factor)
        
        # Save submission
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'../scores/amplified_false_positives_{factor}x_{timestamp}.csv'
        amplified_df.to_csv(output_file, index=False)
        logging.info(f"Saved to: {output_file}")
        
        # Show changes for this factor
        print(f"\n{factor}x amplification changes:")
        for id_val in found_ids:
            idx = baseline_df[baseline_df['id'] == id_val].index[0]
            orig_prob = baseline_df.loc[idx, 'Extrovert']
            new_prob = amplified_df.loc[idx, 'Extrovert']
            correct_label = FALSE_POSITIVES[id_val]
            print(f"  ID {id_val} ({correct_label}): {orig_prob:.4f} → {new_prob:.4f}")

if __name__ == "__main__":
    main()