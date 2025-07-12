#!/usr/bin/env python3
"""
Check the 15 Very Low Confidence cases (near 50-50 predictions)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def check_very_low_confidence():
    """Analyze the 15 very low confidence cases"""
    
    print("="*60)
    print("ANALIZA 15 PRZYPADKÓW VERY LOW CONFIDENCE (<10%)")
    print("="*60)
    
    # Load train data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Train model to get predictions
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    print("\nTraining model to identify uncertain cases...")
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        random_seed=42
    )
    
    model = learner.train(train_df[feature_cols + ['Personality']])
    
    # Get predictions on train
    predictions = model.predict(train_df[feature_cols])
    
    probabilities = []
    pred_classes = []
    
    for pred in predictions:
        prob_I = float(str(pred))
        probabilities.append(prob_I)
        pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
    
    train_df['probability'] = probabilities
    train_df['confidence'] = np.abs(np.array(probabilities) - 0.5) * 2
    train_df['predicted'] = pred_classes
    train_df['is_correct'] = train_df['predicted'] == train_df['Personality']
    
    # Find very low confidence cases
    very_low_conf = train_df[train_df['confidence'] < 0.1].copy()
    very_low_conf = very_low_conf.sort_values('confidence')
    
    print(f"\nZnaleziono {len(very_low_conf)} przypadków z confidence < 10%")
    print("\nSzczegóły tych przypadków:")
    print("-" * 100)
    
    for idx, row in very_low_conf.iterrows():
        print(f"\nID: {row['id']}")
        print(f"  P(Introvert): {row['probability']:.4f} → Confidence: {row['confidence']*100:.1f}%")
        print(f"  Actual: {row['Personality']}, Predicted: {row['predicted']} ({'✓' if row['is_correct'] else '✗'})")
        print(f"  Features: Time_alone={row['Time_spent_Alone']}, Social={row['Social_event_attendance']}, "
              f"Friends={row['Friends_circle_size']}, Outside={row['Going_outside']}, "
              f"Posts={row['Post_frequency']}")
        print(f"  Stage_fear={row['Stage_fear']}, Drained={row['Drained_after_socializing']}")
    
    # Get predictions on test set
    print("\n" + "="*60)
    print("SPRAWDZANIE CZY TE ID SĄ W TEST SET")
    print("="*60)
    
    test_predictions = model.predict(test_df[feature_cols])
    test_probs = []
    test_preds = []
    
    for pred in test_predictions:
        prob_I = float(str(pred))
        test_probs.append(prob_I)
        test_preds.append('Introvert' if prob_I > 0.5 else 'Extrovert')
    
    test_df['probability'] = test_probs
    test_df['confidence'] = np.abs(np.array(test_probs) - 0.5) * 2
    test_df['predicted'] = test_preds
    
    # Find test cases with very low confidence
    test_very_low = test_df[test_df['confidence'] < 0.1].copy()
    test_very_low = test_very_low.sort_values('confidence')
    
    print(f"\nW test set znaleziono {len(test_very_low)} przypadków z confidence < 10%")
    
    if len(test_very_low) > 0:
        print("\nNajbardziej niepewne przypadki w test set:")
        for idx, row in test_very_low.head(10).iterrows():
            print(f"\nTest ID: {row['id']}")
            print(f"  P(Introvert): {row['probability']:.4f} → Confidence: {row['confidence']*100:.1f}%")
            print(f"  Predicted: {row['predicted']}")
    
    # Check if any of these IDs were in our flip tests
    print("\n" + "="*60)
    print("SPRAWDZANIE CZY FLIPOWALIŚMY TE ID")
    print("="*60)
    
    # List all flip test files
    flip_files = list(SCORES_DIR.glob("flip_*.csv"))
    print(f"\nZnaleziono {len(flip_files)} plików flip test")
    
    # Extract flipped IDs from filenames
    flipped_ids = []
    for flip_file in flip_files:
        # Format: flip_STRATEGY_N_DIRECTION_id_ID.csv
        parts = flip_file.stem.split('_')
        if len(parts) >= 6 and parts[-2] == 'id':
            flipped_id = int(parts[-1])
            flipped_ids.append(flipped_id)
    
    print(f"\nWszystkie flipowane ID: {sorted(flipped_ids)}")
    
    # Check overlap
    test_very_low_ids = set(test_very_low['id'].values)
    flipped_test_ids = set(flipped_ids)
    
    overlap = test_very_low_ids.intersection(flipped_test_ids)
    
    if overlap:
        print(f"\n⭐ ZNALEZIONO {len(overlap)} WSPÓLNYCH ID!")
        print(f"ID które są bardzo niepewne i były flipowane: {sorted(overlap)}")
        
        # Show details
        for test_id in sorted(overlap):
            row = test_very_low[test_very_low['id'] == test_id].iloc[0]
            print(f"\nID {test_id}:")
            print(f"  Confidence: {row['confidence']*100:.1f}%")
            print(f"  P(Introvert): {row['probability']:.4f}")
            print(f"  Original prediction: {row['predicted']}")
            
            # Find which flip file
            for flip_file in flip_files:
                if f"_id_{test_id}.csv" in str(flip_file):
                    print(f"  Flip file: {flip_file.name}")
    else:
        print("\nBrak wspólnych ID między bardzo niepewnymi przypadkami a flipowanymi ID")
    
    # Save very low confidence cases
    output_file = OUTPUT_DIR / 'very_low_confidence_cases.csv'
    
    # Combine train and test very low confidence
    all_very_low = pd.concat([
        very_low_conf[['id', 'probability', 'confidence', 'Personality', 'predicted', 'is_correct']].assign(dataset='train'),
        test_very_low[['id', 'probability', 'confidence', 'predicted']].assign(dataset='test', Personality=None, is_correct=None)
    ])
    
    all_very_low.to_csv(output_file, index=False)
    print(f"\nZapisano wszystkie przypadki very low confidence do: {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("PODSUMOWANIE:")
    print("="*60)
    print(f"- Train: {len(very_low_conf)} przypadków z confidence < 10%")
    print(f"- Test: {len(test_very_low)} przypadków z confidence < 10%")
    print(f"- Średnia dokładność na train very low: {very_low_conf['is_correct'].mean()*100:.1f}%")
    print(f"- Flipowane ID które są very low confidence: {len(overlap)}")
    
    if len(test_very_low) > 0:
        print(f"\n💡 REKOMENDACJA: Te {len(test_very_low)} przypadki z test set są idealnymi kandydatami do flipowania!")
        print("Są to prawdopodobnie prawdziwi ambiwerci lub przypadki graniczne.")

if __name__ == "__main__":
    check_very_low_confidence()