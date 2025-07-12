#!/usr/bin/env python3
"""
Ensemble approach using models trained on different feature perspectives
Each model focuses on different aspect of personality
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
OUTPUT_DIR.mkdir(exist_ok=True)

def prepare_data():
    """Load and prepare data"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df

def create_feature_sets():
    """Define different feature perspectives"""
    feature_sets = {
        # Social interaction features
        'social': [
            'Social_event_attendance',
            'Friends_circle_size',
            'Going_outside',
            'Post_frequency'
        ],
        
        # Personal preference features
        'personal': [
            'Time_spent_Alone',
            'Stage_fear',
            'Drained_after_socializing'
        ],
        
        # Behavioral features (numeric only)
        'behavioral': [
            'Time_spent_Alone',
            'Social_event_attendance',
            'Going_outside',
            'Post_frequency'
        ],
        
        # Binary features only
        'binary': [
            'Stage_fear',
            'Drained_after_socializing'
        ],
        
        # External features
        'external': [
            'Social_event_attendance',
            'Friends_circle_size',
            'Post_frequency'
        ],
        
        # All features (baseline)
        'all': [
            'Time_spent_Alone',
            'Social_event_attendance', 
            'Friends_circle_size',
            'Going_outside',
            'Post_frequency',
            'Stage_fear',
            'Drained_after_socializing'
        ]
    }
    
    return feature_sets

def train_perspective_model(train_df, test_df, features, perspective_name):
    """Train model on specific feature perspective"""
    print(f"\nTraining {perspective_name} model with features: {features}")
    
    # Prepare training data
    train_data = train_df[features + ['Personality']].copy()
    test_data = test_df[features].copy()
    
    # Train model
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        compute_oob_performances=True,
        random_seed=42
    )
    
    model = learner.train(train_data)
    
    # Get OOB score
    oob_accuracy = None
    if hasattr(model, 'out_of_bag_evaluation'):
        oob_eval = model.out_of_bag_evaluation()
        if oob_eval and hasattr(oob_eval, 'accuracy'):
            oob_accuracy = oob_eval.accuracy
            print(f"  OOB Accuracy: {oob_accuracy:.6f}")
    
    # Make predictions
    predictions = model.predict(test_data)
    
    # Get probabilities and convert to class labels
    # YDF returns probabilities for Extrovert class
    pred_classes = []
    probabilities = []
    
    for pred in predictions:
        # YDF returns probability values
        prob = float(str(pred))
        probabilities.append(prob)
        
        # Convert to class label
        # YDF returns P(Introvert), not P(Extrovert)!
        pred_class = 'Introvert' if prob > 0.5 else 'Extrovert'
        pred_classes.append(pred_class)
    
    return pred_classes, np.array(probabilities), oob_accuracy

def ensemble_predictions(all_predictions, all_probabilities, weights=None):
    """Combine predictions from different models"""
    
    if weights is None:
        # Equal weights
        weights = np.ones(len(all_predictions)) / len(all_predictions)
    
    # Weighted average of probabilities
    ensemble_proba = np.zeros(len(all_probabilities[0]))
    
    for i, (proba, weight) in enumerate(zip(all_probabilities, weights)):
        ensemble_proba += proba * weight
    
    # Convert to predictions
    # Since YDF returns P(Introvert), ensemble_proba is also P(Introvert)
    ensemble_pred = ['Introvert' if p > 0.5 else 'Extrovert' for p in ensemble_proba]
    
    return ensemble_pred, ensemble_proba

def analyze_disagreements(all_predictions, test_df):
    """Analyze where models disagree"""
    n_models = len(all_predictions)
    n_samples = len(all_predictions[0])
    
    # Count agreements
    agreement_counts = []
    for i in range(n_samples):
        votes = [pred[i] for pred in all_predictions]
        n_extrovert = sum(1 for v in votes if v == 'Extrovert')
        agreement_counts.append(n_extrovert)
    
    # Find cases with maximum disagreement
    max_disagreement_indices = []
    for i, count in enumerate(agreement_counts):
        if count == n_models // 2:  # Split vote
            max_disagreement_indices.append(i)
    
    print(f"\n{'='*60}")
    print("DISAGREEMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"Total samples: {n_samples}")
    print(f"Complete agreement: {sum(1 for c in agreement_counts if c == 0 or c == n_models)}")
    print(f"Split votes: {len(max_disagreement_indices)}")
    
    # Save disagreement cases
    if max_disagreement_indices:
        disagreement_df = test_df.iloc[max_disagreement_indices].copy()
        disagreement_df['agreement_count'] = [agreement_counts[i] for i in max_disagreement_indices]
        disagreement_df.to_csv(OUTPUT_DIR / 'ensemble_disagreements.csv', index=False)
        print(f"\nSaved {len(disagreement_df)} maximum disagreement cases")
        print("Sample IDs with split votes:", disagreement_df['id'].head(10).tolist())
    
    return agreement_counts

def main():
    # Load data
    train_df, test_df = prepare_data()
    
    # Get feature sets
    feature_sets = create_feature_sets()
    
    # Train models on different perspectives
    all_predictions = []
    all_probabilities = []
    oob_scores = []
    perspective_names = []
    
    print(f"\n{'='*60}")
    print("TRAINING PERSPECTIVE MODELS")
    print(f"{'='*60}")
    
    for perspective, features in feature_sets.items():
        if perspective == 'all':  # Skip baseline for now
            continue
            
        try:
            predictions, probabilities, oob_score = train_perspective_model(
                train_df, test_df, features, perspective
            )
            
            all_predictions.append(predictions)
            all_probabilities.append(probabilities)
            oob_scores.append(oob_score)
            perspective_names.append(perspective)
            
        except Exception as e:
            print(f"  Error training {perspective} model: {e}")
    
    # Analyze disagreements
    agreement_counts = analyze_disagreements(all_predictions, test_df)
    
    # Create ensembles with different strategies
    print(f"\n{'='*60}")
    print("ENSEMBLE STRATEGIES")
    print(f"{'='*60}")
    
    # 1. Equal weight ensemble
    print("\n1. Equal Weight Ensemble")
    equal_pred, equal_proba = ensemble_predictions(all_predictions, all_probabilities)
    
    # 2. OOB-weighted ensemble
    print("\n2. OOB-Weighted Ensemble")
    # Normalize OOB scores as weights
    valid_oob = [s for s in oob_scores if isinstance(s, (int, float))]
    if valid_oob:
        oob_weights = np.array(valid_oob)
        oob_weights = oob_weights / oob_weights.sum()
        weighted_pred, weighted_proba = ensemble_predictions(
            all_predictions[:len(valid_oob)], 
            all_probabilities[:len(valid_oob)], 
            oob_weights
        )
    
    # 3. Majority voting
    print("\n3. Majority Voting")
    majority_pred = []
    for i in range(len(test_df)):
        votes = [pred[i] for pred in all_predictions]
        extrovert_votes = sum(1 for v in votes if v == 'Extrovert')
        majority_pred.append('Extrovert' if extrovert_votes > len(votes)/2 else 'Introvert')
    
    # 4. Conservative ensemble (high confidence only)
    print("\n4. Conservative Ensemble (high confidence)")
    conservative_pred = []
    confidence_threshold = 0.7
    for i in range(len(test_df)):
        avg_proba = np.mean([p[i] for p in all_probabilities])
        if avg_proba > confidence_threshold:
            conservative_pred.append('Extrovert')
        elif avg_proba < (1 - confidence_threshold):
            conservative_pred.append('Introvert')
        else:
            # Use majority vote for uncertain cases
            votes = [pred[i] for pred in all_predictions]
            extrovert_votes = sum(1 for v in votes if v == 'Extrovert')
            conservative_pred.append('Extrovert' if extrovert_votes > len(votes)/2 else 'Introvert')
    
    # Save all predictions
    results_df = pd.DataFrame({
        'id': test_df['id'],
        'equal_weight': equal_pred,
        'equal_proba': equal_proba,
        'majority_vote': majority_pred,
        'conservative': conservative_pred
    })
    
    if 'weighted_pred' in locals():
        results_df['oob_weighted'] = weighted_pred
        results_df['oob_weighted_proba'] = weighted_proba
    
    # Add individual model predictions and probabilities
    for name, pred, prob in zip(perspective_names, all_predictions, all_probabilities):
        results_df[f'model_{name}'] = pred
        results_df[f'model_{name}_proba'] = prob
    
    results_df.to_csv(OUTPUT_DIR / 'ensemble_predictions.csv', index=False)
    
    # Create submission files
    print(f"\n{'='*60}")
    print("CREATING SUBMISSIONS")
    print(f"{'='*60}")
    
    # Best ensemble submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': equal_pred  # or choose best strategy
    })
    submission.to_csv(OUTPUT_DIR / 'submission_ensemble_equal.csv', index=False)
    print(f"\nCreated submission with {len(submission)} predictions")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Models trained: {len(all_predictions)}")
    print(f"Perspectives used: {', '.join(perspective_names)}")
    print(f"\nPrediction differences between strategies:")
    
    # Compare strategies
    n_diff_majority_equal = sum(1 for i in range(len(test_df)) 
                                if majority_pred[i] != equal_pred[i])
    print(f"  Majority vs Equal: {n_diff_majority_equal} differences")
    
    n_diff_conservative_equal = sum(1 for i in range(len(test_df)) 
                                    if conservative_pred[i] != equal_pred[i])
    print(f"  Conservative vs Equal: {n_diff_conservative_equal} differences")
    
    # Compare individual models pairwise
    print(f"\nPairwise model differences:")
    for i in range(len(all_predictions)):
        for j in range(i+1, len(all_predictions)):
            n_diff = sum(1 for k in range(len(test_df)) 
                        if all_predictions[i][k] != all_predictions[j][k])
            if n_diff > 0:
                print(f"  {perspective_names[i]} vs {perspective_names[j]}: {n_diff} differences")
                # Show first few different IDs
                diff_ids = [test_df.iloc[k]['id'] for k in range(len(test_df)) 
                           if all_predictions[i][k] != all_predictions[j][k]][:5]
                print(f"    Sample different IDs: {diff_ids}")

if __name__ == "__main__":
    main()