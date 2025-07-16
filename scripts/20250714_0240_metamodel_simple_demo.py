#!/usr/bin/env python3
"""
Simplified MetaModel Demo - Quick implementation to test the concept.
"""

import pandas as pd
import numpy as np
import ydf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class SimpleFlipEvaluator:
    """Simplified flip evaluation using YDF."""
    
    def evaluate_flip_impact(self, train_data, test_idx, n_folds=3):
        """Evaluate impact of flipping a single sample's label."""
        scores = []
        
        # Prepare features
        feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                       'Going_outside', 'Drained_after_socializing', 
                       'Friends_circle_size', 'Post_frequency']
        
        # Create folds
        for fold in range(n_folds):
            # Split data
            train_fold, val_fold = train_test_split(train_data, test_size=0.2, random_state=fold)
            
            # Score without flip
            score_original = self._train_and_score(train_fold, val_fold, feature_cols)
            
            # Apply flip if sample is in training fold
            if test_idx in train_fold.index:
                train_fold_flipped = train_fold.copy()
                current = train_fold_flipped.loc[test_idx, 'Personality']
                new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
                train_fold_flipped.loc[test_idx, 'Personality'] = new_label
                
                score_flipped = self._train_and_score(train_fold_flipped, val_fold, feature_cols)
                improvement = score_flipped - score_original
            else:
                improvement = 0
            
            scores.append(improvement)
        
        return np.mean(scores), np.std(scores)
    
    def _train_and_score(self, train_df, val_df, feature_cols):
        """Train YDF and return validation accuracy."""
        # Prepare data
        train_data = train_df[feature_cols + ['Personality']].copy()
        val_data = val_df[feature_cols + ['Personality']].copy()
        
        # Handle missing values
        train_data[feature_cols] = train_data[feature_cols].fillna(-1)
        val_data[feature_cols] = val_data[feature_cols].fillna(-1)
        
        # Train
        model = ydf.RandomForestLearner(
            label='Personality',
            num_trees=20,  # Very few trees for speed
            max_depth=6
        ).train(train_data)
        
        # Evaluate
        predictions = model.predict(val_data)
        accuracy = (predictions == val_data['Personality']).mean()
        
        return accuracy


def find_flip_candidates(train_data, n_candidates=100):
    """Find potential mislabeled samples using simple heuristics."""
    
    feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                   'Going_outside', 'Drained_after_socializing', 
                   'Friends_circle_size', 'Post_frequency']
    
    # Prepare data
    data = train_data.copy()
    data[feature_cols] = data[feature_cols].fillna(-1)
    
    # Convert categorical
    for col in ['Stage_fear', 'Drained_after_socializing']:
        data[col] = data[col].map({'Yes': 1, 'No': 0, -1: -1})
    
    # Calculate simple uncertainty scores
    scores = []
    
    # 1. KNN disagreement
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[feature_cols])
    
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(scaled_features)
    
    for idx in range(len(data)):
        # Get neighbors
        distances, indices = knn.kneighbors([scaled_features[idx]])
        neighbor_indices = indices[0][1:]  # Exclude self
        
        # Check label agreement
        current_label = data.iloc[idx]['Personality']
        neighbor_labels = data.iloc[neighbor_indices]['Personality'].values
        disagreement = np.mean(neighbor_labels != current_label)
        
        # Null pattern mismatch
        has_nulls = train_data.iloc[idx][feature_cols].isna().any()
        is_introvert = current_label == 'Introvert'
        
        # Introverts typically have nulls, extroverts don't
        null_mismatch = (has_nulls and not is_introvert) or (not has_nulls and is_introvert)
        
        # Combined score
        uncertainty = disagreement + 0.5 * null_mismatch
        
        scores.append({
            'idx': idx,
            'uncertainty': uncertainty,
            'knn_disagreement': disagreement,
            'null_mismatch': null_mismatch,
            'label': current_label
        })
    
    # Sort by uncertainty
    scores.sort(key=lambda x: x['uncertainty'], reverse=True)
    
    return scores[:n_candidates]


def create_flip_submission(train_data, test_data, flip_candidates, evaluator):
    """Create submission with evaluated flips."""
    
    # Evaluate each candidate
    logging.info(f"Evaluating {len(flip_candidates)} flip candidates...")
    
    evaluated_flips = []
    for candidate in tqdm(flip_candidates[:20], desc="Evaluating flips"):  # Limit to 20 for demo
        idx = candidate['idx']
        mean_impact, std_impact = evaluator.evaluate_flip_impact(train_data, idx)
        
        evaluated_flips.append({
            'idx': idx,
            'uncertainty': candidate['uncertainty'],
            'mean_impact': mean_impact,
            'std_impact': std_impact,
            'label': candidate['label'],
            'confidence': mean_impact / (std_impact + 1e-6)
        })
    
    # Sort by impact
    evaluated_flips.sort(key=lambda x: x['mean_impact'], reverse=True)
    
    # Show results
    logging.info("\nTop flip candidates by impact:")
    for i, flip in enumerate(evaluated_flips[:10]):
        logging.info(f"{i+1}. idx={flip['idx']}, label={flip['label']}, "
                    f"impact={flip['mean_impact']:.4f} ± {flip['std_impact']:.4f}")
    
    # Save detailed results
    results_df = pd.DataFrame(evaluated_flips)
    results_df.to_csv('output/metamodel_simple_flip_evaluation.csv', index=False)
    
    # Find similar patterns in test set
    logging.info("\nFinding similar samples in test set...")
    
    # Get top positive impact flips
    good_flips = [f for f in evaluated_flips if f['mean_impact'] > 0.01]
    
    if good_flips:
        logging.info(f"Found {len(good_flips)} flips with positive impact")
        
        # For demo, create a simple submission
        # In practice, you would use the trained metamodel to find similar patterns in test
        test_flip_ids = []
        
        # Example: flip some test samples based on similarity
        # This is simplified - real implementation would use the neural network
        for flip in good_flips[:5]:
            # Find test samples with similar characteristics
            train_sample = train_data.iloc[flip['idx']]
            
            # Simple heuristic: find test samples with same null pattern
            has_nulls = train_sample[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                                     'Going_outside', 'Drained_after_socializing', 
                                     'Friends_circle_size', 'Post_frequency']].isna().any()
            
            if has_nulls:
                # Find test samples with nulls
                test_with_nulls = test_data[test_data.isna().any(axis=1)]
                if len(test_with_nulls) > 0:
                    # Pick a random one for demo
                    test_idx = test_with_nulls.sample(1, random_state=42).index[0]
                    test_flip_ids.append(test_data.iloc[test_idx]['id'])
        
        logging.info(f"Selected {len(test_flip_ids)} test samples for flipping: {test_flip_ids}")
        
        # Create submission
        if test_flip_ids:
            # Load baseline predictions
            baseline = pd.read_csv('../sample_submission.csv')
            
            # Apply flips
            submission = baseline.copy()
            for test_id in test_flip_ids:
                idx = submission[submission['id'] == test_id].index[0]
                current = submission.loc[idx, 'Personality']
                new_label = 'Introvert' if current == 'Extrovert' else 'Extrovert'
                submission.loc[idx, 'Personality'] = new_label
                logging.info(f"Flipped ID {test_id}: {current} → {new_label}")
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_file = f'../scores/metamodel_demo_flips_{timestamp}.csv'
            submission.to_csv(submission_file, index=False)
            logging.info(f"Saved submission to {submission_file}")
    else:
        logging.info("No high-impact flips found")


def main():
    """Run simplified metamodel demo."""
    # Load data
    logging.info("Loading data...")
    train_df = pd.read_csv('../../train.csv')
    test_df = pd.read_csv('../../test.csv')
    
    # Find candidates
    logging.info("Finding flip candidates...")
    candidates = find_flip_candidates(train_df, n_candidates=50)
    
    # Initialize evaluator
    evaluator = SimpleFlipEvaluator()
    
    # Create submission
    create_flip_submission(train_df, test_df, candidates, evaluator)
    
    logging.info("\nDemo complete!")
    logging.info("This simplified version demonstrates the concept.")
    logging.info("The full implementation would:")
    logging.info("1. Train a neural network on flip evaluation results")
    logging.info("2. Use the network to predict flip quality for all test samples")
    logging.info("3. Create submissions based on top predictions")


if __name__ == "__main__":
    main()