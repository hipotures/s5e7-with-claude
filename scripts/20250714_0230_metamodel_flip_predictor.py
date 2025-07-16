#!/usr/bin/env python3
"""
MetaModel Flip Predictor - Neural network trained to predict good flips
using fast YDF evaluations on subsets.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import ydf
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FlipFeatureExtractor:
    """Extract features that indicate potential mislabeling."""
    
    def __init__(self, train_data, n_neighbors=10):
        self.train_data = train_data
        self.n_neighbors = n_neighbors
        self.feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                            'Going_outside', 'Drained_after_socializing', 
                            'Friends_circle_size', 'Post_frequency']
        
        # Prepare data for KNN
        self._prepare_knn_data()
        
    def _prepare_knn_data(self):
        """Prepare data for KNN analysis."""
        # Handle missing values and convert categorical
        data = self.train_data.copy()
        
        # Fill missing values
        data[self.feature_cols] = data[self.feature_cols].fillna(-1)
        
        # Convert Yes/No to 1/0
        for col in ['Stage_fear', 'Drained_after_socializing']:
            data[col] = data[col].map({'Yes': 1, 'No': 0, -1: -1})
        
        # Scale features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(data[self.feature_cols])
        self.labels = (data['Personality'] == 'Extrovert').astype(int).values
        
        # Fit KNN
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        self.knn.fit(self.scaled_features)
    
    def extract_features(self, sample_idx, predictions_dict=None):
        """Extract features for a single sample."""
        sample = self.train_data.iloc[sample_idx]
        
        features = {}
        
        # 1. Basic sample characteristics
        features['null_count'] = sample[self.feature_cols].isna().sum()
        
        # 2. Feature extremity (how far from mean)
        sample_scaled = self.scaled_features[sample_idx]
        features['feature_extremity'] = np.mean(np.abs(sample_scaled))
        features['max_feature_deviation'] = np.max(np.abs(sample_scaled))
        
        # 3. KNN-based features
        distances, indices = self.knn.kneighbors([sample_scaled])
        neighbor_indices = indices[0][1:]  # Exclude self
        neighbor_labels = self.labels[neighbor_indices]
        
        features['knn_disagreement'] = np.mean(neighbor_labels != self.labels[sample_idx])
        features['knn_entropy'] = entropy([sum(neighbor_labels == 0), sum(neighbor_labels == 1)])
        features['mean_knn_distance'] = np.mean(distances[0][1:])
        
        # 4. Class-specific features
        current_label = self.labels[sample_idx]
        features['is_extrovert'] = float(current_label)
        
        # 5. Null pattern features
        has_nulls = features['null_count'] > 0
        features['null_introvert_pattern'] = float(has_nulls and current_label == 0)
        features['null_extrovert_pattern'] = float(has_nulls and current_label == 1)
        
        # 6. Model uncertainty features (if predictions provided)
        if predictions_dict:
            preds = [predictions_dict[model][sample_idx] for model in predictions_dict]
            features['prediction_std'] = np.std(preds)
            features['prediction_entropy'] = entropy([np.mean(preds), 1 - np.mean(preds)])
            features['prediction_margin'] = abs(np.mean(preds) - 0.5)
        
        return list(features.values())
    
    def get_feature_names(self):
        """Return feature names for interpretability."""
        return [
            'null_count', 'feature_extremity', 'max_feature_deviation',
            'knn_disagreement', 'knn_entropy', 'mean_knn_distance',
            'is_extrovert', 'null_introvert_pattern', 'null_extrovert_pattern',
            'prediction_std', 'prediction_entropy', 'prediction_margin'
        ]


class YDFSubsetEvaluator:
    """Fast evaluation using YDF on data subsets."""
    
    def __init__(self, n_subsets=5, subset_size=0.8, n_trees=30):
        self.n_subsets = n_subsets
        self.subset_size = subset_size
        self.n_trees = n_trees
        
    def evaluate_flip(self, train_data, flip_idx, flip_direction):
        """Evaluate a single flip across multiple subsets."""
        scores = []
        
        # Prepare data
        feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                       'Going_outside', 'Drained_after_socializing', 
                       'Friends_circle_size', 'Post_frequency']
        
        for i in range(self.n_subsets):
            # Create train/validation split
            n_samples = len(train_data)
            n_train = int(n_samples * self.subset_size)
            
            # Random indices excluding flip sample
            all_indices = list(range(n_samples))
            all_indices.remove(flip_idx)
            np.random.shuffle(all_indices)
            
            train_indices = all_indices[:n_train]
            val_indices = all_indices[n_train:]
            
            # Prepare datasets
            train_subset = train_data.iloc[train_indices].copy()
            val_subset = train_data.iloc[val_indices].copy()
            
            # Train on original data
            score_original = self._train_and_evaluate(train_subset, val_subset, feature_cols)
            
            # Apply flip
            train_subset_flipped = train_subset.copy()
            if flip_idx in train_indices:
                idx_in_subset = train_indices.index(flip_idx)
                current_label = train_subset_flipped.iloc[idx_in_subset]['Personality']
                new_label = 'Introvert' if current_label == 'Extrovert' else 'Extrovert'
                train_subset_flipped.iloc[idx_in_subset, train_subset_flipped.columns.get_loc('Personality')] = new_label
            
            # Train with flip
            score_flipped = self._train_and_evaluate(train_subset_flipped, val_subset, feature_cols)
            
            # Calculate improvement
            improvement = score_flipped - score_original
            scores.append(improvement)
        
        return {
            'mean_improvement': np.mean(scores),
            'std_improvement': np.std(scores),
            'positive_rate': sum(s > 0 for s in scores) / len(scores)
        }
    
    def _train_and_evaluate(self, train_data, val_data, feature_cols):
        """Train YDF and evaluate on validation set."""
        # Prepare data
        train_df = train_data[feature_cols + ['Personality']].copy()
        val_df = val_data[feature_cols + ['Personality']].copy()
        
        # Handle missing values
        train_df[feature_cols] = train_df[feature_cols].fillna(-1)
        val_df[feature_cols] = val_df[feature_cols].fillna(-1)
        
        # Train model
        model = ydf.RandomForestLearner(
            label='Personality',
            num_trees=self.n_trees,
            max_depth=8
        ).train(train_df)
        
        # Evaluate
        predictions = model.predict(val_df)
        accuracy = (predictions == val_df['Personality']).mean()
        
        return accuracy


class FlipDataset(Dataset):
    """PyTorch dataset for flip training samples."""
    
    def __init__(self, samples):
        self.features = torch.FloatTensor([s['features'] for s in samples])
        self.labels = torch.FloatTensor([s['label'] for s in samples])
        self.improvements = torch.FloatTensor([s['improvement'] for s in samples])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.improvements[idx]


class FlipPredictorNN(nn.Module):
    """Neural network to predict flip quality."""
    
    def __init__(self, input_dim=12, hidden_dims=[64, 32, 16]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output: probability of good flip
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()


class MetaModelTrainer:
    """Train the metamodel using YDF evaluations."""
    
    def __init__(self, train_data):
        self.train_data = train_data
        self.feature_extractor = FlipFeatureExtractor(train_data)
        self.ydf_evaluator = YDFSubsetEvaluator()
        self.neural_model = FlipPredictorNN()
        
    def generate_training_data(self, n_candidates=500, use_parallel=True):
        """Generate flip candidates and evaluate them."""
        logging.info(f"Generating training data for {n_candidates} candidates...")
        
        # Select diverse candidates
        candidates = self._select_candidates(n_candidates)
        
        if use_parallel:
            # Parallel evaluation
            training_samples = self._parallel_evaluate(candidates)
        else:
            # Sequential evaluation
            training_samples = []
            for idx in tqdm(candidates, desc="Evaluating flips"):
                sample = self._evaluate_single_flip(idx)
                if sample:
                    training_samples.append(sample)
        
        logging.info(f"Generated {len(training_samples)} training samples")
        return training_samples
    
    def _parallel_evaluate(self, candidates, n_workers=4):
        """Evaluate flips in parallel."""
        training_samples = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit tasks
            future_to_idx = {
                executor.submit(self._evaluate_single_flip, idx): idx 
                for idx in candidates
            }
            
            # Collect results
            for future in tqdm(as_completed(future_to_idx), total=len(candidates), desc="Evaluating flips"):
                try:
                    sample = future.result()
                    if sample:
                        training_samples.append(sample)
                except Exception as e:
                    logging.warning(f"Error evaluating flip: {e}")
        
        return training_samples
    
    def _evaluate_single_flip(self, idx):
        """Evaluate a single flip candidate."""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(idx)
            
            # Get current label
            current_label = self.train_data.iloc[idx]['Personality']
            flip_direction = 'E2I' if current_label == 'Extrovert' else 'I2E'
            
            # Evaluate flip
            flip_result = self.ydf_evaluator.evaluate_flip(
                self.train_data, idx, flip_direction
            )
            
            # Create training sample
            # Label: 1 if good flip (improves accuracy)
            label = 1 if flip_result['mean_improvement'] > 0.001 else 0
            
            return {
                'features': features,
                'label': label,
                'improvement': flip_result['mean_improvement'],
                'idx': idx,
                'flip_direction': flip_direction
            }
        except Exception as e:
            logging.warning(f"Error processing index {idx}: {e}")
            return None
    
    def _select_candidates(self, n_candidates):
        """Select diverse candidates for evaluation."""
        n_samples = len(self.train_data)
        
        # Strategy: Mix of random and targeted selection
        candidates = []
        
        # 1. Random samples (50%)
        n_random = n_candidates // 2
        random_indices = np.random.choice(n_samples, n_random, replace=False)
        candidates.extend(random_indices)
        
        # 2. High uncertainty samples (25%)
        # Based on KNN disagreement
        knn_disagreements = []
        for i in range(n_samples):
            features = self.feature_extractor.extract_features(i)
            knn_disagreements.append((i, features[3]))  # knn_disagreement is at index 3
        
        knn_disagreements.sort(key=lambda x: x[1], reverse=True)
        n_uncertain = n_candidates // 4
        candidates.extend([idx for idx, _ in knn_disagreements[:n_uncertain]])
        
        # 3. Samples with null values (25%)
        null_indices = self.train_data[self.train_data.isnull().any(axis=1)].index.tolist()
        if null_indices:
            n_null = n_candidates - len(candidates)
            null_sample = np.random.choice(null_indices, min(n_null, len(null_indices)), replace=False)
            candidates.extend(null_sample)
        
        # Remove duplicates and limit to n_candidates
        candidates = list(set(candidates))[:n_candidates]
        
        return candidates
    
    def train(self, training_samples=None, epochs=50, batch_size=32):
        """Train the neural metamodel."""
        if training_samples is None:
            training_samples = self.generate_training_data()
        
        # Create dataset
        dataset = FlipDataset(training_samples)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training loop
        self.neural_model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for features, labels, improvements in train_loader:
                # Forward pass
                predictions = self.neural_model(features)
                loss = criterion(predictions, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                logging.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.neural_model
    
    def save_model(self, path):
        """Save trained model and components."""
        torch.save({
            'model_state': self.neural_model.state_dict(),
            'feature_extractor': self.feature_extractor,
            'model_config': {
                'input_dim': 12,
                'hidden_dims': [64, 32, 16]
            }
        }, path)
        logging.info(f"Model saved to {path}")


class FlipPredictor:
    """Use trained metamodel to predict good flips."""
    
    def __init__(self, model_path, train_data):
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        self.metamodel = FlipPredictorNN(
            input_dim=checkpoint['model_config']['input_dim'],
            hidden_dims=checkpoint['model_config']['hidden_dims']
        )
        self.metamodel.load_state_dict(checkpoint['model_state'])
        self.metamodel.eval()
        
        self.feature_extractor = checkpoint['feature_extractor']
        self.train_data = train_data
        
    def predict_flips(self, test_data, top_k=50):
        """Predict top K flip candidates in test data."""
        flip_scores = []
        
        # For test data, we need to adapt the feature extractor
        test_extractor = FlipFeatureExtractor(test_data)
        
        for idx in tqdm(range(len(test_data)), desc="Scoring flip candidates"):
            # Extract features
            features = test_extractor.extract_features(idx, predictions_dict=None)
            
            # Predict flip quality
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                flip_score = self.metamodel(features_tensor).item()
            
            flip_scores.append({
                'id': test_data.iloc[idx]['id'],
                'flip_score': flip_score,
                'idx': idx
            })
        
        # Sort by score
        flip_scores.sort(key=lambda x: x['flip_score'], reverse=True)
        
        return flip_scores[:top_k]


def main():
    """Main training pipeline."""
    # Load data
    logging.info("Loading data...")
    train_df = pd.read_csv('../../train.csv')
    test_df = pd.read_csv('../../test.csv')
    
    # Initialize trainer
    trainer = MetaModelTrainer(train_df)
    
    # Generate training data
    training_samples = trainer.generate_training_data(n_candidates=300, use_parallel=True)
    
    # Save training data for analysis
    training_df = pd.DataFrame(training_samples)
    training_df.to_csv('output/metamodel_training_data.csv', index=False)
    logging.info(f"Saved {len(training_df)} training samples")
    
    # Train model
    logging.info("Training neural metamodel...")
    trained_model = trainer.train(training_samples, epochs=30)
    
    # Save model
    model_path = 'output/metamodel_flip_predictor.pth'
    trainer.save_model(model_path)
    
    # Test predictions
    logging.info("Generating test predictions...")
    predictor = FlipPredictor(model_path, train_df)
    top_flips = predictor.predict_flips(test_df, top_k=100)
    
    # Save predictions
    predictions_df = pd.DataFrame(top_flips)
    predictions_df.to_csv('output/metamodel_flip_predictions.csv', index=False)
    logging.info(f"Top 10 flip candidates:")
    for i, flip in enumerate(top_flips[:10]):
        logging.info(f"{i+1}. ID {flip['id']}: score={flip['flip_score']:.4f}")


if __name__ == "__main__":
    main()