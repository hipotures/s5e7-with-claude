# MetaModel Flip Predictor Design Document

## System Architecture

### Overview
A meta-learning system that predicts which samples are likely mislabeled (good flip candidates) using neural networks trained on fast YDF evaluations.

### Core Components

```
┌─────────────────────────────────────────────────────┐
│                  MetaModel Pipeline                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Feature Extractor                               │
│     └─> Sample features + Uncertainty metrics       │
│                                                     │
│  2. YDF Fast Evaluator                             │
│     └─> Subset-based flip evaluation               │
│                                                     │
│  3. Neural MetaModel                               │
│     └─> Flip quality prediction                    │
│                                                     │
│  4. Validation Loop                                │
│     └─> Cross-validation on flip performance       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Detailed Design

### 1. Feature Engineering for MetaModel

```python
class FlipFeatureExtractor:
    """Extract features that indicate potential mislabeling."""
    
    def extract_features(self, sample_id, train_data, model_predictions):
        features = {
            # Sample characteristics
            'null_count': count_nulls(sample),
            'feature_extremity': calculate_extremity_score(sample),
            
            # Model uncertainty
            'prediction_entropy': entropy(predictions),
            'prediction_variance': variance(predictions),
            'prediction_margin': min_margin_to_decision_boundary,
            
            # Neighborhood analysis
            'knn_disagreement': knn_label_disagreement(sample, k=10),
            'local_density': local_outlier_factor(sample),
            
            # Cross-model agreement
            'model_agreement_score': cross_model_agreement(predictions),
            'gradient_magnitude': gradient_based_importance(sample),
            
            # Historical flip performance
            'similar_flip_success_rate': historical_similarity_score()
        }
        return features
```

### 2. YDF Fast Evaluation Module

```python
class YDFSubsetEvaluator:
    """Fast evaluation using YDF on data subsets."""
    
    def __init__(self, n_subsets=10, subset_size=0.8):
        self.n_subsets = n_subsets
        self.subset_size = subset_size
        
    def evaluate_flip(self, train_data, flip_id, flip_direction):
        """Evaluate a single flip across multiple subsets."""
        scores = []
        
        for i in range(self.n_subsets):
            # Create subset excluding test sample
            subset = self._create_subset(train_data, exclude_id=flip_id)
            
            # Train YDF on original subset
            model_original = self._train_ydf_fast(subset)
            score_original = self._evaluate_holdout(model_original, flip_id)
            
            # Train YDF with flipped label
            subset_flipped = self._apply_flip(subset, flip_id, flip_direction)
            model_flipped = self._train_ydf_fast(subset_flipped)
            score_flipped = self._evaluate_holdout(model_flipped, flip_id)
            
            # Calculate improvement
            improvement = score_flipped - score_original
            scores.append(improvement)
            
        return {
            'mean_improvement': np.mean(scores),
            'std_improvement': np.std(scores),
            'positive_rate': sum(s > 0 for s in scores) / len(scores)
        }
    
    def _train_ydf_fast(self, data):
        """Train minimal YDF for speed."""
        learner = tfdf.keras.RandomForestModel(
            num_trees=50,  # Fewer trees for speed
            max_depth=8,
            winner_take_all=True
        )
        return learner.fit(data, verbose=0)
```

### 3. Neural MetaModel Architecture

```python
class FlipPredictorNN(nn.Module):
    """Neural network to predict flip quality."""
    
    def __init__(self, input_dim=20, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output: probability of good flip
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
```

### 4. Training Pipeline

```python
class MetaModelTrainer:
    """Train the metamodel using YDF evaluations."""
    
    def __init__(self, feature_extractor, ydf_evaluator, neural_model):
        self.feature_extractor = feature_extractor
        self.ydf_evaluator = ydf_evaluator
        self.neural_model = neural_model
        
    def generate_training_data(self, train_data, n_candidates=1000):
        """Generate flip candidates and evaluate them."""
        training_samples = []
        
        # Select candidate samples (high uncertainty, outliers, etc.)
        candidates = self._select_candidates(train_data, n_candidates)
        
        for candidate_id in tqdm(candidates, desc="Evaluating flips"):
            # Extract features
            features = self.feature_extractor.extract_features(
                candidate_id, train_data, self.base_predictions
            )
            
            # Evaluate flip with YDF
            flip_result = self.ydf_evaluator.evaluate_flip(
                train_data, candidate_id, 
                flip_direction=self._get_flip_direction(candidate_id)
            )
            
            # Label: 1 if good flip (improves score), 0 otherwise
            label = 1 if flip_result['mean_improvement'] > 0.001 else 0
            
            training_samples.append({
                'features': features,
                'label': label,
                'improvement': flip_result['mean_improvement']
            })
            
        return training_samples
    
    def train(self, train_data, epochs=50):
        """Train the neural metamodel."""
        # Generate training data
        training_samples = self.generate_training_data(train_data)
        
        # Create PyTorch dataset
        dataset = FlipDataset(training_samples)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            for batch in train_loader:
                features, labels = batch
                
                # Forward pass
                predictions = self.neural_model(features)
                loss = criterion(predictions, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

### 5. Inference Pipeline

```python
class FlipPredictor:
    """Use trained metamodel to predict good flips."""
    
    def __init__(self, trained_metamodel, feature_extractor):
        self.metamodel = trained_metamodel
        self.feature_extractor = feature_extractor
        
    def predict_flips(self, test_data, train_data, top_k=50):
        """Predict top K flip candidates."""
        flip_scores = []
        
        for test_id in test_data.index:
            # Extract features
            features = self.feature_extractor.extract_features(
                test_id, train_data, model_predictions=None
            )
            
            # Predict flip quality
            with torch.no_grad():
                flip_score = self.metamodel(
                    torch.tensor(features).float()
                ).item()
            
            flip_scores.append({
                'id': test_id,
                'flip_score': flip_score,
                'current_prediction': self._get_current_prediction(test_id)
            })
        
        # Return top K candidates
        flip_scores.sort(key=lambda x: x['flip_score'], reverse=True)
        return flip_scores[:top_k]
```

## Implementation Strategy

### Phase 1: Data Generation
1. Create synthetic mislabeled samples in training data
2. Evaluate flip performance using YDF subsets
3. Build initial training dataset

### Phase 2: Model Development
1. Implement feature extraction pipeline
2. Build and train neural metamodel
3. Validate on held-out flips

### Phase 3: Production Pipeline
1. Apply to test set predictions
2. Generate ranked flip candidates
3. Create submission files for top candidates

## Advantages

1. **Fast Evaluation**: YDF provides quick feedback on flip quality
2. **Generalization**: Neural network learns patterns of mislabeling
3. **Uncertainty Quantification**: Multiple subset evaluations provide confidence
4. **Scalable**: Can evaluate thousands of candidates efficiently

## Evaluation Metrics

- **Flip Precision**: % of predicted flips that improve score
- **Flip Recall**: % of true errors found
- **Ranking Quality**: nDCG of flip score ranking
- **Efficiency**: Flips evaluated per second

## Next Steps

1. Implement feature extractor with domain-specific features
2. Set up YDF subset evaluation pipeline
3. Design neural network architecture experiments
4. Create training data generation pipeline
5. Build end-to-end training and inference system