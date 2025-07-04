#!/usr/bin/env python3
"""
STRATEGY 3: ADVERSARIAL TRAINING FOR AMBIGUOUS CASES
====================================================

HYPOTHESIS:
Current models overfit to exact ambiguous patterns in training data.
By creating adversarial examples around the decision boundary, we can
help the model generalize better to unseen ambiguous cases.

APPROACH:
1. Identify ambiguous cases and decision boundary
2. Generate adversarial examples using multiple techniques
3. Train with adversarial loss that penalizes overconfidence
4. Use temperature scaling for calibrated probabilities

KEY INNOVATION:
Instead of memorizing specific ambiguous patterns, the model learns
robust decision boundaries that generalize to new ambiguous cases.

Author: Claude (Strategy Implementation)
Date: 2025-01-04
"""

# PURPOSE: Use adversarial training to improve model robustness on ambiguous cases
# HYPOTHESIS: Models overfit to exact patterns; adversarial examples improve generalization
# EXPECTED: Adversarial training will create more robust decision boundaries
# RESULT: Multiple adversarial generation methods with temperature-scaled neural network

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)

class AdversarialGenerator:
    """
    Generates adversarial examples for robust training.
    """
    def __init__(self, epsilon=0.1, n_neighbors=5):
        self.epsilon = epsilon
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors)
    
    def fit(self, X, y):
        """Fit the generator on training data."""
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = y
        self.nn_model.fit(self.X_scaled)
    
    def generate_boundary_perturbations(self, X_ambiguous, y_ambiguous, n_samples=5):
        """
        Generate adversarial examples by perturbing ambiguous cases.
        """
        X_scaled = self.scaler.transform(X_ambiguous)
        adversarial_X = []
        adversarial_y = []
        
        for i in range(len(X_ambiguous)):
            x_original = X_scaled[i]
            y_original = y_ambiguous[i]
            
            # Find nearest neighbors with opposite class
            distances, indices = self.nn_model.kneighbors([x_original])
            opposite_class_neighbors = [
                idx for idx in indices[0] if self.y[idx] != y_original
            ]
            
            if len(opposite_class_neighbors) > 0:
                # Generate perturbations towards opposite class
                for _ in range(n_samples):
                    # Pick random opposite class neighbor
                    target_idx = np.random.choice(opposite_class_neighbors)
                    target_point = self.X_scaled[target_idx]
                    
                    # Create perturbation
                    direction = target_point - x_original
                    perturbation = self.epsilon * np.random.uniform(0.5, 1.5) * direction
                    
                    # Add noise
                    noise = np.random.normal(0, 0.01, size=x_original.shape)
                    
                    adversarial_point = x_original + perturbation + noise
                    adversarial_X.append(adversarial_point)
                    
                    # Label depends on perturbation strength
                    if np.random.random() < 0.7:  # 70% keep original label
                        adversarial_y.append(y_original)
                    else:  # 30% flip label
                        adversarial_y.append(1 - y_original)
        
        if len(adversarial_X) > 0:
            adversarial_X = self.scaler.inverse_transform(adversarial_X)
            return np.array(adversarial_X), np.array(adversarial_y)
        else:
            return np.array([]), np.array([])
    
    def generate_mixup_samples(self, X_ambiguous, y_ambiguous, X_clear, y_clear, alpha=0.2):
        """
        Generate samples using mixup between ambiguous and clear cases.
        """
        n_samples = min(len(X_ambiguous), len(X_clear))
        mixup_X = []
        mixup_y = []
        
        for i in range(n_samples):
            # Random ambiguous and clear samples
            idx_ambig = np.random.randint(len(X_ambiguous))
            idx_clear = np.random.randint(len(X_clear))
            
            # Mixup coefficient
            lam = np.random.beta(alpha, alpha)
            
            # Create mixed sample
            mixed_x = lam * X_ambiguous[idx_ambig] + (1 - lam) * X_clear[idx_clear]
            mixed_y = lam * y_ambiguous[idx_ambig] + (1 - lam) * y_clear[idx_clear]
            
            mixup_X.append(mixed_x)
            mixup_y.append(mixed_y)
        
        return np.array(mixup_X), np.array(mixup_y)
    
    def generate_smote_variants(self, X_ambiguous, y_ambiguous, sampling_strategy='auto'):
        """
        Use SMOTE specifically on ambiguous region.
        """
        if len(np.unique(y_ambiguous)) < 2:
            # If only one class, can't use SMOTE
            return X_ambiguous, y_ambiguous
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=min(3, len(X_ambiguous)-1),
            random_state=42
        )
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X_ambiguous, y_ambiguous)
            return X_resampled, y_resampled
        except:
            # If SMOTE fails, return original
            return X_ambiguous, y_ambiguous

class TemperatureScaledModel(nn.Module):
    """
    Neural network with temperature scaling for calibrated probabilities.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(TemperatureScaledModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x, return_logits=False):
        logits = self.model(x)
        
        if return_logits:
            return logits
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        return torch.sigmoid(scaled_logits)
    
    def calibrate_temperature(self, val_loader, device):
        """
        Optimize temperature on validation set.
        """
        self.eval()
        nll_criterion = nn.BCEWithLogitsLoss()
        
        # Collect all logits and labels
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = self.forward(inputs, return_logits=True)
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Optimize temperature
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_temperature():
            scaled_logits = logits / self.temperature
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_temperature)

def identify_ambiguous_cases(df):
    """
    Identify ambiguous cases using multiple criteria.
    """
    # Known markers
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    
    # Check for markers
    has_marker = pd.Series([False] * len(df))
    for col, val in markers.items():
        if col in df.columns:
            has_marker |= (np.abs(df[col] - val) < 1e-6)
    
    # Behavioral patterns
    low_alone = df['Time_spent_Alone'] < 2.5
    moderate_social = df['Social_event_attendance'].between(3, 4)
    medium_friends = df['Friends_circle_size'].between(6, 7)
    
    # Consistency patterns
    low_drain = df['Drained_after_socializing'] < 3
    low_stage_fear = df['Stage_fear'] < 3
    
    # Combine criteria
    ambiguous = (
        has_marker | 
        (low_alone & moderate_social) |
        (medium_friends & low_drain & low_stage_fear)
    )
    
    return ambiguous

def create_adversarial_features(df):
    """
    Create features including adversarial-robust ones.
    """
    features = df.copy()
    
    # Robust features that are harder to adversarially perturb
    features['social_introvert_ratio'] = (
        features['Time_spent_Alone'] / 
        (features['Social_event_attendance'] + features['Going_outside'] + 1)
    )
    
    features['energy_depletion_score'] = (
        features['Drained_after_socializing'] * features['Social_event_attendance'] /
        (features['Time_spent_Alone'] + 1)
    )
    
    features['social_anxiety_score'] = (
        features['Stage_fear'] / (features['Friends_circle_size'] + 1)
    )
    
    # Interaction features
    features['social_media_personality'] = (
        features['Post_frequency'] * features['Friends_circle_size'] /
        (features['Social_event_attendance'] + features['Going_outside'] + 1)
    )
    
    # Non-linear transformations (harder to adversarially attack)
    features['log_alone_time'] = np.log1p(features['Time_spent_Alone'])
    features['sqrt_social_events'] = np.sqrt(features['Social_event_attendance'])
    features['friends_squared'] = features['Friends_circle_size'] ** 2
    
    # Ambiguity detection features
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    
    for col, val in markers.items():
        if col in features.columns:
            features[f'dist_from_marker_{col}'] = np.abs(features[col] - val)
    
    return features

class AdversarialXGBoost:
    """
    XGBoost model trained with adversarial examples.
    """
    def __init__(self, base_params=None):
        if base_params is None:
            base_params = {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 5,
                'reg_lambda': 2,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
        
        self.model = xgb.XGBClassifier(**base_params)
        self.adversarial_weight = 0.5
    
    def fit(self, X_train, y_train, X_adversarial=None, y_adversarial=None,
            X_val=None, y_val=None):
        """
        Train with optional adversarial examples.
        """
        if X_adversarial is not None and len(X_adversarial) > 0:
            # Combine original and adversarial data
            X_combined = np.vstack([X_train, X_adversarial])
            y_combined = np.hstack([y_train, y_adversarial])
            
            # Create sample weights (lower weight for adversarial)
            weights = np.ones(len(X_combined))
            weights[len(X_train):] = self.adversarial_weight
            
            # Train with early stopping if validation set provided
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_combined, y_combined,
                    sample_weight=weights,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.model.fit(X_combined, y_combined, sample_weight=weights)
        else:
            # Standard training
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)

def train_adversarial_neural_network(X_train, y_train, X_val, y_val, 
                                    X_adversarial=None, y_adversarial=None,
                                    epochs=100):
    """
    Train neural network with adversarial examples and temperature scaling.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training on device: {device}")
    
    # Combine training data if adversarial examples provided
    if X_adversarial is not None and len(X_adversarial) > 0:
        X_combined = np.vstack([X_train, X_adversarial])
        y_combined = np.hstack([y_train, y_adversarial])
        
        # Create weights
        weights = np.ones(len(X_combined))
        weights[len(X_train):] = 0.5  # Lower weight for adversarial
    else:
        X_combined = X_train
        y_combined = y_train
        weights = np.ones(len(X_combined))
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_combined),
        torch.FloatTensor(y_combined),
        torch.FloatTensor(weights)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = TemperatureScaledModel(input_dim=X_train.shape[1]).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for inputs, labels, sample_weights in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            sample_weights = sample_weights.to(device)
            
            optimizer.zero_grad()
            
            logits = model(inputs, return_logits=True)
            losses = criterion(logits, labels)
            
            # Apply sample weights
            weighted_loss = (losses.squeeze() * sample_weights).mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            train_losses.append(weighted_loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                probas = model(inputs)
                loss = criterion(model(inputs, return_logits=True), labels).mean()
                
                val_losses.append(loss.item())
                val_preds.extend(probas.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
        
        if epoch % 20 == 0:
            val_acc = accuracy_score(val_true, (np.array(val_preds) > 0.5).astype(int))
            print(f"    Epoch {epoch}: Train Loss: {np.mean(train_losses):.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model and calibrate temperature
    model.load_state_dict(best_model_state)
    print("  Calibrating temperature...")
    model.calibrate_temperature(val_loader, device)
    
    return model

def main():
    """
    Execute Strategy 3: Adversarial Training for Ambiguous Cases
    """
    print("="*80)
    print("STRATEGY 3: ADVERSARIAL TRAINING FOR AMBIGUOUS CASES")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    # Save original IDs
    test_ids = test_df['id']
    
    # Prepare features
    print("\nPreparing adversarial-robust features...")
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                    'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    train_enhanced = create_adversarial_features(train_df[feature_cols])
    test_enhanced = create_adversarial_features(test_df[feature_cols])
    
    # Identify ambiguous cases
    print("\nIdentifying ambiguous cases...")
    ambiguous_mask = identify_ambiguous_cases(train_df)
    print(f"Found {ambiguous_mask.sum()} ambiguous cases ({ambiguous_mask.mean()*100:.2f}%)")
    
    # Separate ambiguous and clear cases
    X_all = train_enhanced.values
    y_all = train_df['Target'].values
    
    X_ambiguous = X_all[ambiguous_mask]
    y_ambiguous = y_all[ambiguous_mask]
    X_clear = X_all[~ambiguous_mask]
    y_clear = y_all[~ambiguous_mask]
    
    print(f"\nAmbiguous cases - Extrovert ratio: {y_ambiguous.mean():.2%}")
    print(f"Clear cases - Extrovert ratio: {y_clear.mean():.2%}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # Generate adversarial examples
    print("\nGenerating adversarial examples...")
    adversarial_gen = AdversarialGenerator(epsilon=0.1)
    adversarial_gen.fit(X_train, y_train)
    
    # Identify ambiguous cases in training set
    train_ambiguous_mask = identify_ambiguous_cases(
        pd.DataFrame(X_train, columns=train_enhanced.columns)
    )
    X_train_ambiguous = X_train[train_ambiguous_mask]
    y_train_ambiguous = y_train[train_ambiguous_mask]
    
    # Generate different types of adversarial examples
    print("  Generating boundary perturbations...")
    X_adv_boundary, y_adv_boundary = adversarial_gen.generate_boundary_perturbations(
        X_train_ambiguous, y_train_ambiguous, n_samples=3
    )
    print(f"    Generated {len(X_adv_boundary)} boundary perturbations")
    
    print("  Generating mixup samples...")
    X_adv_mixup, y_adv_mixup = adversarial_gen.generate_mixup_samples(
        X_train_ambiguous, y_train_ambiguous,
        X_train[~train_ambiguous_mask], y_train[~train_ambiguous_mask],
        alpha=0.2
    )
    print(f"    Generated {len(X_adv_mixup)} mixup samples")
    
    print("  Generating SMOTE variants...")
    X_adv_smote, y_adv_smote = adversarial_gen.generate_smote_variants(
        X_train_ambiguous, y_train_ambiguous
    )
    print(f"    Generated {len(X_adv_smote) - len(X_train_ambiguous)} SMOTE samples")
    
    # Combine all adversarial examples
    if len(X_adv_boundary) > 0:
        X_adversarial = np.vstack([X_adv_boundary, X_adv_mixup, X_adv_smote])
        y_adversarial = np.hstack([y_adv_boundary, y_adv_mixup, y_adv_smote])
    else:
        X_adversarial = np.vstack([X_adv_mixup, X_adv_smote])
        y_adversarial = np.hstack([y_adv_mixup, y_adv_smote])
    
    print(f"\nTotal adversarial examples: {len(X_adversarial)}")
    
    # Train models with adversarial examples
    print("\nTraining adversarial XGBoost...")
    adv_xgb = AdversarialXGBoost()
    adv_xgb.fit(X_train, y_train, X_adversarial, y_adversarial, X_val, y_val)
    
    print("\nTraining adversarial neural network...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_adversarial_scaled = scaler.transform(X_adversarial)
    
    adv_nn = train_adversarial_neural_network(
        X_train_scaled, y_train, X_val_scaled, y_val,
        X_adversarial_scaled, y_adversarial,
        epochs=100
    )
    
    # Make predictions
    print("\nMaking predictions on test set...")
    X_test = test_enhanced.values
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost predictions
    xgb_probas = adv_xgb.predict_proba(X_test)[:, 1]
    
    # Neural network predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adv_nn.eval()
    with torch.no_grad():
        nn_probas = adv_nn(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy().squeeze()
    
    # Ensemble predictions (weighted average)
    ensemble_probas = 0.6 * xgb_probas + 0.4 * nn_probas
    
    # Identify ambiguous cases in test set
    test_ambiguous_mask = identify_ambiguous_cases(test_df)
    print(f"\nFound {test_ambiguous_mask.sum()} ambiguous cases in test set")
    
    # Apply special handling for ambiguous cases
    final_predictions = (ensemble_probas > 0.5).astype(int)
    
    # For highly ambiguous cases, apply the 96.2% rule with some flexibility
    for i in range(len(final_predictions)):
        if test_ambiguous_mask.iloc[i]:
            # Check confidence
            confidence = abs(ensemble_probas[i] - 0.5)
            
            if confidence < 0.15:  # Low confidence
                # Apply 96.2% rule - predict Extrovert
                final_predictions[i] = 1
            elif ensemble_probas[i] < 0.25:  # Very low probability but ambiguous
                # Still might be that rare 3.8% introvert
                final_predictions[i] = 0
    
    print(f"\nPrediction distribution:")
    print(f"  Introverts: {(final_predictions == 0).sum()} ({(final_predictions == 0).mean()*100:.2f}%)")
    print(f"  Extroverts: {(final_predictions == 1).sum()} ({(final_predictions == 1).mean()*100:.2f}%)")
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_ids,
        'Target': final_predictions
    })
    
    submission_path = 'submission_strategy3_adversarial_training.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    
    # Cross-validation to estimate performance
    print("\n" + "="*60)
    print("PERFORMANCE ESTIMATION (5-Fold CV)")
    print("="*60)
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        X_fold_train, X_fold_val = X_all[train_idx], X_all[val_idx]
        y_fold_train, y_fold_val = y_all[train_idx], y_all[val_idx]
        
        # Generate adversarial examples for this fold
        fold_gen = AdversarialGenerator(epsilon=0.1)
        fold_gen.fit(X_fold_train, y_fold_train)
        
        fold_ambiguous_mask = identify_ambiguous_cases(
            pd.DataFrame(X_fold_train, columns=train_enhanced.columns)
        )
        
        if fold_ambiguous_mask.sum() > 0:
            X_fold_adv, y_fold_adv = fold_gen.generate_boundary_perturbations(
                X_fold_train[fold_ambiguous_mask],
                y_fold_train[fold_ambiguous_mask],
                n_samples=2
            )
        else:
            X_fold_adv, y_fold_adv = np.array([]), np.array([])
        
        # Train model
        fold_model = AdversarialXGBoost()
        fold_model.fit(X_fold_train, y_fold_train, X_fold_adv, y_fold_adv)
        
        # Predict
        fold_predictions = fold_model.predict(X_fold_val)
        fold_score = accuracy_score(y_fold_val, fold_predictions)
        cv_scores.append(fold_score)
        print(f"  Fold {fold+1}: {fold_score:.6f}")
    
    print(f"\nMean CV Score: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores)*2:.6f})")
    
    # Analyze adversarial impact
    print("\n" + "="*60)
    print("ADVERSARIAL TRAINING IMPACT ANALYSIS")
    print("="*60)
    
    # Train model without adversarial examples for comparison
    print("\nTraining baseline model (no adversarial examples)...")
    baseline_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    baseline_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      early_stopping_rounds=50, verbose=False)
    
    # Compare on validation set
    baseline_preds = baseline_model.predict(X_val)
    adv_preds = adv_xgb.predict(X_val)
    
    baseline_acc = accuracy_score(y_val, baseline_preds)
    adv_acc = accuracy_score(y_val, adv_preds)
    
    print(f"\nValidation accuracy comparison:")
    print(f"  Baseline model: {baseline_acc:.6f}")
    print(f"  Adversarial model: {adv_acc:.6f}")
    print(f"  Improvement: {(adv_acc - baseline_acc)*100:.4f}%")
    
    # Check performance on ambiguous cases specifically
    val_ambiguous_mask = identify_ambiguous_cases(
        pd.DataFrame(X_val, columns=train_enhanced.columns)
    )
    
    if val_ambiguous_mask.sum() > 0:
        baseline_ambig_acc = accuracy_score(
            y_val[val_ambiguous_mask], 
            baseline_preds[val_ambiguous_mask]
        )
        adv_ambig_acc = accuracy_score(
            y_val[val_ambiguous_mask], 
            adv_preds[val_ambiguous_mask]
        )
        
        print(f"\nAmbiguous cases accuracy:")
        print(f"  Baseline model: {baseline_ambig_acc:.6f}")
        print(f"  Adversarial model: {adv_ambig_acc:.6f}")
        print(f"  Improvement: {(adv_ambig_acc - baseline_ambig_acc)*100:.4f}%")

if __name__ == "__main__":
    main()