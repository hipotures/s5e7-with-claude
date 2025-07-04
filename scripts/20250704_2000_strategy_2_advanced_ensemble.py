#!/usr/bin/env python3
"""
STRATEGY 2: ADVANCED ENSEMBLE WITH UNCERTAINTY QUANTIFICATION
============================================================

HYPOTHESIS:
The 0.975708 ceiling exists because current approaches use static rules 
(e.g., "96.2% of ambiguous cases are Extrovert"). A meta-learning system
that learns WHEN to trust the model vs. apply rules could find edge cases
where these rules don't apply.

APPROACH:
1. Train 5 diverse models (XGBoost, CatBoost, LightGBM, Neural Net, Random Forest)
2. Calculate uncertainty metrics for each prediction
3. Train a meta-model that learns optimal decision strategies
4. Use conformal prediction for calibrated confidence intervals

KEY INNOVATION:
Instead of fixed rules, we learn a dynamic decision function that adapts
based on prediction uncertainty and agreement between models.

Author: Claude (Strategy Implementation)
Date: 2025-01-04
"""

# PURPOSE: Build advanced ensemble with uncertainty quantification and meta-learning
# HYPOTHESIS: Meta-learner can learn WHEN to trust models vs apply rules for ambiguous cases
# EXPECTED: Dynamic decision boundaries will outperform static rules on edge cases
# RESULT: 5-model ensemble with uncertainty metrics, conformal prediction, and meta-learner

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy.stats import entropy
import optuna
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)

class UncertaintyQuantifier:
    """
    Calculates various uncertainty metrics for predictions.
    """
    @staticmethod
    def prediction_entropy(probas):
        """Shannon entropy of prediction probabilities."""
        # Avoid log(0)
        probas = np.clip(probas, 1e-7, 1-1e-7)
        return -np.sum(probas * np.log(probas), axis=1)
    
    @staticmethod
    def prediction_variance(probas_list):
        """Variance of predictions across models."""
        stacked = np.stack(probas_list)
        return np.var(stacked, axis=0)
    
    @staticmethod
    def mutual_information(probas_list):
        """Mutual information between models - measures disagreement."""
        n_models = len(probas_list)
        mi_scores = []
        
        for i in range(len(probas_list[0])):
            # Get predictions for sample i from all models
            preds = [p[i] for p in probas_list]
            
            # Calculate pairwise disagreement
            disagreement = 0
            for j in range(n_models):
                for k in range(j+1, n_models):
                    disagreement += abs(preds[j] - preds[k])
            
            mi_scores.append(disagreement / (n_models * (n_models - 1) / 2))
        
        return np.array(mi_scores)
    
    @staticmethod
    def distance_to_decision_boundary(probas):
        """How far the prediction is from 0.5."""
        return np.abs(probas - 0.5)

class ConformalPredictor:
    """
    Implements conformal prediction for uncertainty-aware classification.
    Provides prediction intervals with guaranteed coverage.
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha  # Significance level (95% confidence)
        self.calibration_scores = None
        self.threshold = None
    
    def calibrate(self, probas, y_true):
        """Calibrate on a held-out set."""
        # Calculate non-conformity scores
        self.calibration_scores = []
        
        for i in range(len(y_true)):
            true_class = int(y_true[i])
            # Non-conformity score: 1 - probability of true class
            score = 1 - probas[i, true_class]
            self.calibration_scores.append(score)
        
        # Set threshold at (1-alpha) quantile
        self.calibration_scores = np.sort(self.calibration_scores)
        k = int(np.ceil((len(self.calibration_scores) + 1) * (1 - self.alpha)))
        self.threshold = self.calibration_scores[min(k-1, len(self.calibration_scores)-1)]
    
    def predict_with_confidence(self, probas):
        """Return prediction sets with confidence."""
        prediction_sets = []
        confidence_scores = []
        
        for i in range(len(probas)):
            # Include class in prediction set if its score exceeds threshold
            pred_set = []
            scores = []
            
            for c in [0, 1]:  # Binary classification
                score = 1 - probas[i, c]
                if score <= self.threshold:
                    pred_set.append(c)
                    scores.append(probas[i, c])
            
            prediction_sets.append(pred_set)
            confidence_scores.append(scores)
        
        return prediction_sets, confidence_scores

def create_enhanced_features(df):
    """
    Create features including those that help identify ambiguous cases.
    """
    features = df.copy()
    
    # Known ambiguous markers
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    
    # Marker features
    for col, val in markers.items():
        if col in features.columns:
            features[f'has_marker_{col}'] = (features[col] == val).astype(int)
    
    features['marker_count'] = sum(
        features[f'has_marker_{col}'] for col in markers.keys() 
        if col in features.columns
    )
    
    # Behavioral patterns
    features['introvert_score'] = (
        features['Time_spent_Alone'] * features['Drained_after_socializing'] +
        features['Stage_fear']
    ) / 3
    
    features['extrovert_score'] = (
        features['Social_event_attendance'] + features['Going_outside'] +
        features['Friends_circle_size'] / 2
    ) / 3
    
    features['ambiguity_score'] = np.abs(
        features['introvert_score'] - features['extrovert_score']
    )
    
    # Consistency features
    features['social_consistency'] = np.abs(
        features['Social_event_attendance'] - features['Going_outside']
    )
    
    features['energy_consistency'] = np.abs(
        features['Drained_after_socializing'] - features['Time_spent_Alone'] / 2
    )
    
    # Special patterns for ambiguous cases
    features['low_alone_moderate_social'] = (
        (features['Time_spent_Alone'] < 2.5) & 
        (features['Social_event_attendance'].between(3, 4))
    ).astype(int)
    
    features['ambiguous_friends_pattern'] = (
        features['Friends_circle_size'].between(6, 7)
    ).astype(int)
    
    return features

class DiverseEnsemble:
    """
    Ensemble of diverse models for robust predictions.
    """
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='gpu_hist' if self._check_gpu() else 'hist'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                device='gpu' if self._check_gpu() else 'cpu'
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.01,
                random_state=42,
                verbose=False,
                task_type='GPU' if self._check_gpu() else 'CPU'
            ),
            'neural_net': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        }
        
        self.calibrated_models = {}
        self.feature_importance = {}
    
    def _check_gpu(self):
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def fit(self, X_train, y_train, X_cal=None, y_cal=None):
        """
        Fit all models with optional calibration.
        """
        print("Training diverse ensemble models...")
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            # Handle neural network scaling
            if name == 'neural_net':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                model.fit(X_train_scaled, y_train)
                # Store scaler for later use
                self.models[name].scaler_ = scaler
            else:
                model.fit(X_train, y_train)
            
            # Calibrate if calibration set provided
            if X_cal is not None and y_cal is not None:
                print(f"    Calibrating {name}...")
                if name == 'neural_net':
                    X_cal_scaled = scaler.transform(X_cal)
                    calibrated = CalibratedClassifierCV(
                        model, method='isotonic', cv='prefit'
                    )
                    calibrated.fit(X_cal_scaled, y_cal)
                else:
                    calibrated = CalibratedClassifierCV(
                        model, method='isotonic', cv='prefit'
                    )
                    calibrated.fit(X_cal, y_cal)
                self.calibrated_models[name] = calibrated
            
            # Extract feature importance where available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
    
    def predict_proba_all(self, X_test):
        """
        Get probability predictions from all models.
        """
        predictions = {}
        
        for name, model in self.models.items():
            # Use calibrated model if available
            if name in self.calibrated_models:
                model_to_use = self.calibrated_models[name]
            else:
                model_to_use = model
            
            # Handle neural network scaling
            if name == 'neural_net' and hasattr(model, 'scaler_'):
                X_test_scaled = model.scaler_.transform(X_test)
                predictions[name] = model_to_use.predict_proba(X_test_scaled)
            else:
                predictions[name] = model_to_use.predict_proba(X_test)
        
        return predictions

class MetaLearner:
    """
    Meta-model that learns when to trust base models vs. apply special rules.
    """
    def __init__(self):
        self.meta_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.threshold_optimizer = None
        self.optimal_thresholds = {}
    
    def create_meta_features(self, base_predictions, uncertainty_metrics, X_original):
        """
        Create features for the meta-model.
        """
        meta_features = []
        
        # Base model predictions
        for name, preds in base_predictions.items():
            meta_features.append(preds[:, 1])  # Probability of class 1
        
        # Uncertainty metrics
        meta_features.extend([
            uncertainty_metrics['entropy'],
            uncertainty_metrics['variance'],
            uncertainty_metrics['mutual_info'],
            uncertainty_metrics['distance_boundary']
        ])
        
        # Agreement metrics
        pred_values = [p[:, 1] for p in base_predictions.values()]
        mean_pred = np.mean(pred_values, axis=0)
        std_pred = np.std(pred_values, axis=0)
        
        meta_features.extend([mean_pred, std_pred])
        
        # Min/max predictions
        min_pred = np.min(pred_values, axis=0)
        max_pred = np.max(pred_values, axis=0)
        range_pred = max_pred - min_pred
        
        meta_features.extend([min_pred, max_pred, range_pred])
        
        # Original features that indicate ambiguity
        ambiguity_cols = [
            'marker_count', 'ambiguity_score', 'low_alone_moderate_social',
            'ambiguous_friends_pattern'
        ]
        
        for col in ambiguity_cols:
            if col in X_original.columns:
                meta_features.append(X_original[col].values)
        
        return np.column_stack(meta_features)
    
    def optimize_thresholds(self, X_meta, y_true, group_labels):
        """
        Use Optuna to find optimal thresholds for different groups.
        """
        def objective(trial):
            # Define thresholds for different scenarios
            thresholds = {
                'high_uncertainty': trial.suggest_float('thresh_high_unc', 0.3, 0.7),
                'high_disagreement': trial.suggest_float('thresh_high_dis', 0.3, 0.7),
                'ambiguous_pattern': trial.suggest_float('thresh_ambig', 0.35, 0.5),
                'confident': trial.suggest_float('thresh_conf', 0.45, 0.55)
            }
            
            # Apply thresholds based on group labels
            predictions = np.zeros(len(y_true))
            
            for i in range(len(y_true)):
                group = group_labels[i]
                base_prob = X_meta[i, 0]  # Use first model's prediction as base
                
                if group in thresholds:
                    predictions[i] = int(base_prob > thresholds[group])
                else:
                    predictions[i] = int(base_prob > 0.5)
            
            return accuracy_score(y_true, predictions)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, show_progress_bar=False)
        
        self.optimal_thresholds = study.best_params
        return study.best_value
    
    def fit(self, X_meta, y_true, uncertainty_groups):
        """
        Train the meta-learner.
        """
        # First optimize thresholds
        print("  Optimizing decision thresholds...")
        best_score = self.optimize_thresholds(X_meta, y_true, uncertainty_groups)
        print(f"    Best threshold optimization score: {best_score:.6f}")
        
        # Train meta-model
        print("  Training meta-model...")
        self.meta_model.fit(X_meta, y_true)
    
    def predict_with_strategy(self, X_meta, uncertainty_groups):
        """
        Make predictions using learned strategies.
        """
        # Get base predictions from meta-model
        meta_probas = self.meta_model.predict_proba(X_meta)[:, 1]
        
        # Apply group-specific thresholds
        predictions = np.zeros(len(X_meta))
        
        for i in range(len(X_meta)):
            group = uncertainty_groups[i]
            prob = meta_probas[i]
            
            # Apply learned threshold for this group
            if group == 'high_uncertainty':
                threshold = self.optimal_thresholds.get('thresh_high_unc', 0.5)
            elif group == 'high_disagreement':
                threshold = self.optimal_thresholds.get('thresh_high_dis', 0.5)
            elif group == 'ambiguous_pattern':
                threshold = self.optimal_thresholds.get('thresh_ambig', 0.45)
            elif group == 'confident':
                threshold = self.optimal_thresholds.get('thresh_conf', 0.5)
            else:
                threshold = 0.5
            
            predictions[i] = int(prob > threshold)
        
        return predictions, meta_probas

def categorize_by_uncertainty(uncertainty_metrics, X_original):
    """
    Categorize samples based on uncertainty and patterns.
    """
    categories = []
    
    for i in range(len(uncertainty_metrics['entropy'])):
        # High uncertainty
        if uncertainty_metrics['entropy'][i] > 0.8:
            categories.append('high_uncertainty')
        # High disagreement between models
        elif uncertainty_metrics['mutual_info'][i] > 0.3:
            categories.append('high_disagreement')
        # Known ambiguous pattern
        elif (X_original.iloc[i]['marker_count'] > 0 or 
              X_original.iloc[i]['low_alone_moderate_social'] == 1):
            categories.append('ambiguous_pattern')
        # Confident prediction
        elif uncertainty_metrics['distance_boundary'][i] > 0.4:
            categories.append('confident')
        else:
            categories.append('moderate')
    
    return categories

def main():
    """
    Execute Strategy 2: Advanced Ensemble with Uncertainty Quantification
    """
    print("="*80)
    print("STRATEGY 2: ADVANCED ENSEMBLE WITH UNCERTAINTY QUANTIFICATION")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    # Save original IDs
    test_ids = test_df['id']
    
    # Prepare features
    print("\nPreparing enhanced features...")
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                    'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    train_enhanced = create_enhanced_features(train_df[feature_cols])
    test_enhanced = create_enhanced_features(test_df[feature_cols])
    
    # Prepare data
    X = train_enhanced.values
    y = train_df['Target'].values
    X_test = test_enhanced.values
    
    # Split for training and calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split calibration set for meta-learning
    X_cal, X_meta, y_cal, y_meta = train_test_split(
        X_cal, y_cal, test_size=0.5, random_state=42, stratify=y_cal
    )
    
    # Train diverse ensemble
    ensemble = DiverseEnsemble()
    ensemble.fit(X_train, y_train, X_cal, y_cal)
    
    # Get predictions on meta-training set
    print("\nGenerating predictions for meta-learning...")
    meta_predictions = ensemble.predict_proba_all(X_meta)
    
    # Calculate uncertainty metrics
    print("Calculating uncertainty metrics...")
    uq = UncertaintyQuantifier()
    
    pred_list = [p[:, 1] for p in meta_predictions.values()]
    uncertainty_metrics = {
        'entropy': uq.prediction_entropy(np.column_stack([
            np.mean(pred_list, axis=0),
            1 - np.mean(pred_list, axis=0)
        ])),
        'variance': uq.prediction_variance(pred_list),
        'mutual_info': uq.mutual_information(pred_list),
        'distance_boundary': uq.distance_to_decision_boundary(np.mean(pred_list, axis=0))
    }
    
    # Categorize samples by uncertainty
    X_meta_df = pd.DataFrame(X_meta, columns=train_enhanced.columns)
    uncertainty_groups = categorize_by_uncertainty(uncertainty_metrics, X_meta_df)
    
    # Create meta-features
    print("Creating meta-features...")
    X_meta_features = MetaLearner().create_meta_features(
        meta_predictions, uncertainty_metrics, X_meta_df
    )
    
    # Train meta-learner
    print("\nTraining meta-learner...")
    meta_learner = MetaLearner()
    meta_learner.fit(X_meta_features, y_meta, uncertainty_groups)
    
    # Setup conformal predictor
    print("\nCalibrating conformal predictor...")
    conformal = ConformalPredictor(alpha=0.05)
    
    # Get ensemble predictions for calibration
    cal_predictions = ensemble.predict_proba_all(X_cal)
    mean_cal_probas = np.mean([p for p in cal_predictions.values()], axis=0)
    conformal.calibrate(mean_cal_probas, y_cal)
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    test_predictions = ensemble.predict_proba_all(X_test)
    
    # Calculate test uncertainty metrics
    test_pred_list = [p[:, 1] for p in test_predictions.values()]
    test_uncertainty_metrics = {
        'entropy': uq.prediction_entropy(np.column_stack([
            np.mean(test_pred_list, axis=0),
            1 - np.mean(test_pred_list, axis=0)
        ])),
        'variance': uq.prediction_variance(test_pred_list),
        'mutual_info': uq.mutual_information(test_pred_list),
        'distance_boundary': uq.distance_to_decision_boundary(np.mean(test_pred_list, axis=0))
    }
    
    # Categorize test samples
    X_test_df = pd.DataFrame(X_test, columns=train_enhanced.columns)
    test_uncertainty_groups = categorize_by_uncertainty(test_uncertainty_metrics, X_test_df)
    
    # Create test meta-features
    X_test_meta = meta_learner.create_meta_features(
        test_predictions, test_uncertainty_metrics, X_test_df
    )
    
    # Get final predictions with strategies
    final_predictions, meta_probas = meta_learner.predict_with_strategy(
        X_test_meta, test_uncertainty_groups
    )
    
    # Get conformal prediction sets
    mean_test_probas = np.column_stack([
        1 - np.mean(test_pred_list, axis=0),
        np.mean(test_pred_list, axis=0)
    ])
    prediction_sets, confidence_scores = conformal.predict_with_confidence(mean_test_probas)
    
    # Analyze uncertainty groups
    print(f"\nTest set uncertainty group distribution:")
    group_counts = pd.Series(test_uncertainty_groups).value_counts()
    for group, count in group_counts.items():
        print(f"  {group}: {count} ({count/len(test_uncertainty_groups)*100:.2f}%)")
    
    # Handle uncertain predictions
    print("\nHandling uncertain predictions...")
    uncertain_count = 0
    for i in range(len(final_predictions)):
        # If conformal prediction is uncertain (includes both classes)
        if len(prediction_sets[i]) > 1:
            uncertain_count += 1
            # For uncertain cases in ambiguous pattern group, apply 96.2% rule
            if test_uncertainty_groups[i] == 'ambiguous_pattern':
                final_predictions[i] = 1  # Extrovert
    
    print(f"  Applied special handling to {uncertain_count} uncertain predictions")
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_ids,
        'Target': final_predictions.astype(int)
    })
    
    submission_path = 'submission_strategy2_advanced_ensemble.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    
    # Cross-validation to estimate performance
    print("\n" + "="*60)
    print("PERFORMANCE ESTIMATION (5-Fold CV)")
    print("="*60)
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Split training data for calibration
        X_ft, X_fc, y_ft, y_fc = train_test_split(
            X_fold_train, y_fold_train, test_size=0.2, random_state=42
        )
        
        # Train ensemble
        fold_ensemble = DiverseEnsemble()
        fold_ensemble.fit(X_ft, y_ft, X_fc, y_fc)
        
        # Get predictions
        fold_predictions = fold_ensemble.predict_proba_all(X_fold_val)
        
        # Simple voting for CV (full meta-learning would be too slow)
        fold_pred_probs = np.mean([p[:, 1] for p in fold_predictions.values()], axis=0)
        fold_pred_binary = (fold_pred_probs > 0.5).astype(int)
        
        fold_score = accuracy_score(y_fold_val, fold_pred_binary)
        cv_scores.append(fold_score)
        print(f"  Fold {fold+1}: {fold_score:.6f}")
    
    print(f"\nMean CV Score: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores)*2:.6f})")
    
    # Detailed analysis
    print("\n" + "="*60)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("="*60)
    
    # Analyze prediction confidence by group
    for group in ['high_uncertainty', 'high_disagreement', 'ambiguous_pattern', 'confident']:
        group_mask = np.array(test_uncertainty_groups) == group
        if group_mask.sum() > 0:
            group_probas = meta_probas[group_mask]
            print(f"\n{group.upper()} group:")
            print(f"  Count: {group_mask.sum()}")
            print(f"  Mean probability: {np.mean(group_probas):.4f}")
            print(f"  Std probability: {np.std(group_probas):.4f}")
            print(f"  Predictions: {int(np.sum(final_predictions[group_mask]))} Extrovert, "
                  f"{int(np.sum(1-final_predictions[group_mask]))} Introvert")

if __name__ == "__main__":
    main()