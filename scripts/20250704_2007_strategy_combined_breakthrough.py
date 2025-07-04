#!/usr/bin/env python3
"""
PURPOSE: Combined breakthrough strategy synthesizing all discoveries - MBTI clustering,
uncertainty quantification, and dynamic threshold optimization to exceed 0.975708

HYPOTHESIS: By combining: (1) 12-cluster MBTI type detection, (2) ensemble models with
different focuses (balanced/ambiguous/confident), (3) Optuna-optimized thresholds, and
(4) special handling for marker values and uncertainty patterns, we can correctly
classify the 2.43% ambiguous cases

EXPECTED: Achieve >97.57% accuracy by identifying ambiguous clusters (40-60% Extrovert),
applying 10x weight to ambiguous cases during training, and using optimized thresholds
(~0.45 for ambiguous, ~0.15 for confidence)

RESULT: Implemented comprehensive BreakthroughEnsemble class that finds personality
clusters, trains 3 specialized models, optimizes decision thresholds with Optuna,
and applies the 96.2% rule for high-uncertainty cases near probability 0.5
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)

def preprocess_data(df):
    """Preprocess data - handle categorical columns."""
    df_processed = df.copy()
    
    # Handle categorical columns
    categorical_mappings = {
        'Stage_fear': {'No': 0, 'Yes': 1},
        'Drained_after_socializing': {'No': 0, 'Yes': 1}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)
    
    # Handle Target column if exists
    if 'Personality' in df_processed.columns:
        df_processed['Target'] = (df_processed['Personality'] == 'Extrovert').astype(int)
    
    return df_processed

def create_comprehensive_features(df):
    """Create all features including ambiguity detection."""
    features = df.copy()
    
    # Known ambiguous markers
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    
    # Marker detection
    features['marker_count'] = 0
    for col, val in markers.items():
        if col in features.columns:
            features[f'has_marker_{col}'] = (np.abs(features[col] - val) < 1e-6).astype(int)
            features['marker_count'] += features[f'has_marker_{col}']
    
    # Behavioral scores
    features['introvert_score'] = (
        features['Time_spent_Alone'] * features['Drained_after_socializing'] +
        features['Stage_fear']
    )
    
    features['extrovert_score'] = (
        features['Social_event_attendance'] + features['Going_outside'] +
        features['Friends_circle_size'] / 2
    )
    
    # Ambiguity indicators
    features['personality_balance'] = np.abs(
        features['introvert_score'] - features['extrovert_score']
    )
    
    # Key ambiguous patterns
    features['low_alone_moderate_social'] = (
        (features['Time_spent_Alone'] < 2.5) & 
        (features['Social_event_attendance'].between(3, 4))
    ).astype(int)
    
    features['medium_friends_low_energy'] = (
        (features['Friends_circle_size'].between(6, 7)) &
        (features['Drained_after_socializing'] == 0)
    ).astype(int)
    
    # Interaction features
    features['social_consistency'] = np.abs(
        features['Social_event_attendance'] - features['Going_outside']
    )
    
    features['social_media_alignment'] = (
        features['Post_frequency'] / (features['Social_event_attendance'] + 1)
    )
    
    # MBTI-inspired dimensions
    features['E_I_dimension'] = (
        features['extrovert_score'] - features['introvert_score']
    )
    
    features['social_energy_ratio'] = (
        features['Social_event_attendance'] / (features['Time_spent_Alone'] + 1)
    )
    
    return features

def identify_ambiguous_cases(X, features_df):
    """Identify ambiguous cases using multiple criteria."""
    ambiguous = np.zeros(len(X), dtype=bool)
    
    # Criterion 1: Has marker values
    if 'marker_count' in features_df.columns:
        ambiguous |= (features_df['marker_count'] > 0).values
    
    # Criterion 2: Specific behavioral pattern
    if 'low_alone_moderate_social' in features_df.columns:
        ambiguous |= (features_df['low_alone_moderate_social'] == 1).values
    
    # Criterion 3: Low personality balance (neither clearly I nor E)
    if 'personality_balance' in features_df.columns:
        ambiguous |= (features_df['personality_balance'] < 5).values
    
    return ambiguous

class BreakthroughEnsemble:
    """Ensemble specifically designed to break the 0.975708 barrier."""
    
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler()
        self.kmeans = None
        self.ambiguous_threshold = 0.45  # Default, will be optimized
        self.confidence_threshold = 0.15
        
    def _create_base_model(self, seed, focus='balanced'):
        """Create XGBoost model with different focus."""
        params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': seed,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'tree_method': 'gpu_hist'
        }
        
        # Adjust parameters based on focus
        if focus == 'ambiguous':
            params['max_depth'] = 8
            params['reg_alpha'] = 5
            params['reg_lambda'] = 2
        elif focus == 'confident':
            params['max_depth'] = 5
            params['reg_alpha'] = 1
            params['reg_lambda'] = 1
        
        return xgb.XGBClassifier(**params)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train ensemble with focus on ambiguous cases."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Find clusters (potential MBTI types)
        print("Finding personality clusters...")
        self.kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_train_scaled)
        
        # Analyze cluster distributions
        print("\nCluster analysis:")
        ambiguous_clusters = []
        for i in range(12):
            mask = clusters == i
            if mask.sum() > 20:
                extrovert_ratio = y_train[mask].mean()
                print(f"  Cluster {i}: {mask.sum()} samples, {extrovert_ratio:.1%} Extrovert")
                if 0.4 < extrovert_ratio < 0.6:
                    ambiguous_clusters.append(i)
        
        print(f"\nAmbiguous clusters: {ambiguous_clusters}")
        
        # Train multiple models with different perspectives
        print("\nTraining ensemble models...")
        
        # Model 1: Standard balanced
        model1 = self._create_base_model(42, 'balanced')
        if X_val is not None:
            model1.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      early_stopping_rounds=50, 
                      verbose=False)
        else:
            model1.fit(X_train, y_train)
        self.models.append(('balanced', model1))
        
        # Model 2: Focus on ambiguous cases
        # Create sample weights
        ambiguous_mask = np.isin(clusters, ambiguous_clusters)
        weights = np.ones(len(y_train))
        weights[ambiguous_mask] = 10  # Much higher weight for ambiguous
        
        model2 = self._create_base_model(123, 'ambiguous')
        if X_val is not None:
            model2.fit(X_train, y_train, 
                      sample_weight=weights,
                      eval_set=[(X_val, y_val)], 
                      early_stopping_rounds=50, 
                      verbose=False)
        else:
            model2.fit(X_train, y_train, sample_weight=weights)
        self.models.append(('ambiguous_focus', model2))
        
        # Model 3: Conservative (high confidence)
        model3 = self._create_base_model(456, 'confident')
        if X_val is not None:
            model3.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      early_stopping_rounds=50, 
                      verbose=False)
        else:
            model3.fit(X_train, y_train)
        self.models.append(('confident', model3))
        
        # Optimize thresholds if validation data available
        if X_val is not None and y_val is not None:
            self._optimize_thresholds(X_val, y_val)
    
    def _optimize_thresholds(self, X_val, y_val):
        """Optimize decision thresholds using Optuna."""
        print("\nOptimizing thresholds...")
        
        def objective(trial):
            # Thresholds to optimize
            ambig_thresh = trial.suggest_float('ambiguous_threshold', 0.35, 0.50)
            conf_thresh = trial.suggest_float('confidence_threshold', 0.10, 0.25)
            
            # Make predictions with these thresholds
            predictions = self._predict_with_thresholds(
                X_val, ambig_thresh, conf_thresh
            )
            
            return accuracy_score(y_val, predictions)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        # Store best thresholds
        self.ambiguous_threshold = study.best_params['ambiguous_threshold']
        self.confidence_threshold = study.best_params['confidence_threshold']
        
        print(f"  Best thresholds: ambiguous={self.ambiguous_threshold:.3f}, "
              f"confidence={self.confidence_threshold:.3f}")
        print(f"  Validation accuracy: {study.best_value:.6f}")
    
    def _predict_with_thresholds(self, X, ambig_thresh, conf_thresh):
        """Make predictions with specific thresholds."""
        # Get predictions from all models
        X_scaled = self.scaler.transform(X)
        
        probas = []
        for name, model in self.models:
            probas.append(model.predict_proba(X)[:, 1])
        
        # Ensemble probabilities
        probas_array = np.array(probas)
        mean_proba = np.mean(probas_array, axis=0)
        std_proba = np.std(probas_array, axis=0)
        
        # Identify ambiguous cases
        clusters = self.kmeans.predict(X_scaled)
        predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            # High uncertainty or disagreement
            if std_proba[i] > conf_thresh:
                # Apply special threshold for uncertain cases
                predictions[i] = int(mean_proba[i] > ambig_thresh)
            else:
                # Standard threshold for confident predictions
                predictions[i] = int(mean_proba[i] > 0.5)
            
            # Override for very low confidence near 0.5
            if abs(mean_proba[i] - 0.5) < 0.05:
                # Apply 96.2% rule - most ambiguous are Extrovert
                predictions[i] = 1
        
        return predictions
    
    def predict(self, X):
        """Make final predictions."""
        return self._predict_with_thresholds(
            X, self.ambiguous_threshold, self.confidence_threshold
        )
    
    def predict_proba(self, X):
        """Get probability predictions."""
        probas = []
        for name, model in self.models:
            probas.append(model.predict_proba(X)[:, 1])
        
        return np.mean(probas, axis=0)

def main():
    """Execute the breakthrough strategy."""
    print("="*80)
    print("COMBINED BREAKTHROUGH STRATEGY")
    print("="*80)
    print()
    
    # Load and preprocess data
    print("Loading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    test_ids = test_df['id']
    
    # Create features
    print("Creating comprehensive features...")
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                    'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    train_features = create_comprehensive_features(train_df[feature_cols])
    test_features = create_comprehensive_features(test_df[feature_cols])
    
    # Prepare data
    X = train_features.values
    y = train_df['Target'].values
    X_test = test_features.values
    
    # Identify ambiguous cases
    print("\nAnalyzing ambiguous cases...")
    train_ambiguous = identify_ambiguous_cases(X, train_features)
    test_ambiguous = identify_ambiguous_cases(X_test, test_features)
    
    print(f"Training ambiguous cases: {train_ambiguous.sum()} ({train_ambiguous.mean():.2%})")
    print(f"Test ambiguous cases: {test_ambiguous.sum()} ({test_ambiguous.mean():.2%})")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train breakthrough ensemble
    print("\nTraining breakthrough ensemble...")
    ensemble = BreakthroughEnsemble()
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)
    
    # Special handling for highly ambiguous cases
    print("\nApplying special rules for ambiguous cases...")
    special_handled = 0
    
    for i in range(len(predictions)):
        if test_ambiguous[i]:
            # Check confidence
            confidence = abs(probabilities[i] - 0.5)
            
            if confidence < 0.1:  # Very uncertain
                # Apply 96.2% rule
                if predictions[i] == 0:
                    predictions[i] = 1
                    special_handled += 1
            elif probabilities[i] < 0.2 and test_features.iloc[i]['marker_count'] > 0:
                # Exception: might be rare introvert ambivert
                predictions[i] = 0
    
    print(f"  Applied special handling to {special_handled} cases")
    
    # Save submission
    print(f"\nPrediction distribution:")
    print(f"  Introverts: {(predictions == 0).sum()} ({(predictions == 0).mean():.2%})")
    print(f"  Extroverts: {(predictions == 1).sum()} ({(predictions == 1).mean():.2%})")
    
    submission = pd.DataFrame({
        'id': test_ids,
        'Target': predictions.astype(int)
    })
    
    submission_path = 'submission_breakthrough_combined.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    
    # Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION PERFORMANCE")
    print("="*60)
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Train fold ensemble
        fold_ensemble = BreakthroughEnsemble()
        fold_ensemble.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
        
        # Predict
        fold_predictions = fold_ensemble.predict(X_fold_val)
        fold_score = accuracy_score(y_fold_val, fold_predictions)
        cv_scores.append(fold_score)
        print(f"  Fold {fold+1}: {fold_score:.6f}")
    
    print(f"\nMean CV Score: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores)*2:.6f})")
    
    # Analyze edge cases
    print("\n" + "="*60)
    print("EDGE CASE ANALYSIS")
    print("="*60)
    
    # Find cases with extreme uncertainty
    extreme_uncertain = (np.abs(probabilities - 0.5) < 0.02)
    print(f"\nExtremely uncertain predictions: {extreme_uncertain.sum()}")
    
    if extreme_uncertain.sum() > 0:
        print("Examples of extreme uncertainty:")
        uncertain_indices = np.where(extreme_uncertain)[0][:5]
        for idx in uncertain_indices:
            print(f"  Sample {idx}: prob={probabilities[idx]:.4f}, "
                  f"predicted={predictions[idx]}, "
                  f"ambiguous={test_ambiguous[idx]}")

if __name__ == "__main__":
    main()