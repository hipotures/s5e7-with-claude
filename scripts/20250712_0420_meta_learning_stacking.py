#!/usr/bin/env python3
"""
Meta-learning approach using stacking ensemble
Train multiple diverse models and use meta-model to combine predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def prepare_data():
    """Load and prepare data"""
    print("="*60)
    print("META-LEARNING WITH STACKING")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"\nOriginal shapes:")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    
    # Encode target
    le = LabelEncoder()
    train_df['target'] = le.fit_transform(train_df['Personality'])
    
    # Features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Handle categorical features
    for col in ['Stage_fear', 'Drained_after_socializing']:
        train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})
        test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})
    
    # Handle missing values
    for col in feature_cols:
        if train_df[col].isnull().any():
            # Fill with median for numeric columns
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    
    return X_train, y_train, X_test, train_df, test_df, le

def create_base_models():
    """Create diverse base models"""
    
    base_models = {
        # Tree-based models
        'rf_100': RandomForestClassifier(n_estimators=100, random_state=42),
        'rf_300': RandomForestClassifier(n_estimators=300, random_state=42),
        'rf_deep': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        
        # Boosting models
        'xgb': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            device='cuda:0',
            tree_method='hist',
            random_state=42
        ),
        
        'lgb': lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            device='gpu',
            random_state=42,
            verbose=-1
        ),
        
        'gb': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        
        # Other models
        'svm': SVC(probability=True, random_state=42),
        'nb': GaussianNB(),
        'knn3': KNeighborsClassifier(n_neighbors=3),
        'knn7': KNeighborsClassifier(n_neighbors=7),
        
        # Logistic regression variants
        'lr': LogisticRegression(random_state=42, max_iter=1000),
        'lr_l1': LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
    }
    
    return base_models

def train_base_models_cv(X_train, y_train, base_models, n_folds=5):
    """Train base models using cross-validation and generate meta-features"""
    
    print("\n" + "="*60)
    print("TRAINING BASE MODELS WITH CROSS-VALIDATION")
    print("="*60)
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for meta-features
    n_models = len(base_models)
    meta_train = np.zeros((len(X_train), n_models * 2))  # probabilities for both classes
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        
        for model_idx, (model_name, model) in enumerate(base_models.items()):
            print(f"  Training {model_name}...", end=' ')
            
            # Clone model
            model_clone = model.__class__(**model.get_params())
            
            # Train
            model_clone.fit(X_fold_train, y_fold_train)
            
            # Predict probabilities
            fold_proba = model_clone.predict_proba(X_fold_val)
            
            # Store in meta features
            meta_train[val_idx, model_idx*2:(model_idx+1)*2] = fold_proba
            
            # Calculate fold accuracy
            fold_pred = fold_proba.argmax(axis=1)
            fold_acc = (fold_pred == y_train[val_idx]).mean()
            print(f"Accuracy: {fold_acc:.4f}")
    
    return meta_train

def train_final_base_models(X_train, y_train, X_test, base_models):
    """Train base models on full training data and get test predictions"""
    
    print("\n" + "="*60)
    print("TRAINING FINAL BASE MODELS")
    print("="*60)
    
    n_models = len(base_models)
    meta_test = np.zeros((len(X_test), n_models * 2))
    
    trained_models = {}
    
    for model_idx, (model_name, model) in enumerate(base_models.items()):
        print(f"Training {model_name} on full data...", end=' ')
        
        # Train on full data
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Predict test probabilities
        test_proba = model.predict_proba(X_test)
        meta_test[:, model_idx*2:(model_idx+1)*2] = test_proba
        
        # Training accuracy
        train_pred = model.predict(X_train)
        train_acc = (train_pred == y_train).mean()
        print(f"Train accuracy: {train_acc:.4f}")
    
    return trained_models, meta_test

def train_meta_model(meta_train, y_train, meta_test):
    """Train meta-model on base model predictions"""
    
    print("\n" + "="*60)
    print("TRAINING META-MODEL")
    print("="*60)
    
    # Try different meta-models
    meta_models = {
        'lr_meta': LogisticRegression(random_state=42, max_iter=1000),
        'rf_meta': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgb_meta': xgb.XGBClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            device='cuda:0',
            tree_method='hist',
            random_state=42
        )
    }
    
    best_score = 0
    best_meta_model = None
    best_meta_name = None
    
    # Cross-validate meta-models
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for meta_name, meta_model in meta_models.items():
        scores = []
        
        for train_idx, val_idx in kf.split(meta_train, y_train):
            meta_model_clone = meta_model.__class__(**meta_model.get_params())
            meta_model_clone.fit(meta_train[train_idx], y_train[train_idx])
            
            val_pred = meta_model_clone.predict(meta_train[val_idx])
            val_score = (val_pred == y_train[val_idx]).mean()
            scores.append(val_score)
        
        avg_score = np.mean(scores)
        print(f"{meta_name}: CV accuracy = {avg_score:.6f} ± {np.std(scores):.6f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_meta_model = meta_model
            best_meta_name = meta_name
    
    print(f"\nBest meta-model: {best_meta_name} with CV accuracy {best_score:.6f}")
    
    # Train best meta-model on full meta-train data
    best_meta_model.fit(meta_train, y_train)
    
    # Final predictions
    final_predictions = best_meta_model.predict(meta_test)
    
    return best_meta_model, final_predictions

def analyze_model_contributions(meta_train, y_train, base_model_names):
    """Analyze how different models contribute to final predictions"""
    
    print("\n" + "="*60)
    print("MODEL CONTRIBUTION ANALYSIS")
    print("="*60)
    
    # Calculate correlation between base model predictions
    n_models = len(base_model_names)
    correlation_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            # Use probability of positive class
            prob_i = meta_train[:, i*2+1]
            prob_j = meta_train[:, j*2+1]
            correlation_matrix[i, j] = np.corrcoef(prob_i, prob_j)[0, 1]
    
    print("\nModel correlation matrix:")
    print("       ", end="")
    for name in base_model_names:
        print(f"{name[:6]:>8}", end="")
    print()
    
    for i, name1 in enumerate(base_model_names):
        print(f"{name1[:6]:>6} ", end="")
        for j, name2 in enumerate(base_model_names):
            if i == j:
                print("    1.00", end="")
            else:
                print(f"{correlation_matrix[i, j]:8.2f}", end="")
        print()
    
    # Find most and least correlated pairs
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    correlations = correlation_matrix[mask == 1]
    
    print(f"\nAverage correlation: {np.mean(correlations):.3f}")
    print(f"Min correlation: {np.min(correlations):.3f}")
    print(f"Max correlation: {np.max(correlations):.3f}")

def create_diverse_predictions(trained_models, X_test, test_df, le):
    """Create different ensemble predictions for testing"""
    
    print("\n" + "="*60)
    print("CREATING DIVERSE PREDICTIONS")
    print("="*60)
    
    predictions = {}
    
    # 1. Individual model predictions
    for model_name, model in trained_models.items():
        pred = model.predict(X_test)
        predictions[f'single_{model_name}'] = le.inverse_transform(pred)
    
    # 2. Simple majority voting
    all_preds = np.array([model.predict(X_test) for model in trained_models.values()])
    majority_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, all_preds)
    predictions['majority_vote'] = le.inverse_transform(majority_pred)
    
    # 3. Weighted average (by training accuracy)
    all_probas = []
    weights = []
    
    for model in trained_models.values():
        proba = model.predict_proba(X_test)[:, 1]
        all_probas.append(proba)
        # Simple equal weights for now
        weights.append(1.0)
    
    weights = np.array(weights) / np.sum(weights)
    weighted_proba = np.average(all_probas, axis=0, weights=weights)
    weighted_pred = (weighted_proba > 0.5).astype(int)
    predictions['weighted_avg'] = le.inverse_transform(weighted_pred)
    
    # Save best predictions
    for name, pred in predictions.items():
        if name in ['single_xgb', 'single_lgb', 'weighted_avg']:
            submission = pd.DataFrame({
                'id': test_df['id'],
                'Personality': pred
            })
            filename = f'submission_meta_{name}.csv'
            submission.to_csv(SCORES_DIR / filename, index=False)
            print(f"Created: {filename}")
    
    return predictions

def create_blended_submission(meta_test, trained_models, X_test, test_df, le):
    """Create blended submission using different strategies"""
    
    print("\n" + "="*60)
    print("CREATING BLENDED SUBMISSIONS")
    print("="*60)
    
    # Get probabilities from all models
    all_probas = []
    model_names = list(trained_models.keys())
    
    for model in trained_models.values():
        proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
        all_probas.append(proba)
    
    all_probas = np.array(all_probas)
    
    # Strategy 1: Trim mean (remove extreme predictions)
    trim_probas = []
    for i in range(len(X_test)):
        sample_probas = all_probas[:, i]
        # Remove highest and lowest
        trimmed = np.sort(sample_probas)[1:-1]
        trim_probas.append(np.mean(trimmed))
    
    trim_pred = (np.array(trim_probas) > 0.5).astype(int)
    
    # Strategy 2: Median prediction
    median_probas = np.median(all_probas, axis=0)
    median_pred = (median_probas > 0.5).astype(int)
    
    # Save submissions
    submissions = {
        'trim_mean': le.inverse_transform(trim_pred),
        'median': le.inverse_transform(median_pred)
    }
    
    for name, pred in submissions.items():
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': pred
        })
        filename = f'submission_blend_{name}.csv'
        submission.to_csv(SCORES_DIR / filename, index=False)
        print(f"Created: {filename}")

def main():
    # Prepare data
    X_train, y_train, X_test, train_df, test_df, le = prepare_data()
    
    # Create base models
    base_models = create_base_models()
    base_model_names = list(base_models.keys())
    
    # Train base models with CV to get meta-features
    meta_train = train_base_models_cv(X_train, y_train, base_models)
    
    # Train final base models and get test meta-features
    trained_models, meta_test = train_final_base_models(X_train, y_train, X_test, base_models)
    
    # Train meta-model
    meta_model, final_predictions = train_meta_model(meta_train, y_train, meta_test)
    
    # Convert predictions back to labels
    final_labels = le.inverse_transform(final_predictions)
    
    # Create main submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': final_labels
    })
    submission.to_csv(SCORES_DIR / 'submission_meta_stacking.csv', index=False)
    print(f"\nCreated main submission: submission_meta_stacking.csv")
    
    # Analyze model contributions
    analyze_model_contributions(meta_train, y_train, base_model_names)
    
    # Create diverse predictions
    create_diverse_predictions(trained_models, X_test, test_df, le)
    
    # Create blended submissions
    create_blended_submission(meta_test, trained_models, X_test, test_df, le)
    
    print("\n" + "="*60)
    print("META-LEARNING COMPLETE")
    print("="*60)
    print("\nSubmissions created:")
    print("- submission_meta_stacking.csv (main stacking ensemble)")
    print("- submission_meta_single_*.csv (individual models)")
    print("- submission_blend_*.csv (blended predictions)")

if __name__ == "__main__":
    main()