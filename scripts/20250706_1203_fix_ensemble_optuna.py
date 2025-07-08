#!/usr/bin/env python3
"""
Fixed ensemble creation from Optuna studies.
Properly handles OOF predictions for weight optimization.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import warnings
import os
import sys
import json
import hashlib
import joblib
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import logging

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
CORRECTED_DATA_DIR = Path("output/")
STUDIES_DIR = Path("output/optuna_studies")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scores")
LOGS_DIR = Path("output/optimization_logs")
MOST_AMBIGUOUS = DATA_DIR / "enhanced_ambiguous_600.csv"

# Configuration
TARGET_COLUMN = "Personality"
CV_FOLDS = 5
CV_RANDOM_STATE = 42
TARGET_SCORE = 0.975708

console = Console()

# Logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = LOGS_DIR / f"ensemble_fixed_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(log_file)]
)
logger = logging.getLogger(__name__)

def get_study_name(model_name: str, dataset_name: str) -> str:
    """Generate study name"""
    base = f"{model_name}_{dataset_name}_{CV_FOLDS}fold"
    return hashlib.md5(base.encode()).hexdigest()[:12]

def create_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Create all features including engineered ones"""
    df = df.copy()
    
    # Null pattern features
    null_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    for col in null_cols:
        df[f'{col}_is_null'] = df[col].isnull().astype(float)
    
    df['null_count'] = df[null_cols].isnull().sum(axis=1)
    df['has_nulls'] = (df['null_count'] > 0).astype(float)
    
    # Weighted null score
    null_weights = {
        'Time_spent_Alone': 2.5,
        'Social_event_attendance': 1.8,
        'Going_outside': 1.5,
        'Friends_circle_size': 1.2,
        'Post_frequency': 1.0
    }
    
    df['weighted_null_score'] = sum(
        df[f'{col}_is_null'] * weight 
        for col, weight in null_weights.items()
    )
    
    # Distance features
    df['dist_to_ambiguous'] = np.sqrt(
        (df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].mean()) - 1.85)**2 +
        (df['Social_event_attendance'].fillna(df['Social_event_attendance'].mean()) - 3.75)**2 +
        (df['Friends_circle_size'].fillna(df['Friends_circle_size'].mean()) - 6.5)**2
    )
    
    df['ambiguity_score'] = 1 / (1 + df['dist_to_ambiguous'])
    df['high_ambiguity'] = (df['ambiguity_score'] > params.get('ambig_score_thresh', 0.3)).astype(float)
    
    # Interactions
    df['social_interaction'] = (
        df['Social_event_attendance'].fillna(0) * 
        df['Friends_circle_size'].fillna(0)
    )
    
    df['introvert_score'] = (
        df['Time_spent_Alone'].fillna(5) + 
        (10 - df['Social_event_attendance'].fillna(5)) +
        (15 - df['Friends_circle_size'].fillna(7.5))
    ) / 3
    
    return df

def load_and_preprocess_data(dataset_name: str, params: dict):
    """Load and preprocess dataset"""
    # Load data
    train_path = CORRECTED_DATA_DIR / dataset_name
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Features
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency']
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    
    # Class-aware imputation
    for col in numerical_cols:
        if col in train_df.columns:
            intro_mean = train_df[train_df[TARGET_COLUMN] == 'Introvert'][col].mean()
            extro_mean = train_df[train_df[TARGET_COLUMN] == 'Extrovert'][col].mean()
            
            test_mean = train_df[col].mean()
            test_df[col] = test_df[col].fillna(test_mean)
            
            intro_mask = train_df[TARGET_COLUMN] == 'Introvert'
            extro_mask = train_df[TARGET_COLUMN] == 'Extrovert'
            
            train_df.loc[intro_mask, col] = train_df.loc[intro_mask, col].fillna(intro_mean)
            train_df.loc[extro_mask, col] = train_df.loc[extro_mask, col].fillna(extro_mean)
    
    # Convert categorical
    for col in categorical_cols:
        if col in train_df.columns:
            mapping = {'Yes': 1, 'No': 0}
            train_df[col] = train_df[col].map(mapping)
            test_df[col] = test_df[col].map(mapping)
            
            mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 0.5
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)
    
    # Create features
    train_df = create_features(train_df, params)
    test_df = create_features(test_df, params)
    
    # Target
    train_df['target'] = (train_df[TARGET_COLUMN] == 'Extrovert').astype(int)
    
    # Feature columns
    feature_cols = (numerical_cols + categorical_cols + 
                   [f'{col}_is_null' for col in numerical_cols] +
                   ['null_count', 'has_nulls', 'weighted_null_score',
                    'dist_to_ambiguous', 'ambiguity_score', 'high_ambiguity',
                    'social_interaction', 'introvert_score'])
    
    feature_cols = [col for col in feature_cols if col in train_df.columns]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    
    return X_train, y_train, X_test, test_df

def create_model_from_params(model_type: str, params: dict):
    """Create model instance from parameters"""
    if model_type == 'xgb':
        model_params = {
            'n_estimators': params.get('n_estimators', 1000),
            'learning_rate': params.get('learning_rate', 0.01),
            'max_depth': params.get('max_depth', 6),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'reg_lambda': params.get('reg_lambda', 1.0),
            'reg_alpha': params.get('reg_alpha', 0.0),
            'gamma': params.get('gamma', 0.0),
            'min_child_weight': params.get('min_child_weight', 1),
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'use_label_encoder': False,
            'random_state': CV_RANDOM_STATE,
            'n_jobs': 1
        }
        return xgb.XGBClassifier(**model_params)
        
    elif model_type == 'gbm':
        model_params = {
            'n_estimators': params.get('n_estimators', 1000),
            'learning_rate': params.get('learning_rate', 0.01),
            'max_depth': params.get('max_depth', -1),
            'num_leaves': params.get('num_leaves', 31),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'reg_lambda': params.get('reg_lambda', 1.0),
            'reg_alpha': params.get('reg_alpha', 0.0),
            'min_child_weight': params.get('min_child_weight', 1),
            'min_split_gain': params.get('min_split_gain', 0.0),
            'device': 'gpu',
            'gpu_device_id': 0,
            'objective': 'binary',
            'metric': 'binary_error',
            'verbosity': -1,
            'random_state': CV_RANDOM_STATE,
            'n_jobs': 1
        }
        return lgb.LGBMClassifier(**model_params)
        
    elif model_type == 'cat':
        model_params = {
            'iterations': params.get('n_estimators', 1000),
            'learning_rate': params.get('learning_rate', 0.03),
            'depth': params.get('max_depth', 6),
            'l2_leaf_reg': params.get('reg_lambda', 3.0),
            'border_count': params.get('border_count', 128),
            'random_strength': params.get('random_strength', 1.0),
            'bagging_temperature': params.get('bagging_temperature', 1.0),
            'thread_count': 16,
            'random_state': CV_RANDOM_STATE,
            'verbose': False
        }
        
        try:
            model_params['task_type'] = 'GPU'
            model_params['devices'] = '0'
            return cb.CatBoostClassifier(**model_params)
        except:
            model_params.pop('task_type', None)
            model_params.pop('devices', None)
            return cb.CatBoostClassifier(**model_params)

def optimize_ensemble_weights_fixed(model_predictions):
    """Fixed ensemble weight optimization using single dataset CV"""
    n_models = len(model_predictions)
    
    # Use first model's data to get ground truth (all should be same)
    _, y_train, _, _ = load_and_preprocess_data(
        model_predictions[0]['dataset'], 
        model_predictions[0]['params']
    )
    
    # Get all OOF predictions for the SAME dataset split
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    all_oof_preds = np.zeros((len(y_train), n_models))
    
    for i, model_info in enumerate(model_predictions):
        oof_pred = np.zeros(len(y_train))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y_train)), y_train)):
            # Load data for this model's dataset
            X_train_full, y_train_full, _, _ = load_and_preprocess_data(
                model_info['dataset'], 
                model_info['params']
            )
            
            X_fold_train = X_train_full.iloc[train_idx]
            y_fold_train = y_train_full.iloc[train_idx]
            X_fold_val = X_train_full.iloc[val_idx]
            
            # Create and train model
            model = create_model_from_params(model_info['model_type'], model_info['params'])
            
            # Sample weights
            sample_weights = np.ones(len(train_idx))
            if MOST_AMBIGUOUS.exists() and 'ambig_weight' in model_info['params']:
                ambiguous_df = pd.read_csv(MOST_AMBIGUOUS)
                ambiguous_ids = set(ambiguous_df['id'].values)
                train_ids = X_fold_train.index
                ambiguous_mask = train_ids.isin(ambiguous_ids)
                sample_weights[ambiguous_mask] = model_info['params']['ambig_weight']
            
            # Train
            if model_info['model_type'] == 'cat':
                model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            else:
                model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            
            # Predict
            oof_pred[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        
        all_oof_preds[:, i] = oof_pred
    
    # Optimize weights
    def ensemble_score(weights):
        weights = weights / weights.sum()
        ensemble_pred = np.average(all_oof_preds, weights=weights, axis=1)
        score = accuracy_score(y_train, (ensemble_pred > 0.5).astype(int))
        return -score
    
    initial_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    
    result = minimize(
        ensemble_score,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    optimal_weights = result.x / result.x.sum()
    optimal_score = -result.fun
    
    return optimal_weights, optimal_score

def main():
    """Create optimized ensemble"""
    console.print(Panel.fit("🚀 Fixed Ensemble Creation from Optuna Studies", 
                           style="bold magenta"))
    
    # Best models to ensemble
    best_models = [
        {"dataset": "train_corrected_07.csv", "model_type": "gbm", "cv_score": 0.975869},
        {"dataset": "train_corrected_07.csv", "model_type": "xgb", "cv_score": 0.975815},
        {"dataset": "train_corrected_07.csv", "model_type": "cat", "cv_score": 0.975815},
        {"dataset": "train_corrected_04.csv", "model_type": "cat", "cv_score": 0.974898},
        {"dataset": "train_corrected_06.csv", "model_type": "cat", "cv_score": 0.974844},
    ]
    
    # Load parameters from Optuna for each model
    for model_info in best_models:
        study_name = get_study_name(model_info['model_type'], model_info['dataset'])
        db_path = STUDIES_DIR / f"{study_name}.db"
        
        if db_path.exists():
            try:
                storage = f"sqlite:///{db_path}"
                study = optuna.load_study(study_name=study_name, storage=storage)
                model_info['params'] = study.best_params
                logger.info(f"Loaded params for {model_info['model_type']} on {model_info['dataset']}")
            except Exception as e:
                console.print(f"[red]Error loading study: {e}[/red]")
                return
    
    # Display models
    table = Table(title="Models for Ensemble", box=box.ROUNDED)
    table.add_column("Dataset", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("CV Score", style="green")
    
    for model in best_models:
        table.add_row(
            model['dataset'].replace('train_corrected_', 'tc'),
            model['model_type'].upper(),
            f"{model['cv_score']:.6f}"
        )
    
    console.print("\n")
    console.print(table)
    
    console.print("\n[yellow]Optimizing ensemble weights...[/yellow]")
    optimal_weights, oof_score = optimize_ensemble_weights_fixed(best_models)
    
    console.print(f"\n[green]Optimal weights: {optimal_weights}[/green]")
    console.print(f"[green]Ensemble OOF score: {oof_score:.6f}[/green]")
    
    # Train final models and generate ensemble predictions
    console.print("\n[cyan]Training final models...[/cyan]")
    
    test_predictions = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Training models...", total=len(best_models))
        
        for model_info in best_models:
            # Load data
            X_train, y_train, X_test, test_df = load_and_preprocess_data(
                model_info['dataset'], 
                model_info['params']
            )
            
            # Create and train model
            model = create_model_from_params(model_info['model_type'], model_info['params'])
            
            # Sample weights
            sample_weights = np.ones(len(y_train))
            if MOST_AMBIGUOUS.exists() and 'ambig_weight' in model_info['params']:
                ambiguous_df = pd.read_csv(MOST_AMBIGUOUS)
                ambiguous_ids = set(ambiguous_df['id'].values)
                train_ids = X_train.index
                ambiguous_mask = train_ids.isin(ambiguous_ids)
                sample_weights[ambiguous_mask] = model_info['params']['ambig_weight']
            
            # Train
            if model_info['model_type'] == 'cat':
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Predict
            test_pred = model.predict_proba(X_test)[:, 1]
            test_predictions.append(test_pred)
            
            progress.update(task, advance=1)
    
    # Create ensemble predictions
    test_pred_array = np.column_stack(test_predictions)
    ensemble_pred_proba = np.average(test_pred_array, weights=optimal_weights, axis=1)
    
    # Generate submissions with different thresholds
    thresholds = [0.48, 0.49, 0.495, 0.50, 0.505, 0.51]
    
    console.print("\n[yellow]Generating ensemble submissions...[/yellow]")
    
    # Use the first test_df (all should have same IDs)
    _, _, _, test_df = load_and_preprocess_data(best_models[0]['dataset'], best_models[0]['params'])
    
    for threshold in thresholds:
        ensemble_pred = (ensemble_pred_proba > threshold).astype(int)
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': ensemble_pred
        })
        submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
        
        # Save
        filename = f"subm-{oof_score:.6f}-{timestamp}-ensemble5-t{threshold}.csv"
        submission.to_csv(SCORES_DIR / filename, index=False)
        
        if threshold == 0.50:
            console.print(f"\n[green]✓ Main submission: {filename}[/green]")
    
    # Save ensemble details
    details = {
        'models': [
            {
                'dataset': m['dataset'],
                'model_type': m['model_type'],
                'cv_score': m['cv_score'],
                'weight': float(optimal_weights[i])
            }
            for i, m in enumerate(best_models)
        ],
        'oof_score': oof_score,
        'optimal_weights': optimal_weights.tolist(),
        'timestamp': timestamp
    }
    
    with open(LOGS_DIR / f'ensemble_details_{timestamp}.json', 'w') as f:
        json.dump(details, f, indent=2)
    
    console.print(f"\n[bold green]✅ Ensemble complete![/bold green]")
    console.print(f"Best individual CV: {max(m['cv_score'] for m in best_models):.6f}")
    console.print(f"Ensemble OOF: {oof_score:.6f}")
    console.print(f"Gap to target: {TARGET_SCORE - oof_score:.6f}")

if __name__ == "__main__":
    main()