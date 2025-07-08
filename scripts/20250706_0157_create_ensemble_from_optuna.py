#!/usr/bin/env python3
"""
Create ensemble from best models in Optuna studies.
Uses SQLite databases to extract best parameters and create optimal ensemble.
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
import time
import json
import hashlib
import pickle
import joblib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
import logging

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
CORRECTED_DATA_DIR = Path("output/")
STUDIES_DIR = Path("output/optuna_studies")
MODELS_DIR = Path("output/saved_models")
ENSEMBLE_DIR = Path("output/ensemble_results")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scores")
LOGS_DIR = Path("output/optimization_logs")

# Create directories
for dir_path in [MODELS_DIR, ENSEMBLE_DIR, SCORES_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Corrected datasets
CORRECTED_DATASETS = [
    "train_corrected_01.csv",  # 78 extreme introverts
    "train_corrected_02.csv",  # 81 (78+3)
    "train_corrected_03.csv",  # 6 negative typicality
    "train_corrected_04.csv",  # 192 psychological contradictions
    "train_corrected_05.csv",  # 34 eleven-hour extroverts
    "train_corrected_06.csv",  # 192 conservative combined
    "train_corrected_07.csv",  # 218 comprehensive
    "train_corrected_08.csv",  # 3 ultra-conservative
]

# Competition parameters
TARGET_COLUMN = "Personality"
ID_COLUMN = "id"
TARGET_SCORE = 0.975708
CV_FOLDS = 5
CV_RANDOM_STATE = 42

# Enhanced ambiguous samples
MOST_AMBIGUOUS = DATA_DIR / "enhanced_ambiguous_600.csv"

# Console for Rich output
console = Console()

# Logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = LOGS_DIR / f"ensemble_creation_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ModelResult:
    dataset: str
    model_type: str
    score: float
    params: Dict[str, Any]
    study_name: str
    n_trials: int

@dataclass
class EnsembleResult:
    weights: np.ndarray
    oof_score: float
    models: List[str]
    test_predictions: np.ndarray

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_study_name(model_name: str, dataset_name: str) -> str:
    """Generate study name (same as in optimization script)"""
    base = f"{model_name}_{dataset_name}_{CV_FOLDS}fold"
    return hashlib.md5(base.encode()).hexdigest()[:12]

def extract_best_results_from_studies() -> List[ModelResult]:
    """Extract best results from all Optuna studies"""
    results = []
    
    console.print(Panel.fit("📊 Extracting Best Results from Optuna Studies", 
                           style="bold cyan"))
    
    for dataset in CORRECTED_DATASETS:
        for model_type in ['xgb', 'gbm', 'cat']:
            study_name = get_study_name(model_type, dataset)
            db_path = STUDIES_DIR / f"{study_name}.db"
            
            if not db_path.exists():
                continue
                
            try:
                storage = f"sqlite:///{db_path}"
                study = optuna.load_study(study_name=study_name, storage=storage)
                
                if len(study.trials) > 0:
                    result = ModelResult(
                        dataset=dataset,
                        model_type=model_type,
                        score=study.best_value,
                        params=study.best_params,
                        study_name=study_name,
                        n_trials=len(study.trials)
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"Could not load study {study_name}: {e}")
    
    # Sort by score
    results.sort(key=lambda x: x.score, reverse=True)
    
    return results

def create_results_table(results: List[ModelResult]) -> Table:
    """Create Rich table with results"""
    table = Table(title="🏆 Best Models from Optimization", box=box.ROUNDED)
    
    table.add_column("Dataset", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("Trials", style="yellow")
    table.add_column("Gap to Target", style="red")
    
    for result in results[:15]:  # Show top 15
        gap = TARGET_SCORE - result.score
        gap_str = f"{gap:.6f}" if gap > 0 else f"[bold green]+{-gap:.6f}[/bold green]"
        
        table.add_row(
            result.dataset.replace("train_corrected_", "tc"),
            result.model_type.upper(),
            f"{result.score:.6f}",
            str(result.n_trials),
            gap_str
        )
    
    return table

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================

def create_features(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Create features including engineered ones from optimization"""
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
    
    # Ambiguity features
    if 'alone_thresh' in params:
        df['ambiguous_pattern'] = (
            (df['Time_spent_Alone'] < params.get('alone_thresh', 2.5)) & 
            (df['Social_event_attendance'] < params.get('social_thresh', 4.5)) &
            (df['Friends_circle_size'] < params.get('friends_thresh', 7))
        ).astype(float)
    
    # Distance to ambiguous centroid
    df['dist_to_ambiguous'] = np.sqrt(
        (df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].mean()) - 1.85)**2 +
        (df['Social_event_attendance'].fillna(df['Social_event_attendance'].mean()) - 3.75)**2 +
        (df['Friends_circle_size'].fillna(df['Friends_circle_size'].mean()) - 6.5)**2
    )
    
    df['ambiguity_score'] = 1 / (1 + df['dist_to_ambiguous'])
    df['high_ambiguity'] = (df['ambiguity_score'] > params.get('ambig_score_thresh', 0.3)).astype(float)
    
    # Interaction features
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

def load_and_preprocess_data(dataset_name: str, params: Dict[str, Any]) -> Tuple:
    """Load and preprocess dataset with all features"""
    # Load data
    train_path = CORRECTED_DATA_DIR / dataset_name
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Features
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency']
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    
    # Class-aware imputation for numerical features
    for col in numerical_cols:
        if col in train_df.columns:
            intro_mean = train_df[train_df[TARGET_COLUMN] == 'Introvert'][col].mean()
            extro_mean = train_df[train_df[TARGET_COLUMN] == 'Extrovert'][col].mean()
            
            # For test set, use overall mean
            test_mean = train_df[col].mean()
            test_df[col] = test_df[col].fillna(test_mean)
            
            # For train, use class means
            intro_mask = train_df[TARGET_COLUMN] == 'Introvert'
            extro_mask = train_df[TARGET_COLUMN] == 'Extrovert'
            
            train_df.loc[intro_mask, col] = train_df.loc[intro_mask, col].fillna(intro_mean)
            train_df.loc[extro_mask, col] = train_df.loc[extro_mask, col].fillna(extro_mean)
    
    # Convert categorical
    for col in categorical_cols:
        if col in train_df.columns:
            # Map before filling
            mapping = {'Yes': 1, 'No': 0}
            train_df[col] = train_df[col].map(mapping)
            test_df[col] = test_df[col].map(mapping)
            
            # Fill with mode
            mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 0.5
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)
    
    # Create features
    train_df = create_features(train_df, params)
    test_df = create_features(test_df, params)
    
    # Target
    train_df['target'] = (train_df[TARGET_COLUMN] == 'Extrovert').astype(int)
    
    # All feature columns
    feature_cols = (numerical_cols + categorical_cols + 
                   [f'{col}_is_null' for col in numerical_cols] +
                   ['null_count', 'has_nulls', 'weighted_null_score',
                    'dist_to_ambiguous', 'ambiguity_score', 'high_ambiguity',
                    'social_interaction', 'introvert_score'])
    
    # Add ambiguous pattern if exists
    if 'ambiguous_pattern' in train_df.columns:
        feature_cols.append('ambiguous_pattern')
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in train_df.columns]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    
    return X_train, y_train, X_test, train_df, test_df

# ==============================================================================
# MODEL CREATION AND TRAINING
# ==============================================================================

def create_model_from_params(model_type: str, params: Dict[str, Any]):
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
        
        # Try GPU first, fall back to CPU
        try:
            model_params['task_type'] = 'GPU'
            model_params['devices'] = '0'
            return cb.CatBoostClassifier(**model_params)
        except:
            model_params.pop('task_type', None)
            model_params.pop('devices', None)
            return cb.CatBoostClassifier(**model_params)

def train_model_full(model_result: ModelResult, X_train, y_train, X_test):
    """Train model on full dataset and return predictions"""
    console.print(f"Training {model_result.model_type} on {model_result.dataset}...")
    
    model = create_model_from_params(model_result.model_type, model_result.params)
    
    # Sample weights
    sample_weights = np.ones(len(y_train))
    
    # Load ambiguous IDs if available
    if MOST_AMBIGUOUS.exists() and 'ambig_weight' in model_result.params:
        ambiguous_df = pd.read_csv(MOST_AMBIGUOUS)
        ambiguous_ids = set(ambiguous_df['id'].values)
        train_ids = X_train.index
        ambiguous_mask = train_ids.isin(ambiguous_ids)
        sample_weights[ambiguous_mask] = model_result.params['ambig_weight']
    
    # Train model
    if model_result.model_type == 'cat':
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Get predictions
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    
    # Calculate training score
    train_score = accuracy_score(y_train, (train_pred > 0.5).astype(int))
    console.print(f"  Training score: {train_score:.6f}")
    
    return model, test_pred

# ==============================================================================
# OUT-OF-FOLD PREDICTIONS
# ==============================================================================

def get_oof_predictions(model_results: List[ModelResult], progress) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Get out-of-fold predictions for ensemble optimization"""
    all_oof_preds = []
    all_y_true = []
    
    task = progress.add_task("[cyan]Generating OOF predictions...", total=len(model_results))
    
    for model_result in model_results:
        # Load data
        X_train, y_train, X_test, train_df, test_df = load_and_preprocess_data(
            model_result.dataset, model_result.params
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
        oof_preds = np.zeros(len(y_train))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Create model
            model = create_model_from_params(model_result.model_type, model_result.params)
            
            # Sample weights
            sample_weights = np.ones(len(train_idx))
            if MOST_AMBIGUOUS.exists() and 'ambig_weight' in model_result.params:
                ambiguous_df = pd.read_csv(MOST_AMBIGUOUS)
                ambiguous_ids = set(ambiguous_df['id'].values)
                train_ids = X_fold_train.index
                ambiguous_mask = train_ids.isin(ambiguous_ids)
                sample_weights[ambiguous_mask] = model_result.params['ambig_weight']
            
            # Train
            if model_result.model_type == 'cat':
                model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            else:
                model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            
            # Predict
            oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        
        all_oof_preds.append(oof_preds)
        all_y_true.append(y_train.values)
        
        # OOF score
        oof_score = accuracy_score(y_train, (oof_preds > 0.5).astype(int))
        logger.info(f"OOF Score for {model_result.dataset} ({model_result.model_type}): {oof_score:.6f}")
        
        progress.update(task, advance=1)
    
    return all_oof_preds, all_y_true

# ==============================================================================
# ENSEMBLE OPTIMIZATION
# ==============================================================================

def optimize_ensemble_weights(oof_predictions: List[np.ndarray], y_true_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Find optimal ensemble weights using OOF predictions"""
    n_models = len(oof_predictions)
    
    # Use only the first y_true since all should be identical
    # (all models trained on same dataset size)
    y_true = y_true_list[0]
    
    # Stack predictions as columns
    all_preds = np.column_stack(oof_predictions)
    
    def ensemble_score(weights):
        weights = weights / weights.sum()
        ensemble_pred = np.average(all_preds, weights=weights, axis=1)
        score = accuracy_score(y_true, (ensemble_pred > 0.5).astype(int))
        return -score  # Minimize negative score
    
    # Optimize
    initial_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    
    console.print("\n[yellow]Optimizing ensemble weights...[/yellow]")
    
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

# ==============================================================================
# MAIN ENSEMBLE FUNCTION
# ==============================================================================

def create_ensemble():
    """Create ensemble from best models"""
    console.print(Panel.fit("🚀 Creating Ensemble from Optuna Studies", 
                           style="bold magenta"))
    
    # Extract best results
    all_results = extract_best_results_from_studies()
    
    if not all_results:
        console.print("[red]No Optuna studies found![/red]")
        return
    
    # Display results table
    console.print("\n")
    console.print(create_results_table(all_results))
    
    # Select top models for ensemble (score >= 0.97)
    selected_results = [r for r in all_results if r.score >= 0.970]
    
    # Further filter to get best model per dataset
    best_per_dataset = {}
    for result in selected_results:
        if result.dataset not in best_per_dataset or result.score > best_per_dataset[result.dataset].score:
            best_per_dataset[result.dataset] = result
    
    ensemble_models = list(best_per_dataset.values())
    console.print(f"\n[green]Selected {len(ensemble_models)} models for ensemble[/green]")
    
    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Get OOF predictions
        oof_predictions, y_true_list = get_oof_predictions(ensemble_models, progress)
        
        # Optimize weights
        if len(ensemble_models) > 1:
            optimal_weights, oof_score = optimize_ensemble_weights(oof_predictions, y_true_list)
            console.print(f"\n[green]Optimal OOF Score: {oof_score:.6f}[/green]")
        else:
            optimal_weights = np.array([1.0])
            oof_score = 0.0
        
        # Train final models and get test predictions
        test_predictions = []
        model_names = []
        
        train_task = progress.add_task("[cyan]Training final models...", total=len(ensemble_models))
        
        for i, model_result in enumerate(ensemble_models):
            # Load data
            X_train, y_train, X_test, train_df, test_df = load_and_preprocess_data(
                model_result.dataset, model_result.params
            )
            
            # Train model
            model, test_pred = train_model_full(model_result, X_train, y_train, X_test)
            
            test_predictions.append(test_pred)
            model_names.append(f"{model_result.model_type}_{model_result.dataset}")
            
            # Save model
            model_path = MODELS_DIR / f"{model_result.model_type}_{model_result.dataset.replace('.csv', '')}_{timestamp}.pkl"
            joblib.dump(model, model_path)
            
            progress.update(train_task, advance=1)
    
    # Create ensemble predictions
    console.print("\n[yellow]Generating ensemble submissions...[/yellow]")
    
    test_pred_array = np.column_stack(test_predictions)
    
    # Load test data for IDs
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # 1. Weighted average ensemble
    ensemble_pred_proba = np.average(test_pred_array, weights=optimal_weights, axis=1)
    
    # Try different thresholds
    thresholds = [0.48, 0.49, 0.495, 0.50, 0.505, 0.51, 0.52]
    
    best_models_str = "_".join([f"{r.model_type}{r.dataset.replace('train_corrected_', 'tc').replace('.csv', '')}" 
                               for r in ensemble_models[:3]])
    
    for threshold in thresholds:
        ensemble_pred = (ensemble_pred_proba > threshold).astype(int)
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': ensemble_pred
        })
        submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
        
        # Save with expected score in filename
        expected_score = oof_score if threshold == 0.5 else oof_score - 0.0001
        filename = f"subm-{expected_score:.6f}-{timestamp}-ensemble_t{threshold}-{best_models_str}.csv"
        submission.to_csv(SCORES_DIR / filename, index=False)
        console.print(f"  Saved: {filename}")
    
    # 2. Apply post-processing for known issues
    # IDs with 11-hour alone time that are labeled as extroverts
    problem_ids = [18876, 20363, 20934, 20950, 21008]
    
    ensemble_pp = (ensemble_pred_proba > 0.5).astype(int)
    flipped = 0
    
    for problem_id in problem_ids:
        idx = test_df[test_df['id'] == problem_id].index
        if len(idx) > 0 and ensemble_pp[idx[0]] == 1:  # If predicted extrovert
            ensemble_pp[idx[0]] = 0  # Change to introvert
            flipped += 1
    
    if flipped > 0:
        submission_pp = pd.DataFrame({
            'id': test_df['id'],
            'Personality': ensemble_pp
        })
        submission_pp['Personality'] = submission_pp['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
        
        filename = f"subm-{oof_score:.6f}-{timestamp}-ensemble_postproc-{best_models_str}.csv"
        submission_pp.to_csv(SCORES_DIR / filename, index=False)
        console.print(f"  Saved: {filename} (flipped {flipped} predictions)")
    
    # Save ensemble details
    details = {
        'models': [
            {
                'dataset': r.dataset,
                'model_type': r.model_type,
                'score': r.score,
                'weight': float(optimal_weights[i])
            }
            for i, r in enumerate(ensemble_models)
        ],
        'oof_score': oof_score,
        'optimal_weights': optimal_weights.tolist(),
        'timestamp': timestamp,
        'best_individual_score': max(r.score for r in ensemble_models),
        'gap_to_target': TARGET_SCORE - max(r.score for r in ensemble_models)
    }
    
    with open(ENSEMBLE_DIR / f'ensemble_details_{timestamp}.json', 'w') as f:
        json.dump(details, f, indent=2)
    
    # Final summary
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        f"[bold green]✅ Ensemble Complete![/bold green]\n\n"
        f"Best Individual Model: {max(r.score for r in ensemble_models):.6f}\n"
        f"Ensemble OOF Score: {oof_score:.6f}\n"
        f"Target Score: {TARGET_SCORE:.6f}\n"
        f"Gap: {TARGET_SCORE - max(r.score for r in ensemble_models):.6f}\n\n"
        f"Submissions saved to: {SCORES_DIR}",
        style="green"
    ))
    
    # Display weights table
    weights_table = Table(title="Ensemble Weights", box=box.SIMPLE)
    weights_table.add_column("Model", style="cyan")
    weights_table.add_column("Dataset", style="magenta")
    weights_table.add_column("Weight", style="green")
    
    for i, (model_result, weight) in enumerate(zip(ensemble_models, optimal_weights)):
        weights_table.add_row(
            model_result.model_type.upper(),
            model_result.dataset.replace("train_corrected_", "tc"),
            f"{weight:.4f}"
        )
    
    console.print("\n")
    console.print(weights_table)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main entry point"""
    try:
        create_ensemble()
    except KeyboardInterrupt:
        console.print("\n[yellow]Ensemble creation interrupted by user[/yellow]")
        logger.info("Ensemble creation interrupted")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error(f"Error during ensemble creation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()