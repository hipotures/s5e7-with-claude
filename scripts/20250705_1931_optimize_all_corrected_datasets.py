#!/usr/bin/env python3
"""
Optimizes all corrected datasets with time-based limits and rich monitoring.
Each dataset gets equal time for optimization across all models.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
import os
import sys
import time
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import multiprocessing as mp
from queue import Empty
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import logging
import threading

warnings.filterwarnings('ignore')

# Silence Optuna output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Redirect stdout for cleaner display
import io
import contextlib

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

# Time limit per dataset (in seconds)
TIME_PER_DATASET = 30  # 3600 seconds (1 hour) per dataset for production

# Corrected datasets to process
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

# Data paths
DATA_DIR = "/mnt/ml/kaggle/playground-series-s5e7/"
CORRECTED_DATA_DIR = "output/"
TEST_FILE = "test.csv"
MOST_AMBIGUOUS = "/mnt/ml/kaggle/playground-series-s5e7/enhanced_ambiguous_600.csv"

# Output directories
SCORES_DIR = "scores"
LOGS_DIR = "output/optimization_logs"
STUDIES_DIR = "output/optuna_studies"

# Resource assignment
MODEL_RESOURCE_ASSIGNMENT = {
    'xgb': ['gpu0', 'gpu1'],
    'gbm': ['gpu0', 'gpu1'],
    'cat': ['cpu']
}

# Model configuration
MODEL_CONFIG = {
    'xgb': {'enabled': True},
    'gbm': {'enabled': True},
    'cat': {'enabled': True}
}

# Cross-validation
CV_FOLDS = 5
CV_RANDOM_STATE = 42

# Competition parameters
TARGET_COLUMN = "Personality"
ID_COLUMN = "id"

AMBIGUOUS_FEATURES = {
    'numerical': ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency'],
    'categorical': ['Stage_fear', 'Drained_after_socializing']
}

# ==============================================================================
# OPTIMIZATION TRACKER
# ==============================================================================

class OptimizationTracker:
    """Tracks optimization progress with rich display"""
    
    def __init__(self, console: Console):
        self.console = console
        self.results = {}
        self.current_dataset = None
        self.start_time = None
        self.dataset_start_time = None
        self.recent_trials = []
        self.logger = self._setup_logger()
        self.current_trial_info = None  # Add current trial tracking
        
    def _setup_logger(self):
        """Setup file logging"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(LOGS_DIR, f'optimization_{timestamp}.log')
        
        logger = logging.getLogger('optimization')
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def start_dataset(self, dataset_name):
        """Start tracking a new dataset"""
        self.current_dataset = dataset_name
        self.dataset_start_time = time.time()
        self.results[dataset_name] = {
            'start_time': self.dataset_start_time,
            'models': {},
            'best_score': 0,
            'best_model': None,
            'total_trials': 0
        }
        self.logger.info(f"Started optimization for {dataset_name}")
        
    def start_trial(self, model_name, trial_num):
        """Mark start of a new trial"""
        self.current_trial_info = {
            'model': model_name,
            'trial': trial_num,
            'start_time': time.time(),
            'status': 'Initializing...',
            'dataset': self.current_dataset
        }
        
    def update_trial(self, model_name, trial_num, score, params):
        """Update with new trial results"""
        # Clear current trial info
        self.current_trial_info = None
        
        if self.current_dataset not in self.results:
            return
            
        dataset_results = self.results[self.current_dataset]
        
        # Initialize model results if needed
        if model_name not in dataset_results['models']:
            dataset_results['models'][model_name] = {
                'trials': 0,
                'best_score': 0,
                'best_params': {},
                'history': []
            }
        
        model_results = dataset_results['models'][model_name]
        model_results['trials'] += 1
        model_results['history'].append({
            'trial': trial_num,
            'score': score,
            'timestamp': time.time()
        })
        
        # Update best score for model
        if score > model_results['best_score']:
            model_results['best_score'] = score
            model_results['best_params'] = params
            
        # Update overall best
        if score > dataset_results['best_score']:
            dataset_results['best_score'] = score
            dataset_results['best_model'] = model_name
            
        dataset_results['total_trials'] += 1
        
        # Add to recent trials
        self.recent_trials.append({
            'dataset': self.current_dataset,
            'model': model_name,
            'trial': trial_num,
            'score': score,
            'timestamp': time.time()
        })
        self.recent_trials = self.recent_trials[-10:]  # Keep last 10
        
        # Log to file
        self.logger.info(f"{self.current_dataset} - {model_name} - Trial {trial_num}: {score:.6f}")
        
    def get_display_data(self):
        """Get formatted data for rich display"""
        data = {
            'current_dataset': self.current_dataset,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'dataset_elapsed': time.time() - self.dataset_start_time if self.dataset_start_time else 0,
            'time_remaining': TIME_PER_DATASET - (time.time() - self.dataset_start_time) if self.dataset_start_time else 0,
            'results': self.results,
            'recent_trials': self.recent_trials
        }
        return data

# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================

def create_summary_table(data):
    """Create summary table for all datasets"""
    table = Table(title="Dataset Optimization Summary", expand=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Best Score", style="green")
    table.add_column("Best Model", style="yellow")
    table.add_column("Total Trials", style="blue")
    table.add_column("Status", style="magenta")
    
    for dataset_name in CORRECTED_DATASETS:
        if dataset_name in data['results']:
            result = data['results'][dataset_name]
            status = "Running" if dataset_name == data['current_dataset'] else "Completed"
            table.add_row(
                dataset_name,
                f"{result['best_score']:.6f}",
                result['best_model'] or "N/A",
                str(result['total_trials']),
                status
            )
        else:
            table.add_row(dataset_name, "-", "-", "-", "Pending")
            
    return table

def create_current_status_panel(data):
    """Create panel showing current optimization status"""
    if not data['current_dataset']:
        return Panel("No active optimization", title="Current Status")
        
    content = f"""
Dataset: {data['current_dataset']}
Time Elapsed: {data['dataset_elapsed']:.1f}s / {TIME_PER_DATASET}s
Time Remaining: {max(0, data['time_remaining']):.1f}s
Progress: {min(100, data['dataset_elapsed']/TIME_PER_DATASET*100):.1f}%
"""
    
    if data['current_dataset'] in data['results']:
        result = data['results'][data['current_dataset']]
        content += f"\nBest Score: {result['best_score']:.6f}"
        content += f"\nBest Model: {result['best_model'] or 'N/A'}"
        content += f"\nTotal Trials: {result['total_trials']}"
        
    return Panel(content, title="Current Optimization")

def create_current_trial_panel(tracker):
    """Create panel showing current running trial"""
    if not hasattr(tracker, 'current_trial_info') or not tracker.current_trial_info:
        return Panel("Waiting for first trial...", title="Current Trial", height=6)
    
    info = tracker.current_trial_info
    elapsed = time.time() - info['start_time']
    
    # Special formatting for system messages
    if info['model'] == 'system':
        content = f"""Status: {info['status']}
Dataset: {info.get('dataset', 'N/A')}
Elapsed: {elapsed:.1f}s"""
    else:
        content = f"""Model: {info['model'].upper()}
Trial: #{info['trial']}
Status: {info['status']}
Elapsed: {elapsed:.1f}s"""
    
    return Panel(content, title="Current Trial", height=6)

def create_recent_trials_table(data):
    """Create table of recent trials"""
    table = Table(title="Recent Trials (Last 10)", expand=True)
    table.add_column("Dataset", style="cyan", overflow="fold")
    table.add_column("Model", style="yellow")
    table.add_column("Trial", style="blue")
    table.add_column("Score", style="green")
    table.add_column("Age", style="magenta")
    
    current_time = time.time()
    for trial in reversed(data['recent_trials']):
        age = current_time - trial['timestamp']
        age_str = f"{age:.0f}s ago"
        table.add_row(
            trial['dataset'][-20:],  # Show last 20 chars
            trial['model'].upper(),
            str(trial['trial']),
            f"{trial['score']:.6f}",
            age_str
        )
        
    return table

def create_model_performance_table(data):
    """Create table showing model performance for current dataset"""
    if not data['current_dataset'] or data['current_dataset'] not in data['results']:
        return Table(title="Model Performance")
        
    table = Table(title=f"Model Performance - {data['current_dataset']}", expand=True)
    table.add_column("Model", style="yellow")
    table.add_column("Trials", style="blue")
    table.add_column("Best Score", style="green")
    table.add_column("Improvement", style="cyan")
    
    result = data['results'][data['current_dataset']]
    for model_name in ['xgb', 'gbm', 'cat']:
        if model_name in result['models']:
            model_data = result['models'][model_name]
            if len(model_data['history']) > 1:
                first_score = model_data['history'][0]['score']
                improvement = (model_data['best_score'] - first_score) * 100
                improvement_str = f"+{improvement:.3f}%"
            else:
                improvement_str = "-"
                
            table.add_row(
                model_name.upper(),
                str(model_data['trials']),
                f"{model_data['best_score']:.6f}",
                improvement_str
            )
            
    return table

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class WorkerTask:
    """Task for worker process"""
    task_id: int
    model_type: str
    resource: str
    trial_number: int
    params: Dict[str, Any]
    train_data: Any
    test_data: Any
    ambiguous_ids: set
    features: List[str]

@dataclass
class WorkerResult:
    """Result from worker process"""
    task_id: int
    model_type: str
    resource: str
    trial_number: int
    success: bool
    score: float = 0.0
    predictions: List = None
    duration: float = 0.0
    error: str = None

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_directories():
    """Create necessary directories"""
    for dir_path in [SCORES_DIR, LOGS_DIR, STUDIES_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        
def get_study_name(model_name, dataset_name):
    """Generate unique study name for Optuna"""
    base = f"{model_name}_{dataset_name}_{CV_FOLDS}fold"
    return hashlib.md5(base.encode()).hexdigest()[:12]

def load_data(dataset_name=None):
    """Load competition data with optional corrected dataset"""
    if dataset_name:
        train_path = os.path.join(CORRECTED_DATA_DIR, dataset_name)
        # Don't print in normal mode, only log
        logging.info(f"Loading corrected dataset: {train_path}")
    else:
        train_path = os.path.join(DATA_DIR, "train.csv")
        logging.info(f"Loading original dataset: {train_path}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))
    
    # Load ambiguous cases
    if os.path.exists(MOST_AMBIGUOUS):
        ambiguous_df = pd.read_csv(MOST_AMBIGUOUS)
        ambiguous_ids = set(ambiguous_df['id'].values)
        logging.info(f"Loaded {len(ambiguous_ids)} ambiguous training samples")
    else:
        ambiguous_ids = set()
        
    return train_df, test_df, ambiguous_ids

def preprocess_data(train_df, test_df):
    """Preprocess data"""
    numerical_cols = AMBIGUOUS_FEATURES['numerical']
    categorical_cols = AMBIGUOUS_FEATURES['categorical']
    
    # Fill missing values
    for col in numerical_cols:
        mean_val = train_df[col].mean()
        train_df[col] = train_df[col].fillna(mean_val)
        test_df[col] = test_df[col].fillna(mean_val)
    
    for col in categorical_cols:
        train_df[col] = train_df[col].fillna('Missing')
        test_df[col] = test_df[col].fillna('Missing')
        train_df[col] = train_df[col].map({'Yes': 1, 'No': 0, 'Missing': 0.5})
        test_df[col] = test_df[col].map({'Yes': 1, 'No': 0, 'Missing': 0.5})
    
    # Convert target
    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].map({'Extrovert': 1, 'Introvert': 0})
    
    return train_df, test_df

def create_ambiguity_features(df, alone_thresh, social_thresh, friends_thresh, ambig_score_thresh):
    """Create ambiguity detection features"""
    df_feat = df.copy()
    
    # Pattern detection
    df_feat['ambiguous_pattern'] = (
        (df['Time_spent_Alone'] < alone_thresh) & 
        (df['Social_event_attendance'] < social_thresh) &
        (df['Friends_circle_size'] < friends_thresh)
    ).astype(float)
    
    # Distance from ambiguous centroid
    df_feat['dist_to_ambiguous'] = np.sqrt(
        (df['Time_spent_Alone'] - 1.85)**2 +
        (df['Social_event_attendance'] - 3.75)**2 +
        (df['Friends_circle_size'] - 6.5)**2
    )
    
    # Ambiguity score
    df_feat['ambiguity_score'] = 1 / (1 + df_feat['dist_to_ambiguous'])
    
    # High ambiguity flag
    df_feat['high_ambiguity'] = (df_feat['ambiguity_score'] > ambig_score_thresh).astype(float)
    
    return df_feat, df_feat['high_ambiguity']

def save_submission(predictions, score, model_name, dataset_name):
    """Save submission file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"subm-{score:.5f}-{timestamp}-{model_name}-{dataset_name[:8]}.csv"
    filepath = os.path.join(SCORES_DIR, filename)
    
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))
    submission = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        TARGET_COLUMN: predictions
    })
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].map({1: 'Extrovert', 0: 'Introvert'})
    submission.to_csv(filepath, index=False)
    
    return filepath

# ==============================================================================
# MODEL FUNCTIONS
# ==============================================================================

def get_model_params(model_type, params_dict, resource):
    """Get model-specific parameters"""
    if model_type == "xgb":
        gpu_id = 0 if 'gpu0' in resource else (1 if 'gpu1' in resource else -1)
        return {
            'n_estimators': params_dict['n_estimators'],
            'learning_rate': params_dict['learning_rate'],
            'max_depth': params_dict['max_depth'],
            'subsample': params_dict['subsample'],
            'colsample_bytree': params_dict['colsample_bytree'],
            'reg_lambda': params_dict['reg_lambda'],
            'reg_alpha': params_dict['reg_alpha'],
            'gamma': params_dict['gamma'],
            'min_child_weight': params_dict['min_child_weight'],
            'tree_method': 'gpu_hist' if gpu_id >= 0 else 'hist',
            'gpu_id': gpu_id if gpu_id >= 0 else None,
            'random_state': CV_RANDOM_STATE,
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'use_label_encoder': False
        }
    elif model_type == "gbm":
        return {
            'n_estimators': params_dict['n_estimators'],
            'learning_rate': params_dict['learning_rate'],
            'max_depth': params_dict['max_depth'],
            'num_leaves': params_dict['num_leaves'],
            'subsample': params_dict['subsample'],
            'colsample_bytree': params_dict['colsample_bytree'],
            'reg_lambda': params_dict['reg_lambda'],
            'reg_alpha': params_dict['reg_alpha'],
            'min_child_weight': params_dict['min_child_weight'],
            'device': 'gpu' if 'gpu' in resource else 'cpu',
            'gpu_device_id': 0 if 'gpu0' in resource else (1 if 'gpu1' in resource else 0),
            'random_state': CV_RANDOM_STATE,
            'objective': 'binary',
            'metric': 'binary_error',
            'verbosity': -1
        }
    elif model_type == "cat":
        return {
            'iterations': params_dict['n_estimators'],
            'learning_rate': params_dict['learning_rate'],
            'depth': params_dict['max_depth'],
            'l2_leaf_reg': params_dict['reg_lambda'],
            'border_count': params_dict['border_count'],
            'task_type': 'GPU' if 'gpu' in resource else 'CPU',
            'devices': '0' if 'gpu0' in resource else ('1' if 'gpu1' in resource else None),
            'random_state': CV_RANDOM_STATE,
            'verbose': False
        }

def create_model(model_type, params):
    """Create model instance"""
    if model_type == "xgb":
        return xgb.XGBClassifier(**{k: v for k, v in params.items() if v is not None})
    elif model_type == "gbm":
        return lgb.LGBMClassifier(**{k: v for k, v in params.items() if v is not None})
    elif model_type == "cat":
        return cb.CatBoostClassifier(**{k: v for k, v in params.items() if v is not None})

# ==============================================================================
# OPTIMIZATION SCHEDULER
# ==============================================================================

class TimeBasedOptimizer:
    """Optimizer that runs for fixed time per dataset"""
    
    def __init__(self, tracker: OptimizationTracker):
        self.tracker = tracker
        self.console = Console()
        
    def optimize_dataset(self, dataset_name):
        """Optimize a single dataset for TIME_PER_DATASET seconds"""
        self.tracker.start_dataset(dataset_name)
        
        # Update status to show we're loading data
        if hasattr(self.tracker, 'current_trial_info'):
            self.tracker.current_trial_info = {
                'model': 'system',
                'trial': 0,
                'start_time': time.time(),
                'status': 'Loading dataset...',
                'dataset': dataset_name
            }
        
        # Load data
        train_df, test_df, ambiguous_ids = load_data(dataset_name)
        train_df, test_df = preprocess_data(train_df, test_df)
        
        features = AMBIGUOUS_FEATURES['numerical'] + AMBIGUOUS_FEATURES['categorical']
        
        # Update status to show we're creating studies
        if hasattr(self.tracker, 'current_trial_info'):
            self.tracker.current_trial_info['status'] = 'Initializing Optuna studies...'
        
        # Create Optuna studies
        studies = {}
        for model_name in ['xgb', 'gbm', 'cat']:
            if MODEL_CONFIG[model_name]['enabled']:
                study_name = get_study_name(model_name, dataset_name)
                storage_name = f"sqlite:///{STUDIES_DIR}/{study_name}.db"
                
                # Save mapping for reference
                mapping_file = os.path.join(STUDIES_DIR, "study_mapping.txt")
                with open(mapping_file, 'a') as f:
                    f.write(f"{study_name}.db = {model_name} + {dataset_name}\n")
                
                try:
                    study = optuna.load_study(study_name=study_name, storage=storage_name)
                    self.tracker.logger.info(f"Loaded existing study: {study_name} ({model_name} + {dataset_name})")
                except:
                    study = optuna.create_study(
                        study_name=study_name,
                        storage=storage_name,
                        direction='maximize',
                        load_if_exists=True
                    )
                    self.tracker.logger.info(f"Created new study: {study_name} ({model_name} + {dataset_name})")
                    
                studies[model_name] = study
        
        # Clear loading status
        if hasattr(self.tracker, 'current_trial_info'):
            self.tracker.current_trial_info = None
        
        # Optimization loop
        start_time = time.time()
        trial_count = {model: 0 for model in studies.keys()}
        
        while time.time() - start_time < TIME_PER_DATASET:
            # Rotate through models
            for model_name, study in studies.items():
                if time.time() - start_time >= TIME_PER_DATASET:
                    break
                    
                # Create objective function
                def objective(trial):
                    # Sample parameters
                    params = {
                        'alone_thresh': trial.suggest_float('alone_thresh', 2.0, 3.5),
                        'social_thresh': trial.suggest_float('social_thresh', 3.5, 5.5),
                        'friends_thresh': trial.suggest_int('friends_thresh', 6, 9),
                        'ambig_score_thresh': trial.suggest_float('ambig_score_thresh', 0.2, 0.4),
                        'proba_low_thresh': trial.suggest_float('proba_low_thresh', 0.1, 0.25),
                        'proba_high_thresh': trial.suggest_float('proba_high_thresh', 0.45, 0.55),
                        'ambig_weight': trial.suggest_float('ambig_weight', 5.0, 20.0),
                        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 0.003, 0.1),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 10.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                    }
                    
                    if model_name == "xgb":
                        params['gamma'] = trial.suggest_float('gamma', 0.0, 0.5)
                    elif model_name == "gbm":
                        params['num_leaves'] = trial.suggest_int('num_leaves', 20, 300)
                    elif model_name == "cat":
                        params['border_count'] = trial.suggest_int('border_count', 32, 255)
                    
                    # Create features
                    X_train, train_high_ambig = create_ambiguity_features(
                        train_df[features], 
                        params['alone_thresh'], 
                        params['social_thresh'],
                        params['friends_thresh'], 
                        params['ambig_score_thresh']
                    )
                    
                    extended_features = features + ['ambiguous_pattern', 'dist_to_ambiguous', 
                                                   'ambiguity_score', 'high_ambiguity']
                    
                    X = X_train[extended_features]
                    y = train_df[TARGET_COLUMN]
                    
                    # Sample weights
                    sample_weights = np.ones(len(train_df))
                    sample_weights[train_df[ID_COLUMN].isin(ambiguous_ids)] = params['ambig_weight']
                    
                    # Cross-validation
                    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
                    cv_scores = []
                    
                    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                        # Update trial status with fold progress
                        if hasattr(self.tracker, 'current_trial_info') and self.tracker.current_trial_info:
                            self.tracker.current_trial_info['status'] = f'Training fold {fold+1}/{CV_FOLDS}...'
                        X_train_fold, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train_fold, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        weights_train = sample_weights[train_idx]
                        
                        # Get model parameters
                        model_params = get_model_params(
                            model_name, 
                            params, 
                            MODEL_RESOURCE_ASSIGNMENT[model_name][0]
                        )
                        
                        # Create and train model
                        model = create_model(model_name, model_params)
                        
                        if model_name == "cat":
                            model.fit(X_train_fold, y_train_fold, sample_weight=weights_train,
                                     eval_set=(X_val, y_val), early_stopping_rounds=50)
                        else:
                            model.fit(X_train_fold, y_train_fold, sample_weight=weights_train)
                        
                        # Predictions
                        proba = model.predict_proba(X_val)[:, 1]
                        pred = (proba > params['proba_high_thresh']).astype(int)
                        
                        # Apply ambiguous rule
                        val_high_ambig_mask = train_high_ambig.iloc[val_idx].values.astype(bool)
                        pred[val_high_ambig_mask] = 1
                        
                        # Revert very low probability
                        very_low_prob = (proba < params['proba_low_thresh']) & val_high_ambig_mask
                        pred[very_low_prob] = 0
                        
                        accuracy = (pred == y_val.values).mean()
                        cv_scores.append(accuracy)
                    
                    mean_score = np.mean(cv_scores)
                    
                    # Update tracker
                    self.tracker.update_trial(
                        model_name, 
                        trial_count[model_name], 
                        mean_score,
                        params
                    )
                    
                    trial_count[model_name] += 1
                    
                    return mean_score
                
                # Run one trial
                try:
                    # Mark trial start
                    self.tracker.start_trial(model_name, trial_count[model_name])
                    
                    # Suppress Optuna output during optimization
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        study.optimize(objective, n_trials=1, show_progress_bar=False)
                except Exception as e:
                    self.tracker.logger.error(f"Error in optimization: {str(e)}")
                    # Clear current trial on error
                    self.tracker.current_trial_info = None
                    
        # Save best model predictions
        self._save_best_predictions(dataset_name, studies, train_df, test_df, ambiguous_ids, features)
        
    def _save_best_predictions(self, dataset_name, studies, train_df, test_df, ambiguous_ids, features):
        """Save predictions from best model"""
        best_score = 0
        best_model = None
        best_params = None
        
        for model_name, study in studies.items():
            try:
                if len(study.trials) == 0:
                    continue
                if study.best_value > best_score:
                    best_score = study.best_value
                    best_model = model_name
                    best_params = study.best_params
            except ValueError:
                # No completed trials for this model
                self.tracker.logger.warning(f"No completed trials for {model_name} in {dataset_name}")
                continue
                
        if best_model and best_params:
            # Create features for test
            X_test, test_high_ambig = create_ambiguity_features(
                test_df[features],
                best_params['alone_thresh'],
                best_params['social_thresh'],
                best_params['friends_thresh'],
                best_params['ambig_score_thresh']
            )
            
            extended_features = features + ['ambiguous_pattern', 'dist_to_ambiguous',
                                           'ambiguity_score', 'high_ambiguity']
            
            X_train_full = create_ambiguity_features(
                train_df[features],
                best_params['alone_thresh'],
                best_params['social_thresh'],
                best_params['friends_thresh'],
                best_params['ambig_score_thresh']
            )[0][extended_features]
            
            y_train = train_df[TARGET_COLUMN]
            X_test_final = X_test[extended_features]
            
            # Sample weights
            sample_weights = np.ones(len(train_df))
            sample_weights[train_df[ID_COLUMN].isin(ambiguous_ids)] = best_params['ambig_weight']
            
            # Train final model
            model_params = get_model_params(
                best_model,
                best_params,
                MODEL_RESOURCE_ASSIGNMENT[best_model][0]
            )
            
            final_model = create_model(best_model, model_params)
            
            if best_model == "cat":
                final_model.fit(X_train_full, y_train, sample_weight=sample_weights)
            else:
                final_model.fit(X_train_full, y_train, sample_weight=sample_weights)
            
            # Make predictions
            proba_test = final_model.predict_proba(X_test_final)[:, 1]
            predictions = (proba_test > best_params['proba_high_thresh']).astype(int)
            test_high_ambig_mask = test_high_ambig.values.astype(bool)
            predictions[test_high_ambig_mask] = 1
            very_intro = (proba_test < best_params['proba_low_thresh']) & test_high_ambig_mask
            predictions[very_intro] = 0
            
            # Save submission
            save_submission(predictions, best_score, best_model, dataset_name)
            
    def run(self):
        """Run optimization for all datasets"""
        self.tracker.start_time = time.time()
        
        # Show startup message
        if hasattr(self.tracker, 'current_trial_info'):
            self.tracker.current_trial_info = {
                'model': 'system',
                'trial': 0,
                'start_time': time.time(),
                'status': 'Starting optimization system...',
                'dataset': 'Initializing'
            }
        
        if hasattr(self, 'debug_mode') and self.debug_mode:
            # Debug mode - no live display
            for dataset_name in CORRECTED_DATASETS:
                self.console.print(f"\n[cyan]Starting optimization for: {dataset_name}[/cyan]")
                try:
                    self.optimize_dataset(dataset_name)
                    self.console.print(f"[green]✓ Completed: {dataset_name}[/green]")
                except Exception as e:
                    self.console.print(f"[red]✗ Error in {dataset_name}: {str(e)}[/red]")
                    self.console.print(traceback.format_exc())
                    self.tracker.logger.error(f"Error in {dataset_name}: {str(e)}\n{traceback.format_exc()}")
        else:
            # Normal mode with live display
            def update_display(live):
                """Update the live display"""
                data = self.tracker.get_display_data()
                layout = Layout()
                layout.split_column(
                    Layout(create_summary_table(data), size=len(CORRECTED_DATASETS) + 3),
                    Layout(create_current_status_panel(data), size=8),
                    Layout(create_current_trial_panel(self.tracker), size=6),
                    Layout(create_model_performance_table(data), size=6),
                    Layout(create_recent_trials_table(data), size=10)
                )
                live.update(layout)
            
            with Live(console=self.console, refresh_per_second=1, screen=True) as live:
                # Show initial display immediately
                update_display(live)
                
                # Create a thread to update display periodically
                import threading
                stop_display = threading.Event()
                
                def display_updater():
                    while not stop_display.is_set():
                        update_display(live)
                        time.sleep(0.5)  # Update every 0.5 seconds
                
                display_thread = threading.Thread(target=display_updater)
                display_thread.start()
                
                try:
                    for dataset_name in CORRECTED_DATASETS:
                        try:
                            # Optimize dataset
                            self.optimize_dataset(dataset_name)
                        except Exception as e:
                            # Log error but continue with next dataset
                            self.tracker.logger.error(f"Error in {dataset_name}: {str(e)}\n{traceback.format_exc()}")
                            error_file = f"output/error_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            with open(error_file, 'w') as f:
                                f.write(f"Error in {dataset_name}: {str(e)}\n\n")
                                f.write(traceback.format_exc())
                finally:
                    stop_display.set()
                    display_thread.join()
                
        # Final summary
        self._print_final_summary()
        
    def _print_final_summary(self):
        """Print final summary of all optimizations"""
        self.console.print("\n[bold green]OPTIMIZATION COMPLETE![/bold green]\n")
        
        summary_table = Table(title="Final Results", show_header=True)
        summary_table.add_column("Dataset", style="cyan")
        summary_table.add_column("Best Score", style="green")
        summary_table.add_column("Best Model", style="yellow")
        summary_table.add_column("Improvement", style="magenta")
        summary_table.add_column("Total Trials", style="blue")
        
        baseline_score = None
        
        for dataset_name in CORRECTED_DATASETS:
            if dataset_name in self.tracker.results:
                result = self.tracker.results[dataset_name]
                
                if baseline_score is None and "01" in dataset_name:
                    baseline_score = result['best_score']
                    
                improvement = ""
                if baseline_score and result['best_score'] > baseline_score:
                    diff = (result['best_score'] - baseline_score) * 100
                    improvement = f"+{diff:.3f}%"
                    
                summary_table.add_row(
                    dataset_name,
                    f"{result['best_score']:.6f}",
                    result['best_model'] or "N/A",
                    improvement,
                    str(result['total_trials'])
                )
                
        self.console.print(summary_table)
        
        # Save summary to file
        summary_path = os.path.join(LOGS_DIR, f"optimization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(summary_path, 'w') as f:
            f.write("OPTIMIZATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            for dataset_name in CORRECTED_DATASETS:
                if dataset_name in self.tracker.results:
                    result = self.tracker.results[dataset_name]
                    f.write(f"{dataset_name}:\n")
                    f.write(f"  Best Score: {result['best_score']:.6f}\n")
                    f.write(f"  Best Model: {result['best_model']}\n")
                    f.write(f"  Total Trials: {result['total_trials']}\n\n")
                    
        self.console.print(f"\n[green]Summary saved to: {summary_path}[/green]")

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main entry point"""
    console = Console()
    
    # Add debug mode
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        console.print("[yellow]DEBUG MODE ENABLED - No live display[/yellow]")
    
    console.print("[bold blue]CORRECTED DATASETS OPTIMIZATION[/bold blue]")
    console.print(f"Time per dataset: {TIME_PER_DATASET}s")
    console.print(f"Total datasets: {len(CORRECTED_DATASETS)}")
    console.print(f"Estimated total time: {TIME_PER_DATASET * len(CORRECTED_DATASETS)}s")
    console.print("-" * 60)
    
    # Create directories
    create_directories()
    
    # Create tracker and optimizer
    tracker = OptimizationTracker(console)
    optimizer = TimeBasedOptimizer(tracker)
    
    # Add debug mode to optimizer
    optimizer.debug_mode = debug_mode
    
    # Run optimization
    try:
        optimizer.run()
    except KeyboardInterrupt:
        console.print("\n[red]Optimization interrupted by user[/red]")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        console.print(f"[red]Traceback:[/red]")
        console.print(traceback.format_exc())
        tracker.logger.error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        
        # Save error details to file
        error_file = f"output/error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_file, 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())
        console.print(f"[yellow]Error details saved to: {error_file}[/yellow]")
        
    console.print("\n[green]Optimization complete![/green]")

if __name__ == "__main__":
    main()
