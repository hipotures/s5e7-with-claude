#!/usr/bin/env python3
"""
Parallel optimization for all corrected datasets with multiprocessing.
Each model runs on its assigned resource (XGB/GBM on GPUs, CAT on CPU).
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
from rich import box
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import logging
import threading

warnings.filterwarnings('ignore')

# Silence Optuna output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

# Time limit per dataset (in seconds)
TIME_PER_DATASET = 3600  # 1 hour per dataset

# CPU thread limit for CatBoost (to leave CPU resources for GPU models)
CATBOOST_THREAD_COUNT = 16  # Adjust based on your CPU

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

# Resource assignment - PARALLEL EXECUTION
# Fixed assignment: each model gets dedicated resource for better balance
MODEL_RESOURCE_ASSIGNMENT = {
    'xgb': ['gpu0'],  # XGBoost always on GPU 0
    'gbm': ['gpu1'],  # LightGBM always on GPU 1
    'cat': ['cpu']    # CatBoost always on CPU
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
# DATA STRUCTURES
# ==============================================================================

@dataclass
class WorkerTask:
    """Task for worker process"""
    task_id: int
    dataset_name: str
    model_type: str
    resource: str
    trial_number: int
    trial_params: Dict[str, Any]
    study_name: str

@dataclass
class WorkerResult:
    """Result from worker process"""
    task_id: int
    dataset_name: str
    model_type: str
    resource: str
    trial_number: int
    success: bool
    score: float = 0.0
    predictions: List = None
    duration: float = 0.0
    error: str = None

# ==============================================================================
# OPTIMIZATION TRACKER
# ==============================================================================

class ParallelOptimizationTracker:
    """Tracks parallel optimization progress with rich display"""
    
    def __init__(self, console: Console):
        self.console = console
        self.results = {}
        self.current_dataset = None
        self.start_time = None
        self.dataset_start_time = None
        self.recent_trials = []
        self.logger = self._setup_logger()
        self.running_tasks = {}  # task_id -> task info
        self.resource_status = {}  # resource -> status
        
        # Initialize resource status
        all_resources = set()
        for resources in MODEL_RESOURCE_ASSIGNMENT.values():
            all_resources.update(resources)
        self.resource_status = {res: 'idle' for res in all_resources}
        
    def _setup_logger(self):
        """Setup file logging"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(LOGS_DIR, f'parallel_optimization_{timestamp}.log')
        
        logger = logging.getLogger('parallel_optimization')
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
        
    def start_task(self, task: WorkerTask):
        """Mark start of a new task"""
        self.running_tasks[task.task_id] = {
            'dataset': task.dataset_name,
            'model': task.model_type,
            'trial': task.trial_number,
            'resource': task.resource,
            'start_time': time.time(),
            'status': 'Initializing...'
        }
        self.resource_status[task.resource] = f'{task.model_type} trial {task.trial_number}'
        self.logger.info(f"Started {task.model_type} trial {task.trial_number} on {task.resource}")
        
    def complete_task(self, result: WorkerResult):
        """Mark task completion"""
        if result.task_id in self.running_tasks:
            del self.running_tasks[result.task_id]
        self.resource_status[result.resource] = 'idle'
        
        if result.success:
            self.update_trial(result.dataset_name, result.model_type, 
                            result.trial_number, result.score)
        
    def update_trial(self, dataset_name, model_name, trial_num, score):
        """Update with new trial results"""
        if dataset_name not in self.results:
            return
            
        dataset_results = self.results[dataset_name]
        
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
            
        # Update overall best
        if score > dataset_results['best_score']:
            dataset_results['best_score'] = score
            dataset_results['best_model'] = model_name
            
        dataset_results['total_trials'] += 1
        
        # Add to recent trials
        self.recent_trials.append({
            'dataset': dataset_name,
            'model': model_name,
            'trial': trial_num,
            'score': score,
            'timestamp': time.time()
        })
        self.recent_trials = self.recent_trials[-10:]  # Keep last 10
        
        # Log to file
        self.logger.info(f"{dataset_name} - {model_name} - Trial {trial_num}: {score:.6f}")
        
    def get_display_data(self):
        """Get formatted data for rich display"""
        data = {
            'current_dataset': self.current_dataset,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'dataset_elapsed': time.time() - self.dataset_start_time if self.dataset_start_time else 0,
            'time_remaining': TIME_PER_DATASET - (time.time() - self.dataset_start_time) if self.dataset_start_time else 0,
            'results': self.results,
            'recent_trials': self.recent_trials,
            'running_tasks': self.running_tasks,
            'resource_status': self.resource_status
        }
        return data

# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================

def create_summary_table(data):
    """Create summary table for all datasets"""
    table = Table(title="Dataset Optimization Summary", expand=True, box=box.ROUNDED)
    table.add_column("Dataset", style="cyan", min_width=25)
    table.add_column("Best Score", style="green")
    table.add_column("Best Model", style="yellow")
    table.add_column("Total Trials", style="blue")
    table.add_column("Status")
    
    for dataset_name in CORRECTED_DATASETS:
        if dataset_name in data['results']:
            result = data['results'][dataset_name]
            if dataset_name == data['current_dataset']:
                status = "[bold yellow]Running[/bold yellow]"
            else:
                status = "[green]Completed[/green]"
            table.add_row(
                dataset_name,
                f"{result['best_score']:.6f}",
                result['best_model'] or "N/A",
                str(result['total_trials']),
                status
            )
        else:
            table.add_row(dataset_name, "-", "-", "-", "[dim]Pending[/dim]")
            
    return table

def create_resource_status_panel(data):
    """Create panel showing resource utilization"""
    content = ""
    for resource, status in sorted(data['resource_status'].items()):
        emoji = "🟢" if status == 'idle' else "🔴"
        content += f"{emoji} {resource}: {status}\n"
    
    return Panel(content.strip(), title="Resource Status", height=len(data['resource_status']) + 2)

def create_running_tasks_panel(data):
    """Create panel showing currently running tasks"""
    if not data['running_tasks']:
        return Panel("No active tasks", title="Running Tasks", height=4)
    
    content = ""
    for task_id, info in data['running_tasks'].items():
        elapsed = time.time() - info['start_time']
        content += f"{info['model'].upper()} trial {info['trial']} on {info['resource']} ({elapsed:.1f}s)\n"
    
    return Panel(content.strip(), title="Running Tasks", height=min(len(data['running_tasks']) + 2, 8))

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

def create_recent_trials_table(data):
    """Create table of recent trials"""
    table = Table(title="Recent Trials (Last 10)", expand=True, box=box.ROUNDED)
    table.add_column("Dataset", style="cyan", overflow="fold", min_width=25)
    table.add_column("Model", style="yellow")
    table.add_column("Trial", style="blue")
    table.add_column("Score", style="green")
    table.add_column("Age", style="magenta")
    
    current_time = time.time()
    for trial in reversed(data['recent_trials']):
        age = current_time - trial['timestamp']
        age_str = f"{age:.0f}s ago"
        table.add_row(
            trial['dataset'],  # Show full dataset name
            trial['model'].upper(),
            str(trial['trial']),
            f"{trial['score']:.6f}",
            age_str
        )
        
    return table

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

def set_gpu_environment(resource):
    """Set GPU environment variables based on resource assignment"""
    if resource == 'gpu0':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    elif resource == 'gpu1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    elif resource == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        # Default - use all GPUs
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

def load_data(dataset_name=None):
    """Load competition data with optional corrected dataset"""
    if dataset_name:
        train_path = os.path.join(CORRECTED_DATA_DIR, dataset_name)
    else:
        train_path = os.path.join(DATA_DIR, "train.csv")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))
    
    # Load ambiguous cases
    if os.path.exists(MOST_AMBIGUOUS):
        ambiguous_df = pd.read_csv(MOST_AMBIGUOUS)
        ambiguous_ids = set(ambiguous_df['id'].values)
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
    # Remove .csv extension and use dataset number
    dataset_short = dataset_name.replace('train_corrected_', 'tc').replace('.csv', '')
    filename = f"subm-{score:.5f}-{timestamp}-{model_name}-{dataset_short}.csv"
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
            'thread_count': CATBOOST_THREAD_COUNT,  # Limit CPU threads to leave resources for GPU models
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
# WORKER PROCESS
# ==============================================================================

def worker_process(task_queue: mp.Queue, result_queue: mp.Queue, data_dict: dict):
    """Worker process that pulls tasks from queue and trains models"""
    
    # Try to set process name
    try:
        import setproctitle
        setproctitle.setproctitle("optuna-worker")
    except ImportError:
        pass
    
    while True:
        try:
            # Get task from queue (blocking with timeout)
            task = task_queue.get(timeout=1)
            
            if task == "STOP":
                break
                
            start_time = time.time()
            
            # Set GPU environment
            set_gpu_environment(task.resource)
            
            # Update process name with model type
            try:
                import setproctitle
                setproctitle.setproctitle(f"optuna-{task.model_type}-{task.resource}")
            except ImportError:
                pass
            
            # Load dataset for this task
            train_df, test_df, ambiguous_ids = load_data(task.dataset_name)
            train_df, test_df = preprocess_data(train_df, test_df)
            features = AMBIGUOUS_FEATURES['numerical'] + AMBIGUOUS_FEATURES['categorical']
            
            try:
                # Extract parameters
                params = task.trial_params
                
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
                    X_train_fold, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_fold, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    weights_train = sample_weights[train_idx]
                    
                    # Get model parameters
                    model_params = get_model_params(
                        task.model_type, 
                        params, 
                        task.resource
                    )
                    
                    # Create and train model
                    model = create_model(task.model_type, model_params)
                    
                    if task.model_type == "cat":
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
                
                # Train final model on full data for predictions
                model_params = get_model_params(task.model_type, params, task.resource)
                final_model = create_model(task.model_type, model_params)
                
                if task.model_type == "cat":
                    final_model.fit(X, y, sample_weight=sample_weights)
                else:
                    final_model.fit(X, y, sample_weight=sample_weights)
                
                # Make test predictions
                X_test, test_high_ambig = create_ambiguity_features(
                    test_df[features],
                    params['alone_thresh'],
                    params['social_thresh'],
                    params['friends_thresh'],
                    params['ambig_score_thresh']
                )
                
                X_test_final = X_test[extended_features]
                proba_test = final_model.predict_proba(X_test_final)[:, 1]
                predictions = (proba_test > params['proba_high_thresh']).astype(int)
                test_high_ambig_mask = test_high_ambig.values.astype(bool)
                predictions[test_high_ambig_mask] = 1
                very_intro = (proba_test < params['proba_low_thresh']) & test_high_ambig_mask
                predictions[very_intro] = 0
                
                # Send result
                result = WorkerResult(
                    task_id=task.task_id,
                    dataset_name=task.dataset_name,
                    model_type=task.model_type,
                    resource=task.resource,
                    trial_number=task.trial_number,
                    success=True,
                    score=mean_score,
                    predictions=predictions.tolist(),
                    duration=time.time() - start_time
                )
                result_queue.put(result)
                
            except Exception as e:
                # Send error result
                result = WorkerResult(
                    task_id=task.task_id,
                    dataset_name=task.dataset_name,
                    model_type=task.model_type,
                    resource=task.resource,
                    trial_number=task.trial_number,
                    success=False,
                    error=str(e) + "\n" + traceback.format_exc()
                )
                result_queue.put(result)
                
        except Empty:
            # No task available, continue waiting
            continue
        except Exception as e:
            print(f"Worker: Fatal error: {e}")
            traceback.print_exc()
            break

# ==============================================================================
# PARALLEL OPTIMIZATION SCHEDULER
# ==============================================================================

class ParallelOptimizationScheduler:
    """Manages parallel allocation of models to resources"""
    
    def __init__(self, tracker: ParallelOptimizationTracker):
        self.tracker = tracker
        self.console = Console()
        
        # Extract all unique resources
        all_resources = set()
        for resources in MODEL_RESOURCE_ASSIGNMENT.values():
            all_resources.update(resources)
        
        # Resource tracking
        self.resource_status = {resource: 'free' for resource in all_resources}
        
        # Process management
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        
        # Task tracking
        self.next_task_id = 0
        self.running_tasks = {}  # task_id -> (model, resource, start_time, trial, dataset)
        
    def start_workers(self):
        """Start worker processes"""
        num_workers = len(self.resource_status)
        for i in range(num_workers):
            # Empty data dict - workers will load their own data
            p = mp.Process(target=worker_process, 
                          args=(self.task_queue, self.result_queue, {}))
            p.start()
            self.workers.append(p)
        self.tracker.logger.info(f"Started {num_workers} worker processes")
        
    def stop_workers(self):
        """Stop all worker processes"""
        for _ in self.workers:
            self.task_queue.put("STOP")
        
        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self.tracker.logger.info("Stopped all worker processes")
        
    def _get_free_resource_for_model(self, model_type):
        """Get free resource from model's assigned resources"""
        assigned_resources = MODEL_RESOURCE_ASSIGNMENT.get(model_type, [])
        
        for resource in assigned_resources:
            if self.resource_status.get(resource) == 'free':
                return resource
        return None
        
    def _create_task(self, dataset_name, model_type, resource, study):
        """Create task for worker"""
        # Ask Optuna for next trial
        trial = study.ask()
        
        # Sample parameters
        trial_params = {
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
        
        if model_type == "xgb":
            trial_params['gamma'] = trial.suggest_float('gamma', 0.0, 0.5)
        elif model_type == "gbm":
            trial_params['num_leaves'] = trial.suggest_int('num_leaves', 20, 300)
        elif model_type == "cat":
            trial_params['border_count'] = trial.suggest_int('border_count', 32, 255)
        
        # Create task
        task = WorkerTask(
            task_id=self.next_task_id,
            dataset_name=dataset_name,
            model_type=model_type,
            resource=resource,
            trial_number=trial.number,
            trial_params=trial_params,
            study_name=get_study_name(model_type, dataset_name)
        )
        
        self.next_task_id += 1
        return task, trial
        
    def optimize_dataset(self, dataset_name):
        """Optimize a single dataset for TIME_PER_DATASET seconds"""
        self.tracker.start_dataset(dataset_name)
        
        # Create Optuna studies for all models
        studies = {}
        model_trials = {}
        
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
                model_trials[model_name] = 0
        
        # Optimization loop
        start_time = time.time()
        active_trials = {}  # task_id -> (model, trial)
        
        while time.time() - start_time < TIME_PER_DATASET:
            # Schedule new tasks for free resources
            for model_name, study in studies.items():
                resource = self._get_free_resource_for_model(model_name)
                if resource:
                    task, trial = self._create_task(dataset_name, model_name, resource, study)
                    
                    # Update status
                    self.resource_status[resource] = 'busy'
                    self.running_tasks[task.task_id] = (model_name, resource, time.time(), trial, dataset_name)
                    active_trials[task.task_id] = (model_name, trial)
                    model_trials[model_name] += 1
                    
                    # Send to queue
                    self.task_queue.put(task)
                    self.tracker.start_task(task)
                    
                    self.tracker.logger.info(f"Scheduled: {model_name} trial {task.trial_number} on {resource}")
            
            # Process results
            try:
                while True:
                    # Non-blocking get
                    result = self.result_queue.get_nowait()
                    
                    # Free resources
                    if result.task_id in self.running_tasks:
                        model_type, resource, start_time_task, trial, _ = self.running_tasks[result.task_id]
                        self.resource_status[resource] = 'free'
                        del self.running_tasks[result.task_id]
                        
                        # Update Optuna
                        if result.task_id in active_trials:
                            model_name, trial = active_trials[result.task_id]
                            study = studies[model_name]
                            
                            if result.success:
                                study.tell(trial, result.score)
                                
                                # Check if new best BEFORE updating tracker
                                current_best = self.tracker.results[dataset_name].get('best_score', 0)
                                is_new_best = result.score > current_best
                                
                                self.tracker.complete_task(result)
                                
                                # Save submission if new best
                                if is_new_best:
                                    save_submission(result.predictions, result.score, 
                                                  model_name, dataset_name)
                            else:
                                study.tell(trial, None, state=optuna.trial.TrialState.FAIL)
                                self.tracker.logger.error(f"Failed trial for {model_name}: {result.error}")
                            
                            del active_trials[result.task_id]
                            
            except Empty:
                pass
            
            # Small sleep to prevent busy waiting
            time.sleep(0.1)
        
        # Wait for remaining tasks to complete (with timeout)
        wait_start = time.time()
        while self.running_tasks and time.time() - wait_start < 60:
            try:
                result = self.result_queue.get(timeout=1)
                
                if result.task_id in self.running_tasks:
                    model_type, resource, start_time_task, trial, _ = self.running_tasks[result.task_id]
                    self.resource_status[resource] = 'free'
                    del self.running_tasks[result.task_id]
                    
                    if result.task_id in active_trials:
                        model_name, trial = active_trials[result.task_id]
                        study = studies[model_name]
                        
                        if result.success:
                            study.tell(trial, result.score)
                            
                            # Check if new best BEFORE updating tracker
                            current_best = self.tracker.results[dataset_name].get('best_score', 0)
                            is_new_best = result.score > current_best
                            
                            self.tracker.complete_task(result)
                            
                            if is_new_best:
                                save_submission(result.predictions, result.score, 
                                              model_name, dataset_name)
                        else:
                            study.tell(trial, None, state=optuna.trial.TrialState.FAIL)
                        
                        del active_trials[result.task_id]
                        
            except Empty:
                pass
                
    def run(self):
        """Run optimization for all datasets"""
        self.tracker.start_time = time.time()
        
        # Start workers
        self.start_workers()
        
        # Create display update thread
        def update_display(live):
            """Update the live display"""
            data = self.tracker.get_display_data()
            layout = Layout()
            # Create middle row layout
            middle_row = Layout()
            middle_row.split_row(
                Layout(create_resource_status_panel(data)),
                Layout(create_running_tasks_panel(data))
            )
            
            layout.split_column(
                Layout(create_summary_table(data), size=len(CORRECTED_DATASETS) + 5),  # +5 to ensure bottom border
                Layout(create_current_status_panel(data), size=7),
                Layout(middle_row, size=8),
                Layout(create_recent_trials_table(data))  # No fixed size - use remaining space
            )
            live.update(layout)
        
        with Live(console=self.console, refresh_per_second=1, screen=True) as live:
            # Show initial display
            update_display(live)
            
            # Create display update thread
            stop_display = threading.Event()
            
            def display_updater():
                while not stop_display.is_set():
                    update_display(live)
                    time.sleep(0.5)
            
            display_thread = threading.Thread(target=display_updater)
            display_thread.start()
            
            try:
                for dataset_name in CORRECTED_DATASETS:
                    try:
                        self.optimize_dataset(dataset_name)
                    except Exception as e:
                        self.tracker.logger.error(f"Error in {dataset_name}: {str(e)}\n{traceback.format_exc()}")
                        
            finally:
                stop_display.set()
                display_thread.join()
                self.stop_workers()
        
        # Save final summary
        self._save_summary()
        
        # Show final results and wait for ESC
        self._show_final_results()
        
    def _save_summary(self):
        """Save optimization summary"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(LOGS_DIR, f'parallel_optimization_summary_{timestamp}.txt')
        
        with open(summary_file, 'w') as f:
            f.write("PARALLEL OPTIMIZATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            for dataset_name, result in self.tracker.results.items():
                f.write(f"\nDataset: {dataset_name}\n")
                f.write(f"Best Score: {result['best_score']:.6f}\n")
                f.write(f"Best Model: {result['best_model']}\n")
                f.write(f"Total Trials: {result['total_trials']}\n")
                
                if result['models']:
                    f.write("\nModel Details:\n")
                    for model_name, model_data in result['models'].items():
                        f.write(f"  {model_name}: {model_data['trials']} trials, "
                               f"best score: {model_data['best_score']:.6f}\n")
    
    def _show_final_results(self):
        """Show final results and wait for ESC key"""
        from rich.console import Console
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.align import Align
        
        console = Console()
        
        # Create final display
        data = self.tracker.get_display_data()
        
        # Mark all as completed
        for dataset in data['results']:
            data['results'][dataset]['status'] = 'Completed'
        
        # Create layout
        layout = Layout()
        
        # Create middle row layout
        middle_row = Layout()
        middle_row.split_row(
            Layout(create_resource_status_panel(data)),
            Layout(create_running_tasks_panel(data))
        )
        
        layout.split_column(
            Layout(create_summary_table(data), size=len(CORRECTED_DATASETS) + 5),
            Layout(self._create_final_summary_panel(data), size=10),
            Layout(middle_row, size=8),
            Layout(create_recent_trials_table(data))
        )
        
        # Display with message
        console.clear()
        console.print(layout)
        console.print("\n[bold yellow]Optimization Complete! Press ESC to exit...[/bold yellow]")
        
        # Wait for ESC
        try:
            import sys
            if sys.platform != 'win32':
                # Linux/Mac - use getch
                import termios, tty
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    while True:
                        ch = sys.stdin.read(1)
                        if ord(ch) == 27:  # ESC
                            break
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            else:
                # Windows
                import msvcrt
                while True:
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'\x1b':  # ESC
                            break
        except:
            # Fallback - just wait for Enter
            console.print("\n[dim]Press Enter to exit...[/dim]")
            input()
    
    def _create_final_summary_panel(self, data):
        """Create final summary panel"""
        # Find best overall score
        best_score = 0
        best_dataset = None
        best_model = None
        total_trials = 0
        
        for dataset, result in data['results'].items():
            total_trials += result.get('total_trials', 0)
            if result['best_score'] > best_score:
                best_score = result['best_score']
                best_dataset = dataset
                best_model = result['best_model']
        
        # Calculate gap to target
        target = 0.975708
        gap = target - best_score
        
        content = f"""
[bold]Best Overall Score:[/bold] {best_score:.6f} ({best_model})
[bold]Best Dataset:[/bold] {best_dataset}
[bold]Gap to Target:[/bold] {gap:.6f} ({gap*100:.4f}%)
[bold]Total Trials:[/bold] {total_trials}
[bold]Total Time:[/bold] {data['elapsed_time']/60:.1f} minutes
"""
        
        return Panel(content.strip(), title="Final Summary", border_style="green")

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main entry point"""
    create_directories()
    
    console = Console()
    tracker = ParallelOptimizationTracker(console)
    scheduler = ParallelOptimizationScheduler(tracker)
    
    try:
        scheduler.run()
    except KeyboardInterrupt:
        console.print("\n[red]Optimization interrupted by user[/red]")
        scheduler.stop_workers()
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        console.print(traceback.format_exc())
        scheduler.stop_workers()

if __name__ == "__main__":
    main()
