#!/usr/bin/env python3
"""
Universal Kaggle competition optimizer with fixed resource assignment.
Each model can be assigned to specific GPU(s) or CPU.
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
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from queue import Empty
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION PARAMETERS (UPPER_CASE)
# ==============================================================================

# Data paths
DATA_DIR = "/mnt/ml/kaggle/playground-series-s5e7/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_TEMPLATE = "sample_submission.csv"

# Output directory
SCORES_DIR = "scores"

# Optimization parameters
SINGLE_MODEL_TRIALS = 500  # Number of trials for fine-tuning
# Always continue from previous optimization

# Fixed Resource Assignment
# Format: 'model': ['resource1', 'resource2', ...]
# Resources can be: 'gpu0', 'gpu1', 'cpu'
MODEL_RESOURCE_ASSIGNMENT = {
    'xgb': ['gpu0', 'gpu1'],  # XGBoost on both GPUs (CPU is 120x slower)
    'gbm': ['gpu0', 'gpu1'],  # LightGBM on both GPUs
    'cat': ['cpu']            # CatBoost on CPU
}

# Alternative example configurations:
# All models share all resources:
# MODEL_RESOURCE_ASSIGNMENT = {
#     'xgb': ['gpu0', 'gpu1', 'cpu'],
#     'gbm': ['gpu0', 'gpu1', 'cpu'],
#     'cat': ['gpu0', 'gpu1', 'cpu']
# }

# Each model gets dedicated resource:
# MODEL_RESOURCE_ASSIGNMENT = {
#     'xgb': ['gpu0'],
#     'gbm': ['gpu1'],
#     'cat': ['cpu']
# }

# Model configuration
MODEL_CONFIG = {
    'xgb': {'enabled': True},   # Focus on XGBoost only
    'gbm': {'enabled': False},  # Disabled
    'cat': {'enabled': False}   # Disabled
}

# Cross-validation
CV_FOLDS = 5
CV_RANDOM_STATE = 42

# Competition specific parameters (to be generalized later)
TARGET_COLUMN = "Personality"
ID_COLUMN = "id"

# Features for ambiguous detection (competition specific)
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
    model_type: str
    resource: str  # 'gpu0', 'gpu1', 'cpu'
    trial_number: int
    trial_params: Dict[str, Any]
    study_name: str

@dataclass
class WorkerResult:
    """Result from worker process"""
    task_id: int
    model_type: str
    resource: str
    trial_number: int
    success: bool
    score: Optional[float] = None
    predictions: Optional[List[int]] = None
    error: Optional[str] = None
    duration: Optional[float] = None

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_resource(resource: str) -> Tuple[str, Optional[int]]:
    """Parse resource string into type and ID"""
    if resource == 'cpu':
        return 'cpu', None
    elif resource.startswith('gpu'):
        gpu_id = int(resource[3:])
        return 'gpu', gpu_id
    else:
        raise ValueError(f"Unknown resource format: {resource}")

def create_directories():
    """Create necessary directories"""
    os.makedirs(SCORES_DIR, exist_ok=True)

def get_study_name(model_name):
    """Generate unique study name for Optuna"""
    base = f"{model_name}_{CV_FOLDS}fold"
    return hashlib.md5(base.encode()).hexdigest()[:12]

def load_data():
    """Load competition data"""
    print(f"\nLoading data from {DATA_DIR}")
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))
    
    # Load ambiguous cases if available
    ambiguous_file = os.path.join(DATA_DIR, "most_ambiguous_2.43pct.csv")
    if os.path.exists(ambiguous_file):
        ambiguous_df = pd.read_csv(ambiguous_file)
        ambiguous_ids = set(ambiguous_df['id'].values)
        print(f"Loaded {len(ambiguous_ids)} ambiguous training samples")
    else:
        ambiguous_ids = set()
        print("No ambiguous samples file found")
    
    return train_df, test_df, ambiguous_ids

def preprocess_data(train_df, test_df):
    """Preprocess data (competition specific)"""
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
        (df['Friends_circle_size'] - 6.18)**2
    )
    
    df_feat['ambiguity_score'] = 1 / (1 + df_feat['dist_to_ambiguous'])
    
    # High ambiguity flag
    df_feat['high_ambiguity'] = (df_feat['ambiguity_score'] > ambig_score_thresh).astype(float)
    
    return df_feat, df_feat['ambiguity_score'] > ambig_score_thresh

def save_submission(predictions, score, model_name, study_name, trial_number):
    """Save submission file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"subm-{score:.5f}-{timestamp}-{model_name}-{study_name}-{trial_number}.csv"
    filepath = os.path.join(SCORES_DIR, filename)
    
    # Load test data for ID column
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))
    
    # Create submission
    submission = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        TARGET_COLUMN: predictions
    })
    
    # Map back to original labels
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].map({1: 'Extrovert', 0: 'Introvert'})
    
    submission.to_csv(filepath, index=False)
    print(f"\nSaved submission: {filename}")
    
    return filepath

# ==============================================================================
# WORKER PROCESS FUNCTIONS
# ==============================================================================

def set_gpu_environment(resource: str):
    """Set CUDA environment based on resource"""
    resource_type, gpu_id = parse_resource(resource)
    
    if resource_type == 'gpu' and gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # No GPU for CPU tasks

def get_model_params(model_type, params_dict, resource):
    """Get model parameters with GPU/CPU configuration"""
    resource_type, gpu_id = parse_resource(resource)
    use_gpu = (resource_type == 'gpu')
    
    if model_type == "xgb":
        params = params_dict.copy()
        params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': CV_RANDOM_STATE,
            'use_label_encoder': False,
            'verbosity': 0
        })
        if use_gpu:
            params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'gpu_id': 0  # Always 0 after CUDA_VISIBLE_DEVICES is set
            })
        else:
            params.update({
                'tree_method': 'hist',
                'predictor': 'cpu_predictor'
            })
    
    elif model_type == "gbm":
        params = params_dict.copy()
        params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': CV_RANDOM_STATE,
            'verbosity': -1
        })
        if use_gpu:
            params['device'] = 'gpu'
            params['gpu_device_id'] = 0  # Always 0 after CUDA_VISIBLE_DEVICES is set
        else:
            params['device'] = 'cpu'
    
    elif model_type == "cat":
        params = params_dict.copy()
        params.update({
            'random_seed': CV_RANDOM_STATE,
            'verbose': False
        })
        if use_gpu:
            params['task_type'] = 'GPU'
            params['devices'] = '0'  # Always 0 after CUDA_VISIBLE_DEVICES is set
        else:
            params['task_type'] = 'CPU'
    
    return params

def create_model(model_type, params):
    """Create model instance based on type"""
    if model_type == "xgb":
        return xgb.XGBClassifier(**params)
    elif model_type == "gbm":
        return lgb.LGBMClassifier(**params)
    elif model_type == "cat":
        return cb.CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def worker_process(task_queue: mp.Queue, result_queue: mp.Queue, data_dict: dict):
    """Worker process that pulls tasks from queue and trains models"""
    
    # Unpack shared data
    train_df = data_dict['train_df']
    test_df = data_dict['test_df']
    ambiguous_ids = data_dict['ambiguous_ids']
    features = data_dict['features']
    
    while True:
        try:
            # Get task from queue (blocking with timeout)
            task = task_queue.get(timeout=1)
            
            if task == "STOP":
                break
                
            start_time = time.time()
            
            # Set GPU environment
            set_gpu_environment(task.resource)
            
            print(f"\nWorker: Starting {task.model_type} trial {task.trial_number} on {task.resource}")
            
            try:
                # Extract parameters
                trial_params = task.trial_params
                
                # Ambiguity parameters
                alone_thresh = trial_params['alone_thresh']
                social_thresh = trial_params['social_thresh']
                friends_thresh = trial_params['friends_thresh']
                ambig_score_thresh = trial_params['ambig_score_thresh']
                proba_low_thresh = trial_params['proba_low_thresh']
                proba_high_thresh = trial_params['proba_high_thresh']
                ambig_weight = trial_params['ambig_weight']
                
                # Model parameters
                model_params_dict = {k: v for k, v in trial_params.items() 
                                    if k not in ['alone_thresh', 'social_thresh', 'friends_thresh', 
                                               'ambig_score_thresh', 'proba_low_thresh', 
                                               'proba_high_thresh', 'ambig_weight']}
                
                model_params = get_model_params(task.model_type, model_params_dict, task.resource)
                
                # Create features
                X_train, train_high_ambig = create_ambiguity_features(
                    train_df[features], alone_thresh, social_thresh, friends_thresh, ambig_score_thresh
                )
                X_test, test_high_ambig = create_ambiguity_features(
                    test_df[features], alone_thresh, social_thresh, friends_thresh, ambig_score_thresh
                )
                
                # Add original features
                extended_features = features + ['ambiguous_pattern', 'dist_to_ambiguous', 
                                               'ambiguity_score', 'high_ambiguity']
                
                X = X_train[extended_features]
                y = train_df[TARGET_COLUMN]
                X_test_final = X_test[extended_features]
                
                # Sample weights
                sample_weights = np.ones(len(train_df))
                sample_weights[train_df[ID_COLUMN].isin(ambiguous_ids)] = ambig_weight
                
                # Cross-validation
                cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
                cv_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                    X_train_fold, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_fold, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    weights_train = sample_weights[train_idx]
                    
                    # Create and train model
                    model = create_model(task.model_type, model_params)
                    
                    if task.model_type == "cat":
                        model.fit(X_train_fold, y_train_fold, sample_weight=weights_train,
                                 eval_set=(X_val, y_val), early_stopping_rounds=50)
                    else:
                        model.fit(X_train_fold, y_train_fold, sample_weight=weights_train)
                    
                    # Predictions
                    proba = model.predict_proba(X_val)[:, 1]
                    pred = (proba > proba_high_thresh).astype(int)
                    
                    # Apply ambiguous rule
                    val_high_ambig = train_high_ambig.iloc[val_idx]
                    pred[val_high_ambig] = 1  # Force to Extrovert
                    
                    # Revert very low probability
                    very_low_prob = (proba < proba_low_thresh) & val_high_ambig
                    pred[very_low_prob] = 0
                    
                    accuracy = (pred == y_val.values).mean()
                    cv_scores.append(accuracy)
                
                mean_score = np.mean(cv_scores)
                
                # Train final model on full data for submission
                final_model = create_model(task.model_type, model_params)
                if task.model_type == "cat":
                    final_model.fit(X, y, sample_weight=sample_weights)
                else:
                    final_model.fit(X, y, sample_weight=sample_weights)
                
                # Make predictions
                proba_test = final_model.predict_proba(X_test_final)[:, 1]
                predictions = (proba_test > proba_high_thresh).astype(int)
                predictions[test_high_ambig] = 1
                very_intro = (proba_test < proba_low_thresh) & test_high_ambig
                predictions[very_intro] = 0
                
                duration = time.time() - start_time
                
                # Send result
                result = WorkerResult(
                    task_id=task.task_id,
                    model_type=task.model_type,
                    resource=task.resource,
                    trial_number=task.trial_number,
                    success=True,
                    score=mean_score,
                    predictions=predictions.tolist(),
                    duration=duration
                )
                
                result_queue.put(result)
                print(f"Worker: Completed {task.model_type} trial {task.trial_number} "
                      f"on {task.resource} - Score: {mean_score:.6f} ({duration:.1f}s)")
                
            except Exception as e:
                # Send error result
                result = WorkerResult(
                    task_id=task.task_id,
                    model_type=task.model_type,
                    resource=task.resource,
                    trial_number=task.trial_number,
                    success=False,
                    error=str(e) + "\n" + traceback.format_exc()
                )
                result_queue.put(result)
                print(f"Worker: Error in {task.model_type} trial {task.trial_number}: {e}")
                
        except Empty:
            # No task available, continue waiting
            continue
        except Exception as e:
            print(f"Worker: Fatal error: {e}")
            traceback.print_exc()
            break

# ==============================================================================
# FIXED RESOURCE SCHEDULER
# ==============================================================================

class FixedResourceScheduler:
    """Manages allocation of models to fixed assigned resources"""
    
    def __init__(self, train_df, test_df, ambiguous_ids, features):
        self.train_df = train_df
        self.test_df = test_df
        self.ambiguous_ids = ambiguous_ids
        self.features = features
        
        # Extract all unique resources from assignments
        all_resources = set()
        for resources in MODEL_RESOURCE_ASSIGNMENT.values():
            all_resources.update(resources)
        
        # Resource tracking
        self.resource_status = {resource: 'free' for resource in all_resources}
        
        # Model tracking
        self.model_status = {model: 'free' for model in MODEL_CONFIG.keys() if MODEL_CONFIG[model]['enabled']}
        self.model_trials = {model: 0 for model in self.model_status.keys()}
        self.model_resource_count = {model: {res: 0 for res in all_resources} for model in self.model_status.keys()}
        
        # Running tasks
        self.running_tasks = {}  # task_id -> (model, resource, start_time)
        self.next_task_id = 0
        
        # Optuna studies
        self.studies = {}
        self.best_scores = {}
        
        # Initialize studies
        for model_type in self.model_status.keys():
            self._init_study(model_type)
        
        # Process management
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        
        # Prepare data for workers
        self.data_dict = {
            'train_df': train_df,
            'test_df': test_df,
            'ambiguous_ids': ambiguous_ids,
            'features': features
        }
        
        print("\nResource configuration:")
        for model, resources in MODEL_RESOURCE_ASSIGNMENT.items():
            if model in self.model_status:
                print(f"  {model}: {', '.join(resources)}")
        print(f"  Total unique resources: {len(all_resources)}")
        
    def _init_study(self, model_type):
        """Initialize or load Optuna study for a model"""
        study_name = get_study_name(model_type)
        journal_path = f"study-{model_type}-finetune.journal"
        
        # Use JournalStorage for better crash recovery
        from optuna.storages import JournalStorage, JournalFileStorage
        storage = JournalStorage(JournalFileStorage(journal_path))
        
        # Always continue from existing study
        print(f"\nOptimization for {model_type.upper()}...")
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage
            )
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) > 0:
                print(f"Continuing from trial {len(study.trials)}, best score: {study.best_value:.6f}")
                self.best_scores[model_type] = study.best_value
            else:
                print(f"Loaded study with {len(study.trials)} trials (0 completed)")
                self.best_scores[model_type] = -np.inf
        except:
            # Create new if doesn't exist
            print(f"Creating new study...")
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction='maximize'
            )
            self.best_scores[model_type] = -np.inf
        
        self.studies[model_type] = study
    
    def start_workers(self):
        """Start worker processes"""
        num_workers = len(self.resource_status)
        for i in range(num_workers):
            p = mp.Process(target=worker_process, 
                          args=(self.task_queue, self.result_queue, self.data_dict))
            p.start()
            self.workers.append(p)
        print(f"\nStarted {num_workers} worker processes")
    
    def stop_workers(self):
        """Stop all worker processes"""
        for _ in self.workers:
            self.task_queue.put("STOP")
        
        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        print("\nStopped all worker processes")
    
    def _get_free_resource_for_model(self, model_type):
        """Get free resource from model's assigned resources"""
        assigned_resources = MODEL_RESOURCE_ASSIGNMENT.get(model_type, [])
        
        for resource in assigned_resources:
            if self.resource_status.get(resource) == 'free':
                return resource
        return None
    
    def _get_next_model(self):
        """Get next model to train based on trials count"""
        # Find model with least trials that's free and has available resources
        available_models = []
        
        for model, status in self.model_status.items():
            if status == 'free' and self._get_free_resource_for_model(model) is not None:
                available_models.append(model)
        
        if not available_models:
            return None
        
        # Return model with minimum trials
        return min(available_models, key=lambda m: self.model_trials[m])
    
    def _create_task(self, model_type, resource):
        """Create a new training task"""
        study = self.studies[model_type]
        
        # Create new trial
        trial = study.ask()
        
        # Suggest parameters
        trial_params = {
            'alone_thresh': trial.suggest_float('alone_thresh', 2.0, 3.5),
            'social_thresh': trial.suggest_float('social_thresh', 3.5, 5.5),
            'friends_thresh': trial.suggest_int('friends_thresh', 6, 9),
            'ambig_score_thresh': trial.suggest_float('ambig_score_thresh', 0.2, 0.4),
            'proba_low_thresh': trial.suggest_float('proba_low_thresh', 0.1, 0.25),
            'proba_high_thresh': trial.suggest_float('proba_high_thresh', 0.45, 0.55),
            'ambig_weight': trial.suggest_float('ambig_weight', 5, 20)
        }
        
        # Add model-specific parameters
        if model_type == "xgb":
            trial_params.update({
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 10.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            })
        elif model_type == "gbm":
            trial_params.update({
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 10.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 1, 10)
            })
        elif model_type == "cat":
            trial_params.update({
                'iterations': trial.suggest_int('iterations', 500, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 10)
            })
        
        # Create task
        task = WorkerTask(
            task_id=self.next_task_id,
            model_type=model_type,
            resource=resource,
            trial_number=trial.number,
            trial_params=trial_params,
            study_name=get_study_name(model_type)
        )
        
        self.next_task_id += 1
        return task, trial
    
    def schedule_next_task(self):
        """Schedule next task if resources and models are available"""
        # Get next model
        model_type = self._get_next_model()
        if model_type is None:
            return False
        
        # Check if model reached trial limit
        if self.model_trials[model_type] >= SINGLE_MODEL_TRIALS:
            return False
        
        # Get resource for this model
        resource = self._get_free_resource_for_model(model_type)
        if resource is None:
            return False
        
        # Create and schedule task
        task, trial = self._create_task(model_type, resource)
        
        # Update status
        self.resource_status[resource] = 'busy'
        self.model_status[model_type] = 'busy'
        self.model_trials[model_type] += 1
        self.model_resource_count[model_type][resource] += 1
        self.running_tasks[task.task_id] = (model_type, resource, time.time(), trial)
        
        # Send to queue
        self.task_queue.put(task)
        
        print(f"\nScheduled: {model_type} trial {task.trial_number} on {resource}")
        return True
    
    def process_results(self):
        """Process results from workers"""
        try:
            while True:
                # Non-blocking get
                result = self.result_queue.get_nowait()
                
                # Get task info
                if result.task_id in self.running_tasks:
                    model_type, resource, start_time, trial = self.running_tasks[result.task_id]
                    
                    # Free resources
                    self.resource_status[resource] = 'free'
                    self.model_status[model_type] = 'free'
                    del self.running_tasks[result.task_id]
                    
                    if result.success:
                        # Update Optuna
                        study = self.studies[model_type]
                        study.tell(trial, result.score)
                        
                        # Check if new best
                        if result.score > self.best_scores[model_type]:
                            self.best_scores[model_type] = result.score
                            print(f"\nNew best score for {model_type}: {result.score:.6f}")
                            
                            # Save submission
                            predictions = np.array(result.predictions)
                            save_submission(predictions, result.score, model_type,
                                          get_study_name(model_type), result.trial_number)
                    else:
                        # Failed trial
                        study = self.studies[model_type]
                        study.tell(trial, None, state=optuna.trial.TrialState.FAIL)
                        print(f"\nFailed trial for {model_type}: {result.error}")
                        
        except Empty:
            pass
    
    def run(self):
        """Main optimization loop"""
        print("\n" + "="*80)
        print("FIXED RESOURCE OPTIMIZATION")
        print("="*80)
        
        self.start_workers()
        start_time = time.time()
        last_status_time = time.time()
        
        try:
            while True:
                # Check for STOP file
                if os.path.exists('STOP'):
                    print("\nSTOP file detected - finishing current tasks...")
                    break
                
                # Check if all models completed their trials
                all_done = all(trials >= SINGLE_MODEL_TRIALS 
                             for trials in self.model_trials.values())
                if all_done:
                    print("\nAll models completed their trials")
                    break
                
                # Schedule new tasks
                while self.schedule_next_task():
                    pass  # Keep scheduling until no more slots
                
                # Process results
                self.process_results()
                
                # Brief sleep to avoid busy waiting
                time.sleep(0.1)
                
                # Status update every 30 seconds
                current_time = time.time()
                if current_time - last_status_time >= 30:
                    running = len(self.running_tasks)
                    print(f"\nStatus: {running} tasks running, Trials: {self.model_trials}")
                    
                    # Resource usage by model
                    print("Resource usage by model:")
                    for model in self.model_status.keys():
                        resource_usage = []
                        for resource, count in self.model_resource_count[model].items():
                            if count > 0:
                                resource_usage.append(f"{resource}:{count}")
                        if resource_usage:
                            print(f"  {model}: {', '.join(resource_usage)}")
                    
                    last_status_time = current_time
        
        finally:
            # Wait for remaining tasks
            print("\nWaiting for remaining tasks to complete...")
            while self.running_tasks:
                self.process_results()
                time.sleep(0.1)
            
            self.stop_workers()
            
            # Final summary
            print("\n" + "="*80)
            print("OPTIMIZATION COMPLETE!")
            print("="*80)
            print("\nFinal results:")
            for model, score in self.best_scores.items():
                if score > -np.inf:
                    print(f"  {model.upper()}: {score:.6f}")
            
            print("\nResource usage summary:")
            for model in self.model_status.keys():
                print(f"\n{model}:")
                total = 0
                for resource, count in sorted(self.model_resource_count[model].items()):
                    if count > 0:
                        print(f"  {resource}: {count} trials")
                        total += count
                print(f"  Total: {total} trials")
            
            total_time = time.time() - start_time
            total_trials = sum(self.model_trials.values())
            if total_trials > 0:
                print(f"\nTotal time: {total_time:.1f}s")
                print(f"Total trials: {total_trials}")
                print(f"Average time per trial: {total_time/total_trials:.1f}s")

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main entry point"""
    print("="*80)
    print("UNIVERSAL KAGGLE OPTIMIZER - FIXED RESOURCE ASSIGNMENT")
    print("="*80)
    
    # Create directories
    create_directories()
    
    # Load and preprocess data
    train_df, test_df, ambiguous_ids = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Define features
    features = AMBIGUOUS_FEATURES['numerical'] + AMBIGUOUS_FEATURES['categorical']
    
    # Create and run scheduler
    scheduler = FixedResourceScheduler(train_df, test_df, ambiguous_ids, features)
    scheduler.run()
    
    print("\nRemember to remove STOP file before next run: rm STOP")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
