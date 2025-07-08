#!/usr/bin/env python3
"""
Create ensemble WITHOUT ambiguous sample weights to avoid overfitting.
Use only the corrected datasets and standard training.
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
from pathlib import Path
import hashlib
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
import json

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
CORRECTED_DATA_DIR = Path("output/")
STUDIES_DIR = Path("output/optuna_studies")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scores")

CV_FOLDS = 5
CV_RANDOM_STATE = 42
TARGET_COLUMN = "Personality"

console = Console()

def get_study_name(model_name: str, dataset_name: str) -> str:
    """Generate study name"""
    base = f"{model_name}_{dataset_name}_{CV_FOLDS}fold"
    return hashlib.md5(base.encode()).hexdigest()[:12]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features WITHOUT complex engineering to reduce overfitting"""
    df = df.copy()
    
    # Simple null indicators only
    null_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    for col in null_cols:
        df[f'{col}_is_null'] = df[col].isnull().astype(float)
    
    df['null_count'] = df[null_cols].isnull().sum(axis=1)
    
    return df

def load_and_preprocess_data(dataset_name: str):
    """Load data WITHOUT ambiguous sample handling"""
    train_path = CORRECTED_DATA_DIR / dataset_name
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Features
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency']
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    
    # Simple imputation - use median instead of mean for robustness
    for col in numerical_cols:
        if col in train_df.columns:
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
    
    # Convert categorical
    for col in categorical_cols:
        if col in train_df.columns:
            mapping = {'Yes': 1, 'No': 0}
            train_df[col] = train_df[col].map(mapping)
            test_df[col] = test_df[col].map(mapping)
            
            # Fill with mode
            mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 0
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)
    
    # Create simple features
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # Target
    train_df['target'] = (train_df[TARGET_COLUMN] == 'Extrovert').astype(int)
    
    # Feature columns - keep it simple
    feature_cols = numerical_cols + categorical_cols + [f'{col}_is_null' for col in numerical_cols] + ['null_count']
    feature_cols = [col for col in feature_cols if col in train_df.columns]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    
    return X_train, y_train, X_test, test_df

def create_robust_model(model_type: str, params: dict):
    """Create model with reduced complexity to avoid overfitting"""
    if model_type == 'xgb':
        # Reduce complexity
        model_params = {
            'n_estimators': min(params.get('n_estimators', 100), 200),  # Cap at 200
            'learning_rate': max(params.get('learning_rate', 0.01), 0.05),  # Higher LR
            'max_depth': min(params.get('max_depth', 6), 4),  # Max depth 4
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'reg_lambda': max(params.get('reg_lambda', 1.0), 5.0),  # More regularization
            'reg_alpha': max(params.get('reg_alpha', 0.0), 2.0),
            'gamma': max(params.get('gamma', 0.0), 1.0),
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'use_label_encoder': False,
            'random_state': CV_RANDOM_STATE
        }
        return xgb.XGBClassifier(**model_params)
        
    elif model_type == 'gbm':
        model_params = {
            'n_estimators': min(params.get('n_estimators', 100), 200),
            'learning_rate': max(params.get('learning_rate', 0.01), 0.05),
            'max_depth': min(params.get('max_depth', -1), 4) if params.get('max_depth', -1) > 0 else 4,
            'num_leaves': min(params.get('num_leaves', 31), 15),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'reg_lambda': max(params.get('reg_lambda', 1.0), 5.0),
            'reg_alpha': max(params.get('reg_alpha', 0.0), 2.0),
            'device': 'gpu',
            'gpu_device_id': 0,
            'objective': 'binary',
            'metric': 'binary_error',
            'verbosity': -1,
            'random_state': CV_RANDOM_STATE
        }
        return lgb.LGBMClassifier(**model_params)
        
    elif model_type == 'cat':
        model_params = {
            'iterations': min(params.get('n_estimators', 100), 200),
            'learning_rate': max(params.get('learning_rate', 0.03), 0.05),
            'depth': min(params.get('max_depth', 6), 4),
            'l2_leaf_reg': max(params.get('reg_lambda', 3.0), 10.0),
            'border_count': min(params.get('border_count', 128), 64),
            'random_strength': max(params.get('random_strength', 1.0), 2.0),
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

def main():
    """Create ensemble without ambiguous weights"""
    console.print(Panel.fit("🎯 Creating Robust Ensemble (No Ambiguous Weights)", 
                           style="bold green"))
    
    # Best models to use (but with reduced complexity)
    best_models = [
        {"dataset": "train_corrected_07.csv", "model_type": "gbm"},
        {"dataset": "train_corrected_04.csv", "model_type": "cat"},
        {"dataset": "train_corrected_06.csv", "model_type": "cat"},
        {"dataset": "train_corrected_01.csv", "model_type": "xgb"},
    ]
    
    # Load parameters from Optuna
    for model_info in best_models:
        study_name = get_study_name(model_info['model_type'], model_info['dataset'])
        db_path = STUDIES_DIR / f"{study_name}.db"
        
        if db_path.exists():
            try:
                storage = f"sqlite:///{db_path}"
                study = optuna.load_study(study_name=study_name, storage=storage)
                model_info['params'] = study.best_params
                model_info['optuna_score'] = study.best_value
            except:
                console.print(f"[red]Could not load study for {model_info['model_type']} on {model_info['dataset']}[/red]")
                model_info['params'] = {}
    
    # Get OOF predictions
    console.print("\n[cyan]Generating OOF predictions WITHOUT ambiguous weights...[/cyan]")
    
    all_oof_preds = []
    all_test_preds = []
    y_train_ref = None
    
    for model_info in best_models:
        console.print(f"\nProcessing {model_info['model_type']} on {model_info['dataset']}...")
        
        # Load data
        X_train, y_train, X_test, test_df = load_and_preprocess_data(model_info['dataset'])
        
        if y_train_ref is None:
            y_train_ref = y_train
        
        # Create robust model
        model = create_robust_model(model_info['model_type'], model_info.get('params', {}))
        
        # CV for OOF predictions - NO SAMPLE WEIGHTS
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
        oof_pred = np.zeros(len(y_train))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            # Train WITHOUT sample weights
            if model_info['model_type'] == 'cat':
                model.fit(X_fold_train, y_fold_train)
            else:
                model.fit(X_fold_train, y_fold_train)
            
            oof_pred[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        
        oof_score = accuracy_score(y_train, (oof_pred > 0.5).astype(int))
        console.print(f"  OOF Score (no weights): {oof_score:.6f}")
        model_info['oof_score'] = oof_score
        
        all_oof_preds.append(oof_pred)
        
        # Train final model on full data
        model.fit(X_train, y_train)
        test_pred = model.predict_proba(X_test)[:, 1]
        all_test_preds.append(test_pred)
    
    # Optimize ensemble weights
    console.print("\n[yellow]Optimizing ensemble weights...[/yellow]")
    
    all_oof_array = np.column_stack(all_oof_preds)
    
    def ensemble_score(weights):
        weights = weights / weights.sum()
        ensemble_pred = np.average(all_oof_array, weights=weights, axis=1)
        score = accuracy_score(y_train_ref, (ensemble_pred > 0.5).astype(int))
        return -score
    
    initial_weights = np.ones(len(best_models)) / len(best_models)
    bounds = [(0.0, 1.0) for _ in range(len(best_models))]
    
    result = minimize(
        ensemble_score,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    optimal_weights = result.x / result.x.sum()
    ensemble_oof_score = -result.fun
    
    console.print(f"\n[green]Ensemble OOF Score (no ambig weights): {ensemble_oof_score:.6f}[/green]")
    
    # Display results
    table = Table(title="Models Without Ambiguous Weights", box=box.ROUNDED)
    table.add_column("Dataset", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Original CV", style="yellow")
    table.add_column("New OOF", style="green")
    table.add_column("Weight", style="blue")
    
    for i, model in enumerate(best_models):
        table.add_row(
            model['dataset'].replace('train_corrected_', 'tc'),
            model['model_type'].upper(),
            f"{model.get('optuna_score', 0):.6f}",
            f"{model['oof_score']:.6f}",
            f"{optimal_weights[i]:.4f}"
        )
    
    console.print("\n")
    console.print(table)
    
    # Generate final predictions
    all_test_array = np.column_stack(all_test_preds)
    ensemble_test_pred = np.average(all_test_array, weights=optimal_weights, axis=1)
    
    # Save submissions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for threshold in [0.49, 0.50, 0.51]:
        test_pred = (ensemble_test_pred > threshold).astype(int)
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': test_pred
        })
        submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
        
        filename = f"subm-{ensemble_oof_score:.6f}-{timestamp}-robust-no-ambig-t{threshold}.csv"
        submission.to_csv(SCORES_DIR / filename, index=False)
        
        if threshold == 0.50:
            console.print(f"\n[green]✓ Main submission: {filename}[/green]")
    
    console.print("\n[bold green]✅ Robust ensemble complete![/bold green]")
    console.print("This should have much less overfitting because:")
    console.print("  • NO ambiguous sample weights")
    console.print("  • Reduced model complexity")
    console.print("  • Simpler feature engineering")
    console.print("  • More regularization")

if __name__ == "__main__":
    main()