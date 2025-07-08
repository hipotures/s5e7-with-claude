#!/usr/bin/env python3
"""
Create simple submissions from best Optuna models without ensemble.
Since different datasets have different sizes, we'll generate individual submissions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from pathlib import Path
import hashlib
import joblib
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Configuration
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
CORRECTED_DATA_DIR = Path("output/")
STUDIES_DIR = Path("output/optuna_studies")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scores")
MOST_AMBIGUOUS = DATA_DIR / "enhanced_ambiguous_600.csv"

CV_FOLDS = 5
CV_RANDOM_STATE = 42
TARGET_COLUMN = "Personality"

console = Console()

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
    
    # Filter to existing columns
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

def generate_submission(dataset: str, model_type: str, cv_score: float):
    """Generate submission for a single model"""
    console.print(f"\n[cyan]Processing {model_type.upper()} on {dataset}...[/cyan]")
    
    # Load best parameters from Optuna
    study_name = get_study_name(model_type, dataset)
    db_path = STUDIES_DIR / f"{study_name}.db"
    
    if not db_path.exists():
        console.print(f"[red]Study not found: {study_name}[/red]")
        return
    
    try:
        storage = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_params = study.best_params
        best_value = study.best_value
    except Exception as e:
        console.print(f"[red]Error loading study: {e}[/red]")
        return
    
    # Load and preprocess data
    X_train, y_train, X_test, test_df = load_and_preprocess_data(dataset, best_params)
    
    # Create and train model
    model = create_model_from_params(model_type, best_params)
    
    # Sample weights if using ambiguous samples
    sample_weights = np.ones(len(y_train))
    if MOST_AMBIGUOUS.exists() and 'ambig_weight' in best_params:
        ambiguous_df = pd.read_csv(MOST_AMBIGUOUS)
        ambiguous_ids = set(ambiguous_df['id'].values)
        train_ids = X_train.index
        ambiguous_mask = train_ids.isin(ambiguous_ids)
        sample_weights[ambiguous_mask] = best_params['ambig_weight']
    
    # Train model
    console.print(f"Training {model_type.upper()}...")
    if model_type == 'cat':
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Generate predictions
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    thresholds = [0.48, 0.49, 0.495, 0.50, 0.505, 0.51]
    
    for threshold in thresholds:
        test_pred = (test_pred_proba > threshold).astype(int)
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': test_pred
        })
        submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
        
        # Save submission
        dataset_short = dataset.replace('train_corrected_', 'tc').replace('.csv', '')
        filename = f"subm-{cv_score:.6f}-{timestamp}-{model_type}-{dataset_short}-t{threshold}.csv"
        submission.to_csv(SCORES_DIR / filename, index=False)
        
        if threshold == 0.50:
            console.print(f"[green]✓ Saved main submission: {filename}[/green]")

def main():
    """Generate submissions for best models"""
    console.print(Panel.fit("🚀 Generating Individual Submissions from Best Models", 
                           style="bold magenta"))
    
    # Best models to generate submissions for
    best_models = [
        ("train_corrected_07.csv", "gbm", 0.975869),  # Best overall
        ("train_corrected_07.csv", "xgb", 0.975815),
        ("train_corrected_07.csv", "cat", 0.975815),
        ("train_corrected_04.csv", "cat", 0.974898),
        ("train_corrected_06.csv", "cat", 0.974844),
    ]
    
    # Create table
    table = Table(title="Models to Generate Submissions", box="rounded")
    table.add_column("Dataset", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("CV Score", style="green")
    table.add_column("Gap", style="red")
    
    for dataset, model, score in best_models:
        gap = 0.975708 - score
        gap_str = f"+{-gap:.6f}" if gap < 0 else f"-{gap:.6f}"
        table.add_row(
            dataset.replace('train_corrected_', 'tc'),
            model.upper(),
            f"{score:.6f}",
            gap_str
        )
    
    console.print("\n")
    console.print(table)
    console.print("\n")
    
    # Generate submissions
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating submissions...", total=len(best_models))
        
        for dataset, model_type, cv_score in best_models:
            generate_submission(dataset, model_type, cv_score)
            progress.update(task, advance=1)
    
    console.print("\n[bold green]✅ All submissions generated![/bold green]")
    console.print(f"Check {SCORES_DIR} for submission files.")

if __name__ == "__main__":
    main()