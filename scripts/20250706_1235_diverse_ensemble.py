#!/usr/bin/env python3
"""
Create diverse ensemble to reduce overfitting.
Mix models trained on different datasets and with different complexities.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
from pathlib import Path
import hashlib
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
import warnings

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

def create_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create only essential features"""
    df = df.copy()
    
    # Null indicators for key features
    df['Time_alone_null'] = df['Time_spent_Alone'].isnull().astype(float)
    df['Social_null'] = df['Social_event_attendance'].isnull().astype(float)
    df['null_count'] = df[['Time_spent_Alone', 'Social_event_attendance', 
                           'Going_outside', 'Friends_circle_size', 'Post_frequency']].isnull().sum(axis=1)
    
    return df

def load_data_for_ensemble(dataset_name: str, simple: bool = False):
    """Load data with option for simple preprocessing"""
    train_path = CORRECTED_DATA_DIR / dataset_name
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Basic features
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency']
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    
    # Convert categorical
    for df in [train_df, test_df]:
        df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
        df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
        
        # Simple imputation
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].mean())
        for col in categorical_cols:
            df[col] = df[col].fillna(0.5)
    
    if simple:
        # Only most important features
        feature_cols = ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone']
    else:
        # Add simple engineered features
        train_df = create_simple_features(train_df)
        test_df = create_simple_features(test_df)
        feature_cols = numerical_cols + categorical_cols + ['Time_alone_null', 'Social_null', 'null_count']
    
    # Target
    train_df['target'] = (train_df[TARGET_COLUMN] == 'Extrovert').astype(int)
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    
    return X_train, y_train, X_test, test_df

def create_diverse_models():
    """Create diverse set of models"""
    models = []
    
    # 1. Ultra-simple XGBoost (like original winner)
    models.append({
        'name': 'xgb_ultra_simple',
        'model': xgb.XGBClassifier(n_estimators=5, max_depth=2, learning_rate=0.3, random_state=42),
        'dataset': 'train.csv',  # Original data
        'simple': True
    })
    
    # 2. Simple logistic regression
    models.append({
        'name': 'logistic',
        'model': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'dataset': 'train_corrected_07.csv',
        'simple': True
    })
    
    # 3. Moderate complexity models from different datasets
    models.append({
        'name': 'xgb_moderate',
        'model': xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, 
                                  reg_lambda=5.0, random_state=42),
        'dataset': 'train_corrected_04.csv',
        'simple': False
    })
    
    models.append({
        'name': 'gbm_moderate',
        'model': lgb.LGBMClassifier(n_estimators=50, max_depth=3, num_leaves=7,
                                   learning_rate=0.1, reg_lambda=5.0, verbosity=-1, random_state=42),
        'dataset': 'train_corrected_06.csv',
        'simple': False
    })
    
    models.append({
        'name': 'cat_moderate',
        'model': cb.CatBoostClassifier(iterations=50, depth=3, learning_rate=0.1,
                                      l2_leaf_reg=10.0, verbose=False, random_state=42),
        'dataset': 'train_corrected_01.csv',
        'simple': False
    })
    
    # 4. One complex model from best Optuna results
    study_name = get_study_name('gbm', 'train_corrected_07.csv')
    db_path = STUDIES_DIR / f"{study_name}.db"
    
    if db_path.exists():
        try:
            storage = f"sqlite:///{db_path}"
            study = optuna.load_study(study_name=study_name, storage=storage)
            params = study.best_params
            
            complex_gbm = lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.01),
                max_depth=params.get('max_depth', -1),
                num_leaves=params.get('num_leaves', 31),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                reg_lambda=params.get('reg_lambda', 1.0),
                reg_alpha=params.get('reg_alpha', 0.0),
                device='gpu',
                gpu_device_id=0,
                verbosity=-1,
                random_state=CV_RANDOM_STATE
            )
            
            models.append({
                'name': 'gbm_complex',
                'model': complex_gbm,
                'dataset': 'train_corrected_07.csv',
                'simple': False
            })
        except:
            pass
    
    return models

def optimize_diverse_ensemble():
    """Create and optimize diverse ensemble"""
    console.print(Panel.fit("🌈 Creating Diverse Ensemble", style="bold magenta"))
    
    models = create_diverse_models()
    console.print(f"Created {len(models)} diverse models")
    
    # Get OOF predictions for each model
    all_oof_preds = []
    all_test_preds = []
    y_train_common = None
    
    for model_info in models:
        console.print(f"\n[cyan]Processing {model_info['name']}...[/cyan]")
        
        # Handle original train.csv
        if model_info['dataset'] == 'train.csv':
            train_df = pd.read_csv(DATA_DIR / "train.csv")
            test_df = pd.read_csv(DATA_DIR / "test.csv")
            
            # Simple preprocessing
            for df in [train_df, test_df]:
                df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
                df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
                df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].mean())
            
            X_train = train_df[['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone']]
            y_train = (train_df['Personality'] == 'Extrovert').astype(int)
            X_test = test_df[['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone']]
        else:
            X_train, y_train, X_test, test_df = load_data_for_ensemble(
                model_info['dataset'], 
                model_info['simple']
            )
        
        if y_train_common is None:
            y_train_common = y_train
        
        # Cross-validation for OOF predictions
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
        oof_pred = np.zeros(len(y_train))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            # Clone model for this fold
            if 'cat' in model_info['name']:
                fold_model = model_info['model']
                fold_model.fit(X_fold_train, y_fold_train)
            else:
                fold_model = model_info['model']
                fold_model.fit(X_fold_train, y_fold_train)
            
            # Get predictions
            if hasattr(fold_model, 'predict_proba'):
                oof_pred[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]
            else:
                oof_pred[val_idx] = fold_model.decision_function(X_fold_val)
                # Normalize to 0-1
                oof_pred[val_idx] = 1 / (1 + np.exp(-oof_pred[val_idx]))
        
        # Train on full data for test predictions
        model_info['model'].fit(X_train, y_train)
        
        if hasattr(model_info['model'], 'predict_proba'):
            test_pred = model_info['model'].predict_proba(X_test)[:, 1]
        else:
            test_pred = model_info['model'].decision_function(X_test)
            test_pred = 1 / (1 + np.exp(-test_pred))
        
        all_oof_preds.append(oof_pred)
        all_test_preds.append(test_pred)
        
        # Report OOF score
        oof_score = accuracy_score(y_train, (oof_pred > 0.5).astype(int))
        console.print(f"  OOF Score: {oof_score:.6f}")
    
    # Optimize ensemble weights using differential evolution (more robust)
    console.print("\n[yellow]Optimizing ensemble weights with differential evolution...[/yellow]")
    
    all_oof_array = np.column_stack(all_oof_preds)
    
    def ensemble_score(weights):
        weights = weights / weights.sum()
        ensemble_pred = np.average(all_oof_array, weights=weights, axis=1)
        score = accuracy_score(y_train_common, (ensemble_pred > 0.5).astype(int))
        return -score  # Minimize negative score
    
    # Use differential evolution for more robust optimization
    bounds = [(0.0, 1.0) for _ in range(len(models))]
    result = differential_evolution(ensemble_score, bounds, seed=42, maxiter=100)
    
    optimal_weights = result.x / result.x.sum()
    ensemble_oof_score = -result.fun
    
    console.print(f"\n[green]Optimal weights found![/green]")
    console.print(f"Ensemble OOF Score: {ensemble_oof_score:.6f}")
    
    # Display weights
    table = Table(title="Ensemble Weights", box=box.SIMPLE)
    table.add_column("Model", style="cyan")
    table.add_column("Dataset", style="magenta")
    table.add_column("Weight", style="green")
    
    for i, model_info in enumerate(models):
        table.add_row(
            model_info['name'],
            model_info['dataset'].replace('train_corrected_', 'tc'),
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
        
        filename = f"subm-{ensemble_oof_score:.6f}-{timestamp}-diverse-ensemble-t{threshold}.csv"
        submission.to_csv(SCORES_DIR / filename, index=False)
        
        if threshold == 0.50:
            console.print(f"\n[green]✓ Main submission: {filename}[/green]")

def main():
    """Create diverse ensemble"""
    optimize_diverse_ensemble()
    
    console.print("\n[bold green]✅ Diverse ensemble complete![/bold green]")
    console.print("This ensemble should be more robust because:")
    console.print("  • Mix of simple and complex models")
    console.print("  • Different datasets (original + corrected)")
    console.print("  • Different algorithms (XGB, GBM, CAT, Logistic)")
    console.print("  • Differential evolution for weight optimization")

if __name__ == "__main__":
    main()