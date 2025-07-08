#!/usr/bin/env python3
"""
Create simple, robust models to reduce overfitting.
Focus on fewer features and simpler models.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
CORRECTED_DATA_DIR = Path("output/")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scores")

CV_FOLDS = 5
CV_RANDOM_STATE = 42
TARGET_COLUMN = "Personality"

console = Console()

def load_data_simple(dataset_name: str):
    """Load data with minimal preprocessing"""
    # Load data
    train_path = CORRECTED_DATA_DIR / dataset_name
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Only the most important features (based on previous analysis)
    important_features = [
        'Drained_after_socializing',
        'Stage_fear', 
        'Time_spent_Alone',
        'Social_event_attendance',
        'Friends_circle_size'
    ]
    
    # Simple preprocessing
    for df in [train_df, test_df]:
        # Convert categorical to numeric
        df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
        df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
        
        # Simple mean imputation
        for col in important_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
    
    # Target
    train_df['target'] = (train_df[TARGET_COLUMN] == 'Extrovert').astype(int)
    
    X_train = train_df[important_features]
    y_train = train_df['target']
    X_test = test_df[important_features]
    
    return X_train, y_train, X_test, test_df

def create_simple_models():
    """Create simple, robust models"""
    models = {
        'xgb_simple': xgb.XGBClassifier(
            n_estimators=100,        # Fewer trees
            max_depth=3,             # Shallower
            learning_rate=0.1,       # Higher learning rate
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,          # More regularization
            reg_alpha=2.0,           # More L1 regularization
            gamma=1.0,               # More conservative splits
            tree_method='gpu_hist',
            gpu_id=0,
            random_state=CV_RANDOM_STATE
        ),
        
        'gbm_simple': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            num_leaves=7,            # Fewer leaves
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            reg_alpha=2.0,
            min_child_samples=50,    # Require more samples
            device='gpu',
            gpu_device_id=0,
            verbosity=-1,
            random_state=CV_RANDOM_STATE
        ),
        
        'cat_simple': cb.CatBoostClassifier(
            iterations=100,
            depth=3,
            learning_rate=0.1,
            l2_leaf_reg=10.0,        # High regularization
            border_count=32,         # Fewer borders
            random_strength=2.0,     # More randomness
            bagging_temperature=1.0,
            task_type='GPU',
            devices='0',
            random_state=CV_RANDOM_STATE,
            verbose=False
        )
    }
    
    return models

def evaluate_models():
    """Evaluate simple models on different datasets"""
    console.print(Panel.fit("🎯 Creating Simple, Robust Models", style="bold cyan"))
    
    datasets = [
        'train_corrected_07.csv',  # Best comprehensive
        'train_corrected_04.csv',  # Psychological contradictions
        'train_corrected_06.csv'   # Conservative combined
    ]
    
    models = create_simple_models()
    results = []
    
    for dataset in datasets:
        console.print(f"\n[yellow]Processing {dataset}...[/yellow]")
        X_train, y_train, X_test, test_df = load_data_simple(dataset)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
        
        for model_name, model in models.items():
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # Train
                if 'cat' in model_name:
                    model.fit(X_fold_train, y_fold_train)
                else:
                    model.fit(X_fold_train, y_fold_train)
                
                # Evaluate
                val_pred = model.predict(X_fold_val)
                score = accuracy_score(y_fold_val, val_pred)
                scores.append(score)
            
            cv_score = np.mean(scores)
            cv_std = np.std(scores)
            
            results.append({
                'dataset': dataset,
                'model': model_name,
                'cv_score': cv_score,
                'cv_std': cv_std,
                'features': len(X_train.columns)
            })
            
            console.print(f"  {model_name}: {cv_score:.6f} ± {cv_std:.6f}")
    
    # Display results
    table = Table(title="Simple Models Results", box=box.ROUNDED)
    table.add_column("Dataset", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("CV Score", style="green")
    table.add_column("Std Dev", style="yellow")
    
    for r in sorted(results, key=lambda x: x['cv_score'], reverse=True)[:10]:
        table.add_row(
            r['dataset'].replace('train_corrected_', 'tc'),
            r['model'],
            f"{r['cv_score']:.6f}",
            f"±{r['cv_std']:.6f}"
        )
    
    console.print("\n")
    console.print(table)
    
    # Generate best submission
    best_result = max(results, key=lambda x: x['cv_score'])
    console.print(f"\n[green]Best simple model: {best_result['model']} on {best_result['dataset']}[/green]")
    console.print(f"CV Score: {best_result['cv_score']:.6f}")
    
    # Train final model
    X_train, y_train, X_test, test_df = load_data_simple(best_result['dataset'])
    model = create_simple_models()[best_result['model']]
    model.fit(X_train, y_train)
    
    # Generate predictions
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Save submissions with different thresholds
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for threshold in [0.49, 0.50, 0.51]:
        test_pred = (test_pred_proba > threshold).astype(int)
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': test_pred
        })
        submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
        
        dataset_short = best_result['dataset'].replace('train_corrected_', 'tc').replace('.csv', '')
        filename = f"subm-{best_result['cv_score']:.6f}-{timestamp}-simple-{best_result['model']}-{dataset_short}-t{threshold}.csv"
        submission.to_csv(SCORES_DIR / filename, index=False)
        
        if threshold == 0.50:
            console.print(f"\n[green]✓ Saved: {filename}[/green]")

def create_very_simple_xgb():
    """Create ultra-simple XGBoost like the original winner"""
    console.print("\n[cyan]Creating ultra-simple XGBoost (5 trees, depth 2)...[/cyan]")
    
    # Use original training data (not corrected)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Only 2 most important features
    features = ['Drained_after_socializing', 'Stage_fear']
    
    # Preprocess
    for df in [train_df, test_df]:
        df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
        df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
        
        for col in features:
            df[col] = df[col].fillna(0.5)  # Neutral value for missing
    
    # Prepare data
    X_train = train_df[features]
    y_train = (train_df['Personality'] == 'Extrovert').astype(int)
    X_test = test_df[features]
    
    # Ultra-simple model
    model = xgb.XGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.3,
        random_state=CV_RANDOM_STATE
    )
    
    # CV evaluation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        val_pred = model.predict(X_train.iloc[val_idx])
        scores.append(accuracy_score(y_train.iloc[val_idx], val_pred))
    
    cv_score = np.mean(scores)
    console.print(f"Ultra-simple XGBoost CV: {cv_score:.6f}")
    
    # Train final and save
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': test_pred
    })
    submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"subm-{cv_score:.6f}-{timestamp}-ultra-simple-xgb.csv"
    submission.to_csv(SCORES_DIR / filename, index=False)
    console.print(f"[green]✓ Saved ultra-simple: {filename}[/green]")

def main():
    """Run all simple model strategies"""
    evaluate_models()
    create_very_simple_xgb()
    
    console.print("\n[bold green]✅ Simple models complete![/bold green]")
    console.print("These should have less overfitting due to:")
    console.print("  • Fewer features (only 2-5)")
    console.print("  • Shallower trees (depth 2-3)")
    console.print("  • More regularization")
    console.print("  • Fewer trees (5-100)")

if __name__ == "__main__":
    main()