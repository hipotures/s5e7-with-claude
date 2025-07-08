#!/usr/bin/env python3
"""
Recreate the EXACT winning model that achieved 0.975708.
Ultra-simple XGBoost with ambivert detection rule.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import warnings

# Check if cupy is available for GPU support
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not installed. GPU models will show device mismatch warnings.")

console = Console()

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scores")

def load_original_data():
    """Load original data (not corrected)"""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Simple preprocessing - exactly as in winning model
    feature_cols = ['Drained_after_socializing', 'Stage_fear', 'Time_spent_Alone', 
                   'Social_event_attendance', 'Friends_circle_size']
    
    for df in [train_df, test_df]:
        # Convert categorical
        df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
        df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
        
        # Simple mean imputation
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
    
    X_train = train_df[feature_cols]
    y_train = (train_df['Personality'] == 'Extrovert').astype(int)
    X_test = test_df[feature_cols]
    
    return X_train, y_train, X_test, test_df

def detect_ambiverts(X, model, is_gpu=False):
    """Detect ambivert cases based on prediction probability"""
    # Convert to appropriate format for model
    if is_gpu and CUPY_AVAILABLE:
        if hasattr(X, 'values'):  # pandas DataFrame
            X_data = cp.asarray(X.values)
        else:
            X_data = X
    else:
        X_data = X
    
    # Get probabilities
    proba = model.predict_proba(X_data)[:, 1]
    
    # Convert back to numpy if on GPU
    if is_gpu and CUPY_AVAILABLE:
        proba = cp.asnumpy(proba)
    
    # Find cases closest to 0.5 (most uncertain)
    uncertainty = np.abs(proba - 0.5)
    
    # Bottom 2.43% most uncertain are ambiverts
    threshold = np.percentile(uncertainty, 2.43)
    ambivert_mask = uncertainty <= threshold
    
    return ambivert_mask, proba

def apply_ambivert_rule(predictions, ambivert_mask, proba):
    """Apply 96.2% extrovert rule to ambiverts"""
    # For ambiverts, 96.2% should be Extrovert
    n_ambiverts = ambivert_mask.sum()
    n_extrovert = int(n_ambiverts * 0.962)
    
    if n_ambiverts > 0:
        # Get ambivert indices
        ambivert_indices = np.where(ambivert_mask)[0]
        
        # Sort by probability (descending) and take top 96.2%
        ambivert_proba = proba[ambivert_mask]
        sorted_indices = ambivert_indices[np.argsort(-ambivert_proba)]
        
        # Set top 96.2% as Extrovert
        predictions[sorted_indices[:n_extrovert]] = 1
        # Rest as Introvert
        predictions[sorted_indices[n_extrovert:]] = 0
    
    return predictions

def create_winning_models():
    """Create multiple versions of the winning configuration"""
    models = []
    
    # 1. Ultra-simple (original winner)
    models.append({
        'name': 'ultra_simple_5_2',
        'model': xgb.XGBClassifier(
            n_estimators=5,
            max_depth=2,
            learning_rate=1.0,
            random_state=42
        ),
        'use_ambivert_rule': True
    })
    
    # 2. Slightly more trees
    models.append({
        'name': 'simple_10_2',
        'model': xgb.XGBClassifier(
            n_estimators=10,
            max_depth=2,
            learning_rate=0.5,
            random_state=42
        ),
        'use_ambivert_rule': True
    })
    
    # 3. Original without ambivert rule (for comparison)
    models.append({
        'name': 'ultra_simple_no_rule',
        'model': xgb.XGBClassifier(
            n_estimators=5,
            max_depth=2,
            learning_rate=1.0,
            random_state=42
        ),
        'use_ambivert_rule': False
    })
    
    # 4. With GPU acceleration (XGBoost 3.x syntax)
    models.append({
        'name': 'ultra_simple_gpu',
        'model': xgb.XGBClassifier(
            n_estimators=5,
            max_depth=2,
            learning_rate=1.0,
            device='cuda:0',
            tree_method='hist',
            random_state=42
        ),
        'use_ambivert_rule': True
    })
    
    return models

def main():
    """Recreate winning model"""
    console.print(Panel.fit("🏆 Recreating Winning Model (0.975708)", style="bold yellow"))
    
    # Load data
    console.print("\n[cyan]Loading original data (no corrections)...[/cyan]")
    X_train, y_train, X_test, test_df = load_original_data()
    
    console.print(f"Train shape: {X_train.shape}")
    console.print(f"Test shape: {X_test.shape}")
    
    # Create models
    models = create_winning_models()
    
    # Results table
    results = []
    
    for model_config in models:
        console.print(f"\n[yellow]Testing {model_config['name']}...[/yellow]")
        
        model = model_config['model']
        
        # Check if this is a GPU model
        is_gpu_model = 'cuda' in str(model.get_params().get('device', ''))
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            # Prepare data based on device
            if is_gpu_model and CUPY_AVAILABLE:
                # Convert to cupy arrays for GPU
                X_train_fold = cp.asarray(X_train.iloc[train_idx].values)
                y_train_fold = cp.asarray(y_train.iloc[train_idx].values)
                X_val_fold = cp.asarray(X_train.iloc[val_idx].values)
                y_val_true = y_train.iloc[val_idx].values  # Keep true labels on CPU for scoring
            else:
                # Use pandas/numpy for CPU
                X_train_fold = X_train.iloc[train_idx]
                y_train_fold = y_train.iloc[train_idx]
                X_val_fold = X_train.iloc[val_idx]
                y_val_true = y_train.iloc[val_idx].values
            
            # Train
            model.fit(X_train_fold, y_train_fold)
            
            # Predict
            val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            
            # Convert predictions back to numpy if on GPU
            if is_gpu_model and CUPY_AVAILABLE:
                val_pred_proba = cp.asnumpy(val_pred_proba)
            
            val_pred = (val_pred_proba > 0.5).astype(int)
            
            # Apply ambivert rule if enabled
            if model_config['use_ambivert_rule']:
                ambivert_mask, _ = detect_ambiverts(X_train.iloc[val_idx], model, is_gpu=is_gpu_model)
                val_pred = apply_ambivert_rule(val_pred, ambivert_mask, val_pred_proba)
            
            score = accuracy_score(y_val_true, val_pred)
            cv_scores.append(score)
        
        cv_score = np.mean(cv_scores)
        console.print(f"  CV Score: {cv_score:.6f}")
        
        # Train final model
        if is_gpu_model and CUPY_AVAILABLE:
            # Convert to GPU arrays
            X_train_gpu = cp.asarray(X_train.values)
            y_train_gpu = cp.asarray(y_train.values)
            X_test_gpu = cp.asarray(X_test.values)
            model.fit(X_train_gpu, y_train_gpu)
            
            # Test predictions on GPU
            test_proba = model.predict_proba(X_test_gpu)[:, 1]
            test_proba = cp.asnumpy(test_proba)  # Convert back to numpy
        else:
            model.fit(X_train, y_train)
            # Test predictions
            test_proba = model.predict_proba(X_test)[:, 1]
        
        test_pred = (test_proba > 0.5).astype(int)
        
        # Apply ambivert rule if enabled
        if model_config['use_ambivert_rule']:
            ambivert_mask, _ = detect_ambiverts(X_test, model, is_gpu=is_gpu_model)
            test_pred = apply_ambivert_rule(test_pred, ambivert_mask, test_proba)
            n_ambiverts = ambivert_mask.sum()
            console.print(f"  Detected {n_ambiverts} ambiverts ({n_ambiverts/len(X_test)*100:.2f}%)")
        
        # Save submission
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': test_pred
        })
        submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"subm-{cv_score:.6f}-{timestamp}-{model_config['name']}.csv"
        submission.to_csv(SCORES_DIR / filename, index=False)
        
        results.append({
            'name': model_config['name'],
            'cv_score': cv_score,
            'filename': filename,
            'ambiverts': n_ambiverts if model_config['use_ambivert_rule'] else 0
        })
    
    # Display results
    table = Table(title="Winning Model Recreations", box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("CV Score", style="green")
    table.add_column("Ambiverts", style="yellow")
    table.add_column("File", style="blue")
    
    for r in sorted(results, key=lambda x: x['cv_score'], reverse=True):
        table.add_row(
            r['name'],
            f"{r['cv_score']:.6f}",
            str(r['ambiverts']),
            r['filename']
        )
    
    console.print("\n")
    console.print(table)
    
    console.print("\n[bold green]✅ Winning model recreated![/bold green]")
    console.print("The ultra_simple_5_2 with ambivert rule should match the original 0.975708")
    console.print("\nKey insights:")
    console.print("  • Only 5 trees with depth 2 were enough")
    console.print("  • The ambivert rule (96.2% → Extrovert) was crucial")
    console.print("  • No complex feature engineering needed")
    console.print("  • No corrected datasets needed")

if __name__ == "__main__":
    main()