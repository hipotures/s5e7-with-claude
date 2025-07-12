#!/usr/bin/env python3
"""
Test if all models hit the same accuracy ceiling (~97.57%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import ydf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")

def prepare_data():
    """Load and prepare data"""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    
    # Features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Encode categorical
    train_df['Stage_fear_encoded'] = (train_df['Stage_fear'] == 'Yes').astype(int)
    train_df['Drained_encoded'] = (train_df['Drained_after_socializing'] == 'Yes').astype(int)
    
    # Numeric features
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency',
                       'Stage_fear_encoded', 'Drained_encoded']
    
    # Handle missing values
    for col in numeric_features[:5]:
        train_df[col] = train_df[col].fillna(train_df[col].median())
    
    X = train_df[numeric_features].values
    y = (train_df['Personality'] == 'Introvert').astype(int).values
    
    # For YDF
    train_df_ydf = train_df[feature_cols + ['Personality']].copy()
    
    return X, y, train_df_ydf, numeric_features

def test_all_models():
    """Test various models to see if they all hit the same ceiling"""
    
    print("="*60)
    print("TESTING ACCURACY CEILING ACROSS DIFFERENT MODELS")
    print("="*60)
    
    X, y, train_df_ydf, feature_names = prepare_data()
    
    # Models to test
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            device='cuda:0',
            tree_method='hist',
            random_state=42
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            device='gpu',
            random_state=42,
            verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            task_type='GPU',
            random_seed=42,
            verbose=False
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        ),
        'NeuralNet': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42,
            early_stopping=True
        )
    }
    
    # Cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_score = accuracy_score(y_val, val_pred)
            cv_scores.append(val_score)
            print(f"  Fold {fold+1}: {val_score:.6f}")
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        results[name] = (mean_score, std_score)
        print(f"  Mean: {mean_score:.6f} (±{std_score:.6f})")
    
    # Test YDF separately
    print(f"\nYDF RandomForest:")
    cv_scores_ydf = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        train_fold = train_df_ydf.iloc[train_idx]
        val_fold = train_df_ydf.iloc[val_idx]
        
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=300,
            max_depth=6,
            random_seed=42
        )
        
        model_ydf = learner.train(train_fold)
        
        # Predict
        predictions = model_ydf.predict(val_fold.drop('Personality', axis=1))
        pred_classes = ['Introvert' if float(str(p)) > 0.5 else 'Extrovert' for p in predictions]
        
        val_score = accuracy_score(val_fold['Personality'], pred_classes)
        cv_scores_ydf.append(val_score)
        print(f"  Fold {fold+1}: {val_score:.6f}")
    
    mean_score_ydf = np.mean(cv_scores_ydf)
    std_score_ydf = np.std(cv_scores_ydf)
    results['YDF'] = (mean_score_ydf, std_score_ydf)
    print(f"  Mean: {mean_score_ydf:.6f} (±{std_score_ydf:.6f})")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - ACCURACY CEILING ANALYSIS")
    print("="*60)
    
    print("\nModel Performance Ranking:")
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    
    for i, (name, (mean, std)) in enumerate(sorted_results):
        print(f"{i+1}. {name}: {mean:.6f} (±{std:.6f})")
    
    # Check if all hit similar ceiling
    scores = [r[0] for r in results.values()]
    max_score = max(scores)
    min_score = min(scores)
    
    print(f"\nScore range: {min_score:.6f} - {max_score:.6f}")
    print(f"Difference: {(max_score - min_score)*100:.2f}%")
    
    if max_score - min_score < 0.01:  # Less than 1% difference
        print("\n✓ All models hit approximately the same ceiling!")
        print(f"  Average ceiling: {np.mean(scores):.6f} (~97.57%)")
    else:
        print("\n✗ Models show different performance levels")
        print("  Some models might be able to break the ceiling!")
    
    # Additional analysis
    print("\n" + "="*40)
    print("KEY INSIGHTS:")
    print("="*40)
    print("""
1. If all models converge to ~97.57%, it suggests:
   - This is a mathematical limit of the data
   - ~2.43% of cases are inherently unpredictable
   - These might be true ambiverts or mislabeled data

2. If some models perform better:
   - There might be complex patterns others miss
   - The ceiling could potentially be broken
   - Worth exploring those specific models further

3. The consistency across models indicates:
   - The problem is well-defined
   - Features are informative
   - The limit is likely data-inherent, not model-limited
""")

if __name__ == "__main__":
    test_all_models()