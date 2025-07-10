#!/usr/bin/env python3
"""
Train model and analyze prediction uncertainty to find potential errors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")

def prepare_data():
    """Load and prepare data"""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Encode binary features
    for col in ['Drained_after_socializing', 'Stage_fear']:
        train_df[f'{col}_num'] = train_df[col].map({'Yes': 1, 'No': 0})
        test_df[f'{col}_num'] = test_df[col].map({'Yes': 1, 'No': 0})
    
    # Encode target
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['Personality'])
    
    # Features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Drained_after_socializing_num', 
                   'Stage_fear_num']
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    return X_train, y_train, X_test, test_df, train_df, le

def train_models_get_probabilities(X_train, y_train, X_test):
    """Train multiple models and get probability predictions"""
    print("\n" + "="*60)
    print("TRAINING MODELS AND GETTING PROBABILITIES")
    print("="*60)
    
    predictions = {}
    
    # 1. XGBoost
    print("\n1. Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    predictions['xgb'] = xgb_proba
    
    # 2. LightGBM
    print("2. Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    predictions['lgb'] = lgb_proba
    
    # 3. CatBoost
    print("3. Training CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    cat_proba = cat_model.predict_proba(X_test)[:, 1]
    predictions['cat'] = cat_proba
    
    return predictions, xgb_model

def analyze_uncertainty(predictions, test_df):
    """Analyze prediction uncertainty"""
    print("\n" + "="*60)
    print("ANALIZA NIEPEWNOŚCI PREDYKCJI")
    print("="*60)
    
    # Calculate average probability and standard deviation
    prob_matrix = np.column_stack([predictions['xgb'], predictions['lgb'], predictions['cat']])
    avg_prob = prob_matrix.mean(axis=1)
    std_prob = prob_matrix.std(axis=1)
    
    # Calculate uncertainty metrics
    uncertainty_df = pd.DataFrame({
        'id': test_df['id'],
        'avg_prob': avg_prob,
        'std_prob': std_prob,
        'xgb_prob': predictions['xgb'],
        'lgb_prob': predictions['lgb'],
        'cat_prob': predictions['cat'],
        'uncertainty': np.minimum(avg_prob, 1 - avg_prob),  # Distance from 0.5
        'disagreement': std_prob,
        'prediction': (avg_prob > 0.5).astype(int)
    })
    
    # Find most uncertain predictions
    print("\n1. NAJBARDZIEJ NIEPEWNE PREDYKCJE (blisko 0.5):")
    print("-"*40)
    most_uncertain = uncertainty_df.nsmallest(20, 'uncertainty')
    
    for _, row in most_uncertain.head(10).iterrows():
        pred_label = 'Extrovert' if row['prediction'] == 1 else 'Introvert'
        print(f"ID {int(row['id'])}: {pred_label} (prob={row['avg_prob']:.3f}, "
              f"uncertainty={row['uncertainty']:.3f})")
    
    # Find where models disagree most
    print("\n2. NAJWIĘKSZE ROZBIEŻNOŚCI MIĘDZY MODELAMI:")
    print("-"*40)
    most_disagreement = uncertainty_df.nlargest(20, 'disagreement')
    
    for _, row in most_disagreement.head(10).iterrows():
        print(f"ID {int(row['id'])}: XGB={row['xgb_prob']:.3f}, "
              f"LGB={row['lgb_prob']:.3f}, CAT={row['cat_prob']:.3f} "
              f"(std={row['disagreement']:.3f})")
    
    return uncertainty_df

def analyze_cv_errors(X_train, y_train, train_df):
    """Analyze cross-validation errors"""
    print("\n" + "="*60)
    print("ANALIZA BŁĘDÓW CROSS-VALIDATION")
    print("="*60)
    
    # Use 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get CV predictions
    print("\nRunning cross-validation...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                             random_state=42, verbosity=0)
    
    cv_proba = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    cv_pred = (cv_proba > 0.5).astype(int)
    
    # Find misclassified samples
    misclassified = train_df[cv_pred != y_train].copy()
    misclassified['true_label'] = misclassified['Personality']
    misclassified['pred_label'] = ['Extrovert' if p == 1 else 'Introvert' for p in cv_pred[cv_pred != y_train]]
    misclassified['probability'] = cv_proba[cv_pred != y_train]
    
    print(f"\nMisclassified in CV: {len(misclassified)} out of {len(train_df)} "
          f"({len(misclassified)/len(train_df)*100:.1f}%)")
    
    # Find consistently misclassified (high confidence but wrong)
    high_conf_errors = misclassified[(misclassified['probability'] > 0.7) | 
                                     (misclassified['probability'] < 0.3)]
    
    print(f"\nHigh confidence errors: {len(high_conf_errors)}")
    print("\nExamples of high confidence errors:")
    for _, row in high_conf_errors.head(5).iterrows():
        print(f"ID {row['id']}: True={row['true_label']}, "
              f"Pred={row['pred_label']} (prob={row['probability']:.3f})")
    
    return misclassified, high_conf_errors

def create_flip_candidates(uncertainty_df, high_conf_errors):
    """Create flip candidates based on analysis"""
    print("\n" + "="*60)
    print("TWORZENIE KANDYDATÓW DO FLIPÓW")
    print("="*60)
    
    candidates = []
    
    # 1. Most uncertain predictions
    for _, row in uncertainty_df.nsmallest(5, 'uncertainty').iterrows():
        pred_label = 'Extrovert' if row['prediction'] == 1 else 'Introvert'
        candidates.append({
            'id': int(row['id']),
            'current': pred_label,
            'strategy': 'high_uncertainty',
            'score': row['uncertainty'],
            'prob': row['avg_prob']
        })
    
    # 2. High disagreement between models
    for _, row in uncertainty_df.nlargest(5, 'disagreement').iterrows():
        pred_label = 'Extrovert' if row['prediction'] == 1 else 'Introvert'
        candidates.append({
            'id': int(row['id']),
            'current': pred_label,
            'strategy': 'model_disagreement',
            'score': row['disagreement'],
            'prob': row['avg_prob']
        })
    
    print(f"\nUtworzono {len(candidates)} kandydatów")
    
    # Save for submission creation
    candidates_df = pd.DataFrame(candidates)
    output_path = WORKSPACE_DIR / "scripts/output/uncertainty_flip_candidates.csv"
    candidates_df.to_csv(output_path, index=False)
    print(f"Zapisano kandydatów: {output_path}")
    
    return candidates

def main():
    # Prepare data
    X_train, y_train, X_test, test_df, train_df, le = prepare_data()
    
    # Train models and get probabilities
    predictions, model = train_models_get_probabilities(X_train, y_train, X_test)
    
    # Analyze uncertainty
    uncertainty_df = analyze_uncertainty(predictions, test_df)
    
    # Analyze CV errors
    misclassified, high_conf_errors = analyze_cv_errors(X_train, y_train, train_df)
    
    # Create flip candidates
    candidates = create_flip_candidates(uncertainty_df, high_conf_errors)
    
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    print("\nNowa strategia oparta na:")
    print("1. Niepewności predykcji (prob ~0.5)")
    print("2. Rozbieżności między modelami")
    print("3. Błędach cross-validation")
    print("\nTo podejście jest bardziej naukowe niż szukanie wzorców!")

if __name__ == "__main__":
    main()