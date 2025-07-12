#!/usr/bin/env python3
"""
Test model accuracy on original personality dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
COMP_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7")
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")

def prepare_original_data():
    """Load and prepare original data"""
    print("="*60)
    print("LOADING ORIGINAL DATA")
    print("="*60)
    
    # Load original data
    orig_df = pd.read_csv(COMP_DIR / "personality_dataset.csv")
    print(f"Original dataset shape: {orig_df.shape}")
    print(f"Columns: {list(orig_df.columns)}")
    
    # Check for duplicates
    duplicates = orig_df.duplicated().sum()
    print(f"\nDuplicates: {duplicates} ({duplicates/len(orig_df)*100:.1f}%)")
    
    # Remove duplicates for cleaner analysis
    orig_df_clean = orig_df.drop_duplicates()
    print(f"After removing duplicates: {orig_df_clean.shape}")
    
    # Check class distribution
    print("\nClass distribution:")
    print(orig_df_clean['Personality'].value_counts())
    print(orig_df_clean['Personality'].value_counts(normalize=True))
    
    # Prepare features and target
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    X = orig_df_clean[feature_cols].copy()
    y = (orig_df_clean['Personality'] == 'Extrovert').astype(int)
    
    # Handle missing values
    print(f"\nMissing values per column:")
    print(X.isnull().sum())
    
    # Convert Yes/No to 1/0
    binary_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in binary_cols:
        X[col] = (X[col] == 'Yes').astype(int)
    
    # Simple imputation for numeric columns
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                    'Going_outside', 'Post_frequency']
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    
    return X, y, orig_df_clean

def test_on_synthetic_data():
    """Test models on synthetic training data for comparison"""
    print("\n" + "="*60)
    print("TESTING ON SYNTHETIC DATA")
    print("="*60)
    
    # Load synthetic data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    print(f"Synthetic train shape: {train_df.shape}")
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    X = train_df[feature_cols].copy()
    y = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # Convert Yes/No to 1/0
    binary_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in binary_cols:
        X[col] = (X[col] == 'Yes').astype(int)
    
    # Handle missing values for numeric columns
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                    'Going_outside', 'Post_frequency']
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1),
        'CatBoost': CatBoostClassifier(n_estimators=100, random_state=42, verbose=False)
    }
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"\n{name} on synthetic data:")
        print(f"  CV Accuracy: {scores.mean():.6f} (+/- {scores.std()*2:.6f})")

def cross_validate_models(X, y):
    """Cross-validate different models on the data"""
    print("\n" + "="*60)
    print("CROSS-VALIDATION ON ORIGINAL DATA")
    print("="*60)
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test different models
    models = {
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1),
        'CatBoost': CatBoostClassifier(n_estimators=100, random_state=42, verbose=False)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[name] = scores
        
        print(f"  CV Accuracy: {scores.mean():.6f} (+/- {scores.std()*2:.6f})")
        print(f"  Individual folds: {scores}")
        
        # Fit on full data to check training accuracy
        model.fit(X, y)
        train_pred = model.predict(X)
        train_acc = accuracy_score(y, train_pred)
        print(f"  Training accuracy: {train_acc:.6f}")
    
    return results

def analyze_errors_in_original():
    """Check if there are inherent errors in original data"""
    print("\n" + "="*60)
    print("ANALYZING POTENTIAL ERRORS IN ORIGINAL DATA")
    print("="*60)
    
    # Load original with duplicates
    orig_df = pd.read_csv(COMP_DIR / "personality_dataset.csv")
    
    # Find duplicate feature sets with different labels
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    # Group by features
    grouped = orig_df.groupby(feature_cols)['Personality'].apply(lambda x: x.value_counts().to_dict()).reset_index()
    
    # Find conflicts
    conflicts = []
    for idx, row in grouped.iterrows():
        personalities = row['Personality']
        if len(personalities) > 1:  # Multiple different labels for same features
            conflicts.append({
                'features': row[feature_cols].to_dict(),
                'labels': personalities,
                'total_count': sum(personalities.values())
            })
    
    print(f"\nFound {len(conflicts)} feature combinations with conflicting labels")
    
    if conflicts:
        print("\nExamples of conflicts:")
        for i, conflict in enumerate(conflicts[:5]):
            print(f"\nConflict {i+1}:")
            print(f"  Labels: {conflict['labels']}")
            print(f"  Total occurrences: {conflict['total_count']}")

def test_on_corrected_synthetic():
    """Test what happens if we correct the synthetic data"""
    print("\n" + "="*60)
    print("THEORETICAL ACCURACY WITH CORRECTIONS")
    print("="*60)
    
    # Load mismatches
    mismatches_df = pd.read_csv(Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output/generation_mismatches.csv"))
    train_mismatches = mismatches_df[mismatches_df['syn_source'] == 'train']
    
    print(f"Train mismatches to correct: {len(train_mismatches)}")
    
    # Load synthetic train
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    
    # Simulate correction
    corrected_df = train_df.copy()
    corrections_made = 0
    
    for _, mismatch in train_mismatches.iterrows():
        if pd.notna(mismatch['orig_personality']):
            mask = corrected_df['id'] == mismatch['syn_id']
            if mask.any():
                corrected_df.loc[mask, 'Personality'] = mismatch['orig_personality']
                corrections_made += 1
    
    print(f"Corrections applied: {corrections_made}")
    
    # Test on corrected data
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    X = corrected_df[feature_cols].fillna(corrected_df[feature_cols].median())
    y = (corrected_df['Personality'] == 'Extrovert').astype(int)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\nXGBoost on CORRECTED synthetic data:")
    print(f"  CV Accuracy: {scores.mean():.6f} (+/- {scores.std()*2:.6f})")
    print(f"  Improvement from corrections: {scores.mean() - 0.97997:.6f}")

def main():
    # Test on original data
    X, y, orig_df = prepare_original_data()
    results = cross_validate_models(X, y)
    
    # Test on synthetic for comparison
    test_on_synthetic_data()
    
    # Analyze potential errors
    analyze_errors_in_original()
    
    # Test theoretical improvement
    test_on_corrected_synthetic()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("1. Original data accuracy is surprisingly LOW (~94-95%)")
    print("2. This suggests the original data has inherent noise/ambiguity")
    print("3. Synthetic data achieves ~98% because it's 'cleaner'")
    print("4. The 171 mismatches might actually be corrections, not errors!")

if __name__ == "__main__":
    main()