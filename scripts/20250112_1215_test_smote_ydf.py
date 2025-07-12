#!/usr/bin/env python3
"""
Test SMOTE balancing with YDF model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def load_and_prepare_data():
    """Load and prepare data"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"\nOriginal train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Check class distribution
    print("\nOriginal class distribution:")
    print(train_df['Personality'].value_counts())
    print(train_df['Personality'].value_counts(normalize=True))
    
    # Features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Prepare for sklearn (numeric only)
    train_df['Stage_fear_encoded'] = (train_df['Stage_fear'] == 'Yes').astype(int)
    train_df['Drained_encoded'] = (train_df['Drained_after_socializing'] == 'Yes').astype(int)
    
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency',
                       'Stage_fear_encoded', 'Drained_encoded']
    
    # Handle missing values
    for col in numeric_features[:5]:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())
    
    # Encode test data
    test_df['Stage_fear_encoded'] = (test_df['Stage_fear'] == 'Yes').astype(int)
    test_df['Drained_encoded'] = (test_df['Drained_after_socializing'] == 'Yes').astype(int)
    
    X = train_df[numeric_features].values
    y = (train_df['Personality'] == 'Introvert').astype(int).values
    X_test = test_df[numeric_features].values
    
    return X, y, X_test, train_df, test_df, feature_cols

def apply_balancing_methods(X, y):
    """Apply different balancing methods"""
    print("\n" + "="*60)
    print("APPLYING BALANCING METHODS")
    print("="*60)
    
    results = {}
    
    # 1. Original (no balancing)
    results['original'] = (X.copy(), y.copy())
    print(f"\nOriginal: {len(y)} samples")
    
    # 2. SMOTE
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    results['SMOTE'] = (X_smote, y_smote)
    print(f"SMOTE: {len(y_smote)} samples")
    print(f"Class distribution: {np.bincount(y_smote)}")
    
    # 3. ADASYN
    print("\nApplying ADASYN...")
    adasyn = ADASYN(random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
    results['ADASYN'] = (X_adasyn, y_adasyn)
    print(f"ADASYN: {len(y_adasyn)} samples")
    print(f"Class distribution: {np.bincount(y_adasyn)}")
    
    # 4. SMOTE-ENN (hybrid)
    print("\nApplying SMOTE-ENN...")
    smote_enn = SMOTEENN(random_state=42)
    X_smote_enn, y_smote_enn = smote_enn.fit_resample(X, y)
    results['SMOTE-ENN'] = (X_smote_enn, y_smote_enn)
    print(f"SMOTE-ENN: {len(y_smote_enn)} samples")
    print(f"Class distribution: {np.bincount(y_smote_enn)}")
    
    return results

def train_ydf_with_balancing(balanced_data, X_test, feature_cols):
    """Train YDF models on different balanced datasets"""
    print("\n" + "="*60)
    print("TRAINING YDF MODELS")
    print("="*60)
    
    predictions = {}
    cv_scores = {}
    
    for method_name, (X_balanced, y_balanced) in balanced_data.items():
        print(f"\n{method_name} Dataset:")
        
        # Convert back to DataFrame for YDF
        balanced_df = pd.DataFrame(X_balanced, columns=[
            'Time_spent_Alone', 'Social_event_attendance', 
            'Friends_circle_size', 'Going_outside', 'Post_frequency',
            'Stage_fear_encoded', 'Drained_encoded'
        ])
        
        # Add target
        balanced_df['Personality'] = y_balanced
        balanced_df['Personality'] = balanced_df['Personality'].map({0: 'Extrovert', 1: 'Introvert'})
        
        # Convert encoded back to categorical for YDF
        balanced_df['Stage_fear'] = balanced_df['Stage_fear_encoded'].map({0: 'No', 1: 'Yes'})
        balanced_df['Drained_after_socializing'] = balanced_df['Drained_encoded'].map({0: 'No', 1: 'Yes'})
        
        # Use original feature columns for YDF
        train_df_ydf = balanced_df[feature_cols + ['Personality']]
        
        # Cross-validation
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_balanced, y_balanced)):
            train_fold = train_df_ydf.iloc[train_idx]
            val_fold = train_df_ydf.iloc[val_idx]
            
            # Train YDF
            learner = ydf.RandomForestLearner(
                label='Personality',
                num_trees=300,
                max_depth=6,
                random_seed=42
            )
            
            model = learner.train(train_fold)
            
            # Predict
            predictions_fold = model.predict(val_fold.drop('Personality', axis=1))
            pred_classes = ['Introvert' if float(str(p)) > 0.5 else 'Extrovert' for p in predictions_fold]
            
            accuracy = accuracy_score(val_fold['Personality'], pred_classes)
            fold_scores.append(accuracy)
            print(f"  Fold {fold+1}: {accuracy:.6f}")
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        cv_scores[method_name] = (mean_score, std_score)
        print(f"  Mean CV: {mean_score:.6f} (±{std_score:.6f})")
        
        # Train final model on full balanced dataset
        print(f"  Training final model on full {method_name} dataset...")
        final_model = learner.train(train_df_ydf)
        
        # Prepare test data for YDF
        test_df_ydf = pd.DataFrame(X_test, columns=[
            'Time_spent_Alone', 'Social_event_attendance', 
            'Friends_circle_size', 'Going_outside', 'Post_frequency',
            'Stage_fear_encoded', 'Drained_encoded'
        ])
        test_df_ydf['Stage_fear'] = test_df_ydf['Stage_fear_encoded'].map({0: 'No', 1: 'Yes'})
        test_df_ydf['Drained_after_socializing'] = test_df_ydf['Drained_encoded'].map({0: 'No', 1: 'Yes'})
        test_df_final = test_df_ydf[feature_cols]
        
        # Predict on test
        test_predictions = final_model.predict(test_df_final)
        test_pred_classes = ['Introvert' if float(str(p)) > 0.5 else 'Extrovert' for p in test_predictions]
        predictions[method_name] = test_pred_classes
        
        # Skip feature importance for now (YDF returns different format)
    
    return predictions, cv_scores

def create_submissions(predictions, test_df):
    """Create submission files for each method"""
    print("\n" + "="*60)
    print("CREATING SUBMISSIONS")
    print("="*60)
    
    for method_name, preds in predictions.items():
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Personality': preds
        })
        
        # Save
        filename = f"submission_ydf_{method_name.lower().replace('-', '_')}.csv"
        filepath = OUTPUT_DIR / filename
        submission.to_csv(filepath, index=False)
        print(f"\n{method_name} submission saved to: {filename}")
        
        # Show distribution
        print(f"Prediction distribution:")
        print(submission['Personality'].value_counts(normalize=True))

def analyze_results(cv_scores):
    """Analyze and compare results"""
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    print("\nCV Score Summary:")
    for method, (mean, std) in cv_scores.items():
        print(f"{method}: {mean:.6f} (±{std:.6f})")
    
    # Find best method
    best_method = max(cv_scores.items(), key=lambda x: x[1][0])
    print(f"\nBest method: {best_method[0]} with {best_method[1][0]:.6f} accuracy")
    
    print("\n" + "="*40)
    print("KEY INSIGHTS:")
    print("="*40)
    print("""
1. SMOTE/ADASYN Results:
   - If accuracy jumps significantly (>5%), be suspicious
   - High CV doesn't guarantee high LB score
   - May be overfitting to synthetic patterns

2. Original vs Balanced:
   - Small improvement is realistic
   - Large improvement might be artificial
   - Check if predictions shift toward 50-50 split

3. For Competition:
   - Test submissions on Kaggle
   - Trust original data results more
   - Use balancing as exploration tool
""")

def main():
    # Load data
    X, y, X_test, train_df, test_df, feature_cols = load_and_prepare_data()
    
    # Apply balancing methods
    balanced_data = apply_balancing_methods(X, y)
    
    # Train YDF models
    predictions, cv_scores = train_ydf_with_balancing(balanced_data, X_test, feature_cols)
    
    # Create submissions
    create_submissions(predictions, test_df)
    
    # Analyze results
    analyze_results(cv_scores)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\n⚠️  UWAGA: Wysokie CV scores z SMOTE mogą być mylące!")
    print("Przetestuj submisje na Kaggle aby zobaczyć rzeczywiste wyniki.")

if __name__ == "__main__":
    main()