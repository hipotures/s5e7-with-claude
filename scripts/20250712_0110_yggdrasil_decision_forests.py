#!/usr/bin/env python3
"""
Test Yggdrasil Decision Forests (YDF) library on S5E7 dataset
YDF is Google's high-performance decision forest library
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"
OUTPUT_DIR.mkdir(exist_ok=True)

def prepare_data():
    """Load and prepare data for YDF"""
    print("="*60)
    print("LOADING DATA FOR YGGDRASIL")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # YDF can handle categorical data directly
    # But we need to handle missing values first
    for col in ['Stage_fear', 'Drained_after_socializing']:
        # Fill missing values with a special category
        train_df[col] = train_df[col].fillna('Unknown')
        test_df[col] = test_df[col].fillna('Unknown')
        # Now convert to categorical
        train_df[col] = pd.Categorical(train_df[col])
        test_df[col] = pd.Categorical(test_df[col])
    
    # Keep Personality as target
    train_labels = train_df['Personality']
    
    # Drop id and target
    train_features = train_df.drop(['id', 'Personality'], axis=1)
    test_features = test_df.drop(['id'], axis=1)
    
    print(f"Train shape: {train_features.shape}")
    print(f"Test shape: {test_features.shape}")
    print(f"\nFeatures: {list(train_features.columns)}")
    print(f"\nTarget distribution:")
    print(train_labels.value_counts())
    
    return train_features, train_labels, test_features, train_df, test_df

def test_ydf_models(train_features, train_labels, test_features):
    """Test different YDF models"""
    print("\n" + "="*60)
    print("TESTING YDF MODELS")
    print("="*60)
    
    results = []
    
    # 1. Random Forest
    print("\n1. Random Forest")
    start_time = time.time()
    
    rf_model = ydf.RandomForestLearner(
        label="Personality",
        num_trees=300,
        max_depth=20,
        min_examples=5,
        winner_take_all=True,  # For better accuracy
        random_seed=42
    )
    
    # Create dataset for training
    train_data = train_features.copy()
    train_data['Personality'] = train_labels
    
    # Train
    rf_model = rf_model.train(train_data)
    train_time = time.time() - start_time
    
    # Evaluate
    evaluation = rf_model.evaluate(train_data)
    print(f"Training time: {train_time:.2f}s")
    print(f"Accuracy: {evaluation.accuracy:.6f}")
    
    # Cross-validation
    rf_oob_accuracy = rf_model.self_evaluation().accuracy
    print(f"OOB Accuracy: {rf_oob_accuracy:.6f}")
    
    results.append({
        'model': 'Random Forest',
        'accuracy': evaluation.accuracy,
        'oob_accuracy': rf_oob_accuracy,
        'train_time': train_time
    })
    
    # 2. Gradient Boosted Trees
    print("\n2. Gradient Boosted Trees")
    start_time = time.time()
    
    gbt_model = ydf.GradientBoostedTreesLearner(
        label="Personality",
        num_trees=300,
        shrinkage=0.1,
        max_depth=6,
        min_examples=5,
        random_seed=42
    )
    
    gbt_model = gbt_model.train(train_data)
    train_time = time.time() - start_time
    
    evaluation = gbt_model.evaluate(train_data)
    print(f"Training time: {train_time:.2f}s")
    print(f"Accuracy: {evaluation.accuracy:.6f}")
    
    results.append({
        'model': 'Gradient Boosted Trees',
        'accuracy': evaluation.accuracy,
        'oob_accuracy': None,
        'train_time': train_time
    })
    
    # 3. CART (single tree)
    print("\n3. CART (Decision Tree)")
    start_time = time.time()
    
    cart_model = ydf.CartLearner(
        label="Personality",
        max_depth=10,
        min_examples=20,
        random_seed=42
    )
    
    cart_model = cart_model.train(train_data)
    train_time = time.time() - start_time
    
    evaluation = cart_model.evaluate(train_data)
    print(f"Training time: {train_time:.2f}s")
    print(f"Accuracy: {evaluation.accuracy:.6f}")
    
    results.append({
        'model': 'CART',
        'accuracy': evaluation.accuracy,
        'oob_accuracy': None,
        'train_time': train_time
    })
    
    return rf_model, gbt_model, cart_model, results

def analyze_feature_importance(models, feature_names):
    """Analyze feature importance from different models"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_names = ['Random Forest', 'Gradient Boosted Trees', 'CART']
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        # Get variable importances
        importances = model.variable_importances()
        
        print(f"\n{name} - Top features:")
        # YDF returns a list-like object, iterate properly
        for i, imp in enumerate(importances):
            if i >= 5:
                break
            print(f"  {imp.name}: {imp.importance:.4f}")
        
        # Plot
        if importances:
            features = []
            values = []
            for i, imp in enumerate(importances):
                if i >= 7:
                    break
                features.append(imp.name)
                values.append(imp.importance)
            
            axes[idx].barh(features, values)
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{name} Feature Importance')
            axes[idx].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ydf_feature_importances.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_model_predictions(rf_model, gbt_model, train_data, test_features):
    """Analyze predictions and uncertainties"""
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    # Get predictions on test set
    rf_pred = rf_model.predict(test_features)
    gbt_pred = gbt_model.predict(test_features)
    
    # Get prediction probabilities
    rf_proba = rf_model.predict_proba(test_features)
    gbt_proba = gbt_model.predict_proba(test_features)
    
    # Convert to DataFrames for analysis
    test_analysis = pd.DataFrame({
        'id': test_features.index,
        'rf_pred': rf_pred,
        'gbt_pred': gbt_pred,
        'rf_proba_extrovert': rf_proba[:, 0],  # Assuming Extrovert is first class
        'gbt_proba_extrovert': gbt_proba[:, 0],
        'agreement': rf_pred == gbt_pred
    })
    
    # Calculate uncertainty (entropy or difference from 0.5)
    test_analysis['rf_uncertainty'] = 1 - np.abs(test_analysis['rf_proba_extrovert'] - 0.5) * 2
    test_analysis['gbt_uncertainty'] = 1 - np.abs(test_analysis['gbt_proba_extrovert'] - 0.5) * 2
    
    print(f"\nModel agreement: {test_analysis['agreement'].mean():.2%}")
    print(f"\nMost uncertain predictions (RF):")
    print(test_analysis.nlargest(10, 'rf_uncertainty')[['id', 'rf_proba_extrovert', 'rf_uncertainty']])
    
    # Find disagreements
    disagreements = test_analysis[~test_analysis['agreement']]
    print(f"\nNumber of disagreements: {len(disagreements)}")
    
    if len(disagreements) > 0:
        print("\nTop disagreements:")
        print(disagreements.head(10)[['id', 'rf_pred', 'gbt_pred', 'rf_proba_extrovert', 'gbt_proba_extrovert']])
    
    # Save analysis
    test_analysis.to_csv(OUTPUT_DIR / 'ydf_prediction_analysis.csv', index=False)
    
    return test_analysis

def create_ydf_submission(model, test_features, test_df, model_name):
    """Create submission file"""
    print("\n" + "="*60)
    print(f"CREATING {model_name.upper()} SUBMISSION")
    print("="*60)
    
    # Get predictions
    predictions = model.predict(test_features)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': predictions
    })
    
    # Save
    filename = f"ydf_{model_name.lower().replace(' ', '_')}_submission.csv"
    submission.to_csv(WORKSPACE_DIR / "scores" / filename, index=False)
    print(f"Saved: {filename}")
    
    return submission

def test_on_hard_cases(model, train_features, train_labels):
    """Test model specifically on our known hard cases"""
    print("\n" + "="*60)
    print("TESTING ON HARD CASES")
    print("="*60)
    
    try:
        # Load hard cases
        hard_cases_df = pd.read_csv(OUTPUT_DIR / 'removed_hard_cases_info.csv')
        hard_case_ids = hard_cases_df['id'].values
        
        # Create train data with labels
        train_data = train_features.copy()
        train_data['Personality'] = train_labels
        train_data['id'] = range(len(train_data))
        
        # Filter to hard cases
        hard_cases_data = train_data[train_data['id'].isin(hard_case_ids)]
        
        if len(hard_cases_data) > 0:
            # Evaluate on hard cases
            hard_cases_eval = model.evaluate(hard_cases_data)
            print(f"Accuracy on hard cases: {hard_cases_eval.accuracy:.6f}")
            print(f"Number of hard cases: {len(hard_cases_data)}")
            
            # Compare with overall accuracy
            overall_eval = model.evaluate(train_data)
            print(f"Overall accuracy: {overall_eval.accuracy:.6f}")
            print(f"Difference: {overall_eval.accuracy - hard_cases_eval.accuracy:.6f}")
            
    except Exception as e:
        print(f"Could not test on hard cases: {e}")

def main():
    # Load and prepare data
    train_features, train_labels, test_features, train_df, test_df = prepare_data()
    
    # Test YDF models
    rf_model, gbt_model, cart_model, results = test_ydf_models(
        train_features, train_labels, test_features
    )
    
    # Analyze feature importance
    analyze_feature_importance(
        [rf_model, gbt_model, cart_model], 
        train_features.columns
    )
    
    # Analyze predictions
    train_data = train_features.copy()
    train_data['Personality'] = train_labels
    test_analysis = analyze_model_predictions(
        rf_model, gbt_model, train_data, test_features
    )
    
    # Test on hard cases
    test_on_hard_cases(rf_model, train_features, train_labels)
    
    # Create submissions for best models
    create_ydf_submission(rf_model, test_features, test_df, "random_forest")
    create_ydf_submission(gbt_model, test_features, test_df, "gradient_boosted_trees")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    results_df = pd.DataFrame(results)
    print("\nModel Performance:")
    print(results_df.to_string(index=False))
    
    print("\nYggdrasil Decision Forests advantages:")
    print("- Very fast training")
    print("- Native handling of missing values")
    print("- Built-in cross-validation (OOB)")
    print("- Excellent interpretability")
    print("- No need for preprocessing")

if __name__ == "__main__":
    main()