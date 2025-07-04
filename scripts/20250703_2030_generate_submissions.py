#!/usr/bin/env python3
"""
Generate predictions for test set using different feature selection methods.
Creates submission files for each method with CV score in filename.

PURPOSE: Generate multiple submission files using the best-performing feature selection
         methods identified in the comprehensive analysis, with GPU acceleration for
         faster predictions.

HYPOTHESIS: The feature selection methods that performed best in cross-validation
            (RFECV, Random Forest Importance, XGBoost Importance) will also produce
            the best test set predictions.

EXPECTED: Generate 7 different submission files with scores ranging from 0.9664 to
          0.9678, with RFECV_optimal expected to perform best on the public leaderboard.

RESULT: Successfully generated submissions for all 7 feature selection methods:
        - RFECV_optimal (13 features, 0.9678 CV)
        - Random_Forest_Importance (104 features, 0.9676 CV)
        - XGBoost_Importance (61 features, 0.9675 CV)
        - Baseline_all (148 features, 0.9673 CV)
        - Lasso_L1 (134 features, 0.9673 CV)
        - Correlation_Filter (49 features, 0.9669 CV)
        - Permutation_Importance (45 features, 0.9664 CV)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, RFE, RFECV,
    f_classif, mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import warnings
import os
from typing import Dict, List, Tuple
import logging

warnings.filterwarnings('ignore')

# Try to import CuPy for GPU support
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Global XGBoost parameters
N_ESTIMATORS = 1000
MAX_DEPTH = 8
USE_GPU = True  # Set to False to use CPU (GPU may show device mismatch warnings during prediction)

# GPU parameters if enabled
if USE_GPU and CUPY_AVAILABLE:
    XGB_PARAMS = {
        'tree_method': 'hist',
        'device': 'cuda',
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'learning_rate': 0.1,
        'random_state': 42
    }
    logger.info("GPU mode enabled for XGBoost (CUDA with CuPy)")
elif USE_GPU and not CUPY_AVAILABLE:
    XGB_PARAMS = {
        'tree_method': 'hist',
        'device': 'cuda',
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'learning_rate': 0.1,
        'random_state': 42
    }
    logger.info("GPU mode enabled for XGBoost (CUDA without CuPy - may show device warnings)")
else:
    XGB_PARAMS = {
        'tree_method': 'hist',
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    logger.info("CPU mode enabled for XGBoost")


class FeatureGenerator:
    """Generate features using same logic as in feature_selection_comprehensive.py"""
    
    def __init__(self):
        self.generated_features = []
        
    def generate_features(self, df: pd.DataFrame, numeric_cols: List[str], 
                         categorical_cols: List[str]) -> pd.DataFrame:
        """Generate comprehensive feature set."""
        logger.info("  Generating features...")
        df_features = df.copy()
        original_cols = list(df.columns)
        
        # First, create NaN indicator columns
        for col in df.columns:
            if df[col].isna().any():
                df_features[f'{col}_was_nan'] = df[col].isna().astype(int)
        
        # 1. Statistical features for numeric columns
        for col in numeric_cols:
            # Basic statistics
            df_features[f'{col}_squared'] = df[col] ** 2
            df_features[f'{col}_cubed'] = df[col] ** 3
            df_features[f'{col}_sqrt'] = np.sqrt(df[col])
            df_features[f'{col}_log1p'] = np.log1p(df[col])
            
            # Standardized versions
            if df[col].std() > 0:
                df_features[f'{col}_standardized'] = (df[col] - df[col].mean()) / df[col].std()
            
            # Rank transform
            df_features[f'{col}_rank'] = df[col].rank() / len(df)
            
            # Binning
            df_features[f'{col}_bin5'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
            df_features[f'{col}_bin10'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
        
        # 2. Interaction features between numeric columns
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Arithmetic operations
                df_features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                df_features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                df_features[f'{col1}_times_{col2}'] = df[col1] * df[col2]
                df_features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
                # Min/max
                df_features[f'{col1}_max_{col2}'] = df[[col1, col2]].max(axis=1)
                df_features[f'{col1}_min_{col2}'] = df[[col1, col2]].min(axis=1)
                
                # Ratios and differences
                df_features[f'{col1}_ratio_{col2}'] = df[col1] / (df[col2] + 1)
                df_features[f'{col1}_diff_{col2}'] = np.abs(df[col1] - df[col2])
        
        # 3. Aggregation features across numeric columns
        df_features['numeric_sum'] = df[numeric_cols].sum(axis=1, skipna=True)
        df_features['numeric_mean'] = df[numeric_cols].mean(axis=1, skipna=True)
        df_features['numeric_std'] = df[numeric_cols].std(axis=1, skipna=True)
        df_features['numeric_min'] = df[numeric_cols].min(axis=1, skipna=True)
        df_features['numeric_max'] = df[numeric_cols].max(axis=1, skipna=True)
        df_features['numeric_range'] = df_features['numeric_max'] - df_features['numeric_min']
        df_features['numeric_skew'] = df[numeric_cols].skew(axis=1, skipna=True)
        df_features['numeric_kurtosis'] = df[numeric_cols].kurtosis(axis=1, skipna=True)
        
        # Percentiles
        df_features['numeric_q25'] = df[numeric_cols].quantile(0.25, axis=1, interpolation='linear')
        df_features['numeric_q75'] = df[numeric_cols].quantile(0.75, axis=1, interpolation='linear')
        df_features['numeric_iqr'] = df_features['numeric_q75'] - df_features['numeric_q25']
        
        # 4. Domain-specific features for this dataset
        # Introvert/extrovert patterns
        df_features['introvert_score'] = (
            df['Time_spent_Alone'] * 2 - 
            df['Social_event_attendance'] - 
            df['Friends_circle_size'] / 2
        )
        
        df_features['social_activity_ratio'] = (
            df['Social_event_attendance'] / (df['Time_spent_Alone'] + 1)
        )
        
        df_features['online_offline_balance'] = (
            df['Post_frequency'] / (df['Going_outside'] + 1)
        )
        
        # Finally, fill NaN values for all columns
        for col in df_features.columns:
            if df_features[col].isna().any():
                if df_features[col].dtype == 'object':
                    df_features[col] = df_features[col].fillna('Missing')
                else:
                    df_features[col] = df_features[col].fillna(-9999)
        
        new_features = [col for col in df_features.columns if col not in original_cols]
        logger.info(f"    Generated {len(new_features)} new features")
        
        return df_features


def prepare_data(df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare data with feature engineering."""
    # Encode Yes/No columns
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' and col not in ['Personality', 'id']:
            if set(df_encoded[col].dropna().unique()) <= {'Yes', 'No'}:
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
    
    # Define column types
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Going_outside', 'Friends_circle_size', 'Post_frequency']
    categorical_cols = []
    
    # Generate features
    generator = FeatureGenerator()
    df_features = generator.generate_features(df_encoded, numeric_cols, categorical_cols)
    
    # Prepare X and y
    if is_train:
        X = df_features.drop(columns=['Personality', 'id'])
        y = df_features['Personality']
        
        # Encode target
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        return X, y
    else:
        # For test set, preserve id column
        ids = df_features['id']
        X = df_features.drop(columns=['id'])
        return X, ids


def get_feature_selection_results():
    """Define feature selection methods and their results from the analysis."""
    return [
        {
            'name': 'RFECV_optimal',
            'score': 0.9678,
            'n_features': 13,
            'method': 'rfecv'
        },
        {
            'name': 'Random_Forest_Importance',
            'score': 0.9676,
            'n_features': 104,
            'method': 'rf_importance',
            'threshold': 0.001
        },
        {
            'name': 'XGBoost_Importance',
            'score': 0.9675,
            'n_features': 61,
            'method': 'xgb_importance',
            'threshold': 0.001
        },
        {
            'name': 'Baseline_all',
            'score': 0.9673,
            'n_features': 148,
            'method': 'all'
        },
        {
            'name': 'Lasso_L1',
            'score': 0.9673,
            'n_features': 134,
            'method': 'lasso',
            'alpha': 0.01
        },
        {
            'name': 'Correlation_Filter',
            'score': 0.9669,
            'n_features': 49,
            'method': 'correlation',
            'threshold': 0.95
        },
        {
            'name': 'Permutation_Importance',
            'score': 0.9664,
            'n_features': 45,
            'method': 'permutation',
            'threshold': 0.001
        }
    ]


def apply_feature_selection(X_train, y_train, X_test, method_info):
    """Apply specific feature selection method to both train and test data."""
    method = method_info['method']
    
    if method == 'all':
        return X_train, X_test
    
    elif method == 'rfecv':
        estimator = xgb.XGBClassifier(**XGB_PARAMS)
        selector = RFECV(estimator, step=5, cv=3, scoring='accuracy', n_jobs=-1, min_features_to_select=5)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.support_]
        
    elif method == 'rfe':
        estimator = xgb.XGBClassifier(**XGB_PARAMS)
        selector = RFE(estimator, n_features_to_select=method_info['n_features'], step=5)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.support_]
        
    elif method == 'xgb_importance':
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        selected_features = importances[importances > method_info['threshold']].index
        
    elif method == 'rf_importance':
        model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1, max_depth=8)
        model.fit(X_train, y_train)
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        selected_features = importances[importances > method_info['threshold']].index
        
    elif method == 'permutation':
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)
        perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
        importances = pd.Series(perm_importance.importances_mean, index=X_train.columns)
        selected_features = importances[importances > method_info['threshold']].index
        
    elif method == 'correlation':
        corr_matrix = X_train.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > method_info['threshold'])]
        selected_features = [col for col in X_train.columns if col not in to_drop]
        
    elif method == 'lasso':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1/method_info['alpha'], random_state=42)
        lasso.fit(X_scaled, y_train)
        selected_features = X_train.columns[np.abs(lasso.coef_[0]) > 1e-5]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return X_train[selected_features], X_test[selected_features]


def main():
    # Paths
    data_dir = "../datasets/playground-series-s5e7"
    output_dir = "./submissions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading datasets...")
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    # Prepare data with feature engineering
    logger.info("\nPreparing train data...")
    X_train, y_train = prepare_data(train_df, is_train=True)
    
    logger.info("\nPreparing test data...")
    X_test, test_ids = prepare_data(test_df, is_train=False)
    
    logger.info(f"\nFeature-engineered shapes:")
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"X_test: {X_test.shape}")
    
    # Get feature selection configurations
    methods = get_feature_selection_results()
    
    # For each method, apply feature selection and generate predictions
    logger.info("\n" + "="*80)
    logger.info("GENERATING PREDICTIONS FOR EACH METHOD")
    logger.info("="*80)
    
    for method_info in methods:
        logger.info(f"\n{method_info['name']}:")
        logger.info(f"  CV Score: {method_info['score']:.4f}")
        logger.info(f"  Features: {method_info['n_features']}")
        
        # Apply feature selection
        X_train_selected, X_test_selected = apply_feature_selection(
            X_train.copy(), y_train, X_test.copy(), method_info
        )
        logger.info(f"  Selected features shape: {X_train_selected.shape}")
        
        # Train final model
        logger.info("  Training final model...")
        model = xgb.XGBClassifier(**XGB_PARAMS)
        
        # Convert to CuPy arrays if GPU is enabled and CuPy is available
        if USE_GPU and CUPY_AVAILABLE:
            X_train_gpu = cp.asarray(X_train_selected.values)
            X_test_gpu = cp.asarray(X_test_selected.values)
            model.fit(X_train_gpu, y_train)
            y_pred = model.predict(X_test_gpu)
        else:
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
        
        # Convert back to original labels
        label_map = {0: 'Extrovert', 1: 'Introvert'}
        predictions = [label_map[pred] for pred in y_pred]
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_ids,
            'Personality': predictions
        })
        
        # Generate filename with score
        filename = f"submission_{method_info['name']}_score{method_info['score']:.4f}_features{method_info['n_features']}.csv"
        filepath = os.path.join(output_dir, filename)
        
        submission.to_csv(filepath, index=False)
        logger.info(f"  Saved: {filename}")
        
        # Show prediction distribution
        pred_dist = submission['Personality'].value_counts()
        logger.info(f"  Predictions: {pred_dist['Introvert']} Introverts, {pred_dist['Extrovert']} Extroverts")
    
    logger.info("\n" + "="*80)
    logger.info("ALL SUBMISSIONS GENERATED SUCCESSFULLY!")
    logger.info(f"Files saved in: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
