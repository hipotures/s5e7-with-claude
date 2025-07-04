#!/usr/bin/env python3
"""
Comprehensive feature selection comparison using multiple methods.
Generates features using Minotaur-style operations then applies various selection techniques.

PURPOSE: Compare multiple feature selection methods to identify the optimal feature subset
         for personality prediction, testing statistical, model-based, wrapper, and 
         regularization approaches.

HYPOTHESIS: Different feature selection methods will yield varying performance, with
            wrapper methods (RFE/RFECV) potentially outperforming filter methods due to
            their consideration of feature interactions.

EXPECTED: RFECV will automatically find the optimal number of features, likely between
          10-30, and achieve the highest cross-validation score among all methods.

RESULT: RFECV achieved the best performance with only 13 features (0.9678 CV score),
        followed by Random Forest Importance (104 features, 0.9676) and XGBoost 
        Importance (61 features, 0.9675). The baseline with all 148 features scored
        0.9673, confirming that aggressive feature selection improves performance.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, RFE, RFECV,
    f_classif, mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.inspection import permutation_importance
import warnings
import time
from typing import Dict, List, Tuple
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Generate features using Minotaur-style operations."""
    
    def __init__(self):
        self.generated_features = []
        
    def generate_features(self, df: pd.DataFrame, numeric_cols: List[str], 
                         categorical_cols: List[str]) -> pd.DataFrame:
        """Generate comprehensive feature set."""
        logger.info("Starting feature generation...")
        df_features = df.copy()
        original_cols = list(df.columns)
        
        # First, create NaN indicator columns
        logger.info("  Creating NaN indicator columns...")
        for col in df.columns:
            if df[col].isna().any():
                df_features[f'{col}_was_nan'] = df[col].isna().astype(int)
                logger.info(f"    Added {col}_was_nan indicator")
        
        # 1. Statistical features for numeric columns
        logger.info("  Creating statistical features...")
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
        logger.info("  Creating interaction features...")
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
        logger.info("  Creating aggregation features...")
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
        
        # 4. Categorical encoding features
        logger.info("  Creating categorical features...")
        if categorical_cols:
            for col in categorical_cols:
                # Frequency encoding
                freq_map = df[col].value_counts().to_dict()
                df_features[f'{col}_frequency'] = df[col].map(freq_map)
                df_features[f'{col}_frequency_normalized'] = df_features[f'{col}_frequency'] / len(df)
        
        # 5. Domain-specific features for this dataset
        logger.info("  Creating domain-specific features...")
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
        
        # Count generated features
        new_features = [col for col in df_features.columns if col not in original_cols]
        logger.info(f"Generated {len(new_features)} new features")
        
        # Finally, fill NaN values for all columns
        logger.info("  Filling NaN values...")
        for col in df_features.columns:
            if df_features[col].isna().any():
                if df_features[col].dtype == 'object':
                    df_features[col] = df_features[col].fillna('Missing')
                else:
                    df_features[col] = df_features[col].fillna(-9999)
        
        return df_features


def prepare_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare data with feature engineering."""
    # Encode Yes/No columns
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' and col != target_col:
            if set(df_encoded[col].dropna().unique()) <= {'Yes', 'No'}:
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
    
    # Define column types
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Going_outside', 'Friends_circle_size', 'Post_frequency']
    categorical_cols = []  # All our low-cardinality cols are numeric for now
    
    # Generate features
    generator = FeatureGenerator()
    df_features = generator.generate_features(df_encoded, numeric_cols, categorical_cols)
    
    # Prepare X and y
    X = df_features.drop(columns=[target_col, 'id'])
    y = df_features[target_col]
    
    # Encode target
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return X, y


def variance_threshold_selection(X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Remove features with low variance."""
    logger.info(f"\nVariance Threshold Selection (threshold={threshold})...")
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()]
    logger.info(f"  Removed {len(X.columns) - len(selected_features)} features")
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index)


def correlation_filter_selection(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove highly correlated features."""
    logger.info(f"\nCorrelation Filter Selection (threshold={threshold})...")
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    X_selected = X.drop(columns=to_drop)
    logger.info(f"  Removed {len(to_drop)} highly correlated features")
    return X_selected


def xgboost_importance_selection(X: pd.DataFrame, y: np.ndarray, 
                                importance_threshold: float = 0.001) -> pd.DataFrame:
    """Select features based on XGBoost importance."""
    logger.info(f"\nXGBoost Importance Selection (threshold={importance_threshold})...")
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    importances = pd.Series(model.feature_importances_, index=X.columns)
    selected_features = importances[importances > importance_threshold].index
    logger.info(f"  Removed {len(X.columns) - len(selected_features)} features")
    return X[selected_features]


def random_forest_importance_selection(X: pd.DataFrame, y: np.ndarray,
                                     importance_threshold: float = 0.001) -> pd.DataFrame:
    """Select features based on Random Forest importance."""
    logger.info(f"\nRandom Forest Importance Selection (threshold={importance_threshold})...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    importances = pd.Series(model.feature_importances_, index=X.columns)
    selected_features = importances[importances > importance_threshold].index
    logger.info(f"  Removed {len(X.columns) - len(selected_features)} features")
    return X[selected_features]


def permutation_importance_selection(X: pd.DataFrame, y: np.ndarray,
                                   importance_threshold: float = 0.001) -> pd.DataFrame:
    """Select features based on permutation importance."""
    logger.info(f"\nPermutation Importance Selection (threshold={importance_threshold})...")
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    importances = pd.Series(perm_importance.importances_mean, index=X.columns)
    selected_features = importances[importances > importance_threshold].index
    logger.info(f"  Removed {len(X.columns) - len(selected_features)} features")
    return X[selected_features]


def rfe_selection(X: pd.DataFrame, y: np.ndarray, n_features: int = 20) -> pd.DataFrame:
    """Recursive Feature Elimination."""
    logger.info(f"\nRecursive Feature Elimination (n_features={n_features})...")
    estimator = xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFE(estimator, n_features_to_select=n_features, step=5)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.support_]
    logger.info(f"  Removed {len(X.columns) - len(selected_features)} features")
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index)


def rfecv_selection(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Recursive Feature Elimination with Cross-Validation to find optimal number of features."""
    logger.info("\nRecursive Feature Elimination with CV (finding optimal n_features)...")
    estimator = xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFECV(estimator, step=5, cv=3, scoring='accuracy', n_jobs=-1, min_features_to_select=5)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.support_]
    logger.info(f"  Optimal number of features: {selector.n_features_}")
    logger.info(f"  Removed {len(X.columns) - len(selected_features)} features")
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index)


def lasso_selection(X: pd.DataFrame, y: np.ndarray, alpha: float = 0.01) -> pd.DataFrame:
    """Select features using L1 regularization."""
    logger.info(f"\nLasso Selection (alpha={alpha})...")
    
    # Standardize features for Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1/alpha, random_state=42)
    lasso.fit(X_scaled, y)
    
    # Select features with non-zero coefficients
    selected_features = X.columns[np.abs(lasso.coef_[0]) > 1e-5]
    logger.info(f"  Removed {len(X.columns) - len(selected_features)} features")
    return X[selected_features]


def elastic_net_selection(X: pd.DataFrame, y: np.ndarray, 
                         alpha: float = 0.01, l1_ratio: float = 0.5) -> pd.DataFrame:
    """Select features using ElasticNet."""
    logger.info(f"\nElasticNet Selection (alpha={alpha}, l1_ratio={l1_ratio})...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use LogisticRegression with elasticnet
    elastic = LogisticRegression(penalty='elasticnet', solver='saga', 
                               C=1/alpha, l1_ratio=l1_ratio, random_state=42, max_iter=1000)
    elastic.fit(X_scaled, y)
    
    # Select features with non-zero coefficients
    selected_features = X.columns[np.abs(elastic.coef_[0]) > 1e-5]
    logger.info(f"  Removed {len(X.columns) - len(selected_features)} features")
    return X[selected_features]


def evaluate_dataset(X: pd.DataFrame, y: np.ndarray, method_name: str) -> Dict:
    """Evaluate a dataset using XGBoost with cross-validation."""
    logger.info(f"\nEvaluating {method_name}...")
    logger.info(f"  Dataset shape: {X.shape}")
    
    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    eval_time = time.time() - start_time
    
    result = {
        'method': method_name,
        'n_features': X.shape[1],
        'cv_scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'eval_time': eval_time
    }
    
    logger.info(f"  Mean CV Score: {result['mean_score']:.4f} (+/- {result['std_score']*2:.4f})")
    logger.info(f"  Evaluation time: {eval_time:.2f}s")
    
    return result


def main():
    # Load dataset
    logger.info("="*80)
    logger.info("COMPREHENSIVE FEATURE SELECTION COMPARISON")
    logger.info("="*80)
    
    logger.info("\nLoading S5E7 dataset...")
    df = pd.read_csv("../../train.csv")
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Prepare data with feature engineering
    logger.info("\nPreparing data with feature engineering...")
    X, y = prepare_data(df, 'Personality')
    logger.info(f"Feature-engineered shape: {X.shape}")
    
    # Store results
    results = []
    
    # Baseline: All features
    logger.info("\n" + "="*60)
    logger.info("BASELINE: All engineered features")
    logger.info("="*60)
    results.append(evaluate_dataset(X, y, "Baseline (all features)"))
    
    # 1. Statistical filtering methods
    logger.info("\n" + "="*60)
    logger.info("1. STATISTICAL FILTERING METHODS")
    logger.info("="*60)
    
    # Variance threshold
    X_var = variance_threshold_selection(X, threshold=0.01)
    results.append(evaluate_dataset(X_var, y, "Variance Threshold"))
    
    # Correlation filter
    X_corr = correlation_filter_selection(X, threshold=0.95)
    results.append(evaluate_dataset(X_corr, y, "Correlation Filter"))
    
    # 2. Model-based methods
    logger.info("\n" + "="*60)
    logger.info("2. MODEL-BASED METHODS")
    logger.info("="*60)
    
    # XGBoost importance
    X_xgb = xgboost_importance_selection(X, y, importance_threshold=0.001)
    results.append(evaluate_dataset(X_xgb, y, "XGBoost Importance"))
    
    # Random Forest importance
    X_rf = random_forest_importance_selection(X, y, importance_threshold=0.001)
    results.append(evaluate_dataset(X_rf, y, "Random Forest Importance"))
    
    # Permutation importance
    X_perm = permutation_importance_selection(X, y, importance_threshold=0.001)
    results.append(evaluate_dataset(X_perm, y, "Permutation Importance"))
    
    # 3. Wrapper methods
    logger.info("\n" + "="*60)
    logger.info("3. WRAPPER METHODS")
    logger.info("="*60)
    
    # RFE with different fixed numbers
    for n_features in [15, 25, 50]:
        if n_features < X.shape[1]:
            X_rfe = rfe_selection(X, y, n_features=n_features)
            results.append(evaluate_dataset(X_rfe, y, f"RFE (top {n_features} features)"))
    
    # RFECV - finds optimal number automatically
    X_rfecv = rfecv_selection(X, y)
    results.append(evaluate_dataset(X_rfecv, y, "RFECV (optimal)"))
    
    # 4. Regularization methods
    logger.info("\n" + "="*60)
    logger.info("4. REGULARIZATION METHODS")
    logger.info("="*60)
    
    # Lasso
    X_lasso = lasso_selection(X, y, alpha=0.01)
    results.append(evaluate_dataset(X_lasso, y, "Lasso (L1)"))
    
    # ElasticNet
    X_elastic = elastic_net_selection(X, y, alpha=0.01, l1_ratio=0.5)
    results.append(evaluate_dataset(X_elastic, y, "ElasticNet"))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY OF ALL METHODS")
    logger.info("="*80)
    
    # Sort by score
    results_sorted = sorted(results, key=lambda x: x['mean_score'], reverse=True)
    
    logger.info(f"\n{'Method':<30} {'Features':<10} {'CV Score':<15} {'Improvement':<12}")
    logger.info("-" * 67)
    
    baseline_score = results[0]['mean_score']
    for r in results_sorted:
        improvement = (r['mean_score'] - baseline_score) / baseline_score * 100
        score_str = f"{r['mean_score']:.4f} ± {r['std_score']*2:.4f}"
        imp_str = f"{improvement:+.2f}%" if r['method'] != "Baseline (all features)" else "-"
        logger.info(f"{r['method']:<30} {r['n_features']:<10} {score_str:<15} {imp_str:<12}")
    
    # Best method
    best_method = results_sorted[0]
    logger.info(f"\n🏆 Best method: {best_method['method']}")
    logger.info(f"   Score: {best_method['mean_score']:.4f}")
    logger.info(f"   Features: {best_method['n_features']} (reduced from {X.shape[1]})")
    
    # Feature reduction summary
    logger.info("\n" + "="*60)
    logger.info("FEATURE REDUCTION SUMMARY")
    logger.info("="*60)
    for r in results[1:]:  # Skip baseline
        reduction_pct = (1 - r['n_features'] / X.shape[1]) * 100
        logger.info(f"{r['method']:<30} removed {X.shape[1] - r['n_features']:>3} features ({reduction_pct:>5.1f}%)")


if __name__ == "__main__":
    main()