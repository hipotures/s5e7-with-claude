#!/usr/bin/env python3
"""
COMPREHENSIVE MISSING VALUE HANDLING STRATEGIES TEST
===================================================

This script implements and tests various strategies for handling missing values
in the personality classification dataset, with the goal of potentially breaking
through the 0.975708 accuracy ceiling.

Author: Claude
Date: 2025-07-05 09:39
"""

# PURPOSE: Test multiple missing value handling strategies to find patterns that might improve accuracy
# HYPOTHESIS: Missing values may encode personality information that can be leveraged for better predictions
# EXPECTED: Identify the best missing value handling strategy and potentially break the 0.975708 ceiling
# RESULT: [To be determined after execution]

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

class MissingValueAnalyzer:
    """Comprehensive analysis of missing value patterns and their relationship with target."""
    
    def __init__(self):
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data."""
        print("Loading data...")
        self.train_df = pd.read_csv("../../train.csv")
        self.test_df = pd.read_csv("../../test.csv")
        
        # Store original missing patterns before any preprocessing
        self.train_missing_original = self.train_df.isnull()
        self.test_missing_original = self.test_df.isnull()
        
        # Separate features and target
        self.feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                            'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        
        # Store target before preprocessing
        self.y_train = (self.train_df['Personality'] == 'Extrovert').astype(int)
        
        # Convert categorical to numeric for easier processing
        for col in ['Stage_fear', 'Drained_after_socializing']:
            if col in self.train_df.columns:
                # Store original values including NaN
                train_vals = self.train_df[col].copy()
                test_vals = self.test_df[col].copy()
                
                # Map non-null values
                mapping = {'Yes': 1, 'No': 0}
                self.train_df[col] = train_vals.map(mapping)
                self.test_df[col] = test_vals.map(mapping)
        
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        
        # Analyze missing patterns
        self.analyze_missing_patterns()
        
    def analyze_missing_patterns(self):
        """Analyze missing value patterns in the dataset."""
        print("\n" + "="*60)
        print("MISSING VALUE ANALYSIS")
        print("="*60)
        
        # Missing value statistics
        print("\nMissing values per feature (Training):")
        for col in self.feature_cols:
            missing_count = self.train_df[col].isnull().sum()
            missing_pct = missing_count / len(self.train_df) * 100
            print(f"  {col:.<30} {missing_count:>5} ({missing_pct:>5.2f}%)")
        
        # Missing patterns
        missing_patterns = self.train_missing_original[self.feature_cols].astype(int).astype(str).agg(''.join, axis=1)
        pattern_counts = missing_patterns.value_counts()
        
        print(f"\nUnique missing patterns: {len(pattern_counts)}")
        print("\nTop 10 most common patterns:")
        for pattern, count in pattern_counts.head(10).items():
            pct = count / len(self.train_df) * 100
            print(f"  Pattern {pattern}: {count:>5} ({pct:>5.2f}%)")
        
        # Correlation between missingness and target
        print("\nCorrelation between missingness and target:")
        correlations = {}
        chi2_results = {}
        
        for col in self.feature_cols:
            is_missing = self.train_df[col].isnull().astype(int)
            corr = np.corrcoef(is_missing, self.y_train)[0, 1]
            correlations[col] = corr
            
            # Chi-square test
            contingency_table = pd.crosstab(is_missing, self.y_train)
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            chi2_results[col] = {'chi2': chi2, 'p_value': p_value}
            
            print(f"  {col:.<30} corr={corr:>6.3f}, p={p_value:>6.4f}")
        
        self.results['missing_correlations'] = correlations
        self.results['chi2_tests'] = chi2_results
        
    def strategy_1_missing_indicators(self):
        """Strategy 1: Using missing indicators with imputation."""
        print("\n" + "="*60)
        print("STRATEGY 1: MISSING INDICATORS")
        print("="*60)
        
        X_train = self.train_df[self.feature_cols]
        
        # Create pipeline with missing indicators
        preprocessor = FeatureUnion([
            ('imputer', SimpleImputer(strategy='median')),
            ('indicators', MissingIndicator())
        ])
        
        # Test multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False),
            'LightGBM': lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        strategy1_results = {}
        
        for model_name, model in models.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            scores = cross_val_score(pipeline, X_train, self.y_train, cv=cv, scoring='accuracy')
            mean_score = scores.mean()
            std_score = scores.std()
            
            print(f"{model_name}: {mean_score:.6f} (+/- {std_score:.6f})")
            strategy1_results[model_name] = {
                'mean': mean_score,
                'std': std_score,
                'scores': scores.tolist()
            }
        
        self.results['strategy1_missing_indicators'] = strategy1_results
        
    def strategy_2_pattern_submodels(self):
        """Strategy 2: Create separate models for each missing pattern."""
        print("\n" + "="*60)
        print("STRATEGY 2: PATTERN SUBMODELS")
        print("="*60)
        
        X_train = self.train_df[self.feature_cols]
        
        # Identify missing patterns
        missing_patterns = X_train.isnull().astype(int).astype(str).agg(''.join, axis=1)
        unique_patterns = missing_patterns.value_counts()
        
        print(f"Creating submodels for {len(unique_patterns)} unique patterns...")
        
        # Only create submodels for patterns with sufficient data
        min_samples = 50
        valid_patterns = unique_patterns[unique_patterns >= min_samples].index
        
        print(f"Patterns with >= {min_samples} samples: {len(valid_patterns)}")
        
        # Train submodels
        submodels = {}
        for pattern in valid_patterns:
            mask = missing_patterns == pattern
            X_subset = X_train[mask]
            y_subset = self.y_train[mask]
            
            # Get available columns for this pattern
            available_cols = [col for i, col in enumerate(self.feature_cols) 
                             if pattern[i] == '0']
            
            if len(available_cols) > 0:
                # Simple imputation for any remaining missing values
                imputer = SimpleImputer(strategy='median')
                X_imputed = imputer.fit_transform(X_subset[available_cols])
                
                # Train model
                model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
                model.fit(X_imputed, y_subset)
                
                submodels[pattern] = {
                    'model': model,
                    'imputer': imputer,
                    'features': available_cols,
                    'n_samples': len(X_subset)
                }
        
        # Evaluate using cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, self.y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]
            
            # Train submodels for this fold
            fold_submodels = self._train_pattern_submodels(X_fold_train, y_fold_train, valid_patterns)
            
            # Make predictions
            predictions = self._predict_with_submodels(X_fold_val, fold_submodels)
            
            score = accuracy_score(y_fold_val, predictions)
            scores.append(score)
            print(f"Fold {fold + 1}: {score:.6f}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"\nPattern Submodels: {mean_score:.6f} (+/- {std_score:.6f})")
        
        self.results['strategy2_pattern_submodels'] = {
            'mean': mean_score,
            'std': std_score,
            'scores': scores,
            'n_patterns': len(valid_patterns),
            'n_submodels': len(submodels)
        }
        
    def _train_pattern_submodels(self, X, y, valid_patterns):
        """Helper function to train pattern submodels."""
        missing_patterns = X.isnull().astype(int).astype(str).agg(''.join, axis=1)
        submodels = {}
        
        for pattern in valid_patterns:
            mask = missing_patterns == pattern
            if mask.sum() < 10:  # Skip if too few samples
                continue
                
            X_subset = X[mask]
            y_subset = y[mask]
            
            available_cols = [col for i, col in enumerate(self.feature_cols) 
                             if pattern[i] == '0']
            
            if len(available_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                X_imputed = imputer.fit_transform(X_subset[available_cols])
                
                model = xgb.XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False)
                model.fit(X_imputed, y_subset)
                
                submodels[pattern] = {
                    'model': model,
                    'imputer': imputer,
                    'features': available_cols
                }
        
        # Default model for unknown patterns
        imputer_default = SimpleImputer(strategy='median')
        X_imputed_default = imputer_default.fit_transform(X)
        model_default = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
        model_default.fit(X_imputed_default, y)
        
        submodels['default'] = {
            'model': model_default,
            'imputer': imputer_default,
            'features': self.feature_cols
        }
        
        return submodels
        
    def _predict_with_submodels(self, X, submodels):
        """Helper function to make predictions with pattern submodels."""
        missing_patterns = X.isnull().astype(int).astype(str).agg(''.join, axis=1)
        predictions = np.zeros(len(X))
        
        for i, pattern in enumerate(missing_patterns):
            if pattern in submodels:
                submodel_info = submodels[pattern]
            else:
                submodel_info = submodels['default']
            
            X_row = X.iloc[[i]][submodel_info['features']]
            X_imputed = submodel_info['imputer'].transform(X_row)
            predictions[i] = submodel_info['model'].predict(X_imputed)[0]
        
        return predictions
        
    def strategy_3_advanced_features(self):
        """Strategy 3: Create advanced missing-pattern features."""
        print("\n" + "="*60)
        print("STRATEGY 3: ADVANCED MISSING FEATURES")
        print("="*60)
        
        # Create advanced features based on missing patterns
        X_train = self.train_df[self.feature_cols].copy()
        
        # Add missing indicators
        for col in self.feature_cols:
            X_train[f'{col}_missing'] = X_train[col].isnull().astype(int)
        
        # Add missing count
        X_train['total_missing'] = X_train[self.feature_cols].isnull().sum(axis=1)
        
        # Add pattern-based features
        X_train['has_social_missing'] = (X_train['Social_event_attendance'].isnull() | 
                                         X_train['Going_outside'].isnull()).astype(int)
        X_train['has_psychological_missing'] = (X_train['Stage_fear'].isnull() | 
                                               X_train['Drained_after_socializing'].isnull()).astype(int)
        
        # Missing pattern hash (for categorical encoding)
        pattern_hash = X_train[self.feature_cols].isnull().astype(int).astype(str).agg(''.join, axis=1)
        pattern_encoding = pd.factorize(pattern_hash)[0]
        X_train['pattern_code'] = pattern_encoding
        
        # Impute original features
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_train[self.feature_cols]),
            columns=self.feature_cols
        )
        
        # Combine all features
        X_final = pd.concat([X_imputed, X_train.iloc[:, len(self.feature_cols):]], axis=1)
        
        # Test with XGBoost
        model = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42, use_label_encoder=False)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_final, self.y_train, cv=cv, scoring='accuracy')
        
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"Advanced Features XGBoost: {mean_score:.6f} (+/- {std_score:.6f})")
        
        # Feature importance analysis
        model.fit(X_final, self.y_train)
        feature_importance = pd.DataFrame({
            'feature': X_final.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:.<30} {row['importance']:.4f}")
        
        self.results['strategy3_advanced_features'] = {
            'mean': mean_score,
            'std': std_score,
            'scores': scores.tolist(),
            'top_features': feature_importance.head(10).to_dict('records')
        }
        
    def compare_with_baseline(self):
        """Compare all strategies with a simple baseline."""
        print("\n" + "="*60)
        print("BASELINE COMPARISON")
        print("="*60)
        
        # Simple baseline: median imputation only
        X_train = self.train_df[self.feature_cols]
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_train)
        
        model = xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_imputed, self.y_train, cv=cv, scoring='accuracy')
        
        baseline_mean = scores.mean()
        baseline_std = scores.std()
        print(f"Baseline (simple imputation): {baseline_mean:.6f} (+/- {baseline_std:.6f})")
        
        self.results['baseline'] = {
            'mean': baseline_mean,
            'std': baseline_std,
            'scores': scores.tolist()
        }
        
        # Summary comparison
        print("\n" + "="*60)
        print("SUMMARY COMPARISON")
        print("="*60)
        
        print(f"Baseline:                     {baseline_mean:.6f}")
        
        if 'strategy1_missing_indicators' in self.results:
            best_s1 = max(self.results['strategy1_missing_indicators'].items(), 
                         key=lambda x: x[1]['mean'])
            print(f"Strategy 1 (best):            {best_s1[1]['mean']:.6f} ({best_s1[0]})")
        
        if 'strategy2_pattern_submodels' in self.results:
            print(f"Strategy 2:                   {self.results['strategy2_pattern_submodels']['mean']:.6f}")
        
        if 'strategy3_advanced_features' in self.results:
            print(f"Strategy 3:                   {self.results['strategy3_advanced_features']['mean']:.6f}")
        
    def visualize_missing_patterns(self):
        """Create visualizations of missing patterns."""
        print("\n" + "="*60)
        print("VISUALIZING MISSING PATTERNS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Missing value heatmap
        ax1 = axes[0, 0]
        missing_matrix = self.train_df[self.feature_cols].isnull().astype(int)
        im1 = ax1.imshow(missing_matrix.T, cmap='viridis', aspect='auto')
        ax1.set_yticks(range(len(self.feature_cols)))
        ax1.set_yticklabels(self.feature_cols)
        ax1.set_xlabel('Sample Index')
        ax1.set_title('Missing Value Pattern Matrix')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Missing correlation with target
        ax2 = axes[0, 1]
        correlations = []
        for col in self.feature_cols:
            corr = np.corrcoef(self.train_df[col].isnull().astype(int), self.y_train)[0, 1]
            correlations.append(corr)
        
        bars = ax2.bar(range(len(self.feature_cols)), correlations)
        ax2.set_xticks(range(len(self.feature_cols)))
        ax2.set_xticklabels(self.feature_cols, rotation=45, ha='right')
        ax2.set_ylabel('Correlation with Target')
        ax2.set_title('Missingness Correlation with Personality')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Color bars based on correlation
        for i, bar in enumerate(bars):
            if correlations[i] > 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        # 3. Pattern frequency distribution
        ax3 = axes[1, 0]
        pattern_counts = self.train_missing_original[self.feature_cols].astype(int).astype(str).agg(''.join, axis=1).value_counts()
        top_patterns = pattern_counts.head(10)
        ax3.bar(range(len(top_patterns)), top_patterns.values)
        ax3.set_xticks(range(len(top_patterns)))
        ax3.set_xticklabels([f'P{i+1}' for i in range(len(top_patterns))])
        ax3.set_ylabel('Frequency')
        ax3.set_title('Top 10 Missing Patterns Frequency')
        
        # 4. Missing by personality type
        ax4 = axes[1, 1]
        personality_missing = []
        for personality in [0, 1]:  # Introvert, Extrovert
            mask = self.y_train == personality
            missing_rates = self.train_df[mask][self.feature_cols].isnull().mean()
            personality_missing.append(missing_rates)
        
        x = np.arange(len(self.feature_cols))
        width = 0.35
        ax4.bar(x - width/2, personality_missing[0], width, label='Introvert')
        ax4.bar(x + width/2, personality_missing[1], width, label='Extrovert')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.feature_cols, rotation=45, ha='right')
        ax4.set_ylabel('Missing Rate')
        ax4.set_title('Missing Rates by Personality Type')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('output/missing_patterns_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualization saved to: output/missing_patterns_visualization.png")
        
    def save_results(self):
        """Save all results to JSON file."""
        import os
        os.makedirs('output', exist_ok=True)
        
        with open('output/missing_value_strategies_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\nResults saved to: output/missing_value_strategies_results.json")


def main():
    """Main execution function."""
    analyzer = MissingValueAnalyzer()
    
    # Load and analyze data
    analyzer.load_and_preprocess_data()
    
    # Run all strategies
    analyzer.strategy_1_missing_indicators()
    analyzer.strategy_2_pattern_submodels()
    analyzer.strategy_3_advanced_features()
    
    # Compare with baseline
    analyzer.compare_with_baseline()
    
    # Create visualizations
    analyzer.visualize_missing_patterns()
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()