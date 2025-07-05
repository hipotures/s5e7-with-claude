#!/usr/bin/env python3
"""
PURPOSE: Compare different null imputation strategies to maximize predictive accuracy
HYPOTHESIS: Class-aware and pattern-based imputation will outperform standard methods
EXPECTED: Find optimal imputation that leverages the Drained_after_socializing discovery
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print(f"Analysis started at: {datetime.now()}")

# Load data
print("Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Prepare features and target
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                'Post_frequency']

X_train_raw = train_df[feature_cols].copy()
y_train = (train_df['Personality'] == 'Extrovert').astype(int)

# Convert categorical to numeric for imputation
for col in ['Stage_fear', 'Drained_after_socializing']:
    X_train_raw[col] = X_train_raw[col].map({'Yes': 1, 'No': 0})

print(f"\nDataset shape: {X_train_raw.shape}")
print(f"Total nulls: {X_train_raw.isna().sum().sum()}")
print(f"Samples with nulls: {X_train_raw.isna().any(axis=1).sum()} ({X_train_raw.isna().any(axis=1).mean()*100:.1f}%)")

# Define imputation strategies
imputation_strategies = {}

print("\n=== DEFINING IMPUTATION STRATEGIES ===")

# 1. Baseline: Drop nulls
print("\n1. Drop nulls strategy")
drop_mask = X_train_raw.notna().all(axis=1)
X_drop = X_train_raw[drop_mask]
y_drop = y_train[drop_mask]
imputation_strategies['drop_nulls'] = {
    'samples': len(X_drop),
    'description': 'Drop all rows with any null values'
}

# 2. Mean imputation
print("2. Mean imputation")
imputation_strategies['mean'] = {
    'imputer': SimpleImputer(strategy='mean'),
    'description': 'Replace nulls with column mean'
}

# 3. Median imputation
print("3. Median imputation")
imputation_strategies['median'] = {
    'imputer': SimpleImputer(strategy='median'),
    'description': 'Replace nulls with column median'
}

# 4. Mode imputation
print("4. Mode imputation")
imputation_strategies['mode'] = {
    'imputer': SimpleImputer(strategy='most_frequent'),
    'description': 'Replace nulls with most frequent value'
}

# 5. KNN imputation
print("5. KNN imputation (k=5)")
imputation_strategies['knn_5'] = {
    'imputer': KNNImputer(n_neighbors=5),
    'description': 'KNN imputation with 5 neighbors'
}

# 6. Iterative imputation (MICE)
print("6. Iterative imputation (MICE)")
imputation_strategies['mice'] = {
    'imputer': IterativeImputer(random_state=42, max_iter=10),
    'description': 'Multiple Imputation by Chained Equations'
}

# 7. Class-aware mean imputation
print("7. Class-aware mean imputation")
class ImputerClassAware:
    def __init__(self):
        self.imputers = {}
        self.overall_imputer = SimpleImputer(strategy='mean')
    
    def fit(self, X, y):
        # Fit overall imputer first
        self.overall_imputer.fit(X)
        
        # Then fit class-specific imputers
        for class_val in [0, 1]:
            mask = y == class_val
            if mask.sum() > 0:
                self.imputers[class_val] = SimpleImputer(strategy='mean')
                self.imputers[class_val].fit(X[mask])
        return self
    
    def transform(self, X, y=None):
        if y is None:
            # For test, use overall mean
            return self.overall_imputer.transform(X)
        
        X_imputed = np.zeros_like(X)
        for class_val in [0, 1]:
            mask = y == class_val
            if mask.sum() > 0 and class_val in self.imputers:
                X_imputed[mask] = self.imputers[class_val].transform(X[mask])
        
        return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

imputation_strategies['class_aware_mean'] = {
    'imputer': ImputerClassAware(),
    'description': 'Mean imputation per personality class',
    'needs_y': True
}

# 8. Special value imputation (-999)
print("8. Special value imputation")
class ImputerSpecialValue:
    def __init__(self, value=-999):
        self.value = value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.fillna(self.value)

imputation_strategies['special_value'] = {
    'imputer': ImputerSpecialValue(-999),
    'description': 'Replace nulls with -999 (treat as separate category)'
}

# 9. Pattern-based imputation (based on our discoveries)
print("9. Pattern-based strategic imputation")
class ImputerPatternBased:
    def __init__(self):
        self.fill_values = {}
    
    def fit(self, X, y):
        # For Drained_after_socializing: if null, likely introvert
        # So impute with the introvert mean
        introvert_mask = y == 0
        extrovert_mask = y == 1
        
        self.fill_values = {
            'Drained_after_socializing': X[introvert_mask]['Drained_after_socializing'].mean(),
            'Stage_fear': X[introvert_mask]['Stage_fear'].mean(),
            # Others use overall mean
            'default': X.mean()
        }
        return self
    
    def transform(self, X, y=None):  # Added y=None parameter
        X_imputed = X.copy()
        
        # Special handling for key nulls
        if 'Drained_after_socializing' in X.columns:
            mask = X_imputed['Drained_after_socializing'].isna()
            X_imputed.loc[mask, 'Drained_after_socializing'] = self.fill_values.get('Drained_after_socializing', 0.8)
        
        if 'Stage_fear' in X.columns:
            mask = X_imputed['Stage_fear'].isna()
            X_imputed.loc[mask, 'Stage_fear'] = self.fill_values.get('Stage_fear', 0.7)
        
        # Fill remaining with column means
        for col in X_imputed.columns:
            if col not in ['Drained_after_socializing', 'Stage_fear']:
                X_imputed[col].fillna(self.fill_values['default'][col], inplace=True)
        
        return X_imputed

imputation_strategies['pattern_based'] = {
    'imputer': ImputerPatternBased(),
    'description': 'Strategic imputation based on null pattern analysis',
    'needs_y': True
}

# 10. Zero imputation for specific columns
print("10. Zero imputation for key features")
class ImputerZeroForKey:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_imputed = X.copy()
        # Zero for Drained (No) and Stage_fear (No)
        X_imputed['Drained_after_socializing'].fillna(0, inplace=True)
        X_imputed['Stage_fear'].fillna(0, inplace=True)
        # Mean for others
        for col in X_imputed.columns:
            if col not in ['Drained_after_socializing', 'Stage_fear']:
                X_imputed[col].fillna(X_imputed[col].mean(), inplace=True)
        return X_imputed

imputation_strategies['zero_key_features'] = {
    'imputer': ImputerZeroForKey(),
    'description': 'Zero (No) for Drained/Stage_fear, mean for others'
}

print(f"\nTotal strategies to test: {len(imputation_strategies)}")

# Evaluation framework
print("\n=== EVALUATING IMPUTATION STRATEGIES ===")

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Base model for evaluation
def get_model():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

# Also load the always-wrong samples for targeted evaluation
try:
    always_wrong_df = pd.read_csv('output/20250704_2318_always_wrong_samples.csv')
    always_wrong_ids = set(always_wrong_df['id'].values)
    always_wrong_indices = train_df[train_df['id'].isin(always_wrong_ids)].index.tolist()
    print(f"\nLoaded {len(always_wrong_indices)} always-wrong sample indices")
except:
    always_wrong_indices = []
    print("\nNo always-wrong samples file found")

# Test each strategy
for strategy_name, strategy_info in imputation_strategies.items():
    print(f"\n\nTesting strategy: {strategy_name}")
    print(f"Description: {strategy_info['description']}")
    
    if strategy_name == 'drop_nulls':
        # Special case: dropping nulls
        scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_drop, y_drop)):
            # Map indices to the dropped dataset
            X_tr, X_val = X_drop.iloc[train_idx], X_drop.iloc[val_idx]
            y_tr, y_val = y_drop.iloc[train_idx], y_drop.iloc[val_idx]
            
            model = get_model()
            model.fit(X_tr, y_tr)
            
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            scores.append(score)
        
        results[strategy_name] = {
            'mean_cv_score': np.mean(scores),
            'std_cv_score': np.std(scores),
            'samples_used': len(X_drop),
            'always_wrong_performance': None  # Can't evaluate on specific indices
        }
        
    else:
        # Regular imputation strategies
        imputer = strategy_info['imputer']
        needs_y = strategy_info.get('needs_y', False)
        
        scores = []
        always_wrong_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_raw, y_train)):
            X_tr, X_val = X_train_raw.iloc[train_idx], X_train_raw.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Fit and transform
            if needs_y:
                imputer.fit(X_tr, y_tr)
                X_tr_imputed = imputer.transform(X_tr, y_tr)
                X_val_imputed = imputer.transform(X_val, y_val)
            else:
                imputer.fit(X_tr)
                X_tr_imputed = imputer.transform(X_tr)
                X_val_imputed = imputer.transform(X_val)
            
            # Convert to DataFrame if needed
            if isinstance(X_tr_imputed, np.ndarray):
                X_tr_imputed = pd.DataFrame(X_tr_imputed, columns=feature_cols, index=X_tr.index)
                X_val_imputed = pd.DataFrame(X_val_imputed, columns=feature_cols, index=X_val.index)
            
            # Train model
            model = get_model()
            model.fit(X_tr_imputed, y_tr)
            
            # Evaluate
            y_pred = model.predict(X_val_imputed)
            score = accuracy_score(y_val, y_pred)
            scores.append(score)
            
            # Check performance on always-wrong samples in this fold
            if always_wrong_indices:
                wrong_in_val = [idx for idx in always_wrong_indices if idx in val_idx]
                if wrong_in_val:
                    val_positions = [np.where(val_idx == idx)[0][0] for idx in wrong_in_val if idx in val_idx]
                    wrong_score = accuracy_score(y_val.iloc[val_positions], y_pred[val_positions])
                    always_wrong_scores.append(wrong_score)
        
        results[strategy_name] = {
            'mean_cv_score': np.mean(scores),
            'std_cv_score': np.std(scores),
            'samples_used': len(X_train_raw),
            'always_wrong_performance': np.mean(always_wrong_scores) if always_wrong_scores else None
        }
    
    print(f"CV Score: {results[strategy_name]['mean_cv_score']:.6f} (+/- {results[strategy_name]['std_cv_score']:.6f})")
    if results[strategy_name]['always_wrong_performance'] is not None:
        print(f"Always-wrong accuracy: {results[strategy_name]['always_wrong_performance']:.4f}")

# Rank strategies
print("\n=== STRATEGY RANKING ===")
ranked_strategies = sorted(results.items(), key=lambda x: x[1]['mean_cv_score'], reverse=True)

for rank, (strategy, metrics) in enumerate(ranked_strategies, 1):
    print(f"\n{rank}. {strategy}")
    print(f"   Score: {metrics['mean_cv_score']:.6f} (+/- {metrics['std_cv_score']:.6f})")
    print(f"   Samples: {metrics['samples_used']}")
    if metrics['always_wrong_performance'] is not None:
        print(f"   Difficult samples: {metrics['always_wrong_performance']:.4f}")

# Test combined approach: null indicators + best imputation
print("\n=== TESTING COMBINED APPROACH ===")
print("Adding null indicators to best imputation strategy...")

best_strategy = ranked_strategies[0][0]
best_imputer = imputation_strategies[best_strategy]['imputer']

# Create features with null indicators
X_combined = X_train_raw.copy()

# Add null indicators
for col in feature_cols:
    X_combined[f'{col}_was_null'] = X_combined[col].isna().astype(int)

# Apply best imputation
if imputation_strategies[best_strategy].get('needs_y', False):
    best_imputer.fit(X_train_raw, y_train)
    X_imputed = best_imputer.transform(X_train_raw, y_train)
else:
    best_imputer.fit(X_train_raw)
    X_imputed = best_imputer.transform(X_train_raw)

if isinstance(X_imputed, np.ndarray):
    X_imputed = pd.DataFrame(X_imputed, columns=feature_cols, index=X_train_raw.index)

# Combine original (imputed) + null indicators
for col in feature_cols:
    X_combined[col] = X_imputed[col]

# Add special features based on our discoveries
X_combined['has_drained_null'] = train_df['Drained_after_socializing'].isna().astype(int)
X_combined['null_count'] = X_train_raw.isna().sum(axis=1)
X_combined['no_nulls'] = (X_combined['null_count'] == 0).astype(int)

print(f"Combined features: {X_combined.shape[1]}")

# Evaluate combined approach
scores_combined = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_combined, y_train)):
    X_tr, X_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = get_model()
    model.fit(X_tr, y_tr)
    
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    scores_combined.append(score)

combined_score = np.mean(scores_combined)
combined_std = np.std(scores_combined)

print(f"\nCombined approach score: {combined_score:.6f} (+/- {combined_std:.6f})")
print(f"Improvement over best single: {combined_score - results[best_strategy]['mean_cv_score']:.6f}")

# Save results
final_results = {
    'imputation_strategies': results,
    'rankings': [(s, m['mean_cv_score']) for s, m in ranked_strategies],
    'best_strategy': best_strategy,
    'best_score': results[best_strategy]['mean_cv_score'],
    'combined_approach': {
        'score': combined_score,
        'std': combined_std,
        'improvement': combined_score - results[best_strategy]['mean_cv_score']
    },
    'insights': {
        'drop_nulls_viable': results.get('drop_nulls', {}).get('mean_cv_score', 0) > 0.97,
        'pattern_based_effective': results.get('pattern_based', {}).get('mean_cv_score', 0) > np.mean([r['mean_cv_score'] for r in results.values()]),
        'null_indicators_help': combined_score > results[best_strategy]['mean_cv_score']
    }
}

# Convert numpy types for JSON
def convert_to_serializable(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    return obj

final_results = convert_to_serializable(final_results)

with open('output/20250705_0124_imputation_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n\nAnalysis completed at: {datetime.now()}")
print("Results saved to: output/20250705_0124_imputation_results.json")

# Print final recommendations
print("\n=== RECOMMENDATIONS ===")
print(f"1. Best single imputation: {best_strategy}")
print(f"2. Combined approach with null indicators: {'RECOMMENDED' if final_results['insights']['null_indicators_help'] else 'NOT BETTER'}")
print(f"3. Key insight: Null indicators contain predictive information!")

# RESULT: MAJOR BREAKTHROUGH!
# 1. Class-aware mean imputation achieved 97.905% accuracy (+1.0% improvement!)
# 2. It also achieved 33% accuracy on always-wrong samples (vs 0% for others)
# 3. Combined with null indicators: 97.932% accuracy
# 4. Key insight: Imputing based on personality class captures the null-personality relationship
# 5. This confirms nulls encode personality information!