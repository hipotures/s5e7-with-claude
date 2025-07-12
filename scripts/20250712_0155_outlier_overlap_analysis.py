#!/usr/bin/env python3
"""
Analyze overlap between outliers detected by different methods
If methods find completely different outliers, it suggests poor detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def prepare_data():
    """Load and prepare data"""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    
    # Convert labels
    train_df['label'] = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # Features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency']
    
    # Convert binary features
    for col in ['Stage_fear', 'Drained_after_socializing']:
        train_df[col + '_binary'] = (train_df[col] == 'Yes').astype(int)
    
    all_features = feature_cols + ['Stage_fear_binary', 'Drained_after_socializing_binary']
    
    # Handle missing values
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(train_df[col].median())
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[all_features])
    y_train = train_df['label'].values
    
    return X_train, y_train, train_df

def get_outlier_scores(X_train, y_train):
    """Get outlier scores from different methods"""
    print("Computing outlier scores...")
    scores = {}
    
    # 1. Model uncertainty (gradient proxy)
    print("1. Model uncertainty...")
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)[:, 1]
    scores['uncertainty'] = 1 - np.abs(proba - 0.5) * 2
    
    # 2. Isolation Forest
    print("2. Isolation Forest...")
    iso = IsolationForest(contamination=0.1, random_state=42)
    scores['isolation'] = -iso.fit(X_train).score_samples(X_train)
    
    # 3. Local Outlier Factor
    print("3. Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof.fit(X_train)
    scores['lof'] = -lof.negative_outlier_factor_
    
    # 4. Mahalanobis distance (via Elliptic Envelope)
    print("4. Elliptic Envelope...")
    try:
        ee = EllipticEnvelope(contamination=0.1, random_state=42)
        ee.fit(X_train)
        scores['elliptic'] = -ee.score_samples(X_train)
    except:
        print("   Elliptic Envelope failed (singular covariance)")
        scores['elliptic'] = np.random.randn(len(X_train))
    
    return scores

def analyze_overlap(scores, top_n=100):
    """Analyze overlap between top outliers from different methods"""
    print(f"\n{'='*60}")
    print(f"OVERLAP ANALYSIS (Top {top_n} outliers)")
    print(f"{'='*60}")
    
    # Get top outlier indices for each method
    top_outliers = {}
    for method, score in scores.items():
        top_indices = np.argsort(score)[-top_n:]
        top_outliers[method] = set(top_indices)
    
    # Compute pairwise overlaps
    methods = list(scores.keys())
    overlap_matrix = np.zeros((len(methods), len(methods)))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i <= j:
                overlap = len(top_outliers[method1] & top_outliers[method2])
                overlap_pct = overlap / top_n * 100
                overlap_matrix[i, j] = overlap_pct
                overlap_matrix[j, i] = overlap_pct
                
                if i < j:
                    print(f"{method1} vs {method2}: {overlap}/{top_n} ({overlap_pct:.1f}%) overlap")
    
    # Visualize overlap matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix, 
                xticklabels=methods, 
                yticklabels=methods,
                annot=True, 
                fmt='.1f',
                cmap='YlOrRd',
                vmin=0, vmax=100)
    plt.title(f'Outlier Detection Method Overlap (Top {top_n})')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'outlier_method_overlap.png', dpi=300)
    plt.close()
    
    return overlap_matrix, top_outliers

def analyze_correlation(scores):
    """Analyze correlation between outlier scores"""
    print("\n" + "="*60)
    print("SCORE CORRELATION ANALYSIS")
    print("="*60)
    
    # Create DataFrame
    scores_df = pd.DataFrame(scores)
    
    # Compute correlations
    corr_matrix = scores_df.corr(method='spearman')
    
    print("\nSpearman correlation between outlier scores:")
    print(corr_matrix.round(3))
    
    # Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1)
    plt.title('Correlation between Outlier Detection Methods')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'outlier_score_correlation.png', dpi=300)
    plt.close()
    
    return corr_matrix

def analyze_outlier_characteristics(X_train, y_train, train_df, top_outliers):
    """Analyze characteristics of detected outliers"""
    print("\n" + "="*60)
    print("OUTLIER CHARACTERISTICS")
    print("="*60)
    
    # Get consensus outliers (appear in multiple methods)
    all_outliers = set()
    consensus_outliers = []
    
    for indices in top_outliers.values():
        all_outliers.update(indices)
    
    for idx in all_outliers:
        count = sum(1 for indices in top_outliers.values() if idx in indices)
        if count >= 3:  # Appears in at least 3 methods
            consensus_outliers.append(idx)
    
    print(f"\nConsensus outliers (in 3+ methods): {len(consensus_outliers)}")
    
    if len(consensus_outliers) > 0:
        # Analyze their characteristics
        consensus_mask = np.zeros(len(X_train), dtype=bool)
        consensus_mask[consensus_outliers] = True
        
        # Check accuracy on consensus outliers
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train[~consensus_mask], y_train[~consensus_mask])
        
        acc_consensus = rf.score(X_train[consensus_mask], y_train[consensus_mask])
        acc_normal = rf.score(X_train[~consensus_mask], y_train[~consensus_mask])
        
        print(f"\nModel accuracy:")
        print(f"  On consensus outliers: {acc_consensus:.3f}")
        print(f"  On normal samples: {acc_normal:.3f}")
        print(f"  Difference: {acc_normal - acc_consensus:.3f}")
        
        # Check personality distribution
        outlier_personalities = train_df.iloc[consensus_outliers]['Personality'].value_counts()
        normal_personalities = train_df.iloc[~consensus_mask]['Personality'].value_counts(normalize=True)
        
        print(f"\nPersonality distribution in consensus outliers:")
        print(outlier_personalities)

def main():
    # Load data
    X_train, y_train, train_df = prepare_data()
    
    # Get outlier scores
    scores = get_outlier_scores(X_train, y_train)
    
    # Analyze overlap
    overlap_matrix, top_outliers = analyze_overlap(scores, top_n=100)
    
    # Analyze correlation
    corr_matrix = analyze_correlation(scores)
    
    # Analyze characteristics
    analyze_outlier_characteristics(X_train, y_train, train_df, top_outliers)
    
    # Summary
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    
    avg_overlap = np.mean(overlap_matrix[np.triu_indices_from(overlap_matrix, k=1)])
    avg_corr = np.mean(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)])
    
    print(f"\n1. Average overlap between methods: {avg_overlap:.1f}%")
    print(f"2. Average correlation between scores: {avg_corr:.3f}")
    
    if avg_overlap < 30:
        print("\n⚠️ Low overlap suggests methods detect different types of outliers")
        print("   This explains why removing outliers has little consistent effect")
    
    if avg_corr < 0.5:
        print("\n⚠️ Low correlation confirms methods measure different properties")
        print("   'Outlier' is not a well-defined concept for this dataset")

if __name__ == "__main__":
    main()