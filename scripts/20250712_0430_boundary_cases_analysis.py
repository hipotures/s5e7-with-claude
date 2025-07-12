#!/usr/bin/env python3
"""
Boundary cases analysis - find and analyze edge cases
Focus on samples near decision boundary and ambiguous cases
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import ydf
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def load_and_prepare_data():
    """Load and prepare data for analysis"""
    print("="*60)
    print("BOUNDARY CASES ANALYSIS")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"\nData shapes:")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    
    return train_df, test_df

def find_boundary_cases(train_df):
    """Find cases near decision boundary using model confidence"""
    
    print("\n1. FINDING BOUNDARY CASES")
    print("-" * 40)
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Train model
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        compute_oob_performances=True,
        random_seed=42
    )
    
    model = learner.train(train_df[feature_cols + ['Personality']])
    
    # Get OOB predictions
    oob_predictions = []
    oob_probabilities = []
    
    # Use cross-validation to get predictions for all samples
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_probabilities = np.zeros(len(train_df))
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df['Personality'])):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        
        # Train on fold
        fold_model = learner.train(fold_train[feature_cols + ['Personality']])
        
        # Predict on validation
        fold_predictions = fold_model.predict(fold_val[feature_cols])
        
        # Extract probabilities
        for i, (idx, pred) in enumerate(zip(val_idx, fold_predictions)):
            prob_I = float(str(pred))
            all_probabilities[idx] = prob_I
    
    # Calculate confidence (distance from 0.5)
    train_df['probability'] = all_probabilities
    train_df['confidence'] = np.abs(all_probabilities - 0.5) * 2
    train_df['predicted_class'] = (all_probabilities <= 0.5).astype(int)  # 0 for E, 1 for I
    train_df['actual_class'] = (train_df['Personality'] == 'Introvert').astype(int)
    train_df['is_correct'] = train_df['predicted_class'] == train_df['actual_class']
    
    # Find boundary cases (low confidence)
    boundary_threshold = 0.1  # Confidence < 0.1 means probability between 0.45 and 0.55
    boundary_cases = train_df[train_df['confidence'] < boundary_threshold].copy()
    
    print(f"\nBoundary cases (confidence < {boundary_threshold}): {len(boundary_cases)}")
    print(f"Percentage of total: {len(boundary_cases)/len(train_df)*100:.2f}%")
    
    # Analyze boundary cases
    print("\nBoundary cases analysis:")
    print(f"  Accuracy on boundary: {boundary_cases['is_correct'].mean():.3f}")
    print(f"  Overall accuracy: {train_df['is_correct'].mean():.3f}")
    
    # Distribution of actual classes in boundary cases
    boundary_dist = boundary_cases['Personality'].value_counts(normalize=True)
    print(f"\nActual class distribution in boundary cases:")
    print(f"  Extrovert: {boundary_dist.get('Extrovert', 0):.3f}")
    print(f"  Introvert: {boundary_dist.get('Introvert', 0):.3f}")
    
    return train_df, boundary_cases

def analyze_misclassified_cases(train_df):
    """Analyze misclassified cases"""
    
    print("\n2. MISCLASSIFIED CASES ANALYSIS")
    print("-" * 40)
    
    # Find misclassified cases
    misclassified = train_df[~train_df['is_correct']].copy()
    
    print(f"\nMisclassified cases: {len(misclassified)}")
    print(f"Misclassification rate: {len(misclassified)/len(train_df)*100:.2f}%")
    
    # Analyze confidence of misclassified
    print(f"\nConfidence distribution of misclassified:")
    print(f"  Mean confidence: {misclassified['confidence'].mean():.3f}")
    print(f"  Median confidence: {misclassified['confidence'].median():.3f}")
    print(f"  Min confidence: {misclassified['confidence'].min():.3f}")
    print(f"  Max confidence: {misclassified['confidence'].max():.3f}")
    
    # High confidence misclassifications
    high_conf_threshold = 0.8
    high_conf_misclass = misclassified[misclassified['confidence'] > high_conf_threshold]
    
    print(f"\nHigh confidence misclassifications (conf > {high_conf_threshold}): {len(high_conf_misclass)}")
    print(f"  Percentage of misclassified: {len(high_conf_misclass)/len(misclassified)*100:.2f}%")
    
    return misclassified, high_conf_misclass

def visualize_boundary_cases(train_df, boundary_cases, misclassified):
    """Visualize boundary cases using dimensionality reduction"""
    
    print("\n3. VISUALIZING BOUNDARY CASES")
    print("-" * 40)
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # Handle missing values
    X = train_df[feature_cols].fillna(train_df[feature_cols].median())
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All samples with boundary cases highlighted
    colors = ['red' if p == 'Extrovert' else 'blue' for p in train_df['Personality']]
    alphas = [0.1 if idx not in boundary_cases.index else 0.8 for idx in range(len(train_df))]
    sizes = [10 if idx not in boundary_cases.index else 50 for idx in range(len(train_df))]
    
    for i in range(len(train_df)):
        ax1.scatter(X_pca[i, 0], X_pca[i, 1], c=colors[i], alpha=alphas[i], s=sizes[i])
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax1.set_title('PCA: Boundary Cases Highlighted')
    ax1.legend(['Extrovert', 'Introvert', 'Boundary'], loc='best')
    
    # Plot 2: Confidence heatmap
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=train_df['confidence'], 
               cmap='RdYlGn', alpha=0.6, s=20)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Confidence')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax2.set_title('PCA: Model Confidence Heatmap')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boundary_cases_pca.png', dpi=300)
    plt.close()
    
    # t-SNE visualization
    print("\nComputing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled[:5000])  # Limit to 5000 for speed
    
    train_subset = train_df.iloc[:5000]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot different categories
    categories = []
    
    # Correct predictions
    correct_E = train_subset[(train_subset['Personality'] == 'Extrovert') & train_subset['is_correct']]
    correct_I = train_subset[(train_subset['Personality'] == 'Introvert') & train_subset['is_correct']]
    
    # Misclassified
    misclass_E = train_subset[(train_subset['Personality'] == 'Extrovert') & ~train_subset['is_correct']]
    misclass_I = train_subset[(train_subset['Personality'] == 'Introvert') & ~train_subset['is_correct']]
    
    # Boundary
    boundary_subset = train_subset[train_subset['confidence'] < 0.1]
    
    # Plot
    ax.scatter(X_tsne[correct_E.index, 0], X_tsne[correct_E.index, 1], 
              c='lightcoral', alpha=0.3, s=20, label='Correct E')
    ax.scatter(X_tsne[correct_I.index, 0], X_tsne[correct_I.index, 1], 
              c='lightblue', alpha=0.3, s=20, label='Correct I')
    ax.scatter(X_tsne[misclass_E.index, 0], X_tsne[misclass_E.index, 1], 
              c='darkred', alpha=0.8, s=50, label='Misclass E', marker='x')
    ax.scatter(X_tsne[misclass_I.index, 0], X_tsne[misclass_I.index, 1], 
              c='darkblue', alpha=0.8, s=50, label='Misclass I', marker='x')
    ax.scatter(X_tsne[boundary_subset.index, 0], X_tsne[boundary_subset.index, 1], 
              c='yellow', alpha=0.8, s=80, label='Boundary', marker='o', edgecolors='black')
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE: Boundary Cases and Misclassifications')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boundary_cases_tsne.png', dpi=300)
    plt.close()

def analyze_feature_patterns(boundary_cases, train_df):
    """Analyze feature patterns in boundary cases"""
    
    print("\n4. FEATURE PATTERNS IN BOUNDARY CASES")
    print("-" * 40)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # Compare distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        
        # Plot distributions
        train_df[feature].hist(bins=30, alpha=0.5, label='All', ax=ax, density=True)
        boundary_cases[feature].hist(bins=30, alpha=0.7, label='Boundary', ax=ax, density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature} Distribution')
        ax.legend()
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boundary_cases_features.png', dpi=300)
    plt.close()
    
    # Statistical comparison
    print("\nFeature statistics comparison (Boundary vs All):")
    for feature in feature_cols:
        all_mean = train_df[feature].mean()
        boundary_mean = boundary_cases[feature].mean()
        diff_pct = (boundary_mean - all_mean) / all_mean * 100
        
        print(f"\n{feature}:")
        print(f"  All mean: {all_mean:.3f}")
        print(f"  Boundary mean: {boundary_mean:.3f}")
        print(f"  Difference: {diff_pct:+.1f}%")

def find_outliers(train_df):
    """Find outliers using multiple methods"""
    
    print("\n5. OUTLIER DETECTION")
    print("-" * 40)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # Prepare data
    X = train_df[feature_cols].fillna(train_df[feature_cols].median())
    X_scaled = StandardScaler().fit_transform(X)
    
    # Method 1: Statistical outliers (IQR method)
    outliers_iqr = set()
    for col in feature_cols:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        col_outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)].index
        outliers_iqr.update(col_outliers)
    
    print(f"IQR method outliers: {len(outliers_iqr)}")
    
    # Method 2: DBSCAN clustering
    dbscan = DBSCAN(eps=1.5, min_samples=10)
    clusters = dbscan.fit_predict(X_scaled)
    outliers_dbscan = train_df.index[clusters == -1]
    
    print(f"DBSCAN outliers: {len(outliers_dbscan)}")
    
    # Method 3: Isolation Forest
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers_if = iso_forest.fit_predict(X_scaled)
    outliers_if_idx = train_df.index[outliers_if == -1]
    
    print(f"Isolation Forest outliers: {len(outliers_if_idx)}")
    
    # Consensus outliers
    consensus_outliers = set(outliers_iqr) & set(outliers_dbscan) & set(outliers_if_idx)
    print(f"\nConsensus outliers (detected by all methods): {len(consensus_outliers)}")
    
    # Analyze outliers
    if len(consensus_outliers) > 0:
        outlier_df = train_df.loc[list(consensus_outliers)]
        print(f"\nOutlier personality distribution:")
        print(outlier_df['Personality'].value_counts(normalize=True))
        
        print(f"\nOutlier confidence stats:")
        print(f"  Mean confidence: {outlier_df['confidence'].mean():.3f}")
        print(f"  Misclassification rate: {(~outlier_df['is_correct']).mean():.3f}")
    
    return outliers_iqr, outliers_dbscan, outliers_if_idx

def create_boundary_submission(train_df, test_df):
    """Create submission focusing on boundary cases"""
    
    print("\n6. CREATING BOUNDARY-FOCUSED SUBMISSION")
    print("-" * 40)
    
    # Train specialized model for boundary cases
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Create sample weights - give more weight to boundary cases
    weights = np.ones(len(train_df))
    boundary_mask = train_df['confidence'] < 0.2
    weights[boundary_mask] = 2.0  # Double weight for boundary cases
    
    # Train weighted model
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=500,
        max_depth=20,
        random_seed=42,
        winner_take_all=False  # Use soft voting
    )
    
    # YDF doesn't support sample weights directly, so we'll oversample boundary cases
    boundary_samples = train_df[boundary_mask]
    train_weighted = pd.concat([train_df, boundary_samples], ignore_index=True)
    
    print(f"Training on {len(train_weighted)} samples (original: {len(train_df)})")
    
    model = learner.train(train_weighted[feature_cols + ['Personality']])
    
    # Predict test
    predictions = model.predict(test_df[feature_cols])
    
    pred_classes = []
    confidences = []
    
    for pred in predictions:
        prob_I = float(str(pred))
        pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
        confidences.append(abs(prob_I - 0.5) * 2)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': pred_classes
    })
    
    submission.to_csv(SCORES_DIR / 'submission_boundary_focused.csv', index=False)
    print("Created: submission_boundary_focused.csv")
    
    # Analyze test predictions
    confidence_array = np.array(confidences)
    print(f"\nTest prediction confidence stats:")
    print(f"  Mean confidence: {confidence_array.mean():.3f}")
    print(f"  Low confidence (<0.1): {(confidence_array < 0.1).sum()} samples")
    print(f"  High confidence (>0.9): {(confidence_array > 0.9).sum()} samples")
    
    return submission, confidence_array

def main():
    # Load data
    train_df, test_df = load_and_prepare_data()
    
    # Find boundary cases
    train_df, boundary_cases = find_boundary_cases(train_df)
    
    # Analyze misclassified cases
    misclassified, high_conf_misclass = analyze_misclassified_cases(train_df)
    
    # Visualize boundary cases
    visualize_boundary_cases(train_df, boundary_cases, misclassified)
    
    # Analyze feature patterns
    analyze_feature_patterns(boundary_cases, train_df)
    
    # Find outliers
    outliers_iqr, outliers_dbscan, outliers_if = find_outliers(train_df)
    
    # Create boundary-focused submission
    submission, test_confidences = create_boundary_submission(train_df, test_df)
    
    # Save analysis results
    analysis_results = {
        'total_samples': len(train_df),
        'boundary_cases': len(boundary_cases),
        'misclassified': len(misclassified),
        'high_conf_misclass': len(high_conf_misclass),
        'outliers_iqr': len(outliers_iqr),
        'outliers_dbscan': len(outliers_dbscan),
        'outliers_if': len(outliers_if)
    }
    
    pd.DataFrame([analysis_results]).to_csv(OUTPUT_DIR / 'boundary_analysis_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("BOUNDARY ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey findings:")
    print(f"- {len(boundary_cases)} boundary cases ({len(boundary_cases)/len(train_df)*100:.1f}%)")
    print(f"- Boundary case accuracy: {boundary_cases['is_correct'].mean():.3f}")
    print(f"- {len(high_conf_misclass)} high-confidence misclassifications")
    print(f"- {len(outliers_iqr)} statistical outliers")
    print("\nVisualizations saved to output/")
    print("Submission created: submission_boundary_focused.csv")

if __name__ == "__main__":
    main()