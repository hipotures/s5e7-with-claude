#!/usr/bin/env python3
"""
Deep analysis of high confidence errors
Find patterns in misclassified samples with high confidence
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ydf
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def identify_high_confidence_errors(train_df):
    """Identify high confidence misclassifications using cross-validation"""
    
    print("="*60)
    print("HIGH CONFIDENCE ERRORS ANALYSIS")
    print("="*60)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Use 5-fold CV to get predictions for all samples
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_predictions = []
    all_probabilities = []
    all_indices = []
    
    print("\nRunning cross-validation to identify errors...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df['Personality'])):
        print(f"  Fold {fold_idx + 1}/5", end=" ")
        
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        
        # Train model
        learner = ydf.RandomForestLearner(
            label='Personality',
            num_trees=300,
            random_seed=42
        )
        
        model = learner.train(fold_train[feature_cols + ['Personality']])
        
        # Predict
        predictions = model.predict(fold_val[feature_cols])
        
        for idx, pred in zip(val_idx, predictions):
            prob_I = float(str(pred))
            pred_class = 'Introvert' if prob_I > 0.5 else 'Extrovert'
            confidence = abs(prob_I - 0.5) * 2
            
            all_predictions.append(pred_class)
            all_probabilities.append(prob_I)
            all_indices.append(idx)
        
        print("✓")
    
    # Create results dataframe
    cv_results = pd.DataFrame({
        'index': all_indices,
        'actual': train_df.iloc[all_indices]['Personality'].values,
        'predicted': all_predictions,
        'probability': all_probabilities,
        'confidence': [abs(p - 0.5) * 2 for p in all_probabilities]
    }).sort_values('index').reset_index(drop=True)
    
    # Add original features
    cv_results = cv_results.merge(
        train_df.reset_index()[['index'] + feature_cols + ['id']], 
        on='index'
    )
    
    # Find high confidence errors
    cv_results['is_error'] = cv_results['actual'] != cv_results['predicted']
    high_conf_errors = cv_results[(cv_results['is_error']) & (cv_results['confidence'] > 0.8)]
    
    print(f"\nFound {len(high_conf_errors)} high confidence errors (conf > 0.8)")
    print(f"Error rate in high confidence: {len(high_conf_errors) / cv_results[cv_results['confidence'] > 0.8].shape[0]:.3%}")
    
    return cv_results, high_conf_errors

def analyze_error_patterns(high_conf_errors, train_df):
    """Analyze patterns in high confidence errors"""
    
    print("\n" + "="*60)
    print("ERROR PATTERN ANALYSIS")
    print("="*60)
    
    # 1. Direction of errors
    error_directions = high_conf_errors.groupby(['actual', 'predicted']).size()
    print("\n1. Error directions:")
    for (actual, predicted), count in error_directions.items():
        print(f"   {actual} → {predicted}: {count} errors")
    
    # 2. Feature statistics
    print("\n2. Feature comparison (Errors vs Correct):")
    
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # Get correct high confidence predictions
    all_high_conf = train_df[train_df.index.isin(
        high_conf_errors[high_conf_errors['confidence'] > 0.8].index
    )]
    
    for feature in numeric_features:
        error_mean = high_conf_errors[feature].mean()
        all_mean = train_df[feature].mean()
        
        if not pd.isna(error_mean):
            diff = (error_mean - all_mean) / all_mean * 100
            print(f"   {feature}: {error_mean:.2f} vs {all_mean:.2f} ({diff:+.1f}%)")
    
    # 3. Clustering analysis
    print("\n3. Clustering high confidence errors...")
    
    # Prepare data for clustering
    cluster_features = high_conf_errors[numeric_features].fillna(
        high_conf_errors[numeric_features].median()
    )
    
    if len(cluster_features) > 10:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cluster_features)
        
        # Find optimal clusters
        n_clusters = min(5, len(high_conf_errors) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        high_conf_errors['cluster'] = clusters
        
        print(f"   Found {n_clusters} error clusters")
        
        # Analyze each cluster
        for cluster_id in range(n_clusters):
            cluster_errors = high_conf_errors[high_conf_errors['cluster'] == cluster_id]
            print(f"\n   Cluster {cluster_id} ({len(cluster_errors)} errors):")
            
            # Dominant error type
            error_types = cluster_errors.groupby(['actual', 'predicted']).size()
            for (actual, predicted), count in error_types.items():
                print(f"     {actual} → {predicted}: {count}")
            
            # Key features
            print("     Key features:")
            for feature in numeric_features[:3]:
                mean_val = cluster_errors[feature].mean()
                if not pd.isna(mean_val):
                    print(f"       {feature}: {mean_val:.2f}")
    
    # 4. Common characteristics
    print("\n4. Common characteristics of errors:")
    
    # Binary features analysis
    stage_fear_errors = high_conf_errors['Stage_fear'].value_counts()
    drained_errors = high_conf_errors['Drained_after_socializing'].value_counts()
    
    print(f"   Stage fear: {stage_fear_errors.to_dict()}")
    print(f"   Drained: {drained_errors.to_dict()}")
    
    # Extreme values
    print("\n5. Extreme value analysis:")
    for feature in numeric_features:
        q1 = train_df[feature].quantile(0.25)
        q3 = train_df[feature].quantile(0.75)
        
        extreme_low = high_conf_errors[high_conf_errors[feature] < q1]
        extreme_high = high_conf_errors[high_conf_errors[feature] > q3]
        
        if len(extreme_low) > 0 or len(extreme_high) > 0:
            print(f"   {feature}: {len(extreme_low)} below Q1, {len(extreme_high)} above Q3")
    
    return high_conf_errors

def create_error_visualizations(high_conf_errors, cv_results):
    """Create visualizations for error analysis"""
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Confidence distribution for errors vs correct
    ax1 = axes[0, 0]
    errors = cv_results[cv_results['is_error']]
    correct = cv_results[~cv_results['is_error']]
    
    ax1.hist(errors['confidence'], bins=30, alpha=0.5, label='Errors', color='red', density=True)
    ax1.hist(correct['confidence'], bins=30, alpha=0.5, label='Correct', color='green', density=True)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Density')
    ax1.set_title('Confidence Distribution: Errors vs Correct')
    ax1.legend()
    ax1.axvline(0.8, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Error rate by confidence bins
    ax2 = axes[0, 1]
    conf_bins = np.linspace(0, 1, 21)
    error_rates = []
    bin_centers = []
    
    for i in range(len(conf_bins)-1):
        mask = (cv_results['confidence'] >= conf_bins[i]) & (cv_results['confidence'] < conf_bins[i+1])
        if mask.sum() > 10:  # Minimum samples
            error_rate = cv_results[mask]['is_error'].mean()
            error_rates.append(error_rate)
            bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
    
    ax2.plot(bin_centers, error_rates, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Error Rate')
    ax2.set_title('Error Rate vs Confidence')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature importance for errors
    ax3 = axes[1, 0]
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # Calculate feature differences for errors
    feature_diffs = []
    for feature in numeric_features:
        error_mean = high_conf_errors[feature].mean()
        all_mean = cv_results[feature].mean()
        if not pd.isna(error_mean) and not pd.isna(all_mean):
            diff = abs(error_mean - all_mean) / all_mean * 100
            feature_diffs.append(diff)
        else:
            feature_diffs.append(0)
    
    ax3.barh(numeric_features, feature_diffs)
    ax3.set_xlabel('% Difference from Average')
    ax3.set_title('Feature Differences in High Conf Errors')
    
    # Plot 4: Error types
    ax4 = axes[1, 1]
    error_types = high_conf_errors.groupby(['actual', 'predicted']).size()
    error_labels = [f"{a}→{p}" for (a, p) in error_types.index]
    
    ax4.pie(error_types.values, labels=error_labels, autopct='%1.0f%%')
    ax4.set_title('High Confidence Error Types')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'high_confidence_errors_analysis.png', dpi=300)
    plt.close()
    
    print("   Saved: high_confidence_errors_analysis.png")

def create_error_aware_submission(train_df, test_df, high_conf_errors):
    """Create submission that's aware of high confidence error patterns"""
    
    print("\n" + "="*60)
    print("CREATING ERROR-AWARE SUBMISSION")
    print("="*60)
    
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Identify "dangerous" patterns from errors
    print("\nIdentifying dangerous patterns...")
    
    # Pattern 1: High confidence Introverts that are actually Extroverts
    I_to_E_errors = high_conf_errors[
        (high_conf_errors['actual'] == 'Extrovert') & 
        (high_conf_errors['predicted'] == 'Introvert')
    ]
    
    # Pattern 2: High confidence Extroverts that are actually Introverts  
    E_to_I_errors = high_conf_errors[
        (high_conf_errors['actual'] == 'Introvert') & 
        (high_conf_errors['predicted'] == 'Extrovert')
    ]
    
    print(f"   I→E errors: {len(I_to_E_errors)}")
    print(f"   E→I errors: {len(E_to_I_errors)}")
    
    # Train model with error awareness
    # Create sample weights - downweight samples similar to errors
    weights = np.ones(len(train_df))
    
    # Reduce weight for samples similar to high conf errors
    for idx, error in high_conf_errors.iterrows():
        # Find similar samples in training data
        similar_mask = (
            (train_df['Personality'] == error['actual']) &
            (abs(train_df['Time_spent_Alone'] - error['Time_spent_Alone']) < 1) &
            (abs(train_df['Social_event_attendance'] - error['Social_event_attendance']) < 1)
        )
        weights[similar_mask] *= 0.8
    
    # Train with modified approach
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=500,
        random_seed=42,
        winner_take_all=False
    )
    
    model = learner.train(train_df[feature_cols + ['Personality']])
    
    # Predict with adjustments
    predictions = model.predict(test_df[feature_cols])
    
    pred_classes = []
    adjusted_count = 0
    
    for i, pred in enumerate(predictions):
        prob_I = float(str(pred))
        base_pred = 'Introvert' if prob_I > 0.5 else 'Extrovert'
        confidence = abs(prob_I - 0.5) * 2
        
        # Check if this looks like a potential error pattern
        row = test_df.iloc[i]
        
        # Adjust for patterns similar to I→E errors
        if base_pred == 'Introvert' and confidence > 0.8:
            if len(I_to_E_errors) > 0:
                # Check similarity to I→E error patterns
                similar_to_error = False
                for _, error in I_to_E_errors.head(5).iterrows():
                    if (abs(row['Time_spent_Alone'] - error['Time_spent_Alone']) < 0.5 and
                        row['Stage_fear'] == error['Stage_fear']):
                        similar_to_error = True
                        break
                
                if similar_to_error and prob_I < 0.95:  # Not extremely confident
                    pred_classes.append('Extrovert')
                    adjusted_count += 1
                else:
                    pred_classes.append(base_pred)
            else:
                pred_classes.append(base_pred)
        else:
            pred_classes.append(base_pred)
    
    print(f"\nAdjusted {adjusted_count} predictions based on error patterns")
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': pred_classes
    })
    
    submission.to_csv(SCORES_DIR / 'submission_error_aware.csv', index=False)
    print("Created: submission_error_aware.csv")
    
    return submission

def main():
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Identify high confidence errors
    cv_results, high_conf_errors = identify_high_confidence_errors(train_df)
    
    # Save detailed error analysis
    high_conf_errors.to_csv(OUTPUT_DIR / 'high_confidence_errors_detailed.csv', index=False)
    print(f"\nSaved detailed error analysis to high_confidence_errors_detailed.csv")
    
    # Analyze error patterns
    high_conf_errors = analyze_error_patterns(high_conf_errors, train_df)
    
    # Create visualizations
    create_error_visualizations(high_conf_errors, cv_results)
    
    # Create error-aware submission
    submission = create_error_aware_submission(train_df, test_df, high_conf_errors)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total high confidence errors: {len(high_conf_errors)}")
    print(f"Error rate in high confidence (>0.8): {len(high_conf_errors) / len(cv_results[cv_results['confidence'] > 0.8]):.3%}")
    print(f"Most common error: {high_conf_errors.groupby(['actual', 'predicted']).size().idxmax()}")
    
    # Find specific IDs that are consistently misclassified
    error_ids = high_conf_errors.groupby('id').size().sort_values(ascending=False)
    if len(error_ids) > 0:
        print(f"\nMost problematic IDs:")
        for id_val, count in error_ids.head(5).items():
            print(f"  ID {id_val}: misclassified in {count} folds")

if __name__ == "__main__":
    main()