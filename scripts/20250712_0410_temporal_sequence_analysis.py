#!/usr/bin/env python3
"""
Temporal and sequential analysis of personality data
Looking for patterns in ID sequences and data ordering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ydf

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def analyze_temporal_patterns(train_df, test_df):
    """Analyze temporal patterns in the data"""
    
    print("="*60)
    print("TEMPORAL PATTERN ANALYSIS")
    print("="*60)
    
    # Sort by ID to analyze sequence
    train_sorted = train_df.sort_values('id').reset_index(drop=True)
    test_sorted = test_df.sort_values('id').reset_index(drop=True)
    
    # Add position index
    train_sorted['position'] = range(len(train_sorted))
    test_sorted['position'] = range(len(test_sorted))
    
    # Analyze personality distribution over ID sequence
    print("\n1. PERSONALITY DISTRIBUTION OVER ID SEQUENCE")
    print("-" * 40)
    
    # Create bins for analysis
    n_bins = 20
    train_sorted['id_bin'] = pd.cut(train_sorted['position'], bins=n_bins, labels=range(n_bins))
    
    # Calculate personality ratio per bin
    personality_by_bin = train_sorted.groupby('id_bin')['Personality'].value_counts(normalize=True).unstack()
    
    print("\nExtrovert ratio by position bin:")
    for bin_id, ratio in personality_by_bin['Extrovert'].items():
        print(f"  Bin {bin_id}: {ratio:.3f}")
    
    # Statistical test for trend
    x = np.arange(n_bins)
    y = personality_by_bin['Extrovert'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    print(f"\nLinear trend analysis:")
    print(f"  Slope: {slope:.6f}")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Personality ratio over bins
    ax1.plot(personality_by_bin.index, personality_by_bin['Extrovert'], 
             'o-', label='Extrovert ratio', linewidth=2)
    ax1.axhline(y=train_df['Personality'].value_counts(normalize=True)['Extrovert'], 
                color='red', linestyle='--', label='Overall average')
    ax1.set_xlabel('Position Bin')
    ax1.set_ylabel('Extrovert Ratio')
    ax1.set_title('Personality Distribution Over ID Sequence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rolling average
    window_size = 1000
    train_sorted['is_extrovert'] = (train_sorted['Personality'] == 'Extrovert').astype(int)
    rolling_avg = train_sorted['is_extrovert'].rolling(window=window_size, center=True).mean()
    
    ax2.plot(train_sorted['position'], rolling_avg, label=f'Rolling avg (window={window_size})')
    ax2.axhline(y=train_df['Personality'].value_counts(normalize=True)['Extrovert'], 
                color='red', linestyle='--', label='Overall average')
    ax2.set_xlabel('Position in Dataset')
    ax2.set_ylabel('Extrovert Ratio')
    ax2.set_title('Rolling Average of Personality Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_personality_distribution.png', dpi=300)
    plt.close()
    
    return train_sorted, test_sorted

def analyze_feature_sequences(train_sorted):
    """Analyze how features change over the sequence"""
    
    print("\n2. FEATURE SEQUENCE ANALYSIS")
    print("-" * 40)
    
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    # Calculate rolling statistics
    window = 1000
    feature_trends = {}
    
    for feature in numeric_features:
        rolling_mean = train_sorted[feature].rolling(window=window, center=True).mean()
        rolling_std = train_sorted[feature].rolling(window=window, center=True).std()
        
        # Trend analysis
        positions = train_sorted['position'].values
        values = train_sorted[feature].fillna(train_sorted[feature].mean()).values
        
        slope, _, r_value, p_value, _ = stats.linregress(positions, values)
        
        feature_trends[feature] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std
        }
        
        print(f"\n{feature}:")
        print(f"  Trend slope: {slope:.6f}")
        print(f"  R-squared: {r_value**2:.4f}")
        print(f"  P-value: {p_value:.4f}")
    
    # Visualize feature trends
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (feature, trend_data) in enumerate(feature_trends.items()):
        if idx < len(axes):
            ax = axes[idx]
            
            # Plot rolling mean
            ax.plot(train_sorted['position'], trend_data['rolling_mean'], 
                   label=f'Rolling mean (window={window})', alpha=0.8)
            
            # Add trend line
            x = train_sorted['position'].values
            y_trend = trend_data['slope'] * x + train_sorted[feature].mean()
            ax.plot(x, y_trend, 'r--', label='Trend line', alpha=0.7)
            
            ax.set_xlabel('Position')
            ax.set_ylabel(feature)
            ax.set_title(f'{feature} Over Sequence\n(slope={trend_data["slope"]:.6f}, p={trend_data["p_value"]:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(feature_trends) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_sequence_trends.png', dpi=300)
    plt.close()
    
    return feature_trends

def analyze_id_patterns(train_df, test_df):
    """Analyze patterns in ID numbers themselves"""
    
    print("\n3. ID NUMBER PATTERN ANALYSIS")
    print("-" * 40)
    
    # Extract ID components
    train_df['id_last_digit'] = train_df['id'] % 10
    train_df['id_last_two'] = train_df['id'] % 100
    train_df['id_mod_50'] = train_df['id'] % 50
    
    test_df['id_last_digit'] = test_df['id'] % 10
    test_df['id_last_two'] = test_df['id'] % 100
    test_df['id_mod_50'] = test_df['id'] % 50
    
    # Analyze personality distribution by ID patterns
    print("\nPersonality distribution by last digit:")
    last_digit_dist = train_df.groupby(['id_last_digit', 'Personality']).size().unstack(fill_value=0)
    last_digit_ratio = last_digit_dist.div(last_digit_dist.sum(axis=1), axis=0)
    
    for digit in range(10):
        if digit in last_digit_ratio.index:
            e_ratio = last_digit_ratio.loc[digit, 'Extrovert']
            print(f"  Digit {digit}: {e_ratio:.3f} Extrovert")
    
    # Chi-square test
    chi2, p_value = stats.chi2_contingency(last_digit_dist)[:2]
    print(f"\nChi-square test for last digit:")
    print(f"  Chi2: {chi2:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    # Analyze patterns ending in specific numbers (like 34)
    print("\nAnalyzing IDs ending in 34:")
    ids_ending_34 = train_df[train_df['id_last_two'] == 34]
    print(f"  Count: {len(ids_ending_34)}")
    print(f"  Extrovert ratio: {(ids_ending_34['Personality'] == 'Extrovert').mean():.3f}")
    print(f"  Overall ratio: {(train_df['Personality'] == 'Extrovert').mean():.3f}")
    
    return train_df, test_df

def create_sequential_features(train_df, test_df):
    """Create features based on sequential patterns"""
    
    print("\n4. CREATING SEQUENTIAL FEATURES")
    print("-" * 40)
    
    # Sort by ID
    train_sorted = train_df.sort_values('id').reset_index(drop=True)
    test_sorted = test_df.sort_values('id').reset_index(drop=True)
    
    # Create lag and lead features
    lag_features = []
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    for feature in numeric_features:
        # Lag features
        train_sorted[f'{feature}_lag1'] = train_sorted[feature].shift(1)
        train_sorted[f'{feature}_lag2'] = train_sorted[feature].shift(2)
        train_sorted[f'{feature}_lead1'] = train_sorted[feature].shift(-1)
        
        test_sorted[f'{feature}_lag1'] = test_sorted[feature].shift(1)
        test_sorted[f'{feature}_lag2'] = test_sorted[feature].shift(2)
        test_sorted[f'{feature}_lead1'] = test_sorted[feature].shift(-1)
        
        # Rolling features
        train_sorted[f'{feature}_roll3'] = train_sorted[feature].rolling(3, center=True).mean()
        test_sorted[f'{feature}_roll3'] = test_sorted[feature].rolling(3, center=True).mean()
        
        lag_features.extend([f'{feature}_lag1', f'{feature}_lag2', 
                           f'{feature}_lead1', f'{feature}_roll3'])
    
    # Position-based features
    train_sorted['position_pct'] = train_sorted.index / len(train_sorted)
    test_sorted['position_pct'] = test_sorted.index / len(test_sorted)
    
    train_sorted['id_gap'] = train_sorted['id'].diff()
    test_sorted['id_gap'] = test_sorted['id'].diff()
    
    print(f"Created {len(lag_features)} lag/lead features")
    print(f"Created position-based features")
    
    return train_sorted, test_sorted, lag_features

def train_sequential_model(train_seq, test_seq, lag_features):
    """Train model with sequential features"""
    
    print("\n5. TRAINING SEQUENTIAL MODEL")
    print("-" * 40)
    
    # Prepare features
    base_features = ['Time_spent_Alone', 'Social_event_attendance', 
                    'Friends_circle_size', 'Going_outside', 'Post_frequency',
                    'Stage_fear', 'Drained_after_socializing']
    
    seq_features = ['position_pct', 'id_gap', 'id_last_digit', 'id_mod_50']
    all_features = base_features + lag_features + seq_features
    
    # Remove rows with NaN (due to lag/lead)
    train_clean = train_seq.dropna(subset=all_features + ['Personality'])
    test_clean = test_seq.copy()
    
    # Fill NaN in test with appropriate values
    for col in all_features:
        if col in test_clean.columns:
            if test_clean[col].dtype in ['float64', 'int64']:
                test_clean[col] = test_clean[col].fillna(test_clean[col].median())
            else:
                # For categorical columns, use mode
                mode_val = test_clean[col].mode()
                if len(mode_val) > 0:
                    test_clean[col] = test_clean[col].fillna(mode_val[0])
                else:
                    test_clean[col] = test_clean[col].fillna('Unknown')
    
    print(f"Train size after cleaning: {len(train_clean)}")
    
    # Train baseline model
    print("\nTraining baseline model...")
    baseline_learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        compute_oob_performances=True,
        random_seed=42
    )
    baseline_model = baseline_learner.train(train_clean[base_features + ['Personality']])
    
    # Train sequential model
    print("Training sequential model...")
    seq_learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        compute_oob_performances=True,
        random_seed=42
    )
    seq_model = seq_learner.train(train_clean[all_features + ['Personality']])
    
    # Compare OOB performances
    print(f"\nModel comparison:")
    try:
        if hasattr(baseline_model, 'out_of_bag_evaluation'):
            baseline_oob = baseline_model.out_of_bag_evaluation()
            seq_oob = seq_model.out_of_bag_evaluation()
            print(f"  Baseline OOB accuracy: {baseline_oob.accuracy:.6f}")
            print(f"  Sequential OOB accuracy: {seq_oob.accuracy:.6f}")
            print(f"  Improvement: {seq_oob.accuracy - baseline_oob.accuracy:+.6f}")
        else:
            print("  OOB evaluation not available, skipping comparison")
    except Exception as e:
        print(f"  Error getting OOB: {e}")
    
    # Feature importance
    try:
        seq_importance = seq_model.variable_importances()
        print("\nTop 10 most important features:")
        
        # Variable importances returns a dict-like object
        importance_items = []
        for feature_name, importance_info in seq_importance.items():
            if hasattr(importance_info, 'importance'):
                importance_items.append((feature_name, importance_info.importance))
            else:
                # Handle different formats
                importance_items.append((feature_name, importance_info))
        
        # Sort by importance
        importance_items.sort(key=lambda x: x[1], reverse=True)
        
        for name, imp in importance_items[:10]:
            print(f"  {name}: {imp:.4f}")
    except Exception as e:
        print(f"  Error getting feature importance: {e}")
    
    # Make predictions
    predictions = seq_model.predict(test_clean[all_features])
    pred_classes = []
    
    for pred in predictions:
        prob_I = float(str(pred))
        pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_seq['id'],
        'Personality': pred_classes
    })
    
    submission.to_csv(SCORES_DIR / 'submission_sequential_model.csv', index=False)
    print("\nCreated: submission_sequential_model.csv")
    
    return seq_model, submission

def analyze_batch_effects(train_df):
    """Analyze if data was collected in batches"""
    
    print("\n6. BATCH EFFECT ANALYSIS")
    print("-" * 40)
    
    # Sort by ID
    train_sorted = train_df.sort_values('id').reset_index(drop=True)
    
    # Look for sudden changes in feature distributions
    window = 500
    features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']
    
    batch_boundaries = []
    
    for feature in features:
        # Calculate rolling statistics
        rolling_mean = train_sorted[feature].rolling(window=window).mean()
        rolling_std = train_sorted[feature].rolling(window=window).std()
        
        # Detect sudden changes
        mean_diff = rolling_mean.diff().abs()
        std_diff = rolling_std.diff().abs()
        
        # Find potential batch boundaries (large changes)
        threshold = mean_diff.quantile(0.99)
        boundaries = train_sorted.index[mean_diff > threshold].tolist()
        
        if boundaries:
            print(f"\n{feature} - potential batch boundaries at positions:")
            for b in boundaries[:5]:  # Show first 5
                print(f"  Position {b} (ID: {train_sorted.iloc[b]['id']})")
            
            batch_boundaries.extend(boundaries)
    
    # Find consensus boundaries
    from collections import Counter
    boundary_counts = Counter(batch_boundaries)
    consensus_boundaries = [b for b, count in boundary_counts.items() if count >= 2]
    
    if consensus_boundaries:
        print(f"\nConsensus batch boundaries found at {len(consensus_boundaries)} positions")
        
        # Analyze personality distribution in each batch
        consensus_boundaries = sorted(consensus_boundaries)
        boundaries_with_edges = [0] + consensus_boundaries + [len(train_sorted)]
        
        print("\nPersonality distribution by batch:")
        for i in range(len(boundaries_with_edges) - 1):
            start = boundaries_with_edges[i]
            end = boundaries_with_edges[i + 1]
            batch_data = train_sorted.iloc[start:end]
            e_ratio = (batch_data['Personality'] == 'Extrovert').mean()
            print(f"  Batch {i+1} (pos {start}-{end}): {e_ratio:.3f} Extrovert")
    else:
        print("\nNo clear batch boundaries detected")
    
    return batch_boundaries

def main():
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # 1. Temporal pattern analysis
    train_sorted, test_sorted = analyze_temporal_patterns(train_df, test_df)
    
    # 2. Feature sequence analysis
    feature_trends = analyze_feature_sequences(train_sorted)
    
    # 3. ID pattern analysis
    train_df, test_df = analyze_id_patterns(train_df, test_df)
    
    # 4. Create sequential features
    train_seq, test_seq, lag_features = create_sequential_features(train_df, test_df)
    
    # 5. Train sequential model
    seq_model, submission = train_sequential_model(train_seq, test_seq, lag_features)
    
    # 6. Batch effect analysis
    batch_boundaries = analyze_batch_effects(train_df)
    
    print("\n" + "="*60)
    print("SEQUENTIAL ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()