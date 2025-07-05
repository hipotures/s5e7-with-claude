#!/usr/bin/env python3
"""
PURPOSE: Deep analysis of null patterns and their relationship to personality types
HYPOTHESIS: Null values encode hidden information about personality - the 25% difference in null rates is not random
EXPECTED: Discover specific null patterns that strongly predict personality type
RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
from datetime import datetime

print(f"Analysis started at: {datetime.now()}")

# Load data
print("Loading data...")
train_df = pd.read_csv('../../train.csv')
test_df = pd.read_csv('../../test.csv')

# Create target variable
y_train = (train_df['Personality'] == 'Extrovert').astype(int)

print("\n=== PHASE 1: NULL CORRELATION ANALYSIS ===")

# Create null indicator features
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                'Post_frequency']

null_indicators = pd.DataFrame()
for col in feature_cols:
    null_indicators[f'{col}_is_null'] = train_df[col].isna().astype(int)

# Correlation with target
print("\nCorrelation between null indicators and personality (Extrovert=1):")
null_correlations = {}
for col in null_indicators.columns:
    corr = null_indicators[col].corr(y_train)
    null_correlations[col] = corr
    print(f"{col}: {corr:.4f}")

# Sort by absolute correlation
sorted_correlations = sorted(null_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nTop null indicators by absolute correlation:")
for col, corr in sorted_correlations[:5]:
    print(f"{col}: {corr:.4f}")

print("\n=== PHASE 2: CHI-SQUARE TESTS ===")

# Chi-square test for independence
chi_square_results = {}
for col in feature_cols:
    # Create contingency table
    contingency = pd.crosstab(train_df[col].isna(), train_df['Personality'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    chi_square_results[col] = {'chi2': chi2, 'p_value': p_value}
    print(f"{col}: χ² = {chi2:.2f}, p-value = {p_value:.2e}")

print("\n=== PHASE 3: BAYESIAN ANALYSIS ===")

# Calculate P(Personality | Null in Feature)
bayesian_analysis = {}
for col in feature_cols:
    # P(Extrovert | Null)
    null_mask = train_df[col].isna()
    if null_mask.sum() > 0:
        p_extrovert_given_null = (train_df[null_mask]['Personality'] == 'Extrovert').mean()
        p_introvert_given_null = 1 - p_extrovert_given_null
        
        # P(Extrovert | Not Null)
        not_null_mask = ~null_mask
        p_extrovert_given_not_null = (train_df[not_null_mask]['Personality'] == 'Extrovert').mean()
        
        bayesian_analysis[col] = {
            'P(E|Null)': p_extrovert_given_null,
            'P(I|Null)': p_introvert_given_null,
            'P(E|NotNull)': p_extrovert_given_not_null,
            'Null_Count': null_mask.sum(),
            'Null_Rate': null_mask.mean()
        }
        
        print(f"\n{col}:")
        print(f"  P(Extrovert|Null) = {p_extrovert_given_null:.3f}")
        print(f"  P(Extrovert|NotNull) = {p_extrovert_given_not_null:.3f}")
        print(f"  Difference = {p_extrovert_given_null - p_extrovert_given_not_null:.3f}")

print("\n=== PHASE 4: MUTUAL INFORMATION ===")

# Calculate mutual information between null patterns and personality
mi_scores = mutual_info_classif(null_indicators, y_train, random_state=42)
mi_results = dict(zip(null_indicators.columns, mi_scores))

print("\nMutual Information scores:")
for col, score in sorted(mi_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{col}: {score:.4f}")

print("\n=== PHASE 5: NULL PATTERN CLUSTERING ===")

# Create null pattern matrix
null_patterns = train_df[feature_cols].isna().astype(int)

# Add null count features
null_patterns['total_nulls'] = null_patterns.sum(axis=1)
null_patterns['null_ratio'] = null_patterns['total_nulls'] / len(feature_cols)

# Cluster by null patterns
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(null_patterns)

# Analyze personality distribution in each cluster
print(f"\nClustering samples into {n_clusters} groups by null patterns:")
cluster_analysis = []

for cluster in range(n_clusters):
    cluster_mask = cluster_labels == cluster
    cluster_size = cluster_mask.sum()
    
    if cluster_size > 0:
        extrovert_ratio = (train_df[cluster_mask]['Personality'] == 'Extrovert').mean()
        avg_nulls = null_patterns[cluster_mask]['total_nulls'].mean()
        
        # Get dominant null pattern
        cluster_patterns = null_patterns[cluster_mask][feature_cols]
        most_common_pattern = cluster_patterns.mode().iloc[0] if len(cluster_patterns) > 0 else None
        
        cluster_analysis.append({
            'cluster': cluster,
            'size': cluster_size,
            'extrovert_ratio': extrovert_ratio,
            'avg_nulls': avg_nulls,
            'pattern': most_common_pattern.values.tolist() if most_common_pattern is not None else None
        })
        
        print(f"\nCluster {cluster}: {cluster_size} samples ({cluster_size/len(train_df)*100:.1f}%)")
        print(f"  Extrovert ratio: {extrovert_ratio:.3f}")
        print(f"  Average nulls: {avg_nulls:.2f}")

# Find clusters with extreme personality ratios
extreme_clusters = sorted(cluster_analysis, key=lambda x: abs(x['extrovert_ratio'] - 0.74), reverse=True)[:3]
print("\n=== MOST PREDICTIVE NULL CLUSTERS ===")
for cluster_info in extreme_clusters:
    print(f"\nCluster {cluster_info['cluster']}:")
    print(f"  Size: {cluster_info['size']} samples")
    print(f"  Extrovert ratio: {cluster_info['extrovert_ratio']:.3f} (baseline: 0.740)")
    print(f"  Pattern: {cluster_info['pattern']}")

print("\n=== PHASE 6: SPECIFIC NULL COMBINATIONS ===")

# Analyze specific meaningful combinations
special_patterns = {
    'only_drained_null': (train_df['Drained_after_socializing'].isna() & 
                          train_df[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                                   'Going_outside', 'Friends_circle_size', 'Post_frequency']].notna().all(axis=1)),
    
    'only_stage_fear_null': (train_df['Stage_fear'].isna() & 
                             train_df[['Time_spent_Alone', 'Drained_after_socializing', 'Social_event_attendance',
                                      'Going_outside', 'Friends_circle_size', 'Post_frequency']].notna().all(axis=1)),
    
    'social_nulls': (train_df[['Social_event_attendance', 'Friends_circle_size']].isna().any(axis=1)),
    
    'no_nulls': train_df[feature_cols].notna().all(axis=1),
    
    'high_nulls': (train_df[feature_cols].isna().sum(axis=1) >= 4)
}

print("\nAnalysis of special null patterns:")
pattern_results = {}
for pattern_name, mask in special_patterns.items():
    if mask.sum() > 0:
        extrovert_rate = (train_df[mask]['Personality'] == 'Extrovert').mean()
        pattern_results[pattern_name] = {
            'count': mask.sum(),
            'percentage': mask.mean() * 100,
            'extrovert_rate': extrovert_rate,
            'lift': extrovert_rate / 0.74  # 74% is baseline extrovert rate
        }
        
        print(f"\n{pattern_name}:")
        print(f"  Count: {mask.sum()} ({mask.mean()*100:.1f}%)")
        print(f"  Extrovert rate: {extrovert_rate:.3f} (lift: {extrovert_rate/0.74:.2f})")

print("\n=== PHASE 7: NULL PATTERNS IN TEST SET ===")

# Check if test set has similar null patterns
test_null_patterns = test_df[feature_cols].isna().astype(int)
test_null_summary = {
    'total_null_rate': test_df[feature_cols].isna().any(axis=1).mean(),
    'avg_nulls_per_sample': test_df[feature_cols].isna().sum(axis=1).mean()
}

print(f"\nTest set null characteristics:")
print(f"  Samples with nulls: {test_null_summary['total_null_rate']*100:.1f}%")
print(f"  Average nulls per sample: {test_null_summary['avg_nulls_per_sample']:.2f}")

# Compare null rates between train and test
print("\nNull rate comparison (train vs test):")
for col in feature_cols:
    train_null_rate = train_df[col].isna().mean()
    test_null_rate = test_df[col].isna().mean()
    print(f"{col}: Train={train_null_rate:.3f}, Test={test_null_rate:.3f}, Diff={abs(train_null_rate-test_null_rate):.3f}")

# Convert numpy types to Python types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj

# Save results
results = {
    'null_correlations': convert_to_serializable(null_correlations),
    'chi_square_results': convert_to_serializable(chi_square_results),
    'bayesian_analysis': convert_to_serializable(bayesian_analysis),
    'mutual_information': convert_to_serializable(mi_results),
    'cluster_analysis': convert_to_serializable(cluster_analysis),
    'pattern_results': convert_to_serializable(pattern_results),
    'test_null_summary': convert_to_serializable(test_null_summary)
}

with open('output/20250705_0122_null_pattern_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Create visualization of null patterns by personality
plt.figure(figsize=(12, 8))

# Subplot 1: Null rates by feature and personality
plt.subplot(2, 2, 1)
null_rates_by_personality = []
for personality in ['Introvert', 'Extrovert']:
    subset = train_df[train_df['Personality'] == personality]
    null_rates = [subset[col].isna().mean() for col in feature_cols]
    null_rates_by_personality.append(null_rates)

x = np.arange(len(feature_cols))
width = 0.35
plt.bar(x - width/2, null_rates_by_personality[0], width, label='Introvert', alpha=0.8)
plt.bar(x + width/2, null_rates_by_personality[1], width, label='Extrovert', alpha=0.8)
plt.xlabel('Features')
plt.ylabel('Null Rate')
plt.title('Null Rates by Feature and Personality Type')
plt.xticks(x, [col.replace('_', ' ') for col in feature_cols], rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Subplot 2: Correlation heatmap
plt.subplot(2, 2, 2)
corr_values = [null_correlations[f'{col}_is_null'] for col in feature_cols]
plt.bar(range(len(feature_cols)), corr_values)
plt.xlabel('Features')
plt.ylabel('Correlation with Extrovert')
plt.title('Null Indicator Correlation with Personality')
plt.xticks(range(len(feature_cols)), [col.replace('_', ' ') for col in feature_cols], rotation=45, ha='right')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()

# Subplot 3: Cluster sizes and personality distribution
plt.subplot(2, 2, 3)
cluster_sizes = [c['size'] for c in cluster_analysis]
cluster_extrovert_rates = [c['extrovert_ratio'] for c in cluster_analysis]
plt.scatter(cluster_sizes, cluster_extrovert_rates, s=100, alpha=0.6)
plt.axhline(y=0.74, color='red', linestyle='--', label='Baseline (74%)')
plt.xlabel('Cluster Size')
plt.ylabel('Extrovert Ratio')
plt.title('Null Pattern Clusters')
plt.legend()
plt.tight_layout()

# Subplot 4: Special pattern analysis
plt.subplot(2, 2, 4)
pattern_names = list(pattern_results.keys())
pattern_lifts = [pattern_results[p]['lift'] for p in pattern_names]
plt.bar(range(len(pattern_names)), pattern_lifts)
plt.axhline(y=1.0, color='red', linestyle='--', label='No lift')
plt.xlabel('Pattern')
plt.ylabel('Lift (vs baseline)')
plt.title('Predictive Power of Special Null Patterns')
plt.xticks(range(len(pattern_names)), pattern_names, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

plt.savefig('output/20250705_0122_null_pattern_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n\nAnalysis completed at: {datetime.now()}")
print("\nResults saved to:")
print("  - output/20250705_0122_null_pattern_results.json")
print("  - output/20250705_0122_null_pattern_analysis.png")

# Print key insights
print("\n=== KEY INSIGHTS ===")
print("1. Most correlated null indicator:", sorted_correlations[0])
print("2. Most predictive null pattern:", extreme_clusters[0])
print("3. Highest lift special pattern:", max(pattern_results.items(), key=lambda x: x[1]['lift']))

# RESULT: Key discoveries:
# 1. Drained_after_socializing null is HIGHLY predictive: P(E|Null)=0.402 vs P(E|NotNull)=0.762
# 2. Missing Drained_after_socializing → 60% likely to be Introvert (lift=0.55)
# 3. No nulls → 83% likely to be Extrovert (lift=1.12)
# 4. Cluster 5 (only Drained null) captures 976 samples with 40.5% extrovert rate
# 5. Test set has similar null patterns - strategy should generalize