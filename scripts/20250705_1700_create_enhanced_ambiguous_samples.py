#!/usr/bin/env python3
"""
PURPOSE: Create enhanced ambiguous samples file combining multiple insights:
         1. Original ambiguity score approach (boundary proximity)
         2. Null pattern insights (introverts have more nulls)
         3. Cluster-based error patterns (clusters 2 & 7)
         4. The 5 critical records that separate accuracy levels

HYPOTHESIS: Combining all discovered patterns will identify a better set of
            ambiguous samples that are more predictive of misclassifications.

EXPECTED: Create a comprehensive ambiguous samples file that includes:
          - Original 450 boundary samples
          - High null-pattern samples
          - Samples from problematic clusters
          - Critical boundary cases

RESULT: [To be filled after execution]
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREATING ENHANCED AMBIGUOUS SAMPLES")
print("="*80)

# Load data
print("\nLoading data...")
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

# Load original ambiguous samples
original_ambiguous = pd.read_csv("/mnt/ml/kaggle/playground-series-s5e7/most_ambiguous_2.43pct.csv")
print(f"Loaded {len(original_ambiguous)} original ambiguous samples")

# Define features
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 
                     'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_features = ['Stage_fear', 'Drained_after_socializing']
all_features = numerical_features + categorical_features

# Preprocessing
print("\nPreprocessing data...")
train_encoded = train_df.copy()
test_encoded = test_df.copy()

# Encode categorical features
for col in categorical_features:
    train_encoded[col] = train_encoded[col].map({'Yes': 1, 'No': 0})
    test_encoded[col] = test_encoded[col].map({'Yes': 1, 'No': 0})

# Encode target
train_encoded['Personality_binary'] = (train_encoded['Personality'] == 'Extrovert').astype(int)

print("\n" + "="*60)
print("1. ORIGINAL AMBIGUITY SCORE APPROACH")
print("="*60)

# Recreate ambiguity scores
def calculate_ambiguity_scores(df):
    """Calculate ambiguity scores as in original approach"""
    # Handle missing values for scoring
    df_score = df.copy()
    for col in numerical_features:
        df_score[col] = df_score[col].fillna(df[col].mean())
    
    # Typical extrovert score
    typical_extrovert_score = (
        (5 - df_score['Time_spent_Alone']) + 
        df_score['Social_event_attendance'] + 
        df_score['Going_outside'] + 
        df_score['Friends_circle_size']/3 + 
        df_score['Post_frequency']
    )
    
    # Typical introvert score  
    typical_introvert_score = (
        df_score['Time_spent_Alone'] + 
        (10 - df_score['Social_event_attendance']) + 
        (10 - df_score['Going_outside']) + 
        (30 - df_score['Friends_circle_size'])/3 + 
        (10 - df_score['Post_frequency'])
    )
    
    # Ambiguity score (lower = more ambiguous)
    ambiguity_score = np.abs(typical_extrovert_score - typical_introvert_score)
    
    return ambiguity_score, typical_extrovert_score, typical_introvert_score

train_encoded['ambiguity_score'], train_encoded['extrovert_score'], train_encoded['introvert_score'] = \
    calculate_ambiguity_scores(train_encoded)

# Verify we can recreate the original ambiguous set
original_ids = set(original_ambiguous['id'].values)
train_sorted = train_encoded.nsmallest(450, 'ambiguity_score')
recreated_ids = set(train_sorted['id'].values)
overlap = len(original_ids.intersection(recreated_ids))
print(f"Recreated {overlap}/450 original ambiguous samples")

print("\n" + "="*60)
print("2. NULL PATTERN INSIGHTS")
print("="*60)

# Add null pattern features
train_encoded['null_count'] = train_df[all_features].isna().sum(axis=1)
train_encoded['has_drained_null'] = train_df['Drained_after_socializing'].isna().astype(int)
train_encoded['has_stage_null'] = train_df['Stage_fear'].isna().astype(int)

# Calculate null pattern scores
print("\nNull patterns by personality:")
intro_mask = train_encoded['Personality_binary'] == 0
print(f"Introverts with nulls: {(train_encoded[intro_mask]['null_count'] > 0).mean():.1%}")
print(f"Extroverts with nulls: {(train_encoded[~intro_mask]['null_count'] > 0).mean():.1%}")

# Null ambiguity: cases where null pattern contradicts personality
train_encoded['null_ambiguity'] = 0
# Extroverts with many nulls (unusual)
train_encoded.loc[(train_encoded['Personality_binary'] == 1) & 
                  (train_encoded['null_count'] >= 2), 'null_ambiguity'] = 1
# Introverts with no nulls (unusual)
train_encoded.loc[(train_encoded['Personality_binary'] == 0) & 
                  (train_encoded['null_count'] == 0), 'null_ambiguity'] = 1

null_ambiguous = train_encoded[train_encoded['null_ambiguity'] == 1]
print(f"\nFound {len(null_ambiguous)} null-ambiguous samples:")
print(f"  Extroverts with many nulls: {len(null_ambiguous[null_ambiguous['Personality_binary'] == 1])}")
print(f"  Introverts with no nulls: {len(null_ambiguous[null_ambiguous['Personality_binary'] == 0])}")

print("\n" + "="*60)
print("3. CLUSTER-BASED ERROR PATTERNS")
print("="*60)

# Prepare data for clustering
X_cluster = train_encoded[all_features].copy()
# Fill NaN for clustering
for col in numerical_features:
    X_cluster[col] = X_cluster[col].fillna(train_encoded[col].mean())
for col in categorical_features:
    X_cluster[col] = X_cluster[col].fillna(-1)  # Special value for missing

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Perform clustering
print("\nPerforming K-means clustering (k=15)...")
kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
train_encoded['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze error rates by cluster
cluster_stats = []
for cluster_id in range(15):
    cluster_mask = train_encoded['cluster'] == cluster_id
    cluster_data = train_encoded[cluster_mask]
    
    if len(cluster_data) > 0:
        # For this analysis, we'll use ambiguity as proxy for error likelihood
        avg_ambiguity = cluster_data['ambiguity_score'].mean()
        cluster_size = len(cluster_data)
        
        cluster_stats.append({
            'cluster': cluster_id,
            'size': cluster_size,
            'avg_ambiguity': avg_ambiguity,
            'pct_introverts': (cluster_data['Personality_binary'] == 0).mean()
        })

cluster_df = pd.DataFrame(cluster_stats).sort_values('avg_ambiguity')
print("\nClusters with lowest ambiguity (most problematic):")
print(cluster_df.head())

# Get samples from problematic clusters (2 and 7 from our analysis)
problematic_clusters = [2, 7]
cluster_ambiguous = train_encoded[train_encoded['cluster'].isin(problematic_clusters)]
print(f"\nSamples from problematic clusters: {len(cluster_ambiguous)}")

print("\n" + "="*60)
print("4. BOUNDARY CASES ANALYSIS")
print("="*60)

# Find extreme boundary cases
# Cases where extrovert/introvert scores are very close
train_encoded['score_diff'] = np.abs(train_encoded['extrovert_score'] - train_encoded['introvert_score'])
train_encoded['normalized_extrovert'] = train_encoded['extrovert_score'] / (
    train_encoded['extrovert_score'] + train_encoded['introvert_score'])

# Boundary cases: normalized score near 0.5
train_encoded['is_boundary'] = np.abs(train_encoded['normalized_extrovert'] - 0.5) < 0.05
boundary_cases = train_encoded[train_encoded['is_boundary']]
print(f"Found {len(boundary_cases)} extreme boundary cases")

# Special patterns from our analysis
print("\nChecking for special patterns...")
# Mixed profile: high alone time but also high social
train_encoded['mixed_profile'] = (
    (train_encoded['Time_spent_Alone'] > 3) & 
    (train_encoded['Social_event_attendance'] > 4) &
    (train_encoded['Friends_circle_size'] > 8)
).astype(int)

mixed_cases = train_encoded[train_encoded['mixed_profile'] == 1]
print(f"Found {len(mixed_cases)} mixed profile cases")

print("\n" + "="*60)
print("5. CREATING ENHANCED AMBIGUOUS SAMPLES")
print("="*60)

# Combine all ambiguous indicators
train_encoded['ambiguity_sources'] = 0
train_encoded.loc[train_encoded['id'].isin(original_ids), 'ambiguity_sources'] += 1
train_encoded.loc[train_encoded['null_ambiguity'] == 1, 'ambiguity_sources'] += 1
train_encoded.loc[train_encoded['cluster'].isin(problematic_clusters), 'ambiguity_sources'] += 1
train_encoded.loc[train_encoded['is_boundary'], 'ambiguity_sources'] += 1
train_encoded.loc[train_encoded['mixed_profile'] == 1, 'ambiguity_sources'] += 1

# Create composite ambiguity score
# Weight different sources
weights = {
    'original': 1.0,
    'null': 0.8,
    'cluster': 0.6,
    'boundary': 1.0,
    'mixed': 0.7
}

train_encoded['composite_ambiguity'] = (
    (train_encoded['id'].isin(original_ids)).astype(float) * weights['original'] +
    train_encoded['null_ambiguity'] * weights['null'] +
    (train_encoded['cluster'].isin(problematic_clusters)).astype(float) * weights['cluster'] +
    train_encoded['is_boundary'].astype(float) * weights['boundary'] +
    train_encoded['mixed_profile'] * weights['mixed']
)

# Also consider the original ambiguity score
train_encoded['final_ambiguity_score'] = (
    train_encoded['composite_ambiguity'] + 
    (1 - train_encoded['ambiguity_score'] / train_encoded['ambiguity_score'].max())
)

# Select top ambiguous samples
n_samples = 600  # Increased from 450 to include more patterns
enhanced_ambiguous = train_encoded.nlargest(n_samples, 'final_ambiguity_score')

print(f"\nSelected {n_samples} enhanced ambiguous samples")
print("\nBreakdown by source:")
print(f"  From original 450: {len(enhanced_ambiguous[enhanced_ambiguous['id'].isin(original_ids)])}")
print(f"  With null ambiguity: {enhanced_ambiguous['null_ambiguity'].sum()}")
print(f"  From problematic clusters: {enhanced_ambiguous['cluster'].isin(problematic_clusters).sum()}")
print(f"  Boundary cases: {enhanced_ambiguous['is_boundary'].sum()}")
print(f"  Mixed profiles: {enhanced_ambiguous['mixed_profile'].sum()}")
print(f"  Multiple sources: {(enhanced_ambiguous['ambiguity_sources'] >= 2).sum()}")

# Prepare output
output_df = enhanced_ambiguous[['id'] + all_features + ['Personality']].copy()

# Add metadata columns
output_df['ambiguity_score'] = enhanced_ambiguous['ambiguity_score']
output_df['null_count'] = enhanced_ambiguous['null_count']
output_df['cluster'] = enhanced_ambiguous['cluster']
output_df['composite_ambiguity'] = enhanced_ambiguous['composite_ambiguity']
output_df['in_original_450'] = enhanced_ambiguous['id'].isin(original_ids).astype(int)

# Save enhanced ambiguous samples
output_path = 'output/enhanced_ambiguous_samples_600.csv'
output_df.to_csv(output_path, index=False)
print(f"\nSaved enhanced ambiguous samples to: {output_path}")

# Also save a version with just 450 samples for direct comparison
enhanced_450 = train_encoded.nlargest(450, 'final_ambiguity_score')
output_450 = enhanced_450[['id'] + all_features + ['Personality']].copy()
output_450['ambiguity_score'] = enhanced_450['ambiguity_score']
output_450['composite_ambiguity'] = enhanced_450['composite_ambiguity']

output_path_450 = 'output/enhanced_ambiguous_samples_450.csv'
output_450.to_csv(output_path_450, index=False)
print(f"Saved 450-sample version to: {output_path_450}")

# Analysis summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Created {n_samples} enhanced ambiguous samples combining:")
print("- Original boundary proximity approach")
print("- Null pattern insights (extroverts with nulls, introverts without)")
print("- Cluster-based patterns (focusing on clusters 2 & 7)")
print("- Extreme boundary cases (50/50 probability)")
print("- Mixed behavioral profiles")
print(f"\nOverlap with original: {len(enhanced_ambiguous[enhanced_ambiguous['id'].isin(original_ids)])/450:.1%}")
print("\nThese samples should better represent the true ambiguous cases that")
print("cause misclassifications in the personality prediction task.")

# Save detailed analysis
analysis_df = train_encoded[['id', 'Personality', 'ambiguity_score', 'null_count', 
                            'cluster', 'composite_ambiguity', 'final_ambiguity_score',
                            'ambiguity_sources']].copy()
analysis_df = analysis_df.sort_values('final_ambiguity_score', ascending=False)
analysis_df.to_csv('output/ambiguity_analysis_full.csv', index=False)
print(f"\nSaved full ambiguity analysis to: output/ambiguity_analysis_full.csv")