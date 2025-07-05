"""
PURPOSE: Comprehensive null value analysis for the personality dataset
HYPOTHESIS: Understanding null patterns can reveal data collection issues or feature relationships
EXPECTED: Identify which features have nulls, their distributions, and combinations
RESULT: To be determined after analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Read the data
print(f"Analysis started at: {datetime.now()}")
train_df = pd.read_csv("../../train.csv")
test_df = pd.read_csv("../../test.csv")

print("=== DATASET OVERVIEW ===")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

print("\n=== NULL VALUES IN TRAINING DATA ===")
print("\nNull counts per column:")
null_counts = train_df.isnull().sum()
null_percentages = (train_df.isnull().sum() / len(train_df)) * 100
null_info = pd.DataFrame({
    'Null_Count': null_counts,
    'Null_Percentage': null_percentages
})
print(null_info[null_info['Null_Count'] > 0])

print("\n=== NULL VALUES IN TEST DATA ===")
print("\nNull counts per column:")
null_counts_test = test_df.isnull().sum()
null_percentages_test = (test_df.isnull().sum() / len(test_df)) * 100
null_info_test = pd.DataFrame({
    'Null_Count': null_counts_test,
    'Null_Percentage': null_percentages_test
})
print(null_info_test[null_info_test['Null_Count'] > 0])

print("\n=== TOTAL ROWS WITH ANY NULL VALUES ===")
print(f"Train: {train_df.isnull().any(axis=1).sum()} rows ({(train_df.isnull().any(axis=1).sum() / len(train_df)) * 100:.2f}%)")
print(f"Test: {test_df.isnull().any(axis=1).sum()} rows ({(test_df.isnull().any(axis=1).sum() / len(test_df)) * 100:.2f}%)")

print("\n=== NULL PATTERNS BY PERSONALITY TYPE ===")
print("\nNull percentages by personality type:")
personality_null_stats = []
for personality in ['Introvert', 'Extrovert']:
    subset = train_df[train_df['Personality'] == personality]
    null_pct = (subset.isnull().any(axis=1).sum() / len(subset)) * 100
    personality_null_stats.append({
        'Personality': personality,
        'Total_Rows': len(subset),
        'Rows_with_Nulls': subset.isnull().any(axis=1).sum(),
        'Null_Percentage': null_pct
    })

personality_df = pd.DataFrame(personality_null_stats)
print(personality_df)

print("\n\nNull counts per feature by personality type:")
for personality in ['Introvert', 'Extrovert']:
    print(f"\n{personality}:")
    subset = train_df[train_df['Personality'] == personality]
    null_counts = subset.isnull().sum()
    null_pcts = (null_counts / len(subset)) * 100
    for col in train_df.columns[1:-1]:  # Skip id and Personality
        if null_counts[col] > 0:
            print(f"  {col}: {null_counts[col]} ({null_pcts[col]:.2f}%)")

print("\n=== NULL COMBINATION PATTERNS ===")
print("\nMost common null combinations (top 20):")

# Create a pattern for each row showing which columns are null
null_patterns = train_df.iloc[:, 1:-1].isnull().astype(int).astype(str)
null_patterns_str = null_patterns.apply(lambda x: ''.join(x), axis=1)

# Count combinations
pattern_counts = null_patterns_str.value_counts()
print(f"\nTotal unique null patterns: {len(pattern_counts)}")
print(f"Pattern with no nulls: {pattern_counts.get('0000000', 0)} rows")
print("\nTop 20 null patterns:")
for i, (pattern, count) in enumerate(pattern_counts.head(20).items()):
    if pattern != '0000000':
        # Decode which columns are null
        null_cols = []
        feature_names = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                        'Post_frequency']
        for j, char in enumerate(pattern):
            if char == '1':
                null_cols.append(feature_names[j])
        print(f"{i+1}. Pattern {pattern}: {count} rows ({count/len(train_df)*100:.2f}%) - Nulls in: {null_cols}")

# Save detailed null analysis
null_analysis = pd.DataFrame({
    'Feature': null_info.index[null_info['Null_Count'] > 0],
    'Train_Null_Count': null_info[null_info['Null_Count'] > 0]['Null_Count'].values,
    'Train_Null_Pct': null_info[null_info['Null_Count'] > 0]['Null_Percentage'].values,
    'Test_Null_Count': [null_counts_test[feat] for feat in null_info.index[null_info['Null_Count'] > 0]],
    'Test_Null_Pct': [null_percentages_test[feat] for feat in null_info.index[null_info['Null_Count'] > 0]]
})

null_analysis.to_csv('output/null_analysis_summary.csv', index=False)
print("\n\nNull analysis summary saved to output/null_analysis_summary.csv")

# Use ydata-profiling for detailed analysis
print("\n=== GENERATING YDATA PROFILING REPORT ===")
try:
    from ydata_profiling import ProfileReport
    
    # Generate profile for train data
    profile = ProfileReport(train_df, title="Personality Dataset Analysis", explorative=True)
    profile.to_file("output/train_data_profile.html")
    print("YData profiling report saved to output/train_data_profile.html")
    
    # Generate minimal profile for test data
    test_profile = ProfileReport(test_df, title="Test Dataset Analysis", minimal=True)
    test_profile.to_file("output/test_data_profile.html")
    print("YData profiling report for test data saved to output/test_data_profile.html")
    
except ImportError:
    print("ydata-profiling not installed. Skipping detailed profiling.")
except Exception as e:
    print(f"Error generating profile: {e}")

print(f"\nAnalysis completed at: {datetime.now()}")