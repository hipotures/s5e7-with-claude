#!/usr/bin/env python3
"""Find the exact 16→2 mapping rule by analyzing edge cases.

PURPOSE: Discover the exact rule used to map 16 MBTI types to 2 classes by
         analyzing edge cases, training decision trees, and testing specific
         threshold rules.

HYPOTHESIS: There's a deterministic rule (possibly complex) that maps 16 MBTI
            types to Introvert/Extrovert, with ~2.43% of samples being edge cases
            that don't follow the simple first-letter rule.

EXPECTED: Find strong introverts labeled as extroverts and vice versa, discover
          specific patterns like ISFJ that cause ambiguity, and identify threshold
          rules that achieve ~97.5% accuracy matching our target.

RESULT: Found edge cases: strong introverts labeled as Extrovert and strong
        extroverts labeled as Introvert. Identified ISFJ pattern as particularly
        problematic. Decision trees revealed Drained_after_socializing as most
        important feature. The most ambiguous 2.43% have very low ambiguity scores,
        confirming they're on type boundaries. Best rule found was MBTI-aware
        with special handling for ISFJ and ENFJ patterns.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("FINDING EXACT 16→2 MAPPING RULE")
print("="*60)

# Load data
train_df = pd.read_csv("../../train.csv")

# Preprocessing
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
            'Post_frequency']

numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

for col in numerical_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mean())

for col in categorical_cols:
    train_df[col] = train_df[col].fillna('Missing')
    train_df[col] = train_df[col].map({'Yes': 1, 'No': 0, 'Missing': 0.5})

# Binary target
train_df['is_extrovert'] = (train_df['Personality'] == 'Extrovert').astype(int)

# Create all possible MBTI indicators
print("\nCreating detailed MBTI indicators...")

# E/I indicators
train_df['high_alone_time'] = (train_df['Time_spent_Alone'] > 5).astype(int)
train_df['low_social_events'] = (train_df['Social_event_attendance'] < 5).astype(int)
train_df['many_friends'] = (train_df['Friends_circle_size'] > 8).astype(int)
train_df['high_posting'] = (train_df['Post_frequency'] > 6).astype(int)
train_df['gets_drained'] = train_df['Drained_after_socializing']
train_df['has_stage_fear'] = train_df['Stage_fear']

# Specific combinations that might indicate MBTI types
train_df['isfj_pattern'] = (
    (train_df['gets_drained'] == 0) &  # Doesn't get drained (unusual for I)
    (train_df['many_friends'] == 1) &  # Many friends
    (train_df['low_social_events'] == 1)  # But low events (contradiction)
).astype(int)

train_df['infp_pattern'] = (
    (train_df['high_alone_time'] == 1) &
    (train_df['has_stage_fear'] == 1) &
    (train_df['Post_frequency'] < 3)  # Very low posting
).astype(int)

train_df['enfj_pattern'] = (
    (train_df['gets_drained'] == 1) &  # Gets drained (unusual for E)
    (train_df['Social_event_attendance'] > 7) &
    (train_df['many_friends'] == 1)
).astype(int)

# Calculate "typicalness" scores
train_df['typical_introvert_score'] = (
    train_df['high_alone_time'] + 
    train_df['low_social_events'] + 
    train_df['gets_drained'] + 
    train_df['has_stage_fear'] +
    (1 - train_df['many_friends']) +
    (1 - train_df['high_posting'])
) / 6

train_df['typical_extrovert_score'] = (
    (1 - train_df['high_alone_time']) + 
    (1 - train_df['low_social_events']) + 
    (1 - train_df['gets_drained']) + 
    (1 - train_df['has_stage_fear']) +
    train_df['many_friends'] +
    train_df['high_posting']
) / 6

# Find edge cases - people who should be one type but are labeled as another
print("\n" + "="*60)
print("ANALYZING EDGE CASES")
print("="*60)

# Strong introverts labeled as extroverts
strong_i_as_e = train_df[
    (train_df['typical_introvert_score'] > 0.7) & 
    (train_df['is_extrovert'] == 1)
]
print(f"\nStrong introverts labeled as Extrovert: {len(strong_i_as_e)} ({len(strong_i_as_e)/len(train_df)*100:.2f}%)")

# Strong extroverts labeled as introverts
strong_e_as_i = train_df[
    (train_df['typical_extrovert_score'] > 0.7) & 
    (train_df['is_extrovert'] == 0)
]
print(f"Strong extroverts labeled as Introvert: {len(strong_e_as_i)} ({len(strong_e_as_i)/len(train_df)*100:.2f}%)")

# Ambiguous cases (neither strongly I nor E)
ambiguous = train_df[
    (train_df['typical_introvert_score'].between(0.4, 0.6)) &
    (train_df['typical_extrovert_score'].between(0.4, 0.6))
]
print(f"Ambiguous cases: {len(ambiguous)} ({len(ambiguous)/len(train_df)*100:.2f}%)")
print(f"  Of these, {(ambiguous['is_extrovert'].sum()/len(ambiguous)*100):.1f}% are labeled Extrovert")

# Try to find the exact rule using Decision Tree
print("\n" + "="*60)
print("LEARNING THE MAPPING RULE WITH DECISION TREE")
print("="*60)

# Prepare features for rule learning
rule_features = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
    'Post_frequency', 'isfj_pattern', 'infp_pattern', 'enfj_pattern',
    'typical_introvert_score', 'typical_extrovert_score'
]

X = train_df[rule_features]
y = train_df['is_extrovert']

# Train a simple decision tree to find rules
for max_depth in [2, 3, 4, 5]:
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X, y)
    
    accuracy = accuracy_score(y, dt.predict(X))
    print(f"\nDepth {max_depth} tree accuracy: {accuracy:.6f}")
    
    if accuracy > 0.974:  # Close to our target
        print("Tree rules:")
        # Get feature importances
        importances = pd.DataFrame({
            'Feature': rule_features,
            'Importance': dt.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(importances[importances['Importance'] > 0].head())

# Find specific threshold rules
print("\n" + "="*60)
print("TESTING SPECIFIC THRESHOLD RULES")
print("="*60)

# Test different combinations
test_rules = [
    {
        'name': 'Simple E/I',
        'rule': lambda df: (
            (df['Time_spent_Alone'] < 5) & 
            (df['Social_event_attendance'] > 3)
        ).astype(int)
    },
    {
        'name': 'Drained-based',
        'rule': lambda df: (
            (df['Drained_after_socializing'] == 0) | 
            ((df['Drained_after_socializing'] == 1) & (df['Social_event_attendance'] > 7))
        ).astype(int)
    },
    {
        'name': 'Complex with exceptions',
        'rule': lambda df: (
            # Base rule
            ((df['Time_spent_Alone'] < 4.5) & (df['Drained_after_socializing'] == 0)) |
            # Exception 1: Social introverts
            ((df['Time_spent_Alone'] < 3) & (df['Friends_circle_size'] > 9)) |
            # Exception 2: Antisocial extroverts
            ((df['Stage_fear'] == 0) & (df['Social_event_attendance'] > 4))
        ).astype(int)
    },
    {
        'name': 'MBTI-aware',
        'rule': lambda df: (
            # Standard E types
            ((df['Time_spent_Alone'] < 5) & (df['gets_drained'] == 0)) |
            # ISFJ exception (social introvert)
            (df['isfj_pattern'] == 1) |
            # ENFJ exception (gets drained but still E)
            ((df['enfj_pattern'] == 1) & (df['Social_event_attendance'] > 6))
        ).astype(int)
    }
]

best_accuracy = 0
best_rule = None

for rule_dict in test_rules:
    pred = rule_dict['rule'](train_df)
    accuracy = accuracy_score(y, pred)
    errors = (pred != y).sum()
    error_rate = errors / len(y) * 100
    
    print(f"\n{rule_dict['name']}:")
    print(f"  Accuracy: {accuracy:.6f}")
    print(f"  Errors: {errors} ({error_rate:.2f}%)")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_rule = rule_dict['name']
    
    # Check if this matches our target
    if 0.974 < accuracy < 0.976:
        print(f"  ✓ This could be the rule! Error rate {error_rate:.2f}% ≈ 2.43%")

print("\n" + "="*60)
print("FINAL ANALYSIS")
print("="*60)

# Analyze the exact 2.43% that would be misclassified
target_errors = int(len(train_df) * 0.0243)
print(f"\nLooking for exactly {target_errors} errors (~2.43%)...")

# Sort by how "ambiguous" each person is
train_df['ambiguity_score'] = np.abs(
    train_df['typical_extrovert_score'] - train_df['typical_introvert_score']
)

# The most ambiguous people might be the ones misclassified
most_ambiguous = train_df.nsmallest(target_errors, 'ambiguity_score')

print(f"\nTop {target_errors} most ambiguous people:")
print(f"  {(most_ambiguous['is_extrovert'] == 1).sum()} are labeled Extrovert")
print(f"  {(most_ambiguous['is_extrovert'] == 0).sum()} are labeled Introvert")

# Check their patterns
print("\nTheir characteristics:")
for col in ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']:
    avg_all = train_df[col].mean()
    avg_ambiguous = most_ambiguous[col].mean()
    print(f"  {col}: {avg_ambiguous:.2f} (all: {avg_all:.2f})")

# Save the most ambiguous for further analysis
most_ambiguous.to_csv('most_ambiguous_2.43pct.csv', index=False)
print("\nSaved most ambiguous 2.43% to 'most_ambiguous_2.43pct.csv'")

print("\n" + "="*60)
print("CONCLUSIONS:")
print("="*60)
print(f"Best rule found: {best_rule} (accuracy: {best_accuracy:.6f})")
print("\nThe 16→2 mapping likely uses:")
print("1. Primary rule based on Drained_after_socializing")
print("2. Secondary rules for edge cases")
print("3. ~2.43% are ambiguous cases that could go either way")
print("\nTo beat 0.975708, we need to correctly classify these ambiguous cases!")