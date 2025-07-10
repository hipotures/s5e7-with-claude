#!/usr/bin/env python3
"""
Analyze what's special about IDs ending in 34
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")

def analyze_pattern_34():
    """Analyze all IDs ending in 34 vs others"""
    print("="*60)
    print("ANALIZA WZORCA '34'")
    print("="*60)
    
    # Load both train and test data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Mark IDs ending in 34
    train_df['ends_34'] = train_df['id'] % 100 == 34
    test_df['ends_34'] = test_df['id'] % 100 == 34
    
    # Count in each dataset
    print("\n1. WYSTĘPOWANIE WZORCA:")
    print("-"*40)
    print(f"Train: {train_df['ends_34'].sum()} z {len(train_df)} ({train_df['ends_34'].sum()/len(train_df)*100:.1f}%)")
    print(f"Test: {test_df['ends_34'].sum()} z {len(test_df)} ({test_df['ends_34'].sum()/len(test_df)*100:.1f}%)")
    
    # Analyze personality distribution
    print("\n2. ROZKŁAD PERSONALITY W TRAIN:")
    print("-"*40)
    
    # For IDs ending in 34
    train_34 = train_df[train_df['ends_34']]
    e_ratio_34 = (train_34['Personality'] == 'Extrovert').sum() / len(train_34)
    print(f"ID kończące się na 34: {e_ratio_34:.3f} Extrovert")
    
    # For other IDs
    train_not_34 = train_df[~train_df['ends_34']]
    e_ratio_not_34 = (train_not_34['Personality'] == 'Extrovert').sum() / len(train_not_34)
    print(f"Pozostałe ID: {e_ratio_not_34:.3f} Extrovert")
    print(f"Różnica: {e_ratio_34 - e_ratio_not_34:+.3f}")
    
    # Feature analysis
    print("\n3. ŚREDNIE WARTOŚCI CECH:")
    print("-"*40)
    
    features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 
                'Going_outside', 'Post_frequency']
    
    for feature in features:
        mean_34 = train_34[feature].mean()
        mean_not_34 = train_not_34[feature].mean()
        diff = mean_34 - mean_not_34
        print(f"{feature:25} | 34: {mean_34:5.2f} | inne: {mean_not_34:5.2f} | diff: {diff:+5.2f}")
    
    # Binary features
    print("\n4. CECHY BINARNE:")
    print("-"*40)
    
    for feature in ['Drained_after_socializing', 'Stage_fear']:
        yes_34 = (train_34[feature] == 'Yes').sum() / len(train_34)
        yes_not_34 = (train_not_34[feature] == 'Yes').sum() / len(train_not_34)
        diff = yes_34 - yes_not_34
        print(f"{feature:25} | 34: {yes_34:.3f} | inne: {yes_not_34:.3f} | diff: {diff:+.3f}")
    
    # Check specific IDs around our errors
    print("\n5. ANALIZA OKOLIC BŁĘDÓW:")
    print("-"*40)
    
    error_ids = [18634, 20934]
    for error_id in error_ids:
        print(f"\nOkoło ID {error_id}:")
        nearby_ids = list(range(error_id - 5, error_id + 6))
        
        # Check train data
        nearby_train = train_df[train_df['id'].isin(nearby_ids)]
        if len(nearby_train) > 0:
            print("W train:")
            for _, row in nearby_train.iterrows():
                marker = " <-- ERROR" if row['id'] == error_id else ""
                print(f"  ID {row['id']}: {row['Personality']}{marker}")
    
    # Pattern analysis
    print("\n6. INNE WZORCE NUMERYCZNE:")
    print("-"*40)
    
    # Check last 2 digits
    train_df['last_2_digits'] = train_df['id'] % 100
    
    # Find which endings have highest E ratio
    digit_stats = []
    for ending in range(100):
        mask = train_df['last_2_digits'] == ending
        if mask.sum() > 50:  # At least 50 samples
            e_ratio = (train_df[mask]['Personality'] == 'Extrovert').sum() / mask.sum()
            digit_stats.append({
                'ending': ending,
                'count': mask.sum(),
                'e_ratio': e_ratio
            })
    
    digit_stats_df = pd.DataFrame(digit_stats)
    digit_stats_df = digit_stats_df.sort_values('e_ratio', ascending=False)
    
    print("TOP 10 końcówek z najwyższym E-ratio:")
    for _, row in digit_stats_df.head(10).iterrows():
        marker = " <--" if row['ending'] == 34 else ""
        print(f"  ...{row['ending']:02d}: {row['e_ratio']:.3f} (n={row['count']}){marker}")
    
    print("\nBOTTOM 10 końcówek z najniższym E-ratio:")
    for _, row in digit_stats_df.tail(10).iterrows():
        marker = " <--" if row['ending'] == 34 else ""
        print(f"  ...{row['ending']:02d}: {row['e_ratio']:.3f} (n={row['count']}){marker}")
    
    # ID 20932 analysis (the other error)
    print("\n7. ANALIZA ID 20932 (trzeci błąd):")
    print("-"*40)
    print("ID 20932 kończy się na 32, nie 34!")
    print("Jest 2 numery przed 20934")
    print("To sugeruje że błędy mogą być w OKOLICY, nie tylko na końcówce 34")

if __name__ == "__main__":
    analyze_pattern_34()