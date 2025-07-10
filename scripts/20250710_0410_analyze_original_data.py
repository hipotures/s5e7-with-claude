#!/usr/bin/env python3
"""
Analyze original personality datasets vs synthetic data
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
COMP_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7")
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")

def load_and_compare_datasets():
    """Load all datasets and compare"""
    print("="*60)
    print("ANALIZA ORYGINALNYCH DANYCH")
    print("="*60)
    
    # Load original datasets
    try:
        orig1 = pd.read_csv(COMP_DIR / "personality_datasert.csv")
        print(f"\n1. personality_datasert.csv: {orig1.shape}")
        print(f"Columns: {list(orig1.columns)}")
    except Exception as e:
        print(f"Error loading datasert: {e}")
        orig1 = None
    
    try:
        orig2 = pd.read_csv(COMP_DIR / "personality_dataset.csv")
        print(f"\n2. personality_dataset.csv: {orig2.shape}")
        print(f"Columns: {list(orig2.columns)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        orig2 = None
    
    # Load synthetic data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"\n3. Synthetic train.csv: {train_df.shape}")
    print(f"4. Synthetic test.csv: {test_df.shape}")
    
    return orig1, orig2, train_df, test_df

def analyze_original_data(orig1, orig2):
    """Analyze original datasets"""
    print("\n" + "="*60)
    print("SZCZEGÓŁOWA ANALIZA ORYGINALNYCH DANYCH")
    print("="*60)
    
    for name, df in [("datasert", orig1), ("dataset", orig2)]:
        if df is not None:
            print(f"\n{name.upper()}:")
            print("-"*40)
            
            # Check if has personality column
            if 'Personality' in df.columns:
                personality_dist = df['Personality'].value_counts()
                print(f"Personality distribution:")
                for val, count in personality_dist.items():
                    print(f"  {val}: {count} ({count/len(df)*100:.1f}%)")
            
            # Check for duplicates
            print(f"\nDuplicates: {df.duplicated().sum()}")
            
            # Sample data
            print(f"\nFirst 3 rows:")
            print(df.head(3))
            
            # Check for nulls
            print(f"\nNull counts:")
            null_counts = df.isnull().sum()
            for col, count in null_counts.items():
                if count > 0:
                    print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")

def find_exact_matches(orig_df, train_df, test_df):
    """Find exact matches between original and synthetic data"""
    print("\n" + "="*60)
    print("SZUKANIE DOKŁADNYCH DOPASOWAŃ")
    print("="*60)
    
    if orig_df is None:
        print("No original data to compare")
        return
    
    # Normalize column names
    orig_cols = [col.strip().replace(' ', '_') for col in orig_df.columns]
    orig_df.columns = orig_cols
    
    # Common features
    common_features = []
    for col in orig_df.columns:
        if col in train_df.columns and col != 'id' and col != 'Personality':
            common_features.append(col)
    
    print(f"Common features: {common_features}")
    
    if len(common_features) > 3:
        # Look for exact matches
        print("\nSzukanie dokładnych dopasowań...")
        
        matches_train = []
        matches_test = []
        
        # Sample check (first 100 original records)
        for idx, orig_row in orig_df.head(100).iterrows():
            # Check in train
            match_mask = True
            for feat in common_features[:3]:  # Use first 3 features
                if pd.notna(orig_row[feat]):
                    match_mask = match_mask & (train_df[feat] == orig_row[feat])
            
            train_matches = train_df[match_mask]
            if len(train_matches) > 0:
                matches_train.append({
                    'orig_idx': idx,
                    'train_ids': train_matches['id'].values,
                    'orig_personality': orig_row.get('Personality', 'Unknown'),
                    'train_personality': train_matches['Personality'].values
                })
            
            # Check in test
            match_mask = True
            for feat in common_features[:3]:
                if pd.notna(orig_row[feat]):
                    match_mask = match_mask & (test_df[feat] == orig_row[feat])
            
            test_matches = test_df[match_mask]
            if len(test_matches) > 0:
                matches_test.append({
                    'orig_idx': idx,
                    'test_ids': test_matches['id'].values,
                    'orig_personality': orig_row.get('Personality', 'Unknown')
                })
        
        print(f"\nZnaleziono {len(matches_train)} dopasowań w train")
        print(f"Znaleziono {len(matches_test)} dopasowań w test")
        
        # Check for conflicts
        if matches_train:
            print("\nPrzykłady dopasowań train:")
            for match in matches_train[:5]:
                orig_p = match['orig_personality']
                train_p = match['train_personality'][0] if len(match['train_personality']) > 0 else 'None'
                conflict = "KONFLIKT!" if orig_p != train_p else "OK"
                print(f"  Orig idx {match['orig_idx']} ({orig_p}) → Train ID {match['train_ids'][0]} ({train_p}) {conflict}")

def analyze_generation_method():
    """Analyze how synthetic data was generated"""
    print("\n" + "="*60)
    print("ANALIZA METODY GENEROWANIA")
    print("="*60)
    
    print("\nMożliwe metody generowania syntetycznych danych:")
    print("1. SMOTE lub podobne - interpolacja między istniejącymi punktami")
    print("2. GANs - generowanie nowych przykładów")
    print("3. Dodanie szumu do oryginalnych danych")
    print("4. Rule-based generation z zachowaniem rozkładów")
    
    print("\nCo to oznacza dla błędów:")
    print("• Jeśli użyto interpolacji → błędy mogą być na 'granicach'")
    print("• Jeśli dodano szum → oryginalne etykiety mogą być zachowane")
    print("• Jeśli rule-based → mogą być systematyczne błędy w regułach")

def check_specific_hypothesis(orig_df, train_df, test_df):
    """Check specific hypothesis about errors"""
    print("\n" + "="*60)
    print("TESTOWANIE HIPOTEZ")
    print("="*60)
    
    if orig_df is None:
        return
    
    # Hypothesis 1: Edge cases in original might be errors in synthetic
    print("\n1. HIPOTEZA: Edge cases w oryginalnych danych")
    print("-"*40)
    
    if 'Personality' in orig_df.columns:
        # Find edge cases in original
        orig_df_clean = orig_df.dropna(subset=['Personality'])
        
        # Count feature combinations
        if 'Time_spent_Alone' in orig_df.columns and 'Social_event_attendance' in orig_df.columns:
            edge_cases = orig_df_clean[
                ((orig_df_clean['Time_spent_Alone'] > 10) & (orig_df_clean['Personality'] == 'Extrovert')) |
                ((orig_df_clean['Social_event_attendance'] > 8) & (orig_df_clean['Personality'] == 'Introvert'))
            ]
            
            print(f"Edge cases w oryginalnych: {len(edge_cases)}")
            if len(edge_cases) > 0:
                print("Przykłady:")
                print(edge_cases.head())

def main():
    # Load all datasets
    orig1, orig2, train_df, test_df = load_and_compare_datasets()
    
    # Analyze original data
    analyze_original_data(orig1, orig2)
    
    # Find matches between original and synthetic
    if orig2 is not None:  # Use the correct file
        find_exact_matches(orig2, train_df, test_df)
    
    # Analyze generation method
    analyze_generation_method()
    
    # Check hypotheses
    if orig2 is not None:
        check_specific_hypothesis(orig2, train_df, test_df)
    
    print("\n" + "="*60)
    print("WNIOSKI")
    print("="*60)
    print("1. Oryginalne dane mogą zawierać klucz do błędów")
    print("2. Proces generowania mógł wprowadzić systematyczne błędy")
    print("3. Edge cases z oryginalnych mogą być źle oznaczone w syntetycznych")

if __name__ == "__main__":
    main()