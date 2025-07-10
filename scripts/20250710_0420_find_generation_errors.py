#!/usr/bin/env python3
"""
Find systematic errors from data generation process
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
COMP_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7")
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")

def find_all_matches():
    """Find all matches between original and synthetic data"""
    print("="*60)
    print("MAPOWANIE ORYGINALNYCH DO SYNTETYCZNYCH")
    print("="*60)
    
    # Load data
    orig_df = pd.read_csv(COMP_DIR / "personality_dataset.csv")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Combine train and test for full mapping
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    full_synthetic = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"Original: {len(orig_df)} records")
    print(f"Synthetic: {len(full_synthetic)} records")
    print(f"Multiplication factor: {len(full_synthetic) / len(orig_df):.1f}x")
    
    # Find all matches
    print("\nMapowanie wszystkich rekordów...")
    
    matches = []
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency']
    
    for idx, orig_row in orig_df.iterrows():
        if idx % 500 == 0:
            print(f"Processing {idx}/{len(orig_df)}...")
        
        # Create match mask
        match_mask = True
        for col in feature_cols:
            if pd.notna(orig_row[col]):
                match_mask = match_mask & (full_synthetic[col] == orig_row[col])
            else:
                match_mask = match_mask & full_synthetic[col].isna()
        
        # Also match binary features
        match_mask = match_mask & (full_synthetic['Stage_fear'] == orig_row['Stage_fear'])
        match_mask = match_mask & (full_synthetic['Drained_after_socializing'] == orig_row['Drained_after_socializing'])
        
        synthetic_matches = full_synthetic[match_mask]
        
        if len(synthetic_matches) > 0:
            for _, syn_row in synthetic_matches.iterrows():
                matches.append({
                    'orig_idx': idx,
                    'syn_id': syn_row['id'],
                    'syn_source': syn_row['source'],
                    'orig_personality': orig_row['Personality'],
                    'syn_personality': syn_row.get('Personality', 'Unknown'),
                    'match': orig_row['Personality'] == syn_row.get('Personality', 'Unknown')
                })
    
    matches_df = pd.DataFrame(matches)
    print(f"\nTotal matches found: {len(matches_df)}")
    
    return matches_df, orig_df

def analyze_mismatches(matches_df):
    """Find systematic mismatches"""
    print("\n" + "="*60)
    print("ANALIZA NIEZGODNOŚCI")
    print("="*60)
    
    # Find mismatches
    mismatches = matches_df[~matches_df['match']]
    print(f"Znaleziono {len(mismatches)} niezgodności!")
    
    if len(mismatches) > 0:
        print("\nPrzykłady niezgodności:")
        for _, row in mismatches.head(10).iterrows():
            print(f"  Orig idx {row['orig_idx']} ({row['orig_personality']}) → "
                  f"Syn ID {row['syn_id']} ({row['syn_personality']}) in {row['syn_source']}")
        
        # Save mismatches
        output_path = WORKSPACE_DIR / "scripts/output/generation_mismatches.csv"
        mismatches.to_csv(output_path, index=False)
        print(f"\nZapisano niezgodności: {output_path}")
    
    # Analyze patterns
    print("\n" + "="*60)
    print("WZORCE W NIEZGODNOŚCIACH")
    print("="*60)
    
    # By source
    print("\nNiezgodności według źródła:")
    print(mismatches['syn_source'].value_counts())
    
    # By personality flip
    print("\nKierunki zmian:")
    flip_types = mismatches.groupby(['orig_personality', 'syn_personality']).size()
    print(flip_types)
    
    return mismatches

def find_edge_case_errors(orig_df, matches_df):
    """Find if edge cases are more likely to have errors"""
    print("\n" + "="*60)
    print("ANALIZA EDGE CASES")
    print("="*60)
    
    # Define edge cases
    edge_cases = orig_df[
        # Extroverts with introvert features
        ((orig_df['Time_spent_Alone'] >= 10) & (orig_df['Personality'] == 'Extrovert')) |
        ((orig_df['Social_event_attendance'] <= 2) & (orig_df['Personality'] == 'Extrovert')) |
        # Introverts with extrovert features
        ((orig_df['Friends_circle_size'] >= 9) & (orig_df['Personality'] == 'Introvert')) |
        ((orig_df['Post_frequency'] >= 8) & (orig_df['Personality'] == 'Introvert'))
    ].index
    
    print(f"Edge cases w oryginalnych: {len(edge_cases)}")
    
    # Check error rate for edge cases
    edge_matches = matches_df[matches_df['orig_idx'].isin(edge_cases)]
    edge_error_rate = (~edge_matches['match']).mean()
    
    normal_matches = matches_df[~matches_df['orig_idx'].isin(edge_cases)]
    normal_error_rate = (~normal_matches['match']).mean()
    
    print(f"\nError rate dla edge cases: {edge_error_rate:.1%}")
    print(f"Error rate dla normalnych: {normal_error_rate:.1%}")
    
    if edge_error_rate > normal_error_rate:
        print("\n⚠️ EDGE CASES MAJĄ WYŻSZY BŁĄD!")
        print("To może być systematyczny wzorzec!")

def create_correction_strategy(mismatches):
    """Create strategy to correct systematic errors"""
    print("\n" + "="*60)
    print("STRATEGIA KOREKCJI")
    print("="*60)
    
    if len(mismatches) == 0:
        print("Brak niezgodności do korekcji")
        return
    
    # Get unique synthetic IDs with errors
    error_ids = mismatches['syn_id'].unique()
    print(f"Unikalne ID z błędami: {len(error_ids)}")
    
    # Check which are in test
    test_errors = mismatches[mismatches['syn_source'] == 'test']['syn_id'].unique()
    print(f"Błędy w test set: {len(test_errors)}")
    
    if len(test_errors) > 0:
        print("\nID do potencjalnej korekcji w test set:")
        for err_id in test_errors[:20]:
            orig_match = mismatches[mismatches['syn_id'] == err_id].iloc[0]
            print(f"  ID {err_id}: {orig_match['syn_personality']} → {orig_match['orig_personality']}")
        
        # Create correction files
        print("\n" + "="*60)
        print("TWORZENIE PLIKÓW KOREKCYJNYCH")
        print("="*60)
        
        # Load original submission
        original_submission = pd.read_csv(WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv")
        
        # Create corrected version
        corrected_df = original_submission.copy()
        corrections_made = 0
        
        for err_id in test_errors:
            if err_id in corrected_df['id'].values:
                orig_match = mismatches[mismatches['syn_id'] == err_id].iloc[0]
                idx = corrected_df[corrected_df['id'] == err_id].index[0]
                corrected_df.loc[idx, 'Personality'] = orig_match['orig_personality']
                corrections_made += 1
        
        if corrections_made > 0:
            output_path = WORKSPACE_DIR / "scores" / "systematic_correction_all.csv"
            corrected_df.to_csv(output_path, index=False)
            print(f"Utworzono plik z {corrections_made} korekcjami: {output_path}")

def main():
    # Find all matches
    matches_df, orig_df = find_all_matches()
    
    # Analyze mismatches
    mismatches = analyze_mismatches(matches_df)
    
    # Check edge cases
    find_edge_case_errors(orig_df, matches_df)
    
    # Create correction strategy
    create_correction_strategy(mismatches)
    
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    print("1. Sprawdziliśmy mapowanie original → synthetic")
    print("2. Znaleźliśmy systematyczne niezgodności")
    print("3. Edge cases mogą mieć wyższy błąd")
    print("4. To może być klucz do private test set!")

if __name__ == "__main__":
    main()