#!/usr/bin/env python3
"""
Analyze potential data batches/sources based on ID ranges and characteristics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"

def analyze_data_batches():
    """Analyze characteristics of different ID ranges"""
    print("="*60)
    print("ANALIZA POTENCJALNYCH PARTII DANYCH")
    print("="*60)
    
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Combine train and test for full analysis
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    
    # Get personalities for test data
    test_personalities = []
    for _, row in test_df.iterrows():
        idx = original_df[original_df['id'] == row['id']].index
        if len(idx) > 0:
            test_personalities.append(original_df.loc[idx[0], 'Personality'])
        else:
            test_personalities.append(None)
    test_df['Personality'] = test_personalities
    
    # Combine
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sort_values('id').reset_index(drop=True)
    
    # Analyze ID gaps
    print("\n1. ANALIZA LUKI W ID:")
    print("-"*40)
    
    id_diffs = full_df['id'].diff()
    large_gaps = full_df[id_diffs > 10].copy()
    large_gaps['gap_size'] = id_diffs[id_diffs > 10].astype(int)
    
    print("Duże luki w numeracji ID:")
    for _, row in large_gaps.iterrows():
        print(f"Przed ID {row['id']}: luka {row['gap_size']} numerów")
    
    # Define potential batches based on gaps and patterns
    batch_boundaries = [0]
    batch_boundaries.extend(large_gaps.index.tolist())
    batch_boundaries.append(len(full_df))
    
    # If no large gaps, use regular intervals
    if len(batch_boundaries) <= 2:
        print("\nBrak dużych luk - dzielę na równe części")
        n_batches = 5
        batch_size = len(full_df) // n_batches
        batch_boundaries = [i * batch_size for i in range(n_batches + 1)]
        batch_boundaries[-1] = len(full_df)
    
    # Analyze each batch
    print("\n2. CHARAKTERYSTYKA PARTII:")
    print("-"*40)
    
    batch_stats = []
    
    for i in range(len(batch_boundaries) - 1):
        start_idx = batch_boundaries[i]
        end_idx = batch_boundaries[i + 1]
        batch_df = full_df.iloc[start_idx:end_idx]
        
        # Calculate statistics
        stats = {
            'batch': i + 1,
            'start_id': batch_df['id'].min(),
            'end_id': batch_df['id'].max(),
            'n_records': len(batch_df),
            'n_train': (batch_df['source'] == 'train').sum(),
            'n_test': (batch_df['source'] == 'test').sum(),
            'e_ratio': (batch_df['Personality'] == 'Extrovert').sum() / len(batch_df),
            'null_ratio': batch_df[['Time_spent_Alone', 'Social_event_attendance', 
                                   'Friends_circle_size', 'Going_outside', 
                                   'Post_frequency']].isnull().sum().sum() / (len(batch_df) * 5)
        }
        
        # Feature distributions
        for col in ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']:
            if col in batch_df.columns:
                stats[f'{col}_mean'] = batch_df[col].mean()
                stats[f'{col}_std'] = batch_df[col].std()
        
        batch_stats.append(stats)
    
    batch_stats_df = pd.DataFrame(batch_stats)
    
    # Print batch characteristics
    for _, batch in batch_stats_df.iterrows():
        print(f"\nPartia {batch['batch']} (ID {batch['start_id']}-{batch['end_id']}):")
        print(f"  Liczba rekordów: {batch['n_records']} (train: {batch['n_train']}, test: {batch['n_test']})")
        print(f"  E-ratio: {batch['e_ratio']:.3f}")
        print(f"  Null-ratio: {batch['null_ratio']:.3f}")
        print(f"  Time_alone: {batch.get('Time_spent_Alone_mean', 0):.1f} ± {batch.get('Time_spent_Alone_std', 0):.1f}")
    
    # Visualize batch characteristics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # E-ratio by batch
    ax = axes[0, 0]
    bars = ax.bar(batch_stats_df['batch'], batch_stats_df['e_ratio'])
    ax.axhline(y=batch_stats_df['e_ratio'].mean(), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Batch')
    ax.set_ylabel('E-ratio')
    ax.set_title('Extrovert Ratio by Batch')
    ax.set_ylim(0.6, 0.85)
    
    # Color bars by deviation from mean
    mean_ratio = batch_stats_df['e_ratio'].mean()
    for bar, ratio in zip(bars, batch_stats_df['e_ratio']):
        if abs(ratio - mean_ratio) > 0.05:
            bar.set_color('red')
        else:
            bar.set_color('skyblue')
    
    # Null ratio by batch
    ax = axes[0, 1]
    ax.bar(batch_stats_df['batch'], batch_stats_df['null_ratio'])
    ax.set_xlabel('Batch')
    ax.set_ylabel('Null ratio')
    ax.set_title('Missing Data Ratio by Batch')
    
    # Time alone distribution
    ax = axes[1, 0]
    ax.errorbar(batch_stats_df['batch'], 
                batch_stats_df['Time_spent_Alone_mean'],
                yerr=batch_stats_df['Time_spent_Alone_std'],
                marker='o', capsize=5)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Time spent alone')
    ax.set_title('Time Alone Distribution by Batch')
    
    # ID ranges
    ax = axes[1, 1]
    for _, batch in batch_stats_df.iterrows():
        ax.barh(batch['batch'], 
                batch['end_id'] - batch['start_id'],
                left=batch['start_id'],
                height=0.5,
                alpha=0.7,
                color=plt.cm.tab10(batch['batch'] - 1))
    ax.set_ylabel('Batch')
    ax.set_xlabel('ID range')
    ax.set_title('ID Ranges by Batch')
    
    plt.suptitle('Data Batch Analysis - Potential Different Sources', fontsize=16)
    plt.tight_layout()
    
    output_path = WORKSPACE_DIR / "scripts/output/data_batch_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nWykres zapisany: {output_path}")
    plt.close()
    
    # Find suspicious boundaries
    print("\n3. PODEJRZANE GRANICE MIĘDZY PARTIAMI:")
    print("-"*40)
    
    # Look for sudden changes in E-ratio
    for i in range(len(batch_stats_df) - 1):
        current = batch_stats_df.iloc[i]
        next_batch = batch_stats_df.iloc[i + 1]
        
        e_ratio_change = abs(next_batch['e_ratio'] - current['e_ratio'])
        null_ratio_change = abs(next_batch['null_ratio'] - current['null_ratio'])
        
        if e_ratio_change > 0.05 or null_ratio_change > 0.05:
            print(f"\nGranica między Partią {current['batch']} a {next_batch['batch']}:")
            print(f"  ID range: {current['end_id']} → {next_batch['start_id']}")
            print(f"  E-ratio change: {current['e_ratio']:.3f} → {next_batch['e_ratio']:.3f} (Δ={e_ratio_change:.3f})")
            print(f"  Null-ratio change: {current['null_ratio']:.3f} → {next_batch['null_ratio']:.3f} (Δ={null_ratio_change:.3f})")
            
            # Find specific IDs near boundary
            boundary_ids = full_df[(full_df['id'] >= current['end_id'] - 10) & 
                                  (full_df['id'] <= next_batch['start_id'] + 10)]['id'].values
            print(f"  IDs near boundary: {boundary_ids[:10]}...")
    
    return batch_stats_df, full_df

def find_batch_anomalies(batch_stats_df, full_df):
    """Find records that don't match their batch characteristics"""
    print("\n" + "="*60)
    print("SZUKANIE ANOMALII W PARTIACH")
    print("="*60)
    
    anomalies = []
    
    # For each batch, find records that don't fit
    for _, batch in batch_stats_df.iterrows():
        batch_df = full_df[(full_df['id'] >= batch['start_id']) & 
                          (full_df['id'] <= batch['end_id'])]
        
        # If batch has high E-ratio, find Introverts
        if batch['e_ratio'] > 0.8:
            introverts = batch_df[batch_df['Personality'] == 'Introvert']
            for _, record in introverts.head(3).iterrows():
                anomalies.append({
                    'id': record['id'],
                    'personality': 'Introvert',
                    'batch': batch['batch'],
                    'reason': f"I in high-E batch ({batch['e_ratio']:.1%} E)",
                    'score': batch['e_ratio']
                })
        
        # If batch has low E-ratio, find Extroverts
        elif batch['e_ratio'] < 0.7:
            extroverts = batch_df[batch_df['Personality'] == 'Extrovert']
            for _, record in extroverts.head(3).iterrows():
                anomalies.append({
                    'id': record['id'],
                    'personality': 'Extrovert',
                    'batch': batch['batch'],
                    'reason': f"E in low-E batch ({batch['e_ratio']:.1%} E)",
                    'score': 1 - batch['e_ratio']
                })
    
    if anomalies:
        print("\nZnalezione anomalie:")
        for anom in sorted(anomalies, key=lambda x: x['score'], reverse=True)[:10]:
            print(f"ID {anom['id']} ({anom['personality']}): {anom['reason']}")
    
    return anomalies

if __name__ == "__main__":
    batch_stats_df, full_df = analyze_data_batches()
    anomalies = find_batch_anomalies(batch_stats_df, full_df)
    
    # Save results
    batch_stats_df.to_csv(WORKSPACE_DIR / "scripts/output/batch_statistics.csv", index=False)
    if anomalies:
        anomalies_df = pd.DataFrame(anomalies)
        anomalies_df.to_csv(WORKSPACE_DIR / "scripts/output/batch_anomalies.csv", index=False)
    
    print("\n" + "="*60)
    print("WNIOSKI:")
    print("="*60)
    print("1. Dane prawdopodobnie pochodzą z różnych źródeł/ankiet")
    print("2. Każda partia ma nieco inne charakterystyki")
    print("3. Błędy mogą występować na granicach partii")
    print("4. Niektóre rekordy nie pasują do swojej partii")