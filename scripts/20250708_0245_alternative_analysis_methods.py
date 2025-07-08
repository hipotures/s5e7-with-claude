#!/usr/bin/env python3
"""
Alternative analysis methods to find mislabeled records
"""

import pandas as pd
import numpy as np
from pathlib import Path

def brainstorm_new_approaches():
    print("="*60)
    print("ALTERNATYWNE METODY ANALIZY")
    print("="*60)
    
    print("\n1. ANALIZA CZASOWA / SEKWENCYJNA:")
    print("-"*40)
    print("• Wzorce co N rekordów (np. co 100, 500, 1000)")
    print("• Cykliczne błędy (np. co 7 jak dni tygodnia)")
    print("• Zmiana rozkładu w czasie (drift)")
    print("• Punkty przełamania (change points)")
    
    print("\n2. ANALIZA RELACJI MIĘDZY CECHAMI:")
    print("-"*40)
    print("• Korelacje niemożliwe (np. max Friends + min Social)")
    print("• Reguły logiczne MBTI (np. I musi mieć Drained=Yes)")
    print("• Stosunek cech (np. Post_frequency/Friends_circle)")
    print("• Interakcje 3+ cech razem")
    
    print("\n3. ANALIZA ENTROPII / INFORMACJI:")
    print("-"*40)
    print("• Rekordy z minimalną entropią (wszystko 0 lub max)")
    print("• Rekordy z maksymalną entropią (chaos)")
    print("• Najbardziej 'informatywne' rekordy")
    print("• Anomalie w przestrzeni informatycznej")
    
    print("\n4. ANALIZA GRAFOWA / SIECIOWA:")
    print("-"*40)
    print("• Rekordy bez 'sąsiadów' (isolated nodes)")
    print("• Mosty między klastrami")
    print("• Huby - rekordy połączone z wieloma innymi")
    print("• Cykle w grafie podobieństwa")
    
    print("\n5. ANALIZA JĘZYKOWA / SEMANTYCZNA:")
    print("-"*40)
    print("• Może ID zawierają wzorzec? (końcówki, sumy cyfr)")
    print("• Hash ID → cechy (pseudolosowość)")
    print("• Alfabetyczna kolejność?")
    print("• Numerologia ID (np. podzielne przez X)")
    
    print("\n6. ANALIZA BAYESOWSKA:")
    print("-"*40)
    print("• P(E|cechy) vs P(I|cechy)")
    print("• Największe rozbieżności prior/posterior")
    print("• Rekordy łamiące założenia Bayesa")
    print("• Paradoksy warunkowania")
    
    print("\n7. TEORIA GIER / ADVERSARIAL:")
    print("-"*40)
    print("• Rekordy które 'oszukują' model")
    print("• Adversarial examples")
    print("• Worst-case dla każdego klasyfikatora")
    print("• Nash equilibrium między cechami")
    
    print("\n8. ANALIZA KOMPRESJI:")
    print("-"*40)
    print("• Rekordy nie dające się skompresować")
    print("• Kolmogorov complexity")
    print("• Minimum Description Length")
    print("• Anomalie w kompresji")

def implement_selected_method():
    """Implement most promising method - Sequential/Temporal analysis"""
    print("\n\n" + "="*60)
    print("IMPLEMENTACJA: ANALIZA SEKWENCYJNA")
    print("="*60)
    
    # Load data
    DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
    WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
    ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"
    
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Add sequence number based on ID
    test_df['seq_num'] = test_df['id'] - test_df['id'].min()
    
    print("\n1. ANALIZA CYKLICZNA:")
    print("-"*40)
    
    # Check patterns every N records
    patterns = {}
    for period in [5, 7, 10, 50, 100, 137, 255]:  # Including primes
        # Count personality distribution in each position
        positions = {}
        for i in range(period):
            mask = test_df['seq_num'] % period == i
            subset_ids = test_df[mask]['id'].values
            
            # Get labels for this subset
            labels = []
            for sid in subset_ids:
                idx = original_df[original_df['id'] == sid].index
                if len(idx) > 0:
                    labels.append(original_df.loc[idx[0], 'Personality'])
            
            if labels:
                e_ratio = labels.count('Extrovert') / len(labels)
                positions[i] = e_ratio
        
        # Find outliers
        ratios = list(positions.values())
        if ratios:
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            
            for pos, ratio in positions.items():
                if abs(ratio - mean_ratio) > 2 * std_ratio:
                    patterns[f"period_{period}_pos_{pos}"] = {
                        'period': period,
                        'position': pos,
                        'e_ratio': ratio,
                        'anomaly': ratio - mean_ratio
                    }
    
    if patterns:
        print("Znalezione anomalie cykliczne:")
        for key, pattern in sorted(patterns.items(), key=lambda x: abs(x[1]['anomaly']), reverse=True)[:5]:
            print(f"Co {pattern['period']} rekordów, pozycja {pattern['position']}: "
                  f"E_ratio={pattern['e_ratio']:.3f} (anomalia: {pattern['anomaly']:+.3f})")
    
    print("\n2. ANALIZA DRYFTU:")
    print("-"*40)
    
    # Split data into chunks and check distribution changes
    chunk_size = 500
    n_chunks = len(test_df) // chunk_size
    
    chunk_stats = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_ids = test_df.iloc[start_idx:end_idx]['id'].values
        
        # Get personality distribution
        e_count = 0
        total = 0
        for cid in chunk_ids:
            idx = original_df[original_df['id'] == cid].index
            if len(idx) > 0:
                if original_df.loc[idx[0], 'Personality'] == 'Extrovert':
                    e_count += 1
                total += 1
        
        if total > 0:
            chunk_stats.append({
                'chunk': i,
                'start_id': chunk_ids[0],
                'end_id': chunk_ids[-1],
                'e_ratio': e_count / total
            })
    
    # Find chunks with unusual distribution
    e_ratios = [c['e_ratio'] for c in chunk_stats]
    mean_ratio = np.mean(e_ratios)
    std_ratio = np.std(e_ratios)
    
    print(f"Średni E_ratio: {mean_ratio:.3f} (std: {std_ratio:.3f})")
    print("\nChunks z anomalią:")
    for chunk in chunk_stats:
        if abs(chunk['e_ratio'] - mean_ratio) > 1.5 * std_ratio:
            print(f"Chunk {chunk['chunk']} (ID {chunk['start_id']}-{chunk['end_id']}): "
                  f"E_ratio={chunk['e_ratio']:.3f}")
    
    print("\n3. ANALIZA LOKALNEJ SPÓJNOŚCI:")
    print("-"*40)
    
    # Check if neighbors have similar labels
    inconsistencies = []
    
    for i in range(1, len(test_df) - 1):
        current_id = test_df.iloc[i]['id']
        prev_id = test_df.iloc[i-1]['id']
        next_id = test_df.iloc[i+1]['id']
        
        # Get labels
        labels = {}
        for name, rid in [('prev', prev_id), ('curr', current_id), ('next', next_id)]:
            idx = original_df[original_df['id'] == rid].index
            if len(idx) > 0:
                labels[name] = original_df.loc[idx[0], 'Personality']
        
        # Check if current is different from both neighbors
        if len(labels) == 3:
            if labels['curr'] != labels['prev'] and labels['curr'] != labels['next'] and labels['prev'] == labels['next']:
                inconsistencies.append({
                    'id': current_id,
                    'label': labels['curr'],
                    'neighbors': labels['prev']
                })
    
    print(f"Znaleziono {len(inconsistencies)} rekordów różniących się od obu sąsiadów")
    if inconsistencies:
        print("Pierwsze 5:")
        for inc in inconsistencies[:5]:
            print(f"ID {inc['id']}: {inc['label']} (sąsiedzi: {inc['neighbors']})")
    
    return patterns, chunk_stats, inconsistencies

def create_sequential_flip_files():
    """Create flip files based on sequential analysis"""
    print("\n" + "="*60)
    print("TWORZENIE PLIKÓW NA PODSTAWIE ANALIZY SEKWENCYJNEJ")
    print("="*60)
    
    # Implement the actual file creation here
    # For now, just show the concept
    
    print("\nProponowane flipy:")
    print("1. Rekordy z anomalii cyklicznych")
    print("2. Rekordy z chunków o nietypowym rozkładzie")
    print("3. Rekordy różniące się od sąsiadów")
    print("4. Punkty przełamania trendów")
    print("5. Lokalne ekstrema w sekwencji")

if __name__ == "__main__":
    brainstorm_new_approaches()
    patterns, chunks, inconsistencies = implement_selected_method()
    create_sequential_flip_files()