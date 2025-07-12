#!/usr/bin/env python3
"""
Explain why diagonal stripes appear in only one PCA cluster
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")

def explain_stripes():
    """Explain the diagonal stripe pattern in PCA"""
    
    print("="*60)
    print("WYJAŚNIENIE SKOŚNYCH PASÓW W PCA")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    
    # Numeric features - all are integers!
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    print("\n1. DLACZEGO POWSTAJĄ SKOŚNE PASY?")
    print("-" * 40)
    print("Wszystkie zmienne numeryczne to LICZBY CAŁKOWITE:")
    for feat in numeric_features:
        values = train_df[feat].dropna().unique()
        print(f"  - {feat}: {min(values):.0f} do {max(values):.0f} (tylko całkowite)")
    
    print("\nTo tworzy 'siatkę' możliwych kombinacji w przestrzeni cech.")
    print("PCA przekształca tę siatkę, ale zachowuje regularną strukturę.")
    
    # Separate by personality
    introverts = train_df[train_df['Personality'] == 'Introvert']
    extroverts = train_df[train_df['Personality'] == 'Extrovert']
    
    print("\n2. DLACZEGO TYLKO JEDEN KLASTER MA PASY?")
    print("-" * 40)
    
    # Analyze feature distributions for each personality type
    print("\nPorównanie rozkładów cech:")
    
    for feat in numeric_features:
        intro_mean = introverts[feat].mean()
        extro_mean = extroverts[feat].mean()
        diff = abs(intro_mean - extro_mean)
        
        print(f"\n{feat}:")
        print(f"  Introwertycy: średnia = {intro_mean:.2f}")
        print(f"  Ekstrawertycy: średnia = {extro_mean:.2f}")
        print(f"  Różnica: {diff:.2f} {'⭐' if diff > 3 else ''}")
    
    # Count unique combinations
    print("\n3. ANALIZA UNIKALNYCH KOMBINACJI:")
    print("-" * 40)
    
    intro_combos = introverts[numeric_features].value_counts()
    extro_combos = extroverts[numeric_features].value_counts()
    
    print(f"Introwertycy: {len(intro_combos)} unikalnych kombinacji z {len(introverts)} próbek")
    print(f"Ekstrawertycy: {len(extro_combos)} unikalnych kombinacji z {len(extroverts)} próbek")
    
    # Find most common patterns
    print("\n4. NAJPOPULARNIEJSZE WZORCE:")
    print("-" * 40)
    
    print("\nIntrowertycy - TOP 5 kombinacji:")
    for combo, count in intro_combos.head(5).items():
        print(f"  {dict(zip(numeric_features, combo))} -> {count} wystąpień")
    
    print("\nEkstrawertycy - TOP 5 kombinacji:")
    for combo, count in extro_combos.head(5).items():
        print(f"  {dict(zip(numeric_features, combo))} -> {count} wystąpień")
    
    # Explain the visual pattern
    print("\n5. WYJAŚNIENIE WIZUALNE:")
    print("-" * 40)
    print("""
Skośne pasy powstają, ponieważ:

1. KLASTER INTROWERTYKÓW (ten z pasami):
   - Wysokie wartości: Time_alone (7-11), niskie: Social_events (0-3)
   - Te cechy są UJEMNIE skorelowane (więcej czasu sam = mniej imprez)
   - Dyskretne wartości tworzą regularne "stopnie" w PCA
   
2. KLASTER EKSTRAWERTYKÓW (bez pasów):
   - Bardziej zróżnicowane kombinacje cech
   - Mniej ekstremalne wartości
   - Rozmycie granic przez większą różnorodność

3. PCA podkreśla główną oś zmienności:
   - PC1 głównie różnicuje Intro vs Extro
   - PC2 wychwytuje subtelniejsze wzorce
   - Dyskretne wartości + silna korelacja = widoczne pasy
""")
    
    # Check correlations
    print("\n6. KORELACJE MIĘDZY CECHAMI:")
    print("-" * 40)
    
    intro_corr = introverts[numeric_features].corr()
    extro_corr = extroverts[numeric_features].corr()
    
    # Find strongest correlations
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            feat1, feat2 = numeric_features[i], numeric_features[j]
            intro_r = intro_corr.loc[feat1, feat2]
            extro_r = extro_corr.loc[feat1, feat2]
            
            if abs(intro_r) > 0.5 or abs(extro_r) > 0.5:
                print(f"{feat1} vs {feat2}:")
                print(f"  Introwertycy: r = {intro_r:.3f} {'⭐' if abs(intro_r) > 0.7 else ''}")
                print(f"  Ekstrawertycy: r = {extro_r:.3f}")
    
    print("\n" + "="*60)
    print("PODSUMOWANIE:")
    print("="*60)
    print("""
Skośne pasy w klastrze introwertyków powstają przez:
1. Wszystkie cechy to liczby całkowite (dyskretna siatka)
2. Silne korelacje między cechami (np. Time_alone vs Social_events)
3. Introwertycy mają bardziej ekstremalne i regularne wzorce
4. PCA zachowuje tę regularną strukturę jako widoczne "stopnie"

Klaster ekstrawertyków nie ma pasów, bo:
- Bardziej zróżnicowane kombinacje cech
- Słabsze korelacje
- Mniej ekstremalne wartości
""")

if __name__ == "__main__":
    explain_stripes()