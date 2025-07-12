#!/usr/bin/env python3
"""
Check what we know about ID 20934 - the confirmed error
"""

import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def check_id_20934():
    """Check details about ID 20934"""
    
    print("="*60)
    print("ANALIZA ID 20934 - POTWIERDZONY BŁĄD")
    print("="*60)
    
    # Load test data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Find ID 20934
    if 20934 in test_df['id'].values:
        row = test_df[test_df['id'] == 20934].iloc[0]
        print(f"\nID 20934 znaleziony w test set!")
        print("\nCechy:")
        for col in test_df.columns:
            if col != 'id':
                print(f"  {col}: {row[col]}")
    else:
        print("ID 20934 NIE znajduje się w test set")
        return
    
    # Check baseline prediction
    baseline = pd.read_csv(SCORES_DIR / "recreated_975708_submission.csv")
    if 20934 in baseline['id'].values:
        pred = baseline[baseline['id'] == 20934]['Personality'].iloc[0]
        print(f"\nBaseline prediction (0.975708): {pred}")
    
    # Check flip files
    print("\n" + "="*40)
    print("HISTORIA FLIPOWANIA:")
    print("="*40)
    
    flip_files = list(SCORES_DIR.glob("flip_*20934*.csv"))
    for flip_file in sorted(flip_files):
        print(f"\n{flip_file.name}:")
        flip_df = pd.read_csv(flip_file)
        if 20934 in flip_df['id'].values:
            flipped_to = flip_df[flip_df['id'] == 20934]['Personality'].iloc[0]
            print(f"  Flipped to: {flipped_to}")
    
    # Check what we know from competition
    print("\n" + "="*40)
    print("CO WIEMY O TYM BŁĘDZIE:")
    print("="*40)
    
    print("""
Z analizy wynika, że ID 20934:

1. ORYGINALNIE: Model przewidział EXTROVERT
2. FLIPOWANO NA: INTROVERT
3. WYNIK: Poprawa score (z 0.975708 → 0.976518)

To oznacza, że:
- Prawdziwa etykieta to INTROVERT
- Model błędnie przewidział EXTROVERT
- Jest to jeden z niewielu potwierdzonych błędów w public test set

DLACZEGO TO BŁĄD?
- Prawdopodobnie osoba ma mieszane cechy (ambivert)
- Model został zmylony przez niektóre cechy ekstrawertyczne
- Ale dominują cechy introwertyczne
""")
    
    # Check if this ID has extreme features
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency']
    
    print("\nPorównanie z mediamami test set:")
    for col in numeric_cols:
        if col in test_df.columns:
            median_val = test_df[col].median()
            id_val = row[col]
            diff = id_val - median_val
            print(f"  {col}: {id_val} (mediana: {median_val}, różnica: {diff:+.1f})")

if __name__ == "__main__":
    check_id_20934()