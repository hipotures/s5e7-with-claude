#!/usr/bin/env python3
"""
Tworzy pliki flip test dla ID: 18887, 20153, 23177
"""

import pandas as pd
import shutil

def create_flip_test(baseline_file, output_file, flip_id, new_value):
    """Tworzy plik z pojedynczym flipem"""
    
    # Wczytaj baseline
    df = pd.read_csv(baseline_file)
    
    # Znajdź właściwą kolumnę
    if 'personality' in df.columns:
        target_col = 'personality'
    elif 'Personality' in df.columns:
        target_col = 'Personality'
    else:
        raise ValueError("Nie znaleziono kolumny personality")
    
    # Zmień wartość dla określonego ID
    df.loc[df['id'] == flip_id, target_col] = new_value
    
    # Zapisz
    df.to_csv(output_file, index=False)
    print(f"Utworzono: {output_file}")
    
    # Weryfikacja
    changed_rows = df[df['id'] == flip_id]
    if len(changed_rows) > 0:
        print(f"  ID {flip_id}: {changed_rows[target_col].values[0]}")
    else:
        print(f"  UWAGA: Nie znaleziono ID {flip_id}!")

def main():
    baseline_file = 'subm-0.96950-20250704_121621-xgb-381482788433-148.csv'
    
    # Sprawdź strukturę baseline
    df = pd.read_csv(baseline_file)
    print(f"Kolumny w baseline: {df.columns.tolist()}")
    print(f"Liczba wierszy: {len(df)}")
    
    # Testy do utworzenia
    tests = [
        (18887, 'Extrovert', 'flip_TEST_1_I2E_id_18887.csv'),
        (20153, 'Extrovert', 'flip_TEST_2_I2E_id_20153.csv'),
        (23177, 'Introvert', 'flip_TEST_3_E2I_id_23177.csv')
    ]
    
    print("\nTworzę pliki flip test...")
    
    for flip_id, new_value, output_file in tests:
        create_flip_test(baseline_file, output_file, flip_id, new_value)

if __name__ == "__main__":
    main()