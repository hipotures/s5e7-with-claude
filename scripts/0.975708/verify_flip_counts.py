#!/usr/bin/env python3
"""
Weryfikuje liczbę różnic (flipów) między plikiem wzorcowym a plikami w katalogach minus-1 i minus-2
"""

import pandas as pd
import os

def load_submission(filepath):
    """Wczytuje plik submission CSV"""
    df = pd.read_csv(filepath)
    
    # Znajdź kolumnę z przewidywaniami
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'Id'})
    
    target_col = None
    for col in ['Personality', 'personality', 'target', 'Target', 'prediction']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"Nie znaleziono kolumny z przewidywaniami w pliku: {filepath}")
    
    # Konwertuj wartości na jednolity format (E/I)
    result = {}
    for idx, val in df.set_index('Id')[target_col].items():
        if val in ['Extrovert', 'extrovert', 1, '1']:
            result[idx] = 'E'
        elif val in ['Introvert', 'introvert', 0, '0']:
            result[idx] = 'I'
        else:
            result[idx] = str(val)
    
    return result

def count_differences(baseline_data, file_data):
    """Liczy różnice między dwoma plikami"""
    differences = []
    
    all_ids = set(baseline_data.keys()) | set(file_data.keys())
    
    for id_val in sorted(all_ids):
        baseline_val = baseline_data.get(id_val)
        file_val = file_data.get(id_val)
        
        if baseline_val != file_val:
            if baseline_val is None:
                diff_str = f"{id_val}:?->{file_val}"
            elif file_val is None:
                diff_str = f"{id_val}:{baseline_val}->?"
            else:
                diff_str = f"{id_val}:{baseline_val}->{file_val}"
            differences.append(diff_str)
    
    return differences

def main():
    # Plik wzorcowy
    baseline_file = 'subm-0.96950-20250704_121621-xgb-381482788433-148.csv'
    
    print(f"Plik wzorcowy: {baseline_file}")
    print("=" * 80)
    
    # Wczytaj baseline
    baseline_data = load_submission(baseline_file)
    
    # Sprawdź pliki w minus-1
    print("\nKatalog minus-1 (oczekiwane: 1 różnica):")
    print("-" * 40)
    
    minus1_files = [f for f in os.listdir('minus-1') if f.endswith('.csv')]
    for filename in sorted(minus1_files):
        filepath = os.path.join('minus-1', filename)
        try:
            file_data = load_submission(filepath)
            differences = count_differences(baseline_data, file_data)
            
            print(f"\n{filename}:")
            print(f"  Liczba różnic: {len(differences)}")
            if len(differences) <= 5:  # Wypisz różnice tylko jeśli jest ich mało
                for diff in differences:
                    print(f"    {diff}")
            
            if len(differences) != 1:
                print(f"  ⚠️ UWAGA: Oczekiwano 1 różnicy, znaleziono {len(differences)}")
                
        except Exception as e:
            print(f"  BŁĄD: {e}")
    
    # Sprawdź pliki w minus-2
    print("\n\nKatalog minus-2 (oczekiwane: 2 różnice):")
    print("-" * 40)
    
    minus2_files = [f for f in os.listdir('minus-2') if f.endswith('.csv')]
    for filename in sorted(minus2_files):
        filepath = os.path.join('minus-2', filename)
        try:
            file_data = load_submission(filepath)
            differences = count_differences(baseline_data, file_data)
            
            print(f"\n{filename}:")
            print(f"  Liczba różnic: {len(differences)}")
            if len(differences) <= 5:  # Wypisz różnice tylko jeśli jest ich mało
                for diff in differences:
                    print(f"    {diff}")
            else:
                print(f"  (za dużo różnic, aby wypisać wszystkie)")
            
            if len(differences) != 2:
                print(f"  ⚠️ UWAGA: Oczekiwano 2 różnic, znaleziono {len(differences)}")
                
        except Exception as e:
            print(f"  BŁĄD: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()