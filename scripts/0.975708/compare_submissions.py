#!/usr/bin/env python3
"""
Porównuje wszystkie pliki submission CSV z wynikiem 0.975708
Generuje raport różnic między plikami
"""

import pandas as pd
import hashlib
import os
import json
from datetime import datetime
from itertools import combinations

def calculate_md5(filepath):
    """Oblicza MD5 hash pliku"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_submission(filepath):
    """Wczytuje plik submission CSV"""
    df = pd.read_csv(filepath)
    # Zakładamy że kolumny to 'Id' i 'Personality' lub 'target'
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'Id'})
    
    # Znajdź kolumnę z przewidywaniami
    target_col = None
    for col in ['Personality', 'target', 'Target', 'prediction']:
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

def compare_submissions(file1_data, file2_data):
    """Porównuje dwa pliki submission i zwraca różnice"""
    differences = []
    
    # Sprawdź różnice
    all_ids = set(file1_data.keys()) | set(file2_data.keys())
    
    for id_val in sorted(all_ids):
        val1 = file1_data.get(id_val)
        val2 = file2_data.get(id_val)
        
        if val1 != val2:
            if val1 is None:
                differences.append(f"{id_val}:?->{val2}")
            elif val2 is None:
                differences.append(f"{id_val}:{val1}->?")
            else:
                differences.append(f"{id_val}:{val1}->{val2}")
    
    return differences

def main():
    # Ścieżka do katalogu z plikami
    directory = "/home/xai/WORKSPACE/scripts/0.975708"
    
    # Wczytaj listę plików
    print("Wczytuję listę plików...")
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') and f != 'comp-0.975708.csv']
    csv_files.sort()
    
    print(f"Znaleziono {len(csv_files)} plików CSV")
    
    # Przygotuj słownik z danymi
    file_data = {}
    file_hashes = {}
    
    print("\nWczytuję pliki i obliczam MD5...")
    for i, filename in enumerate(csv_files):
        filepath = os.path.join(directory, filename)
        try:
            file_hashes[filename] = calculate_md5(filepath)
            file_data[filename] = load_submission(filepath)
            print(f"  [{i+1}/{len(csv_files)}] {filename} - MD5: {file_hashes[filename][:8]}...")
        except Exception as e:
            print(f"  BŁĄD przy {filename}: {e}")
            file_hashes[filename] = "ERROR"
            file_data[filename] = {}
    
    # Porównaj każdy z każdym
    print("\nPorównuję pliki...")
    results = []
    
    total_comparisons = len(list(combinations(range(len(csv_files)), 2)))
    comparison_count = 0
    
    for i in range(len(csv_files)):
        print(f"\nPorównuję plik {i+1}/{len(csv_files)}: {csv_files[i][:50]}...")
        
        for j in range(i+1, len(csv_files)):
            comparison_count += 1
            
            file1 = csv_files[i]
            file2 = csv_files[j]
            
            if file_data.get(file1) and file_data.get(file2):
                differences = compare_submissions(file_data[file1], file_data[file2])
                
                if differences:  # Zapisuj tylko jeśli są różnice
                    results.append({
                        'file1': file1,
                        'file2': file2,
                        'md5_1': file_hashes[file1],
                        'md5_2': file_hashes[file2],
                        'differences': ','.join(differences),
                        'diff_count': len(differences),
                        'e_to_i': len([d for d in differences if 'E->I' in d]),
                        'i_to_e': len([d for d in differences if 'I->E' in d])
                    })
    
    print(f"\nZnaleziono {len(results)} par plików z różnicami")
    
    # Zapisz wyniki w różnych formatach
    output_dir = os.path.join(directory, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Format 1: CSV podstawowy
    with open(os.path.join(output_dir, 'comp-0.975708.csv'), 'w') as f:
        f.write('md5_1,md5_2,differences\n')
        for r in results:
            f.write(f"{r['md5_1']},{r['md5_2']},\"{r['differences']}\"\n")
    
    # Format 2: CSV rozszerzony (łatwiejszy do analizy)
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'comp-0.975708-extended.csv'), index=False)
    
    # Format 3: JSON (najłatwiejszy do dalszej analizy)
    with open(os.path.join(output_dir, 'comp-0.975708.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Podsumowanie
    print("\n=== PODSUMOWANIE ===")
    print(f"Porównano: {total_comparisons} par plików")
    print(f"Znaleziono różnice w: {len(results)} parach")
    if results:
        avg_diff = sum(r['diff_count'] for r in results) / len(results)
        print(f"Średnia liczba różnic: {avg_diff:.1f}")
        print(f"Maksymalna liczba różnic: {max(r['diff_count'] for r in results)}")
    
    print("\nWyniki zapisano w katalogu output/:")
    print("  - output/comp-0.975708.csv (format podstawowy)")
    print("  - output/comp-0.975708-extended.csv (format rozszerzony)")
    print("  - output/comp-0.975708.json (format JSON)")

if __name__ == "__main__":
    main()