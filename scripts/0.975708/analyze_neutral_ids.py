#!/usr/bin/env python3
"""
Analizuje które ID są neutralne (nie zmieniają wyniku) na podstawie plików z wynikiem 0.975708
"""

import pandas as pd
import os
from collections import defaultdict

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

def find_neutral_ids():
    """Znajduje ID które są neutralne (pojawiają się w plikach z wynikiem 0.975708)"""
    
    # Plik wzorcowy
    baseline_file = 'subm-0.96950-20250704_121621-xgb-381482788433-148.csv'
    baseline_data = load_submission(baseline_file)
    
    # Zbierz wszystkie pliki CSV z wynikiem 0.975708 (bez baseline i plików wynikowych)
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') 
                 and f != baseline_file 
                 and not f.startswith('comp-')
                 and 'minus' not in f]
    
    # Przechowuj informacje o zmianach
    id_changes = defaultdict(list)
    
    # Analizuj każdy plik
    for filename in csv_files:
        try:
            file_data = load_submission(filename)
            
            # Znajdź różnice
            for id_val in sorted(set(baseline_data.keys()) | set(file_data.keys())):
                baseline_val = baseline_data.get(id_val)
                file_val = file_data.get(id_val)
                
                if baseline_val != file_val and baseline_val is not None and file_val is not None:
                    change = f"{baseline_val}->{file_val}"
                    id_changes[id_val].append((filename, change))
                    
        except Exception as e:
            print(f"BŁĄD przy {filename}: {e}")
    
    # Identyfikuj neutralne ID (te które się zmieniają w plikach z wynikiem 0.975708)
    neutral_ids = set()
    for id_val, changes in id_changes.items():
        if len(changes) > 0:
            neutral_ids.add(id_val)
    
    return neutral_ids, id_changes, baseline_data

def analyze_minus_files(neutral_ids, baseline_data):
    """Analizuje pliki w minus-1 i minus-2 pod kątem neutralnych ID"""
    
    results = {}
    
    for dir_name in ['minus-1', 'minus-2']:
        if not os.path.exists(dir_name):
            continue
            
        print(f"\nAnalizuję katalog {dir_name}:")
        print("-" * 50)
        
        files = [f for f in os.listdir(dir_name) if f.endswith('.csv')]
        
        for filename in sorted(files):
            filepath = os.path.join(dir_name, filename)
            try:
                file_data = load_submission(filepath)
                
                # Znajdź różnice
                all_changes = []
                neutral_changes = []
                impactful_changes = []
                
                for id_val in sorted(set(baseline_data.keys()) | set(file_data.keys())):
                    baseline_val = baseline_data.get(id_val)
                    file_val = file_data.get(id_val)
                    
                    if baseline_val != file_val and baseline_val is not None and file_val is not None:
                        change = f"{id_val}:{baseline_val}->{file_val}"
                        all_changes.append(change)
                        
                        if id_val in neutral_ids:
                            neutral_changes.append(change)
                        else:
                            impactful_changes.append(change)
                
                results[filename] = {
                    'total': len(all_changes),
                    'neutral': len(neutral_changes),
                    'impactful': len(impactful_changes),
                    'neutral_changes': neutral_changes,
                    'impactful_changes': impactful_changes
                }
                
                print(f"\n{filename}:")
                print(f"  Wszystkich zmian: {len(all_changes)}")
                print(f"  Zmian neutralnych: {len(neutral_changes)}")
                print(f"  Zmian wpływających: {len(impactful_changes)}")
                
                if len(impactful_changes) <= 5:
                    print(f"  Zmiany wpływające:")
                    for change in impactful_changes:
                        print(f"    {change}")
                
                if len(neutral_changes) > 0 and len(neutral_changes) <= 5:
                    print(f"  Zmiany neutralne:")
                    for change in neutral_changes:
                        print(f"    {change}")
                        
            except Exception as e:
                print(f"  BŁĄD: {e}")
    
    return results

def main():
    print("Szukam neutralnych ID (na podstawie plików z wynikiem 0.975708)...")
    
    neutral_ids, id_changes, baseline_data = find_neutral_ids()
    
    print(f"\nZnaleziono {len(neutral_ids)} potencjalnie neutralnych ID")
    
    # Pokaż najczęściej zmieniane neutralne ID
    id_frequency = [(id_val, len(changes)) for id_val, changes in id_changes.items()]
    id_frequency.sort(key=lambda x: x[1], reverse=True)
    
    print("\nNajczęściej zmieniane ID (top 10):")
    for id_val, count in id_frequency[:10]:
        baseline_val = baseline_data.get(id_val, '?')
        print(f"  ID {id_val}: {count} zmian [baseline: {baseline_val}]")
    
    # Analizuj pliki w minus-1 i minus-2
    results = analyze_minus_files(neutral_ids, baseline_data)
    
    # Podsumowanie
    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    
    for dir_name in ['minus-1', 'minus-2']:
        files_in_dir = [f for f in results.keys() if f.startswith(dir_name.replace('-', ''))]
        if files_in_dir:
            print(f"\n{dir_name}:")
            for filename in files_in_dir:
                r = results[filename]
                print(f"  {filename}: {r['impactful']} wpływających, {r['neutral']} neutralnych")

if __name__ == "__main__":
    main()