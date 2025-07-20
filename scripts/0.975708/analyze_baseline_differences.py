#!/usr/bin/env python3
"""
Analizuje różnice wszystkich plików względem pliku wzorcowego (baseline).
Tworzy szczegółową listę wszystkich zmian.
"""

import pandas as pd
import json
import os
from collections import defaultdict, Counter

def load_submission(filepath):
    """Wczytuje plik submission CSV"""
    df = pd.read_csv(filepath)
    
    # Znajdź kolumnę z przewidywaniami
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'Id'})
    
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

def analyze_baseline_differences():
    """Analizuje różnice względem pliku wzorcowego"""
    
    # Plik wzorcowy
    baseline_file = 'subm-0.96950-20250704_121621-xgb-381482788433-148.csv'
    baseline_md5 = 'c4c8721683a44601b6e423462ee7db98'
    
    print(f"Plik wzorcowy: {baseline_file}")
    print(f"MD5 wzorca: {baseline_md5}")
    print("-" * 80)
    
    # Wczytaj baseline
    baseline_data = load_submission(baseline_file)
    
    # Zbierz wszystkie pliki CSV (bez baseline i plików wynikowych)
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') 
                 and f != baseline_file 
                 and not f.startswith('comp-')]
    csv_files.sort()
    
    print(f"\nZnaleziono {len(csv_files)} plików do porównania z wzorcem\n")
    
    # Przechowuj wyniki
    all_differences = []
    id_change_counter = defaultdict(lambda: {'E->I': 0, 'I->E': 0})
    
    # Porównaj każdy plik z baseline
    for i, filename in enumerate(csv_files):
        print(f"[{i+1}/{len(csv_files)}] Analizuję: {filename}")
        
        try:
            file_data = load_submission(filename)
            
            # Znajdź różnice
            differences = []
            for id_val in sorted(set(baseline_data.keys()) | set(file_data.keys())):
                baseline_val = baseline_data.get(id_val)
                file_val = file_data.get(id_val)
                
                if baseline_val != file_val:
                    if baseline_val is None:
                        diff_str = f"{id_val}:?->{file_val}"
                    elif file_val is None:
                        diff_str = f"{id_val}:{baseline_val}->?"
                    else:
                        diff_str = f"{id_val}:{baseline_val}->{file_val}"
                        # Zlicz kierunek zmian
                        if baseline_val == 'E' and file_val == 'I':
                            id_change_counter[id_val]['E->I'] += 1
                        elif baseline_val == 'I' and file_val == 'E':
                            id_change_counter[id_val]['I->E'] += 1
                    
                    differences.append(diff_str)
            
            if differences:
                all_differences.append({
                    'filename': filename,
                    'num_differences': len(differences),
                    'differences': differences
                })
                
        except Exception as e:
            print(f"  BŁĄD: {e}")
    
    # Zapisz szczegółowe wyniki
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Pełna lista różnic względem baseline
    with open(os.path.join(output_dir, 'baseline_differences_detailed.json'), 'w') as f:
        json.dump(all_differences, f, indent=2)
    
    # 2. Podsumowanie różnic w CSV
    summary_data = []
    for item in all_differences:
        summary_data.append({
            'filename': item['filename'],
            'num_differences': item['num_differences'],
            'differences': ','.join(item['differences'])
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, 'baseline_differences_summary.csv'), index=False)
    
    # 3. Analiza częstości zmian ID
    id_stats = []
    for id_val, changes in id_change_counter.items():
        id_stats.append({
            'id': id_val,
            'total_changes': changes['E->I'] + changes['I->E'],
            'E_to_I': changes['E->I'],
            'I_to_E': changes['I->E'],
            'baseline_value': baseline_data.get(id_val, '?')
        })
    
    df_id_stats = pd.DataFrame(id_stats)
    df_id_stats = df_id_stats.sort_values('total_changes', ascending=False)
    df_id_stats.to_csv(os.path.join(output_dir, 'id_change_frequency.csv'), index=False)
    
    # 4. Ranking plików wg liczby różnic
    df_ranking = df_summary[['filename', 'num_differences']].sort_values('num_differences')
    df_ranking.to_csv(os.path.join(output_dir, 'files_ranked_by_differences.csv'), index=False)
    
    # Wyświetl podsumowanie
    print("\n" + "=" * 80)
    print("PODSUMOWANIE ANALIZY")
    print("=" * 80)
    
    print(f"\nPrzeanalizowano {len(all_differences)} plików")
    
    if all_differences:
        avg_diff = sum(item['num_differences'] for item in all_differences) / len(all_differences)
        min_diff = min(item['num_differences'] for item in all_differences)
        max_diff = max(item['num_differences'] for item in all_differences)
        
        print(f"Średnia liczba różnic od baseline: {avg_diff:.1f}")
        print(f"Minimalna liczba różnic: {min_diff}")
        print(f"Maksymalna liczba różnic: {max_diff}")
        
        print("\nPliki najbliższe baseline (top 5):")
        for _, row in df_ranking.head(5).iterrows():
            print(f"  {row['filename']}: {row['num_differences']} różnic")
        
        print("\nPliki najdalsze od baseline (top 5):")
        for _, row in df_ranking.tail(5).iterrows():
            print(f"  {row['filename']}: {row['num_differences']} różnic")
        
        print("\nNajczęściej zmieniane ID (top 10):")
        for _, row in df_id_stats.head(10).iterrows():
            print(f"  ID {row['id']}: {row['total_changes']} zmian "
                  f"(E->I: {row['E_to_I']}, I->E: {row['I_to_E']}) "
                  f"[baseline: {row['baseline_value']}]")
    
    print("\n" + "=" * 80)
    print("Wyniki zapisano w katalogu output/:")
    print("  - baseline_differences_detailed.json - pełne szczegóły wszystkich różnic")
    print("  - baseline_differences_summary.csv - podsumowanie różnic")
    print("  - id_change_frequency.csv - częstość zmian dla każdego ID")
    print("  - files_ranked_by_differences.csv - ranking plików wg liczby różnic")

if __name__ == "__main__":
    analyze_baseline_differences()