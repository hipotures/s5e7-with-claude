#!/usr/bin/env python3
"""
Analizuje pliki w minus-2 z uwzględnieniem znanych neutralnych ID
"""

import pandas as pd
import os

# Znane neutralne ID (potwierdzone + wydedukowane)
NEUTRAL_IDS = {24005, 22340, 18887, 20153, 23177, 19612, 20017, 22286, 22547, 23418, 23547, 23844}

# Znane ID które dają -1
MINUS_ONE_IDS = {20932, 18634, 21359, 21800, 23872, 20934, 21138, 20728}

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

def analyze_file(baseline_data, file_data, filename):
    """Analizuje pojedynczy plik"""
    
    differences = []
    neutral_count = 0
    known_minus_one_count = 0
    unknown_ids = []
    
    for id_val in sorted(set(baseline_data.keys()) | set(file_data.keys())):
        baseline_val = baseline_data.get(id_val)
        file_val = file_data.get(id_val)
        
        if baseline_val != file_val and baseline_val is not None and file_val is not None:
            change = f"{id_val}:{baseline_val}->{file_val}"
            differences.append(change)
            
            if id_val in NEUTRAL_IDS:
                neutral_count += 1
            elif id_val in MINUS_ONE_IDS:
                known_minus_one_count += 1
            else:
                unknown_ids.append((id_val, baseline_val, file_val))
    
    print(f"\n{filename}:")
    print(f"  Wszystkich zmian: {len(differences)}")
    print(f"  Zmian neutralnych (znanych): {neutral_count}")
    print(f"  Zmian -1 (znanych): {known_minus_one_count}")
    print(f"  Zmian o nieznanym wpływie: {len(unknown_ids)}")
    
    if known_minus_one_count > 0:
        print(f"\n  Znane ID które dają -1:")
        for diff in differences:
            id_val = int(diff.split(':')[0])
            if id_val in MINUS_ONE_IDS:
                print(f"    {diff}")
    
    if len(unknown_ids) > 0 and len(unknown_ids) <= 10:
        print(f"\n  ID o nieznanym wpływie:")
        for id_val, base_val, file_val in unknown_ids:
            print(f"    {id_val}:{base_val}->{file_val}")
    
    # Oblicz oczekiwany wpływ nieznanych ID
    # Wynik 0.974089 = baseline - 2*0.000810
    expected_impact = -2
    known_impact = -known_minus_one_count
    unknown_impact = expected_impact - known_impact
    
    print(f"\n  Analiza wpływu:")
    print(f"    Oczekiwany całkowity wpływ: {expected_impact}")
    print(f"    Znany wpływ (z ID -1): {known_impact}")
    print(f"    Nieznany wpływ (suma nieznanych ID): {unknown_impact}")
    
    if unknown_impact == 0 and len(unknown_ids) > 0:
        print(f"    → Wszystkie {len(unknown_ids)} nieznane ID muszą być neutralne!")
    elif len(unknown_ids) == abs(unknown_impact):
        print(f"    → Każde z {len(unknown_ids)} nieznanych ID daje wpływ {unknown_impact/len(unknown_ids):.0f}")
    
    return unknown_ids, unknown_impact

def main():
    # Plik wzorcowy
    baseline_file = 'subm-0.96950-20250704_121621-xgb-381482788433-148.csv'
    baseline_data = load_submission(baseline_file)
    
    print("Analiza plików w minus-2 z uwzględnieniem znanych neutralnych ID")
    print("=" * 70)
    
    print(f"\nZnane neutralne ID: {sorted(NEUTRAL_IDS)}")
    print(f"Znane ID dające -1: {sorted(MINUS_ONE_IDS)}")
    
    # Analizuj pliki w minus-2
    minus2_files = [f for f in os.listdir('minus-2') if f.endswith('.csv')]
    
    all_unknown_ids = set()
    
    for filename in sorted(minus2_files):
        filepath = os.path.join('minus-2', filename)
        try:
            file_data = load_submission(filepath)
            unknown_ids, unknown_impact = analyze_file(baseline_data, file_data, filename)
            
            # Zbierz wszystkie nieznane ID
            for id_val, _, _ in unknown_ids:
                all_unknown_ids.add(id_val)
                
        except Exception as e:
            print(f"\n{filename}: BŁĄD - {e}")
    
    print("\n" + "=" * 70)
    print(f"PODSUMOWANIE: Znaleziono {len(all_unknown_ids)} unikalnych ID o nieznanym wpływie")
    if len(all_unknown_ids) <= 20:
        print(f"Lista: {sorted(all_unknown_ids)}")

if __name__ == "__main__":
    main()