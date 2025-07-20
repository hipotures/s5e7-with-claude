#!/usr/bin/env python3
"""
Tworzy pliki flip test dla ID z submission_RFECV
"""

import pandas as pd

def create_flip_test(baseline_file, output_file, flip_id, new_value):
    """Tworzy plik z pojedynczym flipem"""
    
    # Wczytaj baseline
    df = pd.read_csv(baseline_file)
    
    # Zmień wartość dla określonego ID
    df.loc[df['id'] == flip_id, 'Personality'] = new_value
    
    # Zapisz
    df.to_csv(output_file, index=False)
    print(f"Utworzono: {output_file}")
    
    # Weryfikacja
    changed_rows = df[df['id'] == flip_id]
    if len(changed_rows) > 0:
        print(f"  ID {flip_id}: {changed_rows['Personality'].values[0]}")
    else:
        print(f"  UWAGA: Nie znaleziono ID {flip_id}!")

def main():
    baseline_file = 'subm-0.96950-20250704_121621-xgb-381482788433-148.csv'
    
    # Sprawdź strukturę baseline
    df = pd.read_csv(baseline_file)
    print(f"Sprawdzam baseline...")
    print(f"Liczba wierszy: {len(df)}")
    
    # Testy do utworzenia (8 z 9, pomijamy 23036)
    tests = [
        # I->E (3 pliki)
        (19087, 'Extrovert', 'flip_RFECV_1_I2E_id_19087.csv'),
        (19482, 'Extrovert', 'flip_RFECV_2_I2E_id_19482.csv'),
        (20033, 'Extrovert', 'flip_RFECV_3_I2E_id_20033.csv'),
        # E->I (5 z 6 plików)
        (20822, 'Introvert', 'flip_RFECV_4_E2I_id_20822.csv'),
        (21008, 'Introvert', 'flip_RFECV_5_E2I_id_21008.csv'),
        (21932, 'Introvert', 'flip_RFECV_6_E2I_id_21932.csv'),
        (22750, 'Introvert', 'flip_RFECV_7_E2I_id_22750.csv'),
        (22794, 'Introvert', 'flip_RFECV_8_E2I_id_22794.csv'),
        # Pomijamy: (23036, 'Introvert', 'flip_RFECV_9_E2I_id_23036.csv')
    ]
    
    print("\nTworzę pliki flip test...")
    print("(Pomijam ID 23036 - można wydedukować z reszty)\n")
    
    for flip_id, new_value, output_file in tests:
        # Sprawdź jaka jest oryginalna wartość
        orig_val = df[df['id'] == flip_id]['Personality'].values[0]
        print(f"ID {flip_id}: {orig_val} -> {new_value}")
        create_flip_test(baseline_file, output_file, flip_id, new_value)
        print()
    
    print("\nPodsumowanie:")
    print("- Utworzono 8 plików testowych")
    print("- ID 23036 pominięte (można wydedukować)")
    print("- Razem te 9 ID powinno dać -1")

if __name__ == "__main__":
    main()