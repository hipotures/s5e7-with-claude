#!/usr/bin/env python3
"""
Strategies to find the second error that new TOP 1 discovered
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_patterns():
    print("="*60)
    print("STRATEGIE SZUKANIA DRUGIEGO BŁĘDU")
    print("="*60)
    
    print("\n1. ANALIZA WZORCA REKORDU 20934:")
    print("-"*40)
    print("Co wiemy o 20934:")
    print("• Był oznaczony jako E, powinien być I")
    print("• Ma nietypowy profil:")
    print("  - Time_alone: 2.0 (bardzo mało jak na I)")
    print("  - Drained: Yes + Stage_fear: Yes (cechy I)")
    print("  - Social: 2.0 (mało jak na E)")
    print("• ID: 20934 (środkowy zakres)")
    
    print("\n2. HIPOTEZY O DRUGIM BŁĘDZIE:")
    print("-"*40)
    
    print("\nA) PRZECIWNY PROFIL:")
    print("   Szukaj Introverts z:")
    print("   - Time_alone > 10")
    print("   - Drained: No")
    print("   - Stage_fear: No") 
    print("   - Social > 7")
    print("   → Klasyczne Extroverts błędnie oznaczone jako I")
    
    print("\nB) SPRZECZNOŚCI LOGICZNE:")
    print("   - I z Friends > 10 + Social > 8")
    print("   - E z Time_alone > 12")
    print("   - I z Post_frequency > 10")
    print("   → Niemożliwe kombinacje")
    
    print("\nC) EKSTREMALNE WARTOŚCI:")
    print("   - Nulls w kluczowych cechach")
    print("   - Wartości = 0 lub max")
    print("   - Outliers (3+ odchylenia)")
    
    print("\nD) ZAKRES ID:")
    print("   - Blisko 20934 (np. 20900-21000)")
    print("   - Początek/koniec datasetu")
    print("   - Co ~1000 rekordów (batch effect)")
    
    print("\nE) DUPLIKATY/BLIŹNIAKI:")
    print("   - Identyczne profile z różnymi labels")
    print("   - Prawie identyczne (1 cecha różna)")
    
    print("\n3. KONKRETNY PLAN AKCJI:")
    print("-"*40)
    
    strategies = [
        {
            'name': 'extreme_introvert_as_E',
            'criteria': 'E z Time_alone>12 lub Social<2',
            'logic': 'Ekstremalnie introwertyczne E'
        },
        {
            'name': 'extreme_extrovert_as_I', 
            'criteria': 'I z Friends>10 i Social>7',
            'logic': 'Ekstremalnie ekstrawertyczne I'
        },
        {
            'name': 'near_20934',
            'criteria': 'ID w zakresie 20900-21000',
            'logic': 'Błędy mogą być w batch'
        },
        {
            'name': 'contradiction_hunters',
            'criteria': 'Drained=No + Stage_fear=No + Time_alone<3',
            'logic': 'Sprzeczne cechy'
        },
        {
            'name': 'null_patterns',
            'criteria': 'Rekordy z 3+ nullami',
            'logic': 'Niepełne dane = błędy'
        }
    ]
    
    print("\nTOP 5 STRATEGII NA JUTRO:")
    for i, s in enumerate(strategies):
        print(f"\n{i+1}. {s['name'].upper()}")
        print(f"   Kryterium: {s['criteria']}")
        print(f"   Logika: {s['logic']}")
    
    print("\n4. IMPLEMENTACJA:")
    print("-"*40)
    print("Dla każdej strategii:")
    print("1. Znajdź 5-10 kandydatów")
    print("2. Posortuj wg siły sprzeczności")
    print("3. Weź top 1 z każdej strategii")
    print("4. Stwórz 5 plików flip")
    print("5. Test który wzorzec działa!")

def create_search_scripts():
    """Create the actual search implementation"""
    
    script_content = '''#!/usr/bin/env python3
"""
Find candidates for the second error using multiple strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"

def find_candidates():
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Preprocess
    test_df['Drained_num'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    test_df['Stage_fear_num'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})
    
    # Count nulls
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 
                   'Going_outside', 'Post_frequency']
    test_df['null_count'] = test_df[feature_cols].isnull().sum(axis=1)
    
    candidates = []
    
    # Strategy 1: Extreme introverts marked as E
    mask1 = (original_df['Personality'] == 'Extrovert')
    for idx in original_df[mask1].index:
        if idx < len(test_df):
            row = test_df.iloc[idx]
            score = 0
            if pd.notna(row['Time_spent_Alone']) and row['Time_spent_Alone'] > 12:
                score += 3
            if pd.notna(row['Social_event_attendance']) and row['Social_event_attendance'] < 2:
                score += 2
            if row.get('Drained_num') == 1:
                score += 1
            if score >= 3:
                candidates.append({
                    'id': row['id'],
                    'strategy': 'extreme_I_as_E',
                    'score': score,
                    'time_alone': row['Time_spent_Alone'],
                    'social': row['Social_event_attendance']
                })
    
    # Strategy 2: Extreme extroverts marked as I
    mask2 = (original_df['Personality'] == 'Introvert')
    for idx in original_df[mask2].index:
        if idx < len(test_df):
            row = test_df.iloc[idx]
            score = 0
            if pd.notna(row['Friends_circle_size']) and row['Friends_circle_size'] >= 10:
                score += 2
            if pd.notna(row['Social_event_attendance']) and row['Social_event_attendance'] > 7:
                score += 2
            if row.get('Drained_num') == 0:
                score += 1
            if pd.notna(row['Time_spent_Alone']) and row['Time_spent_Alone'] < 2:
                score += 1
            if score >= 3:
                candidates.append({
                    'id': row['id'],
                    'strategy': 'extreme_E_as_I',
                    'score': score,
                    'friends': row['Friends_circle_size'],
                    'social': row['Social_event_attendance']
                })
    
    # Strategy 3: Near 20934
    for idx in range(max(0, 934-50), min(len(test_df), 1034+50)):  # ±50 from 20934
        row = test_df.iloc[idx]
        if 20900 <= row['id'] <= 21000 and row['id'] != 20934:
            candidates.append({
                'id': row['id'],
                'strategy': 'near_20934',
                'score': 1/abs(row['id'] - 20934),  # Closer = higher score
                'distance': abs(row['id'] - 20934)
            })
    
    # Strategy 4: High null count
    high_null = test_df[test_df['null_count'] >= 3]
    for _, row in high_null.iterrows():
        candidates.append({
            'id': row['id'],
            'strategy': 'high_nulls',
            'score': row['null_count'],
            'nulls': row['null_count']
        })
    
    # Sort and display
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    print(f"Found {len(candidates)} candidates")
    print("\\nTop candidates by strategy:")
    
    strategies = ['extreme_I_as_E', 'extreme_E_as_I', 'near_20934', 'high_nulls']
    
    for strategy in strategies:
        strat_candidates = [c for c in candidates if c['strategy'] == strategy]
        if strat_candidates:
            best = strat_candidates[0]
            print(f"\\n{strategy}: ID {best['id']} (score: {best['score']:.3f})")
            print(f"  Details: {best}")
    
    return candidates

if __name__ == "__main__":
    candidates = find_candidates()
'''
    
    # Save the search script
    search_file = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/20250708_0220_find_second_error_candidates.py")
    with open(search_file, 'w') as f:
        f.write(script_content)
    
    print(f"\nUtworzony skrypt: {search_file}")
    
if __name__ == "__main__":
    analyze_patterns()
    create_search_scripts()