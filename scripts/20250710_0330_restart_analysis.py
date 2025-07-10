#!/usr/bin/env python3
"""
Restart analysis - understand why we failed and where to look
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")

def analyze_our_failures():
    """Analyze why our flips failed"""
    print("="*60)
    print("ANALIZA NASZYCH BŁĘDNYCH FLIPÓW")
    print("="*60)
    
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    
    # Our failed flips
    failed_flips = [
        {'id': 20934, 'predicted': 'E', 'we_changed_to': 'I', 'result': 'score dropped'},
        {'id': 18634, 'predicted': 'E', 'we_changed_to': 'I', 'result': 'score dropped'},
        {'id': 20932, 'predicted': 'I', 'we_changed_to': 'E', 'result': 'score dropped'}
    ]
    
    print("\nNasze błędne flipy:")
    for flip in failed_flips:
        test_row = test_df[test_df['id'] == flip['id']].iloc[0]
        print(f"\nID {flip['id']}:")
        print(f"  Model predicted: {flip['predicted']}")
        print(f"  We changed to: {flip['we_changed_to']}")
        print(f"  Result: {flip['result']} → prediction was CORRECT")
        print(f"  Features:")
        print(f"    Time_alone: {test_row.get('Time_spent_Alone', 'NaN')}")
        print(f"    Social: {test_row.get('Social_event_attendance', 'NaN')}")
        print(f"    Friends: {test_row.get('Friends_circle_size', 'NaN')}")
        print(f"    Drained: {test_row.get('Drained_after_socializing', 'NaN')}")
        print(f"    Stage_fear: {test_row.get('Stage_fear', 'NaN')}")

def analyze_what_top_players_might_see():
    """Think about what TOP players found"""
    print("\n" + "="*60)
    print("CO MOGLI ZNALEŹĆ TOP GRACZE?")
    print("="*60)
    
    print("\n1. MOŻLIWE CHARAKTERYSTYKI PRAWDZIWYCH BŁĘDÓW:")
    print("-"*40)
    print("• Nie są w oczywistych miejscach (końcówka 34, ekstremalne profile)")
    print("• Mogą mieć subtelne nieprawidłowości")
    print("• Prawdopodobnie wymagają głębszej analizy")
    
    print("\n2. GDZIE NIE SZUKALIŚMY:")
    print("-"*40)
    print("• Środek rozkładu (średnie wartości wszystkich cech)")
    print("• Konkretne kombinacje cech (np. specific patterns)")
    print("• Analiza błędów modelu na train set")
    print("• Cross-validation errors")
    
    print("\n3. NOWE PODEJŚCIA DO ROZWAŻENIA:")
    print("-"*40)
    print("• Analiza najbardziej niepewnych predykcji (probability ~0.5)")
    print("• Szukanie duplikatów między train i test")
    print("• Analiza feature importance dla błędnie klasyfikowanych")
    print("• Sprawdzenie czy model consistently misclassifies certain patterns")

def analyze_model_uncertainty():
    """Check model uncertainty on predictions"""
    print("\n" + "="*60)
    print("ANALIZA NIEPEWNOŚCI MODELU")
    print("="*60)
    
    # We need to load predictions with probabilities
    # This is what we should analyze:
    
    print("\nCO POWINNIŚMY ZROBIĆ:")
    print("1. Wytrenować model na train set")
    print("2. Uzyskać prawdopodobieństwa dla test set")
    print("3. Znaleźć rekordy gdzie model jest najmniej pewny (prob ~0.5)")
    print("4. To mogą być prawdziwe błędy!")
    
    print("\nDLACZEGO TO MOŻE DZIAŁAĆ:")
    print("• Jeśli rekord jest błędnie oznaczony w podobnych przykładach treningowych")
    print("• Model będzie niepewny przy predykcji")
    print("• TOP gracze mogli użyć tej metody")

def suggest_new_strategy():
    """Suggest completely new approach"""
    print("\n" + "="*60)
    print("NOWA STRATEGIA")
    print("="*60)
    
    print("\n1. TRENUJ MODEL I ANALIZUJ BŁĘDY NA TRAIN SET:")
    print("-"*40)
    print("• Użyj cross-validation")
    print("• Znajdź które rekordy są consistently misclassified")
    print("• Sprawdź czy podobne rekordy są w test set")
    
    print("\n2. SZUKAJ DUPLIKATÓW TRAIN-TEST:")
    print("-"*40)
    print("• Może niektóre rekordy występują w obu zbiorach?")
    print("• Jeśli mają różne etykiety = błąd!")
    
    print("\n3. ANALIZA PROBABILISTYCZNA:")
    print("-"*40)
    print("• Train multiple models (XGB, LGBM, CatBoost)")
    print("• Znajdź rekordy gdzie modele się nie zgadzają")
    print("• Szczególnie gdzie jeden model jest bardzo pewny a drugi nie")
    
    print("\n4. FEATURE ENGINEERING:")
    print("-"*40)
    print("• Stwórz nowe cechy")
    print("• Może prawdziwe błędy są widoczne tylko w transformed space?")

if __name__ == "__main__":
    analyze_our_failures()
    analyze_what_top_players_might_see()
    analyze_model_uncertainty()
    suggest_new_strategy()
    
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    print("\n❌ Nasze podejście było zbyt powierzchowne")
    print("❌ Szukaliśmy oczywistych wzorców które nie istnieją")
    print("✅ Potrzebujemy model-based approach")
    print("✅ Analiza niepewności predykcji jest kluczowa")
    print("✅ Cross-validation errors mogą wskazać prawdziwe błędy")