#!/usr/bin/env python3
"""
Compare 0.975708 vs 0.976518 in context of 20% public test set
"""

def main():
    print("="*60)
    print("PORÓWNANIE: 0.975708 vs 0.976518")
    print("="*60)
    
    # Scores
    your_score = 0.975708
    top3_score = 0.976518
    difference = top3_score - your_score
    
    # Public test set (20%)
    public_size = 1235  # 20% of 6175
    
    print(f"\nTwój score: {your_score}")
    print(f"TOP 3 score: {top3_score}")
    print(f"Różnica: {difference:.6f}")
    
    # Oblicz liczbę poprawnych
    your_correct = int(round(your_score * public_size))
    top3_correct = int(round(top3_score * public_size))
    
    your_errors = public_size - your_correct
    top3_errors = public_size - top3_correct
    
    print(f"\nW PUBLIC SET ({public_size} rekordów):")
    print(f"Ty: {your_correct}/{public_size} poprawnych ({your_errors} błędów)")
    print(f"TOP 3: {top3_correct}/{public_size} poprawnych ({top3_errors} błędów)")
    print(f"\nRóżnica: {top3_correct - your_correct} dodatkowe poprawne odpowiedzi")
    
    # Weryfikacja
    print("\nWeryfikacja:")
    print(f"1 dodatkowa poprawna = +{1/public_size:.6f} do score")
    print(f"Obserwowana różnica = {difference:.6f}")
    print(f"Zgadza się? {abs(difference - 1/public_size) < 0.000001}")
    
    # Co to oznacza dla pełnego test setu
    print("\n" + "="*60)
    print("EKSTRAPOLACJA NA PEŁNY TEST SET:")
    print("="*60)
    
    full_test = 6175
    
    # Jeśli błędy są równomiernie rozłożone
    your_errors_full = int(your_errors * 5)  # 20% → 100%
    top3_errors_full = int(top3_errors * 5)
    
    print(f"\nJeśli błędy równomiernie rozłożone:")
    print(f"Ty: ~{your_errors_full} błędów na {full_test}")
    print(f"TOP 3: ~{top3_errors_full} błędów na {full_test}")
    print(f"Różnica: {your_errors_full - top3_errors_full} błędów")
    
    # Interpretacja
    print("\n" + "="*60)
    print("CO TO OZNACZA:")
    print("="*60)
    
    print("\n1. TOP 3 poprawili TYLKO 1 REKORD w public set")
    print("   - Rekord 20934 (lub podobny)")
    print("   - Zmienili z Extrovert → Introvert")
    
    print("\n2. W pełnym test set (6175) prawdopodobnie:")
    print("   - Znaleźli ~5 takich błędów")
    print("   - Tylko 1 z nich (20%) był w public")
    
    print("\n3. Alternatywne wyjaśnienie:")
    print("   - Może znaleźli 1 SYSTEMATYCZNY błąd")
    print("   - Np. wszystkie rekordy z określonym wzorcem")
    print("   - Wtedy private score może być DUŻO wyższy!")
    
    # Maksymalny możliwy score
    print("\n" + "="*60)
    print("TEORETYCZNE MAKSIMUM:")
    print("="*60)
    
    if your_errors == 30:  # ~2.43% błędów
        print(f"\nJeśli te {your_errors} błędów to prawdziwi ambiverts:")
        print(f"Maksymalny możliwy score = {(public_size - 0)/public_size:.6f}")
        print(f"TOP 3 są na poziomie {top3_errors}/{your_errors} = {top3_errors/your_errors:.1%} drogi do perfekcji")

if __name__ == "__main__":
    main()