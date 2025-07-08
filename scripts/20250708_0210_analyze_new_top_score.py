#!/usr/bin/env python3
"""
Analyze the new #1 score: 0.977327
"""

def main():
    print("="*60)
    print("NOWY LIDER: 0.977327")
    print("="*60)
    
    # Scores
    your_score = 0.975708
    old_top3 = 0.976518
    new_top1 = 0.977327
    
    # Public set
    public_size = 1235
    
    print(f"\nTwój score: {your_score}")
    print(f"Stary TOP 3: {old_top3}")
    print(f"NOWY TOP 1: {new_top1}")
    
    # Calculate improvements
    improvement_from_you = new_top1 - your_score
    improvement_from_old_top = new_top1 - old_top3
    
    print(f"\nPoprawa względem Ciebie: +{improvement_from_you:.6f}")
    print(f"Poprawa względem starego TOP 3: +{improvement_from_old_top:.6f}")
    
    # Calculate flips
    flips_from_you = round(improvement_from_you * public_size)
    flips_from_old_top = round(improvement_from_old_top * public_size)
    
    print(f"\nLiczba dodatkowych poprawnych (względem Ciebie): {flips_from_you}")
    print(f"Liczba dodatkowych poprawnych (względem TOP 3): {flips_from_old_top}")
    
    # Verify
    print("\nWeryfikacja:")
    print(f"1 flip = {1/public_size:.6f}")
    print(f"2 flips = {2/public_size:.6f}")
    print(f"Obserwowana różnica od TOP 3: {improvement_from_old_top:.6f}")
    print(f"Zgadza się z 1 flipem? {abs(improvement_from_old_top - 1/public_size) < 0.000001}")
    
    # Details
    print("\n" + "="*60)
    print("SZCZEGÓŁOWA ANALIZA:")
    print("="*60)
    
    # Errors count
    your_errors = public_size - int(round(your_score * public_size))
    old_top_errors = public_size - int(round(old_top3 * public_size))
    new_top_errors = public_size - int(round(new_top1 * public_size))
    
    print(f"\nLiczba błędów w public set:")
    print(f"Ty: {your_errors}/1235")
    print(f"Stary TOP 3: {old_top_errors}/1235")
    print(f"NOWY TOP 1: {new_top_errors}/1235")
    
    print(f"\nNOWY TOP 1 ma tylko {new_top_errors} błędy!")
    print(f"To {your_errors - new_top_errors} mniej niż Ty")
    print(f"I {old_top_errors - new_top_errors} mniej niż stary TOP 3")
    
    # What it means
    print("\n" + "="*60)
    print("CO TO OZNACZA:")
    print("="*60)
    
    print("\nNOWY TOP 1 znalazł:")
    print(f"• 2 błędy które TY przegapiłeś")
    print(f"• w tym 1 NOWY błąd którego nawet TOP 3 nie znał!")
    
    print("\nMożliwe scenariusze:")
    print("1. Znalazł rekord 20934 + 1 nowy")
    print("2. Znalazł 2 zupełnie nowe błędy")
    print("3. Ma dostęp do większej liczby submisji")
    
    # Path to perfection
    print("\n" + "="*60)
    print("DROGA DO PERFEKCJI:")
    print("="*60)
    
    print(f"Do 100% pozostało: {new_top_errors} błędów")
    print(f"Do 99%: {int(public_size * 0.01) - (public_size - new_top_errors)} flipów")
    print(f"TOP 1 jest na {(public_size - new_top_errors)/public_size*100:.2f}% accuracy")

if __name__ == "__main__":
    main()