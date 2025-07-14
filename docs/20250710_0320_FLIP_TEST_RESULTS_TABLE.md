# Tabela Wyników Wszystkich Flip Testów
Data: 2025-07-10 03:20

## Legenda:
- **0** = bez zmiany score (nie w public test set)
- **-1** = zły flip (score spadł po naszej zmianie)
- **+1** = dobry flip (prawdziwy błąd znaleziony)

## Wyniki (chronologicznie):

| Lp | ID | Kierunek | Score | Status | Uwagi |
|----|-----|----------|-------|--------|-------|
| 1 | 19612 | E→I | 0.975708 | **0** | |
| 2 | 20934 | E→I | 0.974898 | **-1** | ❌ Błędny flip |
| 3 | 23336 | E→I | 0.975708 | **0** | |
| 4 | 23844 | E→I | 0.975708 | **0** | |
| 5 | 20017 | E→I | 0.975708 | **0** | |
| 6 | 20033 | I→E | 0.975708 | **0** | |
| 7 | 22234 | I→E | 0.975708 | **0** | |
| 8 | 22927 | I→E | 0.975708 | **0** | |
| 9 | 19636 | I→E | 0.975708 | **0** | |
| 10 | 22850 | I→E | 0.975708 | **0** | |
| 11 | 20932 | I→E | 0.974898 | **-1** | ❌ Błędny flip |
| 12 | 18524 | E→I | 0.975708 | **0** | |
| 13 | 18566 | I→E | 0.975708 | **0** | |
| 14 | 18534 | E→I | 0.975708 | **0** | |
| 15 | 18634 | E→I | 0.974898 | **-1** | ❌ Błędny flip |
| 16 | 18834 | E→I | 0.975708 | **0** | |
| 17 | 18934 | E→I | 0.975708 | **0** | |
| 18 | 19034 | E→I | 0.975708 | **0** | |
| 19 | 23844 | E→I | 0.975708 | **0** | (powtórka z #4) |
| 20 | 19234 | E→I | 0.975708 | **0** | |
| 21 | 24428 | I→E | 0.975708 | **0** | podobny do 11798 |
| 22 | 23547 | E→I | 0.975708 | **0** | MMD outlier |
| 23 | 21138 | E→I | 0.974898 | **-1** | ❌ MMD outlier + gradient |
| 24 | 20728 | I→E | 0.974898 | **-1** | ❌ MMD outlier |
| 25 | 21359 | E→I | 0.974898 | **-1** | ❌ CORRECTED_DISAGREEMENT |
| 26 | 19612 | E→I | 0.975708 | **0** | CORRECTED_DISAGREEMENT |
| 27 | 20017 | E→I | 0.975708 | **0** | CORRECTED_DISAGREEMENT |
| 28 | 19876 | E→I | 0.975708 | **0** | CORRECTED_DISAGREEMENT |
| 29 | 19774 | E→I | 0.975708 | **0** | CORRECTED_DISAGREEMENT |
| 30 | 18876 | E→I | 0.975708 | **0** | CORRECTED_DISAGREEMENT |
| 31 | 18604 | E→I | 0.975708 | **0** | CORRECTED_DISAGREEMENT |
| 32 | 19514 | E→I | 0.975708 | **0** | CORRECTED_DISAGREEMENT |
| 33 | 18754 | E→I | 0.975708 | **0** | CORRECTED_DISAGREEMENT |
| 34 | 19477 | E→I | 0.975708 | **0** | CORRECTED_DISAGREEMENT |

## Podsumowanie:

| Status | Liczba | Procent | Opis |
|--------|--------|---------|------|
| **0** | 28 | 82.4% | Nie w public test set |
| **-1** | 6 | 17.6% | Błędne flipy (false positives) |
| **+1** | 0 | 0% | Prawdziwy błąd znaleziony (score wzrósł) |

## Kluczowe ID:

### ✅ Potwierdzone błędy (0):
- Żaden

### ❌ Potwierdzone NIE-błędy (6):
- **20934**: Jest poprawnie Extrovert
- **18634**: Jest poprawnie Extrovert
- **20932**: Jest poprawnie Introvert
- **21138**: Jest poprawnie Extrovert (MMD outlier)
- **20728**: Jest poprawnie Introvert (MMD outlier)
- **21359**: Jest poprawnie Extrovert (CORRECTED_DISAGREEMENT)

### ❓ Nieznany status (28):
- Wszystkie pozostałe - mogą być błędami w private test set

## Analiza metod:
- **Original high uncertainty**: 1/5 trafień (20% skuteczność w public)
- **Sequential/Pattern**: 0/6 trafień (0% skuteczność)
- **Boundary analysis**: 0/4 trafień (0% skuteczność, 2 false positives)
- **MMD outliers**: 0/3 trafień (0% skuteczność, 2 false positives)
- **CORRECTED_DISAGREEMENT**: 0/10 trafień (0% skuteczność, 1 false positive)