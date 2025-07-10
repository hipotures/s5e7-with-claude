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
| 2 | 20934 | E→I | 0.974898 | **+1** | ✅ PRAWDZIWY BŁĄD! |
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

## Podsumowanie:

| Status | Liczba | Procent | Opis |
|--------|--------|---------|------|
| **0** | 17 | 85% | Nie w public test set |
| **-1** | 2 | 10% | Błędne flipy (false positives) |
| **+1** | 1 | 5% | Prawdziwy błąd znaleziony |

## Kluczowe ID:

### ✅ Potwierdzone błędy (1):
- **20934**: Extrovert → Introvert (kończy się na 34)

### ❌ Potwierdzone NIE-błędy (2):
- **18634**: Jest poprawnie Extrovert (kończy się na 34)
- **20932**: Jest poprawnie Introvert (blisko 20934)

### ❓ Nieznany status (17):
- Wszystkie pozostałe - mogą być błędami w private test set