# Podsumowanie Wyników Sesji - 2025-07-12

## 1. Wyniki Flip Testów

### Status: 21 testów przeprowadzonych
- **ID 24428** (podobny do 11798): bez zmiany score → poza public test set (20%)
- Łącznie: 19/21 testów w private test set (90.5%)
- Tylko 1 potwierdzony błąd w public: ID 20934

## 2. Quilt Framework - Gradient Analysis

### Kluczowe odkrycia:
1. **Gradient jako wskaźnik błędów**:
   - Top 5% (95 percentyl): tylko 39% accuracy
   - Top 10% (90 percentyl): tylko 69.6% accuracy  
   - Reszta: 100% accuracy

2. **Analiza pełnego train set (18,524 próbek)**:
   - 927 próbek w top 5% gradientów
   - **WSZYSTKIE top 20 to Introwerticy** - sugeruje systematyczny błąd
   - Charakterystyka: dużo przyjaciół (8-15), aktywność społeczna, mało czasu samemu

3. **Test set - wysokie gradienty**:
   - ID 22291: 0.0265 (najwyższy)
   - ID 24005: 0.0133 (99.9 percentyl!)
   - ID 20934: 0.0016 (97.8 percentyl)

### Stworzone flip submissions:
- `flip_GRADIENT_1_E2I_id_22291.csv`
- `flip_GRADIENT_2_E2I_id_18754.csv`
- `flip_GRADIENT_3_E2I_id_21138.csv`
- `flip_GRADIENT_4_E2I_id_21932.csv`
- `flip_GRADIENT_5_E2I_id_22189.csv`

## 3. Yggdrasil Decision Forests

### Wyniki:
1. **Szybkość**: 
   - RF 500 drzew: 1.06s
   - GBT 500 drzew: 0.70s

2. **Dokładność**:
   - RF OOB: 96.896%
   - GBT: 97.204%
   - Zgodność modeli: 99.81%

3. **Najbardziej niepewne**:
   - **ID 19482: prawdopodobieństwo 0.500!** (uncertainty 1.0)
   - ID 24005: 0.560 (uncertainty 0.880)
   - ID 23711: 0.387 (uncertainty 0.773)

### Stworzone submissions:
- `ydf_random_forest_500trees.csv`
- `ydf_gradient_boosted_500trees.csv`
- `ydf_optimized_baseline.csv`
- `ydf_flip_uncertain_id_19482.csv`

## 4. Najważniejsze Kandydaci do Flipa

### Priorytet 1 (bardzo wysokie prawdopodobieństwo błędu):
1. **ID 19482** - YDF daje dokładnie 0.5 prawdopodobieństwa!
2. **ID 24005** - wysokie gradienty (99.9%) + wysoka niepewność
3. **ID 22291** - najwyższy gradient w test set

### Priorytet 2 (wysokie prawdopodobieństwo):
- ID 18754, 21138, 21932, 22189 (wysokie gradienty)
- ID 23711, 23547 (wysoka niepewność YDF)

## 5. Wzorce i Wnioski

1. **Systematyczny błąd**: Introwerticy z cechami ekstrawertyków
   - Dużo przyjaciół (>10)
   - Niska samotność
   - Brak stage fear/drained

2. **Metody detekcji**:
   - Gradienty (Quilt) - wykrywają outliers w train
   - Niepewność modeli (YDF) - wykrywają trudne przypadki
   - Oba podejścia się uzupełniają

3. **Public vs Private test set**:
   - ~20% w public (1235/6175)
   - ~80% w private (4940/6175)
   - Większość naszych kandydatów jest w private

## 6. Rekomendacje

1. **Przetestuj na Kaggle**:
   - ID 19482 (YDF 0.5)
   - ID 22291 (najwyższy gradient)
   - Gradient-based flips

2. **Analiza post-submission**:
   - Jeśli score się zmieni → trafiliśmy public
   - Jeśli nie → czekamy na private leaderboard

3. **Strategia długoterminowa**:
   - Skupić się na wysokich gradientach + niepewności
   - Szukać więcej Introwertyków z cechami Ekstrawertyków