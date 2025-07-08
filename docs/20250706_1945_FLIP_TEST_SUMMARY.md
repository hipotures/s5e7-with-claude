# Flip Test Summary - Testing Single Record Impact

## Odtworzenie oryginalnego wyniku 0.975708

### Oryginalny submission:
- **Plik**: `subm-0.96950-20250704_121621-xgb-381482788433-148.csv`
- **Public Score**: 0.975708
- **Model**: XGBoost (5 trees, depth 2) + ambivert detection

### Odtworzenie (v2):
- **Zgodność**: 98.98% (6112/6175 matches)
- **Mismatches**: tylko 63 rekordy
- **Konfiguracja**: 
  - XGBoost: n_estimators=5, max_depth=2, learning_rate=1.0
  - Dynamic thresholds (0.43 for ambiverts, 0.5 for others)
  - 96.2% rule for most uncertain cases

## Utworzone pliki testowe

### Najbardziej niepewne rekordy (flip candidates):

| File | ID | Probability | Uncertainty | Original → Flipped |
|------|-----|------------|-------------|-------------------|
| flip_test_1_id_23418.csv | 23418 | 0.4887 | 0.0113 | Extrovert → Introvert |
| flip_test_2_id_23844.csv | 23844 | 0.4887 | 0.0113 | Extrovert → Introvert |
| flip_test_3_id_21932.csv | 21932 | 0.4887 | 0.0113 | Extrovert → Introvert |
| flip_test_4_id_23336.csv | 23336 | 0.4887 | 0.0113 | Extrovert → Introvert |
| flip_test_5_id_24011.csv | 24011 | 0.4017 | 0.0983 | Extrovert → Introvert |

## Hipotezy do sprawdzenia

1. **Jeśli 1 flip zmieni score**: 
   - Test set ma dokładnie 6175 rekordów
   - Każdy błąd = -0.000162 do score
   - Score 0.975708 = 6024 poprawnych z 6175

2. **Jeśli potrzeba 2+ flips**:
   - Test set może być mniejszy (np. połowa)
   - Lub niektóre rekordy są ważone inaczej

3. **Oczekiwane zmiany score**:
   - 1 flip: 0.975708 → 0.975546 (spadek o 0.000162)
   - 2 flips: 0.975708 → 0.975384 (spadek o 0.000324)
   - etc.

## Next Steps

1. Submit wszystkie 5 plików do Kaggle
2. Zanotować dokładne scores
3. Obliczyć wpływ pojedynczego flipa
4. Zidentyfikować które rekordy mają największy wpływ

## Dodatkowe obserwacje

- Wiele rekordów ma identyczną probability (0.4887) - może to wskazywać na specyficzny wzorzec
- Najbardziej niepewne rekordy są blisko progu 0.5
- Wszystkie flip candidates to obecnie Extroverts - sugeruje że model ma tendencję do przewidywania Extrovert przy niepewności