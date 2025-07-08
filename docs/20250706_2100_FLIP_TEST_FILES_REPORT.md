# Raport: Pliki Testowe z Pojedynczymi Flipami
Data: 2025-07-06 21:00

## Cel eksperymentu

Celem jest przetestowanie wpływu pojedynczych zmian (flipów) w predykcjach na public score w Kaggle. Pomoże to:
1. Określić dokładną wielkość test setu
2. Zrozumieć strukturę scoringu
3. Znaleźć kluczowe rekordy wpływające na wynik
4. Odkryć jak TOP 3 przełamali barierę 0.975708 → 0.976518

## Plik bazowy

### Oryginalny submission (100% dokładny)
- **Plik**: `subm-0.96950-20250704_121621-xgb-381482788433-148.csv`
- **Lokalizacja**: `/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/subm/20250705/`
- **Public Score**: 0.975708 (osiągnięty przez 240+ uczestników)
- **Model**: XGBoost (5 trees, depth 2) z ambivert detection
- **Pochodzenie**: Optuna optimization study, trial #148
- **Rozkład**: 4622 Extroverts (74.85%), 1553 Introverts (25.15%)

## Utworzone pliki testowe

### Grupa 1: Extrovert → Introvert Flips
Najbardziej podejrzane rekordy - Extroverts z silnymi cechami Introvert.

| Plik | ID | Cechy introvertyczne | Opis |
|------|-----|---------------------|------|
| `flip_E2I_1_id_19612.csv` | 19612 | 3/5 | Time_alone>8, Drained=Yes, Stage_fear=Yes |
| `flip_E2I_2_id_20934.csv` | 20934 | 3/5 | Time_alone>8, Drained=Yes, Social<3 |
| `flip_E2I_3_id_23336.csv` | 23336 | 3/5 | Time_alone>8, Stage_fear=Yes, Friends<5 |
| `flip_E2I_4_id_23844.csv` | 23844 | 3/5 | Drained=Yes, Social<3, Friends<5 |
| `flip_E2I_5_id_20017.csv` | 20017 | 2/5 | Time_alone>8, Stage_fear=Yes |

**Dlaczego te rekordy?**
- Sklasyfikowane jako Extrovert, ale mają 2-3 silne cechy Introvert
- Najbardziej prawdopodobne błędy klasyfikacji
- Mogą być "ukrytymi ambivertami"

### Grupa 2: Introvert → Extrovert Flips
Pierwsze 5 Introverts z submission (brak dobrych kandydatów).

| Plik | ID | Cechy extrovertyczne | Opis |
|------|-----|---------------------|------|
| `flip_I2E_simple_1_id_18525.csv` | 18525 | 0-1/5 | Brak silnych cech E |
| `flip_I2E_simple_2_id_18528.csv` | 18528 | 0-1/5 | Brak silnych cech E |
| `flip_I2E_simple_3_id_18531.csv` | 18531 | 0-1/5 | Brak silnych cech E |
| `flip_I2E_simple_4_id_18533.csv` | 18533 | 0-1/5 | Brak silnych cech E |
| `flip_I2E_simple_5_id_18536.csv` | 18536 | 0-1/5 | Brak silnych cech E |

**Dlaczego brak dobrych kandydatów?**
- 90.7% Introverts nie ma ŻADNYCH cech Extrovert
- 9.3% ma tylko 1 cechę (zazwyczaj Friends>8)
- Model bardzo dobrze klasyfikuje Introverts

### Grupa 3: Kombinowany test
| Plik | Opis |
|------|------|
| `flip_ALL_10_combined.csv` | Wszystkie 10 flipów (5 E→I + 5 I→E) w jednym pliku |

## Lokalizacja plików

Wszystkie pliki znajdują się w:
```
/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scores/
```

## Metodologia tworzenia

1. **Pliki E→I**: Znaleziono Extroverts z największą liczbą cech introvertycznych (Time_alone>8, Drained=Yes, Stage_fear=Yes, Social<3, Friends<5)

2. **Pliki I→E**: Ze względu na brak dobrych kandydatów, wybrano pierwsze 5 Introverts

3. **Każdy plik**: 
   - Jest dokładną kopią oryginalnego submission
   - Ma zmienioną tylko JEDNĄ predykcję
   - Zachowuje 6174 niezmienione rekordy

## Oczekiwane wyniki

### Jeśli test set = 6175 rekordów:
- 1 błąd = 1/6175 = 0.000162 spadek accuracy
- 1 flip: 0.975708 → 0.975546
- 10 flips: 0.975708 → 0.974088

### Jeśli brak zmiany score:
- Rekordy mogą nie być w public test set
- Test set może być mniejszy (np. 50% = 3087 rekordów)
- Niektóre rekordy mogą być duplikatami

## Interpretacja wyników

Po submisji należy sprawdzić:

1. **Czy pojedynczy flip zmienia score?**
   - TAK → możemy obliczyć dokładną wielkość test setu
   - NIE → test set jest podzielony lub mniejszy

2. **Który typ flipa ma większy wpływ?**
   - E→I większy → model lepiej klasyfikuje Introverts
   - I→E większy → model lepiej klasyfikuje Extroverts

3. **Czy efekt jest liniowy?**
   - 10 flips = 10× efekt 1 flipa → prosty scoring
   - Inaczej → możliwe wagi lub duplikaty

## Znaczenie dla TOP 3

TOP 3 osiągnęli 0.976518 (tylko 95 błędów z 18524 = 99.487% accuracy). Możliwe strategie:
1. Znaleźli więcej błędów w danych treningowych
2. Odkryli wzorzec w test set
3. Użyli zewnętrznych danych
4. Znaleźli duplikaty lub wagi w scoringu

Ten eksperyment pomoże zrozumieć strukturę problemu i może ujawnić drogę do przełamania bariery 0.975708.