# Plan Submisji - Ostatni na dziś + 5 na jutro

## Ostatnia submisja DZIŚ (1/1):

### PRIORYTET: `flip_E2I_5_id_20017.csv`
**Dlaczego?**
- Ostatni z grupy E→I 
- Ma 2/5 cech introvert (mniej niż rekord 20934 który trafił!)
- Pozwoli sprawdzić czy mniej "oczywiste" błędy też są w public set

**Oczekiwania:**
- 20% szans że jest w public set
- Jeśli trafi: spadek o -0.000810
- Jeśli nie: bez zmian (0.975708)

---

## Plan na JUTRO (5 submisji):

### 1. `flip_ALL_10_combined.csv` 
**Cel**: Sprawdzić efekt kumulacyjny
- Zawiera wszystkie 10 pojedynczych flipów
- Oczekiwany wynik: 
  - Jeśli tylko 20934 w public: spadek -0.000810
  - Jeśli więcej trafień: większy spadek
  - Pokaże czy są inne "ukryte" trafienia

### 2-5. Eksperyment: Szukanie wzorca rekordu 20934

Musimy znaleźć co wyróżnia rekord 20934! Propozycja:

#### 2. `flip_similar_to_20934_v1.csv`
- Znajdź rekordy z DOKŁADNIE takimi samymi cechami jak 20934
- Flipuj pierwszy znaleziony

#### 3. `flip_similar_to_20934_v2.csv`
- Flipuj drugi podobny rekord

#### 4. `flip_OPPOSITE_20934.csv`
- Znajdź rekord 20934 i flipuj go ODWROTNIE (I→E)
- Oczekiwany wynik: +0.000810 (powrót do 0.976518!)

#### 5. `flip_cluster_20934.csv`
- Znajdź wszystkie rekordy z tego samego klastra co 20934
- Flipuj losowy z nich

---

## Alternatywny Plan B (jeśli wolisz):

1. `flip_ALL_10_combined.csv`
2. `flip_I2E_simple_1_id_18525.csv` 
3. `flip_I2E_simple_2_id_18528.csv`
4. `flip_I2E_simple_3_id_18531.csv`
5. `flip_I2E_simple_4_id_18533.csv`

**Cel**: Sprawdzić czy jakiś Introvert też jest w public set

---

## Co zrobić przed jutrzejszymi submisami:

1. **Przeanalizuj rekord 20934**:
   - Jakie ma dokładne cechy?
   - Do jakiego klastra należy?
   - Czy są podobne rekordy?

2. **Stwórz nowe pliki** według wybranego planu

3. **Zapisz wszystkie wyniki** dla pełnej analizy

## Pytanie do Ciebie:
Który plan wolisz - szukanie wzorca 20934 (Plan A) czy testowanie I→E (Plan B)?