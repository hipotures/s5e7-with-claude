# Raport: Nowe Metody Porównywania Zbiorów Danych (2024-2025)

Data utworzenia: 2025-07-12

## Streszczenie

Ten raport przedstawia najnowsze metody i techniki porównywania zbiorów danych, wykrywania przesunięć (shift) oraz znajdowania różnic między datasetami, oparte na publikacjach i badaniach z lat 2024-2025.

## 1. Metody Wykrywania Dataset Shift

### 1.1 Quilt Framework (Grudzień 2024)
**Opis**: Skalowalne podejście do zarządzania covariate i concept drift poprzez adaptacyjną segmentację danych.

**Kluczowe cechy**:
- Wykorzystuje **gradient-based disparity and gain scores** obliczane na zbiorach treningowych i walidacyjnych
- Automatycznie usuwa segmenty danych wykazujące concept drift
- Wybiera stabilne segmenty danych do efektywnego treningu modelu
- Minimalny koszt obliczeniowy dzięki wykorzystaniu gradientów

**Zastosowanie**: Szczególnie przydatne w środowiskach z ciągłym napływem danych, gdzie rozkład może się zmieniać w czasie.

### 1.2 Automatic Dataset Shift Identification Framework (2024)
**Opis**: Pierwszy w pełni nienadzorowany framework do identyfikacji typu przesunięcia w zbiorze danych.

**Rozróżnia między**:
- **Prevalence shift**: zmiana w rozkładzie etykiet (P(Y) się zmienia)
- **Covariate shift**: zmiana w charakterystykach wejściowych (P(X) się zmienia, ale P(Y|X) pozostaje stałe)
- **Mixed shifts**: jednoczesne wystąpienie obu typów przesunięć

**Technologia**: 
- Self-supervised encoders do wykrywania subtelnych zmian
- Wykorzystuje zarówno encodery jak i outputy modelu zadaniowego
- Automatyczna analiza przyczyn źródłowych (root cause analysis)

### 1.3 CheXstray (2023-2024)
**Opis**: Real-time multi-modal monitoring workflow dla AI w obrazowaniu medycznym.

**Cechy**:
- Statistical Process Control dla wykrywania out-of-distribution
- Monitoring w czasie rzeczywistym
- Szczególnie istotne dla krytycznych zastosowań medycznych

## 2. Metody Statystyczne

### 2.1 Maximum Mean Discrepancy (MMD)
**Status**: Nadal aktywnie rozwijane i używane w 2025

**Charakterystyka**:
- Test statystyczny określający czy dwie próbki pochodzą z różnych rozkładów
- Mierzy największą różnicę w oczekiwaniach po funkcjach w jednostkowej kuli RKHS
- **Zalety**:
  - Nie wymaga estymacji gęstości (w przeciwieństwie do KL divergence)
  - Łatwa estymacja jako średnia empiryczna
  - Czas obliczeniowy O(n²), dostępne aproksymacje liniowe
- **Implementacje 2025**: Pakiet R 'maotai' (kwiecień 2025)

**Zastosowania**:
- Wykrywanie przesunięć w NLP (NER datasets)
- Domain adaptive person re-identification
- Iris recognition z style transfer
- Predykcja mobilności ludzkiej

### 2.2 Kernel Two-Sample Tests (2024)
**Nowe badania**: Zachowanie asymptotyczne gdy wymiary i wielkości próbek → ∞

**Kernels**:
- Gaussian kernels
- Laplace kernels
- Energy distance jako przypadek szczególny

**Zastosowania**: Szczególnie skuteczne dla danych wysokowymiarowych gdzie liczba cech > liczba próbek

## 3. Praktyczne Podejścia do Porównywania Datasetów

### 3.1 Exploratory Data Analysis (EDA)
**Kluczowe aspekty do porównania**:
1. **Centre**: Punkt centralny rozkładu
2. **Spread**: Zmienność danych
3. **Shape**: Kształt rozkładu (symetria, liczba pików, skośność)
4. **Missing patterns**: Wzorce brakujących danych

### 3.2 Testy Statystyczne
- **ANOVA**: Porównanie średnich między grupami
- **Two-sample t-test**: Dla porównania średnich
- **F-test**: Dla porównania wariancji
- **Wald-Wolfowitz test**: Wykrywanie nie-losowości

### 3.3 Miary Odległości
- Euclidean distance
- Mahalanobis distance
- Kullback-Leibler divergence
- Wasserstein distance

## 4. Najlepsze Praktyki (2025)

### 4.1 Proces Porównywania
1. **Zdefiniuj cele**: Co chcesz odkryć poprzez porównanie?
2. **Wyczyść dane**: Usuń błędy, wartości odstające, duplikaty
3. **Dopasuj datasety**: Na kluczowych cechach (okres, populacja)
4. **Wybierz odpowiednie metody**: Zależnie od typu danych i celu
5. **Interpretuj ostrożnie**: Uwzględnij kontekst i ograniczenia

### 4.2 Strategie Mitygacji
- **Dla prevalence shift**: Lekka rekalibracja outputów
- **Dla covariate shift**: Domain adaptation lub fine-tuning modelu
- **Dla mixed shifts**: Kombinacja technik

## 5. Trendy i Kierunki Rozwoju

### 5.1 Automatyzacja
- Automatyczne wykrywanie typu shift
- Self-supervised metody nie wymagające etykiet
- Real-time monitoring systemów produkcyjnych

### 5.2 Efektywność
- Gradient-based metody o niskim koszcie obliczeniowym
- Liniowe aproksymacje dla metod kwadratowych
- Segmentacja danych dla efektywnego treningu

### 5.3 Interpretability
- Root cause analysis dla wykrytych przesunięć
- Wizualizacja różnic między rozkładami
- Explainable shift detection

## 6. Zastosowanie w Kaggle S5E7

Dla naszego problemu personality prediction, najbardziej obiecujące wydają się:

1. **MMD test** - do wykrycia subtelnych różnic między train a test
2. **Automatic Dataset Shift Identification** - do określenia typu shift
3. **Gradient-based scores** - do znalezienia najbardziej "problematycznych" próbek
4. **Self-supervised encoders** - do wykrycia ukrytych wzorców bez wykorzystania etykiet

## 7. Rekomendacje

1. Zastosuj MMD test do porównania train vs test z różnymi kernels
2. Użyj gradient-based scores do identyfikacji outlierów
3. Przeanalizuj prevalence vs covariate shift w naszych danych
4. Rozważ segmentację danych czasowych (jeśli istnieją)
5. Monitoruj stabilność predykcji w czasie

## Bibliografia

- Gretton et al. "A Kernel Two-Sample Test" (JMLR, aktualizacje 2024)
- "A Scalable Approach to Covariate and Concept Drift Management via Adaptive Data Segmentation" (arxiv, 2024)
- "Automatic dataset shift identification to support root cause analysis of AI performance drift" (arxiv, 2024)
- R package 'maotai' documentation (2025)
- Various conference papers from MICCAI, NeurIPS (2023-2024)