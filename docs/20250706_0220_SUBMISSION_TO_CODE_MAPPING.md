# Mapowanie Submissions do Kodów Źródłowych

## Top Submissions (Cel osiągnięty: 0.975708)

### 1. subm-0.96950-20250704_121621-xgb-381482788433-148.csv
- **Score**: 0.975708 ✓
- **Model**: XGBoost (5 trees, depth 2)
- **Źródło**: 
  - Prawdopodobnie z sesji optymalizacji Optuna
  - Hash: 381482788433, Trial: 148
- **Strategia**: Ambivert detection + simple model
- **Related scripts**:
  - `20250704_0007_ambivert_detector.py`
  - `20250704_0008_ambivert_breakthrough_strategy.py`

### 2. subm-0.96949-20250704_120105-sta-c279d162b246-0.csv
- **Score**: 0.975708 ✓
- **Model**: STA (Stacking/Threshold Adjustment)
- **Źródło**: 
  - Hash: c279d162b246, Trial: 0
- **Strategia**: Dynamic thresholds (0.42-0.45 for ambiverts)

### 3. 003_perfect_243_rule.csv
- **Score**: 0.974898
- **Źródło**: Manual rule-based approach
- **Problem**: Zbyt agresywna reguła 2.43%

## Ostatnie Submissions (5-6 lipca 2025)

### Flip Strategies
1. **flip_top_15_candidates.csv** (0.972469)
   - Źródło: `20250705_1715_find_all_flip_candidates.py`
   - Flipowano 15 rekordów

2. **02_final_flip_all_5_flips_submission.csv** (0.974898)
   - Źródło: `20250705_1554_final_flip_strategy.py`
   - Flipowano IDs: 18876, 20363, 20934, 20950, 21008

### Corrected Datasets Submissions
1. **subm-0.97528-20250705_205457-cat-tc07.csv** (LB: 0.974089)
   - Źródło: `20250705_2015_optimize_parallel_corrected_datasets.py`
   - Dataset: train_corrected_07.csv (218 corrections)
   - Problem: CV overfitting (CV: 0.97528 vs LB: 0.974089)

2. **subm-0.96950-20250705_205210-cat-tc03.csv** (0.969xx)
   - Dataset: train_corrected_03.csv (6 corrections)
   - Zbyt konserwatywne podejście

### Ensemble Approaches
1. **01_ensemble_cluster_pattern_submission.csv** (0.973279)
   - Źródło: Prawdopodobnie `20250705_1542_cluster_aware_strategy.py`

2. **submission_RFECV_optimal_score0.9678_features13.csv** (0.974089)
   - Źródło: Feature selection approach

## Pliki Optuna Studies

Wszystkie wyniki są zapisane w SQLite:
```
output/optuna_studies/
├── e8a823232109.db  (xgb + tc01)
├── 0b5b01cca100.db  (gbm + tc01)
├── fb8ffe0a3ba6.db  (cat + tc01)
└── ... (24 baz danych total)
```

## Najnowsze Wyniki (po zakończeniu optymalizacji)

Z Optuna optimization (8834 trials):
- **GBM tc07**: 0.975869 (przekroczył cel o 0.000161!) ✓✓
- **XGB tc07**: 0.975815 ✓
- **CAT tc07**: 0.975815 ✓

Te modele czekają na wygenerowanie submissions przez:
- `20250706_0157_create_ensemble_from_optuna.py`