# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle Playground Series S5E7 competition project for personality classification (Introvert/Extrovert prediction). The project has achieved breakthrough accuracy of **97.997% CV** by discovering that missing values encode personality information.

### Critical Understanding
- **Original 0.975708 barrier** was thought to be a mathematical limit from 16 MBTI types → 2 classes reduction
- **NEW DISCOVERY**: Missing values are not random - they encode personality traits
  - 63.4% of Introverts have nulls vs only 38.5% of Extroverts
  - Missing "Drained_after_socializing" → 60% likely to be Introvert
- **Class-aware imputation** achieved 97.905% accuracy (+1.1% improvement)
- **Null-based features** pushed accuracy to 97.997% with CatBoost

## Key Commands

### Running Scripts
```bash
# Direct script execution
python scripts/<script_name>.py

# Fix dataset paths if needed
bash tools/fix_dataset_paths.sh
```

### Common Development Tasks
- **New experiments**: Create scripts with timestamp prefix `YYYYMMDD_HHMM_<description>.py` (run `date +"%Y%m%d_%H%M"`)
- **Analysis results**: Saved as CSV/JSON in `scripts/output/` directory
- **Submissions**: Generated in `subm/DATE_YYYYMMDD/` directory
- **GPU optimization**: Use Optuna with parameters at top of script (N_TRIALS_XGB, etc.)

### Running GPU Scripts
```bash
# For long-running GPU scripts (Optuna, neural networks)
CUDA_VISIBLE_DEVICES=0 python scripts/20250705_0140_deep_optuna_null_aware_gpu.py

# Monitor GPU usage
nvidia-smi -l 1
```

## Architecture & Structure

### Directory Layout
- `scripts/`: All experimental Python scripts (100+ files)
  - Early exploration: `20250703_*`
  - Ambivert discovery: `20250704_00*`
  - Optimization strategies: `20250704_01-02*`
  - Advanced ML: `20250704_19-20*`
  - **Null pattern breakthrough**: `20250705_*`
- `scripts/output/`: Analysis results, engineered features
- `subm/`: Competition submissions
- `docs/`: Documentation and reports
- `tools/`: Utility scripts

### Key Discoveries
1. **Null Pattern Breakthrough**: Missing values encode personality - 63.4% of Introverts have nulls vs 38.5% of Extroverts
2. **Class-Aware Imputation**: Imputing based on personality class achieves 97.905% accuracy
3. **Critical Features with Nulls**: 
   - "Drained_after_socializing" null → 60% Introvert (strongest signal)
   - "Stage_fear" (92.7% importance after null engineering)
   - Null indicators contribute significantly to predictions
4. **Final Performance**: CatBoost 97.997% CV, Ensemble 97.868% CV

### Ambivert Detection Patterns
```python
# Exact marker values (float precision matters!)
markers = {
    'Social_event_attendance': 5.265106088560886,
    'Going_outside': 4.044319380935631,
    'Post_frequency': 4.982097334878332,
    'Time_spent_Alone': 3.1377639321564557
}

# Behavioral pattern for ambiguous cases
ambiguous = (
    (df['Time_spent_Alone'] < 2.5) & 
    (df['Social_event_attendance'].between(3, 4)) &
    (df['Friends_circle_size'].between(6, 7))
)
```

### Important Classes/Patterns
```python
# AmbivertHandler - Core detection logic
class AmbivertHandler:
    AMBIVERT_MARKER = 0.12345678
    BOUNDARY_MARKER = 0.87654321
    # Detects ambiguous cases using special markers
```

### Script Structure Pattern
Each experimental script follows:
```python
# PURPOSE: [description]
# HYPOTHESIS: [theory being tested]
# EXPECTED: [anticipated outcome]
# RESULT: [actual findings]
```

## Data Paths
- Training data: `../train.csv`
- Test data: `../test.csv`
- Sample submission: `../sample_submission.csv`

## Key Libraries Used
- ML Models: xgboost, lightgbm, autogluon, scikit-learn
- Deep Learning: tensorflow/keras
- Optimization: optuna
- Data Processing: pandas, numpy
- Visualization: matplotlib, seaborn

## Key Scripts to Understand

### Original Ambivert Strategy (0.975708 ceiling)
1. **Ambivert Discovery**: `scripts/20250704_0007_ambivert_detector.py`
2. **Error Pattern Analysis**: `scripts/20250703_2357_analyze_errors_pattern.py`

### Null Pattern Breakthrough (97.997% accuracy)
1. **Null Pattern Analysis**: `scripts/20250705_0122_null_pattern_personality.py`
2. **Imputation Strategies**: `scripts/20250705_0124_null_imputation_strategies.py`
3. **Feature Engineering**: `scripts/20250705_0126_null_feature_engineering.py`
4. **Final Model**: `scripts/20250705_0128_null_aware_breakthrough_model.py`
5. **Deep Optimization**: `scripts/20250705_0140_deep_optuna_null_aware_gpu.py`

## Understanding the Project

1. Read `README_CLAUDE.md` - message from you to you
2. Read `docs/20250705_0137_BREAKTHROUGH_REPORT.md` for latest null pattern breakthrough
3. Read `docs/20250704_2058_ANALYSIS_SUMMARY.md` for original project overview
4. Read `docs/20250705_0120_NULL_ANALYSIS_PLAN.md` for null analysis methodology

## Breakthrough Implementation Steps

### Current Best Strategy (97.997% accuracy)
1. **Create null indicators** for all features (especially Drained_after_socializing)
2. **Apply class-aware imputation** - impute based on personality class during training
3. **Engineer null-based features**:
   - has_drained_null, null_count, weighted_null_score
   - Pattern features: pattern_only_drained, no_nulls
4. **Use CatBoost** as primary model (achieves 97.997% CV)
5. **Apply special rules**:
   - Missing Drained + low prob → Introvert
   - No nulls + high prob → Extrovert
   - High weighted null score → Introvert

## Important Notes
1. This is a git repository - use git for version control
2. No formal testing framework - scripts are experimental/exploratory
3. No dependency management file - libraries assumed to be installed
4. **Latest breakthrough**: Null patterns are the key, not ambivert detection
5. **CatBoost performs best** on this dataset (97.997% CV without extensive optimization)
6. **Optimal ensemble weights** (from Optuna): XGB=52.4%, LGB=13.3%, CAT=34.3%

## Project Memories

### Development Strategies
- Jeśli tworzysz skrypt który działa i produkuje dane i chcesz uruchomić ponownie, modyfikująć w nim rzeczy nie związane z błędnym działaniem, utwórz kolejny plik z nowszą datą i ten plik modyfikuj. Oryginalny plik musi zostać.
- Commituj po każdej dłuższej analizie, jesli wyprodukowałeś nowy, działający kod.

### Workspace Management
- Przed commitem usuń pliki utworzone przez modele do obliczeń (katalogi robocze)

### Server Resources
- Mamy dostęp do odrebnego serwera obliczeniowego 200GB RAM i 2x4090 GPU. Jeśli jest potrzeba dłuższych obliczeń (np. dla optuna), zgłoś potrzebę dłuższej optymalizacji.

### File Path and Access
- Jeżeli tworzysz skrypt w katalogu scripts/ (i w tym katalogu będzie uruchamiany /wyniki zapisujesz do katalogu output/ ) to dostęp do plików (train.csv, test.csv i pozostałych) jest w lokalizacji względnej tak: read_csv("../../train.csv")

### Development Notes
- Jeżeli używamy metod Deep Learning, stosujemy pytorch

### Output File Generation
- Kod na output ma wygenerować plik w katalogu output/ (katalog lokalny względem tego, gdzie ja ręcznie uruchamiam kod). Plik ma mieć nazwę analogiczną do skryptu, który go uruchamia, przykładowo 20250705_0955_advanced_imputation_test.py -> zapisuje do output/20250705_0955_advanced_imputation_test.txt

### Script Creation Best Practices
- Przed stworzeniem skryptu i ewentualnym jego uruchomieniem sprwdzaj 1. Gdzie jesteś pwd (masz być w katalogu scripts/) . 2. Uruchom date +"%Y%m%d_%H%M" żeby wiedzieć, jak nazwać plik skryptu.

## XGBoost Migration Guidance
- Jeśli w skryptach używasz biblioteki xgboost, musisz zapoznać się z najnowaszymi zmianami w wersji 3: 2->3 docs/00000000_0000_xgboost-migration-guide.md