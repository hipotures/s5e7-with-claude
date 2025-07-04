# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle Playground Series S5E7 competition project for personality classification (Introvert/Extrovert prediction). The project has discovered a mathematical accuracy ceiling of 0.975708 due to ~2.43% ambiguous cases (ambiverts) that cannot be classified from the given features.

### Critical Understanding
- **The 0.975708 barrier is NOT random** - it's a mathematical limit from 16 MBTI types → 2 classes (I/E) reduction
- **2.43% of data is ambiguous** (mainly ISFJ/ESFJ types that can be either I or E)
- **96.2% of ambiverts are labeled as Extrovert** in the dataset
- Without full MBTI dimensions (N/S, T/F, J/P are missing), these cases cannot be correctly classified

## Key Commands

### Running Scripts
```bash
# Direct script execution
python scripts/<script_name>.py

# Fix dataset paths if needed
bash tools/fix_dataset_paths.sh
```

### Common Development Tasks
- **New experiments**: Create scripts with timestamp of file creation (run `date` command) prefix `YYYYMMDD_HHMM_<description>.py`
- **Analysis results**: Saved as CSV/JSON in scripts/output directory
- **Submissions**: Generated in `subm/DATE_YYYYMMDD` directory

## Architecture & Structure

### Directory Layout
- `scripts/`: All experimental Python scripts (76+ files)
  - Early exploration: `20250703_*`
  - Ambivert discovery: `20250704_00*`
  - Optimization strategies: `20250704_01-02*`
  - Advanced ML: `20250704_19-20*`
- `subm/`: Competition submissions
- `tools/`: Utility scripts

### Key Discoveries
1. **Accuracy Ceiling**: 0.975708 (mathematical limit, 240+ people achieved exactly this)
2. **Ambiguous Cases**: 2.43% of data (ambiverts/ISFJ-ESFJ boundary)
3. **Critical Features**: 
   - "Drained_after_socializing" (>50% importance)
   - "Stage_fear" (second most important)
   - Together provide >90% predictive power
4. **Breakthrough Strategy**: Detect ambiguous cases and apply 96.2% Extrovert rule

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

1. **Ambivert Discovery**: `scripts/20250704_0007_ambivert_detector.py`
2. **Error Pattern Analysis**: `scripts/20250703_2357_analyze_errors_pattern.py`
3. **Breakthrough Strategy**: `scripts/20250704_0008_ambivert_breakthrough_strategy.py`
4. **Simple Test**: `scripts/20250704_2009_test_breakthrough_simple.py`
5. **Iterative Optimization**: `scripts/20250704_0246_optimize_ambiguous_iterative.py`

## Understanding the Project

1. Read `README_CLAUDE.md` - message from you to you
2. Read `docs/20250704_2058_ANALYSIS_SUMMARY.md` for project overview
3. Read `docs/20250704_2013_RAPORT.md` for implementation details

## Breakthrough Implementation Steps

1. **Detect ambiverts** (2.43% of data)
2. **Apply 96.2% rule** - if uncertain ambivert, predict Extrovert
3. **Use dynamic thresholds**:
   - Ambiverts: 0.42-0.45
   - Others: 0.50
4. **Weight samples** - 10-20x higher weight for ambiverts during training

## Important Notes
1. This is a git repository - use git for version control
2. No formal testing framework - scripts are experimental/exploratory
3. No dependency management file - libraries assumed to be installed
4. Configuration is embedded in scripts rather than external files
5. The project focuses on understanding data structure rather than model optimization
6. This is NOT a "better model" problem - it's about understanding the information-theoretic limit