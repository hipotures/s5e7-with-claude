# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle Playground Series S5E7 competition project for personality classification (Introvert/Extrovert prediction). The project has evolved through multiple breakthrough discoveries:

### Evolution of Understanding
1. **Initial Discovery (0.975708 barrier)**: Found that ~2.43% of data are ambiverts (MBTI types ISFJ/ESFJ) that cannot be definitively classified as I/E
2. **Null Pattern Breakthrough (97.997% CV)**: Discovered missing values encode personality - 63.4% of Introverts have nulls vs 38.5% of Extroverts
3. **Training Error Discovery**: Found ~1.18% labeling errors in training data, leading to corrected datasets strategy

### Current Best Results
- **Highest log score**: 0.975383 (CAT on train_corrected_07.csv)
- **Gap to target**: 0.000325 (only 0.03%!)
- **Best submission**: 0.975010 (XGB)

## Key Commands

### Running Scripts
```bash
# Navigate to scripts directory first
cd scripts/

# Direct script execution
python <script_name>.py

# For parallel optimization (uses 2xGPU + CPU)
python 20250705_2015_optimize_parallel_corrected_datasets.py

# Monitor GPU usage
nvidia-smi -l 1

# Check running processes
ps aux | grep optuna
```

### Common Development Tasks
- **New experiments**: Create scripts with timestamp prefix `YYYYMMDD_HHMM_<description>.py` (run `date +"%Y%m%d_%H%M"`)
- **Check results**: Look in `scores/` for submission files (now named with tc01, tc02, etc.)
- **Logs**: Check `output/optimization_logs/` for detailed optimization history

## Architecture & Structure

### Directory Layout
- `scripts/`: All experimental Python scripts (100+ files)
  - Early exploration: `20250703_*`
  - Ambivert discovery: `20250704_00*`
  - Training error analysis: `20250705_17*`
  - Parallel optimization: `20250705_2015_*`
- `scripts/output/`: Analysis results, corrected datasets
- `scores/`: Competition submissions (in root directory)
- `docs/`: Documentation and reports

### Key Discoveries & Strategies

#### 1. Ambivert Detection (Original Strategy)
```python
# Exact marker values (float precision matters!)
markers = {
    'Social_event_attendance': 5.265106088560886,
    'Going_outside': 4.044319380935631,
    'Post_frequency': 4.982097334878332,
    'Time_spent_Alone': 3.1377639321564557
}
```

#### 2. Null Pattern Strategy (97.997% CV)
- Create null indicators for all features
- Apply class-aware imputation
- Engineer null-based features (null_count, weighted_null_score)
- CatBoost achieves best results

#### 3. Corrected Datasets Strategy (Current Focus)
- 8 datasets with different correction levels (3 to 218 corrections)
- Best results from datasets 04 (psychological contradictions) and 07 (comprehensive)
- Parallel optimization on GPU server

### Parallel Optimization Setup
```python
# Fixed resource assignment for balanced execution
MODEL_RESOURCE_ASSIGNMENT = {
    'xgb': ['gpu0'],  # XGBoost always on GPU 0
    'gbm': ['gpu1'],  # LightGBM always on GPU 1
    'cat': ['cpu']    # CatBoost on CPU (16 threads)
}
```

## Data Paths
- Training data: `../../train.csv` (from scripts directory)
- Test data: `../../test.csv`
- Corrected datasets: `output/train_corrected_*.csv`
- Enhanced ambiguous samples: `/mnt/ml/kaggle/playground-series-s5e7/enhanced_ambiguous_600.csv`

## Key Scripts to Understand

### Training Error Analysis & Correction
1. **Find flip candidates**: `scripts/20250705_1715_find_all_flip_candidates.py`
2. **Analyze training errors**: `scripts/20250705_1720_analyze_training_errors.py`
3. **Create corrected datasets**: `scripts/20250705_1730_create_corrected_datasets.py`
4. **Parallel optimization**: `scripts/20250705_2015_optimize_parallel_corrected_datasets.py`

### Reports & Documentation
1. Read `docs/20250705_1730_CORRECTED_DATASETS_REPORT.md` for correction strategies
2. Read `scripts/output/20250705_2045_RESULTS_ANALYSIS.md` for latest results

## Important Notes

### GPU Server Execution
- Server has 200GB RAM and 2x RTX 4090 GPUs for hard ML space search

### Submission File Naming
- Format: `subm-SCORE-TIMESTAMP-MODEL-tc##.csv` (tc01, tc02, etc.)

### Display & Monitoring
- Rich library provides live monitoring during optimization
- Final results stay on screen until ESC pressed
- All statuses color-coded: Running (yellow), Completed (green), Pending (dim)

## Current Status & Next Steps

1. **Best corrected dataset**: train_corrected_07.csv (comprehensive - 218 corrections)
2. **Highest score**: 0.975383 (only 0.000325 from target!)
3. **Key insight**: Training data quality matters more than model complexity

### To Continue Optimization
```bash
cd scripts/
# Edit TIME_PER_DATASET back to 3600 (1 hour) if needed
python 20250705_2015_optimize_parallel_corrected_datasets.py
```

## XGBoost Migration Guide

### Important: XGBoost 3.x Changes
When using XGBoost, be aware of syntax changes between versions 2.x and 3.x:

#### GPU Configuration
```python
# OLD (XGBoost 2.x)
model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    gpu_id=0
)

# NEW (XGBoost 3.x)
model = xgb.XGBClassifier(
    device='cuda:0',  # or 'cuda:1' for second GPU
    tree_method='hist'  # automatically uses GPU when device='cuda'
)
```

#### Early Stopping
```python
# OLD (XGBoost 2.x)
model.fit(X_train, y_train, 
         eval_set=[(X_val, y_val)],
         early_stopping_rounds=10)

# NEW (XGBoost 3.x)
model = xgb.XGBClassifier(
    early_stopping_rounds=10  # Move to constructor!
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

#### Verbosity
```python
# OLD: silent=True
# NEW: verbosity=0 (0=silent, 1=warning, 2=info, 3=debug)
```

For full migration guide, see: `docs/00000000_0000_xgboost-migration-guide.md`

## Critical Discoveries

### Overfitting Source: enhanced_ambiguous_600.csv
- All 600 samples are from TRAINING set only (none in test)
- 86.2% are Extroverts (not actually ambiguous!)
- With ambig_weight=15, these 600 samples have 33.4% influence on training
- This causes massive overfitting: CV 0.977705 → LB 0.963562 (gap -1.4%)
- **Solution**: Train models WITHOUT ambiguous sample weights

### Winning Model Configuration (0.975708)
The exact configuration is unknown, but we know:
- Model: XGBoost (from filename)
- Score: 0.975708 (achieved by 240+ participants)
- Strategy: Likely involved ambivert detection (2.43% of data)
- Key insight: This score represents a mathematical ceiling in the data

## Project Memories

### Development Best Practices
- Keep original working scripts - create new files with updated timestamps
- Commit after significant analysis or working code
- Remove temporary model files before committing

### Script Creation
- Always check pwd (should be in scripts/)
- Use `date +"%Y%m%d_%H%M"` for timestamp
- Output files go to output/ with matching script name

### XGBoost Migration
- If using XGBoost, check version compatibility (see docs/00000000_0000_xgboost-migration-guide.md)