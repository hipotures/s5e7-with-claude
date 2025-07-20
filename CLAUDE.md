# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle Playground Series S5E7 competition project for personality classification (Introvert/Extrovert prediction). The competition has a mathematical ceiling at 0.975708 due to inherent data ambiguity.

### Key Discoveries Through Competition

1. **Initial Ambivert Hypothesis (0.975708 barrier)**: ~2.43% of data thought to be ambiverts
2. **Null Pattern Discovery**: Missing values encode personality - 63.4% of Introverts have nulls vs 38.5% of Extroverts
3. **Training Error Analysis**: Found ~1.18% labeling errors, created corrected datasets
4. **Flip Testing Phase**: Discovered only 1 true error in public test set (ID 20934)

### Current Understanding
- **Public test set**: 20% of test data (1235/6175 records)
- **Confirmed errors in public**: 1 (ID 20934: E→I)
- **Error rate**: ~0.08% (extremely clean dataset)
- **Best achievable score**: 0.976518 (baseline 0.975708 + 1 flip)

## Analytical Guidelines

### Data Analysis Approach
- **Key Principle**: Approach results with scientific detachment
- Do not express emotional reactions to results
- Analyze results thoroughly, focusing on their contribution to the project
- Base score reference: 0.975708 (public baseline score)
  - Scores above this are improvements
  - Scores below this are regressions
- Maintain a skeptical, research-oriented mindset
- Seek confirmation of findings before publishing results

## Key Commands

### Running Scripts
```bash
# Always work from scripts directory
cd scripts/

# Create new experiment with timestamp
date +"%Y%m%d_%H%M"  # Copy this for filename prefix

# Run optimization (GPU server with 2x RTX 4090)
python 20250705_2015_optimize_parallel_corrected_datasets.py

# Monitor processes
nvidia-smi -l 1
ps aux | grep python
```

### Submission Testing
```bash
# Flip test files go in scores/
# Format: flip_<STRATEGY>_<N>_<DIRECTION>_id_<ID>.csv
# Example: flip_BOUNDARY_1_I2E_id_20932.csv
```

## Architecture & Key Files

### Critical Scripts
- `20250705_2015_optimize_parallel_corrected_datasets.py` - Main parallel optimization
- `20250708_0310_sliding_window_analysis.py` - Sequential pattern analysis
- `20250708_0320_analyze_data_batches.py` - Data batch analysis
- `20250708_0350_analyze_pattern_34.py` - Pattern analysis for IDs ending in 34

### Data Paths
- Train/Test: `../../train.csv`, `../../test.csv` (from scripts/)
- Corrected datasets: `scripts/output/train_corrected_*.csv`
- Submissions: `scores/` (root directory)
- Flip test results: `docs/20250710_0320_FLIP_TEST_RESULTS_TABLE.md`

### Parallel Optimization Setup
```python
MODEL_RESOURCE_ASSIGNMENT = {
    'xgb': ['gpu0'],  # XGBoost on GPU 0
    'gbm': ['gpu1'],  # LightGBM on GPU 1
    'cat': ['cpu']    # CatBoost on CPU
}
```

## Flip Testing Results Summary

### Tested Strategies (20 tests total)
- **Original E→I**: 5 tests, 1 hit (20934)
- **Mirror I→E**: 5 tests, 0 hits
- **Sequential**: 5 tests, 0 hits
- **Boundary**: 4 tests, 2 in public (1 error, 1 false positive)
- **Pattern 34**: 1 test, 0 hits

### Key Finding
- Only 1 confirmed error in public test (ID 20934)
- 2 false positives (18634, 20932)
- 85% of tests are in private set (unknown status)

## XGBoost 3.x Migration

```python
# NEW syntax (XGBoost 3.x)
model = xgb.XGBClassifier(
    device='cuda:0',  # Instead of gpu_id
    tree_method='hist',  # Instead of gpu_hist
    early_stopping_rounds=10  # In constructor, not fit()
)
```

## Important Insights

### Data Quality
- Dataset is extremely clean (0.08% error rate in public)
- Most "patterns" are statistical noise
- Focus on individual anomalies rather than systematic errors

### Submission Strategy
- Each flip changes score by ±0.000810
- Public score only reflects 20% of data
- Private test set performance is unknown

### Current Competition Status
- TOP 1: 0.977327 (likely 2 errors found)
- TOP 3: 0.978137 (likely 1 error found)
- Mathematical ceiling remains at 0.975708 for perfect predictions

## Flip Testing Analysis (scripts/0.975708/)

### Directory Structure
- **Baseline file**: `subm-0.96950-20250704_121621-xgb-381482788433-148.csv` (MD5: c4c8721683a44601b6e423462ee7db98)
- **minus-1/**: Files that scored 0.974898 (1 wrong flip)
- **minus-2/**: Files that scored 0.974089 (2 wrong flips equivalent)
- **trash/**: Duplicate files removed from analysis
- **output/**: Analysis results and comparisons

### Key Analysis Files
- `neutral_ids.txt`: IDs that don't affect score when flipped
- `m-1.txt`: IDs that decrease score by 1 point (0.000810)
- `compare_submissions.py`: Compares all submissions against baseline
- `analyze_baseline_differences.py`: Analyzes differences from baseline
- `verify_flip_counts.py`: Verifies expected flip counts in minus-1/minus-2

### Creating Flip Tests
```bash
# Create single flip test files
python create_flip_tests.py  # or create_rfecv_flip_tests.py

# Flip file naming convention:
# flip_<STRATEGY>_<N>_<DIRECTION>_id_<ID>.csv
# Direction: E2I (Extrovert to Introvert) or I2E (Introvert to Extrovert)
```

### Known ID Classifications
- **Neutral IDs** (confirmed): 24005, 18887, 20153, 23177
- **IDs giving -1**: 20932, 18634, 21359, 21800, 23872, 20934, 21138, 20728
- **ID giving +1**: 20934 when flipped I→E (the only true error found)

## Translation and Identity Profile 

### Professional Identity
- Experienced ML/DL engineer and scientist with strong theoretical foundations
- Proficient in implementing and deploying machine learning models
- Skilled in critical analysis of results and methodologies
- Well-versed in latest trends and scientific publications

### Working Principles
- Always provide balanced, objective analysis
- Identify potential limitations and biases
- Consider alternative approaches
- Maintain scientific skepticism
- Use precise, technical language
- Provide concrete numbers and references
- Ask clarifying questions when context is unclear

### Communication Guidelines
- Use technical, precise language
- Cite specific sources and publications
- Highlight both positive and negative aspects of solutions
- Propose concrete testing and validation strategies
- Maintain professional, research-oriented communication style