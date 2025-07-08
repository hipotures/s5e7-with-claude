# Ambiguous Samples Analysis Report
**Date**: 2025-07-06  
**Purpose**: Document the discovery and usage of ambiguous training samples that achieved 0.96950 accuracy

## Executive Summary

The breakthrough accuracy of 0.96950 was achieved using 450 pre-identified ambiguous training samples combined with Optuna hyperparameter optimization. This report documents:
1. How the ambiguous samples were created
2. The optimization approach that used them
3. The specific parameters that achieved top results
4. File locations and scripts involved

## Key Files and Scripts

### 1. Ambiguous Samples File
- **File**: `/mnt/ml/kaggle/playground-series-s5e7/most_ambiguous_2.43pct.csv`
- **Size**: 450 samples (2.43% of training data)
- **Created by**: `scripts/20250704_0019_find_exact_mapping_rule.py`
- **Creation date**: 2025-07-04

### 2. Optimization Script
- **Script**: `scripts/20250704_0000_optimize_ambiguous_fine_tune.py`
- **Purpose**: Universal Kaggle optimizer with fixed resource assignment
- **Key features**:
  - Multi-GPU support (XGBoost on GPU0+GPU1)
  - Optuna hyperparameter optimization
  - Continues from previous optimization sessions
  - Generates submission files with pattern: `subm-{score}-{timestamp}-{model}-{session}-{trial}.csv`

### 3. Optuna Journal
- **File**: `/mnt/ml/kaggle-helper/study-xgb-finetune.journal`
- **Contains**: Full optimization history including all trials and parameters

### 4. Best Submission
- **File**: `subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv`
- **Score**: 0.96950
- **Model**: XGBoost
- **Session ID**: 381482788433
- **Trial**: 148

## How Ambiguous Samples Were Identified

The script `20250704_0019_find_exact_mapping_rule.py` identified ambiguous samples by:

1. **Creating scoring functions**:
   ```python
   typical_extrovert_score = (
       (5 - Time_spent_Alone) + 
       Social_event_attendance + 
       Going_outside + 
       Friends_circle_size/3 + 
       Post_frequency
   )
   
   typical_introvert_score = (
       Time_spent_Alone + 
       (10 - Social_event_attendance) + 
       (10 - Going_outside) + 
       (30 - Friends_circle_size)/3 + 
       (10 - Post_frequency)
   )
   ```

2. **Computing ambiguity score**:
   ```python
   ambiguity_score = abs(typical_extrovert_score - typical_introvert_score)
   ```

3. **Selecting most ambiguous**: The 450 samples with the **lowest ambiguity scores** (closest to the boundary)

## Optimization Parameters (Trial 148)

### Ambiguity Detection Parameters
- **alone_thresh**: 2.1895448626209046
- **social_thresh**: 5.04647098357926
- **friends_thresh**: 8
- **ambig_score_thresh**: 0.37958112294168955
- **proba_low_thresh**: 0.2252053144258133
- **proba_high_thresh**: 0.4925814615875499
- **ambig_weight**: 6.087743435546929

### XGBoost Model Parameters
- **n_estimators**: 1808
- **learning_rate**: 0.006582324928661372
- **max_depth**: 10
- **subsample**: 0.6076502743755241
- **colsample_bytree**: 0.8936799297310533
- **reg_lambda**: 3.4590042329528896
- **reg_alpha**: 6.601702966168614
- **gamma**: 0.017816855604698388
- **min_child_weight**: 6

### Key Characteristics
- Very low learning rate with many estimators (fine-tuning approach)
- Deep trees (max_depth=10) for complex patterns
- Strong regularization to prevent overfitting
- Conservative subsampling for stability

## How the Optimization Works

1. **Loads ambiguous samples**:
   ```python
   ambiguous_file = os.path.join(DATA_DIR, "most_ambiguous_2.43pct.csv")
   ambiguous_df = pd.read_csv(ambiguous_file)
   ambiguous_ids = set(ambiguous_df['id'].values)
   ```

2. **Applies sample weights**:
   ```python
   sample_weights = np.ones(len(train_df))
   sample_weights[train_df[ID_COLUMN].isin(ambiguous_ids)] = ambig_weight
   ```

3. **Creates ambiguity features**:
   - ambiguous_pattern
   - dist_to_ambiguous
   - ambiguity_score
   - high_ambiguity

4. **Uses special rules**:
   - Forces high ambiguity cases to Extrovert
   - Reverts very low probability cases back to Introvert

## Submission File Naming Convention

Pattern: `subm-{score:.5f}-{timestamp}-{model}-{study_name}-{trial_number}.csv`

Example: `subm-0.96950-20250704_121621-xgb-381482788433-148.csv`
- **score**: 0.96950 (5 decimal places)
- **timestamp**: 20250704_121621 (YYYYMMDD_HHMMSS)
- **model**: xgb (model abbreviation)
- **study_name**: 381482788433 (session identifier)
- **trial_number**: 148 (Optuna trial)

## Commands to Reproduce

1. **Run the optimization**:
   ```bash
   cd scripts/
   python 20250704_0000_optimize_ambiguous_fine_tune.py
   ```

2. **Check Optuna journal**:
   ```bash
   grep "trial_id\":148" /mnt/ml/kaggle-helper/study-xgb-finetune.journal
   ```

3. **Find submissions**:
   ```bash
   find subm/ -name "*381482788433*" -type f
   ```

## Recent Discoveries vs Original Approach

### Original Approach (0.96950)
- Used 450 ambiguous samples based on boundary proximity
- Applied higher weights to ambiguous samples
- Fine-tuned XGBoost with 1808 estimators

### New Discoveries (Post-Analysis)
1. **Null Patterns**: 63.4% introverts have nulls vs 38.5% extroverts
2. **Cluster Analysis**: Clusters 2 & 7 have 5.6% error rates
3. **Critical Finding**: Only 5 records separate 0.975708 from 0.976518

## Recommendations

1. **Combine approaches**: Use original ambiguous samples + null patterns
2. **Update ambiguity detection**: Include null-based features in ambiguity score
3. **Cluster-aware sampling**: Give higher weights to samples in problematic clusters
4. **Focus on the 5 critical records**: Identify and ensure correct classification

## File Locations Summary

```
/mnt/ml/
├── kaggle/playground-series-s5e7/
│   └── most_ambiguous_2.43pct.csv          # 450 ambiguous samples
├── kaggle-helper/
│   └── study-xgb-finetune.journal          # Optuna optimization history
└── competitions/2025/playground-series-s5e7/WORKSPACE/
    ├── scripts/
    │   ├── 20250704_0019_find_exact_mapping_rule.py    # Creates ambiguous samples
    │   └── 20250704_0000_optimize_ambiguous_fine_tune.py # Optimization script
    └── subm/20250705/
        └── subm-0.96950-20250704_121621-xgb-381482788433-148.csv # Best submission
```

## Next Steps

1. Create enhanced ambiguous samples file combining all insights
2. Re-run optimization with updated ambiguity detection
3. Test cluster-specific strategies on the 450 ambiguous samples
4. Analyze the 5 critical records that make the difference