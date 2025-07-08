# Overfitting Analysis Report: CV 0.977705 vs LB 0.963562

**Date**: 2025-07-06 14:45  
**Gap**: -1.4% (CV much higher than LB)

## Executive Summary

The ensemble model achieves 97.77% CV score but only 96.36% on the leaderboard, indicating severe overfitting. Analysis reveals that the model is overfitting to 600 "ambiguous" training samples that are given 5-20x higher weights during training.

## Root Cause Analysis

### 1. The Enhanced Ambiguous 600 Samples

**Key Findings:**
- All 600 samples are from the training set (0 in test set)
- 86.2% are labeled as Extroverts (vs 74.0% in overall training)
- These samples are given `ambig_weight` between 5.0 and 20.0
- With weight=15, these 600 samples have 33.4% influence on model training

**Problem**: The model learns patterns specific to these 600 training samples that don't generalize to the test set.

### 2. Sample Weight Impact

| ambig_weight | Effective Samples | Training Influence |
|--------------|------------------|-------------------|
| 5.0          | 3,000            | 14.3%            |
| 10.0         | 6,000            | 25.1%            |
| 15.0         | 9,000            | 33.4%            |
| 20.0         | 12,000           | 40.1%            |

### 3. CV Score Inflation

The cross-validation uses the same weighted samples, so:
- CV folds contain the weighted ambiguous samples
- Model performs well on these heavily weighted patterns
- CV score is artificially inflated

### 4. Feature Distribution Issues

The "ambiguous" samples have different distributions:
- Lower `Time_spent_Alone`: 2.35 vs 3.14 overall
- Higher `Social_event_attendance`: 5.77 vs 5.27 overall
- Higher `Friends_circle_size`: 9.56 vs 8.00 overall
- Higher `Post_frequency`: 6.07 vs 4.98 overall

These are actually more extroverted patterns, not ambiguous!

## Why the Model Overfits

1. **Training-only patterns**: 600 samples exist only in training
2. **High weights**: 5-20x weight makes model focus on these patterns
3. **Biased distribution**: 86.2% extroverts in "ambiguous" set
4. **CV uses same weights**: Validation also sees inflated importance

## Recommendations

### Immediate Actions

1. **Remove ambig_weight parameter entirely** or reduce to 1.0-2.0
2. **Train models without enhanced_ambiguous_600.csv**
3. **Use standard uniform weights** for all samples

### Better Approaches

1. **Proper ambiguity detection**:
   - Use prediction confidence/entropy
   - Identify samples where P(Extrovert) ≈ 0.5
   - Don't rely on pre-selected IDs

2. **Stratified validation**:
   - Ensure ambiguous patterns appear in both train and validation
   - Use GroupKFold if patterns cluster

3. **Simpler models**:
   - Complex weighting schemes increase overfitting risk
   - Focus on robust features that generalize

4. **Corrected datasets**:
   - The corrected datasets (tc01-tc07) show promise
   - They fix actual labeling errors rather than upweighting ambiguous cases

## Expected Outcomes

By removing the ambiguous sample weights:
- CV score will drop to ~0.970-0.973
- LB score should improve to ~0.970+
- Gap between CV and LB will reduce to <0.5%

## Conclusion

The 1.4% CV-LB gap is caused by overfitting to 600 heavily weighted training samples that don't represent the test distribution. The solution is to remove these artificial weights and focus on models that generalize better.

The corrected datasets approach (fixing labeling errors) is more principled than the ambiguous weighting approach and should be pursued further.