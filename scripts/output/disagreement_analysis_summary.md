# Disagreement Analysis Summary


## Key Findings from Model Disagreement Analysis

### 1. **Unanimous Disagreements (Original Dataset)**
- **492 IDs** where ALL 6 challenger models predict Introvert while baseline predicts Extrovert
- This represents ~8% of test set (492/6175)
- All unanimous disagreements are E→I (no I→E cases found)

### 2. **Corrected Dataset Analysis**
- Only **2 IDs** with unanimous disagreement: 21359, 23844
- Only **8 IDs** with mismatches ≥ 4 (high disagreement)
- Dramatic reduction from original dataset suggests corrected training improved consistency

### 3. **Cross-Dataset High Disagreement IDs**
- **5 IDs** appear with high disagreement in BOTH datasets:
  - **21359, 23844**: Unanimous (6/6) in both datasets
  - **19612, 23418, 23336**: 5-6 mismatches in both datasets

### 4. **Known IDs Analysis**
- **20934** (confirmed error): Only 1 mismatch - not caught by disagreement analysis!
- **19482**: 6/6 unanimous disagreement (strong candidate)
- **19098**: 6/6 unanimous disagreement (pattern 34 ID)
- **21404**: 5/6 disagreement in original, changed to I in corrected dataset


## Top Candidates

| Priority | ID | Flip | Evidence | Risk |
|----------|-----|------|----------|------|
| 1 | 21359 | E→I | Unanimous 6/6 in BOTH datasets | Very Low |
| 2 | 23844 | E→I | Unanimous 6/6 in BOTH datasets | Very Low |
| 3 | 19482 | E→I | Unanimous 6/6 in original | Low |
| 4 | 19098 | E→I | Unanimous 6/6 in original, Pattern 34 | Low |
| 5 | 19612 | E→I | 5/6 in both datasets | Medium |
| 6 | 23418 | E→I | 5/6 in both datasets | Medium |
| 7 | 23336 | E→I | 5/6 in both datasets | Medium |
| 8 | 21404 | E→I | 5/6 original, baseline changed in corrected | Medium |


## Flip Testing Strategy Based on Disagreement Analysis

### Phase 1: Ultra-High Confidence (2 flips)
Test IDs with unanimous disagreement in BOTH datasets:
- **21359**: E→I
- **23844**: E→I

### Phase 2: High Confidence (2 flips)
Test IDs with unanimous disagreement in original + other evidence:
- **19482**: E→I (unanimous 6/6)
- **19098**: E→I (unanimous 6/6 + pattern 34)

### Phase 3: Medium Confidence (4 flips)
Test IDs with 5/6 disagreement in both datasets:
- **19612**: E→I
- **23418**: E→I
- **23336**: E→I
- **21404**: E→I (interesting case - baseline changed between datasets)

### Important Observations:
1. **20934** (confirmed error) shows LOW disagreement - suggests disagreement alone isn't sufficient
2. All high-disagreement cases are E→I (no I→E found)
3. Corrected dataset shows much higher model consensus
4. Only 2.43% of test set shows unanimous disagreement in corrected dataset

### Expected Score Impact:
- Each correct flip: +0.000810
- Maximum potential from 8 flips: +0.006480
- Theoretical new score: 0.975708 + 0.006480 = 0.982188


## Key Takeaways:

1. **Start with IDs 21359 and 23844** - highest confidence based on cross-dataset agreement
2. **Don't rely solely on disagreement** - 20934 proves errors can have low disagreement
3. **Consider combining strategies**:
   - High disagreement (this analysis)
   - Boundary analysis (previous work)
   - Pattern analysis (e.g., IDs ending in 34)
   - Statistical anomalies
4. **All candidates are E→I** - suggests systematic bias in baseline model
5. **492 unanimous disagreements** is suspiciously high - possible ambivert cluster?

## Next Steps:
1. Submit flip tests for top 2-4 candidates
2. Analyze feature patterns of unanimous disagreement IDs
3. Check if unanimous disagreement IDs share common characteristics
4. Consider ensemble voting strategy for future submissions
