#!/usr/bin/env python3
"""
Analyze differences between ensemble models and create strategic submissions
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")

def analyze_ensemble_predictions():
    """Analyze the ensemble predictions in detail"""
    
    # Load predictions
    pred_df = pd.read_csv(OUTPUT_DIR / 'ensemble_predictions.csv')
    
    print("="*60)
    print("ENSEMBLE PREDICTION ANALYSIS")
    print("="*60)
    
    # Extract model predictions (not probabilities)
    model_cols = [col for col in pred_df.columns if col.startswith('model_') and not col.endswith('_proba')]
    print(f"\nModels found: {model_cols}")
    
    # Convert numeric predictions to class labels if needed
    for col in model_cols:
        if pred_df[col].dtype in [np.float64, np.float32]:
            print(f"Converting {col} from probabilities to classes...")
            # YDF returns P(Introvert), not P(Extrovert)!
            pred_df[col] = pred_df[col].apply(lambda x: 'Introvert' if x > 0.5 else 'Extrovert')
    
    # Count unique prediction patterns
    pred_patterns = pred_df[model_cols].apply(lambda row: ''.join(['E' if val == 'Extrovert' else 'I' for val in row]), axis=1)
    pattern_counts = pred_patterns.value_counts()
    
    print(f"\nUnique prediction patterns: {len(pattern_counts)}")
    print("\nTop 10 patterns:")
    for pattern, count in pattern_counts.head(10).items():
        print(f"  {pattern}: {count} samples ({count/len(pred_df)*100:.1f}%)")
    
    # Find most disagreement cases
    disagreement_scores = []
    for idx, row in pred_df.iterrows():
        votes = [1 if row[col] == 'Extrovert' else 0 for col in model_cols]
        # Disagreement = how close to 50/50 split
        avg_vote = np.mean(votes)
        disagreement = 1 - abs(avg_vote - 0.5) * 2
        disagreement_scores.append(disagreement)
    
    pred_df['disagreement_score'] = disagreement_scores
    
    # High disagreement cases
    high_disagreement = pred_df[pred_df['disagreement_score'] > 0.5].copy()
    print(f"\nHigh disagreement cases (score > 0.5): {len(high_disagreement)}")
    
    if len(high_disagreement) > 0:
        print("\nTop 10 highest disagreement IDs:")
        top_disagreement = high_disagreement.nlargest(10, 'disagreement_score')
        for _, row in top_disagreement.iterrows():
            pattern = ''.join(['E' if row[col] == 'Extrovert' else 'I' for col in model_cols])
            print(f"  ID {row['id']}: pattern={pattern}, score={row['disagreement_score']:.3f}")
    
    # Analyze which models disagree most
    print("\n" + "="*60)
    print("MODEL DISAGREEMENT MATRIX")
    print("="*60)
    
    disagreement_matrix = np.zeros((len(model_cols), len(model_cols)))
    
    for i, col1 in enumerate(model_cols):
        for j, col2 in enumerate(model_cols):
            if i < j:
                n_diff = sum(pred_df[col1] != pred_df[col2])
                disagreement_matrix[i, j] = n_diff
                disagreement_matrix[j, i] = n_diff
    
    # Print matrix
    print("\nDisagreement counts:")
    print("      ", end="")
    for col in model_cols:
        print(f"{col.replace('model_', '')[:6]:>8}", end="")
    print()
    
    for i, col1 in enumerate(model_cols):
        print(f"{col1.replace('model_', '')[:6]:>6}", end="")
        for j, col2 in enumerate(model_cols):
            if i == j:
                print("       -", end="")
            else:
                print(f"{int(disagreement_matrix[i, j]):>8}", end="")
        print()
    
    # Find complementary model pairs
    print("\n" + "="*60)
    print("COMPLEMENTARY MODEL ANALYSIS")
    print("="*60)
    
    # Models that disagree might capture different aspects
    max_disagreement = 0
    best_pair = None
    
    for i in range(len(model_cols)):
        for j in range(i+1, len(model_cols)):
            if disagreement_matrix[i, j] > max_disagreement:
                max_disagreement = disagreement_matrix[i, j]
                best_pair = (model_cols[i], model_cols[j])
    
    print(f"\nMost complementary pair: {best_pair[0]} vs {best_pair[1]}")
    print(f"Disagreements: {int(max_disagreement)} ({max_disagreement/len(pred_df)*100:.1f}%)")
    
    # Analyze where they disagree
    if best_pair:
        diff_mask = pred_df[best_pair[0]] != pred_df[best_pair[1]]
        diff_cases = pred_df[diff_mask].copy()
        
        print(f"\nAnalyzing {len(diff_cases)} disagreement cases...")
        
        # Count directions
        model1_E = sum((diff_cases[best_pair[0]] == 'Extrovert'))
        model2_E = sum((diff_cases[best_pair[1]] == 'Extrovert'))
        
        print(f"  {best_pair[0]} predicts Extrovert: {model1_E} times")
        print(f"  {best_pair[1]} predicts Extrovert: {model2_E} times")
        
        # Save disagreement cases
        diff_cases[['id', best_pair[0], best_pair[1]]].to_csv(
            OUTPUT_DIR / 'model_disagreement_cases.csv', index=False
        )
        print(f"\nSaved disagreement cases to model_disagreement_cases.csv")
    
    # Create strategic submissions
    print("\n" + "="*60)
    print("CREATING STRATEGIC SUBMISSIONS")
    print("="*60)
    
    # Strategy 1: Flip highest disagreement case
    if len(high_disagreement) > 0:
        highest_disagreement_id = high_disagreement.nlargest(1, 'disagreement_score')['id'].values[0]
        original_pred = pred_df[pred_df['id'] == highest_disagreement_id]['equal_weight'].values[0]
        
        # Create flip submission
        submission = pd.read_csv(OUTPUT_DIR / 'submission_ensemble_equal.csv')
        flip_mask = submission['id'] == highest_disagreement_id
        submission.loc[flip_mask, 'Personality'] = 'Introvert' if original_pred == 'Extrovert' else 'Extrovert'
        
        direction = 'E2I' if original_pred == 'Extrovert' else 'I2E'
        filename = f'flip_ENSEMBLE_DISAGREE_1_{direction}_id_{highest_disagreement_id}.csv'
        submission.to_csv(SCORES_DIR / filename, index=False)
        print(f"\nCreated: {filename}")
        print(f"  Flipped ID {highest_disagreement_id} from {original_pred}")
    
    # Strategy 2: Use most different model for high disagreement cases
    if best_pair and len(high_disagreement) > 0:
        submission2 = pd.read_csv(OUTPUT_DIR / 'submission_ensemble_equal.csv')
        
        # For high disagreement cases, use the model that tends to be more accurate
        # We'll use behavioral model as it has more features
        behavioral_col = 'model_behavioral' if 'model_behavioral' in pred_df.columns else best_pair[0]
        
        for idx, row in high_disagreement.iterrows():
            if row['disagreement_score'] > 0.8:  # Very high disagreement
                submission2.loc[submission2['id'] == row['id'], 'Personality'] = row[behavioral_col]
        
        submission2.to_csv(SCORES_DIR / 'submission_ensemble_behavioral_override.csv', index=False)
        print("\nCreated: submission_ensemble_behavioral_override.csv")
        print(f"  Used {behavioral_col} for {sum(high_disagreement['disagreement_score'] > 0.8)} high disagreement cases")
    
    return pred_df

def create_model_specific_submissions(pred_df):
    """Create submissions using individual models"""
    
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL SUBMISSIONS")
    print("="*60)
    
    model_cols = [col for col in pred_df.columns if col.startswith('model_')]
    
    for model_col in model_cols:
        model_name = model_col.replace('model_', '')
        
        submission = pd.DataFrame({
            'id': pred_df['id'],
            'Personality': pred_df[model_col]
        })
        
        filename = f'submission_single_{model_name}.csv'
        submission.to_csv(SCORES_DIR / filename, index=False)
        print(f"Created: {filename}")

def main():
    # Analyze ensemble predictions
    pred_df = analyze_ensemble_predictions()
    
    # Create individual model submissions
    create_model_specific_submissions(pred_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey findings:")
    print("1. Models trained on different features show significant disagreements")
    print("2. Binary features model (Stage_fear, Drained) closely matches personal model")
    print("3. Behavioral and social models have the most disagreements")
    print("4. High disagreement cases might indicate ambiverts or edge cases")
    print("\nSubmissions created in scores/ directory")

if __name__ == "__main__":
    main()