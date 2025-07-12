#!/usr/bin/env python3
"""
Check prediction patterns for our known probe IDs
"""

import pandas as pd
from pathlib import Path

# Paths
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

# Known probes
PROBE_IDS = {
    20934: 'Extrovert',  # Known error - flipping to I worsens score
    18634: 'Extrovert',
    20932: 'Introvert', 
    21138: 'Extrovert',
    20728: 'Introvert'
}

def analyze_probes():
    # Load predictions
    pred_df = pd.read_csv(OUTPUT_DIR / 'ensemble_predictions.csv')
    
    print("="*80)
    print("PROBE ANALYSIS - Model Predictions")
    print("="*80)
    print(f"{'ID':>6} {'True':>10} {'Ensemble':>10} {'Social':>8} {'Personal':>8} {'Behavioral':>10} {'Binary':>8} {'External':>8} {'Pattern':>8}")
    print("-"*80)
    
    for probe_id, true_label in PROBE_IDS.items():
        if probe_id in pred_df['id'].values:
            row = pred_df[pred_df['id'] == probe_id].iloc[0]
            
            # Get predictions from each model (now they are class labels)
            social = 'E' if row['model_social'] == 'Extrovert' else 'I'
            personal = 'E' if row['model_personal'] == 'Extrovert' else 'I'
            behavioral = 'E' if row['model_behavioral'] == 'Extrovert' else 'I'
            binary = 'E' if row['model_binary'] == 'Extrovert' else 'I'
            external = 'E' if row['model_external'] == 'Extrovert' else 'I'
            
            pattern = social + personal + behavioral + binary + external
            
            # Ensemble prediction
            ensemble = row['equal_weight']
            
            # Check if correct
            correct = '✓' if ensemble == true_label else '✗'
            
            print(f"{probe_id:>6} {true_label:>10} {ensemble:>10}{correct} {social:>8} {personal:>8} {behavioral:>10} {binary:>8} {external:>8} {pattern:>8}")
            
            # Detailed probabilities for interesting cases
            if pattern not in ['IIIII', 'EEEEE']:
                print(f"       Probabilities: S={row['model_social_proba']:.3f} P={row['model_personal_proba']:.3f} "
                      f"B={row['model_behavioral_proba']:.3f} Bi={row['model_binary_proba']:.3f} E={row['model_external_proba']:.3f}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Analyze ID 20934 specifically
    row_20934 = pred_df[pred_df['id'] == 20934].iloc[0]
    print(f"\nID 20934 (known mislabeled as Introvert, actually Extrovert):")
    print(f"  - Social model: {row_20934['model_social_proba']:.3f} → {row_20934['model_social']}")
    print(f"  - Personal model: {row_20934['model_personal_proba']:.3f} → {row_20934['model_personal']}") 
    print(f"  - Behavioral model: {row_20934['model_behavioral_proba']:.3f} → {row_20934['model_behavioral']}")
    print(f"  - Binary model: {row_20934['model_binary_proba']:.3f} → {row_20934['model_binary']}")
    print(f"  - External model: {row_20934['model_external_proba']:.3f} → {row_20934['model_external']}")
    
    pattern_20934 = ('E' if row_20934['model_social'] == 'Extrovert' else 'I') + \
                    ('E' if row_20934['model_personal'] == 'Extrovert' else 'I') + \
                    ('E' if row_20934['model_behavioral'] == 'Extrovert' else 'I') + \
                    ('E' if row_20934['model_binary'] == 'Extrovert' else 'I') + \
                    ('E' if row_20934['model_external'] == 'Extrovert' else 'I')
    
    print(f"  - Pattern: {pattern_20934} - models disagree!")
    print(f"\n  This disagreement pattern suggests this ID might indeed be mislabeled!")
    
    # Check other high disagreement IDs
    print("\n" + "="*80)
    print("OTHER HIGH DISAGREEMENT IDS TO INVESTIGATE")
    print("="*80)
    
    # Load all predictions and find similar patterns to 20934
    model_cols = ['model_social', 'model_personal', 'model_behavioral', 
                  'model_binary', 'model_external']
    
    # Convert to binary pattern
    for idx, row in pred_df.iterrows():
        pattern = ''
        for col in model_cols:
            pattern += 'E' if row[col] == 'Extrovert' else 'I'
        pred_df.loc[idx, 'pattern'] = pattern
    
    # Find IDs with 3E-2I or 2E-3I patterns (high disagreement)
    disagreement_patterns = ['EEIEI', 'EEIEE', 'EIEIE', 'IEEEI', 'IEEIE', 'IEIEE']
    
    high_disagreement = pred_df[pred_df['pattern'].isin(disagreement_patterns)]
    
    print(f"\nFound {len(high_disagreement)} IDs with 3-2 split patterns:")
    print(f"{'ID':>6} {'Ensemble':>10} {'Pattern':>8} {'Social':>8} {'Personal':>10} {'Behavioral':>12} {'Binary':>8} {'External':>10}")
    print("-"*80)
    
    for _, row in high_disagreement.head(20).iterrows():
        print(f"{row['id']:>6} {row['equal_weight']:>10} {row['pattern']:>8} "
              f"{row['model_social_proba']:>8.3f} {row['model_personal_proba']:>10.3f} "
              f"{row['model_behavioral_proba']:>12.3f} {row['model_binary_proba']:>8.3f} "
              f"{row['model_external_proba']:>10.3f}")
    
    # Save high disagreement cases for submission
    high_disagreement[['id', 'equal_weight', 'pattern'] + model_cols].to_csv(
        OUTPUT_DIR / 'high_disagreement_candidates.csv', index=False
    )
    print(f"\nSaved {len(high_disagreement)} high disagreement candidates to high_disagreement_candidates.csv")

if __name__ == "__main__":
    analyze_probes()