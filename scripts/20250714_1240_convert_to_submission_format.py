#!/usr/bin/env python3
"""
Convert probability files to submission format with Introvert/Extrovert labels.
"""

import pandas as pd
from pathlib import Path

def convert_to_submission(prob_file):
    """Convert probability file to submission format."""
    df = pd.read_csv(prob_file)
    
    # Convert probabilities to labels
    df['Personality'] = df['Extrovert'].apply(lambda x: 'Extrovert' if x > 0.5 else 'Introvert')
    
    # Keep only required columns
    submission = df[['id', 'Personality']]
    
    # Save with same name but _submission suffix
    output_file = str(prob_file).replace('.csv', '_submission.csv')
    submission.to_csv(output_file, index=False)
    print(f"Converted: {prob_file.name} → {Path(output_file).name}")
    
    return output_file

def main():
    """Convert all amplified files."""
    scores_dir = Path('../scores')
    amplified_files = list(scores_dir.glob('amplified_false_positives_*x_*.csv'))
    
    print(f"Found {len(amplified_files)} files to convert")
    
    for file in sorted(amplified_files):
        if '_submission' not in str(file):
            convert_to_submission(file)

if __name__ == "__main__":
    main()