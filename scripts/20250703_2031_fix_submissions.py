#!/usr/bin/env python3
"""
PURPOSE: Fix label mapping in existing submission files by swapping Introvert/Extrovert labels
HYPOTHESIS: Previous submissions have inverted labels due to encoding issue
EXPECTED: Swapping labels will improve submission scores significantly
RESULT: Fixed submissions with corrected label mapping for Kaggle submission
"""

import pandas as pd
import glob
import os

# Find all submission files
submission_files = glob.glob("submissions/submission_*.csv")

print(f"Found {len(submission_files)} submission files to fix")

for file in submission_files:
    print(f"\nProcessing: {file}")
    
    # Load submission
    df = pd.read_csv(file)
    
    # Check current distribution
    before_dist = df['Personality'].value_counts().to_dict()
    print(f"  Before: {before_dist}")
    
    # Swap labels
    df['Personality'] = df['Personality'].map({
        'Introvert': 'Extrovert',
        'Extrovert': 'Introvert'
    })
    
    # Check new distribution
    after_dist = df['Personality'].value_counts().to_dict()
    print(f"  After:  {after_dist}")
    
    # Save back
    df.to_csv(file, index=False)
    print(f"  ✓ Fixed and saved")

print("\n✅ All submissions fixed!")