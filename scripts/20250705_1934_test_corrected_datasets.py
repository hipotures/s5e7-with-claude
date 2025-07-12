#!/usr/bin/env python3
"""Quick test to verify corrected datasets are loadable"""

import pandas as pd
import os

CORRECTED_DATASETS = [
    "train_corrected_01.csv",
    "train_corrected_02.csv", 
    "train_corrected_03.csv",
    "train_corrected_04.csv",
    "train_corrected_05.csv",
    "train_corrected_06.csv",
    "train_corrected_07.csv",
    "train_corrected_08.csv",
]

print("Testing corrected datasets...")
print("="*60)

for dataset in CORRECTED_DATASETS:
    path = f"output/{dataset}"
    try:
        df = pd.read_csv(path)
        intro_count = (df['Personality'] == 'Introvert').sum()
        extro_count = (df['Personality'] == 'Extrovert').sum()
        print(f"✓ {dataset}: {len(df)} rows, {intro_count} Intro, {extro_count} Extro")
    except Exception as e:
        print(f"✗ {dataset}: ERROR - {str(e)}")

print("\nChecking enhanced ambiguous file...")
ambig_path = "/mnt/ml/kaggle/playground-series-s5e7/enhanced_ambiguous_600.csv"
if os.path.exists(ambig_path):
    df = pd.read_csv(ambig_path)
    print(f"✓ Enhanced ambiguous: {len(df)} samples")
else:
    print(f"✗ Enhanced ambiguous file not found at: {ambig_path}")

print("\nAll tests complete!")