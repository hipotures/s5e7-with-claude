#!/usr/bin/env python3
"""
PURPOSE: Rename submission files with priority numbering for easy tracking
HYPOTHESIS: Organizing submissions by priority will help systematic testing on Kaggle
EXPECTED: Rename submission files with numbered prefixes for ordered submission
RESULT: Created prioritized naming scheme for submission files
"""

import os
import shutil
from datetime import datetime

# Create backup directory
backup_dir = f"submissions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(backup_dir, exist_ok=True)

# Priority groups
priority_files = {
    # Top priority - most likely candidates
    1: [
        'submission_DT_depth2_simple.csv',
        'submission_manual_DT_rules.csv',
    ],
    
    # Very simple XGBoost models (1-5 trees, depth 1-2)
    2: [
        'submission_SIMPLE_n1_d1_lr1.0_rs42.csv',
        'submission_SIMPLE_n2_d1_lr1.0_rs42.csv',
        'submission_SIMPLE_n3_d1_lr1.0_rs42.csv',
        'submission_SIMPLE_n1_d2_lr1.0_rs42.csv',
        'submission_SIMPLE_n2_d2_lr1.0_rs42.csv',
    ],
    
    # Single feature models
    3: [
        'submission_SINGLE_Drained_after_socializing_n1.csv',
        'submission_SINGLE_Drained_after_socializing_n2.csv',
        'submission_SINGLE_Stage_fear_n1.csv',
    ],
    
    # Decision tree variations
    4: [
        'submission_DT_d1_gini_rs42.csv',
        'submission_DT_d2_gini_rs42.csv',
        'submission_DT_d2_entropy_rs42.csv',
    ],
    
    # Other simple XGBoost
    5: [
        'submission_SIMPLE_n5_d1_lr1.0_rs42.csv',
        'submission_SIMPLE_n5_d2_lr1.0_rs42.csv',
        'submission_SIMPLE_n10_d1_lr1.0_rs42.csv',
        'submission_SIMPLE_n10_d2_lr1.0_rs42.csv',
    ],
}

# Get all submission files
all_submissions = [f for f in os.listdir('.') if f.startswith('submission_') and f.endswith('.csv')]
print(f"Found {len(all_submissions)} submission files")

# Backup all files first
print(f"\nBacking up files to {backup_dir}/")
for file in all_submissions:
    shutil.copy2(file, os.path.join(backup_dir, file))

# Rename priority files
counter = 1
renamed_files = []

print("\nRenaming priority files:")
print("="*60)

for priority, files in sorted(priority_files.items()):
    print(f"\nPriority {priority} group:")
    for file in files:
        if file in all_submissions:
            new_name = f"{counter:03d}_{file}"
            os.rename(file, new_name)
            renamed_files.append(file)
            print(f"  {counter:03d}: {file}")
            counter += 1

# Rename remaining files
print("\nOther files:")
remaining_files = [f for f in all_submissions if f not in renamed_files]

# Sort remaining by type
simple_files = sorted([f for f in remaining_files if 'SIMPLE' in f])
subsample_files = sorted([f for f in remaining_files if 'SUBSAMPLE' in f])
single_files = sorted([f for f in remaining_files if 'SINGLE' in f])
dt_files = sorted([f for f in remaining_files if 'DT' in f])
other_files = sorted([f for f in remaining_files if f not in simple_files + subsample_files + single_files + dt_files])

# Rename in groups
for group_name, group_files in [
    ("Simple XGBoost", simple_files),
    ("Subsample variations", subsample_files),
    ("Single feature", single_files),
    ("Decision Trees", dt_files),
    ("Other", other_files)
]:
    if group_files:
        print(f"\n{group_name}:")
        for file in group_files:
            new_name = f"{counter:03d}_{file}"
            os.rename(file, new_name)
            if counter <= 50:  # Show first 50
                print(f"  {counter:03d}: {file}")
            elif counter == 51:
                print(f"  ... (and {len(group_files) - 50} more)")
            counter += 1

# Create tracking file
with open('submission_order.txt', 'w') as f:
    f.write("SUBMISSION ORDER FOR KAGGLE\n")
    f.write("="*60 + "\n\n")
    f.write("Submit in this order and record scores:\n\n")
    f.write("Num | Filename | Kaggle Score | Notes\n")
    f.write("-"*60 + "\n")
    
    # List priority files
    counter = 1
    for priority, files in sorted(priority_files.items()):
        f.write(f"\n--- Priority {priority} ---\n")
        for file in files:
            if file in all_submissions:
                f.write(f"{counter:03d} | {file} | _____ | \n")
                counter += 1
    
    f.write("\n--- Remaining files ---\n")
    f.write("(Check submission files starting with numbers 001, 002, etc.)\n")

print(f"\n\nDONE! Renamed {counter-1} files")
print("\nTop priority files to submit first:")
print("-"*40)
for i in range(1, min(11, counter)):
    matching_files = [f for f in os.listdir('.') if f.startswith(f"{i:03d}_submission")]
    if matching_files:
        print(f"{i:03d}: {matching_files[0].replace(f'{i:03d}_', '')}")

print("\nFiles are now numbered 001_submission_*, 002_submission_*, etc.")
print("Submit them in order and track results in submission_order.txt")
print(f"\nBackup of original files saved in: {backup_dir}/")