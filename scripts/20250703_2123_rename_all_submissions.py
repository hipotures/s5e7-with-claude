#!/usr/bin/env python3
"""
PURPOSE: Rename ALL submission files with priority numbering for systematic testing
HYPOTHESIS: A comprehensive renaming system will help track all submission attempts
EXPECTED: Create numbered submission files with tracking spreadsheet
RESULT: Renamed all submissions with priority order and tracking files
"""

import os
import shutil
from datetime import datetime

# Get all submission files
all_submissions = sorted([f for f in os.listdir('.') if f.startswith('submission_') and f.endswith('.csv')])
print(f"Found {len(all_submissions)} submission files")

# Create backup directory
backup_dir = f"submissions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(backup_dir, exist_ok=True)

# Backup all files
print(f"\nBacking up files to {backup_dir}/")
for file in all_submissions:
    if os.path.exists(file):
        shutil.copy2(file, os.path.join(backup_dir, file))

# Define priority order
priority_order = []

# 1. Top priority - Most likely candidates
if 'submission_DT_depth2_simple.csv' in all_submissions:
    priority_order.append('submission_DT_depth2_simple.csv')
if 'submission_manual_DT_rules.csv' in all_submissions:
    priority_order.append('submission_manual_DT_rules.csv')

# 2. Very simple XGBoost (1-5 trees, depth 1-2)
simple_xgb = []
for n in [1, 2, 3, 4, 5]:
    for d in [1, 2]:
        pattern = f'_n{n}_d{d}_'
        simple_xgb.extend([f for f in all_submissions if 'SIMPLE' in f and pattern in f])
priority_order.extend(sorted(set(simple_xgb)))

# 3. Single feature models
single_feat = [f for f in all_submissions if 'SINGLE' in f]
# Sort by feature importance (Drained first)
drained_single = sorted([f for f in single_feat if 'Drained' in f])
other_single = sorted([f for f in single_feat if 'Drained' not in f])
priority_order.extend(drained_single + other_single)

# 4. Decision Trees
dt_files = sorted([f for f in all_submissions if f.startswith('submission_DT_') and f not in priority_order])
priority_order.extend(dt_files)

# 5. Simple XGBoost (6-20 trees)
for n in range(6, 21):
    pattern = f'_n{n}_'
    files = [f for f in all_submissions if 'SIMPLE' in f and pattern in f and f not in priority_order]
    priority_order.extend(sorted(files))

# 6. All remaining files
remaining = [f for f in all_submissions if f not in priority_order]
priority_order.extend(sorted(remaining))

# Rename files
print("\nRenaming files:")
print("="*60)

renamed_count = 0
tracking_data = []

for i, old_name in enumerate(priority_order, 1):
    if os.path.exists(old_name):
        new_name = f"{i:03d}_{old_name}"
        try:
            os.rename(old_name, new_name)
            renamed_count += 1
            
            # Extract key info for tracking
            info = ""
            if 'SIMPLE' in old_name:
                # Extract n, d, lr, rs
                parts = old_name.split('_')
                n = next((p[1:] for p in parts if p.startswith('n')), '?')
                d = next((p[1:] for p in parts if p.startswith('d')), '?')
                lr = next((p[2:] for p in parts if p.startswith('lr')), '?')
                rs = next((p[2:] for p in parts if p.startswith('rs')), '?')
                info = f"XGB: trees={n}, depth={d}, lr={lr}, seed={rs}"
            elif 'SINGLE' in old_name:
                feature = old_name.split('SINGLE_')[1].split('_n')[0]
                info = f"Single feature: {feature}"
            elif 'DT' in old_name:
                info = "Decision Tree"
            
            tracking_data.append((i, old_name, info))
            
            if i <= 20:
                print(f"{i:03d}: {old_name}")
        except Exception as e:
            print(f"Error renaming {old_name}: {e}")

if renamed_count < len(priority_order):
    print(f"\n... showing first 20 of {renamed_count} renamed files")

# Create detailed tracking file
with open('submission_tracking.csv', 'w') as f:
    f.write("Number,Filename,Description,Kaggle_Score,Notes\n")
    for num, filename, desc in tracking_data:
        f.write(f"{num:03d},{filename},{desc},,\n")

# Create simple order file
with open('submission_order.txt', 'w') as f:
    f.write("SUBMISSION ORDER FOR KAGGLE\n")
    f.write("="*60 + "\n\n")
    f.write("Top 30 files to submit first:\n\n")
    
    for num, filename, desc in tracking_data[:30]:
        f.write(f"{num:03d} | {filename:<60} | Score: _____ |\n")
    
    f.write("\n\nKey files to watch:\n")
    f.write("- 001-010: Simplest models (1-5 trees, depth 1-2)\n")
    f.write("- 011-020: Single feature models\n") 
    f.write("- 021-030: Decision Trees\n")
    f.write("\nTarget score: 0.975708\n")

print(f"\n\nSUCCESS! Renamed {renamed_count} files")
print("\nFiles are now numbered 001_ to {:03d}_".format(renamed_count))
print("\nTracking files created:")
print("- submission_tracking.csv (detailed with descriptions)")
print("- submission_order.txt (simple list)")
print(f"\nBackup saved in: {backup_dir}/")

# Show what to submit first
print("\n" + "="*60)
print("SUBMIT THESE FIRST:")
print("="*60)
for num, filename, desc in tracking_data[:10]:
    print(f"{num:03d}: {desc}")