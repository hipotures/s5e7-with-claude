#!/usr/bin/env python3
"""Recreate results table from saved files"""

import os
import re
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# Find all submission files
scores_dir = "../scores" if os.path.exists("../scores") else "scores"
submissions = []

if os.path.exists(scores_dir):
    for f in os.listdir(scores_dir):
        if f.startswith("subm-") and f.endswith(".csv"):
            # Parse filename: subm-SCORE-TIMESTAMP-MODEL-DATASET.csv
            match = re.match(r"subm-(\d\.\d+)-(\d+_\d+)-(\w+)-(.+)\.csv", f)
            if match:
                score, timestamp, model, dataset = match.groups()
                submissions.append({
                    'score': float(score),
                    'timestamp': timestamp,
                    'model': model,
                    'dataset': dataset,
                    'filename': f
                })

# Group by dataset
dataset_results = {}
for sub in submissions:
    ds = sub['dataset']
    if ds not in dataset_results or sub['score'] > dataset_results[ds]['score']:
        dataset_results[ds] = sub

# Map short names back to full names
dataset_map = {
    'train_co': 'train_corrected_??',  # Can't determine which one
}

# Create table
table = Table(title="Optimization Results Summary", box=box.ROUNDED)
table.add_column("Dataset", style="cyan", min_width=25)
table.add_column("Best Score", style="green")
table.add_column("Best Model", style="yellow")
table.add_column("Timestamp", style="blue")

# Sort by score
sorted_results = sorted(dataset_results.items(), key=lambda x: x[1]['score'], reverse=True)

console.print("\n")
for dataset, result in sorted_results[:10]:
    table.add_row(
        dataset,
        f"{result['score']:.6f}",
        result['model'].upper(),
        result['timestamp']
    )

console.print(table)

# Show top scores
console.print("\n[bold]Top 5 Scores:[/bold]")
top_scores = sorted(submissions, key=lambda x: x['score'], reverse=True)[:5]
for i, sub in enumerate(top_scores, 1):
    console.print(f"{i}. {sub['score']:.6f} - {sub['model'].upper()} - {sub['dataset']} ({sub['timestamp']})")

# Calculate distance to target
target = 0.975708
if top_scores:
    best = top_scores[0]['score']
    gap = target - best
    console.print(f"\n[bold]Distance to target {target}:[/bold]")
    console.print(f"Best score: {best:.6f}")
    console.print(f"Gap: {gap:.6f} ({gap*100:.4f}%)")