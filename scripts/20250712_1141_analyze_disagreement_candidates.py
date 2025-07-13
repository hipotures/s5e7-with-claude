#!/usr/bin/env python3
"""
Analyze disagreement analysis files to identify promising flip candidates.
Created: 2025-01-12 11:41
"""

import pandas as pd
import ast
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# Load both disagreement analysis files
df_orig = pd.read_csv('../scores/disagreement_analysis.csv')
df_corrected = pd.read_csv('../scores/disagreement_analysis_corrected.csv')

# Parse challenger predictions
df_orig['challenger_preds_list'] = df_orig['challenger_preds'].apply(ast.literal_eval)
df_corrected['challenger_preds_list'] = df_corrected['challenger_preds'].apply(ast.literal_eval)

# Add unanimous disagreement flag (all 6 challengers agree but differ from baseline)
df_orig['unanimous_disagreement'] = df_orig.apply(
    lambda row: len(set(row['challenger_preds_list'])) == 1 and 
                row['challenger_preds_list'][0] != row['baseline_prediction'], 
    axis=1
)
df_corrected['unanimous_disagreement'] = df_corrected.apply(
    lambda row: len(set(row['challenger_preds_list'])) == 1 and 
                row['challenger_preds_list'][0] != row['baseline_prediction'], 
    axis=1
)

# Known IDs from previous analysis
known_ids = {
    20934: "Confirmed error (E→I)",
    19482: "High disagreement candidate",
    18634: "False positive (boundary)",
    20932: "False positive (boundary)",
    19098: "Pattern 34 candidate",
    21404: "High disagreement candidate"
}

# Analysis 1: Top unanimous disagreements in original dataset
console.print(Panel.fit("Analysis 1: Unanimous Disagreements (Original Dataset)", 
                       style="bold cyan"))
unanimous_orig = df_orig[df_orig['unanimous_disagreement']].copy()
console.print(f"Total IDs with unanimous disagreement: {len(unanimous_orig)}")

# Show top 20
table = Table(title="Top 20 Unanimous Disagreements (Original)", box=box.ROUNDED)
table.add_column("ID", style="cyan")
table.add_column("Baseline", style="yellow")
table.add_column("All Challengers", style="green")
table.add_column("Notes", style="magenta")

for _, row in unanimous_orig.head(20).iterrows():
    notes = known_ids.get(row['id'], "")
    table.add_row(
        str(row['id']),
        row['baseline_prediction'],
        row['challenger_preds_list'][0],
        notes
    )
console.print(table)

# Analysis 2: Top disagreements in corrected dataset
console.print("\n")
console.print(Panel.fit("Analysis 2: High Disagreements (Corrected Dataset)", 
                       style="bold green"))
high_disagreement_corrected = df_corrected[df_corrected['mismatches'] >= 4].copy()
console.print(f"Total IDs with mismatches >= 4: {len(high_disagreement_corrected)}")

table = Table(title="High Disagreements in Corrected Dataset", box=box.ROUNDED)
table.add_column("ID", style="cyan")
table.add_column("Baseline", style="yellow")
table.add_column("Mismatches", style="red")
table.add_column("Challenger Predictions", style="green")
table.add_column("Notes", style="magenta")

for _, row in high_disagreement_corrected.iterrows():
    notes = known_ids.get(row['id'], "")
    pred_counts = Counter(row['challenger_preds_list'])
    pred_summary = ", ".join([f"{p}:{c}" for p, c in pred_counts.items()])
    table.add_row(
        str(row['id']),
        row['baseline_prediction'],
        str(row['mismatches']),
        pred_summary,
        notes
    )
console.print(table)

# Analysis 3: IDs appearing in both analyses with high disagreement
console.print("\n")
console.print(Panel.fit("Analysis 3: Common High-Disagreement IDs", 
                       style="bold red"))

# Merge datasets
merged = pd.merge(
    df_orig[df_orig['mismatches'] >= 5][['id', 'baseline_prediction', 'mismatches', 'challenger_preds_list']],
    df_corrected[df_corrected['mismatches'] >= 4][['id', 'mismatches', 'challenger_preds_list']],
    on='id',
    suffixes=('_orig', '_corrected')
)

table = Table(title="IDs with High Disagreement in Both Datasets", box=box.ROUNDED)
table.add_column("ID", style="cyan")
table.add_column("Baseline", style="yellow")
table.add_column("Orig Mismatches", style="red")
table.add_column("Corrected Mismatches", style="red")
table.add_column("Notes", style="magenta")

for _, row in merged.iterrows():
    notes = known_ids.get(row['id'], "")
    table.add_row(
        str(row['id']),
        row['baseline_prediction'],
        str(row['mismatches_orig']),
        str(row['mismatches_corrected']),
        notes
    )
console.print(table)

# Analysis 4: Statistical summary
console.print("\n")
console.print(Panel.fit("Analysis 4: Statistical Summary", style="bold blue"))

stats_table = Table(title="Disagreement Statistics", box=box.ROUNDED)
stats_table.add_column("Metric", style="cyan")
stats_table.add_column("Original", style="yellow")
stats_table.add_column("Corrected", style="green")

stats_table.add_row("Total samples", str(len(df_orig)), str(len(df_corrected)))
stats_table.add_row("Unanimous disagreements", str(len(unanimous_orig)), 
                   str(len(df_corrected[df_corrected['unanimous_disagreement']])))
stats_table.add_row("Mismatches >= 5", 
                   str(len(df_orig[df_orig['mismatches'] >= 5])),
                   str(len(df_corrected[df_corrected['mismatches'] >= 5])))
stats_table.add_row("Mismatches = 6", 
                   str(len(df_orig[df_orig['mismatches'] == 6])),
                   str(len(df_corrected[df_corrected['mismatches'] == 6])))

console.print(stats_table)

# Analysis 5: Most promising candidates
console.print("\n")
console.print(Panel.fit("Analysis 5: Most Promising Flip Candidates", 
                       style="bold white on red"))

# Candidates: unanimous disagreement in original AND high disagreement in corrected
promising = []

for _, row in unanimous_orig.iterrows():
    id_val = row['id']
    if id_val in df_corrected['id'].values:
        corrected_row = df_corrected[df_corrected['id'] == id_val].iloc[0]
        if corrected_row['mismatches'] >= 4:
            promising.append({
                'id': id_val,
                'baseline': row['baseline_prediction'],
                'flip_to': row['challenger_preds_list'][0],
                'orig_mismatches': row['mismatches'],
                'corrected_mismatches': corrected_row['mismatches'],
                'notes': known_ids.get(id_val, "")
            })

# Sort by corrected mismatches
promising = sorted(promising, key=lambda x: x['corrected_mismatches'], reverse=True)

table = Table(title="Top Flip Candidates (E→I)", box=box.ROUNDED)
table.add_column("Rank", style="white")
table.add_column("ID", style="cyan")
table.add_column("Flip", style="yellow")
table.add_column("Orig", style="red")
table.add_column("Corrected", style="red")
table.add_column("Notes", style="magenta")

for i, candidate in enumerate(promising[:15], 1):
    table.add_row(
        str(i),
        str(candidate['id']),
        f"{candidate['baseline']}→{candidate['flip_to']}",
        str(candidate['orig_mismatches']),
        str(candidate['corrected_mismatches']),
        candidate['notes']
    )

console.print(table)

# Save top candidates
top_candidates = [c['id'] for c in promising[:10]]
console.print(f"\n[bold green]Top 10 candidate IDs saved:[/bold green] {top_candidates}")

# Export for further analysis
pd.DataFrame(promising).to_csv('output/disagreement_flip_candidates.csv', index=False)
console.print("\n[bold cyan]Full candidate list exported to: output/disagreement_flip_candidates.csv[/bold cyan]")