#!/usr/bin/env python3
"""
Quick script to check all Optuna study results and find best models.
"""

import pandas as pd
import optuna
from pathlib import Path
import hashlib
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

# Configuration
STUDIES_DIR = Path("output/optuna_studies")
CV_FOLDS = 5
TARGET_SCORE = 0.975708

# Corrected datasets
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

console = Console()

def get_study_name(model_name: str, dataset_name: str) -> str:
    """Generate study name"""
    base = f"{model_name}_{dataset_name}_{CV_FOLDS}fold"
    return hashlib.md5(base.encode()).hexdigest()[:12]

def main():
    console.print(Panel.fit("📊 Optuna Studies Results Summary", style="bold cyan"))
    
    # Collect all results
    results = []
    
    for dataset in CORRECTED_DATASETS:
        dataset_results = {}
        
        for model_type in ['xgb', 'gbm', 'cat']:
            study_name = get_study_name(model_type, dataset)
            db_path = STUDIES_DIR / f"{study_name}.db"
            
            if not db_path.exists():
                continue
                
            try:
                storage = f"sqlite:///{db_path}"
                study = optuna.load_study(study_name=study_name, storage=storage)
                
                if len(study.trials) > 0:
                    dataset_results[model_type] = {
                        'score': study.best_value,
                        'n_trials': len(study.trials),
                        'params': study.best_params
                    }
            except Exception as e:
                console.print(f"[red]Error loading {study_name}: {e}[/red]")
        
        if dataset_results:
            # Find best model for this dataset
            best_model = max(dataset_results.items(), key=lambda x: x[1]['score'])
            results.append({
                'dataset': dataset,
                'best_model': best_model[0],
                'best_score': best_model[1]['score'],
                'all_models': dataset_results
            })
    
    # Sort by best score
    results.sort(key=lambda x: x['best_score'], reverse=True)
    
    # Create summary table
    table = Table(title="Best Models by Dataset", box=box.ROUNDED)
    table.add_column("Dataset", style="cyan")
    table.add_column("Best Model", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("XGB", style="yellow")
    table.add_column("GBM", style="yellow")
    table.add_column("CAT", style="yellow")
    table.add_column("Gap to Target", style="red")
    
    for result in results:
        gap = TARGET_SCORE - result['best_score']
        gap_str = f"{gap:.6f}" if gap > 0 else f"[bold green]+{-gap:.6f}[/bold green]"
        
        xgb_score = f"{result['all_models']['xgb']['score']:.6f} ({result['all_models']['xgb']['n_trials']})" if 'xgb' in result['all_models'] else "-"
        gbm_score = f"{result['all_models']['gbm']['score']:.6f} ({result['all_models']['gbm']['n_trials']})" if 'gbm' in result['all_models'] else "-"
        cat_score = f"{result['all_models']['cat']['score']:.6f} ({result['all_models']['cat']['n_trials']})" if 'cat' in result['all_models'] else "-"
        
        table.add_row(
            result['dataset'].replace("train_corrected_", "tc"),
            result['best_model'].upper(),
            f"{result['best_score']:.6f}",
            xgb_score,
            gbm_score,
            cat_score,
            gap_str
        )
    
    console.print("\n")
    console.print(table)
    
    # Top 5 overall
    console.print("\n")
    top_table = Table(title="🏆 Top 5 Models Overall", box=box.SIMPLE)
    top_table.add_column("Rank", style="bold")
    top_table.add_column("Dataset", style="cyan")
    top_table.add_column("Model", style="magenta")
    top_table.add_column("Score", style="green")
    top_table.add_column("Gap", style="red")
    
    all_models = []
    for result in results:
        for model_type, model_data in result['all_models'].items():
            all_models.append({
                'dataset': result['dataset'],
                'model': model_type,
                'score': model_data['score'],
                'n_trials': model_data['n_trials']
            })
    
    all_models.sort(key=lambda x: x['score'], reverse=True)
    
    for i, model in enumerate(all_models[:5]):
        gap = TARGET_SCORE - model['score']
        gap_str = f"{gap:.6f}" if gap > 0 else f"[bold green]+{-gap:.6f}[/bold green]"
        
        top_table.add_row(
            str(i + 1),
            model['dataset'].replace("train_corrected_", "tc"),
            model['model'].upper(),
            f"{model['score']:.6f}",
            gap_str
        )
    
    console.print(top_table)
    
    # Summary statistics
    if results:
        best_overall = max(all_models, key=lambda x: x['score'])
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Best Model: {best_overall['model'].upper()} on {best_overall['dataset']}")
        console.print(f"  Best Score: [green]{best_overall['score']:.6f}[/green]")
        console.print(f"  Gap to Target: [{'red' if TARGET_SCORE > best_overall['score'] else 'green'}]{TARGET_SCORE - best_overall['score']:.6f}[/]")
        console.print(f"  Total Studies: {len(all_models)}")
        console.print(f"  Total Trials: {sum(m['n_trials'] for m in all_models)}")

if __name__ == "__main__":
    main()