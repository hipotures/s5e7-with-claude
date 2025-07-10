#!/usr/bin/env python3
"""
Sliding window analysis - analyze local E/I distribution patterns
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
ORIGINAL_SUBMISSION = WORKSPACE_DIR / "subm/20250705/subm-0.96950-20250704_121621-xgb-381482788433-148.csv"

def sliding_window_analysis(window_size=10):
    """Analyze E/I ratio in sliding windows"""
    print(f"\n{'='*60}")
    print(f"ANALIZA OKNA PRZESUWNEGO (rozmiar: {window_size})")
    print(f"{'='*60}")
    
    # Load data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    original_df = pd.read_csv(ORIGINAL_SUBMISSION)
    
    # Sort by ID to ensure sequence
    test_df = test_df.sort_values('id').reset_index(drop=True)
    
    # Get personality for each test record
    personalities = []
    for _, row in test_df.iterrows():
        idx = original_df[original_df['id'] == row['id']].index
        if len(idx) > 0:
            personalities.append(original_df.loc[idx[0], 'Personality'])
        else:
            personalities.append(None)
    
    test_df['personality'] = personalities
    
    # Calculate sliding window statistics
    results = []
    
    for i in range(window_size, len(test_df) - window_size):
        # Get left and right windows
        left_window = test_df.iloc[i-window_size:i]
        right_window = test_df.iloc[i+1:i+window_size+1]
        current = test_df.iloc[i]
        
        # Count E/I in each window
        left_e = (left_window['personality'] == 'Extrovert').sum()
        left_i = (left_window['personality'] == 'Introvert').sum()
        right_e = (right_window['personality'] == 'Extrovert').sum()
        right_i = (right_window['personality'] == 'Introvert').sum()
        
        # Calculate ratios
        left_e_ratio = left_e / window_size if window_size > 0 else 0
        right_e_ratio = right_e / window_size if window_size > 0 else 0
        
        # Calculate asymmetry
        asymmetry = abs(left_e_ratio - right_e_ratio)
        
        results.append({
            'position': i,
            'id': current['id'],
            'personality': current['personality'],
            'left_e': left_e,
            'left_i': left_i,
            'right_e': right_e,
            'right_i': right_i,
            'left_e_ratio': left_e_ratio,
            'right_e_ratio': right_e_ratio,
            'asymmetry': asymmetry,
            'total_e_ratio': (left_e + right_e) / (2 * window_size)
        })
    
    results_df = pd.DataFrame(results)
    
    # Find interesting patterns
    print(f"\nPrzetworzono {len(results_df)} pozycji")
    print(f"Średnia E_ratio (lewo): {results_df['left_e_ratio'].mean():.3f}")
    print(f"Średnia E_ratio (prawo): {results_df['right_e_ratio'].mean():.3f}")
    print(f"Średnia asymetria: {results_df['asymmetry'].mean():.3f}")
    
    # Find extreme asymmetries
    high_asymmetry = results_df.nlargest(10, 'asymmetry')
    print(f"\n10 NAJWIĘKSZYCH ASYMETRII:")
    print("-"*40)
    for _, row in high_asymmetry.iterrows():
        print(f"ID {row['id']} ({row['personality']}): "
              f"L={row['left_e_ratio']:.2f} R={row['right_e_ratio']:.2f} "
              f"Asymmetry={row['asymmetry']:.2f}")
    
    return results_df

def create_visualization(all_results):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Define colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    
    for idx, (window_size, results_df) in enumerate(all_results):
        ax = axes[idx]
        
        # Plot E_ratio for left and right windows
        positions = results_df['position'].values
        
        # Smooth the data for better visualization
        from scipy.ndimage import gaussian_filter1d
        left_smooth = gaussian_filter1d(results_df['left_e_ratio'].values, sigma=5)
        right_smooth = gaussian_filter1d(results_df['right_e_ratio'].values, sigma=5)
        asymmetry_smooth = gaussian_filter1d(results_df['asymmetry'].values, sigma=5)
        
        # Main plots
        ax.plot(positions, left_smooth, label=f'Left window E-ratio', 
                color=colors[0], linewidth=2, alpha=0.8)
        ax.plot(positions, right_smooth, label=f'Right window E-ratio', 
                color=colors[1], linewidth=2, alpha=0.8)
        
        # Asymmetry on secondary axis
        ax2 = ax.twinx()
        ax2.plot(positions, asymmetry_smooth, label='Asymmetry', 
                 color=colors[2], linewidth=1.5, linestyle='--', alpha=0.6)
        
        # Add average line
        avg_e_ratio = results_df['total_e_ratio'].mean()
        ax.axhline(y=avg_e_ratio, color='gray', linestyle=':', alpha=0.5, 
                   label=f'Average E-ratio ({avg_e_ratio:.3f})')
        
        # Highlight extreme asymmetries
        extreme_mask = results_df['asymmetry'] > results_df['asymmetry'].quantile(0.99)
        extreme_points = results_df[extreme_mask]
        
        for _, point in extreme_points.iterrows():
            ax.axvline(x=point['position'], color='red', alpha=0.2, linewidth=1)
            # Add ID annotation for most extreme
            if point['asymmetry'] > results_df['asymmetry'].quantile(0.995):
                ax.annotate(f"ID {point['id']}", 
                           xy=(point['position'], point['left_e_ratio']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # Formatting
        ax.set_title(f'Window Size = {window_size} records', fontsize=14, fontweight='bold')
        ax.set_xlabel('Position in dataset' if idx == 2 else '')
        ax.set_ylabel('E-ratio', color=colors[0])
        ax2.set_ylabel('Asymmetry', color=colors[2])
        
        ax.set_ylim(0.6, 0.9)
        ax2.set_ylim(0, 0.4)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Color the y-axis labels
        ax.tick_params(axis='y', labelcolor=colors[0])
        ax2.tick_params(axis='y', labelcolor=colors[2])
    
    plt.suptitle('Sliding Window Analysis: Local E/I Distribution Patterns', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = WORKSPACE_DIR / "scripts/output/sliding_window_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nWykres zapisany: {output_path}")
    plt.close()

def find_flip_candidates(all_results):
    """Find best flip candidates based on sliding window analysis"""
    print("\n" + "="*60)
    print("KANDYDACI NA PODSTAWIE ANALIZY OKIEN")
    print("="*60)
    
    candidates = []
    
    for window_size, results_df in all_results:
        # 1. Records with highest asymmetry
        high_asymmetry = results_df.nlargest(5, 'asymmetry')
        
        for _, row in high_asymmetry.iterrows():
            # Prefer cases where current label doesn't match the dominant side
            if row['personality'] == 'Extrovert' and row['left_e_ratio'] < 0.5 and row['right_e_ratio'] < 0.5:
                candidates.append({
                    'id': row['id'],
                    'current': row['personality'],
                    'window_size': window_size,
                    'reason': f'E in I-dominated region (L={row["left_e_ratio"]:.2f}, R={row["right_e_ratio"]:.2f})',
                    'score': row['asymmetry']
                })
            elif row['personality'] == 'Introvert' and row['left_e_ratio'] > 0.8 and row['right_e_ratio'] > 0.8:
                candidates.append({
                    'id': row['id'],
                    'current': row['personality'],
                    'window_size': window_size,
                    'reason': f'I in E-dominated region (L={row["left_e_ratio"]:.2f}, R={row["right_e_ratio"]:.2f})',
                    'score': row['asymmetry']
                })
        
        # 2. Records at boundaries (sudden changes)
        ratio_diff = results_df['left_e_ratio'] - results_df['right_e_ratio']
        boundaries = results_df[abs(ratio_diff) > 0.3]
        
        for _, row in boundaries.head(3).iterrows():
            candidates.append({
                'id': row['id'],
                'current': row['personality'],
                'window_size': window_size,
                'reason': f'Boundary point (L-R diff = {abs(row["left_e_ratio"] - row["right_e_ratio"]):.2f})',
                'score': abs(row['left_e_ratio'] - row['right_e_ratio'])
            })
    
    # Deduplicate and sort
    seen_ids = set()
    unique_candidates = []
    for cand in sorted(candidates, key=lambda x: x['score'], reverse=True):
        if cand['id'] not in seen_ids:
            seen_ids.add(cand['id'])
            unique_candidates.append(cand)
    
    print("\nTOP 10 KANDYDATÓW:")
    for i, cand in enumerate(unique_candidates[:10]):
        print(f"{i+1}. ID {cand['id']} ({cand['current']})")
        print(f"   Window: {cand['window_size']}, Score: {cand['score']:.3f}")
        print(f"   Reason: {cand['reason']}")
    
    return unique_candidates

def main():
    # Analyze different window sizes
    window_sizes = [10, 50, 100]
    all_results = []
    
    for window_size in window_sizes:
        results_df = sliding_window_analysis(window_size)
        all_results.append((window_size, results_df))
    
    # Create visualization
    create_visualization(all_results)
    
    # Find candidates
    candidates = find_flip_candidates(all_results)
    
    # Save candidates
    candidates_df = pd.DataFrame(candidates[:20])
    output_path = WORKSPACE_DIR / "scripts/output/sliding_window_candidates.csv"
    candidates_df.to_csv(output_path, index=False)
    print(f"\nKandydaci zapisani: {output_path}")

if __name__ == "__main__":
    main()