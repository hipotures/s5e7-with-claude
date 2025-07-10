#!/usr/bin/env python3
"""
Create visualizations of systematic error discovery
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output/visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

def create_error_distribution_plot():
    """Create plot showing error distribution across datasets"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data
    total_errors = 171
    train_errors = 131
    test_errors = 40
    test_correctable = 35
    
    # Pie chart of error distribution
    sizes = [train_errors, test_correctable, test_errors - test_correctable]
    labels = ['Train Set\n(131 errors)', 'Test Set Correctable\n(35 errors)', 'Test Set NaN\n(5 errors)']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.1, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Distribution of 171 Systematic Errors', fontsize=14, fontweight='bold')
    
    # Bar chart of error types
    error_types = ['Introvert→Extrovert', 'Extrovert→Introvert', '?→Introvert', '?→Extrovert']
    counts = [79, 52, 20, 20]
    
    bars = ax2.bar(error_types, counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731'])
    ax2.set_title('Error Types in Data Generation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Type of Label Change')
    ax2.set_ylabel('Number of Errors')
    ax2.set_xticklabels(error_types, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_impact_visualization():
    """Visualize the potential impact of corrections"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Score improvement potential
    base_score = 0.975708
    per_error_impact = 0.000810
    
    # Scenario analysis
    scenarios = ['Minimum\n(0 in public)', 'Expected\n(7 in public)', 'Maximum\n(35 in public)']
    improvements = [0, 7 * per_error_impact, 35 * per_error_impact]
    final_scores = [base_score + imp for imp in improvements]
    
    # Bar chart with annotations
    bars = ax1.bar(scenarios, improvements, color=['#e74c3c', '#f39c12', '#27ae60'])
    ax1.set_title('Potential Score Improvement from 35 Corrections', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score Improvement')
    ax1.set_ylim(0, max(improvements) * 1.2)
    
    # Add value labels
    for i, (bar, imp, score) in enumerate(zip(bars, improvements, final_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0002,
                f'+{imp:.6f}\n→ {score:.6f}', ha='center', va='bottom', fontsize=10)
    
    # Timeline of discovery
    events = [
        'Initial attempts\n(Pattern 34, etc.)',
        'Flip testing\n(Found 1 error)',
        'Model disagreement\nanalysis',
        'Original data\nanalysis',
        'BREAKTHROUGH:\n171 errors found!'
    ]
    x_pos = list(range(len(events)))
    y_values = [0.2, 0.3, 0.4, 0.5, 1.0]
    
    ax2.plot(x_pos, y_values, 'o-', linewidth=3, markersize=10, color='#3498db')
    ax2.fill_between(x_pos, 0, y_values, alpha=0.3, color='#3498db')
    
    for i, (x, y, event) in enumerate(zip(x_pos, y_values, events)):
        ax2.text(x, y + 0.05, event, ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow' if i == 4 else 'white', alpha=0.8))
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Day {i+1}' for i in range(len(events))])
    ax2.set_ylabel('Progress Level')
    ax2.set_title('Discovery Timeline', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_error_examples():
    """Create visualization of specific error examples"""
    # Load mismatch data
    mismatches_df = pd.read_csv(WORKSPACE_DIR / "scripts/output/generation_mismatches.csv")
    
    # Select interesting test examples
    test_errors = mismatches_df[mismatches_df['syn_source'] == 'test'].head(10)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create table-like visualization
    cell_text = []
    for _, row in test_errors.iterrows():
        orig_label = row['orig_personality'] if pd.notna(row['orig_personality']) else '?'
        syn_label = row['syn_personality'] if pd.notna(row['syn_personality']) else 'NaN'
        cell_text.append([
            f"{row['orig_idx']}", 
            orig_label,
            f"{row['syn_id']}",
            syn_label,
            '❌ Error' if orig_label != syn_label else '✓ OK'
        ])
    
    table = ax.table(cellText=cell_text,
                    colLabels=['Original Index', 'Original Label', 'Synthetic ID', 'Synthetic Label', 'Status'],
                    cellLoc='center',
                    loc='center',
                    colColours=['#f0f0f0']*5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color cells based on error type
    for i in range(len(cell_text)):
        if 'Error' in cell_text[i][4]:
            for j in range(5):
                table[(i+1, j)].set_facecolor('#ffcccc')
    
    ax.axis('off')
    ax.set_title('Examples of Systematic Errors in Test Set', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'error_examples.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plot():
    """Compare our approach vs top competitors"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data
    approaches = ['Random Flips\n(Early attempts)', 'Pattern Search\n(ID ending 34)', 
                  'Model Disagreement\n(CatBoost vs XGB)', 'Systematic Analysis\n(Our breakthrough)']
    errors_found = [0, 0, 0, 35]
    effectiveness = [0, 0, 10, 100]  # Percentage effectiveness
    
    x = np.arange(len(approaches))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, errors_found, width, label='Errors Found', color='#e74c3c')
    bars2 = ax.bar(x + width/2, effectiveness, width, label='Effectiveness %', color='#27ae60')
    
    # Customize
    ax.set_xlabel('Approach', fontsize=12)
    ax.set_ylabel('Count / Percentage', fontsize=12)
    ax.set_title('Comparison of Different Approaches', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    # Add annotation
    ax.annotate('BREAKTHROUGH!', xy=(3, 35), xytext=(2.5, 50),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=14, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'approach_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_leaderboard_projection():
    """Show projected leaderboard position"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Current and projected positions
    positions = list(range(1, 11)) + [400]
    scores = [0.977327, 0.976518, 0.976518, 0.975708, 0.975708, 
              0.975708, 0.975708, 0.975708, 0.975708, 0.975708, 0.975708]
    labels = ['Top 1', 'Top 2', 'Top 3'] + [f'Pos {i}' for i in range(4, 11)] + ['Us (Pos 400)']
    
    # Our projected score
    our_current = 0.975708
    our_projected_min = 0.975708  # If no errors in public
    our_projected_max = 0.975708 + 0.005670  # If 7 errors in public
    
    # Create horizontal bar chart
    y_pos = np.arange(len(positions))
    bars = ax.barh(y_pos, scores, color=['gold', 'silver', '#cd7f32'] + ['#95a5a6']*7 + ['#e74c3c'])
    
    # Add our projection
    ax.barh(len(positions), our_projected_max - our_current, left=our_current, 
            color='#27ae60', alpha=0.7, label='Our Potential')
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Score')
    ax.set_title('Current Leaderboard & Our Projection', fontsize=16, fontweight='bold')
    ax.set_xlim(0.974, 0.978)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.00005, bar.get_y() + bar.get_height()/2,
                f'{score:.6f}', va='center')
    
    # Add projection label
    ax.text(our_projected_max + 0.00005, len(positions), 
            f'{our_projected_max:.6f}', va='center', color='green', fontweight='bold')
    
    # Add arrow showing movement
    ax.annotate('', xy=(our_projected_max, len(positions)), 
                xytext=(our_current, len(positions)),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    
    ax.text(0.9745, len(positions) + 0.5, 
            'Projected movement\nafter corrections', 
            fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'leaderboard_projection.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_infographic():
    """Create a summary infographic"""
    fig = plt.figure(figsize=(14, 10))
    
    # Title
    fig.suptitle('🎯 SYSTEMATIC ERROR DISCOVERY - SUMMARY', fontsize=20, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Key numbers
    ax1 = fig.add_subplot(gs[0, :])
    ax1.text(0.5, 0.5, '171 TOTAL ERRORS FOUND\n35 IN TEST SET', 
             ha='center', va='center', fontsize=24, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
    ax1.axis('off')
    
    # Process
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.text(0.5, 0.7, 'ORIGINAL DATA', ha='center', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.5, '2,900 records', ha='center', fontsize=12)
    ax2.text(0.5, 0.3, '↓ 8.5x', ha='center', fontsize=16)
    ax2.text(0.5, 0.1, 'SYNTHETIC DATA\n24,699 records', ha='center', fontsize=12)
    ax2.axis('off')
    ax2.set_title('Data Generation', fontsize=16)
    
    # Error breakdown
    ax3 = fig.add_subplot(gs[1, 1])
    labels = ['Train', 'Test']
    sizes = [131, 40]
    colors = ['#ff9999', '#66b3ff']
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax3.set_title('Error Distribution', fontsize=16)
    
    # Impact
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.text(0.5, 0.7, 'CURRENT SCORE', ha='center', fontsize=14, fontweight='bold')
    ax4.text(0.5, 0.5, '0.975708', ha='center', fontsize=18, color='red')
    ax4.text(0.5, 0.3, '↓', ha='center', fontsize=16)
    ax4.text(0.5, 0.1, 'PROJECTED\n~0.981378', ha='center', fontsize=16, color='green', fontweight='bold')
    ax4.axis('off')
    ax4.set_title('Score Impact', fontsize=16)
    
    # Timeline
    ax5 = fig.add_subplot(gs[2, :])
    timeline_events = ['Day 1-2: Random attempts', 'Day 3: Pattern search', 
                       'Day 4: Model analysis', 'Day 5: BREAKTHROUGH!']
    x = [1, 2, 3, 4]
    y = [1, 1, 1, 2]
    ax5.plot(x, y, 'o-', markersize=15, linewidth=3, color='#3498db')
    for i, (xi, yi, event) in enumerate(zip(x, y, timeline_events)):
        color = 'yellow' if i == 3 else 'lightblue'
        ax5.text(xi, yi + 0.1, event, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
    ax5.set_xlim(0.5, 4.5)
    ax5.set_ylim(0.5, 2.5)
    ax5.axis('off')
    ax5.set_title('Discovery Timeline', fontsize=16, y=-0.1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_infographic.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Creating visualizations...")
    
    # Create all visualizations
    print("1. Creating error distribution plot...")
    create_error_distribution_plot()
    
    print("2. Creating impact visualization...")
    create_impact_visualization()
    
    print("3. Creating error examples...")
    create_error_examples()
    
    print("4. Creating approach comparison...")
    create_comparison_plot()
    
    print("5. Creating leaderboard projection...")
    create_leaderboard_projection()
    
    print("6. Creating summary infographic...")
    create_summary_infographic()
    
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()