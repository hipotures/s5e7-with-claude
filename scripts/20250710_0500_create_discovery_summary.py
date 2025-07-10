#!/usr/bin/env python3
"""
Create comprehensive summary graphic of the systematic error discovery
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from matplotlib.patheffects import withStroke
import numpy as np
from pathlib import Path

# Output path
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output/visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

def create_discovery_summary():
    """Create a comprehensive infographic summarizing the discovery"""
    
    # Create figure with custom background
    fig = plt.figure(figsize=(16, 12), facecolor='#f8f9fa')
    
    # Main title with emphasis
    title_text = fig.text(0.5, 0.95, '🎯 SYSTEMATYCZNY BŁĄD W GENEROWANIU DANYCH', 
                         fontsize=28, weight='bold', ha='center', va='top',
                         bbox=dict(boxstyle="round,pad=0.8", facecolor='#e74c3c', edgecolor='none', alpha=0.9),
                         color='white')
    
    # Create main sections
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Section 1: Main Discovery (top center)
    discovery_box = FancyBboxPatch((10, 70), 80, 15,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#3498db',
                                  edgecolor='#2c3e50',
                                  linewidth=3,
                                  alpha=0.9)
    ax.add_patch(discovery_box)
    
    ax.text(50, 77.5, '171 rekordów ma zmienione etykiety\nwzględem oryginalnych danych',
            fontsize=20, ha='center', va='center', color='white', weight='bold')
    
    # Section 2: Key Numbers (left side)
    numbers_box = FancyBboxPatch((5, 35), 40, 30,
                                boxstyle="round,pad=0.1",
                                facecolor='#f39c12',
                                edgecolor='#d68910',
                                linewidth=2)
    ax.add_patch(numbers_box)
    
    ax.text(25, 60, '📊 KLUCZOWE LICZBY', fontsize=16, ha='center', weight='bold')
    
    # Draw flow diagram for data augmentation
    ax.text(15, 52, '2,900', fontsize=18, ha='center', weight='bold')
    ax.text(15, 49, 'oryginalne', fontsize=10, ha='center')
    
    arrow1 = FancyArrowPatch((20, 50), (30, 50), 
                            connectionstyle="arc3,rad=0", 
                            arrowstyle='->', 
                            mutation_scale=30,
                            color='#2c3e50',
                            linewidth=3)
    ax.add_patch(arrow1)
    
    ax.text(35, 52, '24,699', fontsize=18, ha='center', weight='bold')
    ax.text(35, 49, 'syntetyczne', fontsize=10, ha='center')
    ax.text(25, 46, '8.5x', fontsize=14, ha='center', style='italic', color='#e74c3c')
    
    # Error statistics
    ax.text(25, 42, '171 niezgodności', fontsize=14, ha='center', weight='bold')
    ax.text(25, 39, '35 w test set', fontsize=14, ha='center', color='#e74c3c', weight='bold')
    ax.text(25, 36, '5.9% błędnych kopii', fontsize=12, ha='center')
    
    # Section 3: Process (right side)
    process_box = FancyBboxPatch((55, 35), 40, 30,
                                boxstyle="round,pad=0.1",
                                facecolor='#27ae60',
                                edgecolor='#229954',
                                linewidth=2)
    ax.add_patch(process_box)
    
    ax.text(75, 60, '🔧 PROCES ODKRYCIA', fontsize=16, ha='center', weight='bold')
    
    # Process steps
    steps = [
        '1. Porównanie danych',
        '2. Dopasowanie cech',
        '3. Wykrycie niezgodności',
        '4. Plik korekcyjny'
    ]
    
    for i, step in enumerate(steps):
        y_pos = 54 - i*4
        circle = Circle((60, y_pos), 1.5, facecolor='white', edgecolor='#27ae60', linewidth=2)
        ax.add_patch(circle)
        ax.text(60, y_pos, str(i+1), fontsize=12, ha='center', va='center', weight='bold')
        ax.text(75, y_pos, step, fontsize=12, ha='center', va='center')
        
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((60, y_pos-1.5), (60, y_pos-2.5),
                                   arrowstyle='->', mutation_scale=20,
                                   color='white', linewidth=2)
            ax.add_patch(arrow)
    
    # Section 4: Impact (bottom)
    impact_box = FancyBboxPatch((10, 10), 80, 20,
                               boxstyle="round,pad=0.1",
                               facecolor='#9b59b6',
                               edgecolor='#8e44ad',
                               linewidth=3)
    ax.add_patch(impact_box)
    
    ax.text(50, 25, '🎯 WPŁYW NA WYNIKI', fontsize=18, ha='center', weight='bold', color='white')
    
    # Score calculation
    ax.text(30, 18, 'Obecny wynik:', fontsize=14, ha='center', color='white')
    ax.text(30, 15, '0.975708', fontsize=16, ha='center', weight='bold', color='white')
    
    ax.text(50, 18, '+', fontsize=20, ha='center', color='white')
    
    ax.text(70, 18, 'Poprawa (35 błędów):', fontsize=14, ha='center', color='white')
    ax.text(70, 15, '≈ 0.028350', fontsize=16, ha='center', weight='bold', color='#f1c40f')
    
    ax.text(50, 12, '= MOŻLIWY TOP 1', fontsize=18, ha='center', weight='bold', color='#f1c40f',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add visual elements for test set breakdown
    test_breakdown = patches.Rectangle((25, 3), 50, 5, 
                                     facecolor='#ecf0f1', 
                                     edgecolor='#34495e',
                                     linewidth=2)
    ax.add_patch(test_breakdown)
    
    # Breakdown bars
    ax.text(50, 5.5, '35 błędów w test set', fontsize=10, ha='center', weight='bold')
    
    # I->E errors (16)
    i_to_e_width = 16/35 * 48
    i_to_e = patches.Rectangle((26, 3.5), i_to_e_width, 1, facecolor='#e74c3c')
    ax.add_patch(i_to_e)
    
    # E->I errors (19)
    e_to_i_width = 19/35 * 48
    e_to_i = patches.Rectangle((26 + i_to_e_width, 3.5), e_to_i_width, 1, facecolor='#3498db')
    ax.add_patch(e_to_i)
    
    ax.text(26 + i_to_e_width/2, 2.5, '16 I→E', fontsize=8, ha='center')
    ax.text(26 + i_to_e_width + e_to_i_width/2, 2.5, '19 E→I', fontsize=8, ha='center')
    
    # Save
    plt.savefig(OUTPUT_DIR / 'discovery_summary_polish.png', dpi=300, bbox_inches='tight', 
                facecolor='#f8f9fa', edgecolor='none')
    plt.close()

def create_english_summary():
    """Create English version of the summary"""
    
    # Create figure with custom background
    fig = plt.figure(figsize=(16, 12), facecolor='#f8f9fa')
    
    # Main title with emphasis
    title_text = fig.text(0.5, 0.95, '🎯 SYSTEMATIC ERROR IN DATA GENERATION', 
                         fontsize=28, weight='bold', ha='center', va='top',
                         bbox=dict(boxstyle="round,pad=0.8", facecolor='#e74c3c', edgecolor='none', alpha=0.9),
                         color='white')
    
    # Create main sections
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Section 1: Main Discovery (top center)
    discovery_box = FancyBboxPatch((10, 70), 80, 15,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#3498db',
                                  edgecolor='#2c3e50',
                                  linewidth=3,
                                  alpha=0.9)
    ax.add_patch(discovery_box)
    
    ax.text(50, 77.5, '171 records have changed labels\ncompared to original data',
            fontsize=20, ha='center', va='center', color='white', weight='bold')
    
    # Section 2: Key Numbers (left side)
    numbers_box = FancyBboxPatch((5, 35), 40, 30,
                                boxstyle="round,pad=0.1",
                                facecolor='#f39c12',
                                edgecolor='#d68910',
                                linewidth=2)
    ax.add_patch(numbers_box)
    
    ax.text(25, 60, '📊 KEY NUMBERS', fontsize=16, ha='center', weight='bold')
    
    # Draw flow diagram for data augmentation
    ax.text(15, 52, '2,900', fontsize=18, ha='center', weight='bold')
    ax.text(15, 49, 'original', fontsize=10, ha='center')
    
    arrow1 = FancyArrowPatch((20, 50), (30, 50), 
                            connectionstyle="arc3,rad=0", 
                            arrowstyle='->', 
                            mutation_scale=30,
                            color='#2c3e50',
                            linewidth=3)
    ax.add_patch(arrow1)
    
    ax.text(35, 52, '24,699', fontsize=18, ha='center', weight='bold')
    ax.text(35, 49, 'synthetic', fontsize=10, ha='center')
    ax.text(25, 46, '8.5x', fontsize=14, ha='center', style='italic', color='#e74c3c')
    
    # Error statistics
    ax.text(25, 42, '171 mismatches', fontsize=14, ha='center', weight='bold')
    ax.text(25, 39, '35 in test set', fontsize=14, ha='center', color='#e74c3c', weight='bold')
    ax.text(25, 36, '5.9% error rate', fontsize=12, ha='center')
    
    # Section 3: Process (right side)
    process_box = FancyBboxPatch((55, 35), 40, 30,
                                boxstyle="round,pad=0.1",
                                facecolor='#27ae60',
                                edgecolor='#229954',
                                linewidth=2)
    ax.add_patch(process_box)
    
    ax.text(75, 60, '🔧 DISCOVERY PROCESS', fontsize=16, ha='center', weight='bold')
    
    # Process steps
    steps = [
        '1. Compare datasets',
        '2. Match features',
        '3. Find mismatches',
        '4. Create corrections'
    ]
    
    for i, step in enumerate(steps):
        y_pos = 54 - i*4
        circle = Circle((60, y_pos), 1.5, facecolor='white', edgecolor='#27ae60', linewidth=2)
        ax.add_patch(circle)
        ax.text(60, y_pos, str(i+1), fontsize=12, ha='center', va='center', weight='bold')
        ax.text(75, y_pos, step, fontsize=12, ha='center', va='center')
        
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((60, y_pos-1.5), (60, y_pos-2.5),
                                   arrowstyle='->', mutation_scale=20,
                                   color='white', linewidth=2)
            ax.add_patch(arrow)
    
    # Section 4: Impact (bottom)
    impact_box = FancyBboxPatch((10, 10), 80, 20,
                               boxstyle="round,pad=0.1",
                               facecolor='#9b59b6',
                               edgecolor='#8e44ad',
                               linewidth=3)
    ax.add_patch(impact_box)
    
    ax.text(50, 25, '🎯 IMPACT ON RESULTS', fontsize=18, ha='center', weight='bold', color='white')
    
    # Score calculation
    ax.text(30, 18, 'Current score:', fontsize=14, ha='center', color='white')
    ax.text(30, 15, '0.975708', fontsize=16, ha='center', weight='bold', color='white')
    
    ax.text(50, 18, '+', fontsize=20, ha='center', color='white')
    
    ax.text(70, 18, 'Improvement (35 errors):', fontsize=14, ha='center', color='white')
    ax.text(70, 15, '≈ 0.028350', fontsize=16, ha='center', weight='bold', color='#f1c40f')
    
    ax.text(50, 12, '= POTENTIAL TOP 1', fontsize=18, ha='center', weight='bold', color='#f1c40f',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add visual elements for test set breakdown
    test_breakdown = patches.Rectangle((25, 3), 50, 5, 
                                     facecolor='#ecf0f1', 
                                     edgecolor='#34495e',
                                     linewidth=2)
    ax.add_patch(test_breakdown)
    
    # Breakdown bars
    ax.text(50, 5.5, '35 errors in test set', fontsize=10, ha='center', weight='bold')
    
    # I->E errors (16)
    i_to_e_width = 16/35 * 48
    i_to_e = patches.Rectangle((26, 3.5), i_to_e_width, 1, facecolor='#e74c3c')
    ax.add_patch(i_to_e)
    
    # E->I errors (19)
    e_to_i_width = 19/35 * 48
    e_to_i = patches.Rectangle((26 + i_to_e_width, 3.5), e_to_i_width, 1, facecolor='#3498db')
    ax.add_patch(e_to_i)
    
    ax.text(26 + i_to_e_width/2, 2.5, '16 I→E', fontsize=8, ha='center')
    ax.text(26 + i_to_e_width + e_to_i_width/2, 2.5, '19 E→I', fontsize=8, ha='center')
    
    # Save
    plt.savefig(OUTPUT_DIR / 'discovery_summary_english.png', dpi=300, bbox_inches='tight', 
                facecolor='#f8f9fa', edgecolor='none')
    plt.close()

def main():
    print("Creating discovery summary graphics...")
    
    print("1. Creating Polish version...")
    create_discovery_summary()
    
    print("2. Creating English version...")
    create_english_summary()
    
    print(f"\nSummary graphics saved to: {OUTPUT_DIR}")
    print("- discovery_summary_polish.png")
    print("- discovery_summary_english.png")

if __name__ == "__main__":
    main()