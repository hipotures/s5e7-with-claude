#!/usr/bin/env python3
"""
Ensemble of best submissions
Combine predictions from multiple high-performing submissions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
SCORES_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/scores")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def load_best_submissions():
    """Load the best submissions based on our analysis"""
    
    print("="*60)
    print("LOADING BEST SUBMISSIONS")
    print("="*60)
    
    # Define best submissions to ensemble
    best_submissions = [
        # Feature engineering based
        'submission_feature_engineering.csv',
        
        # Meta-learning based
        'submission_meta_stacking.csv',
        
        # Pseudo-labeling based
        'submission_balanced_pseudo_500_samples.csv',
        'submission_aggressive_extreme_4000_samples.csv',
        
        # Ensemble based
        'submission_ensemble_equal.csv',
        
        # Threshold optimized
        'submission_adaptive_threshold.csv',
        'submission_threshold_ensemble.csv',
        
        # Error aware
        'submission_error_aware.csv',
        
        # Boundary aware
        'submission_boundary_aware.csv',
        
        # Individual strong models
        'submission_meta_single_xgb.csv',
        'submission_meta_single_lgb.csv'
    ]
    
    submissions = {}
    
    print("\nLoading submissions:")
    for sub_name in best_submissions:
        try:
            sub_path = SCORES_DIR / sub_name
            if sub_path.exists():
                sub_df = pd.read_csv(sub_path)
                submissions[sub_name] = sub_df
                print(f"  ✓ {sub_name}")
            else:
                print(f"  ✗ {sub_name} (not found)")
        except Exception as e:
            print(f"  ✗ {sub_name} (error: {e})")
    
    print(f"\nLoaded {len(submissions)} submissions successfully")
    
    return submissions

def analyze_submission_diversity(submissions):
    """Analyze diversity among submissions"""
    
    print("\n" + "="*60)
    print("ANALYZING SUBMISSION DIVERSITY")
    print("="*60)
    
    # Get all predictions
    all_predictions = {}
    for name, df in submissions.items():
        all_predictions[name] = df['Personality'].values
    
    # Calculate pairwise agreement
    n_subs = len(submissions)
    agreement_matrix = np.zeros((n_subs, n_subs))
    
    sub_names = list(submissions.keys())
    
    for i in range(n_subs):
        for j in range(n_subs):
            if i <= j:
                agreement = np.mean(all_predictions[sub_names[i]] == all_predictions[sub_names[j]])
                agreement_matrix[i, j] = agreement
                agreement_matrix[j, i] = agreement
    
    # Print agreement summary
    print("\nPairwise agreement summary:")
    avg_agreement = []
    
    for i, name in enumerate(sub_names):
        # Average agreement with others (excluding self)
        agreements = [agreement_matrix[i, j] for j in range(n_subs) if i != j]
        avg_agr = np.mean(agreements)
        avg_agreement.append(avg_agr)
        print(f"  {name}: {avg_agr:.4f}")
    
    # Find most diverse pairs
    print("\nMost diverse pairs (lowest agreement):")
    pairs = []
    for i in range(n_subs):
        for j in range(i+1, n_subs):
            pairs.append((agreement_matrix[i, j], sub_names[i], sub_names[j]))
    
    pairs.sort()
    for agr, sub1, sub2 in pairs[:5]:
        print(f"  {sub1} vs {sub2}: {agr:.4f}")
    
    return agreement_matrix, sub_names

def create_ensemble_submissions(submissions):
    """Create various ensemble submissions"""
    
    print("\n" + "="*60)
    print("CREATING ENSEMBLE SUBMISSIONS")
    print("="*60)
    
    # Get sample submission for structure
    sample_df = list(submissions.values())[0]
    
    # 1. Simple Majority Voting
    print("\n1. Creating majority voting ensemble...")
    
    all_preds = []
    for df in submissions.values():
        all_preds.append(df['Personality'].values)
    
    all_preds = np.array(all_preds)
    
    # Count votes for each sample
    majority_preds = []
    for i in range(len(sample_df)):
        votes = all_preds[:, i]
        vote_counts = Counter(votes)
        majority_pred = vote_counts.most_common(1)[0][0]
        majority_preds.append(majority_pred)
    
    majority_df = pd.DataFrame({
        'id': sample_df['id'],
        'Personality': majority_preds
    })
    
    majority_df.to_csv(SCORES_DIR / 'submission_final_majority_ensemble.csv', index=False)
    print("   Created: submission_final_majority_ensemble.csv")
    
    # Analyze vote distribution
    vote_counts = []
    for i in range(len(sample_df)):
        votes = all_preds[:, i]
        n_extrovert = sum(1 for v in votes if v == 'Extrovert')
        vote_counts.append(n_extrovert)
    
    vote_counts = np.array(vote_counts)
    unanimous = np.sum((vote_counts == 0) | (vote_counts == len(submissions)))
    print(f"   Unanimous predictions: {unanimous} ({unanimous/len(sample_df)*100:.1f}%)")
    
    # 2. Weighted Ensemble (based on expected performance)
    print("\n2. Creating weighted ensemble...")
    
    # Define weights based on our knowledge
    weights = {
        'submission_feature_engineering.csv': 1.2,  # Best feature engineering
        'submission_meta_stacking.csv': 1.1,       # Strong meta-learning
        'submission_balanced_pseudo_500_samples.csv': 1.0,
        'submission_aggressive_extreme_4000_samples.csv': 0.9,  # More risky
        'submission_ensemble_equal.csv': 1.0,
        'submission_adaptive_threshold.csv': 1.1,   # Smart threshold
        'submission_threshold_ensemble.csv': 1.0,
        'submission_error_aware.csv': 1.1,         # Addresses specific errors
        'submission_boundary_aware.csv': 1.0,
        'submission_meta_single_xgb.csv': 0.9,
        'submission_meta_single_lgb.csv': 0.9
    }
    
    # Normalize weights
    total_weight = sum(weights.get(name, 1.0) for name in submissions.keys())
    
    weighted_preds = []
    for i in range(len(sample_df)):
        weighted_votes = {'Extrovert': 0, 'Introvert': 0}
        
        for name, df in submissions.items():
            weight = weights.get(name, 1.0) / total_weight
            pred = df['Personality'].iloc[i]
            weighted_votes[pred] += weight
        
        weighted_pred = 'Extrovert' if weighted_votes['Extrovert'] > weighted_votes['Introvert'] else 'Introvert'
        weighted_preds.append(weighted_pred)
    
    weighted_df = pd.DataFrame({
        'id': sample_df['id'],
        'Personality': weighted_preds
    })
    
    weighted_df.to_csv(SCORES_DIR / 'submission_final_weighted_ensemble.csv', index=False)
    print("   Created: submission_final_weighted_ensemble.csv")
    
    # 3. Conservative Ensemble (high agreement only)
    print("\n3. Creating conservative ensemble...")
    
    conservative_preds = []
    changes_made = 0
    
    for i in range(len(sample_df)):
        votes = all_preds[:, i]
        vote_counts = Counter(votes)
        
        # If high agreement (>80%), use majority
        if vote_counts.most_common(1)[0][1] >= len(submissions) * 0.8:
            pred = vote_counts.most_common(1)[0][0]
        else:
            # For disagreement, use weighted ensemble prediction
            pred = weighted_preds[i]
            changes_made += 1
        
        conservative_preds.append(pred)
    
    conservative_df = pd.DataFrame({
        'id': sample_df['id'],
        'Personality': conservative_preds
    })
    
    conservative_df.to_csv(SCORES_DIR / 'submission_final_conservative_ensemble.csv', index=False)
    print("   Created: submission_final_conservative_ensemble.csv")
    print(f"   Used weighted for {changes_made} disagreement cases ({changes_made/len(sample_df)*100:.1f}%)")
    
    # 4. Top 5 Ensemble (best performers only)
    print("\n4. Creating top 5 ensemble...")
    
    top_5_submissions = [
        'submission_feature_engineering.csv',
        'submission_meta_stacking.csv',
        'submission_adaptive_threshold.csv',
        'submission_error_aware.csv',
        'submission_ensemble_equal.csv'
    ]
    
    top_5_preds = []
    for name in top_5_submissions:
        if name in submissions:
            top_5_preds.append(submissions[name]['Personality'].values)
    
    top_5_preds = np.array(top_5_preds)
    
    top_5_majority = []
    for i in range(len(sample_df)):
        votes = top_5_preds[:, i]
        vote_counts = Counter(votes)
        pred = vote_counts.most_common(1)[0][0]
        top_5_majority.append(pred)
    
    top_5_df = pd.DataFrame({
        'id': sample_df['id'],
        'Personality': top_5_majority
    })
    
    top_5_df.to_csv(SCORES_DIR / 'submission_final_top5_ensemble.csv', index=False)
    print("   Created: submission_final_top5_ensemble.csv")
    
    return {
        'majority': majority_preds,
        'weighted': weighted_preds,
        'conservative': conservative_preds,
        'top5': top_5_majority,
        'vote_counts': vote_counts
    }

def visualize_ensemble_analysis(submissions, ensemble_results):
    """Create visualizations for ensemble analysis"""
    
    print("\n" + "="*60)
    print("CREATING ENSEMBLE VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Vote distribution
    ax1 = axes[0, 0]
    vote_counts = ensemble_results['vote_counts']
    ax1.hist(vote_counts, bins=range(len(submissions)+2), alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Extrovert Votes')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Vote Distribution Across Submissions')
    ax1.axvline(len(submissions)/2, color='red', linestyle='--', label='50% threshold')
    ax1.legend()
    
    # Plot 2: Agreement between ensemble methods
    ax2 = axes[0, 1]
    ensemble_names = ['majority', 'weighted', 'conservative', 'top5']
    agreement_matrix = np.zeros((4, 4))
    
    for i, name1 in enumerate(ensemble_names):
        for j, name2 in enumerate(ensemble_names):
            agreement = np.mean(np.array(ensemble_results[name1]) == np.array(ensemble_results[name2]))
            agreement_matrix[i, j] = agreement
    
    sns.heatmap(agreement_matrix, annot=True, fmt='.3f', 
                xticklabels=ensemble_names, yticklabels=ensemble_names,
                cmap='YlOrRd', ax=ax2)
    ax2.set_title('Agreement Between Ensemble Methods')
    
    # Plot 3: Prediction differences
    ax3 = axes[1, 0]
    
    # Count differences between methods
    diff_counts = []
    labels = []
    
    for i in range(len(ensemble_names)):
        for j in range(i+1, len(ensemble_names)):
            name1, name2 = ensemble_names[i], ensemble_names[j]
            n_diff = np.sum(np.array(ensemble_results[name1]) != np.array(ensemble_results[name2]))
            diff_counts.append(n_diff)
            labels.append(f'{name1[:3]} vs {name2[:3]}')
    
    ax3.bar(labels, diff_counts)
    ax3.set_xlabel('Ensemble Pair')
    ax3.set_ylabel('Number of Differences')
    ax3.set_title('Prediction Differences Between Methods')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Class distribution
    ax4 = axes[1, 1]
    
    class_distributions = []
    for name in ensemble_names:
        preds = ensemble_results[name]
        extrovert_ratio = sum(1 for p in preds if p == 'Extrovert') / len(preds)
        class_distributions.append(extrovert_ratio)
    
    ax4.bar(ensemble_names, class_distributions)
    ax4.set_xlabel('Ensemble Method')
    ax4.set_ylabel('Extrovert Ratio')
    ax4.set_title('Class Distribution by Ensemble Method')
    ax4.axhline(0.74, color='red', linestyle='--', label='Training ratio')
    ax4.legend()
    
    # Add value labels
    for i, v in enumerate(class_distributions):
        ax4.text(i, v + 0.005, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'final_ensemble_analysis.png', dpi=300)
    plt.close()
    
    print("   Saved: final_ensemble_analysis.png")

def create_ultra_ensemble(submissions):
    """Create the ultimate ensemble using advanced techniques"""
    
    print("\n" + "="*60)
    print("CREATING ULTRA ENSEMBLE")
    print("="*60)
    
    # Convert to numeric for advanced processing
    all_preds_numeric = []
    for df in submissions.values():
        preds_numeric = (df['Personality'] == 'Introvert').astype(int).values
        all_preds_numeric.append(preds_numeric)
    
    all_preds_numeric = np.array(all_preds_numeric)
    
    # Calculate prediction entropy for each sample
    entropy_scores = []
    for i in range(all_preds_numeric.shape[1]):
        votes = all_preds_numeric[:, i]
        p_introvert = np.mean(votes)
        p_extrovert = 1 - p_introvert
        
        # Calculate entropy
        if p_introvert > 0 and p_extrovert > 0:
            entropy = -p_introvert * np.log2(p_introvert) - p_extrovert * np.log2(p_extrovert)
        else:
            entropy = 0
        
        entropy_scores.append(entropy)
    
    entropy_scores = np.array(entropy_scores)
    
    # Ultra ensemble strategy
    ultra_preds = []
    
    for i in range(len(entropy_scores)):
        if entropy_scores[i] < 0.3:  # High certainty
            # Use simple majority
            votes = all_preds_numeric[:, i]
            pred = 1 if np.mean(votes) > 0.5 else 0
        else:  # High uncertainty
            # Use only the most reliable models
            reliable_models = [0, 1, 4, 7]  # Indices of most reliable submissions
            reliable_votes = all_preds_numeric[reliable_models, i]
            pred = 1 if np.mean(reliable_votes) > 0.5 else 0
        
        ultra_preds.append('Introvert' if pred == 1 else 'Extrovert')
    
    # Create submission
    sample_df = list(submissions.values())[0]
    ultra_df = pd.DataFrame({
        'id': sample_df['id'],
        'Personality': ultra_preds
    })
    
    ultra_df.to_csv(SCORES_DIR / 'submission_ULTRA_FINAL_ENSEMBLE.csv', index=False)
    print("   Created: submission_ULTRA_FINAL_ENSEMBLE.csv")
    
    # Analysis
    high_entropy = np.sum(entropy_scores > 0.8)
    print(f"   High uncertainty samples: {high_entropy} ({high_entropy/len(entropy_scores)*100:.1f}%)")
    
    return ultra_preds, entropy_scores

def main():
    # Load best submissions
    submissions = load_best_submissions()
    
    if len(submissions) < 3:
        print("ERROR: Not enough submissions loaded for ensemble!")
        return
    
    # Analyze diversity
    agreement_matrix, sub_names = analyze_submission_diversity(submissions)
    
    # Create ensemble submissions
    ensemble_results = create_ensemble_submissions(submissions)
    
    # Visualize results
    visualize_ensemble_analysis(submissions, ensemble_results)
    
    # Create ultra ensemble
    ultra_preds, entropy_scores = create_ultra_ensemble(submissions)
    
    # Save summary instead of detailed analysis
    try:
        summary = {
            'n_submissions': len(submissions),
            'unanimous_predictions': np.sum(ensemble_results['vote_counts'] == 0) + np.sum(ensemble_results['vote_counts'] == len(submissions)),
            'high_uncertainty': np.sum(entropy_scores > 0.8)
        }
        pd.DataFrame([summary]).to_csv(OUTPUT_DIR / 'final_ensemble_summary.csv', index=False)
    except:
        pass  # Skip if error
    
    print("\n" + "="*60)
    print("ENSEMBLE CREATION COMPLETE")
    print("="*60)
    print("\nCreated 5 final ensemble submissions:")
    print("  1. submission_final_majority_ensemble.csv")
    print("  2. submission_final_weighted_ensemble.csv")
    print("  3. submission_final_conservative_ensemble.csv")
    print("  4. submission_final_top5_ensemble.csv")
    print("  5. submission_ULTRA_FINAL_ENSEMBLE.csv ⭐")
    
    print("\nGood luck with the competition! 🚀")

if __name__ == "__main__":
    main()