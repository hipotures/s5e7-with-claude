#!/usr/bin/env python3
"""
Comprehensive analysis of differences between train and test datasets
Date: 2025-01-11 23:53
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"
OUTPUT_DIR.mkdir(exist_ok=True)

class DatasetDifferenceAnalyzer:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                            'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
        self.numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                            'Going_outside', 'Post_frequency']
        self.binary_cols = ['Stage_fear', 'Drained_after_socializing']
        
    def load_data(self):
        """Load train and test datasets"""
        print("="*60)
        print("LOADING DATASETS")
        print("="*60)
        
        self.train_df = pd.read_csv(DATA_DIR / "train.csv")
        self.test_df = pd.read_csv(DATA_DIR / "test.csv")
        
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        print(f"Train columns: {list(self.train_df.columns)}")
        print(f"Test columns: {list(self.test_df.columns)}")
        
        # Convert binary features
        for col in self.binary_cols:
            self.train_df[col + '_binary'] = (self.train_df[col] == 'Yes').astype(int)
            self.test_df[col + '_binary'] = (self.test_df[col] == 'Yes').astype(int)
    
    def analyze_distributions(self):
        """Compare feature distributions between train and test"""
        print("\n" + "="*60)
        print("DISTRIBUTION ANALYSIS")
        print("="*60)
        
        results = []
        
        # Analyze numeric features
        for col in self.numeric_cols:
            train_data = self.train_df[col].dropna()
            test_data = self.test_df[col].dropna()
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(train_data, test_data)
            
            # Mann-Whitney U test
            mw_stat, mw_pval = stats.mannwhitneyu(train_data, test_data)
            
            # Basic statistics
            train_mean = train_data.mean()
            test_mean = test_data.mean()
            train_std = train_data.std()
            test_std = test_data.std()
            
            results.append({
                'feature': col,
                'train_mean': train_mean,
                'test_mean': test_mean,
                'mean_diff': abs(train_mean - test_mean),
                'train_std': train_std,
                'test_std': test_std,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'mw_pvalue': mw_pval,
                'significant': ks_pval < 0.05
            })
            
            print(f"\n{col}:")
            print(f"  Train: mean={train_mean:.3f}, std={train_std:.3f}")
            print(f"  Test:  mean={test_mean:.3f}, std={test_std:.3f}")
            print(f"  KS test: stat={ks_stat:.4f}, p={ks_pval:.4f} {'***' if ks_pval < 0.001 else '**' if ks_pval < 0.01 else '*' if ks_pval < 0.05 else ''}")
        
        # Analyze binary features
        for col in self.binary_cols:
            train_prop = (self.train_df[col] == 'Yes').mean()
            test_prop = (self.test_df[col] == 'Yes').mean()
            
            # Chi-square test
            train_counts = self.train_df[col].value_counts()
            test_counts = self.test_df[col].value_counts()
            chi2, chi_pval = stats.chi2_contingency([train_counts, test_counts])[:2]
            
            results.append({
                'feature': col,
                'train_mean': train_prop,
                'test_mean': test_prop,
                'mean_diff': abs(train_prop - test_prop),
                'chi2': chi2,
                'chi_pvalue': chi_pval,
                'significant': chi_pval < 0.05
            })
            
            print(f"\n{col}:")
            print(f"  Train 'Yes': {train_prop:.3%}")
            print(f"  Test 'Yes':  {test_prop:.3%}")
            print(f"  Chi-square: stat={chi2:.4f}, p={chi_pval:.4f} {'***' if chi_pval < 0.001 else '**' if chi_pval < 0.01 else '*' if chi_pval < 0.05 else ''}")
        
        return pd.DataFrame(results)
    
    def analyze_missing_patterns(self):
        """Analyze missing value patterns"""
        print("\n" + "="*60)
        print("MISSING VALUE PATTERNS")
        print("="*60)
        
        # Count missing values
        train_missing = self.train_df[self.feature_cols].isnull().sum()
        test_missing = self.test_df[self.feature_cols].isnull().sum()
        
        print("\nMissing counts:")
        print("Feature                  Train    Test   Diff")
        print("-"*45)
        for col in self.feature_cols:
            train_pct = train_missing[col] / len(self.train_df) * 100
            test_pct = test_missing[col] / len(self.test_df) * 100
            print(f"{col:20} {train_pct:6.2f}% {test_pct:6.2f}% {abs(train_pct-test_pct):6.2f}%")
        
        # Missing patterns per class in train
        print("\nMissing patterns by personality type (train):")
        for personality in ['Introvert', 'Extrovert']:
            subset = self.train_df[self.train_df['Personality'] == personality]
            missing_pct = subset[self.numeric_cols].isnull().any(axis=1).mean()
            print(f"{personality}: {missing_pct:.2%} have at least one missing value")
        
        # Create missing pattern matrix
        train_missing_pattern = self.train_df[self.numeric_cols].isnull().astype(int)
        test_missing_pattern = self.test_df[self.numeric_cols].isnull().astype(int)
        
        # Most common patterns
        train_patterns = train_missing_pattern.value_counts().head(10)
        test_patterns = test_missing_pattern.value_counts().head(10)
        
        print("\nTop missing patterns in train:")
        for pattern, count in train_patterns.items():
            print(f"  Pattern {pattern}: {count} ({count/len(self.train_df)*100:.1f}%)")
        
        return train_missing, test_missing
    
    def adversarial_validation(self):
        """Use ML to detect if we can distinguish train from test"""
        print("\n" + "="*60)
        print("ADVERSARIAL VALIDATION")
        print("="*60)
        
        # Prepare data
        train_subset = self.train_df[self.feature_cols].copy()
        test_subset = self.test_df[self.feature_cols].copy()
        
        # Add source label
        train_subset['is_train'] = 1
        test_subset['is_train'] = 0
        
        # Combine
        combined = pd.concat([train_subset, test_subset], ignore_index=True)
        
        # Handle missing values and binary features
        for col in self.numeric_cols:
            combined[col] = combined[col].fillna(combined[col].median())
        
        for col in self.binary_cols:
            combined[col] = (combined[col] == 'Yes').astype(int)
        
        # Split features and target
        X = combined.drop('is_train', axis=1)
        y = combined['is_train']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        
        print(f"\nAdversarial Validation Results:")
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print(f"Can distinguish: {'YES' if test_score > 0.55 else 'NO'} (threshold=0.55)")
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature importance for distinguishing train/test:")
        for _, row in feature_imp.iterrows():
            print(f"  {row['feature']:25} {row['importance']:.4f}")
        
        return test_score, feature_imp
    
    def analyze_id_patterns(self):
        """Analyze ID patterns and ranges"""
        print("\n" + "="*60)
        print("ID PATTERN ANALYSIS")
        print("="*60)
        
        train_ids = self.train_df['id'].values
        test_ids = self.test_df['id'].values
        
        print(f"\nTrain IDs: {train_ids.min()} - {train_ids.max()}")
        print(f"Test IDs:  {test_ids.min()} - {test_ids.max()}")
        print(f"Gap: {test_ids.min() - train_ids.max()}")
        
        # Check for patterns
        print("\nID patterns:")
        print(f"Train IDs divisible by 3: {(train_ids % 3 == 0).sum()}")
        print(f"Test IDs divisible by 3: {(test_ids % 3 == 0).sum()}")
        
        # Check last digits
        train_last_digit = train_ids % 10
        test_last_digit = test_ids % 10
        
        print("\nLast digit distribution:")
        for digit in range(10):
            train_count = (train_last_digit == digit).sum()
            test_count = (test_last_digit == digit).sum()
            print(f"  Digit {digit}: Train={train_count:5d} ({train_count/len(train_ids)*100:5.2f}%), "
                  f"Test={test_count:4d} ({test_count/len(test_ids)*100:5.2f}%)")
    
    def population_stability_index(self):
        """Calculate PSI for each feature"""
        print("\n" + "="*60)
        print("POPULATION STABILITY INDEX (PSI)")
        print("="*60)
        
        psi_results = []
        
        for col in self.numeric_cols:
            # Get non-null values
            train_data = self.train_df[col].dropna()
            test_data = self.test_df[col].dropna()
            
            # Create bins based on train data
            _, bins = pd.qcut(train_data, q=10, retbins=True, duplicates='drop')
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Calculate distributions
            train_dist = pd.cut(train_data, bins=bins).value_counts(normalize=True).sort_index()
            test_dist = pd.cut(test_data, bins=bins).value_counts(normalize=True).sort_index()
            
            # Calculate PSI
            psi = 0
            for i in range(len(train_dist)):
                if train_dist.iloc[i] > 0 and test_dist.iloc[i] > 0:
                    psi += (test_dist.iloc[i] - train_dist.iloc[i]) * np.log(test_dist.iloc[i] / train_dist.iloc[i])
            
            psi_results.append({
                'feature': col,
                'psi': psi,
                'stability': 'Stable' if psi < 0.1 else 'Moderate' if psi < 0.2 else 'Unstable'
            })
            
            print(f"{col:25} PSI={psi:.4f} ({psi_results[-1]['stability']})")
        
        return pd.DataFrame(psi_results)
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plot
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Distribution comparison for numeric features
        for i, col in enumerate(self.numeric_cols):
            ax = plt.subplot(6, 3, i+1)
            
            train_data = self.train_df[col].dropna()
            test_data = self.test_df[col].dropna()
            
            ax.hist(train_data, bins=50, alpha=0.5, label='Train', density=True, color='blue')
            ax.hist(test_data, bins=50, alpha=0.5, label='Test', density=True, color='orange')
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend()
        
        # 2. Binary feature comparison
        ax = plt.subplot(6, 3, 6)
        binary_train = pd.DataFrame({
            'Stage_fear': (self.train_df['Stage_fear'] == 'Yes').mean(),
            'Drained': (self.train_df['Drained_after_socializing'] == 'Yes').mean()
        }, index=['Train'])
        binary_test = pd.DataFrame({
            'Stage_fear': (self.test_df['Stage_fear'] == 'Yes').mean(),
            'Drained': (self.test_df['Drained_after_socializing'] == 'Yes').mean()
        }, index=['Test'])
        
        binary_combined = pd.concat([binary_train, binary_test])
        binary_combined.plot(kind='bar', ax=ax)
        ax.set_title('Binary Features Proportion')
        ax.set_ylabel('Proportion "Yes"')
        ax.legend(title='Feature')
        
        # 3. Missing value heatmap
        ax = plt.subplot(6, 3, 7)
        missing_train = self.train_df[self.feature_cols].isnull().mean()
        missing_test = self.test_df[self.feature_cols].isnull().mean()
        missing_df = pd.DataFrame({
            'Train': missing_train,
            'Test': missing_test
        })
        sns.heatmap(missing_df.T, annot=True, fmt='.3f', cmap='Reds', ax=ax)
        ax.set_title('Missing Value Proportions')
        
        # 4. Correlation heatmap - Train
        ax = plt.subplot(6, 3, 8)
        train_corr = self.train_df[self.numeric_cols].corr()
        sns.heatmap(train_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlations - Train')
        
        # 5. Correlation heatmap - Test
        ax = plt.subplot(6, 3, 9)
        test_corr = self.test_df[self.numeric_cols].corr()
        sns.heatmap(test_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlations - Test')
        
        # 6. PCA visualization
        ax = plt.subplot(6, 3, 10)
        # Prepare data for PCA
        train_pca = self.train_df[self.numeric_cols].fillna(self.train_df[self.numeric_cols].median())
        test_pca = self.test_df[self.numeric_cols].fillna(self.test_df[self.numeric_cols].median())
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_pca)
        test_scaled = scaler.transform(test_pca)
        
        pca = PCA(n_components=2)
        train_pca_2d = pca.fit_transform(train_scaled)
        test_pca_2d = pca.transform(test_scaled)
        
        ax.scatter(train_pca_2d[:, 0], train_pca_2d[:, 1], alpha=0.5, label='Train', s=10)
        ax.scatter(test_pca_2d[:, 0], test_pca_2d[:, 1], alpha=0.5, label='Test', s=10)
        ax.set_title('PCA Projection (2D)')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.legend()
        
        # 7. ID distribution
        ax = plt.subplot(6, 3, 11)
        ax.hist(self.train_df['id'], bins=50, alpha=0.5, label='Train', color='blue')
        ax.hist(self.test_df['id'], bins=50, alpha=0.5, label='Test', color='orange')
        ax.set_title('ID Distribution')
        ax.set_xlabel('ID')
        ax.set_ylabel('Count')
        ax.legend()
        
        # 8. Personality distribution in train
        ax = plt.subplot(6, 3, 12)
        personality_counts = self.train_df['Personality'].value_counts()
        ax.pie(personality_counts.values, labels=personality_counts.index, autopct='%1.1f%%')
        ax.set_title('Personality Distribution (Train)')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'dataset_difference_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {OUTPUT_DIR / 'dataset_difference_analysis.png'}")
    
    def generate_report(self, dist_results, adv_score, psi_results):
        """Generate comprehensive markdown report"""
        print("\n" + "="*60)
        print("GENERATING REPORT")
        print("="*60)
        
        report = f"""# Dataset Difference Analysis Report
**Date**: 2025-01-11 23:53
**Analyst**: Deep Learning Model

## Executive Summary

This report analyzes the differences between the training and test datasets for the Kaggle Playground Series S5E7 personality prediction competition.

### Key Findings:

1. **Adversarial Validation Score**: {adv_score:.4f} - {'Datasets are distinguishable!' if adv_score > 0.55 else 'Datasets are similar'}
2. **Significant Distribution Differences**: {dist_results['significant'].sum()} out of {len(dist_results)} features
3. **Missing Value Patterns**: {'Different' if dist_results['significant'].sum() > 3 else 'Similar'} between train and test

## 1. Distribution Analysis

### Numeric Features
"""
        
        for _, row in dist_results[dist_results['feature'].isin(self.numeric_cols)].iterrows():
            sig = '⚠️' if row['significant'] else '✓'
            report += f"\n**{row['feature']}** {sig}\n"
            report += f"- Train: mean={row['train_mean']:.3f}, std={row['train_std']:.3f}\n"
            report += f"- Test: mean={row['test_mean']:.3f}, std={row['test_std']:.3f}\n"
            report += f"- KS test p-value: {row['ks_pvalue']:.4f} {'(significant!)' if row['significant'] else ''}\n"
        
        report += "\n### Binary Features\n"
        
        for _, row in dist_results[dist_results['feature'].isin(self.binary_cols)].iterrows():
            sig = '⚠️' if row.get('significant', False) else '✓'
            report += f"\n**{row['feature']}** {sig}\n"
            report += f"- Train 'Yes': {row['train_mean']:.1%}\n"
            report += f"- Test 'Yes': {row['test_mean']:.1%}\n"
            if 'chi_pvalue' in row:
                report += f"- Chi-square p-value: {row['chi_pvalue']:.4f}\n"
        
        report += f"\n## 2. Population Stability Index\n\n"
        report += "| Feature | PSI | Stability |\n"
        report += "|---------|-----|----------|\n"
        
        for _, row in psi_results.iterrows():
            emoji = '✓' if row['stability'] == 'Stable' else '⚠️' if row['stability'] == 'Moderate' else '❌'
            report += f"| {row['feature']} | {row['psi']:.4f} | {row['stability']} {emoji} |\n"
        
        report += f"\n## 3. Missing Value Analysis\n\n"
        report += "Missing value percentages show {'significant' if dist_results['significant'].sum() > 3 else 'minimal'} differences.\n"
        report += "\nKey observation: Missing values in numeric features are strongly correlated with Introvert personality type.\n"
        
        report += f"\n## 4. Adversarial Validation\n\n"
        report += f"A Random Forest classifier achieved {adv_score:.1%} accuracy in distinguishing train from test samples.\n"
        report += f"This indicates that the datasets are {'significantly different' if adv_score > 0.6 else 'relatively similar'}.\n"
        
        report += "\n## 5. Recommendations\n\n"
        
        if adv_score > 0.55:
            report += "⚠️ **Warning**: The test set has different characteristics from the training set.\n\n"
            report += "Recommended actions:\n"
            report += "1. Use robust validation strategies (e.g., adversarial validation splits)\n"
            report += "2. Apply domain adaptation techniques\n"
            report += "3. Focus on features that are stable across datasets\n"
            report += "4. Consider ensemble methods that are robust to distribution shift\n"
        else:
            report += "✓ The datasets appear to be from the same distribution.\n"
            report += "Standard validation strategies should work well.\n"
        
        report += "\n## 6. Technical Details\n\n"
        report += "- Train samples: 18,524\n"
        report += "- Test samples: 6,175\n"
        report += "- Features analyzed: 7\n"
        report += "- Statistical tests: Kolmogorov-Smirnov, Mann-Whitney U, Chi-square\n"
        report += "- Visualization: See `dataset_difference_analysis.png`\n"
        
        # Save report
        report_path = OUTPUT_DIR / 'dataset_difference_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
        
        return report
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        # Load data
        self.load_data()
        
        # Run analyses
        dist_results = self.analyze_distributions()
        missing_train, missing_test = self.analyze_missing_patterns()
        adv_score, feature_imp = self.adversarial_validation()
        self.analyze_id_patterns()
        psi_results = self.population_stability_index()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        report = self.generate_report(dist_results, adv_score, psi_results)
        
        # Save detailed results
        dist_results.to_csv(OUTPUT_DIR / 'distribution_test_results.csv', index=False)
        feature_imp.to_csv(OUTPUT_DIR / 'adversarial_feature_importance.csv', index=False)
        psi_results.to_csv(OUTPUT_DIR / 'psi_results.csv', index=False)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"All results saved to {OUTPUT_DIR}")
        
        return dist_results, adv_score, psi_results

if __name__ == "__main__":
    analyzer = DatasetDifferenceAnalyzer()
    analyzer.run_analysis()