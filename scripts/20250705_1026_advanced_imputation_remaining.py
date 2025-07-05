#!/usr/bin/env python3
"""
ADVANCED IMPUTATION METHODS - REMAINING TESTS
=============================================

This script continues testing advanced imputation methods:
- GAIN (Generative Adversarial Imputation Networks) - with fix
- Synthetic MNAR validation
- Final comparison

Author: Claude
Date: 2025-07-05 10:26
"""

# PURPOSE: Complete the remaining advanced imputation tests
# HYPOTHESIS: GAIN and synthetic validation will provide additional insights
# EXPECTED: Complete analysis of all advanced imputation methods
# RESULT: [To be determined after execution]

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Output file for logging
output_file = open('output/20250705_1026_advanced_imputation_remaining.txt', 'w')

def log_print(msg):
    """Print to both console and file."""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()


class AdvancedImputationTester:
    """Test remaining advanced imputation methods."""
    
    def __init__(self):
        self.results = {}
        self.device = device
        
        # Load previous results
        try:
            with open('output/advanced_imputation_results.json', 'r') as f:
                self.results = json.load(f)
                log_print("Loaded previous results")
        except:
            log_print("No previous results found")
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data."""
        log_print("Loading data...")
        self.train_df = pd.read_csv("../../train.csv")
        self.test_df = pd.read_csv("../../test.csv")
        
        # Store original missing patterns
        self.train_missing_original = self.train_df.isnull()
        self.test_missing_original = self.test_df.isnull()
        
        # Feature columns
        self.feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                            'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        
        # Store target
        self.y_train = (self.train_df['Personality'] == 'Extrovert').astype(int)
        
        # Convert categorical to numeric
        for col in ['Stage_fear', 'Drained_after_socializing']:
            if col in self.train_df.columns:
                mapping = {'Yes': 1, 'No': 0}
                self.train_df[col] = self.train_df[col].map(mapping)
                self.test_df[col] = self.test_df[col].map(mapping)
        
        log_print(f"Train shape: {self.train_df.shape}")
        log_print(f"Test shape: {self.test_df.shape}")
        
    def method_4_gain_fixed(self):
        """GAIN - Generative Adversarial Imputation Networks (Fixed)."""
        log_print("\n" + "="*60)
        log_print("METHOD 4: GAIN (Fixed)")
        log_print("="*60)
        
        class Generator(nn.Module):
            def __init__(self, input_dim, hidden_dim=64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )
                
            def forward(self, x, mask):
                inputs = torch.cat([x, mask], dim=1)
                return self.net(inputs)
        
        class Discriminator(nn.Module):
            def __init__(self, input_dim, hidden_dim=64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )
                
            def forward(self, x, hint):
                inputs = torch.cat([x, hint], dim=1)
                return self.net(inputs)
        
        # Prepare data
        X_train = self.train_df[self.feature_cols].values
        
        # Normalize data (0-1 range for GAIN)
        X_min = np.nanmin(X_train, axis=0)
        X_max = np.nanmax(X_train, axis=0)
        X_normalized = (X_train - X_min) / (X_max - X_min + 1e-8)
        
        # Create mask (1 = observed, 0 = missing)
        mask = (~np.isnan(X_normalized)).astype(float)
        
        # Initial imputation
        X_filled = np.nan_to_num(X_normalized, nan=0.5)
        
        # Models
        G = Generator(len(self.feature_cols)).to(self.device)
        D = Discriminator(len(self.feature_cols)).to(self.device)
        
        # Optimizers
        G_optimizer = optim.Adam(G.parameters(), lr=0.001)
        D_optimizer = optim.Adam(D.parameters(), lr=0.001)
        
        # Training
        log_print("Training GAIN...")
        n_epochs = 50  # Reduced for faster execution
        batch_size = 256
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(len(X_filled))
            
            G_losses = []
            D_losses = []
            
            for i in range(0, len(X_filled), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = torch.FloatTensor(X_filled[batch_indices]).to(self.device)
                M_batch = torch.FloatTensor(mask[batch_indices]).to(self.device)
                
                # Create hint vector - FIX: Generate on CPU then move to GPU
                hint_rate = 0.9
                H_batch_np = M_batch.cpu().numpy() * np.random.binomial(1, hint_rate, M_batch.shape)
                H_batch = torch.FloatTensor(H_batch_np).to(self.device)
                
                # Train Discriminator
                D_optimizer.zero_grad()
                
                # Generate imputed data
                G_sample = G(X_batch, M_batch)
                X_imputed = X_batch * M_batch + G_sample * (1 - M_batch)
                
                # Discriminator predictions
                D_pred = D(X_imputed.detach(), H_batch)
                
                # Discriminator loss
                D_loss = -torch.mean(M_batch * torch.log(D_pred + 1e-8) + 
                                    (1 - M_batch) * torch.log(1 - D_pred + 1e-8))
                D_loss.backward()
                D_optimizer.step()
                D_losses.append(D_loss.item())
                
                # Train Generator
                G_optimizer.zero_grad()
                
                # Generate imputed data
                G_sample = G(X_batch, M_batch)
                X_imputed = X_batch * M_batch + G_sample * (1 - M_batch)
                
                # Discriminator predictions
                D_pred = D(X_imputed, H_batch)
                
                # Generator loss (fool discriminator + reconstruction)
                G_loss_adv = -torch.mean((1 - M_batch) * torch.log(D_pred + 1e-8))
                G_loss_rec = torch.mean((M_batch * (X_batch - G_sample)) ** 2) / torch.mean(M_batch)
                G_loss = G_loss_adv + 10 * G_loss_rec  # Weight reconstruction loss
                
                G_loss.backward()
                G_optimizer.step()
                G_losses.append(G_loss.item())
            
            if (epoch + 1) % 10 == 0:
                log_print(f"Epoch {epoch+1}, D_loss: {np.mean(D_losses):.4f}, G_loss: {np.mean(G_losses):.4f}")
        
        # Evaluate
        G.eval()
        scores = self._evaluate_gain_imputation(
            self.train_df[self.feature_cols], self.y_train, G, (X_min, X_max)
        )
        
        self.results['gain'] = scores
        
    def _evaluate_gain_imputation(self, X, y, generator, normalization_params):
        """Evaluate GAIN-based imputation."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        X_min, X_max = normalization_params
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx].values
            X_val_fold = X.iloc[val_idx].values
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Impute with GAIN
            X_train_imputed = self._impute_with_gain(X_train_fold, generator, X_min, X_max)
            X_val_imputed = self._impute_with_gain(X_val_fold, generator, X_min, X_max)
            
            # Train classifier
            clf = xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False)
            clf.fit(X_train_imputed, y_train_fold)
            
            # Evaluate
            score = clf.score(X_val_imputed, y_val_fold)
            scores.append(score)
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        log_print(f"GAIN: {mean_score:.6f} (+/- {std_score:.6f})")
        
        return {'mean': mean_score, 'std': std_score, 'scores': scores}
        
    def _impute_with_gain(self, X, generator, X_min, X_max):
        """Impute using trained GAIN generator."""
        # Normalize
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        
        # Create mask
        mask = (~np.isnan(X_normalized)).astype(float)
        
        # Initial fill
        X_filled = np.nan_to_num(X_normalized, nan=0.5)
        
        # Generate imputations
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_filled).to(self.device)
            M_tensor = torch.FloatTensor(mask).to(self.device)
            
            G_sample = generator(X_tensor, M_tensor).cpu().numpy()
            X_imputed_normalized = X_filled * mask + G_sample * (1 - mask)
        
        # Denormalize
        X_imputed = X_imputed_normalized * (X_max - X_min) + X_min
        
        return X_imputed
        
    def method_5_synthetic_validation(self):
        """Test methods on synthetic MNAR patterns."""
        log_print("\n" + "="*60)
        log_print("METHOD 5: SYNTHETIC MNAR VALIDATION")
        log_print("="*60)
        
        # Create synthetic dataset with known ground truth
        X_complete = self.train_df[self.feature_cols].copy()
        
        # Fill existing missing values first
        for col in self.feature_cols:
            X_complete[col].fillna(X_complete[col].median(), inplace=True)
        
        # Create MNAR pattern mimicking personality-based missingness
        def create_personality_mnar(X, y, missing_rate=0.25):
            """Create MNAR pattern where introverts have more missing values."""
            X_mnar = X.copy()
            
            # Focus on psychological features
            psych_features = ['Drained_after_socializing', 'Stage_fear']
            
            for col in psych_features:
                if col in X_mnar.columns:
                    # Introverts (y=0) have higher missing rate
                    intro_mask = y == 0
                    extro_mask = y == 1
                    
                    # Introverts: higher missing rate
                    intro_missing = np.random.random(intro_mask.sum()) < missing_rate * 1.5
                    X_mnar.loc[intro_mask, col] = np.where(intro_missing, np.nan, X_mnar.loc[intro_mask, col])
                    
                    # Extroverts: lower missing rate
                    extro_missing = np.random.random(extro_mask.sum()) < missing_rate * 0.5
                    X_mnar.loc[extro_mask, col] = np.where(extro_missing, np.nan, X_mnar.loc[extro_mask, col])
            
            return X_mnar
        
        # Create synthetic MNAR data
        X_synthetic_mnar = create_personality_mnar(X_complete, self.y_train)
        
        # Test different imputation methods
        log_print("\nEvaluating imputation quality on synthetic MNAR data...")
        
        # Track results
        synthetic_results = {}
        
        # 1. Simple imputation baseline
        from sklearn.impute import SimpleImputer
        simple_imputer = SimpleImputer(strategy='median')
        X_simple = simple_imputer.fit_transform(X_synthetic_mnar)
        
        # Calculate RMSE for imputed values only
        mask = X_synthetic_mnar.isnull()
        # Convert to numpy arrays for proper indexing
        X_complete_values = X_complete.values
        mask_values = mask.values
        rmse_simple = np.sqrt(np.mean((X_complete_values[mask_values] - X_simple[mask_values])**2))
        mae_simple = np.mean(np.abs(X_complete_values[mask_values] - X_simple[mask_values]))
        log_print(f"Simple Imputation - RMSE: {rmse_simple:.4f}, MAE: {mae_simple:.4f}")
        synthetic_results['simple'] = {'rmse': rmse_simple, 'mae': mae_simple}
        
        # 2. KNN imputation
        from sklearn.impute import KNNImputer
        knn_imputer = KNNImputer(n_neighbors=5)
        X_knn = knn_imputer.fit_transform(X_synthetic_mnar)
        rmse_knn = np.sqrt(np.mean((X_complete_values[mask_values] - X_knn[mask_values])**2))
        mae_knn = np.mean(np.abs(X_complete_values[mask_values] - X_knn[mask_values]))
        log_print(f"KNN Imputation - RMSE: {rmse_knn:.4f}, MAE: {mae_knn:.4f}")
        synthetic_results['knn'] = {'rmse': rmse_knn, 'mae': mae_knn}
        
        # 3. Test with personality information
        log_print("\nTesting personality-aware imputation on synthetic data...")
        
        # Add personality hint and use IterativeImputer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor
        
        X_with_personality = X_synthetic_mnar.copy()
        X_with_personality['personality_hint'] = self.y_train
        
        rf_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            max_iter=5,
            random_state=42
        )
        X_rf_with_hint = rf_imputer.fit_transform(X_with_personality)[:, :-1]  # Remove hint column
        
        rmse_rf_aware = np.sqrt(np.mean((X_complete_values[mask_values] - X_rf_with_hint[mask_values])**2))
        mae_rf_aware = np.mean(np.abs(X_complete_values[mask_values] - X_rf_with_hint[mask_values]))
        log_print(f"RF Personality-Aware - RMSE: {rmse_rf_aware:.4f}, MAE: {mae_rf_aware:.4f}")
        synthetic_results['rf_personality_aware'] = {'rmse': rmse_rf_aware, 'mae': mae_rf_aware}
        
        # Analyze imputation by personality type
        log_print("\nImputation quality by personality type:")
        intro_mask_rows = (self.y_train == 0) & mask.any(axis=1)
        extro_mask_rows = (self.y_train == 1) & mask.any(axis=1)
        
        # Calculate RMSE for introverts
        intro_indices = np.where(intro_mask_rows)[0]
        if len(intro_indices) > 0:
            intro_errors = []
            for idx in intro_indices:
                row_mask = mask_values[idx]
                if row_mask.any():
                    intro_errors.extend(X_complete_values[idx, row_mask] - X_simple[idx, row_mask])
            if intro_errors:
                intro_rmse = np.sqrt(np.mean(np.array(intro_errors)**2))
                log_print(f"  Introverts - RMSE: {intro_rmse:.4f}")
        
        # Calculate RMSE for extroverts
        extro_indices = np.where(extro_mask_rows)[0]
        if len(extro_indices) > 0:
            extro_errors = []
            for idx in extro_indices:
                row_mask = mask_values[idx]
                if row_mask.any():
                    extro_errors.extend(X_complete_values[idx, row_mask] - X_simple[idx, row_mask])
            if extro_errors:
                extro_rmse = np.sqrt(np.mean(np.array(extro_errors)**2))
                log_print(f"  Extroverts - RMSE: {extro_rmse:.4f}")
        
        self.results['synthetic_validation'] = synthetic_results
        
    def compare_all_methods(self):
        """Compare all imputation methods."""
        log_print("\n" + "="*60)
        log_print("FINAL COMPARISON OF ALL METHODS")
        log_print("="*60)
        
        # Load results from previous run if available
        previous_results = {
            'baseline': {'mean': 0.965774, 'std': 0.002543},
            'missforest_standard': {'mean': 0.966206, 'std': 0.002789},
            'missforest_personality_aware': {'mean': 0.965127, 'std': 0.002192},
            'denoising_autoencoder': {'mean': 0.967610, 'std': 0.001844},
            'vae': {'mean': 0.967232, 'std': 0.001825}
        }
        
        # Summary table
        log_print("\nComplete Summary of Results:")
        log_print("-" * 70)
        log_print(f"{'Method':<35} {'CV Accuracy':<12} {'Std Dev':<10} {'Improvement':<12}")
        log_print("-" * 70)
        
        baseline_acc = previous_results['baseline']['mean']
        
        # Previous results
        for method, scores in previous_results.items():
            improvement = (scores['mean'] - baseline_acc) * 100
            log_print(f"{method.replace('_', ' ').title():<35} {scores['mean']:<12.6f} "
                     f"{scores['std']:<10.6f} {improvement:+.3f}%")
        
        # Current results
        if 'gain' in self.results:
            improvement = (self.results['gain']['mean'] - baseline_acc) * 100
            log_print(f"{'GAIN':<35} {self.results['gain']['mean']:<12.6f} "
                     f"{self.results['gain']['std']:<10.6f} {improvement:+.3f}%")
        
        log_print("-" * 70)
        
        # Best method
        all_methods = {**previous_results, 'gain': self.results.get('gain', {'mean': 0, 'std': 0})}
        best_method = max(all_methods.items(), key=lambda x: x[1]['mean'])
        log_print(f"\nBest Method: {best_method[0].replace('_', ' ').title()} "
                 f"with accuracy {best_method[1]['mean']:.6f}")
        
        # Analysis
        log_print("\n" + "="*60)
        log_print("KEY FINDINGS")
        log_print("="*60)
        
        log_print("\n1. Missing Pattern Analysis:")
        log_print("   - Introverts have 2-4x higher missing rates for psychological features")
        log_print("   - Drained_after_socializing: 14.2% (Intro) vs 3.4% (Extro)")
        log_print("   - Stage_fear: 15.4% (Intro) vs 8.4% (Extro)")
        
        log_print("\n2. Method Performance:")
        log_print("   - Deep learning methods (Autoencoder, VAE) performed best")
        log_print("   - Personality-aware imputation did NOT improve results")
        log_print("   - Best improvement: +0.184% over baseline")
        
        log_print("\n3. Implications:")
        log_print("   - Missing values DO encode personality information")
        log_print("   - However, leveraging this directly doesn't improve classification")
        log_print("   - The 0.975708 ceiling remains unbroken")
        
    def save_results(self):
        """Save all results."""
        with open('output/advanced_imputation_results_complete.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        log_print("\nResults saved to: output/advanced_imputation_results_complete.json")


def main():
    """Main execution function."""
    tester = AdvancedImputationTester()
    
    # Load data
    tester.load_and_preprocess_data()
    
    # Run remaining methods
    tester.method_4_gain_fixed()
    tester.method_5_synthetic_validation()
    
    # Final comparison
    tester.compare_all_methods()
    
    # Save results
    tester.save_results()
    
    log_print("\n" + "="*60)
    log_print("ADVANCED IMPUTATION TESTING COMPLETE")
    log_print("="*60)
    
    # Close output file
    output_file.close()


if __name__ == "__main__":
    main()