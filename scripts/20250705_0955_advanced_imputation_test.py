#!/usr/bin/env python3
"""
ADVANCED IMPUTATION METHODS TEST
================================

This script implements and tests advanced imputation methods including:
- MissForest-style Random Forest imputation
- Denoising Autoencoder (PyTorch)
- Variational Autoencoder (PyTorch)
- GAIN (Generative Adversarial Imputation Networks)

Special focus on personality-aware imputation since missing values encode personality.

Author: Claude
Date: 2025-07-05 09:55
"""

# PURPOSE: Test advanced imputation methods that can leverage personality-encoding missingness
# HYPOTHESIS: Advanced methods that model complex patterns can better utilize the MNAR nature of our data
# EXPECTED: Deep learning methods might capture personality-specific missing patterns better than traditional methods
# RESULT: [To be determined after execution]

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class AdvancedImputationTester:
    """Test various advanced imputation methods."""
    
    def __init__(self):
        self.results = {}
        self.device = device
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data."""
        print("Loading data...")
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
        
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        
        # Calculate missing statistics by personality
        self.analyze_missing_by_personality()
        
    def analyze_missing_by_personality(self):
        """Analyze missing patterns by personality type."""
        print("\n" + "="*60)
        print("MISSING PATTERNS BY PERSONALITY")
        print("="*60)
        
        for col in self.feature_cols:
            introverts_mask = self.y_train == 0
            extroverts_mask = self.y_train == 1
            
            intro_missing = self.train_df[introverts_mask][col].isnull().mean()
            extro_missing = self.train_df[extroverts_mask][col].isnull().mean()
            
            print(f"{col:.<30} Intro: {intro_missing:.3f}, Extro: {extro_missing:.3f}")
            
    def method_1_missforest_style(self):
        """MissForest-style imputation using Random Forest."""
        print("\n" + "="*60)
        print("METHOD 1: MISSFOREST-STYLE IMPUTATION")
        print("="*60)
        
        X_train = self.train_df[self.feature_cols].copy()
        
        # Standard MissForest
        print("\nTesting standard MissForest...")
        rf_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=42),
            max_iter=10,
            random_state=42
        )
        
        cv_scores_standard = self._evaluate_imputation_method(
            X_train, self.y_train, rf_imputer, "MissForest Standard"
        )
        
        # Personality-aware MissForest (using target in imputation for training)
        print("\nTesting personality-aware MissForest...")
        X_train_with_target = X_train.copy()
        X_train_with_target['personality_hint'] = self.y_train
        
        rf_imputer_aware = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=42),
            max_iter=10,
            random_state=42
        )
        
        cv_scores_aware = self._evaluate_personality_aware_imputation(
            X_train, self.y_train, rf_imputer_aware, "MissForest Personality-Aware"
        )
        
        self.results['missforest'] = {
            'standard': cv_scores_standard,
            'personality_aware': cv_scores_aware
        }
        
    def _evaluate_imputation_method(self, X, y, imputer, method_name):
        """Evaluate imputation method using cross-validation."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx].copy()
            X_val_fold = X.iloc[val_idx].copy()
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Impute
            X_train_imputed = imputer.fit_transform(X_train_fold)
            X_val_imputed = imputer.transform(X_val_fold)
            
            # Train classifier
            clf = xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False)
            clf.fit(X_train_imputed, y_train_fold)
            
            # Evaluate
            score = clf.score(X_val_imputed, y_val_fold)
            scores.append(score)
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{method_name}: {mean_score:.6f} (+/- {std_score:.6f})")
        
        return {'mean': mean_score, 'std': std_score, 'scores': scores}
        
    def _evaluate_personality_aware_imputation(self, X, y, imputer, method_name):
        """Evaluate personality-aware imputation (uses target during training imputation)."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx].copy()
            X_val_fold = X.iloc[val_idx].copy()
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Add personality hint for training imputation
            X_train_with_hint = X_train_fold.copy()
            X_train_with_hint['personality_hint'] = y_train_fold
            
            # Add dummy hint for validation (will be ignored but keeps shape consistent)
            X_val_with_hint = X_val_fold.copy()
            X_val_with_hint['personality_hint'] = 0  # Dummy value
            
            # Impute training data with hint
            X_train_imputed_with_hint = imputer.fit_transform(X_train_with_hint)
            X_train_imputed = X_train_imputed_with_hint[:, :-1]  # Remove hint column
            
            # Impute validation data (hint will be imputed but we ignore it)
            X_val_imputed_with_hint = imputer.transform(X_val_with_hint)
            X_val_imputed = X_val_imputed_with_hint[:, :-1]  # Remove hint column
            
            # Train classifier
            clf = xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False)
            clf.fit(X_train_imputed, y_train_fold)
            
            # Evaluate
            score = clf.score(X_val_imputed, y_val_fold)
            scores.append(score)
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{method_name}: {mean_score:.6f} (+/- {std_score:.6f})")
        
        return {'mean': mean_score, 'std': std_score, 'scores': scores}
        
    def method_2_denoising_autoencoder(self):
        """Denoising Autoencoder for imputation."""
        print("\n" + "="*60)
        print("METHOD 2: DENOISING AUTOENCODER")
        print("="*60)
        
        class DenoisingAutoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dims=[32, 16]):
                super().__init__()
                
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last dropout
                
                # Decoder
                decoder_layers = []
                hidden_dims_reversed = hidden_dims[::-1]
                prev_dim = hidden_dims[-1]
                for hidden_dim in hidden_dims_reversed[1:]:
                    decoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        # Prepare data
        X_train = self.train_df[self.feature_cols].values
        scaler = StandardScaler()
        
        # Create complete data for training (using median imputation as ground truth)
        X_complete = np.nan_to_num(X_train, nan=np.nanmedian(X_train, axis=0))
        X_scaled = scaler.fit_transform(X_complete)
        
        # Train autoencoder
        model = DenoisingAutoencoder(len(self.feature_cols)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create corrupted versions
        corruption_rate = 0.3
        mask = np.random.random(X_scaled.shape) < corruption_rate
        X_corrupted = X_scaled.copy()
        X_corrupted[mask] = 0
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        X_corrupted_tensor = torch.FloatTensor(X_corrupted).to(self.device)
        
        # Training
        print("Training denoising autoencoder...")
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = model(X_corrupted_tensor)
            loss = criterion(output, X_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Evaluate using the autoencoder for imputation
        model.eval()
        scores = self._evaluate_autoencoder_imputation(
            self.train_df[self.feature_cols], self.y_train, model, scaler
        )
        
        self.results['denoising_autoencoder'] = scores
        
    def _evaluate_autoencoder_imputation(self, X, y, model, scaler):
        """Evaluate autoencoder-based imputation."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx].values
            X_val_fold = X.iloc[val_idx].values
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Scale and impute with autoencoder
            X_train_imputed = self._impute_with_autoencoder(
                X_train_fold, model, scaler, fit_scaler=True
            )
            X_val_imputed = self._impute_with_autoencoder(
                X_val_fold, model, scaler, fit_scaler=False
            )
            
            # Train classifier
            clf = xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False)
            clf.fit(X_train_imputed, y_train_fold)
            
            # Evaluate
            score = clf.score(X_val_imputed, y_val_fold)
            scores.append(score)
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"Denoising Autoencoder: {mean_score:.6f} (+/- {std_score:.6f})")
        
        return {'mean': mean_score, 'std': std_score, 'scores': scores}
        
    def _impute_with_autoencoder(self, X, model, scaler, fit_scaler=False):
        """Impute missing values using trained autoencoder."""
        # Initial imputation with median
        X_initial = np.nan_to_num(X, nan=np.nanmedian(X[~np.isnan(X).any(axis=1)], axis=0))
        
        # Scale
        if fit_scaler:
            X_scaled = scaler.fit_transform(X_initial)
        else:
            X_scaled = scaler.transform(X_initial)
        
        # Use autoencoder
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            X_reconstructed = model(X_tensor).cpu().numpy()
        
        # Inverse transform
        X_imputed = scaler.inverse_transform(X_reconstructed)
        
        # Keep original non-missing values
        mask = ~np.isnan(X)
        X_imputed[mask] = X[mask]
        
        return X_imputed
        
    def method_3_vae(self):
        """Variational Autoencoder for probabilistic imputation."""
        print("\n" + "="*60)
        print("METHOD 3: VARIATIONAL AUTOENCODER (VAE)")
        print("="*60)
        
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim=8):
                super().__init__()
                
                # Encoder
                self.fc1 = nn.Linear(input_dim, 32)
                self.fc2_mean = nn.Linear(32, latent_dim)
                self.fc2_logvar = nn.Linear(32, latent_dim)
                
                # Decoder
                self.fc3 = nn.Linear(latent_dim, 32)
                self.fc4 = nn.Linear(32, input_dim)
                
            def encode(self, x):
                h = torch.relu(self.fc1(x))
                return self.fc2_mean(h), self.fc2_logvar(h)
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                h = torch.relu(self.fc3(z))
                return self.fc4(h)
            
            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar
        
        # Prepare data
        X_train = self.train_df[self.feature_cols].values
        scaler = StandardScaler()
        
        # Create complete data for training
        X_complete = np.nan_to_num(X_train, nan=np.nanmedian(X_train, axis=0))
        X_scaled = scaler.fit_transform(X_complete)
        
        # Train VAE
        model = VAE(len(self.feature_cols)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # VAE loss function
        def vae_loss(recon_x, x, mu, logvar):
            MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return MSE + KLD
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Training
        print("Training VAE...")
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(X_tensor)
            loss = vae_loss(recon_batch, X_tensor, mu, logvar)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Evaluate
        model.eval()
        scores = self._evaluate_vae_imputation(
            self.train_df[self.feature_cols], self.y_train, model, scaler
        )
        
        self.results['vae'] = scores
        
    def _evaluate_vae_imputation(self, X, y, model, scaler):
        """Evaluate VAE-based imputation with multiple samples."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx].values
            X_val_fold = X.iloc[val_idx].values
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Impute with VAE (multiple samples)
            X_train_imputed = self._impute_with_vae(
                X_train_fold, model, scaler, n_samples=5, fit_scaler=True
            )
            X_val_imputed = self._impute_with_vae(
                X_val_fold, model, scaler, n_samples=5, fit_scaler=False
            )
            
            # Train classifier
            clf = xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False)
            clf.fit(X_train_imputed, y_train_fold)
            
            # Evaluate
            score = clf.score(X_val_imputed, y_val_fold)
            scores.append(score)
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"VAE: {mean_score:.6f} (+/- {std_score:.6f})")
        
        return {'mean': mean_score, 'std': std_score, 'scores': scores}
        
    def _impute_with_vae(self, X, model, scaler, n_samples=5, fit_scaler=False):
        """Impute using VAE with multiple samples."""
        # Initial imputation
        X_initial = np.nan_to_num(X, nan=np.nanmedian(X[~np.isnan(X).any(axis=1)], axis=0))
        
        # Scale
        if fit_scaler:
            X_scaled = scaler.fit_transform(X_initial)
        else:
            X_scaled = scaler.transform(X_initial)
        
        # Generate multiple imputations
        imputations = []
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            for _ in range(n_samples):
                recon, _, _ = model(X_tensor)
                imputations.append(recon.cpu().numpy())
        
        # Average imputations
        X_imputed_scaled = np.mean(imputations, axis=0)
        X_imputed = scaler.inverse_transform(X_imputed_scaled)
        
        # Keep original non-missing values
        mask = ~np.isnan(X)
        X_imputed[mask] = X[mask]
        
        return X_imputed
        
    def method_4_gain(self):
        """GAIN - Generative Adversarial Imputation Networks."""
        print("\n" + "="*60)
        print("METHOD 4: GAIN (Generative Adversarial Imputation Networks)")
        print("="*60)
        
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
        scaler = StandardScaler()
        
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
        print("Training GAIN...")
        n_epochs = 100
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
                
                # Create hint vector
                hint_rate = 0.9
                H_batch = M_batch * np.random.binomial(1, hint_rate, M_batch.shape)
                H_batch = torch.FloatTensor(H_batch).to(self.device)
                
                # Train Discriminator
                D_optimizer.zero_grad()
                
                # Generate imputed data
                G_sample = G(X_batch, M_batch)
                X_imputed = X_batch * M_batch + G_sample * (1 - M_batch)
                
                # Discriminator predictions
                D_pred = D(X_imputed, H_batch)
                
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
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}, D_loss: {np.mean(D_losses):.4f}, G_loss: {np.mean(G_losses):.4f}")
        
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
        print(f"GAIN: {mean_score:.6f} (+/- {std_score:.6f})")
        
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
        print("\n" + "="*60)
        print("METHOD 5: SYNTHETIC MNAR VALIDATION")
        print("="*60)
        
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
        
        # Test all methods on synthetic data
        print("\nEvaluating imputation quality on synthetic MNAR data...")
        
        # Simple imputation baseline
        from sklearn.impute import SimpleImputer
        simple_imputer = SimpleImputer(strategy='median')
        X_simple = simple_imputer.fit_transform(X_synthetic_mnar)
        
        # Calculate RMSE for imputed values only
        mask = X_synthetic_mnar.isnull()
        rmse_simple = np.sqrt(np.mean((X_complete[mask] - X_simple[mask])**2))
        print(f"Simple Imputation RMSE: {rmse_simple:.4f}")
        
        self.results['synthetic_validation'] = {
            'simple_rmse': rmse_simple,
            'missing_pattern': 'personality_based_mnar'
        }
        
    def compare_all_methods(self):
        """Compare all imputation methods."""
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        
        # Baseline with simple imputation
        from sklearn.impute import SimpleImputer
        simple_imputer = SimpleImputer(strategy='median')
        baseline_scores = self._evaluate_imputation_method(
            self.train_df[self.feature_cols], self.y_train, simple_imputer, "Baseline (Median)"
        )
        self.results['baseline'] = baseline_scores
        
        # Summary table
        print("\nSummary of Results:")
        print("-" * 60)
        print(f"{'Method':<30} {'CV Accuracy':<12} {'Std Dev':<10}")
        print("-" * 60)
        
        # Baseline
        print(f"{'Baseline (Median)':<30} {baseline_scores['mean']:<12.6f} {baseline_scores['std']:<10.6f}")
        
        # Other methods
        if 'missforest' in self.results:
            print(f"{'MissForest Standard':<30} {self.results['missforest']['standard']['mean']:<12.6f} "
                  f"{self.results['missforest']['standard']['std']:<10.6f}")
            print(f"{'MissForest Personality-Aware':<30} {self.results['missforest']['personality_aware']['mean']:<12.6f} "
                  f"{self.results['missforest']['personality_aware']['std']:<10.6f}")
        
        if 'denoising_autoencoder' in self.results:
            print(f"{'Denoising Autoencoder':<30} {self.results['denoising_autoencoder']['mean']:<12.6f} "
                  f"{self.results['denoising_autoencoder']['std']:<10.6f}")
        
        if 'vae' in self.results:
            print(f"{'VAE':<30} {self.results['vae']['mean']:<12.6f} "
                  f"{self.results['vae']['std']:<10.6f}")
        
        if 'gain' in self.results:
            print(f"{'GAIN':<30} {self.results['gain']['mean']:<12.6f} "
                  f"{self.results['gain']['std']:<10.6f}")
        
    def save_results(self):
        """Save all results."""
        os.makedirs('output', exist_ok=True)
        
        with open('output/advanced_imputation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\nResults saved to: output/advanced_imputation_results.json")


def main():
    """Main execution function."""
    tester = AdvancedImputationTester()
    
    # Load data
    tester.load_and_preprocess_data()
    
    # Test all methods
    tester.method_1_missforest_style()
    tester.method_2_denoising_autoencoder()
    tester.method_3_vae()
    tester.method_4_gain()
    tester.method_5_synthetic_validation()
    
    # Compare all methods
    tester.compare_all_methods()
    
    # Save results
    tester.save_results()
    
    print("\n" + "="*60)
    print("ADVANCED IMPUTATION TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()