#!/usr/bin/env python3
"""
Quilt Framework Implementation on FULL training set
Analyze gradients on entire train dataset to find all potential mislabeled samples
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"
OUTPUT_DIR.mkdir(exist_ok=True)

class SimpleNN(nn.Module):
    """Simple neural network for gradient analysis"""
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)

def prepare_data():
    """Load and prepare data"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Convert labels
    train_df['label'] = (train_df['Personality'] == 'Extrovert').astype(int)
    
    # Feature columns
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency']
    
    # Convert binary features
    for col in ['Stage_fear', 'Drained_after_socializing']:
        train_df[col + '_binary'] = (train_df[col] == 'Yes').astype(int)
        test_df[col + '_binary'] = (test_df[col] == 'Yes').astype(int)
        
    # Use all features
    all_features = feature_cols + ['Stage_fear_binary', 'Drained_after_socializing_binary']
    
    # Handle missing values
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[all_features])
    X_test = scaler.transform(test_df[all_features])
    y_train = train_df['label'].values
    
    print(f"Full train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    return X_train, y_train, X_test, train_df, test_df, all_features

def compute_gradient_scores_full(model, X, y, batch_size=64):
    """Compute gradient norms for each sample in the full dataset"""
    model.eval()
    gradient_norms = []
    
    # Process in batches
    n_samples = len(X)
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_X = torch.FloatTensor(X[i:batch_end])
        batch_y = torch.FloatTensor(y[i:batch_end])
        
        batch_X.requires_grad = True
        
        # Forward pass
        outputs = model(batch_X)
        criterion = nn.BCELoss()
        loss = criterion(outputs.squeeze(), batch_y)
        
        # Compute gradients w.r.t. inputs
        loss.backward()
        
        # Calculate gradient norm for each sample
        grad_norms = torch.norm(batch_X.grad, dim=1).detach().cpu().numpy()
        gradient_norms.extend(grad_norms)
        
        # Clear gradients
        model.zero_grad()
        batch_X.grad = None
    
    return np.array(gradient_norms)

def train_model_and_analyze_gradients(X_train, y_train, X_test):
    """Train model on full dataset and analyze gradients"""
    print("\n" + "="*60)
    print("TRAINING ON FULL DATASET")
    print("="*60)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    # Create data loader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = SimpleNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("Training model...")
    model.train()
    for epoch in range(20):  # More epochs for full dataset
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/20, Loss: {total_loss/len(train_loader):.4f}")
    
    # Compute gradients on full training set
    print("\nComputing gradients on full training set...")
    train_gradients = compute_gradient_scores_full(model, X_train, y_train)
    
    # Compute gradients on test set with pseudo-labels
    print("Computing gradients on test set...")
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.FloatTensor(X_test)).squeeze().numpy()
    test_pseudo_labels = (test_preds > 0.5).astype(float)
    test_gradients = compute_gradient_scores_full(model, X_test, test_pseudo_labels)
    
    return train_gradients, test_gradients, model

def analyze_train_gradients(train_gradients, train_df):
    """Analyze gradient patterns in training data"""
    print("\n" + "="*60)
    print("ANALYZING TRAINING SET GRADIENTS")
    print("="*60)
    
    # Add gradients to train_df
    train_df['gradient_norm'] = train_gradients
    
    # Statistics
    print(f"\nGradient statistics:")
    print(f"Mean: {train_gradients.mean():.6f}")
    print(f"Std: {train_gradients.std():.6f}")
    print(f"Min: {train_gradients.min():.6f}")
    print(f"Max: {train_gradients.max():.6f}")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(train_gradients, p)
        print(f"{p}th percentile: {val:.6f}")
    
    # Find highest gradient samples
    print("\n" + "="*60)
    print("TOP 20 HIGHEST GRADIENT TRAINING SAMPLES")
    print("="*60)
    
    top_gradient_train = train_df.nlargest(20, 'gradient_norm')
    
    for _, row in top_gradient_train.iterrows():
        print(f"\nID: {int(row['id'])}")
        print(f"  Gradient norm: {row['gradient_norm']:.6f}")
        print(f"  Label: {row['Personality']}")
        print(f"  Friends: {row['Friends_circle_size']}")
        print(f"  Social: {row['Social_event_attendance']}")
        print(f"  Alone: {row['Time_spent_Alone']}")
        print(f"  Stage fear: {row['Stage_fear']}")
        print(f"  Drained: {row['Drained_after_socializing']}")
    
    # Check if our known hard cases have high gradients
    print("\n" + "="*60)
    print("CHECKING KNOWN HARD CASES")
    print("="*60)
    
    # Load removed hard cases if available
    try:
        removed_df = pd.read_csv(OUTPUT_DIR / 'removed_hard_cases_info.csv')
        hard_case_ids = removed_df['id'].values
        
        for hc_id in hard_case_ids[:10]:  # Check first 10
            if hc_id in train_df['id'].values:
                row = train_df[train_df['id'] == hc_id].iloc[0]
                percentile = (train_df['gradient_norm'] < row['gradient_norm']).mean() * 100
                print(f"\nHard case ID {hc_id}:")
                print(f"  Gradient: {row['gradient_norm']:.6f}")
                print(f"  Percentile: {percentile:.1f}%")
    except:
        print("Could not load hard cases info")
    
    # Save high gradient samples
    high_gradient_threshold = np.percentile(train_gradients, 95)
    high_gradient_samples = train_df[train_df['gradient_norm'] > high_gradient_threshold]
    high_gradient_samples.to_csv(OUTPUT_DIR / 'train_high_gradient_samples.csv', index=False)
    print(f"\nSaved {len(high_gradient_samples)} high gradient training samples")
    
    return train_df

def visualize_gradients(train_gradients, test_gradients, train_df):
    """Create visualizations of gradient distributions"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Histogram of train gradients
    axes[0, 0].hist(train_gradients, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.percentile(train_gradients, 95), color='red', linestyle='--', label='95th percentile')
    axes[0, 0].set_xlabel('Gradient Norm')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Training Set Gradient Distribution')
    axes[0, 0].legend()
    
    # 2. Train vs Test gradient distributions
    axes[0, 1].hist(train_gradients, bins=50, alpha=0.5, label='Train', density=True)
    axes[0, 1].hist(test_gradients, bins=50, alpha=0.5, label='Test', density=True)
    axes[0, 1].set_xlabel('Gradient Norm')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Gradient Distribution: Train vs Test')
    axes[0, 1].legend()
    
    # 3. Gradient by personality type
    introverts = train_df[train_df['Personality'] == 'Introvert']['gradient_norm'].values
    extroverts = train_df[train_df['Personality'] == 'Extrovert']['gradient_norm'].values
    
    axes[1, 0].hist(introverts, bins=50, alpha=0.5, label='Introvert', density=True)
    axes[1, 0].hist(extroverts, bins=50, alpha=0.5, label='Extrovert', density=True)
    axes[1, 0].set_xlabel('Gradient Norm')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Gradient Distribution by Personality Type')
    axes[1, 0].legend()
    
    # 4. Scatter plot: gradient vs features
    axes[1, 1].scatter(train_df['Friends_circle_size'], train_df['gradient_norm'], 
                       alpha=0.3, s=10)
    axes[1, 1].set_xlabel('Friends Circle Size')
    axes[1, 1].set_ylabel('Gradient Norm')
    axes[1, 1].set_title('Gradient vs Friends Circle Size')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'quilt_full_train_gradients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved")

def main():
    # Load and prepare data
    X_train, y_train, X_test, train_df, test_df, feature_names = prepare_data()
    
    # Train model and compute gradients
    train_gradients, test_gradients, model = train_model_and_analyze_gradients(
        X_train, y_train, X_test
    )
    
    # Analyze training gradients
    train_df_with_gradients = analyze_train_gradients(train_gradients, train_df)
    
    # Create visualizations
    visualize_gradients(train_gradients, test_gradients, train_df_with_gradients)
    
    # Save test gradients
    test_df['gradient_norm'] = test_gradients
    test_df[['id', 'gradient_norm']].to_csv(
        OUTPUT_DIR / 'test_gradients_full_train.csv', 
        index=False
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("1. Trained model on full training set")
    print("2. Computed gradients for all 18,524 training samples")
    print("3. Identified high gradient samples (potential errors)")
    print("4. Saved results for further analysis")
    
    # Final recommendations
    high_grad_count = (train_gradients > np.percentile(train_gradients, 95)).sum()
    print(f"\nFound {high_grad_count} samples in top 5% gradient")
    print("These are prime candidates for being mislabeled!")

if __name__ == "__main__":
    main()