#!/usr/bin/env python3
"""
Quilt Framework Implementation for S5E7
Based on "A Scalable Approach to Covariate and Concept Drift Management via Adaptive Data Segmentation" (2024)

The Quilt framework uses gradient-based disparity and gain scores to detect concept drift.
Key idea: Use gradients on training and validation sets to identify data segments with drift.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    return X_train, y_train, X_test, train_df, test_df, all_features

def compute_gradient_scores(model, data_loader, criterion):
    """Compute gradient norms for each sample"""
    model.eval()
    gradient_norms = []
    
    for batch_X, batch_y in data_loader:
        batch_X.requires_grad = True
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y.float())
        
        # Compute gradients w.r.t. inputs
        loss.backward()
        
        # Calculate gradient norm for each sample
        grad_norms = torch.norm(batch_X.grad, dim=1).detach().cpu().numpy()
        gradient_norms.extend(grad_norms)
        
        # Clear gradients
        model.zero_grad()
        batch_X.grad = None
    
    return np.array(gradient_norms)

def quilt_analysis(X_train, y_train, X_test):
    """Implement Quilt framework gradient-based analysis"""
    print("\n" + "="*60)
    print("QUILT FRAMEWORK ANALYSIS")
    print("="*60)
    
    # Split train into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Convert to PyTorch tensors
    X_tr_tensor = torch.FloatTensor(X_tr)
    y_tr_tensor = torch.FloatTensor(y_tr)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = SimpleNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model briefly
    print("\nTraining model for gradient analysis...")
    model.train()
    for epoch in range(10):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
    
    print("Computing gradient scores...")
    
    # Compute gradient scores
    train_gradients = compute_gradient_scores(model, train_loader, criterion)
    val_gradients = compute_gradient_scores(model, val_loader, criterion)
    
    # For test data, we need pseudo-labels from model predictions
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).squeeze().numpy()
    test_pseudo_labels = (test_preds > 0.5).astype(float)
    
    # Create test loader with pseudo-labels
    test_dataset = TensorDataset(X_test_tensor, torch.FloatTensor(test_pseudo_labels))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    test_gradients = compute_gradient_scores(model, test_loader, criterion)
    
    # Compute disparity scores (difference between train and val gradients)
    # We'll use percentiles to identify high-gradient samples
    train_high_grad = train_gradients > np.percentile(train_gradients, 90)
    val_high_grad = val_gradients > np.percentile(val_gradients, 90)
    
    print(f"\nHigh gradient samples in train: {train_high_grad.sum()} ({train_high_grad.mean()*100:.1f}%)")
    print(f"High gradient samples in val: {val_high_grad.sum()} ({val_high_grad.mean()*100:.1f}%)")
    
    # Analyze gradient distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(train_gradients, bins=50, alpha=0.7, label='Train', density=True)
    plt.hist(val_gradients, bins=50, alpha=0.7, label='Validation', density=True)
    plt.xlabel('Gradient Norm')
    plt.ylabel('Density')
    plt.title('Gradient Norm Distribution: Train vs Validation')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(train_gradients, bins=50, alpha=0.7, label='Train', density=True)
    plt.hist(test_gradients, bins=50, alpha=0.7, label='Test', density=True)
    plt.xlabel('Gradient Norm')
    plt.ylabel('Density')
    plt.title('Gradient Norm Distribution: Train vs Test')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # Scatter plot of gradient norms
    sample_size = min(1000, len(train_gradients))
    indices = np.random.choice(len(train_gradients), sample_size, replace=False)
    plt.scatter(range(sample_size), train_gradients[indices], alpha=0.5, label='Train samples')
    plt.axhline(y=np.percentile(train_gradients, 90), color='r', linestyle='--', label='90th percentile')
    plt.xlabel('Sample Index')
    plt.ylabel('Gradient Norm')
    plt.title('Sample-wise Gradient Norms')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'quilt_gradient_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return train_gradients, val_gradients, test_gradients, X_tr, y_tr, X_val, y_val

def identify_drift_segments(train_gradients, X_train, y_train, train_df):
    """Identify segments with potential drift based on gradient scores"""
    print("\n" + "="*60)
    print("IDENTIFYING DRIFT SEGMENTS")
    print("="*60)
    
    # Identify high-gradient samples (potential drift)
    threshold_percentiles = [80, 85, 90, 95]
    
    results = []
    for percentile in threshold_percentiles:
        threshold = np.percentile(train_gradients, percentile)
        high_grad_mask = train_gradients > threshold
        n_high_grad = high_grad_mask.sum()
        
        # Get accuracy on high vs low gradient samples
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Split data
        X_low = X_train[~high_grad_mask]
        y_low = y_train[~high_grad_mask]
        X_high = X_train[high_grad_mask]
        y_high = y_train[high_grad_mask]
        
        # Train on low gradient samples
        if len(X_low) > 100:  # Ensure enough samples
            rf.fit(X_low, y_low)
            
            # Test on both
            acc_low = accuracy_score(y_low, rf.predict(X_low))
            if len(X_high) > 0:
                acc_high = accuracy_score(y_high, rf.predict(X_high))
            else:
                acc_high = 0
            
            results.append({
                'percentile': percentile,
                'threshold': threshold,
                'n_high_grad': n_high_grad,
                'pct_high_grad': n_high_grad / len(train_gradients) * 100,
                'acc_low_grad': acc_low,
                'acc_high_grad': acc_high,
                'acc_diff': acc_low - acc_high
            })
            
            print(f"\nPercentile {percentile}:")
            print(f"  Samples above threshold: {n_high_grad} ({n_high_grad/len(train_gradients)*100:.1f}%)")
            print(f"  Accuracy on low gradient: {acc_low:.4f}")
            print(f"  Accuracy on high gradient: {acc_high:.4f}")
            print(f"  Difference: {acc_low - acc_high:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'quilt_drift_segments.csv', index=False)
    
    # Analyze characteristics of high gradient samples
    best_percentile = 90
    threshold = np.percentile(train_gradients, best_percentile)
    high_grad_mask = train_gradients > threshold
    
    print("\n" + "="*60)
    print(f"CHARACTERISTICS OF HIGH GRADIENT SAMPLES (>{best_percentile}th percentile)")
    print("="*60)
    
    # Add gradient info to train_df subset
    train_subset = train_df.iloc[:len(train_gradients)].copy()
    train_subset['gradient_norm'] = train_gradients
    train_subset['high_gradient'] = high_grad_mask
    
    # Compare characteristics
    for col in ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                'Going_outside', 'Post_frequency']:
        if col in train_subset.columns:
            mean_low = train_subset[~high_grad_mask][col].mean()
            mean_high = train_subset[high_grad_mask][col].mean()
            print(f"\n{col}:")
            print(f"  Low gradient: {mean_low:.2f}")
            print(f"  High gradient: {mean_high:.2f}")
            print(f"  Difference: {mean_high - mean_low:.2f}")
    
    # Check personality distribution
    if 'Personality' in train_subset.columns:
        print("\nPersonality distribution:")
        low_dist = train_subset[~high_grad_mask]['Personality'].value_counts(normalize=True)
        high_dist = train_subset[high_grad_mask]['Personality'].value_counts(normalize=True)
        print("Low gradient:", dict(low_dist))
        print("High gradient:", dict(high_dist))
    
    return train_subset, high_grad_mask

def compare_with_test_predictions(test_gradients, test_df):
    """Compare test gradient scores with our previous predictions"""
    print("\n" + "="*60)
    print("TEST SET GRADIENT ANALYSIS")
    print("="*60)
    
    # Load our previous uncertainty analysis if available
    try:
        uncertainty_df = pd.read_csv(OUTPUT_DIR / 'test_uncertainty_analysis.csv')
        
        # Merge with gradient info
        test_analysis = test_df.copy()
        test_analysis['gradient_norm'] = test_gradients
        test_analysis = test_analysis.merge(
            uncertainty_df[['id', 'mean_pred', 'std_pred']], 
            on='id', 
            how='left'
        )
        
        # Correlation analysis
        if 'std_pred' in test_analysis.columns:
            corr = test_analysis[['gradient_norm', 'std_pred']].corr().iloc[0, 1]
            print(f"\nCorrelation between gradient norm and prediction uncertainty: {corr:.3f}")
        
        # Find high gradient test samples
        high_grad_threshold = np.percentile(test_gradients, 90)
        high_grad_test = test_analysis[test_analysis['gradient_norm'] > high_grad_threshold]
        
        print(f"\nHigh gradient test samples: {len(high_grad_test)}")
        print("\nTop 10 highest gradient test samples:")
        top_grad = test_analysis.nlargest(10, 'gradient_norm')[['id', 'gradient_norm', 'mean_pred', 'std_pred']]
        print(top_grad)
        
        # Save for further analysis
        test_analysis.to_csv(OUTPUT_DIR / 'test_gradient_analysis.csv', index=False)
        
    except FileNotFoundError:
        print("Previous uncertainty analysis not found. Running basic gradient analysis.")
        test_analysis = test_df.copy()
        test_analysis['gradient_norm'] = test_gradients
        test_analysis.to_csv(OUTPUT_DIR / 'test_gradient_analysis.csv', index=False)

def main():
    # Load and prepare data
    X_train, y_train, X_test, train_df, test_df, feature_names = prepare_data()
    
    # Run Quilt framework analysis
    train_gradients, val_gradients, test_gradients, X_tr, y_tr, X_val, y_val = quilt_analysis(
        X_train, y_train, X_test
    )
    
    # Identify drift segments
    train_subset, high_grad_mask = identify_drift_segments(
        train_gradients[:len(X_tr)], X_tr, y_tr, train_df
    )
    
    # Compare with test set
    compare_with_test_predictions(test_gradients, test_df)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("1. Gradient-based analysis completed")
    print("2. High gradient samples identified (potential drift/errors)")
    print("3. Results saved to output directory")
    print("\nNext steps:")
    print("- Investigate high gradient samples for labeling errors")
    print("- Compare gradient patterns between train and test")
    print("- Consider removing high gradient samples from training")

if __name__ == "__main__":
    main()