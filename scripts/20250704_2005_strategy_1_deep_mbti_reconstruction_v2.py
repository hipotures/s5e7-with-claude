#!/usr/bin/env python3
"""
STRATEGY 1: DEEP MBTI TYPE RECONSTRUCTION WITH NEURAL EMBEDDINGS (V2)
=====================================================================
Fixed version - handles categorical columns properly

HYPOTHESIS: 
The dataset originally contained 16 MBTI personality types that were collapsed 
into 2 classes (Introvert/Extrovert). The 2.43% "ambiguous" cases are likely 
specific MBTI types that don't map cleanly to I/E (e.g., ISFJ, ESFJ, INFJ, ENFJ).

APPROACH:
1. Use deep learning to learn a 16-dimensional embedding space
2. Apply unsupervised clustering to find natural groupings
3. Map clusters back to I/E with learned, cluster-specific boundaries
4. Special handling for ambiguous MBTI types

KEY INNOVATION:
Instead of treating all ambiverts as one group, we recognize they might be 
4-6 distinct MBTI types, each requiring different handling.

Author: Claude (Strategy Implementation V2)
Date: 2025-01-04
"""

# PURPOSE: Fixed version of MBTI reconstruction that properly handles categorical columns
# HYPOTHESIS: Same as V1 - ambiguous cases are hidden MBTI types needing special handling
# EXPECTED: Proper preprocessing will allow the deep learning approach to work correctly
# RESULT: Fixed categorical encoding and preprocessing for successful MBTI embedding learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)

def preprocess_data(df):
    """
    Preprocess data - handle categorical columns and prepare features.
    """
    df_processed = df.copy()
    
    # Handle categorical columns
    categorical_mappings = {
        'Stage_fear': {'No': 0, 'Yes': 1},
        'Drained_after_socializing': {'No': 0, 'Yes': 1}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)
    
    # Handle Target column if exists
    if 'Personality' in df_processed.columns:
        df_processed['Target'] = (df_processed['Personality'] == 'Extrovert').astype(int)
    
    return df_processed

class MBTIEmbeddingNetwork(nn.Module):
    """
    Deep neural network that learns MBTI-like embeddings from behavioral features.
    The network projects 7 behavioral features into a 16-dimensional space,
    representing potential MBTI types.
    """
    def __init__(self, input_dim=7, embedding_dim=16, hidden_dims=[64, 128, 64]):
        super(MBTIEmbeddingNetwork, self).__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)
        
        # Decoder for reconstruction (autoencoder approach)
        decoder_layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Classification head (for supervised fine-tuning)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_embedding=False):
        embedding = self.encoder(x)
        
        if return_embedding:
            return embedding
        
        reconstruction = self.decoder(embedding)
        classification = self.classifier(embedding)
        
        return reconstruction, classification, embedding

def identify_ambiguous_cases(df):
    """
    Identify potentially ambiguous cases based on known patterns.
    These are likely the hidden MBTI types that don't map cleanly to I/E.
    """
    # Known marker values for ambiverts
    markers = {
        'Social_event_attendance': 5.265106088560886,
        'Going_outside': 4.044319380935631,
        'Post_frequency': 4.982097334878332,
        'Time_spent_Alone': 3.1377639321564557
    }
    
    # Check for marker values
    has_markers = pd.Series([False] * len(df))
    for col, val in markers.items():
        if col in df.columns:
            has_markers |= (df[col] == val)
    
    # Additional patterns for ambiguous cases
    low_alone_time = df['Time_spent_Alone'] < 2.5
    moderate_social = (df['Social_event_attendance'] >= 3) & (df['Social_event_attendance'] <= 4)
    medium_friends = (df['Friends_circle_size'] >= 6) & (df['Friends_circle_size'] <= 7)
    
    # Combine patterns
    ambiguous = has_markers | (low_alone_time & moderate_social & medium_friends)
    
    return ambiguous

def create_mbti_features(df):
    """
    Create features that might help identify MBTI dimensions.
    Based on Jung's cognitive functions and MBTI theory.
    """
    features = df.copy()
    
    # E/I dimension (already in target, but let's enhance)
    features['social_energy'] = (
        features['Social_event_attendance'] + 
        features['Going_outside'] - 
        features['Time_spent_Alone']
    )
    
    # S/N dimension (Sensing vs Intuition) - inferred from posting behavior
    features['abstract_thinking'] = (
        features['Post_frequency'] / (features['Social_event_attendance'] + 1)
    )
    
    # T/F dimension (Thinking vs Feeling) - inferred from social patterns
    features['emotional_expression'] = (
        features['Friends_circle_size'] * features['Post_frequency'] / 10
    )
    
    # J/P dimension (Judging vs Perceiving) - inferred from consistency
    features['behavioral_consistency'] = (
        abs(features['Going_outside'] - features['Social_event_attendance']) +
        abs(features['Post_frequency'] - features['Social_event_attendance'])
    )
    
    # Interaction features that might reveal MBTI types
    features['introvert_pattern'] = (
        features['Time_spent_Alone'] * features['Drained_after_socializing']
    )
    
    features['extrovert_pattern'] = (
        features['Social_event_attendance'] * (1 - features['Stage_fear'])
    )
    
    features['ambivert_score'] = abs(
        features['introvert_pattern'] - features['extrovert_pattern']
    )
    
    return features

def train_mbti_embedding_model(X_train, y_train, X_val, y_val, epochs=100):
    """
    Train the MBTI embedding network using a multi-task approach:
    1. Autoencoder loss for learning meaningful representations
    2. Classification loss for I/E prediction
    3. Contrastive loss to separate different personality types
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Prepare data
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = MBTIEmbeddingNetwork(input_dim=X_train.shape[1]).to(device)
    
    # Loss functions
    reconstruction_loss = nn.MSELoss()
    classification_loss = nn.BCELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, classification, embedding = model(batch_x)
            
            # Calculate losses
            rec_loss = reconstruction_loss(reconstruction, batch_x)
            cls_loss = classification_loss(classification.squeeze(), batch_y)
            
            # Combined loss (weighted)
            total_loss = 0.5 * rec_loss + cls_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_losses.append(total_loss.item())
        
        # Validation phase
        model.eval()
        val_preds = []
        val_true = []
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                reconstruction, classification, embedding = model(batch_x)
                
                rec_loss = reconstruction_loss(reconstruction, batch_x)
                cls_loss = classification_loss(classification.squeeze(), batch_y)
                total_loss = 0.5 * rec_loss + cls_loss
                
                val_losses.append(total_loss.item())
                val_preds.extend(classification.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int).reshape(-1)
        val_acc = accuracy_score(val_true, val_preds_binary)
        
        # Learning rate scheduling
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {np.mean(train_losses):.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def cluster_mbti_embeddings(model, X_data, n_clusters=16):
    """
    Cluster the learned embeddings to find MBTI-like groupings.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Get embeddings
    with torch.no_grad():
        data_tensor = torch.FloatTensor(X_data).to(device)
        embeddings = model(data_tensor, return_embedding=True).cpu().numpy()
    
    # Cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    return clusters, embeddings, kmeans

def create_cluster_specific_predictors(X_train, y_train, clusters_train, n_clusters=16):
    """
    Train a separate predictor for each MBTI cluster.
    Some clusters (ambiguous MBTI types) might need different decision boundaries.
    """
    from xgboost import XGBClassifier
    
    cluster_models = {}
    cluster_thresholds = {}
    
    for cluster_id in range(n_clusters):
        # Get data for this cluster
        mask = clusters_train == cluster_id
        if mask.sum() < 10:  # Skip small clusters
            continue
        
        X_cluster = X_train[mask]
        y_cluster = y_train[mask]
        
        # Check if this is an ambiguous cluster
        extrovert_ratio = y_cluster.mean()
        is_ambiguous = 0.4 < extrovert_ratio < 0.6
        
        # Train cluster-specific model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        if len(np.unique(y_cluster)) > 1:  # Only train if both classes present
            model.fit(X_cluster, y_cluster)
            cluster_models[cluster_id] = model
            
            # Set threshold based on cluster characteristics
            if is_ambiguous:
                # For ambiguous clusters, use the known 96.2% extrovert rule
                cluster_thresholds[cluster_id] = 0.038  # Will classify most as extrovert
            else:
                cluster_thresholds[cluster_id] = 0.5
        else:
            # If only one class, store the prediction
            cluster_models[cluster_id] = int(y_cluster[0])
            cluster_thresholds[cluster_id] = 0.5
    
    return cluster_models, cluster_thresholds

def predict_with_mbti_reconstruction(model, kmeans, cluster_models, cluster_thresholds, 
                                     X_test, feature_names):
    """
    Make predictions using the MBTI reconstruction approach.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Get embeddings and cluster assignments
    with torch.no_grad():
        data_tensor = torch.FloatTensor(X_test).to(device)
        _, base_predictions, embeddings = model(data_tensor)
        base_predictions = base_predictions.cpu().numpy().reshape(-1)
        embeddings = embeddings.cpu().numpy()
    
    # Assign to clusters
    test_clusters = kmeans.predict(embeddings)
    
    # Make cluster-specific predictions
    final_predictions = np.zeros(len(X_test))
    
    for i in range(len(X_test)):
        cluster_id = test_clusters[i]
        
        if cluster_id in cluster_models:
            if isinstance(cluster_models[cluster_id], int):
                # Single class cluster
                final_predictions[i] = cluster_models[cluster_id]
            else:
                # Use cluster-specific model
                cluster_model = cluster_models[cluster_id]
                threshold = cluster_thresholds[cluster_id]
                
                prob = cluster_model.predict_proba(X_test[i:i+1])[0, 1]
                final_predictions[i] = int(prob > threshold)
        else:
            # Fallback to base prediction
            final_predictions[i] = int(base_predictions[i] > 0.5)
    
    return final_predictions, base_predictions, test_clusters

def main():
    """
    Execute Strategy 1: Deep MBTI Type Reconstruction
    """
    print("="*80)
    print("STRATEGY 1: DEEP MBTI TYPE RECONSTRUCTION WITH NEURAL EMBEDDINGS (V2)")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("../../train.csv")
    test_df = pd.read_csv("../../test.csv")
    
    # Preprocess data
    print("Preprocessing data...")
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    # Save original IDs
    test_ids = test_df['id']
    
    # Prepare features
    print("\nPreparing features...")
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Drained_after_socializing',
                    'Stage_fear', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    # Create enhanced features
    train_enhanced = create_mbti_features(train_df[feature_cols])
    test_enhanced = create_mbti_features(test_df[feature_cols])
    
    # Identify ambiguous cases in training data
    print("\nIdentifying ambiguous cases...")
    ambiguous_mask = identify_ambiguous_cases(train_df)
    print(f"Found {ambiguous_mask.sum()} potentially ambiguous cases ({ambiguous_mask.mean()*100:.2f}%)")
    
    # Prepare data for training
    X = train_enhanced.values
    y = train_df['Target'].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(test_enhanced.values)
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train MBTI embedding model
    print("\nTraining MBTI embedding network...")
    mbti_model = train_mbti_embedding_model(X_train, y_train, X_val, y_val, epochs=50)
    
    # Cluster embeddings to find MBTI types
    print("\nClustering embeddings to identify MBTI types...")
    clusters_train, embeddings_train, kmeans = cluster_mbti_embeddings(
        mbti_model, X_train, n_clusters=16
    )
    
    # Analyze clusters
    print("\nAnalyzing discovered MBTI clusters:")
    for cluster_id in range(16):
        mask = clusters_train == cluster_id
        if mask.sum() > 0:
            extrovert_ratio = y_train[mask].mean()
            print(f"  Cluster {cluster_id}: {mask.sum()} samples, "
                  f"{extrovert_ratio*100:.1f}% Extrovert")
    
    # Train cluster-specific models
    print("\nTraining cluster-specific predictors...")
    cluster_models, cluster_thresholds = create_cluster_specific_predictors(
        X_train, y_train, clusters_train
    )
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    predictions, base_predictions, test_clusters = predict_with_mbti_reconstruction(
        mbti_model, kmeans, cluster_models, cluster_thresholds, 
        X_test_scaled, train_enhanced.columns
    )
    
    # Analyze prediction distribution
    print(f"\nPrediction distribution:")
    print(f"  Introverts: {(predictions == 0).sum()} ({(predictions == 0).mean()*100:.2f}%)")
    print(f"  Extroverts: {(predictions == 1).sum()} ({(predictions == 1).mean()*100:.2f}%)")
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_ids,
        'Target': predictions.astype(int)
    })
    
    submission_path = 'submission_strategy1_mbti_reconstruction_v2.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    
    # Cross-validation to estimate performance
    print("\n" + "="*60)
    print("PERFORMANCE ESTIMATION (5-Fold CV on Training Data)")
    print("="*60)
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Train model for this fold
        fold_model = train_mbti_embedding_model(
            X_fold_train, y_fold_train, X_fold_val, y_fold_val, epochs=30
        )
        
        # Cluster and predict
        fold_clusters, _, fold_kmeans = cluster_mbti_embeddings(fold_model, X_fold_train)
        fold_cluster_models, fold_thresholds = create_cluster_specific_predictors(
            X_fold_train, y_fold_train, fold_clusters
        )
        
        fold_predictions, _, _ = predict_with_mbti_reconstruction(
            fold_model, fold_kmeans, fold_cluster_models, fold_thresholds,
            X_fold_val, train_enhanced.columns
        )
        
        fold_score = accuracy_score(y_fold_val, fold_predictions)
        cv_scores.append(fold_score)
        print(f"  Fold {fold+1}: {fold_score:.6f}")
    
    print(f"\nMean CV Score: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores)*2:.6f})")
    
    # Detailed analysis of ambiguous cases
    print("\n" + "="*60)
    print("AMBIGUOUS CASE ANALYSIS")
    print("="*60)
    
    # Check how different clusters handle ambiguous patterns
    ambiguous_val_mask = identify_ambiguous_cases(
        pd.DataFrame(X_val, columns=train_enhanced.columns)
    )
    
    if ambiguous_val_mask.sum() > 0:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        val_clusters = kmeans.predict(mbti_model(
            torch.FloatTensor(X_val).to(device), return_embedding=True
        ).cpu().numpy())
        
        print(f"\nFound {ambiguous_val_mask.sum()} ambiguous cases in validation")
        print("Distribution across clusters:")
        
        for cluster_id in range(16):
            cluster_ambiguous = ambiguous_val_mask & (val_clusters == cluster_id)
            if cluster_ambiguous.sum() > 0:
                print(f"  Cluster {cluster_id}: {cluster_ambiguous.sum()} ambiguous cases")

if __name__ == "__main__":
    main()