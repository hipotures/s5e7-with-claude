#!/usr/bin/env python3
"""
Interactive cluster analysis of uncertain/boundary cases
Creates HTML visualization with clickable points
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import umap
import ydf
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def load_and_prepare_data():
    """Load data and identify uncertain cases"""
    
    print("="*60)
    print("INTERACTIVE CLUSTER ANALYSIS")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    
    print(f"\nData shape: {train_df.shape}")
    
    # Features for clustering
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Encode categorical features
    train_df['Stage_fear_encoded'] = (train_df['Stage_fear'] == 'Yes').astype(int)
    train_df['Drained_encoded'] = (train_df['Drained_after_socializing'] == 'Yes').astype(int)
    
    # Get model predictions to identify uncertain cases
    print("\nTraining model to identify uncertain cases...")
    
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        random_seed=42
    )
    
    # Use a subset for faster OOB predictions
    model = learner.train(train_df[feature_cols + ['Personality']])
    
    # Get predictions on full data (not OOB, but fast)
    predictions = model.predict(train_df[feature_cols])
    
    probabilities = []
    pred_classes = []
    
    for pred in predictions:
        prob_I = float(str(pred))
        probabilities.append(prob_I)
        pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
    
    train_df['probability'] = probabilities
    train_df['confidence'] = np.abs(np.array(probabilities) - 0.5) * 2
    train_df['predicted'] = pred_classes
    train_df['is_correct'] = train_df['predicted'] == train_df['Personality']
    
    # Identify uncertain cases
    train_df['uncertainty_type'] = 'Certain'
    train_df.loc[train_df['confidence'] < 0.1, 'uncertainty_type'] = 'Very Uncertain'
    train_df.loc[(train_df['confidence'] >= 0.1) & (train_df['confidence'] < 0.3), 'uncertainty_type'] = 'Uncertain'
    train_df.loc[~train_df['is_correct'] & (train_df['confidence'] > 0.8), 'uncertainty_type'] = 'High Conf Error'
    
    print(f"\nUncertainty distribution:")
    print(train_df['uncertainty_type'].value_counts())
    
    return train_df

def perform_clustering(train_df):
    """Perform multiple clustering methods"""
    
    print("\n" + "="*60)
    print("PERFORMING CLUSTERING")
    print("="*60)
    
    # Prepare features
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency',
                       'Stage_fear_encoded', 'Drained_encoded']
    
    # Handle missing values
    X = train_df[numeric_features].fillna(train_df[numeric_features].median())
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Store results
    clustering_results = {}
    
    # 1. K-Means
    print("\n1. K-Means clustering...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    train_df['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
    clustering_results['kmeans'] = kmeans
    
    # 2. DBSCAN
    print("2. DBSCAN clustering...")
    dbscan = DBSCAN(eps=1.5, min_samples=50)
    train_df['cluster_dbscan'] = dbscan.fit_predict(X_scaled)
    print(f"   DBSCAN found {len(set(train_df['cluster_dbscan'])) - 1} clusters + noise")
    
    # 3. Hierarchical
    print("3. Hierarchical clustering...")
    hierarchical = AgglomerativeClustering(n_clusters=5)
    train_df['cluster_hierarchical'] = hierarchical.fit_predict(X_scaled)
    
    # 4. Gaussian Mixture
    print("4. Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=5, random_state=42)
    train_df['cluster_gmm'] = gmm.fit_predict(X_scaled)
    train_df['gmm_probability'] = gmm.predict_proba(X_scaled).max(axis=1)
    
    # Analyze cluster characteristics
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS")
    print("="*60)
    
    for method in ['kmeans', 'dbscan', 'hierarchical', 'gmm']:
        print(f"\n{method.upper()} clusters:")
        cluster_col = f'cluster_{method}'
        
        for cluster_id in sorted(train_df[cluster_col].unique()):
            if cluster_id == -1:  # DBSCAN noise
                continue
                
            cluster_data = train_df[train_df[cluster_col] == cluster_id]
            uncertain_ratio = (cluster_data['uncertainty_type'] != 'Certain').mean()
            
            print(f"  Cluster {cluster_id}: {len(cluster_data)} samples, "
                  f"{uncertain_ratio*100:.1f}% uncertain/error")
    
    return train_df, X_scaled, clustering_results

def perform_dimensionality_reduction(X_scaled, train_df):
    """Perform multiple dimensionality reduction methods"""
    
    print("\n" + "="*60)
    print("DIMENSIONALITY REDUCTION")
    print("="*60)
    
    reductions = {}
    
    # 1. PCA
    print("\n1. PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    train_df['pca_1'] = X_pca[:, 0]
    train_df['pca_2'] = X_pca[:, 1]
    reductions['pca'] = pca
    print(f"   Explained variance: {pca.explained_variance_ratio_}")
    
    # 2. t-SNE
    print("\n2. t-SNE (this may take a while)...")
    # Use subset for t-SNE if data is large
    if len(train_df) > 5000:
        indices = np.random.choice(len(train_df), 5000, replace=False)
        X_subset = X_scaled[indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_subset)
        
        # Fill full dataframe
        train_df['tsne_1'] = np.nan
        train_df['tsne_2'] = np.nan
        train_df.loc[indices, 'tsne_1'] = X_tsne[:, 0]
        train_df.loc[indices, 'tsne_2'] = X_tsne[:, 1]
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        train_df['tsne_1'] = X_tsne[:, 0]
        train_df['tsne_2'] = X_tsne[:, 1]
    
    # 3. UMAP
    print("\n3. UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    train_df['umap_1'] = X_umap[:, 0]
    train_df['umap_2'] = X_umap[:, 1]
    reductions['umap'] = reducer
    
    return train_df, reductions

def create_interactive_visualizations(train_df):
    """Create interactive Plotly visualizations"""
    
    print("\n" + "="*60)
    print("CREATING INTERACTIVE VISUALIZATIONS")
    print("="*60)
    
    # Create subplots for different views
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=(
            'K-Means (PCA)', 'K-Means (t-SNE)', 'K-Means (UMAP)', 'Uncertainty Distribution',
            'DBSCAN (PCA)', 'DBSCAN (t-SNE)', 'DBSCAN (UMAP)', 'Confidence vs Error',
            'GMM (PCA)', 'GMM (t-SNE)', 'GMM (UMAP)', 'Feature Importance'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Color schemes
    uncertainty_colors = {
        'Certain': 'lightblue',
        'Uncertain': 'orange',
        'Very Uncertain': 'red',
        'High Conf Error': 'darkred'
    }
    
    # Prepare hover data
    hover_cols = ['id', 'Personality', 'predicted', 'confidence', 'probability',
                  'Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']
    
    # Row 1: K-Means clustering
    for col, method in enumerate(['pca', 'tsne', 'umap'], 1):
        if f'{method}_1' in train_df.columns:
            subset = train_df.dropna(subset=[f'{method}_1', f'{method}_2']) if method == 'tsne' else train_df
            
            for uncertainty_type in uncertainty_colors.keys():
                mask = subset['uncertainty_type'] == uncertainty_type
                if mask.sum() > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=subset.loc[mask, f'{method}_1'],
                            y=subset.loc[mask, f'{method}_2'],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=subset.loc[mask, 'cluster_kmeans'],
                                colorscale='Viridis',
                                line=dict(width=1, color=uncertainty_colors[uncertainty_type])
                            ),
                            name=uncertainty_type,
                            legendgroup=uncertainty_type,
                            showlegend=(col == 1),
                            text=subset.loc[mask].apply(
                                lambda r: f"ID: {r['id']}<br>" +
                                         f"Actual: {r['Personality']}<br>" +
                                         f"Predicted: {r['predicted']}<br>" +
                                         f"Confidence: {r['confidence']:.3f}<br>" +
                                         f"P(Introvert): {r['probability']:.3f}<br>" +
                                         f"Alone: {r['Time_spent_Alone']}<br>" +
                                         f"Social: {r['Social_event_attendance']}<br>" +
                                         f"Friends: {r['Friends_circle_size']}<br>" +
                                         f"K-Means: {r['cluster_kmeans']}",
                                axis=1
                            ),
                            hoverinfo='text'
                        ),
                        row=1, col=col
                    )
    
    # Row 2: DBSCAN clustering
    for col, method in enumerate(['pca', 'tsne', 'umap'], 1):
        if f'{method}_1' in train_df.columns:
            subset = train_df.dropna(subset=[f'{method}_1', f'{method}_2']) if method == 'tsne' else train_df
            
            # Plot DBSCAN clusters
            unique_clusters = sorted(subset['cluster_dbscan'].unique())
            colors = px.colors.qualitative.Set3
            
            for i, cluster in enumerate(unique_clusters):
                cluster_mask = subset['cluster_dbscan'] == cluster
                cluster_name = f'Cluster {cluster}' if cluster != -1 else 'Noise'
                
                fig.add_trace(
                    go.Scatter(
                        x=subset.loc[cluster_mask, f'{method}_1'],
                        y=subset.loc[cluster_mask, f'{method}_2'],
                        mode='markers',
                        marker=dict(
                            size=6 if cluster != -1 else 4,
                            color=colors[i % len(colors)] if cluster != -1 else 'gray',
                            opacity=0.8 if cluster != -1 else 0.3
                        ),
                        name=cluster_name,
                        legendgroup=f'dbscan_{cluster}',
                        showlegend=(col == 1),
                        text=subset.loc[cluster_mask].apply(
                            lambda r: f"ID: {r['id']}<br>" +
                                     f"DBSCAN: {cluster_name}<br>" +
                                     f"Uncertainty: {r['uncertainty_type']}<br>" +
                                     f"Confidence: {r['confidence']:.3f}",
                            axis=1
                        ),
                        hoverinfo='text'
                    ),
                    row=2, col=col
                )
    
    # Row 3: GMM clustering
    for col, method in enumerate(['pca', 'tsne', 'umap'], 1):
        if f'{method}_1' in train_df.columns:
            subset = train_df.dropna(subset=[f'{method}_1', f'{method}_2']) if method == 'tsne' else train_df
            
            fig.add_trace(
                go.Scatter(
                    x=subset[f'{method}_1'],
                    y=subset[f'{method}_2'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=subset['cluster_gmm'],
                        colorscale='Portland',
                        opacity=subset['gmm_probability'],
                        colorbar=dict(title='GMM Cluster') if col == 3 else None
                    ),
                    text=subset.apply(
                        lambda r: f"ID: {r['id']}<br>" +
                                 f"GMM Cluster: {r['cluster_gmm']}<br>" +
                                 f"GMM Probability: {r['gmm_probability']:.3f}<br>" +
                                 f"Uncertainty: {r['uncertainty_type']}",
                        axis=1
                    ),
                    hoverinfo='text',
                    showlegend=False
                ),
                row=3, col=col
            )
    
    # Additional plots
    # Uncertainty distribution bar chart
    uncertainty_counts = train_df['uncertainty_type'].value_counts()
    fig.add_trace(
        go.Bar(
            x=uncertainty_counts.index,
            y=uncertainty_counts.values,
            marker_color=[uncertainty_colors.get(x, 'gray') for x in uncertainty_counts.index],
            text=uncertainty_counts.values,
            textposition='auto',
            showlegend=False
        ),
        row=1, col=4
    )
    
    # Confidence vs Error scatter
    fig.add_trace(
        go.Scatter(
            x=train_df['confidence'],
            y=(~train_df['is_correct']).astype(int) + np.random.normal(0, 0.02, len(train_df)),
            mode='markers',
            marker=dict(
                size=4,
                color=train_df['uncertainty_type'].map(uncertainty_colors),
                opacity=0.5
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=4
    )
    
    # Feature importance for uncertain cases
    uncertain_mask = train_df['uncertainty_type'] != 'Certain'
    if uncertain_mask.sum() > 0:
        feature_means = train_df[uncertain_mask][
            ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 
             'Going_outside', 'Post_frequency']
        ].mean()
        
        fig.add_trace(
            go.Bar(
                x=feature_means.values,
                y=feature_means.index,
                orientation='h',
                marker_color='indianred',
                showlegend=False
            ),
            row=3, col=4
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive Cluster Analysis of Uncertain Cases<br>' +
                   '<sub>Click on points to see details. Uncertain cases have colored borders.</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=1500,
        width=1800,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes
    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_xaxes(title_text='Component 1', row=i, col=j)
            fig.update_yaxes(title_text='Component 2', row=i, col=j)
    
    # Save as HTML
    output_file = OUTPUT_DIR / 'interactive_cluster_analysis.html'
    fig.write_html(
        str(output_file),
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    print(f"\nSaved interactive visualization to: {output_file}")
    
    # Create a second detailed view for uncertain cases only
    create_uncertain_cases_detail(train_df)
    
    return fig

def create_uncertain_cases_detail(train_df):
    """Create detailed view of uncertain cases only"""
    
    print("\nCreating detailed uncertain cases visualization...")
    
    # Filter uncertain cases
    uncertain_df = train_df[train_df['uncertainty_type'] != 'Certain'].copy()
    
    if len(uncertain_df) == 0:
        print("No uncertain cases found!")
        return
    
    # Create parallel coordinates plot
    fig = go.Figure()
    
    # Normalize features for parallel coordinates
    features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 
                'Going_outside', 'Post_frequency', 'confidence']
    
    # Create dimensions
    dimensions = []
    for feat in features:
        if feat in uncertain_df.columns:
            values = uncertain_df[feat].fillna(uncertain_df[feat].median())
            dimensions.append(
                dict(
                    label=feat.replace('_', ' '),
                    values=values,
                    range=[values.min(), values.max()]
                )
            )
    
    # Add categorical dimension for actual personality
    dimensions.append(
        dict(
            label='Actual',
            values=uncertain_df['Personality'].map({'Extrovert': 0, 'Introvert': 1}),
            tickvals=[0, 1],
            ticktext=['E', 'I']
        )
    )
    
    # Add predicted personality
    dimensions.append(
        dict(
            label='Predicted',
            values=uncertain_df['predicted'].map({'Extrovert': 0, 'Introvert': 1}),
            tickvals=[0, 1],
            ticktext=['E', 'I']
        )
    )
    
    # Color by uncertainty type
    color_map = {'Uncertain': 1, 'Very Uncertain': 2, 'High Conf Error': 3}
    colors = uncertain_df['uncertainty_type'].map(color_map).fillna(0)
    
    fig.add_trace(
        go.Parcoords(
            dimensions=dimensions,
            line=dict(
                color=colors,
                colorscale=[[0, 'orange'], [0.5, 'red'], [1, 'darkred']],
                showscale=True,
                colorbar=dict(
                    title='Uncertainty<br>Type',
                    tickvals=[1, 2, 3],
                    ticktext=['Uncertain', 'Very<br>Uncertain', 'High Conf<br>Error']
                )
            ),
            labelfont=dict(size=12),
            tickfont=dict(size=10)
        )
    )
    
    fig.update_layout(
        title={
            'text': 'Uncertain Cases - Parallel Coordinates View<br>' +
                   '<sub>Each line represents one uncertain sample. Drag dimensions to reorder.</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        width=1400,
        margin=dict(l=150, r=150, t=100, b=50)
    )
    
    # Save
    output_file = OUTPUT_DIR / 'uncertain_cases_detail.html'
    fig.write_html(
        str(output_file),
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    print(f"Saved uncertain cases detail to: {output_file}")
    
    # Create cluster summary statistics
    create_cluster_summary(train_df)

def create_cluster_summary(train_df):
    """Create summary statistics for each cluster"""
    
    print("\nCreating cluster summary statistics...")
    
    summary_data = []
    
    for method in ['kmeans', 'dbscan', 'hierarchical', 'gmm']:
        cluster_col = f'cluster_{method}'
        
        for cluster_id in sorted(train_df[cluster_col].unique()):
            if cluster_id == -1 and method == 'dbscan':
                cluster_name = f'{method}_noise'
            else:
                cluster_name = f'{method}_{cluster_id}'
            
            cluster_data = train_df[train_df[cluster_col] == cluster_id]
            
            summary_data.append({
                'method': method,
                'cluster': cluster_id,
                'n_samples': len(cluster_data),
                'pct_extrovert': (cluster_data['Personality'] == 'Extrovert').mean() * 100,
                'pct_uncertain': (cluster_data['uncertainty_type'] != 'Certain').mean() * 100,
                'avg_confidence': cluster_data['confidence'].mean(),
                'pct_errors': (~cluster_data['is_correct']).mean() * 100,
                'avg_alone_time': cluster_data['Time_spent_Alone'].mean(),
                'avg_social_events': cluster_data['Social_event_attendance'].mean(),
                'avg_friends': cluster_data['Friends_circle_size'].mean()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create heatmap of cluster characteristics
    fig = go.Figure()
    
    # Prepare data for heatmap
    metrics = ['pct_extrovert', 'pct_uncertain', 'avg_confidence', 'pct_errors']
    
    for i, method in enumerate(['kmeans', 'dbscan', 'hierarchical', 'gmm']):
        method_data = summary_df[summary_df['method'] == method]
        
        z_data = []
        for metric in metrics:
            z_data.append(method_data[metric].values)
        
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=[f'C{c}' for c in method_data['cluster']],
                y=metrics,
                colorscale='RdBu_r',
                showscale=(i == 0),
                name=method,
                visible=(i == 0)
            )
        )
    
    # Add buttons for method selection
    buttons = []
    for i, method in enumerate(['kmeans', 'dbscan', 'hierarchical', 'gmm']):
        visible = [False] * 4
        visible[i] = True
        buttons.append(
            dict(
                label=method.upper(),
                method='update',
                args=[{'visible': visible}]
            )
        )
    
    fig.update_layout(
        title='Cluster Characteristics by Method',
        updatemenus=[dict(
            type='buttons',
            direction='right',
            x=0.7,
            y=1.15,
            buttons=buttons
        )],
        height=500,
        width=800
    )
    
    # Save
    output_file = OUTPUT_DIR / 'cluster_summary.html'
    fig.write_html(str(output_file), include_plotlyjs='cdn')
    
    # Also save summary table
    summary_df.to_csv(OUTPUT_DIR / 'cluster_summary_statistics.csv', index=False)
    
    print(f"Saved cluster summary to: {output_file}")
    print(f"Saved statistics to: cluster_summary_statistics.csv")

def main():
    # Load and prepare data
    train_df = load_and_prepare_data()
    
    # Perform clustering
    train_df, X_scaled, clustering_results = perform_clustering(train_df)
    
    # Perform dimensionality reduction
    train_df, reductions = perform_dimensionality_reduction(X_scaled, train_df)
    
    # Create interactive visualizations
    fig = create_interactive_visualizations(train_df)
    
    print("\n" + "="*60)
    print("INTERACTIVE ANALYSIS COMPLETE")
    print("="*60)
    print("\nCreated files:")
    print("  - interactive_cluster_analysis.html (main visualization)")
    print("  - uncertain_cases_detail.html (parallel coordinates)")
    print("  - cluster_summary.html (cluster characteristics)")
    print("  - cluster_summary_statistics.csv (detailed stats)")
    
    print("\nOpen the HTML files in a web browser to explore interactively!")

if __name__ == "__main__":
    main()