#!/usr/bin/env python3
"""
Create interactive clustering visualization with Plotly
- Toggle datasets on/off
- Click points to see ID and details
- Multiple clustering methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
WORKSPACE_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE")
OUTPUT_DIR = WORKSPACE_DIR / "scripts/output"

def load_and_prepare_data():
    """Load all data and prepare features"""
    print("="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)
    
    # Load original data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Load removed samples info
    removed_info = pd.read_csv(OUTPUT_DIR / "removed_hard_cases_info.csv")
    removed_ids = set(removed_info['id'].values)
    
    # Create group labels
    train_df['group'] = train_df['id'].apply(lambda x: 'Removed' if x in removed_ids else 'Kept')
    test_df['group'] = 'Test'
    
    # Combine all data
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"Total samples: {len(all_df)}")
    print(f"Groups: {all_df['group'].value_counts().to_dict()}")
    
    # Prepare features
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size',
                   'Going_outside', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
    
    X = all_df[feature_cols].copy()
    
    # Convert binary features
    for col in ['Stage_fear', 'Drained_after_socializing']:
        X[col] = (X[col] == 'Yes').astype(int)
    
    # Handle missing values - store info before filling
    all_df['has_missing'] = X.isnull().any(axis=1)
    
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return all_df, X_scaled, feature_cols

def create_hover_text(df):
    """Create detailed hover text for each point"""
    hover_texts = []
    
    for idx, row in df.iterrows():
        text = f"<b>ID: {row['id']}</b><br>"
        text += f"Group: {row['group']}<br>"
        
        if 'Personality' in row and pd.notna(row['Personality']):
            text += f"Personality: {row['Personality']}<br>"
        
        text += f"Has Missing: {'Yes' if row['has_missing'] else 'No'}<br>"
        
        # Add feature values
        text += "<br><b>Features:</b><br>"
        text += f"Time Alone: {row['Time_spent_Alone']:.1f}<br>" if pd.notna(row['Time_spent_Alone']) else "Time Alone: NaN<br>"
        text += f"Social Events: {row['Social_event_attendance']:.1f}<br>" if pd.notna(row['Social_event_attendance']) else "Social Events: NaN<br>"
        text += f"Friends: {row['Friends_circle_size']:.1f}<br>" if pd.notna(row['Friends_circle_size']) else "Friends: NaN<br>"
        text += f"Going Outside: {row['Going_outside']:.1f}<br>" if pd.notna(row['Going_outside']) else "Going Outside: NaN<br>"
        text += f"Post Freq: {row['Post_frequency']:.1f}<br>" if pd.notna(row['Post_frequency']) else "Post Freq: NaN<br>"
        text += f"Stage Fear: {row['Stage_fear']}<br>"
        text += f"Drained: {row['Drained_after_socializing']}"
        
        hover_texts.append(text)
    
    return hover_texts

def create_interactive_visualization():
    """Create main interactive visualization"""
    print("\n" + "="*60)
    print("CREATING INTERACTIVE VISUALIZATION")
    print("="*60)
    
    # Load and prepare data
    all_df, X_scaled, feature_cols = load_and_prepare_data()
    
    # Compute dimensionality reductions
    print("\nComputing dimensionality reductions...")
    
    # PCA
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    all_df['pca_x'] = pca_coords[:, 0]
    all_df['pca_y'] = pca_coords[:, 1]
    all_df['pca_z'] = pca_coords[:, 2]
    
    # t-SNE
    print("Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_coords = tsne.fit_transform(X_scaled)
    all_df['tsne_x'] = tsne_coords[:, 0]
    all_df['tsne_y'] = tsne_coords[:, 1]
    
    # Create hover text
    all_df['hover_text'] = create_hover_text(all_df)
    
    # Define colors
    color_map = {
        'Kept': '#3498db',     # Blue
        'Removed': '#e74c3c',  # Red
        'Test': '#f39c12'      # Orange
    }
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PCA 2D', 't-SNE', 'PCA 3D', 'Feature Space (First 2)'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter3d'}, {'type': 'scatter'}]],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # Add traces for each group
    for group in ['Kept', 'Removed', 'Test']:
        df_group = all_df[all_df['group'] == group]
        
        # PCA 2D
        fig.add_trace(
            go.Scatter(
                x=df_group['pca_x'],
                y=df_group['pca_y'],
                mode='markers',
                name=f'{group} ({len(df_group)})',
                marker=dict(
                    size=5,
                    color=color_map[group],
                    opacity=0.7 if group != 'Removed' else 0.8,
                    line=dict(width=1, color='black') if group == 'Removed' else dict(width=0)
                ),
                text=df_group['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                legendgroup=group
            ),
            row=1, col=1
        )
        
        # t-SNE
        fig.add_trace(
            go.Scatter(
                x=df_group['tsne_x'],
                y=df_group['tsne_y'],
                mode='markers',
                name=f'{group} ({len(df_group)})',
                marker=dict(
                    size=5,
                    color=color_map[group],
                    opacity=0.7 if group != 'Removed' else 0.8,
                    line=dict(width=1, color='black') if group == 'Removed' else dict(width=0)
                ),
                text=df_group['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                showlegend=False,
                legendgroup=group
            ),
            row=1, col=2
        )
        
        # PCA 3D
        fig.add_trace(
            go.Scatter3d(
                x=df_group['pca_x'],
                y=df_group['pca_y'],
                z=df_group['pca_z'],
                mode='markers',
                name=f'{group} ({len(df_group)})',
                marker=dict(
                    size=3,
                    color=color_map[group],
                    opacity=0.7 if group != 'Removed' else 0.8,
                    line=dict(width=0.5, color='black') if group == 'Removed' else dict(width=0)
                ),
                text=df_group['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                showlegend=False,
                legendgroup=group
            ),
            row=2, col=1
        )
        
        # Feature space (first 2 features after scaling)
        fig.add_trace(
            go.Scatter(
                x=X_scaled[all_df['group'] == group][:, 0],
                y=X_scaled[all_df['group'] == group][:, 1],
                mode='markers',
                name=f'{group} ({len(df_group)})',
                marker=dict(
                    size=5,
                    color=color_map[group],
                    opacity=0.7 if group != 'Removed' else 0.8,
                    line=dict(width=1, color='black') if group == 'Removed' else dict(width=0)
                ),
                text=df_group['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                showlegend=False,
                legendgroup=group
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive Clustering Visualization - Click on points for details',
            'font': {'size': 20}
        },
        height=1000,
        width=1400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    # Update axes
    fig.update_xaxes(title_text=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", row=1, col=1)
    fig.update_yaxes(title_text=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", row=1, col=1)
    
    fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)
    
    fig.update_xaxes(title_text=f"{feature_cols[0]} (scaled)", row=2, col=2)
    fig.update_yaxes(title_text=f"{feature_cols[1]} (scaled)", row=2, col=2)
    
    # Save to HTML
    output_file = OUTPUT_DIR / 'interactive_clustering.html'
    fig.write_html(
        str(output_file),
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    print(f"\nInteractive visualization saved to: {output_file}")
    print("\nFeatures:")
    print("- Click on any point to see ID and all details")
    print("- Click on legend items to toggle groups on/off")
    print("- Use zoom, pan, and other controls")
    print("- Hover over points for quick info")
    
    # Also create a focused single plot
    create_focused_plot(all_df, color_map)
    
    return all_df

def create_focused_plot(all_df, color_map):
    """Create a single, more detailed plot"""
    print("\n" + "="*60)
    print("CREATING FOCUSED PLOT")
    print("="*60)
    
    fig = go.Figure()
    
    # Add traces for each group
    for group in ['Test', 'Kept', 'Removed']:  # Order matters for visibility
        df_group = all_df[all_df['group'] == group]
        
        fig.add_trace(
            go.Scatter(
                x=df_group['pca_x'],
                y=df_group['pca_y'],
                mode='markers',
                name=f'{group} ({len(df_group)} samples)',
                marker=dict(
                    size=8 if group == 'Removed' else 6,
                    color=color_map[group],
                    opacity=0.8 if group == 'Removed' else 0.6,
                    line=dict(width=1, color='black') if group == 'Removed' else dict(width=0)
                ),
                text=df_group['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                customdata=df_group['id']
            )
        )
    
    fig.update_layout(
        title='PCA Visualization - Click points to see details',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        height=800,
        width=1200,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=14)
        )
    )
    
    # Add click event info
    fig.add_annotation(
        text="Click on any point to see its ID and features. Click legend to toggle groups.",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    output_file = OUTPUT_DIR / 'interactive_pca_focused.html'
    fig.write_html(str(output_file))
    
    print(f"Focused plot saved to: {output_file}")

def create_sample_search_page():
    """Create a simple HTML page with search functionality"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Sample Search Tool</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .search-box { margin: 20px 0; }
        input { padding: 5px; width: 200px; }
        button { padding: 5px 10px; }
        .result { margin-top: 20px; padding: 10px; border: 1px solid #ccc; }
        .removed { background-color: #ffe6e6; }
        .kept { background-color: #e6f3ff; }
        .test { background-color: #fff4e6; }
    </style>
</head>
<body>
    <h1>Sample Search Tool</h1>
    <div class="search-box">
        <input type="number" id="sampleId" placeholder="Enter Sample ID">
        <button onclick="searchSample()">Search</button>
    </div>
    <div id="result"></div>
    
    <h2>Quick Links:</h2>
    <ul>
        <li><a href="interactive_clustering.html">Full Interactive Visualization</a></li>
        <li><a href="interactive_pca_focused.html">Focused PCA Plot</a></li>
    </ul>
    
    <script>
        // This would need the data embedded or loaded via AJAX
        function searchSample() {
            const id = document.getElementById('sampleId').value;
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `<p>Search functionality would require embedding data or server setup.</p>
                                   <p>Please use the interactive visualizations to click on points.</p>`;
        }
    </script>
</body>
</html>
    """
    
    with open(OUTPUT_DIR / 'search_tool.html', 'w') as f:
        f.write(html_content)
    
    print(f"Search page saved to: {OUTPUT_DIR / 'search_tool.html'}")

def main():
    # Create interactive visualization
    all_df = create_interactive_visualization()
    
    # Create search page
    create_sample_search_page()
    
    # Print summary
    print("\n" + "="*60)
    print("INTERACTIVE VISUALIZATIONS CREATED")
    print("="*60)
    print("\nFiles created:")
    print(f"1. {OUTPUT_DIR / 'interactive_clustering.html'} - Full multi-plot visualization")
    print(f"2. {OUTPUT_DIR / 'interactive_pca_focused.html'} - Focused PCA plot")
    print(f"3. {OUTPUT_DIR / 'search_tool.html'} - Simple search interface")
    print("\nOpen any HTML file in your browser to explore the data interactively!")
    
    # Save data summary
    summary = all_df.groupby('group')[['id', 'has_missing']].agg({
        'id': 'count',
        'has_missing': 'mean'
    }).round(3)
    summary.columns = ['count', 'missing_rate']
    print("\nData Summary:")
    print(summary)

if __name__ == "__main__":
    main()