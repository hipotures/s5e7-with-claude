#!/usr/bin/env python3
"""
Fixed color selector in cluster visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import ydf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import umap
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/mnt/ml/kaggle/playground-series-s5e7/")
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def load_and_prepare_data():
    """Load data and identify uncertain cases"""
    
    print("="*60)
    print("FIXED COLOR SELECTOR - CLUSTER ANALYSIS")
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
    
    model = learner.train(train_df[feature_cols + ['Personality']])
    
    # Get predictions
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
    
    # Create uncertainty levels
    train_df['uncertainty_level'] = 0  # Certain
    train_df.loc[train_df['confidence'] < 0.3, 'uncertainty_level'] = 1  # Low confidence
    train_df.loc[train_df['confidence'] < 0.1, 'uncertainty_level'] = 2  # Very low confidence
    train_df.loc[~train_df['is_correct'] & (train_df['confidence'] > 0.8), 'uncertainty_level'] = 3  # High confidence error
    
    train_df['uncertainty_type'] = train_df['uncertainty_level'].map({
        0: 'Certain (conf > 0.3)',
        1: 'Low Confidence (0.1-0.3)', 
        2: 'Very Low Confidence (< 0.1)',
        3: 'High Confidence Error'
    })
    
    print(f"\nUncertainty distribution:")
    print(train_df['uncertainty_type'].value_counts())
    
    return train_df

def perform_analysis(train_df):
    """Perform clustering and dimensionality reduction"""
    
    print("\nPerforming clustering and dimensionality reduction...")
    
    # Prepare features
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency',
                       'Stage_fear_encoded', 'Drained_encoded']
    
    X = train_df[numeric_features].fillna(train_df[numeric_features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    train_df['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
    
    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=50)
    train_df['cluster_dbscan'] = dbscan.fit_predict(X_scaled)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    train_df['pca_x'] = X_pca[:, 0]
    train_df['pca_y'] = X_pca[:, 1]
    
    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    train_df['umap_x'] = X_umap[:, 0]
    train_df['umap_y'] = X_umap[:, 1]
    
    # t-SNE on subset
    if len(train_df) > 5000:
        indices = np.random.choice(len(train_df), 5000, replace=False)
        X_subset = X_scaled[indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_subset)
        
        train_df['tsne_x'] = np.nan
        train_df['tsne_y'] = np.nan
        train_df.loc[indices, 'tsne_x'] = X_tsne[:, 0]
        train_df.loc[indices, 'tsne_y'] = X_tsne[:, 1]
    
    return train_df

def create_fixed_visualization(train_df):
    """Create visualization with properly working color selector"""
    
    print("\nCreating visualization with fixed color selector...")
    
    # Prepare data for JavaScript
    data_for_js = []
    
    for idx, row in train_df.iterrows():
        point_data = {
            'id': int(row['id']),
            'index': idx,
            'personality': row['Personality'],
            'predicted': row['predicted'],
            'confidence': float(row['confidence']),
            'probability': float(row['probability']),
            'is_correct': bool(row['is_correct']),
            'uncertainty_level': int(row['uncertainty_level']),
            'uncertainty_type': row['uncertainty_type'],
            'cluster_kmeans': int(row['cluster_kmeans']),
            'cluster_dbscan': int(row['cluster_dbscan']),
            'pca_x': float(row['pca_x']),
            'pca_y': float(row['pca_y']),
            'umap_x': float(row['umap_x']),
            'umap_y': float(row['umap_y']),
            'tsne_x': float(row['tsne_x']) if not pd.isna(row['tsne_x']) else None,
            'tsne_y': float(row['tsne_y']) if not pd.isna(row['tsne_y']) else None,
            'time_alone': float(row['Time_spent_Alone']) if not pd.isna(row['Time_spent_Alone']) else None,
            'social_events': float(row['Social_event_attendance']) if not pd.isna(row['Social_event_attendance']) else None,
            'friends': float(row['Friends_circle_size']) if not pd.isna(row['Friends_circle_size']) else None,
            'going_outside': float(row['Going_outside']) if not pd.isna(row['Going_outside']) else None,
            'post_frequency': float(row['Post_frequency']) if not pd.isna(row['Post_frequency']) else None,
            'stage_fear': row['Stage_fear'],
            'drained': row['Drained_after_socializing']
        }
        data_for_js.append(point_data)
    
    # Create HTML with fixed color handling
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Fixed Color Selector - Cluster Analysis</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        
        #main-container {
            display: flex;
            width: 100%;
            height: 100%;
        }
        
        #controls {
            width: 320px;
            padding: 20px;
            background-color: #f5f5f5;
            overflow-y: auto;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        }
        
        #visualization {
            flex: 1;
            position: relative;
            overflow: hidden;
        }
        
        #side-panel {
            position: absolute;
            right: -350px;
            top: 0;
            width: 350px;
            height: 100%;
            background-color: white;
            box-shadow: -2px 0 5px rgba(0,0,0,0.1);
            transition: right 0.3s ease;
            overflow-y: auto;
            padding: 20px;
            box-sizing: border-box;
        }
        
        #side-panel.active {
            right: 0;
        }
        
        .control-group {
            margin-bottom: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .control-group h3 {
            margin-top: 0;
            color: #333;
        }
        
        label {
            display: block;
            margin: 5px 0;
            color: #666;
        }
        
        select, input[type="range"] {
            width: 100%;
            margin: 5px 0;
            padding: 8px;
            font-size: 14px;
        }
        
        .checkbox-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .checkbox-group label {
            display: flex;
            align-items: center;
            cursor: pointer;
            padding: 5px;
            border-radius: 3px;
            transition: background-color 0.2s;
        }
        
        .checkbox-group label:hover {
            background-color: #f0f0f0;
        }
        
        .checkbox-group input[type="checkbox"] {
            margin-right: 8px;
            width: 16px;
            height: 16px;
            cursor: pointer;
        }
        
        .legend {
            display: flex;
            flex-direction: column;
            gap: 5px;
            margin-top: 10px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid #333;
        }
        
        #tooltip {
            position: absolute;
            padding: 10px;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 5px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 12px;
            z-index: 1000;
        }
        
        .close-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            cursor: pointer;
            font-size: 24px;
            color: #666;
        }
        
        .close-btn:hover {
            color: #000;
        }
        
        .detail-group {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        
        .detail-group h4 {
            margin: 0 0 10px 0;
            color: #2c5aa0;
        }
        
        .detail-item {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        
        .detail-label {
            color: #666;
        }
        
        .detail-value {
            font-weight: bold;
        }
        
        /* Uncertainty colors */
        .uncertainty-0 { fill: #4CAF50; stroke: #4CAF50; }
        .uncertainty-1 { fill: #FF9800; stroke: #FF9800; }
        .uncertainty-2 { fill: #F44336; stroke: #F44336; }
        .uncertainty-3 { fill: #9C27B0; stroke: #9C27B0; }
        
        .point {
            cursor: pointer;
            transition: all 0.2s;
            stroke-width: 1;
        }
        
        .point:hover {
            stroke-width: 3;
            stroke: #000 !important;
        }
        
        .selected {
            stroke: #000 !important;
            stroke-width: 3;
        }
        
        #point-count {
            margin-top: 10px;
            padding: 10px;
            background-color: #e3f2fd;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        
        .info-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-size: 12px;
        }
        
        .current-coloring {
            margin-top: 10px;
            padding: 8px;
            background-color: #e8f5e9;
            border-radius: 3px;
            font-size: 12px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div id="main-container">
        <div id="controls">
            <h2>Cluster Analysis</h2>
            
            <div class="info-box">
                <strong>💡 Tip:</strong> Use "Color By" to change how points are colored
            </div>
            
            <div class="control-group">
                <h3>Visualization Method</h3>
                <select id="method-select">
                    <option value="pca">PCA</option>
                    <option value="umap">UMAP</option>
                    <option value="tsne">t-SNE (subset)</option>
                </select>
            </div>
            
            <div class="control-group">
                <h3>Color By</h3>
                <select id="cluster-select">
                    <option value="uncertainty">Uncertainty Level</option>
                    <option value="kmeans">K-Means Clusters</option>
                    <option value="dbscan">DBSCAN Clusters</option>
                    <option value="personality">Actual Personality</option>
                    <option value="predicted">Predicted Personality</option>
                </select>
                <div class="current-coloring">Currently showing: <strong id="current-color">Uncertainty Level</strong></div>
            </div>
            
            <div class="control-group">
                <h3>Filter by Uncertainty</h3>
                <div class="checkbox-group">
                    <label>
                        <input type="checkbox" class="uncertainty-filter" value="0" checked>
                        Certain (conf > 30%)
                    </label>
                    <label>
                        <input type="checkbox" class="uncertainty-filter" value="1" checked>
                        Low Confidence (10-30%)
                    </label>
                    <label>
                        <input type="checkbox" class="uncertainty-filter" value="2" checked>
                        Very Low Confidence (< 10%)
                    </label>
                    <label>
                        <input type="checkbox" class="uncertainty-filter" value="3" checked>
                        High Confidence Errors
                    </label>
                </div>
                
                <div class="legend" id="uncertainty-legend">
                    <h4>Legend:</h4>
                    <div class="legend-item">
                        <div class="legend-color uncertainty-0"></div>
                        <span>Certain</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color uncertainty-1"></div>
                        <span>Low Confidence</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color uncertainty-2"></div>
                        <span>Very Low Confidence</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color uncertainty-3"></div>
                        <span>High Conf Error</span>
                    </div>
                </div>
                
                <div class="legend" id="other-legend" style="display: none;">
                    <h4>Legend:</h4>
                    <div id="dynamic-legend"></div>
                </div>
            </div>
            
            <div class="control-group">
                <h3>Point Size</h3>
                <input type="range" id="size-slider" min="2" max="10" value="5">
                <label>Size: <span id="size-value">5</span></label>
            </div>
            
            <div class="control-group">
                <h3>Opacity</h3>
                <input type="range" id="opacity-slider" min="10" max="100" value="70">
                <label>Opacity: <span id="opacity-value">70</span>%</label>
            </div>
            
            <div id="point-count"></div>
        </div>
        
        <div id="visualization">
            <svg id="plot"></svg>
            <div id="tooltip"></div>
            <div id="side-panel">
                <span class="close-btn" onclick="closePanel()">&times;</span>
                <h2>Sample Details</h2>
                <div id="panel-content"></div>
            </div>
        </div>
    </div>
    
    <script>
"""
    
    # Add JavaScript with fixed color handling
    html_content += f"""
        const data = {json.dumps(data_for_js)};
        
        // Initialize
        let currentMethod = 'pca';
        let currentClustering = 'uncertainty';
        let selectedPoint = null;
        let pointSize = 5;
        let opacity = 0.7;
        
        // Color schemes
        const uncertaintyColors = ['#4CAF50', '#FF9800', '#F44336', '#9C27B0'];
        const clusterColors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'];
        const personalityColors = {{ 'Extrovert': '#FF6B6B', 'Introvert': '#4ECDC4' }};
        
        // Set up SVG
        const margin = {{top: 50, right: 50, bottom: 50, left: 50}};
        const svg = d3.select('#plot');
        const tooltip = d3.select('#tooltip');
        
        function getPointColor(d) {{
            if (currentClustering === 'uncertainty') {{
                return uncertaintyColors[d.uncertainty_level];
            }} else if (currentClustering === 'personality') {{
                return personalityColors[d.personality];
            }} else if (currentClustering === 'predicted') {{
                return personalityColors[d.predicted];
            }} else if (currentClustering === 'kmeans') {{
                return clusterColors[d.cluster_kmeans % clusterColors.length];
            }} else if (currentClustering === 'dbscan') {{
                return d.cluster_dbscan === -1 ? '#cccccc' : clusterColors[d.cluster_dbscan % clusterColors.length];
            }}
            return '#666666';
        }}
        
        function updateLegend() {{
            const uncertaintyLegend = document.getElementById('uncertainty-legend');
            const otherLegend = document.getElementById('other-legend');
            const dynamicLegend = document.getElementById('dynamic-legend');
            
            if (currentClustering === 'uncertainty') {{
                uncertaintyLegend.style.display = 'block';
                otherLegend.style.display = 'none';
            }} else {{
                uncertaintyLegend.style.display = 'none';
                otherLegend.style.display = 'block';
                
                dynamicLegend.innerHTML = '';
                
                if (currentClustering === 'personality' || currentClustering === 'predicted') {{
                    // Personality legend
                    ['Extrovert', 'Introvert'].forEach(type => {{
                        const item = document.createElement('div');
                        item.className = 'legend-item';
                        item.innerHTML = `
                            <div class="legend-color" style="background-color: ${{personalityColors[type]}}"></div>
                            <span>${{type}}</span>
                        `;
                        dynamicLegend.appendChild(item);
                    }});
                }} else if (currentClustering === 'kmeans') {{
                    // K-means legend
                    for (let i = 0; i < 5; i++) {{
                        const item = document.createElement('div');
                        item.className = 'legend-item';
                        item.innerHTML = `
                            <div class="legend-color" style="background-color: ${{clusterColors[i]}}"></div>
                            <span>Cluster ${{i}}</span>
                        `;
                        dynamicLegend.appendChild(item);
                    }}
                }} else if (currentClustering === 'dbscan') {{
                    // DBSCAN legend
                    const item = document.createElement('div');
                    item.className = 'legend-item';
                    item.innerHTML = `
                        <div class="legend-color" style="background-color: #cccccc"></div>
                        <span>Noise (-1)</span>
                    `;
                    dynamicLegend.appendChild(item);
                    
                    // Add actual clusters
                    const uniqueClusters = [...new Set(data.map(d => d.cluster_dbscan))].filter(c => c !== -1).sort();
                    uniqueClusters.forEach(cluster => {{
                        const item = document.createElement('div');
                        item.className = 'legend-item';
                        item.innerHTML = `
                            <div class="legend-color" style="background-color: ${{clusterColors[cluster % clusterColors.length]}}"></div>
                            <span>Cluster ${{cluster}}</span>
                        `;
                        dynamicLegend.appendChild(item);
                    }});
                }}
            }}
        }}
        
        function updateVisualization() {{
            const width = document.getElementById('visualization').clientWidth;
            const height = document.getElementById('visualization').clientHeight;
            
            svg.attr('width', width).attr('height', height);
            
            const innerWidth = width - margin.left - margin.right;
            const innerHeight = height - margin.top - margin.bottom;
            
            // Get selected uncertainty levels
            const selectedLevels = Array.from(document.querySelectorAll('.uncertainty-filter:checked'))
                .map(cb => parseInt(cb.value));
            
            // Filter data
            const filteredData = data.filter(d => {{
                if (currentMethod === 'tsne' && (d.tsne_x === null || d.tsne_y === null)) {{
                    return false;
                }}
                return selectedLevels.includes(d.uncertainty_level);
            }});
            
            // Update point count
            document.getElementById('point-count').textContent = 
                `Showing ${{filteredData.length}} / ${{data.length}} points`;
            
            // Clear SVG
            svg.selectAll('*').remove();
            
            const g = svg.append('g')
                .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
            
            // Set up scales
            const xExtent = d3.extent(filteredData, d => d[currentMethod + '_x']);
            const yExtent = d3.extent(filteredData, d => d[currentMethod + '_y']);
            
            const xScale = d3.scaleLinear()
                .domain(xExtent)
                .range([0, innerWidth]);
            
            const yScale = d3.scaleLinear()
                .domain(yExtent)
                .range([innerHeight, 0]);
            
            // Sort data so that uncertain points are drawn on top
            const sortedData = filteredData.sort((a, b) => a.uncertainty_level - b.uncertainty_level);
            
            // Draw points
            g.selectAll('.point')
                .data(sortedData)
                .enter()
                .append('circle')
                .attr('class', 'point')
                .attr('cx', d => xScale(d[currentMethod + '_x']))
                .attr('cy', d => yScale(d[currentMethod + '_y']))
                .attr('r', pointSize)
                .attr('fill', d => getPointColor(d))
                .attr('fill-opacity', opacity)
                .attr('stroke', d => getPointColor(d))
                .attr('stroke-opacity', opacity)
                .on('mouseover', function(event, d) {{
                    tooltip.style('opacity', 1)
                        .html(`
                            <strong>ID: ${{d.id}}</strong><br>
                            Type: ${{d.uncertainty_type}}<br>
                            Confidence: ${{(d.confidence * 100).toFixed(1)}}%<br>
                            Actual: ${{d.personality}}<br>
                            Predicted: ${{d.predicted}}<br>
                            K-Means: ${{d.cluster_kmeans}}<br>
                            DBSCAN: ${{d.cluster_dbscan}}
                        `)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 28) + 'px');
                }})
                .on('mouseout', function() {{
                    tooltip.style('opacity', 0);
                }})
                .on('click', function(event, d) {{
                    // Remove previous selection
                    svg.selectAll('.point').classed('selected', false);
                    
                    // Select this point
                    d3.select(this).classed('selected', true);
                    selectedPoint = d;
                    showPanel(d);
                }});
            
            // Add axes
            g.append('g')
                .attr('transform', `translate(0,${{innerHeight}})`)
                .call(d3.axisBottom(xScale));
            
            g.append('g')
                .call(d3.axisLeft(yScale));
            
            // Add title
            const coloringText = {{
                'uncertainty': 'Uncertainty Level',
                'kmeans': 'K-Means Clusters',
                'dbscan': 'DBSCAN Clusters',
                'personality': 'Actual Personality',
                'predicted': 'Predicted Personality'
            }};
            
            svg.append('text')
                .attr('x', width / 2)
                .attr('y', 30)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .style('font-weight', 'bold')
                .text(`${{currentMethod.toUpperCase()}} Visualization - Colored by ${{coloringText[currentClustering]}}`);
        }}
        
        function showPanel(d) {{
            const panel = document.getElementById('side-panel');
            const content = document.getElementById('panel-content');
            
            content.innerHTML = `
                <div class="detail-group">
                    <h4>Basic Information</h4>
                    <div class="detail-item">
                        <span class="detail-label">ID:</span>
                        <span class="detail-value">${{d.id}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Actual Personality:</span>
                        <span class="detail-value">${{d.personality}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Predicted:</span>
                        <span class="detail-value">${{d.predicted}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Correct:</span>
                        <span class="detail-value">${{d.is_correct ? 'Yes' : 'No'}}</span>
                    </div>
                </div>
                
                <div class="detail-group">
                    <h4>Confidence Metrics</h4>
                    <div class="detail-item">
                        <span class="detail-label">Confidence:</span>
                        <span class="detail-value">${{(d.confidence * 100).toFixed(1)}}%</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">P(Introvert):</span>
                        <span class="detail-value">${{d.probability.toFixed(3)}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Uncertainty Type:</span>
                        <span class="detail-value">${{d.uncertainty_type}}</span>
                    </div>
                </div>
                
                <div class="detail-group">
                    <h4>Feature Values</h4>
                    <div class="detail-item">
                        <span class="detail-label">Time Alone:</span>
                        <span class="detail-value">${{d.time_alone !== null ? d.time_alone.toFixed(1) : 'N/A'}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Social Events:</span>
                        <span class="detail-value">${{d.social_events !== null ? d.social_events.toFixed(1) : 'N/A'}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Friends:</span>
                        <span class="detail-value">${{d.friends !== null ? d.friends.toFixed(1) : 'N/A'}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Going Outside:</span>
                        <span class="detail-value">${{d.going_outside !== null ? d.going_outside.toFixed(1) : 'N/A'}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Post Frequency:</span>
                        <span class="detail-value">${{d.post_frequency !== null ? d.post_frequency.toFixed(1) : 'N/A'}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Stage Fear:</span>
                        <span class="detail-value">${{d.stage_fear}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Drained:</span>
                        <span class="detail-value">${{d.drained}}</span>
                    </div>
                </div>
                
                <div class="detail-group">
                    <h4>Cluster Assignments</h4>
                    <div class="detail-item">
                        <span class="detail-label">K-Means:</span>
                        <span class="detail-value">${{d.cluster_kmeans}}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">DBSCAN:</span>
                        <span class="detail-value">${{d.cluster_dbscan === -1 ? 'Noise' : d.cluster_dbscan}}</span>
                    </div>
                </div>
            `;
            
            panel.classList.add('active');
        }}
        
        function closePanel() {{
            document.getElementById('side-panel').classList.remove('active');
            svg.selectAll('.point').classed('selected', false);
            selectedPoint = null;
        }}
        
        // Event listeners
        document.getElementById('method-select').addEventListener('change', function() {{
            currentMethod = this.value;
            updateVisualization();
        }});
        
        document.getElementById('cluster-select').addEventListener('change', function() {{
            currentClustering = this.value;
            document.getElementById('current-color').textContent = this.options[this.selectedIndex].text;
            updateLegend();
            updateVisualization();
        }});
        
        document.querySelectorAll('.uncertainty-filter').forEach(cb => {{
            cb.addEventListener('change', updateVisualization);
        }});
        
        document.getElementById('size-slider').addEventListener('input', function() {{
            pointSize = parseInt(this.value);
            document.getElementById('size-value').textContent = pointSize;
            svg.selectAll('.point').attr('r', pointSize);
        }});
        
        document.getElementById('opacity-slider').addEventListener('input', function() {{
            opacity = parseInt(this.value) / 100;
            document.getElementById('opacity-value').textContent = this.value;
            svg.selectAll('.point')
                .attr('fill-opacity', opacity)
                .attr('stroke-opacity', opacity);
        }});
        
        // Window resize
        window.addEventListener('resize', updateVisualization);
        
        // Initial setup
        updateLegend();
        updateVisualization();
    </script>
</body>
</html>
"""
    
    # Save HTML
    output_file = OUTPUT_DIR / 'cluster_analysis_final.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nCreated visualization with fixed color selector: {output_file}")
    print("\nFixed features:")
    print("✓ Color selector properly changes point colors")
    print("✓ Dynamic legend updates based on selected coloring")
    print("✓ Shows current coloring mode")
    print("✓ All clustering methods work correctly")

def main():
    # Load and prepare data
    train_df = load_and_prepare_data()
    
    # Perform analysis
    train_df = perform_analysis(train_df)
    
    # Create visualization
    create_fixed_visualization(train_df)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()