#!/usr/bin/env python3
"""
Cluster visualization with train/test dataset selector
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
    """Load both train and test data"""
    
    print("="*60)
    print("CLUSTER ANALYSIS WITH TRAIN/TEST SELECTOR")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Add dataset indicator
    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'
    test_df['Personality'] = 'Unknown'  # Placeholder for test
    
    # Combine for processing
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Features for clustering
    feature_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                   'Friends_circle_size', 'Going_outside', 'Post_frequency',
                   'Stage_fear', 'Drained_after_socializing']
    
    # Encode categorical features
    all_df['Stage_fear_encoded'] = (all_df['Stage_fear'] == 'Yes').astype(int)
    all_df['Drained_encoded'] = (all_df['Drained_after_socializing'] == 'Yes').astype(int)
    
    # Train model on train data only
    print("\nTraining model on train data...")
    
    learner = ydf.RandomForestLearner(
        label='Personality',
        num_trees=300,
        random_seed=42
    )
    
    model = learner.train(train_df[feature_cols + ['Personality']])
    
    # Get predictions for all data
    predictions = model.predict(all_df[feature_cols])
    
    probabilities = []
    pred_classes = []
    
    for pred in predictions:
        prob_I = float(str(pred))
        probabilities.append(prob_I)
        pred_classes.append('Introvert' if prob_I > 0.5 else 'Extrovert')
    
    all_df['probability'] = probabilities
    all_df['confidence'] = np.abs(np.array(probabilities) - 0.5) * 2
    all_df['predicted'] = pred_classes
    
    # For train data, check if correct
    all_df['is_correct'] = False
    train_mask = all_df['dataset'] == 'train'
    all_df.loc[train_mask, 'is_correct'] = all_df.loc[train_mask, 'predicted'] == all_df.loc[train_mask, 'Personality']
    
    # Create uncertainty levels
    all_df['uncertainty_level'] = 0  # Certain
    all_df.loc[all_df['confidence'] < 0.3, 'uncertainty_level'] = 1  # Low confidence
    all_df.loc[all_df['confidence'] < 0.1, 'uncertainty_level'] = 2  # Very low confidence
    all_df.loc[(all_df['dataset'] == 'train') & (~all_df['is_correct']) & (all_df['confidence'] > 0.8), 'uncertainty_level'] = 3  # High confidence error
    
    all_df['uncertainty_type'] = all_df['uncertainty_level'].map({
        0: 'Certain (conf > 0.3)',
        1: 'Low Confidence (0.1-0.3)', 
        2: 'Very Low Confidence (< 0.1)',
        3: 'High Confidence Error'
    })
    
    print(f"\nUncertainty distribution (train):")
    print(all_df[all_df['dataset'] == 'train']['uncertainty_type'].value_counts())
    
    print(f"\nUncertainty distribution (test):")
    print(all_df[all_df['dataset'] == 'test']['uncertainty_type'].value_counts())
    
    return all_df

def perform_analysis(all_df):
    """Perform clustering and dimensionality reduction on combined data"""
    
    print("\nPerforming clustering and dimensionality reduction...")
    
    # Prepare features
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 
                       'Friends_circle_size', 'Going_outside', 'Post_frequency',
                       'Stage_fear_encoded', 'Drained_encoded']
    
    X = all_df[numeric_features].fillna(all_df[numeric_features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    all_df['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
    
    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=50)
    all_df['cluster_dbscan'] = dbscan.fit_predict(X_scaled)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    all_df['pca_x'] = X_pca[:, 0]
    all_df['pca_y'] = X_pca[:, 1]
    
    # UMAP
    print("Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    all_df['umap_x'] = X_umap[:, 0]
    all_df['umap_y'] = X_umap[:, 1]
    
    # t-SNE on subset
    print("Computing t-SNE on subset...")
    if len(all_df) > 5000:
        indices = np.random.choice(len(all_df), 5000, replace=False)
        X_subset = X_scaled[indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_subset)
        
        all_df['tsne_x'] = np.nan
        all_df['tsne_y'] = np.nan
        all_df.loc[indices, 'tsne_x'] = X_tsne[:, 0]
        all_df.loc[indices, 'tsne_y'] = X_tsne[:, 1]
    
    return all_df

def create_visualization_with_selector(all_df):
    """Create visualization with dataset selector"""
    
    print("\nCreating visualization with dataset selector...")
    
    # Prepare data for JavaScript
    data_for_js = []
    
    for idx, row in all_df.iterrows():
        point_data = {
            'id': int(row['id']),
            'dataset': row['dataset'],
            'personality': row['Personality'],
            'predicted': row['predicted'],
            'confidence': float(row['confidence']),
            'probability': float(row['probability']),
            'is_correct': bool(row['is_correct']) if row['dataset'] == 'train' else None,
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
    
    # Create HTML
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Cluster Analysis - Train/Test Selector</title>
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
        
        .dataset-selector {
            background-color: #e3f2fd;
            border: 2px solid #2196f3;
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
        
        /* Dataset shapes */
        .train-point { }
        .test-point { stroke-dasharray: 3,3; }
        
        .point {
            cursor: pointer;
            transition: all 0.2s;
            stroke-width: 1.5;
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
        
        .dataset-stats {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div id="main-container">
        <div id="controls">
            <h2>Cluster Analysis</h2>
            
            <div class="control-group dataset-selector">
                <h3>📊 Dataset Selection</h3>
                <div class="checkbox-group">
                    <label>
                        <input type="checkbox" class="dataset-filter" value="train" checked>
                        Train (18,524 samples)
                    </label>
                    <label>
                        <input type="checkbox" class="dataset-filter" value="test" checked>
                        Test (6,175 samples)
                    </label>
                </div>
                <div class="dataset-stats" id="dataset-stats"></div>
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
                    <option value="personality">Actual/Predicted Personality</option>
                    <option value="dataset">Dataset (Train/Test)</option>
                </select>
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
                        High Confidence Errors (train only)
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
            
            <div class="warning-box">
                <strong>Note:</strong> Train points are solid circles, test points have dashed borders.
            </div>
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
    
    # Add JavaScript
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
        const personalityColors = {{ 'Extrovert': '#FF6B6B', 'Introvert': '#4ECDC4', 'Unknown': '#999999' }};
        const datasetColors = {{ 'train': '#2196F3', 'test': '#FF5722' }};
        
        // Set up SVG
        const margin = {{top: 50, right: 50, bottom: 50, left: 50}};
        const svg = d3.select('#plot');
        const tooltip = d3.select('#tooltip');
        
        function getPointColor(d) {{
            if (currentClustering === 'uncertainty') {{
                return uncertaintyColors[d.uncertainty_level];
            }} else if (currentClustering === 'personality') {{
                return personalityColors[d.personality === 'Unknown' ? d.predicted : d.personality];
            }} else if (currentClustering === 'kmeans') {{
                return clusterColors[d.cluster_kmeans % clusterColors.length];
            }} else if (currentClustering === 'dbscan') {{
                return d.cluster_dbscan === -1 ? '#cccccc' : clusterColors[d.cluster_dbscan % clusterColors.length];
            }} else if (currentClustering === 'dataset') {{
                return datasetColors[d.dataset];
            }}
            return '#666666';
        }}
        
        function updateDatasetStats() {{
            const selectedDatasets = Array.from(document.querySelectorAll('.dataset-filter:checked'))
                .map(cb => cb.value);
            
            const filteredData = data.filter(d => selectedDatasets.includes(d.dataset));
            const trainCount = filteredData.filter(d => d.dataset === 'train').length;
            const testCount = filteredData.filter(d => d.dataset === 'test').length;
            
            document.getElementById('dataset-stats').innerHTML = `
                Showing: ${{trainCount}} train, ${{testCount}} test<br>
                Total: ${{filteredData.length}} points
            `;
        }}
        
        function updateVisualization() {{
            const width = document.getElementById('visualization').clientWidth;
            const height = document.getElementById('visualization').clientHeight;
            
            svg.attr('width', width).attr('height', height);
            
            const innerWidth = width - margin.left - margin.right;
            const innerHeight = height - margin.top - margin.bottom;
            
            // Get selected datasets
            const selectedDatasets = Array.from(document.querySelectorAll('.dataset-filter:checked'))
                .map(cb => cb.value);
            
            // Get selected uncertainty levels
            const selectedLevels = Array.from(document.querySelectorAll('.uncertainty-filter:checked'))
                .map(cb => parseInt(cb.value));
            
            // Filter data
            const filteredData = data.filter(d => {{
                if (!selectedDatasets.includes(d.dataset)) {{
                    return false;
                }}
                if (currentMethod === 'tsne' && (d.tsne_x === null || d.tsne_y === null)) {{
                    return false;
                }}
                return selectedLevels.includes(d.uncertainty_level);
            }});
            
            // Update counts
            updateDatasetStats();
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
            
            // Sort data: test points on top of train points, uncertain on top
            const sortedData = filteredData.sort((a, b) => {{
                if (a.dataset !== b.dataset) {{
                    return a.dataset === 'train' ? -1 : 1;
                }}
                return a.uncertainty_level - b.uncertainty_level;
            }});
            
            // Draw points
            g.selectAll('.point')
                .data(sortedData)
                .enter()
                .append('circle')
                .attr('class', d => `point ${{d.dataset}}-point`)
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
                            <strong>ID: ${{d.id}} [${{d.dataset}}]</strong><br>
                            Type: ${{d.uncertainty_type}}<br>
                            Confidence: ${{(d.confidence * 100).toFixed(1)}}%<br>
                            ${{d.dataset === 'train' ? `Actual: ${{d.personality}}<br>` : ''}}
                            Predicted: ${{d.predicted}}<br>
                            ${{d.dataset === 'train' && d.is_correct !== null ? `Correct: ${{d.is_correct ? 'Yes' : 'No'}}` : ''}}
                        `)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 28) + 'px');
                }})
                .on('mouseout', function() {{
                    tooltip.style('opacity', 0);
                }})
                .on('click', function(event, d) {{
                    svg.selectAll('.point').classed('selected', false);
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
            const datasets = selectedDatasets.join(' + ');
            svg.append('text')
                .attr('x', width / 2)
                .attr('y', 30)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .style('font-weight', 'bold')
                .text(`${{currentMethod.toUpperCase()}} - ${{datasets.toUpperCase()}} Data`);
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
                        <span class="detail-label">Dataset:</span>
                        <span class="detail-value">${{d.dataset.toUpperCase()}}</span>
                    </div>
                    ${{d.dataset === 'train' ? `
                    <div class="detail-item">
                        <span class="detail-label">Actual Personality:</span>
                        <span class="detail-value">${{d.personality}}</span>
                    </div>` : ''}}
                    <div class="detail-item">
                        <span class="detail-label">Predicted:</span>
                        <span class="detail-value">${{d.predicted}}</span>
                    </div>
                    ${{d.dataset === 'train' ? `
                    <div class="detail-item">
                        <span class="detail-label">Correct:</span>
                        <span class="detail-value">${{d.is_correct ? 'Yes' : 'No'}}</span>
                    </div>` : ''}}
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
            updateVisualization();
        }});
        
        document.querySelectorAll('.dataset-filter').forEach(cb => {{
            cb.addEventListener('change', updateVisualization);
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
        
        // Initial render
        updateDatasetStats();
        updateVisualization();
    </script>
</body>
</html>
"""
    
    # Save HTML
    output_file = OUTPUT_DIR / 'cluster_analysis_train_test.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nCreated visualization: {output_file}")
    print("\nFeatures:")
    print("✓ Dataset selector checkboxes (Train, Test, Train+Test)")
    print("✓ Train points: solid circles")
    print("✓ Test points: dashed borders")
    print("✓ Shows dataset in tooltip and side panel")
    print("✓ Can color by dataset to see distribution")

def main():
    # Load and prepare data
    all_df = load_and_prepare_data()
    
    # Perform analysis
    all_df = perform_analysis(all_df)
    
    # Create visualization
    create_visualization_with_selector(all_df)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()