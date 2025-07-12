#!/usr/bin/env python3
"""
Create interactive visualization with click-to-copy functionality
- Click on point to show persistent info panel
- IDs are copyable
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import json
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
    
    # Handle missing values
    all_df['has_missing'] = X.isnull().any(axis=1)
    
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    all_df['pca_x'] = pca_coords[:, 0]
    all_df['pca_y'] = pca_coords[:, 1]
    
    return all_df, pca

def create_interactive_with_table():
    """Create visualization with clickable table"""
    print("\n" + "="*60)
    print("CREATING INTERACTIVE VISUALIZATION WITH TABLE")
    print("="*60)
    
    all_df, pca = load_and_prepare_data()
    
    # Create the HTML template with table
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Clustering with Click Details</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            gap: 20px;
        }
        #plot-container {
            flex: 1;
            min-width: 800px;
        }
        #info-panel {
            width: 400px;
            background: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            height: fit-content;
            position: sticky;
            top: 20px;
        }
        #info-panel h3 {
            margin-top: 0;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #ddd;
        }
        .info-label {
            font-weight: bold;
        }
        .copyable {
            background: #fff;
            padding: 2px 6px;
            border-radius: 3px;
            cursor: pointer;
            font-family: monospace;
        }
        .copyable:hover {
            background: #e0e0e0;
        }
        .copy-feedback {
            color: green;
            font-size: 12px;
            display: none;
        }
        #selected-points {
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        .point-item {
            background: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            cursor: pointer;
        }
        .point-item:hover {
            background: #f0f0f0;
        }
        .removed { border-left: 5px solid #e74c3c; }
        .kept { border-left: 5px solid #3498db; }
        .test { border-left: 5px solid #f39c12; }
        #search-box {
            margin-bottom: 20px;
        }
        #search-box input {
            width: 100%;
            padding: 8px;
            font-size: 14px;
        }
        .highlight {
            animation: pulse 1s;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div id="plot-container">
        <div id="myDiv"></div>
    </div>
    
    <div id="info-panel">
        <h3>Click on a point for details</h3>
        
        <div id="search-box">
            <input type="number" id="search-input" placeholder="Search by ID (e.g., 12345)" onkeypress="if(event.key==='Enter') searchPoint()">
        </div>
        
        <div id="point-details" style="display:none;">
            <h4>Point Details</h4>
            <div id="details-content"></div>
            <div class="copy-feedback" id="copy-feedback">Copied!</div>
        </div>
        
        <div id="selected-points">
            <h4>Recent Selections</h4>
            <div id="selections-list"></div>
        </div>
    </div>

    <script>
        // Data will be inserted here by Python
        var allData = INSERT_DATA_HERE;
        
        var selectedPoints = [];
        
        // Create color map
        var colorMap = {
            'Kept': '#3498db',
            'Removed': '#e74c3c',
            'Test': '#f39c12'
        };
        
        // Create traces
        var traces = [];
        ['Kept', 'Removed', 'Test'].forEach(function(group) {
            var groupData = allData.filter(d => d.group === group);
            traces.push({
                x: groupData.map(d => d.pca_x),
                y: groupData.map(d => d.pca_y),
                mode: 'markers',
                type: 'scatter',
                name: group + ' (' + groupData.length + ')',
                marker: {
                    size: group === 'Removed' ? 8 : 6,
                    color: colorMap[group],
                    opacity: group === 'Removed' ? 0.8 : 0.6,
                    line: group === 'Removed' ? {width: 1, color: 'black'} : {width: 0}
                },
                customdata: groupData.map(d => d.id),
                hovertemplate: 'ID: %{customdata}<br>Click for details<extra></extra>'
            });
        });
        
        var layout = {
            title: 'Interactive Clustering - Click points to see details',
            xaxis: {title: 'PC1 (INSERT_PC1_VAR% variance)'},
            yaxis: {title: 'PC2 (INSERT_PC2_VAR% variance)'},
            hovermode: 'closest',
            height: 700,
            clickmode: 'event'
        };
        
        var config = {
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };
        
        Plotly.newPlot('myDiv', traces, layout, config);
        
        // Handle click events
        document.getElementById('myDiv').on('plotly_click', function(data) {
            var point = data.points[0];
            var pointId = point.customdata;
            
            // Find full data for this point
            var pointData = allData.find(d => d.id === pointId);
            showPointDetails(pointData);
        });
        
        function showPointDetails(pointData) {
            if (!pointData) return;
            
            // Add to selected points
            selectedPoints.unshift(pointData);
            if (selectedPoints.length > 10) selectedPoints.pop();
            
            // Show details
            var detailsHtml = '<div class="info-row"><span class="info-label">ID:</span>' +
                '<span class="copyable" onclick="copyToClipboard(' + pointData.id + ')">' + 
                pointData.id + '</span></div>';
            
            detailsHtml += '<div class="info-row"><span class="info-label">Group:</span><span>' + 
                pointData.group + '</span></div>';
            
            if (pointData.personality) {
                detailsHtml += '<div class="info-row"><span class="info-label">Personality:</span><span>' + 
                    pointData.personality + '</span></div>';
            }
            
            detailsHtml += '<div class="info-row"><span class="info-label">Has Missing:</span><span>' + 
                (pointData.has_missing ? 'Yes' : 'No') + '</span></div>';
            
            // Add features
            detailsHtml += '<h5>Features:</h5>';
            ['time_alone', 'social_events', 'friends', 'going_out', 'post_freq', 'stage_fear', 'drained'].forEach(function(feat) {
                if (pointData[feat] !== null && pointData[feat] !== undefined) {
                    detailsHtml += '<div class="info-row"><span class="info-label">' + 
                        feat.replace(/_/g, ' ') + ':</span><span>' + 
                        (typeof pointData[feat] === 'number' ? pointData[feat].toFixed(2) : pointData[feat]) + 
                        '</span></div>';
                }
            });
            
            document.getElementById('details-content').innerHTML = detailsHtml;
            document.getElementById('point-details').style.display = 'block';
            
            // Update selections list
            updateSelectionsList();
            
            // Highlight point on plot
            highlightPoint(pointData.id);
        }
        
        function updateSelectionsList() {
            var html = '';
            selectedPoints.forEach(function(point) {
                html += '<div class="point-item ' + point.group.toLowerCase() + 
                    '" onclick="showPointDetails(allData.find(d => d.id === ' + point.id + '))">' +
                    '<strong>ID ' + point.id + '</strong> - ' + point.group;
                if (point.personality) html += ' (' + point.personality + ')';
                html += '</div>';
            });
            document.getElementById('selections-list').innerHTML = html;
        }
        
        function copyToClipboard(text) {
            navigator.clipboard.writeText(text).then(function() {
                document.getElementById('copy-feedback').style.display = 'block';
                setTimeout(function() {
                    document.getElementById('copy-feedback').style.display = 'none';
                }, 2000);
            });
        }
        
        function searchPoint() {
            var searchId = parseInt(document.getElementById('search-input').value);
            if (isNaN(searchId)) return;
            
            var pointData = allData.find(d => d.id === searchId);
            if (pointData) {
                showPointDetails(pointData);
                // TODO: Could also zoom to point
            } else {
                alert('ID ' + searchId + ' not found');
            }
        }
        
        function highlightPoint(pointId) {
            // This would require updating the plot - simplified for now
            console.log('Highlighting point', pointId);
        }
    </script>
</body>
</html>
    """
    
    # Prepare data for JavaScript
    js_data = []
    for _, row in all_df.iterrows():
        js_data.append({
            'id': int(row['id']),
            'group': row['group'],
            'pca_x': float(row['pca_x']),
            'pca_y': float(row['pca_y']),
            'personality': row.get('Personality', None),
            'has_missing': bool(row['has_missing']),
            'time_alone': float(row['Time_spent_Alone']) if pd.notna(row['Time_spent_Alone']) else None,
            'social_events': float(row['Social_event_attendance']) if pd.notna(row['Social_event_attendance']) else None,
            'friends': float(row['Friends_circle_size']) if pd.notna(row['Friends_circle_size']) else None,
            'going_out': float(row['Going_outside']) if pd.notna(row['Going_outside']) else None,
            'post_freq': float(row['Post_frequency']) if pd.notna(row['Post_frequency']) else None,
            'stage_fear': row['Stage_fear'],
            'drained': row['Drained_after_socializing']
        })
    
    # Insert data and variance into HTML
    html_content = html_template.replace('INSERT_DATA_HERE', json.dumps(js_data))
    html_content = html_content.replace('INSERT_PC1_VAR', f"{pca.explained_variance_ratio_[0]:.1%}")
    html_content = html_content.replace('INSERT_PC2_VAR', f"{pca.explained_variance_ratio_[1]:.1%}")
    
    # Save HTML file
    output_file = OUTPUT_DIR / 'interactive_with_copyable_ids.html'
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive visualization saved to: {output_file}")
    
    # Also create a simple CSV for easy lookup
    all_df[['id', 'group', 'Personality', 'has_missing']].to_csv(
        OUTPUT_DIR / 'sample_lookup.csv', index=False
    )
    print(f"Sample lookup CSV saved to: {OUTPUT_DIR / 'sample_lookup.csv'}")

def main():
    create_interactive_with_table()
    
    print("\n" + "="*60)
    print("ENHANCED INTERACTIVE VISUALIZATION CREATED")
    print("="*60)
    print("\nFeatures:")
    print("- Click on any point to see persistent details panel")
    print("- Click on ID numbers to copy them")
    print("- Search by ID using the search box")
    print("- Recent selections are saved in a list")
    print("- All IDs are easily copyable")
    print("\nNo more hover frustration!")

if __name__ == "__main__":
    main()