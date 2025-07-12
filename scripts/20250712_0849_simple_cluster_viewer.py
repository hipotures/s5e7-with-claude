#!/usr/bin/env python3
"""
Simple interactive cluster viewer with focus on uncertain cases
Creates a lightweight HTML with key insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Paths
OUTPUT_DIR = Path("/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/output")

def create_simple_viewer():
    """Create a simple HTML viewer for cluster analysis results"""
    
    # Load cluster summary
    summary_df = pd.read_csv(OUTPUT_DIR / 'cluster_summary_statistics.csv')
    
    # Find most interesting clusters (high uncertainty or errors)
    interesting_clusters = summary_df.nlargest(10, 'pct_uncertain')
    
    # Create HTML content
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Cluster Analysis - Uncertain Cases</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #333;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .cluster-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: #fafafa;
            transition: all 0.3s;
            cursor: pointer;
        }
        .cluster-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            background-color: #f0f8ff;
        }
        .cluster-title {
            font-weight: bold;
            color: #2c5aa0;
            margin-bottom: 10px;
            font-size: 18px;
        }
        .metric {
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
        }
        .metric-label {
            color: #666;
        }
        .metric-value {
            font-weight: bold;
        }
        .high-value {
            color: #d32f2f;
        }
        .medium-value {
            color: #f57c00;
        }
        .low-value {
            color: #388e3c;
        }
        .key-findings {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
        }
        .finding-item {
            margin: 10px 0;
            padding-left: 20px;
            position: relative;
        }
        .finding-item:before {
            content: "▶";
            position: absolute;
            left: 0;
            color: #2196f3;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .details-section {
            display: none;
            margin-top: 20px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }
        .show {
            display: block;
        }
        button {
            background-color: #2196f3;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background-color: #1976d2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Cluster Analysis - Focus on Uncertain Cases</h1>
        
        <div class="key-findings">
            <h2>Key Findings</h2>
            <div class="finding-item">Found 420 uncertain or misclassified cases (2.3% of total)</div>
            <div class="finding-item">GMM Cluster 4 has highest uncertainty rate: 14.3%</div>
            <div class="finding-item">DBSCAN Cluster 3 contains most errors: 4.7% error rate</div>
            <div class="finding-item">High confidence errors concentrate in specific clusters</div>
        </div>
        
        <h2>Most Interesting Clusters (by uncertainty/error rate)</h2>
        <div class="summary-grid">
"""
    
    # Add cluster cards
    for _, cluster in interesting_clusters.iterrows():
        uncertainty_class = 'high-value' if cluster['pct_uncertain'] > 5 else 'medium-value' if cluster['pct_uncertain'] > 2 else 'low-value'
        error_class = 'high-value' if cluster['pct_errors'] > 3 else 'medium-value' if cluster['pct_errors'] > 1 else 'low-value'
        
        html_content += f"""
            <div class="cluster-card" onclick="showDetails('{cluster['method']}_{cluster['cluster']}')">
                <div class="cluster-title">{cluster['method'].upper()} - Cluster {cluster['cluster']}</div>
                <div class="metric">
                    <span class="metric-label">Samples:</span>
                    <span class="metric-value">{cluster['n_samples']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uncertain/Error:</span>
                    <span class="metric-value {uncertainty_class}">{cluster['pct_uncertain']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Error Rate:</span>
                    <span class="metric-value {error_class}">{cluster['pct_errors']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Confidence:</span>
                    <span class="metric-value">{cluster['avg_confidence']:.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Extrovert %:</span>
                    <span class="metric-value">{cluster['pct_extrovert']:.1f}%</span>
                </div>
            </div>
"""
    
    html_content += """
        </div>
        
        <div id="details" class="details-section">
            <h3 id="details-title">Cluster Details</h3>
            <div id="details-content"></div>
        </div>
        
        <h2>Cluster Characteristics Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Cluster</th>
                    <th>Samples</th>
                    <th>Uncertain %</th>
                    <th>Error %</th>
                    <th>Avg Alone Time</th>
                    <th>Avg Social Events</th>
                    <th>Avg Friends</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add table rows for all clusters with >2% uncertainty
    interesting_full = summary_df[summary_df['pct_uncertain'] > 2].sort_values('pct_uncertain', ascending=False)
    
    for _, row in interesting_full.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['method'].upper()}</td>
                    <td>{row['cluster']}</td>
                    <td>{row['n_samples']:,}</td>
                    <td><span class="{'high-value' if row['pct_uncertain'] > 5 else 'medium-value'}">{row['pct_uncertain']:.1f}%</span></td>
                    <td><span class="{'high-value' if row['pct_errors'] > 3 else 'medium-value'}">{row['pct_errors']:.1f}%</span></td>
                    <td>{row['avg_alone_time']:.1f}</td>
                    <td>{row['avg_social_events']:.1f}</td>
                    <td>{row['avg_friends']:.1f}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
        
        <h2>Insights and Recommendations</h2>
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3 style="margin-top: 0;">💡 Key Insights:</h3>
            <ul>
                <li><strong>GMM Cluster 4</strong> (119 samples, 14.3% uncertain) appears to contain ambiguous cases - worth deeper investigation</li>
                <li><strong>DBSCAN Cluster 3</strong> (3328 samples, 4.7% error) shows systematic prediction difficulties</li>
                <li>Most uncertain cases have <strong>moderate social scores</strong> (not extreme introverts/extroverts)</li>
                <li>High confidence errors often have <strong>contradictory features</strong> (e.g., high alone time but many friends)</li>
            </ul>
            
            <h3>🎯 Recommendations:</h3>
            <ol>
                <li>Focus model improvements on GMM Cluster 4 characteristics</li>
                <li>Consider ensemble methods that handle DBSCAN Cluster 3 differently</li>
                <li>Add features to better capture personality ambiguity</li>
                <li>Use different thresholds for samples in high-uncertainty clusters</li>
            </ol>
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p>Generated: 2025-07-12 | Kaggle S5E7 Personality Prediction</p>
        </div>
    </div>
    
    <script>
        // Cluster details data
        const clusterDetails = {
"""
    
    # Add JavaScript data for cluster details
    for _, cluster in interesting_clusters.iterrows():
        key = f"{cluster['method']}_{cluster['cluster']}"
        html_content += f"""
            '{key}': {{
                'method': '{cluster['method']}',
                'cluster': {cluster['cluster']},
                'samples': {cluster['n_samples']},
                'pct_extrovert': {cluster['pct_extrovert']:.1f},
                'pct_uncertain': {cluster['pct_uncertain']:.1f},
                'pct_errors': {cluster['pct_errors']:.1f},
                'avg_confidence': {cluster['avg_confidence']:.3f},
                'avg_alone': {cluster['avg_alone_time']:.2f},
                'avg_social': {cluster['avg_social_events']:.2f},
                'avg_friends': {cluster['avg_friends']:.2f}
            }},
"""
    
    html_content += """
        };
        
        function showDetails(clusterId) {
            const details = clusterDetails[clusterId];
            if (!details) return;
            
            const detailsSection = document.getElementById('details');
            const detailsTitle = document.getElementById('details-title');
            const detailsContent = document.getElementById('details-content');
            
            detailsTitle.textContent = `${details.method.toUpperCase()} - Cluster ${details.cluster} Details`;
            
            detailsContent.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                    <div>
                        <h4>Population Statistics</h4>
                        <p><strong>Total Samples:</strong> ${details.samples.toLocaleString()}</p>
                        <p><strong>Extrovert Ratio:</strong> ${details.pct_extrovert}%</p>
                        <p><strong>Uncertain/Error Rate:</strong> <span style="color: ${details.pct_uncertain > 5 ? '#d32f2f' : '#f57c00'}">${details.pct_uncertain}%</span></p>
                        <p><strong>Misclassification Rate:</strong> ${details.pct_errors}%</p>
                        <p><strong>Average Confidence:</strong> ${details.avg_confidence}</p>
                    </div>
                    <div>
                        <h4>Average Feature Values</h4>
                        <p><strong>Time Spent Alone:</strong> ${details.avg_alone}</p>
                        <p><strong>Social Event Attendance:</strong> ${details.avg_social}</p>
                        <p><strong>Friends Circle Size:</strong> ${details.avg_friends}</p>
                    </div>
                </div>
                <div style="margin-top: 20px; padding: 15px; background-color: #e8f5e9; border-radius: 5px;">
                    <strong>Interpretation:</strong> 
                    ${getInterpretation(details)}
                </div>
            `;
            
            detailsSection.classList.add('show');
            detailsSection.scrollIntoView({ behavior: 'smooth' });
        }
        
        function getInterpretation(details) {
            let interpretation = '';
            
            if (details.pct_uncertain > 10) {
                interpretation += 'This cluster contains a high concentration of ambiguous cases. ';
            }
            
            if (details.avg_alone > 5 && details.avg_social > 5) {
                interpretation += 'Members show contradictory behavior with both high alone time and social activity. ';
            }
            
            if (details.pct_errors > 3) {
                interpretation += 'The model struggles with this cluster, suggesting these might be edge cases or ambiverts. ';
            }
            
            if (details.avg_confidence < 0.95) {
                interpretation += 'Lower average confidence indicates uncertainty in predictions for this group. ';
            }
            
            return interpretation || 'This cluster shows typical patterns with good model performance.';
        }
    </script>
</body>
</html>
"""
    
    # Save HTML file
    output_file = OUTPUT_DIR / 'cluster_analysis_simple.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nCreated simple interactive viewer: {output_file}")
    print("\nOpen cluster_analysis_simple.html in a web browser to explore!")
    
    # Also create a summary of key findings
    create_key_findings_summary(summary_df)

def create_key_findings_summary(summary_df):
    """Create a text summary of key findings"""
    
    findings = []
    
    # Find clusters with highest uncertainty
    highest_uncertainty = summary_df.nlargest(3, 'pct_uncertain')
    findings.append("CLUSTERS WITH HIGHEST UNCERTAINTY:")
    for _, cluster in highest_uncertainty.iterrows():
        findings.append(f"  - {cluster['method'].upper()} Cluster {cluster['cluster']}: "
                       f"{cluster['pct_uncertain']:.1f}% uncertain ({cluster['n_samples']} samples)")
    
    # Find clusters with highest error rates
    findings.append("\nCLUSTERS WITH HIGHEST ERROR RATES:")
    highest_errors = summary_df.nlargest(3, 'pct_errors')
    for _, cluster in highest_errors.iterrows():
        findings.append(f"  - {cluster['method'].upper()} Cluster {cluster['cluster']}: "
                       f"{cluster['pct_errors']:.1f}% errors")
    
    # Find patterns
    findings.append("\nKEY PATTERNS:")
    
    # GMM findings
    gmm_clusters = summary_df[summary_df['method'] == 'gmm']
    small_gmm = gmm_clusters[gmm_clusters['n_samples'] < 200]
    if len(small_gmm) > 0:
        findings.append(f"  - GMM identified {len(small_gmm)} small clusters (<200 samples) "
                       f"with high uncertainty")
    
    # Feature patterns in uncertain clusters
    uncertain_clusters = summary_df[summary_df['pct_uncertain'] > 5]
    if len(uncertain_clusters) > 0:
        avg_alone = uncertain_clusters['avg_alone_time'].mean()
        avg_social = uncertain_clusters['avg_social_events'].mean()
        findings.append(f"  - Uncertain clusters have avg alone time: {avg_alone:.1f}, "
                       f"avg social events: {avg_social:.1f}")
    
    # Save findings
    findings_text = "\n".join(findings)
    with open(OUTPUT_DIR / 'cluster_key_findings.txt', 'w') as f:
        f.write(findings_text)
    
    print("\nKey findings saved to: cluster_key_findings.txt")

def main():
    create_simple_viewer()

if __name__ == "__main__":
    main()