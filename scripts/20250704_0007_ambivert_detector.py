"""
PURPOSE: Implement a comprehensive AmbivertHandler class to detect and handle ambiverts
using multiple strategies including special marker values, confusion zones, and clustering

HYPOTHESIS: Ambiverts can be detected through: (1) special marker values in features,
(2) samples in confusion zones between classes, (3) distance to cluster centroids,
and (4) model prediction uncertainty

EXPECTED: Identify ~2.43% of samples as ambiverts and provide multiple strategies for
handling them to potentially break the 0.975708 accuracy ceiling

RESULT: Created a flexible framework for ambivert detection with multiple scoring
mechanisms and handling strategies. Key finding: special marker values (e.g., 
Time_spent_Alone=3.1377639321564557) may indicate synthetic ambivert samples
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class AmbivertHandler:
    """
    A class to detect and handle ambiverts in personality prediction
    """
    
    def __init__(self):
        self.special_values = {
            'Time_spent_Alone': 3.1377639321564557,
            'Social_event_attendance': 5.265106088560886,
            'Going_outside': 4.044319380935631,
            'Post_frequency': 4.982097334878332
        }
        
        # Cluster centroids from analysis
        self.centroids = {
            0: {'Time_spent_Alone': 10.000, 'Stage_fear': 0.000, 
                'Social_event_attendance': 3.000, 'Going_outside': 3.000,
                'Drained_after_socializing': 0.000, 'Friends_circle_size': 5.000,
                'Post_frequency': 3.000},
            1: {'Time_spent_Alone': 1.664, 'Stage_fear': 0.058,
                'Social_event_attendance': 5.688, 'Going_outside': 4.999,
                'Drained_after_socializing': 0.031, 'Friends_circle_size': 9.764,
                'Post_frequency': 6.123},
            2: {'Time_spent_Alone': 1.968, 'Stage_fear': 0.045,
                'Social_event_attendance': 5.301, 'Going_outside': 4.913,
                'Drained_after_socializing': 0.023, 'Friends_circle_size': 9.995,
                'Post_frequency': 6.447}
        }
        
        self.confusion_zone = {
            'Time_spent_Alone': (0, 4),
            'Social_event_attendance': (4, 6),
            'Going_outside': (3, 7),
            'Post_frequency': (3, 10),
            'Friends_circle_size': (9, 10),
            'Stage_fear': (0.0, 0.5),
            'Drained_after_socializing': (0.0, 0.5)
        }
    
    def detect_ambiverts(self, df, model_probs=None, uncertainty_threshold=0.015):
        """
        Detect potential ambiverts using multiple strategies
        
        Args:
            df: DataFrame with personality features
            model_probs: Optional model prediction probabilities
            uncertainty_threshold: Threshold for uncertainty-based detection
            
        Returns:
            DataFrame with ambivert scores and flags
        """
        df = df.copy()
        
        # Strategy 1: Special marker values
        df['has_special_values'] = 0
        for col, val in self.special_values.items():
            if col in df.columns:
                df['has_special_values'] += (df[col] == val).astype(int)
        
        # Strategy 2: Confusion zone detection
        df['in_confusion_zone'] = 0
        for col, (min_val, max_val) in self.confusion_zone.items():
            if col in df.columns:
                df['in_confusion_zone'] += ((df[col] >= min_val) & (df[col] <= max_val)).astype(int)
        df['confusion_zone_score'] = df['in_confusion_zone'] / len(self.confusion_zone)
        
        # Strategy 3: Distance to centroids
        feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
                       'Post_frequency']
        
        for cluster_id, centroid in self.centroids.items():
            distances = []
            for _, row in df.iterrows():
                dist = np.sqrt(sum((row[col] - centroid[col])**2 for col in feature_cols if col in df.columns))
                distances.append(dist)
            df[f'dist_to_cluster_{cluster_id}'] = distances
        
        # Find minimum distance to any ambivert cluster
        dist_cols = [f'dist_to_cluster_{i}' for i in [1, 2]]  # Clusters 1 and 2 are main ambivert clusters
        df['min_dist_to_ambivert'] = df[dist_cols].min(axis=1)
        
        # Strategy 4: Model uncertainty (if provided)
        if model_probs is not None:
            df['model_uncertainty'] = 1 - np.max(model_probs, axis=1)
            df['uncertain_prediction'] = df['model_uncertainty'] > uncertainty_threshold
        else:
            df['uncertain_prediction'] = False
        
        # Combined ambivert score
        df['ambivert_score'] = (
            0.3 * (df['has_special_values'] > 0).astype(int) +
            0.3 * df['confusion_zone_score'] +
            0.2 * (df['min_dist_to_ambivert'] < 5).astype(int) +
            0.2 * df['uncertain_prediction'].astype(int)
        )
        
        # Flag likely ambiverts
        df['is_likely_ambivert'] = df['ambivert_score'] > 0.5
        
        return df
    
    def apply_ambivert_strategy(self, predictions, ambivert_flags, strategy='soft_mapping'):
        """
        Apply different strategies to handle ambiverts
        
        Args:
            predictions: Original binary predictions (0 or 1)
            ambivert_flags: Boolean array indicating ambiverts
            strategy: 'soft_mapping', 'random_split', 'majority_extrovert', 'pattern_based'
            
        Returns:
            Modified predictions
        """
        predictions = predictions.copy()
        
        if strategy == 'soft_mapping':
            # Use a probabilistic approach for ambiverts
            # This is where you'd use the actual probabilities
            pass
            
        elif strategy == 'random_split':
            # Randomly assign ambiverts (maintaining ~85% extrovert ratio seen in training)
            ambivert_indices = np.where(ambivert_flags)[0]
            np.random.seed(42)
            extrovert_ratio = 0.85
            n_extroverts = int(len(ambivert_indices) * extrovert_ratio)
            random_indices = np.random.choice(ambivert_indices, n_extroverts, replace=False)
            predictions[random_indices] = 1  # Extrovert
            
        elif strategy == 'majority_extrovert':
            # Assign most ambiverts as extroverts (based on training data pattern)
            predictions[ambivert_flags] = 1
            
        elif strategy == 'pattern_based':
            # Use specific patterns to decide
            # This would require the full feature data
            pass
            
        return predictions


# Example usage function
def demonstrate_ambivert_handling():
    """
    Demonstrate how to use the AmbivertHandler
    """
    print("=== AMBIVERT HANDLING DEMONSTRATION ===\n")
    
    # Load potential ambiverts for testing
    ambiverts_df = pd.read_csv('potential_ambiverts.csv')
    
    # Initialize handler
    handler = AmbivertHandler()
    
    # Detect ambiverts
    result_df = handler.detect_ambiverts(ambiverts_df)
    
    print("1. AMBIVERT DETECTION RESULTS:")
    print(f"   Total records: {len(result_df)}")
    print(f"   Likely ambiverts: {result_df['is_likely_ambivert'].sum()} ({result_df['is_likely_ambivert'].sum()/len(result_df)*100:.1f}%)")
    print(f"   Has special values: {(result_df['has_special_values'] > 0).sum()}")
    print(f"   High confusion zone score: {(result_df['confusion_zone_score'] > 0.7).sum()}")
    
    print("\n2. AMBIVERT SCORE DISTRIBUTION:")
    print(result_df['ambivert_score'].describe())
    
    print("\n3. TOP AMBIVERTS BY SCORE:")
    top_ambiverts = result_df.nlargest(10, 'ambivert_score')[['id', 'Personality', 'ambivert_score', 'has_special_values', 'confusion_zone_score']]
    print(top_ambiverts)
    
    print("\n4. IMPLEMENTATION STEPS:")
    print("   Step 1: Train your model and get prediction probabilities")
    print("   Step 2: Use AmbivertHandler.detect_ambiverts() on test data")
    print("   Step 3: Apply different strategies to ambiverts")
    print("   Step 4: Ensemble multiple strategies for best results")
    
    print("\n5. SUGGESTED ENSEMBLE APPROACH:")
    print("   - Create 3-5 different predictions using different ambivert strategies")
    print("   - Weight them based on validation performance")
    print("   - Special handling for records with special marker values")
    print("   - Consider creating a 'confidence-based' submission")
    
    return handler, result_df

if __name__ == "__main__":
    handler, results = demonstrate_ambivert_handling()