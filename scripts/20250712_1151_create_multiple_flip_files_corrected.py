import pandas as pd
import os

# Define file paths
BASE_PATH = "/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/subm/20250705"
SCORES_PATH = "/mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scores"
BASELINE_FILE = os.path.join(BASE_PATH, "subm-0.96950-20250704_121621-xgb-381482788433-148.csv")
DISAGREEMENT_FILE = os.path.join(SCORES_PATH, "disagreement_analysis_corrected.csv")

# Load the baseline submission file
submission_df = pd.read_csv(BASELINE_FILE)

# Load the disagreement analysis
disagreement_df = pd.read_csv(DISAGREEMENT_FILE)

# Get the top 20 IDs to flip
top_20_candidates = disagreement_df.head(20)

print(f"Preparing to create 20 individual flip files based on {BASELINE_FILE}...")

# Loop through the top 20 candidates and create a separate file for each
for index, row in top_20_candidates.iterrows():
    an_id = row['id']
    baseline_pred = row['baseline_prediction']
    # Determine the flip direction. If baseline is Extrovert, flip to Introvert, and vice-versa.
    flipped_pred = 'Introvert' if baseline_pred == 'Extrovert' else 'Extrovert'
    direction = f"{baseline_pred[0]}2{flipped_pred[0]}"

    # Define the output file name based on the ID and direction
    output_file = os.path.join(SCORES_PATH, f"flip_CORRECTED_DISAGREEMENT_1_{direction}_id_{an_id}.csv")
    
    # Create a fresh copy of the original submission
    flipped_submission_df = submission_df.copy()

    # Flip the 'Personality' for the current ID
    flipped_submission_df.loc[flipped_submission_df['id'] == an_id, 'Personality'] = flipped_pred
    
    # Save the new submission file
    flipped_submission_df.to_csv(output_file, index=False)
    print(f"Successfully created: {output_file}")

print("\nFinished creating all 20 corrected flip files.")
