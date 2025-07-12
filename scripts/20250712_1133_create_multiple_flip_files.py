import pandas as pd
import os

# Define file paths
BASE_PATH = "scores"
BASELINE_FILE = os.path.join(BASE_PATH, "recreated_975708_submission.csv")
DISAGREEMENT_FILE = os.path.join(BASE_PATH, "disagreement_analysis.csv")

# Load the baseline submission file
submission_df = pd.read_csv(BASELINE_FILE)

# Load the disagreement analysis
disagreement_df = pd.read_csv(DISAGREEMENT_FILE)

# Get the top 20 IDs to flip
top_20_ids_to_flip = disagreement_df.head(20)['id'].tolist()

print(f"Preparing to create 20 individual flip files...")

# Loop through the top 20 IDs and create a separate file for each
for an_id in top_20_ids_to_flip:
    # Define the output file name based on the ID
    output_file = os.path.join(BASE_PATH, f"flip_DISAGREEMENT_1_E2I_id_{an_id}.csv")
    
    # Create a fresh copy of the original submission
    flipped_submission_df = submission_df.copy()

    # Flip the 'Personality' for the current ID
    flipped_submission_df.loc[flipped_submission_df['id'] == an_id, 'Personality'] = 'Introvert'
    
    # Save the new submission file
    flipped_submission_df.to_csv(output_file, index=False)
    print(f"Successfully created: {output_file}")

print("\nFinished creating all 20 flip files.")
