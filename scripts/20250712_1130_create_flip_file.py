import pandas as pd
import os

# Define file paths
BASE_PATH = "scores"
BASELINE_FILE = os.path.join(BASE_PATH, "recreated_975708_submission.csv")
DISAGREEMENT_FILE = os.path.join(BASE_PATH, "disagreement_analysis.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "flip_DISAGREEMENT_20_E2I.csv")

# Load the baseline submission file
submission_df = pd.read_csv(BASELINE_FILE)

# Load the disagreement analysis
disagreement_df = pd.read_csv(DISAGREEMENT_FILE)

# Get the top 20 IDs to flip
ids_to_flip = disagreement_df.head(20)['id'].tolist()

# Create a copy of the submission to modify
flipped_submission_df = submission_df.copy()

# Flip the 'Personality' for the selected IDs from Extrovert to Introvert
# We are assuming the baseline for these is 'Extrovert' as per our analysis
flipped_submission_df.loc[flipped_submission_df['id'].isin(ids_to_flip), 'Personality'] = 'Introvert'

# Save the new submission file
flipped_submission_df.to_csv(OUTPUT_FILE, index=False)

print(f"Successfully created '{OUTPUT_FILE}' with {len(ids_to_flip)} flipped predictions.")
