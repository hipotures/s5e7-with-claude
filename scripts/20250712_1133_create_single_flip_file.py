import pandas as pd
import os

# Define file paths
BASE_PATH = "scores"
BASELINE_FILE = os.path.join(BASE_PATH, "recreated_975708_submission.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "flip_DISAGREEMENT_1_E2I_id_24677.csv")
ID_TO_FLIP = 24677

# Load the baseline submission file
submission_df = pd.read_csv(BASELINE_FILE)

# Create a copy of the submission to modify
flipped_submission_df = submission_df.copy()

# Flip the 'Personality' for the selected ID from Extrovert to Introvert
flipped_submission_df.loc[flipped_submission_df['id'] == ID_TO_FLIP, 'Personality'] = 'Introvert'

# Save the new submission file
flipped_submission_df.to_csv(OUTPUT_FILE, index=False)

print(f"Successfully created '{OUTPUT_FILE}' with a single flipped prediction for id {ID_TO_FLIP}.")
