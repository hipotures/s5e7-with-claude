

import pandas as pd
import os

# Define the paths to the submission files
BASE_PATH = "scores"
BASELINE_FILE = os.path.join(BASE_PATH, "recreated_975708_submission.csv")
CHALLENGER_FILES = [
    os.path.join(BASE_PATH, "subm-0.97582-20250706_040034-xgb-tc07.csv"),
    os.path.join(BASE_PATH, "subm-0.97468-20250706_031620-xgb-tc07.csv"),
    os.path.join(BASE_PATH, "subm-0.97538-20250706_032037-cat-tc07.csv"),
    os.path.join(BASE_PATH, "subm-0.97479-20250706_031622-cat-tc07.csv"),
    os.path.join(BASE_PATH, "subm-0.97576-20250706_035101-gbm-tc07.csv"),
    os.path.join(BASE_PATH, "subm-0.97490-20250706_004053-gbm-tc04.csv"),
]

# Load the baseline submission
baseline_df = pd.read_csv(BASELINE_FILE)
baseline_df.rename(columns={'Personality': 'baseline'}, inplace=True)

# Load challenger submissions and merge them into a single DataFrame
all_dfs = [baseline_df]
for i, file in enumerate(CHALLENGER_FILES):
    challenger_df = pd.read_csv(file)
    challenger_df.rename(columns={'Personality': f'challenger_{i+1}'}, inplace=True)
    all_dfs.append(challenger_df.drop(columns=['id']))

# Concatenate all dataframes
merged_df = pd.concat(all_dfs, axis=1)

# Find disagreements
disagreements = []
for index, row in merged_df.iterrows():
    baseline_pred = row['baseline']
    mismatches = 0
    for i in range(len(CHALLENGER_FILES)):
        if row[f'challenger_{i+1}'] != baseline_pred:
            mismatches += 1
    
    if mismatches > 0:
        disagreements.append({
            'id': row['id'],
            'baseline_prediction': baseline_pred,
            'mismatches': mismatches,
            'challenger_preds': [row[f'challenger_{i+1}'] for i in range(len(CHALLENGER_FILES))]
        })

# Create a DataFrame from the disagreements
disagreements_df = pd.DataFrame(disagreements)

# Sort by the number of mismatches
disagreements_df.sort_values(by='mismatches', ascending=False, inplace=True)

# Print the results
print("Analysis of Submission Disagreements")
print("=====================================")
print(f"Baseline file: {BASELINE_FILE}")
print(f"Challenger files: {len(CHALLENGER_FILES)}")
print(f"Total disagreements found: {len(disagreements_df)}")
print("\nTop 20 most disagreed upon samples:")
print(disagreements_df.head(20).to_string())

# Save the full list of disagreements to a file
output_path = os.path.join(BASE_PATH, "disagreement_analysis.csv")
disagreements_df.to_csv(output_path, index=False)
print(f"\nFull disagreement report saved to: {output_path}")

