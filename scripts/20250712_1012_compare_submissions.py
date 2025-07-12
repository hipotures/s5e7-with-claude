import pandas as pd

baseline = pd.read_csv('/mnt/ml/competitions/2025/playground-series-s5e7/scores/recreated_975708_submission.csv')
combined = pd.read_csv('output/submission_combined_data_ensemble.csv')

# Compare
diff = (baseline['Personality'] != combined['Personality']).sum()
print(f'Różnice od baseline: {diff} ({diff/len(baseline)*100:.2f}%)')

# Show some differences
diff_ids = baseline[baseline['Personality'] != combined['Personality']]['id'].head(20)
print('\nPierwsze 20 różnic:')
for id in diff_ids:
    b = baseline[baseline['id']==id]['Personality'].iloc[0]
    c = combined[combined['id']==id]['Personality'].iloc[0]
    print(f'ID {id}: {b} → {c}')

# Check distribution
print('\nRozkład predykcji:')
print('Baseline:', baseline['Personality'].value_counts(normalize=True))
print('Combined:', combined['Personality'].value_counts(normalize=True))