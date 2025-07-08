#!/bin/bash
# Quick test of optimization display

echo "Running optimization display test..."
echo "This will run for 15 seconds to show the display updates"
echo ""

# Set very short time for testing
sed -i 's/TIME_PER_DATASET = 30/TIME_PER_DATASET = 15/' 20250705_1931_optimize_all_corrected_datasets.py

# Run the script
timeout 20s python 20250705_1931_optimize_all_corrected_datasets.py

# Restore original time
sed -i 's/TIME_PER_DATASET = 15/TIME_PER_DATASET = 30/' 20250705_1931_optimize_all_corrected_datasets.py

echo ""
echo "Test complete!"