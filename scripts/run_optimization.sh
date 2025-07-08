#!/bin/bash
# Script to run optimization with different modes

echo "Optimization Runner"
echo "=================="
echo "1. Debug mode (no rich display, see all errors)"
echo "2. Normal mode (rich display)"
echo "3. Quick test (5 seconds per dataset)"
echo ""

if [ "$1" == "debug" ]; then
    echo "Running in DEBUG mode..."
    python 20250705_1931_optimize_all_corrected_datasets.py --debug
elif [ "$1" == "test" ]; then
    echo "Running QUICK TEST (5s per dataset)..."
    # Temporarily modify TIME_PER_DATASET
    sed -i 's/TIME_PER_DATASET = 30/TIME_PER_DATASET = 5/' 20250705_1931_optimize_all_corrected_datasets.py
    python 20250705_1931_optimize_all_corrected_datasets.py
    # Restore original value
    sed -i 's/TIME_PER_DATASET = 5/TIME_PER_DATASET = 30/' 20250705_1931_optimize_all_corrected_datasets.py
else
    echo "Running in NORMAL mode..."
    python 20250705_1931_optimize_all_corrected_datasets.py
fi