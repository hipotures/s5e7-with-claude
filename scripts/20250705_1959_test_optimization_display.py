#!/usr/bin/env python3
"""Quick test of optimization with display"""

import subprocess
import time
import signal
import sys

def test_optimization():
    """Run optimization for 15 seconds to test display"""
    print("Testing optimization display for 15 seconds...")
    print("You should see:")
    print("1. Initial 'Starting optimization system...' message")
    print("2. 'Loading dataset...' status")
    print("3. 'Initializing Optuna studies...' status")
    print("4. Current trial updates with fold progress")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    # Temporarily set TIME_PER_DATASET to 15 seconds
    with open('20250705_1931_optimize_all_corrected_datasets.py', 'r') as f:
        content = f.read()
    
    # Replace TIME_PER_DATASET
    modified = content.replace('TIME_PER_DATASET = 30', 'TIME_PER_DATASET = 15')
    
    with open('20250705_1931_optimize_all_corrected_datasets_test.py', 'w') as f:
        f.write(modified)
    
    # Run the test
    try:
        proc = subprocess.Popen(['python', '20250705_1931_optimize_all_corrected_datasets_test.py'])
        proc.wait()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        proc.terminate()
    finally:
        # Clean up
        import os
        if os.path.exists('20250705_1931_optimize_all_corrected_datasets_test.py'):
            os.remove('20250705_1931_optimize_all_corrected_datasets_test.py')

if __name__ == "__main__":
    test_optimization()