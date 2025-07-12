#!/usr/bin/env python3
"""Test the parallel display fixes"""

import subprocess
import time
import sys

print("Testing parallel optimization display...")
print("This will run for 30 seconds to check:")
print("1. Full dataset names in Recent Trials")
print("2. Resource status panel")
print("3. Running tasks panel")
print("\nStarting in 3 seconds...\n")
time.sleep(3)

# Create a temporary version with 30 second timeout
with open('20250705_2015_optimize_parallel_corrected_datasets.py', 'r') as f:
    content = f.read()

# Replace TIME_PER_DATASET
modified = content.replace('TIME_PER_DATASET = 3600', 'TIME_PER_DATASET = 30')

with open('test_parallel_30s.py', 'w') as f:
    f.write(modified)

# Run the test
try:
    proc = subprocess.Popen([sys.executable, 'test_parallel_30s.py'])
    proc.wait()
except KeyboardInterrupt:
    print("\nTest interrupted")
    proc.terminate()
finally:
    # Clean up
    import os
    if os.path.exists('test_parallel_30s.py'):
        os.remove('test_parallel_30s.py')
    print("\nTest complete!")