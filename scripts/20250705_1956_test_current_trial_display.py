#!/usr/bin/env python3
"""Test the current trial display"""

import time
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

# Import the display functions from the main script
import sys
sys.path.append('.')

# Create mock tracker with current trial info
class MockTracker:
    def __init__(self):
        self.current_trial_info = {
            'model': 'xgb',
            'trial': 42,
            'start_time': time.time(),
            'status': 'Running...',
            'dataset': 'train_corrected_01.csv'
        }
        self.results = {}
        self.recent_trials = []
        
    def get_display_data(self):
        return {
            'current_dataset': 'train_corrected_01.csv',
            'elapsed_time': 120,
            'dataset_elapsed': 60,
            'time_remaining': 3540,
            'results': {},
            'recent_trials': []
        }

def create_current_trial_panel(tracker):
    """Create panel showing current running trial"""
    if not tracker.current_trial_info:
        return Panel("No active trial", title="Current Trial", height=4)
    
    info = tracker.current_trial_info
    elapsed = time.time() - info['start_time']
    
    content = f"""Model: {info['model'].upper()}
Trial: #{info['trial']}
Status: {info['status']}
Elapsed: {elapsed:.1f}s"""
    
    return Panel(content, title="Current Trial", height=6)

# Test the display
console = Console()
tracker = MockTracker()

print("Testing current trial display...")
print("="*60)

# Show the panel
panel = create_current_trial_panel(tracker)
console.print(panel)

# Test with no trial
print("\nWith no active trial:")
tracker.current_trial_info = None
panel = create_current_trial_panel(tracker)
console.print(panel)

# Test with updating trial
print("\nSimulating live update:")
tracker.current_trial_info = {
    'model': 'cat',
    'trial': 123,
    'start_time': time.time(),
    'status': 'Training fold 3/5...',
    'dataset': 'train_corrected_05.csv'
}

with Live(console=console, refresh_per_second=2) as live:
    for i in range(5):
        elapsed = time.time() - tracker.current_trial_info['start_time']
        tracker.current_trial_info['status'] = f'Training fold {i+1}/5...'
        
        content = f"""Model: {tracker.current_trial_info['model'].upper()}
Trial: #{tracker.current_trial_info['trial']}  
Status: {tracker.current_trial_info['status']}
Elapsed: {elapsed:.1f}s"""
        
        panel = Panel(content, title=f"Current Trial (Update {i+1})", height=6)
        live.update(panel)
        time.sleep(1)
        
print("\nTest complete!")