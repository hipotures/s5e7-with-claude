# Optimization Display - FIXED

## Problem Solved
User reported: "gdy skrypt staruje, pojawia sie Loaded 600 ambiguous training samples a potem przez kilkadziesiąt sekund tylko czarna plansza"

## Solution Implemented

### 1. Added Current Trial Display
- Shows what the system is doing during startup
- Displays loading status before first trial starts
- Updates with fold progress during CV training

### 2. Status Messages
The display now shows:
- "Starting optimization system..." - when script starts
- "Loading dataset..." - during data loading
- "Initializing Optuna studies..." - during study creation
- "Training fold X/5..." - during cross-validation

### 3. Immediate Display Update
- Display starts immediately when script runs
- No more black screen during initialization
- Current trial panel shows system status

## Display Layout
```
┌─────────────────────────────────────────┐
│       Dataset Optimization Summary       │
├─────────────────────────────────────────┤
│         Current Optimization            │
├─────────────────────────────────────────┤
│           Current Trial                 │  <-- NEW: Shows live progress
├─────────────────────────────────────────┤
│         Model Performance               │
├─────────────────────────────────────────┤
│         Recent Trials                   │
└─────────────────────────────────────────┘
```

## How to Run

### Normal Mode (with display):
```bash
python 20250705_1931_optimize_all_corrected_datasets.py
```

### Debug Mode (see all output):
```bash
python 20250705_1931_optimize_all_corrected_datasets.py --debug
```

### Quick Test (15 seconds):
```bash
./run_optimization_test.sh
```

## What You'll See Now

1. **At Startup**:
   - Immediate display with "Starting optimization system..."
   - No more waiting for first results

2. **During Loading**:
   - "Loading dataset..." status
   - "Initializing Optuna studies..." status

3. **During Optimization**:
   - Current model being optimized (XGB/GBM/CAT)
   - Trial number
   - Current fold being trained (1/5, 2/5, etc.)
   - Elapsed time for current trial

4. **After Each Trial**:
   - Trial added to "Recent Trials" table
   - Best scores updated if improved
   - Progress bar shows time remaining

## Technical Changes

1. Added `current_trial_info` tracking to OptimizationTracker
2. Created `create_current_trial_panel()` function
3. Added `start_trial()` calls before optimization
4. Updated display layout to include current trial panel
5. Added system status messages during startup
6. Suppressed print statements that interfered with display

## Notes
- First trial may still take 30-60 seconds (training 5 folds)
- But now you see what's happening during that time
- Display updates every 0.5 seconds
- All Optuna output is suppressed to keep display clean