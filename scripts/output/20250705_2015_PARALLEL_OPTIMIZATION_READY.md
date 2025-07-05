# Parallel Optimization - READY TO RUN

## Key Difference: PARALLEL vs SEQUENTIAL

### Sequential (old script: 20250705_1931)
- Models run one after another: XGB → GBM → CAT → XGB → ...
- Only one GPU/CPU used at a time
- ~3x slower

### Parallel (new script: 20250705_2015) 
- All models run simultaneously:
  - XGB on GPU0
  - GBM on GPU1  
  - CAT on CPU
- Full resource utilization
- ~3x faster optimization

## Why Parallel is Better

1. **Full GPU Utilization**
   - Both GPUs work simultaneously
   - CPU also runs CatBoost in parallel
   
2. **3x More Trials**
   - In 1 hour: ~300 trials vs ~100 trials
   - Better hyperparameter exploration
   
3. **True Resource Assignment**
   ```python
   MODEL_RESOURCE_ASSIGNMENT = {
       'xgb': ['gpu0', 'gpu1'],
       'gbm': ['gpu0', 'gpu1'], 
       'cat': ['cpu']
   }
   ```

## How It Works

1. **Worker Processes**
   - 3 worker processes start (one per resource)
   - Each worker pulls tasks from queue
   - Workers set CUDA_VISIBLE_DEVICES correctly

2. **Smart Scheduling**
   - Scheduler assigns models to free resources
   - XGB/GBM alternate between GPU0 and GPU1
   - CAT always runs on CPU

3. **Live Display**
   ```
   ┌─────────────────────────────────────────┐
   │     Dataset Optimization Summary         │
   ├─────────────────────────────────────────┤
   │        Current Optimization             │
   ├─────────────────────────────────────────┤
   │ Resource Status │  Running Tasks        │
   │ 🟢 gpu0: idle   │  XGB trial 42 (15s)   │
   │ 🔴 gpu1: XGB 42 │  GBM trial 38 (23s)   │
   │ 🔴 cpu: CAT 55  │  CAT trial 55 (8s)    │
   ├─────────────────────────────────────────┤
   │         Recent Trials                   │
   └─────────────────────────────────────────┘
   ```

## How to Run

### Production (1 hour per dataset):
```bash
python 20250705_2015_optimize_parallel_corrected_datasets.py
```

### Quick Test (edit TIME_PER_DATASET):
```bash
# Change line 43 to: TIME_PER_DATASET = 300  # 5 minutes
python 20250705_2015_optimize_parallel_corrected_datasets.py
```

### Monitor GPU Usage:
```bash
# In another terminal:
watch -n 1 nvidia-smi
```

## What You'll See

1. **Startup**: "Started 3 worker processes"
2. **Resource Status**: Shows which model is on which resource
3. **Running Tasks**: Live view of all parallel tasks
4. **Simultaneous Updates**: Results from all 3 models

## Technical Implementation

1. **Multiprocessing Queue System**
   - `task_queue`: Scheduler → Workers
   - `result_queue`: Workers → Scheduler

2. **Worker Process**
   - Sets GPU environment (CUDA_VISIBLE_DEVICES)
   - Loads dataset independently
   - Trains model and returns results

3. **Parallel Scheduler**
   - Monitors resource availability
   - Creates tasks with Optuna trials
   - Processes results asynchronously

## Output Files

Same as sequential version:
- `scores/subm-*.csv` - Best submissions
- `output/optimization_logs/` - Detailed logs
- `output/optuna_studies/*.db` - Optimization history
- `output/parallel_optimization_summary_*.txt` - Final report

## Important Notes

1. **Memory Usage**: 3 models training simultaneously = 3x memory
2. **GPU Assignment**: Automatic via CUDA_VISIBLE_DEVICES
3. **Crash Recovery**: SQLite databases persist trials
4. **Debug Mode**: Not implemented (use logs instead)

## Expected Performance

With parallel execution on 8 datasets × 1 hour each:
- XGB: ~100 trials per dataset
- GBM: ~100 trials per dataset  
- CAT: ~100 trials per dataset
- **Total**: ~2400 trials vs ~800 sequential

## Commands for GPU Server

```bash
# Navigate to scripts directory
cd /mnt/ml/competitions/2025/playground-series-s5e7/WORKSPACE/scripts/

# Run in background with logging
nohup python 20250705_2015_optimize_parallel_corrected_datasets.py > parallel_optimization.log 2>&1 &

# Monitor progress
tail -f parallel_optimization.log

# Check GPU usage
nvidia-smi -l 1

# Check running processes
ps aux | grep python
```