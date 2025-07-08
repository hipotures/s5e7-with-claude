# Resource Optimization Configuration

## CPU Thread Management

### Problem
When running parallel optimization:
- XGBoost on GPU0
- LightGBM on GPU1  
- CatBoost on CPU

CatBoost can consume all CPU cores, potentially starving GPU models of CPU resources needed for data preparation and management.

### Solution
Added `CATBOOST_THREAD_COUNT = 16` configuration to limit CatBoost CPU usage.

## Resource Allocation

### XGBoost (GPU)
- Uses `tree_method='gpu_hist'`
- Computations on GPU
- Minimal CPU usage (data loading/prep only)
- No thread limit needed

### LightGBM (GPU)
- Uses `device='gpu'`
- Computations on GPU
- Minimal CPU usage (data loading/prep only)
- No thread limit needed

### CatBoost (CPU)
- Uses `task_type='CPU'`
- **Limited to 16 threads** via `thread_count` parameter
- Leaves CPU headroom for GPU models

## How to Adjust

Edit the configuration at the top of the script:
```python
# CPU thread limit for CatBoost (to leave CPU resources for GPU models)
CATBOOST_THREAD_COUNT = 16  # Adjust based on your CPU
```

### Recommended Settings
- **32-core CPU**: Set to 16-20 threads
- **64-core CPU**: Set to 32-40 threads
- **128-core CPU**: Set to 64-80 threads

Rule of thumb: Use 50-60% of total cores for CatBoost.

## Monitoring Resource Usage

```bash
# Monitor CPU usage by process
htop

# Monitor GPU usage
nvidia-smi -l 1

# Check thread count for running processes
ps -eLf | grep python | wc -l
```

## Expected Impact
- More stable GPU performance
- Better overall throughput
- Reduced resource contention
- Consistent trial times

## Note
If you have a very powerful CPU (64+ cores), you can increase `CATBOOST_THREAD_COUNT` to 32 or more for better CatBoost performance while still leaving plenty of resources for GPU models.