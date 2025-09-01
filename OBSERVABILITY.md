# Runtime Profiling Overview

This framework provides a lightweight runtime profiler to surface key timing metrics with minimal overhead. Profiling is **enabled by default** (`profile_training = True`) and can be disabled by setting `GenericTrainerConfig.profile_training = False`.

What It Measures

- Data Fetch: Time to retrieve the next batch per dataloader.
- Forward: Time for model forward pass (and loss) per dataloader.
- Backward: Time for backward/autograd per dataloader.
- Optimizer: Time for optimizer step per dataloader.

Metric Names

- Training (per dataloader):
  - `profile/train/dl_{loader_name}/time_data_ms`
  - `profile/train/dl_{loader_name}/time_forward_ms`
  - `profile/train/dl_{loader_name}/time_backward_ms`
  - `profile/train/dl_{loader_name}/time_optimizer_ms`
- Validation (per dataloader):
  - `profile/val/dl_{loader_name}/time_data_ms`
  - `profile/val/dl_{loader_name}/time_forward_ms`
- Optional approximation:
  - `profile/train/dl_{loader_name}/time_batch_wait_ms` (time between batches)

DDP Behavior

- Metrics log only on the primary rank.

Logging Frequency

- Respects `logging.log_scalars_every_n_steps` (None = every step). Forward/backward/data logs emit at optimizer boundaries to align with step-based frequency. Optimizer timing logs after each optimizer step.

CUDA Synchronization

- Forward/backward/optimizer timings optionally call `torch.cuda.synchronize()` when the model is on CUDA to improve timing accuracy.

Usage

- Profiling is enabled by default. To disable, set `GenericTrainerConfig.profile_training = False`.
- Optionally provide explicit dataloader names via `MultiDataLoaderConfig.dataloader_names` to produce clean metric keys.

Example YAML Snippet

```yaml
# Profiling is enabled by default, but can be disabled:
training:
  profile_training: false  # Set to false to disable profiling

# With profiling enabled (default):
logging:
  log_scalars_every_n_steps: 50
data:
  train:
    dataloader_names: ["main", "aux"]
```

Example Output

- `profile/train/dl_main/time_forward_ms: 3.21`
- `profile/train/dl_aux/time_backward_ms: 1.87`
- `profile/train/dl_main/time_optimizer_ms: 0.44`
- `profile/val/dl_val_main/time_forward_ms: 2.78`
- Multi‑Dataloader Interpretation
- When multiple dataloaders are interleaved, compare per‑loader timings to spot imbalances. For example, high `time_data_ms` on one loader points to dataset I/O or preprocessing as the bottleneck for that source. Discrepancies between `time_forward_ms` across loaders often indicate different input shapes or sequence lengths. In WEIGHTED schedules, inspect realized proportions alongside timings to ensure allocation matches expectations.

Overhead And Accuracy Notes

- Overhead is designed to be minimal: timers and occasional CUDA syncs. Forward/backward/optimizer timings include optional `torch.cuda.synchronize()` which improves accuracy but adds a small barrier cost. Data fetch timing measures `next(iterator)` wall time (CPU‑side) and typically doesn’t require CUDA sync.
- To quantify overhead in your environment:
  1) Run a short training job with `profile_training=True` (default) and capture steps/sec.
  2) Repeat with `profile_training=False` and identical seeds, batch sizes and loaders.
  3) Compare steps/sec or avg step time in your logs.

Tips

- Provide explicit, stable dataloader names via `MultiDataLoaderConfig.dataloader_names` for clean metric keys.
- Use a non‑None `logging.log_scalars_every_n_steps` (e.g., 50) for high‑frequency training to reduce logger overhead.
- On GPUs, timings are most accurate with CUDA sync enabled (default for forward/backward/optimizer). If you need absolute minimum overhead, remove syncs at the cost of precision.
