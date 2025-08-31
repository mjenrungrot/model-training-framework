# Add lightweight runtime profiling with per-dataloader timings

Closes #13.

Summary

- Implements opt-in runtime profiling controlled by `GenericTrainerConfig.profile_training`.
- Captures per-dataloader timings for data fetch, forward, backward, and optimizer.
- Adds forward timing hooks to the hook system.
- Auto-registers a `RuntimeProfilingHook` to time optimizer steps and approximate batch wait.
- DDP-safe logging (primary rank only) and log frequency gating using `logging.log_scalars_every_n_steps`.

Key Changes

- trainer/hooks.py: Added `on_before_forward`, `on_after_forward` to `TrainerHooks`. Implemented `RuntimeProfilingHook`.
- trainer/core.py: Tracked current dataloader, instrumented forward/backward timing in training and forward timing in validation, logged metrics, and auto-registered profiling hook.
- trainer/multi_dataloader.py: Measured `next()` fetch time, exposed `last_batch_fetch_ms` and `last_loader_name`.
- OBSERVABILITY.md: Documented usage, metric names, CUDA sync, multi-dataloader interpretation, examples.
- tests/trainer/test_runtime_profiling.py: Added comprehensive tests.

Metrics

- Train per loader:
  - `profile/train/dl_{name}/time_data_ms`
  - `profile/train/dl_{name}/time_forward_ms`
  - `profile/train/dl_{name}/time_backward_ms`
  - `profile/train/dl_{name}/time_optimizer_ms`
- Validation per loader:
  - `profile/val/dl_{name}/time_data_ms`
  - `profile/val/dl_{name}/time_forward_ms`
- Optional approximation:
  - `profile/train/dl_{name}/time_batch_wait_ms`

Testing

- 8 new tests:
  - Single, multi-dataloader profiling
  - Disabled flag -> no metrics
  - Validation profiling
  - Logging frequency gating
  - DDP primary-only gate (unit-level)
  - CUDA path (skippable)
  - Weighted sampling profiling
  - Dataloader exhaustion handling
- Full suite: 394 passed, 23 skipped, 7 warnings locally.

Notes

- Data fetch timing uses CPU wall-time of `next(iterator)`; CUDA sync is intentionally not applied by default.
- Name sanitization ensures metric keys are safe.

Next Steps

- Optional toggle to enable CUDA sync for data fetch timing if desired.
- Benchmark overhead in a representative workload and report steps/sec deltas.
