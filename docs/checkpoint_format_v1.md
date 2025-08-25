# Checkpoint Format v1 Documentation

## Overview

The Model Training Framework uses checkpoint format v1 for fault-tolerant training with deterministic resume capabilities. This format supports mid-epoch resume, multi-dataloader training, and mixed precision training with AMP.

## Checkpoint Schema v1

### Core Fields

```python
{
    # Format metadata
    "format_version": 1,
    "is_multi_dataloader_only": True,
    "save_timestamp": float,  # Unix timestamp

    # Training state
    "epoch": int,  # Current epoch (0-based)
    "global_step": int,  # Global step counter

    # Model state
    "model_state_dict": dict,  # PyTorch model state

    # Multi-optimizer support (list of optimizer states)
    "optimizer_state_dicts": list[dict],

    # Multi-scheduler support (list of scheduler states)
    "scheduler_state_dicts": list[dict],

    # AMP scaler state for mixed precision
    "amp_scaler_state": dict | None,

    # RNG states for reproducibility
    "rng_states": {
        "python_random": tuple,  # Python random state
        "numpy_random": tuple,  # NumPy random state
        "torch_cpu": ByteTensor,  # PyTorch CPU RNG state
        "torch_cuda": list[ByteTensor] | None,  # PyTorch CUDA RNG states
    },

    # DataLoader manager state
    "dataloader_manager_state": {
        "choice_rng_state": dict,  # RNG for weighted sampling
        "choice_rng": tuple,  # NumPy RandomState
        "train_iterator_state": dict | None,  # Training iterator state
        "val_iterator_state": dict | None,  # Validation iterator state
    },

    # Explicit choice RNG (for weighted sampling)
    "choice_rng_state": tuple | None,

    # Resume state for fault tolerance
    "resume_state": ResumeState,

    # Optional fields
    "metrics_history": dict | None,
    "config_snapshot": dict | None,
}
```

### Iterator State Structure

```python
{
    "schedule_position": int,  # Current position in schedule
    "loader_states": list[dict],  # Per-loader states
    "total_batches": int,  # Total batches processed
    "loader_cycles": dict[int, int],  # Cycles per loader
    "prefetched_batches": int,  # Prefetched batch count
}
```

### DataLoaderState Structure

```python
{
    "batch_idx": int,  # Current batch index
    "exhausted": bool,  # Whether loader is exhausted
    "sampler_state": dict | None,  # Sampler state if available
    "dataset_state": dict | None,  # Dataset state if available
}
```

## Deterministic Resume Requirements

### Dataset Requirements

For deterministic mid-epoch resume, datasets should implement one of:

1. **CheckpointableIterable Protocol** (for iterable datasets):

```python
class MyIterableDataset(IterableDataset):
    def state_dict(self) -> dict[str, Any]:
        """Return current dataset state."""
        return {"position": self.position, ...}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore dataset state."""
        self.position = state["position"]
```

1. **Stateful Sampler** (for map-style datasets):

```python
class MySampler(Sampler):
    def state_dict(self) -> dict[str, Any]:
        """Return current sampler state."""
        return {"indices": self.indices, "position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore sampler state."""
        self.indices = state["indices"]
        self.position = state["position"]
```

### Fallback Behavior

When datasets/samplers don't implement state methods:

- Framework falls back to skip-based restoration
- Logs warning message with guidance
- May be inefficient for large datasets
- Not suitable for streaming datasets

### Worker Considerations

For multi-worker dataloaders:

- Ensure deterministic worker seeding
- Use `worker_init_fn` to set seeds based on worker ID
- Consider single-process dataloaders for critical reproducibility

## Mixed Precision Training (AMP)

### Configuration

```python
config = GenericTrainerConfig(
    performance=PerformanceConfig(
        use_amp=True,  # Enable mixed precision
    )
)
```

### Automatic Features

When AMP is enabled:

- GradScaler automatically created for CUDA devices
- Forward passes wrapped in `autocast("cuda")`
- Loss scaled before backward pass
- Gradients unscaled before clipping
- Optimizer steps handled by scaler

### Checkpoint Integration

AMP scaler state is automatically:

- Saved in `amp_scaler_state` field
- Restored on checkpoint load
- Preserved across training interruptions

## Usage Examples

### Saving Checkpoints

```python
from model_training_framework.trainer.checkpoints import save_checkpoint

# Save complete training state
save_checkpoint(path=Path("checkpoint.pt"), trainer=trainer)
```

### Loading Checkpoints

```python
from model_training_framework.trainer.checkpoints import load_checkpoint

# Restore complete training state
load_checkpoint(path=Path("checkpoint.pt"), trainer=trainer)
```

### With CheckpointManager

```python
manager = CheckpointManager(config)

# Save with rotation and best tracking
path = manager.save_checkpoint(
    model=model,
    optimizers=optimizers,
    schedulers=schedulers,
    scaler=scaler,  # AMP scaler
    resume_state=resume_state,
    epoch=epoch,
    global_step=step,
    metrics=metrics,
)

# Restore from latest or best
epoch, step, resume_state = manager.restore_from_checkpoint(
    model=model,
    optimizers=optimizers,
    schedulers=schedulers,
    scaler=scaler,
)
```

## Platform Notes

### Windows Compatibility

- Symlinks may fail on Windows without developer mode
- Framework automatically falls back to copy operations
- No functional impact, only performance difference

### DDP/Fabric Considerations

- Schedules broadcast from rank 0 for consistency
- All ranks must use identical configuration
- Checkpoint save/load typically on rank 0 only

## DDP (Distributed Data Parallel) Requirements

### Key Requirements for Distributed Training

1. **DistributedSampler Usage**
   - All DataLoaders must use `DistributedSampler` or compatible sampler
   - Samplers must implement `set_epoch(epoch)` method for proper shuffling
   - Each rank must see the same effective number of batches per epoch

2. **Custom Sampler Requirements**
   - Must implement `set_epoch(epoch)` for deterministic shuffling across epochs
   - Should implement `state_dict()` and `load_state_dict()` for mid-epoch resume
   - Must ensure consistent batch counts across all ranks

3. **Rank Synchronization**
   - Checkpoint loading occurs only on rank 0 and is broadcast to other ranks
   - All logging (WandB, console) happens only on primary rank to avoid duplication
   - Barriers ensure all ranks are synchronized at epoch boundaries

4. **Schedule Consistency**
   - Multi-dataloader schedules are built on rank 0 and broadcast
   - Choice RNG state for weighted sampling is synchronized across ranks
   - All ranks must see identical (dataloader_idx, batch) sequences

### Example Custom Sampler for DDP

```python
class DDPCompatibleSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """Required for deterministic shuffling in DDP."""
        self.epoch = epoch

    def state_dict(self) -> dict:
        """For mid-epoch resume support."""
        return {
            'epoch': self.epoch,
            'indices': self.indices,
            'position': self.position
        }

    def load_state_dict(self, state: dict):
        """Restore sampler state."""
        self.epoch = state['epoch']
        self.indices = state['indices']
        self.position = state['position']
```

### Common DDP Issues and Solutions

1. **Rank Divergence**
   - **Issue**: Different ranks see different numbers of batches
   - **Solution**: Use DistributedSampler with consistent `drop_last` setting

2. **Non-deterministic Resume**
   - **Issue**: Training doesn't resume at exact same point across ranks
   - **Solution**: Implement state_dict/load_state_dict in custom samplers

3. **Duplicate Logging**
   - **Issue**: Metrics logged multiple times (once per rank)
   - **Solution**: Framework automatically restricts logging to rank 0

4. **Checkpoint Loading Overhead**
   - **Issue**: All ranks loading checkpoint causes filesystem contention
   - **Solution**: Framework loads on rank 0 and broadcasts to other ranks

## Migration from Legacy Formats

The framework supports backward compatibility:

- Detects format version automatically
- Falls back to single optimizer/scheduler if needed
- Logs migration recommendations

## Best Practices

1. **Always use format v1** for new projects
2. **Implement state methods** on custom datasets/samplers
3. **Test resume behavior** with partial epoch training
4. **Monitor warnings** for fallback behaviors
5. **Use AMP** for faster training when possible
6. **Regular checkpoints** with step-based saving for preemption safety

## Troubleshooting

### Common Issues

1. **Non-deterministic resume**: Check dataset/sampler state implementation
2. **Skip-based restoration slow**: Implement state_dict/load_state_dict
3. **AMP errors**: Ensure CUDA available and compatible GPU
4. **Checkpoint size**: Consider saving frequency vs storage constraints

### Debug Logging

Enable debug logging to see restoration details:

```python
import logging
logging.getLogger("model_training_framework.trainer").setLevel(logging.DEBUG)
```

This will show:

- Successful state restorations
- Fallback skip operations
- Schedule generation details
- Iterator state transitions
