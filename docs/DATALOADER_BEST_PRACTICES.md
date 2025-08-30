# DataLoader Best Practices

This guide covers best practices for creating high-performance DataLoaders with proper configuration for GPU training and reproducibility.

## Optimal DataLoader Configuration

### Basic High-Performance Setup

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate

def create_optimized_dataloader(
    dataset,
    batch_size,
    is_training=True,
    num_workers=4,
    world_size=1,
    rank=0,
    seed=42
):
    """Create DataLoader with optimal settings for GPU training."""

    # Use DistributedSampler for multi-GPU training
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_training,
            seed=seed,  # Ensure reproducibility
            drop_last=is_training  # Drop incomplete batches for training
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = is_training

    # Create DataLoader with performance optimizations
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Pin memory for faster GPU transfer
        drop_last=is_training,  # Drop incomplete final batch for stable training
        collate_fn=default_collate,
        worker_init_fn=worker_init_fn,  # Set worker seeds
    )
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,        # Prefetch batches (only valid when num_workers>0)
        )
    dataloader = DataLoader(**loader_kwargs)

    return dataloader

def worker_init_fn(worker_id):
    """Initialize each worker with a unique seed for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)
```

### Non-Blocking GPU Transfers

When using pinned memory, enable non-blocking transfers for better CPU-GPU overlap:

```python
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    # Move data to GPU with non_blocking when pin_memory is enabled
    device = next(trainer.model.parameters()).device
    if trainer.config.performance.pin_memory and device.type == 'cuda':
        # Non-blocking transfer overlaps with computation
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
    else:
        # Standard transfer for CPU or when pin_memory is disabled
        x = batch[0].to(device)
        y = batch[1].to(device)

    # Computation proceeds while transfer completes
    outputs = trainer.model(x)
    loss = F.cross_entropy(outputs, y)
    return {"loss": loss}
```

**Important**: Only use `non_blocking=True` when:

1. `pin_memory=True` in DataLoader
2. Training on CUDA device
3. Data is being transferred from CPU to GPU

### Ensure per-epoch shuffling in DDP

When using `DistributedSampler`, call `set_epoch(epoch)` at the start of each epoch to get different shuffles across epochs and ranks:

```python
from torch.utils.data import DistributedSampler

for epoch in range(num_epochs):
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    # proceed with training loop for this epoch
```

## Performance Optimization Guidelines

### 1. Number of Workers

```python
def get_optimal_num_workers(device_type="cuda"):
    """Determine optimal number of workers based on system."""
    import os

    if device_type == "cpu":
        return 0  # No multiprocessing for CPU-only training

    # General guideline: 4 workers per GPU
    num_gpus = torch.cuda.device_count()
    cpu_count = os.cpu_count() or 1

    # Don't exceed available CPUs
    optimal = min(4 * max(num_gpus, 1), cpu_count)

    # Account for memory constraints
    # Reduce if you encounter out-of-memory errors
    return optimal

# Usage
num_workers = get_optimal_num_workers()
```

### 2. Batch Size and Memory Management

```python
def calculate_optimal_batch_size(model, input_shape, device="cuda"):
    """Estimate optimal batch size for GPU memory."""
    if device == "cpu":
        return 32  # Conservative default for CPU

    # Start with a reasonable batch size
    batch_size = 64
    max_batch_size = 512

    while batch_size <= max_batch_size:
        try:
            # Try a forward pass with dummy data
            dummy_input = torch.randn(batch_size, *input_shape).to(device)
            with torch.no_grad():
                _ = model(dummy_input)

            # Clear cache
            if device == "cuda":
                torch.cuda.empty_cache()

            # If successful, try larger batch
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Back off to last successful size
                batch_size //= 2
                break
            else:
                raise

    # Use 80% of max to leave room for gradients
    return int(batch_size * 0.8)
```

### 3. Mixed Precision Training

```python
def setup_mixed_precision():
    """Configure mixed precision training for better performance."""
    # Enable TF32 on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Note: When using GenericTrainer, it handles GradScaler internally
    # This is just for demonstration of manual AMP setup

def training_step_with_amp(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch

    device = next(trainer.model.parameters()).device
    use_amp = trainer.config.performance.use_amp and device.type == 'cuda'

    # Use autocast for mixed precision only on CUDA
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        outputs = trainer.model(x)
        loss = F.cross_entropy(outputs, y)

    # Return the loss tensor; the trainer handles backward/optimizer step/scaler
    return {"loss": loss}
```

## Reproducibility

### Setting Seeds Properly

```python
def set_reproducible_training(seed=42):
    """Set all seeds for reproducible training."""
    import random
    import numpy as np

    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For DataLoader workers
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return seed_worker, g

# Usage
seed_worker, generator = set_reproducible_training(42)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=generator  # For reproducible shuffling
)
```

## Multi-DataLoader Optimizations

### Efficient Multi-Dataset Loading

```python
from model_training_framework.trainer import (
    MultiDataLoaderConfig,
    SamplingStrategy,
)

def create_multi_dataloaders(datasets_dict, batch_sizes, num_workers=4):
    """Create multiple DataLoaders with shared worker pool."""

    loaders = []
    names = []

    for name, (dataset, batch_size) in datasets_dict.items():
        # Adjust workers per loader to avoid oversubscription
        workers_per_loader = max(1, num_workers // len(datasets_dict))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers_per_loader,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True
        )

        loaders.append(loader)
        names.append(name)

    # Configure multi-loader training
    config = MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.WEIGHTED,
        dataloader_weights=[1.0] * len(loaders),  # Equal weighting
        dataloader_names=names,
        cycle_short_loaders=True  # Restart exhausted loaders
    )

    return loaders, config
```

## Common Issues and Solutions

### 1. Out of Memory Errors

```python
def handle_oom_errors():
    """Solutions for out-of-memory errors."""

    # 1. Reduce batch size
    # 2. Reduce number of workers
    # 3. Disable persistent_workers
    # 4. Use gradient accumulation

    config = {
        "batch_size": 16,  # Smaller batch
        "num_workers": 2,  # Fewer workers
        "persistent_workers": False,  # Free memory between epochs
        "gradient_accumulation_steps": 4,  # Simulate larger batch
    }

    return config
```

### 2. Slow Data Loading

```python
def diagnose_dataloader_speed(dataloader):
    """Profile DataLoader performance."""
    import time

    # Warmup
    for _ in range(3):
        next(iter(dataloader))

    # Time loading
    times = []
    for _ in range(10):
        start = time.time()
        _ = next(iter(dataloader))
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"Average batch load time: {avg_time:.3f}s")

    if avg_time > 0.1:  # More than 100ms is slow
        print("Consider:")
        print("- Increasing num_workers")
        print("- Using pin_memory=True (for GPU training)")
        print("- Enabling persistent_workers")
        print("- Increasing prefetch_factor")
```

### 3. Non-Deterministic Results

```python
def ensure_deterministic_dataloader(dataset, batch_size):
    """Create fully deterministic DataLoader."""

    # Set all seeds
    set_reproducible_training(42)

    # Create generator for DataLoader
    g = torch.Generator()
    g.manual_seed(42)

    # Disable non-deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Create DataLoader with fixed randomness
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded for full determinism
        generator=g,
        drop_last=True  # Consistent batch sizes
    )

    return loader
```

## Integration with Model Training Framework

### Complete Example

```python
from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
    PerformanceConfig,
)

# Create optimized DataLoaders
train_loader = create_optimized_dataloader(
    train_dataset,
    batch_size=32,
    is_training=True,
    num_workers=4,
    world_size=torch.cuda.device_count(),
    rank=0
)

val_loader = create_optimized_dataloader(
    val_dataset,
    batch_size=64,  # Larger batch for validation
    is_training=False,
    num_workers=4
)

# Configure trainer
config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["train"],
    ),
    # Enable mixed precision
    performance=PerformanceConfig(
        use_amp=True,  # Enable automatic mixed precision
        dataloader_num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    ),
)

# Create trainer
trainer = GenericTrainer(config, model, [optimizer])

# Train with optimized DataLoaders
trainer.fit(
    train_loaders=[train_loader],
    val_loaders=[val_loader],
    max_epochs=100
)
```

## Profiling and Monitoring

### Adding Profiling Hooks

```python
from model_training_framework.trainer.hooks import TrainerHooks
import time

class DataLoaderProfilingHook(TrainerHooks):
    """Profile DataLoader performance during training."""

    def __init__(self):
        super().__init__()
        self.batch_load_times = []
        self.gpu_transfer_times = []

    def on_train_batch_start(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name, metrics):
        batch_time = time.time() - self.batch_start_time
        self.batch_load_times.append(batch_time)

        # Log slow batches
        if batch_time > 1.0:  # More than 1 second
            print(f"Slow batch {batch_idx}: {batch_time:.2f}s")

        # Periodic summary
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_time = sum(self.batch_load_times[-100:]) / 100
            print(f"Avg batch time (last 100): {avg_time:.3f}s")

    def on_epoch_end(self, trainer, epoch):
        avg_time = sum(self.batch_load_times) / len(self.batch_load_times)
        print(f"Epoch {epoch} - Avg batch time: {avg_time:.3f}s")
        self.batch_load_times = []

# Register hook
trainer.hook_manager.register_hook(DataLoaderProfilingHook())
```

## See Also

- [Multi-DataLoader Guide](MULTI_DATALOADER.md)
- [Configuration Guide](CONFIGURATION.md)
- [Hooks Documentation](HOOKS.md)
