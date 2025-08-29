# Advanced Features

This guide covers advanced features of the Model Training Framework for production use cases.

## Table of Contents

1. [Fault-Tolerant Training](#fault-tolerant-training)
2. [Git Integration](#git-integration)
3. [Experiment Tracking](#experiment-tracking)
4. [Enhanced Logging](#enhanced-logging)
5. [Metrics Management](#metrics-management)
6. [DDP with Multi-Loaders](#ddp-with-multi-loaders)
7. [Mixed Precision Training](#mixed-precision-training)
8. [Preemption Handling](#preemption-handling)

## Fault-Tolerant Training

The framework provides preemption-safe training with automatic checkpointing:

### Configuration

```python
from model_training_framework.trainer import (
    GenericTrainerConfig,
    CheckpointConfig,
    PreemptionConfig,
    FaultToleranceConfig
)
import signal

trainer_config = GenericTrainerConfig(
    checkpoint=CheckpointConfig(
        save_every_n_epochs=1,
        save_every_n_steps=500,
        save_rng=True,  # For deterministic resume
        max_checkpoints=5
    ),
    fault_tolerance=FaultToleranceConfig(
        save_sampler_state=True,  # For exact resume
        save_dataset_state=True,
        verify_deterministic_resume=True,
    ),
    preemption=PreemptionConfig(
        signal=signal.SIGUSR1,
        max_checkpoint_sec=300.0,
        requeue_job=True
    )
)
```

### Automatic Resume

```python
# Training automatically resumes from latest checkpoint
if checkpoint_path.exists():
    trainer.load_checkpoint(checkpoint_path)
    # Resumes from exact batch/sample where it left off
```

### Handling SLURM Preemption

The framework automatically handles SLURM preemption signals:

```python
# In your SLURM script
#SBATCH --signal=USR1@60  # Send SIGUSR1 60s before termination

# The trainer automatically:
# 1. Catches the signal
# 2. Saves checkpoint
# 3. Exits gracefully
# 4. Resumes on next run
```

## Git Integration

Automatic git branch management for experiment isolation:

### Basic Usage

```python
from model_training_framework import ModelTrainingFramework

framework = ModelTrainingFramework(project_root=".")

# Creates temporary branches for each experiment
# Format: slurm-job/<experiment_name>/<timestamp>/<commit_hash>
result = framework.run_single_experiment(
    config=config,
    script_path="scripts/train.py",
    use_git_branch=True  # Enable git integration
)
```

### Branch Management

```python
from model_training_framework.slurm import GitManager

git_manager = GitManager(repo_path=".")

# Create experiment branch
with git_manager.branch_context("experiment-001", create_if_not_exists=True):
    # Work on the specified branch
    # Automatically returns to original branch when done
    pass

# Commit changes
commit_hash = git_manager.commit_changes(
    message="Experiment 001 configuration",
    files=["configs/experiment_001.yaml"],
    add_all=False
)
```

## Experiment Tracking

### Weights & Biases Integration

```python
from model_training_framework.config.schemas import LoggingConfig

config = GenericTrainerConfig(
    logging=LoggingConfig(
        logger_type="wandb",
        wandb_project="my-project",
        wandb_entity="my-team",
        wandb_name="run-001",
        wandb_mode="online",  # or "offline", "disabled"
        wandb_id="unique-run-id",  # For resuming runs
        wandb_resume="allow",  # or "must", "never"
        log_scalars_every_n_steps=50,
        log_images_every_n_steps=500
    )
)
```

### TensorBoard Integration

```python
config = GenericTrainerConfig(
    logging=LoggingConfig(
        logger_type="tensorboard",
        tensorboard_dir="./tb_logs",
        log_scalars_every_n_steps=10
    )
)
```

### Composite Logging

Use multiple logging backends simultaneously:

```python
config = GenericTrainerConfig(
    logging=LoggingConfig(
        logger_type="composite",
        composite_loggers=["console", "tensorboard", "wandb"],
        wandb_project="my-project",
        tensorboard_dir="./tb_logs"
    )
)
```

## Enhanced Logging

### Structured Console Output

```python
config = GenericTrainerConfig(
    logging=LoggingConfig(
        logger_type="console",
        console_log_level="INFO",
        log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
)
```

### Per-Loader Metrics

```python
config = GenericTrainerConfig(
    logging=LoggingConfig(
        log_per_loader_metrics=True,  # Track metrics per dataloader
        log_global_metrics=True,      # Also compute global aggregates
        log_loader_proportions=True   # Track loader usage (for WEIGHTED strategy)
    )
)
```

## Metrics Management

### Advanced Metrics Tracking

```python
from model_training_framework.trainer import MetricsManager, AggregationStrategy

# Configure metrics aggregation
config = GenericTrainerConfig(
    metrics=MetricsConfig(
        aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        track_proportions=True,
        per_loader_metrics=True,
    )
)

# Access metrics during training
def on_epoch_end(trainer, epoch, metrics):
    # Per-loader metrics
    loader_a_loss = metrics.get("train/dl_loader_a/loss")

    # Global aggregated metrics
    global_loss = metrics.get("train/loss")

    # Loader proportions
    proportions = trainer.metrics_manager.get_loader_proportions()
```

### Custom Metrics

```python
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch
    outputs = trainer.model(x)
    loss = F.cross_entropy(outputs, y)

    # Custom metrics
    accuracy = (outputs.argmax(1) == y).float().mean()
    f1_score = calculate_f1(outputs, y)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "custom/learning_rate": trainer.optimizers[0].param_groups[0]['lr']
    }
```

## DDP with Multi-Loaders

### Configuration for Distributed Training

```python
from lightning.fabric import Fabric
from model_training_framework.trainer import DDPConfig

fabric = Fabric(accelerator="gpu", devices=4, strategy="ddp")
fabric.launch()

config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.ROUND_ROBIN,
        dataloader_names=["shard_1", "shard_2"],
    ),
    ddp=DDPConfig(
        sync_schedules_across_ranks=True,
        validate_schedule_consistency=True,
        all_reduce_metrics=True  # Aggregate metrics across ranks
    )
)

# Fabric handles distributed setup
model, *optimizers = fabric.setup(model, *optimizers)
trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=optimizers,
    fabric=fabric,
)
```

### Rank-Aware Processing

```python
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    # Access rank information
    rank = trainer.fabric.global_rank if trainer.fabric else 0
    world_size = trainer.fabric.world_size if trainer.fabric else 1

    # Rank-specific logging
    if rank == 0:
        trainer.log("main_process/special_metric", value)

    # All-reduce for global metrics
    if trainer.fabric:
        global_loss = trainer.fabric.all_reduce(loss, reduce_op="mean")

    return {"loss": loss}
```

## Mixed Precision Training

### Automatic Mixed Precision (AMP)

```python
from model_training_framework.config.schemas import PerformanceConfig

config = GenericTrainerConfig(
    performance=PerformanceConfig(
        use_amp=True,  # Enable mixed precision
        amp_dtype="float16",  # or "bfloat16" for newer GPUs
    )
)

# AMP is automatically handled:
# - GradScaler created for CUDA devices
# - Forward passes wrapped in autocast
# - Loss scaled before backward
# - Gradients unscaled before clipping
```

### Manual AMP Control

```python
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    with torch.cuda.amp.autocast(enabled=trainer.config.performance.use_amp):
        outputs = trainer.model(batch)
        loss = criterion(outputs, targets)

    # Scaler is managed automatically by trainer
    return {"loss": loss}
```

## Preemption Handling

### Signal-Based Checkpointing

```python
import signal

# Configure preemption handling
config = GenericTrainerConfig(
    preemption=PreemptionConfig(
        signal=signal.SIGUSR1,  # Signal to catch
        timeout_minutes=5,       # Max time for checkpoint save
        grace_period_seconds=60, # Grace period before forced exit
        checkpoint_on_preemption=True,
        exit_on_max_preemptions=True,
        max_preemptions=3
    )
)

# Set up signal handler
def handle_preemption(signum, frame):
    trainer.save_checkpoint(emergency=True)
    if should_requeue:
        os.system(f"scontrol requeue {os.environ['SLURM_JOB_ID']}")
    sys.exit(0)

signal.signal(signal.SIGUSR1, handle_preemption)
```

### Automatic Recovery

```python
# The trainer automatically:
# 1. Detects previous checkpoints
# 2. Loads the latest valid checkpoint
# 3. Resumes from exact instruction level
# 4. Continues training seamlessly

trainer = GenericTrainer(config, model, optimizers)

# Automatic resume if checkpoint exists
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    latest = trainer.checkpoint_manager.find_latest_checkpoint()
    if latest:
        trainer.load_checkpoint(latest)
        print(f"Resumed from epoch {trainer.current_epoch}, step {trainer.global_step}")
```

## Best Practices

### Production Configuration

```python
# Comprehensive production config
production_config = GenericTrainerConfig(
    # Checkpointing
    checkpoint=CheckpointConfig(
        save_every_n_steps=1000,
        save_every_n_epochs=1,
        max_checkpoints=3,
        save_last=True,
        save_best=True,
        monitor="val/loss",
        mode="min"
    ),

    # Fault tolerance
    fault_tolerance=FaultToleranceConfig(
        save_sampler_state=True,
        save_dataset_state=True,
        verify_deterministic_resume=True,
    ),

    # Logging
    logging=LoggingConfig(
        logger_type="composite",
        composite_loggers=["console", "wandb", "tensorboard"],
        log_per_loader_metrics=True,
        all_reduce_metrics=True
    ),

    # Performance
    performance=PerformanceConfig(
        use_amp=True,
        compile_model=False,  # Set True for torch.compile
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    ),

    # Preemption
    preemption=PreemptionConfig(
        signal=signal.SIGUSR1,
        checkpoint_on_preemption=True,
        requeue_job=True
    )
)
```

### Error Handling

```python
from model_training_framework.trainer import (
    TrainerError,
    PreemptionTimeoutError,
    CheckpointTimeoutError
)

try:
    trainer.fit(train_loaders, val_loaders, max_epochs=100)
except PreemptionTimeoutError as e:
    # Handle preemption timeout
    logger.error(f"Preemption checkpoint failed: {e}")
    # Force save minimal state
    trainer.save_emergency_checkpoint()
except CheckpointTimeoutError as e:
    # Handle checkpoint timeout
    logger.error(f"Regular checkpoint failed: {e}")
    # Continue training without checkpoint
except TrainerError as e:
    # Handle general trainer errors
    logger.error(f"Training failed: {e}")
    raise
```

## See Also

- [Multi-DataLoader Guide](MULTI_DATALOADER.md) - Multi-loader training patterns
- [Hooks System](HOOKS.md) - Custom training hooks
- [API Reference](API.md) - Complete API documentation
- [Example Code](../demo/example3_production/) - Production example
