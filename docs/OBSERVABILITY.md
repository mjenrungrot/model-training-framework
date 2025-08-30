# Observability Guide for Model Training Framework

This guide covers the observability features in the training framework, including logging, metrics, and hooks.

## Table of Contents

- [Logging](#logging)
- [Metrics](#metrics)
- [Hooks System](#hooks-system)
- [Early Stopping](#early-stopping)

## Logging

### Logger Types

The framework supports multiple logging backends through a unified `LoggerProtocol`:

- **WandB**: Weights & Biases integration for experiment tracking
- **TensorBoard**: Local visualization and metrics tracking
- **Console**: Structured console output for development
- **Composite**: Combine multiple loggers simultaneously

### Basic Configuration

```python
from model_training_framework.config.schemas import LoggingConfig

# Single logger
config = LoggingConfig(
    logger_type="wandb",
    wandb_project="my-project",
    wandb_entity="my-team",
    wandb_name="run-name",
    wandb_mode="online",
    wandb_id="abc123",
    wandb_resume="allow",
)

# Composite logger with explicit list
config = LoggingConfig(
    logger_type="composite",
    composite_loggers=["console", "tensorboard", "wandb"],
    wandb_project="my-project",
    tensorboard_dir="./tb_logs"
)
```

Field semantics:

- wandb_mode: one of "online", "offline", "disabled". If omitted, honors the WANDB_MODE environment variable.
- wandb_resume: one of "allow", "must", "never". Use together with wandb_id to continue a previous run.
- wandb_id: stable unique run ID to resume or de-duplicate runs.
- wandb_name: human-readable run display name (does not affect resume).

Note: WandB runs are created only on the primary rank when using DDP. If wandb_mode is not set in the config, the WANDB_MODE environment variable (if present) is respected.

### Composite Logger

The composite logger allows using multiple logging backends simultaneously:

```python
# Automatic composite (console + one backend)
config = LoggingConfig(
    logger_type="composite",
    wandb_project="my-project"  # Will create console + wandb
)

# Explicit composite configuration
config = LoggingConfig(
    logger_type="composite",
    composite_loggers=["console", "tensorboard"],  # Specific loggers
    tensorboard_dir="./logs"
)
```

**Note**: When including "wandb" in `composite_loggers`, ensure `wandb_project` is provided in the configuration to avoid warnings.

## Metrics

### Multi-DataLoader Metrics

The framework tracks metrics per dataloader and provides global aggregates:

```python
config = LoggingConfig(
    log_per_loader_metrics=True,  # Track per-loader metrics
    log_global_metrics=True,      # Compute global aggregates
    log_loader_proportions=True   # Track loader usage (WEIGHTED strategy)
)
```

### Cross-Rank Aggregation (DDP)

For distributed training, enable cross-rank metric aggregation to get accurate global metrics:

```python
config = LoggingConfig(
    all_reduce_metrics=True  # Aggregate metrics across all ranks
)
```

When enabled, metrics are aggregated using all-reduce operations before logging, ensuring that metrics reflect the global state across all processes rather than just rank 0.

### Aggregation Strategies

The `MetricsManager` supports multiple aggregation strategies:

```python
from model_training_framework.trainer.metrics import AggregationStrategy

# In your trainer configuration
metrics_manager = MetricsManager(
    aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE  # Default
    # Other options: SIMPLE_AVERAGE, SUM, MAX, MIN
)
```

## Hooks System

The hooks system allows injecting custom behavior at various training lifecycle points.

### Built-in Hooks

#### Logging Hook

Basic logging of training progress:

```python
config = HooksConfig(
    enable_logging_hook=True
)
```

#### Gradient Monitor Hook

Monitor gradient statistics during training:

```python
config = HooksConfig(
    enable_gradient_monitor=True,
    gradient_monitor_config={
        "log_frequency": 100,  # Log every N steps
        "param_filter": ["encoder", "decoder"]  # Monitor specific parameters
    }
)
```

The `param_filter` option is useful for large models where monitoring all gradients would be expensive. Only parameters with names containing any of the filter strings will be monitored.

#### Model Checkpoint Hook

Custom checkpoint behavior:

```python
config = HooksConfig(
    enable_model_checkpoint_hook=True,
    model_checkpoint_config={
        "save_top_k": 3,
        "monitor": "val/loss"
    }
)
```

#### Early Stopping Hook

Stop training when metrics stop improving:

```python
config = HooksConfig(
    enable_early_stopping_hook=True,
    early_stopping_config={
        "monitor": "val/loss",
        "patience": 10,
        "mode": "min",  # "min" for loss, "max" for accuracy
        "min_delta": 0.0001
    }
)
```

### Custom Hooks

Create custom hooks by subclassing `TrainerHooks`:

```python
from model_training_framework.trainer.hooks import TrainerHooks

class MyCustomHook(TrainerHooks):
    def on_epoch_start(self, trainer, epoch):
        print(f"Starting epoch {epoch}")

    def on_train_batch_end(self, trainer, batch, loader_idx, loader_name, metrics):
        if metrics.get("loss", 0) > 10:
            print(f"High loss detected: {metrics['loss']}")

# Register via configuration
config = HooksConfig(
    hook_classes=["mypackage.hooks.MyCustomHook"],
    hook_configs={"mypackage.hooks.MyCustomHook": {}}
)

# Or register directly after trainer creation
trainer = GenericTrainer(config, model, optimizers)
trainer.hook_manager.register_hook(MyCustomHook())
```

### Hook Execution Order

Hooks are executed in registration order:

1. User-provided custom hooks (first)
2. Built-in hooks (last)

This ensures user hooks can override or enhance built-in behavior.

## Early Stopping

The framework supports two early stopping mechanisms:

### Legacy Early Stopping

Using the `EarlyStopping` utility:

```python
from model_training_framework.config.schemas import GenericTrainerConfig

# Configure via GenericTrainerConfig
config = GenericTrainerConfig(
    early_stopping_patience=10,
    early_stopping_metric="val_loss",
    early_stopping_mode="min"
)

trainer = GenericTrainer(config, model, optimizers)
```

### Hook-based Early Stopping

Using the `EarlyStoppingHook`:

```python
config = HooksConfig(
    enable_early_stopping_hook=True,
    early_stopping_config={
        "monitor": "val/loss",
        "patience": 10
    }
)
```

### Controlling Early Stopping Source

You can control which early stopping mechanism is active:

```python
from model_training_framework.config.schemas import ValidationConfig

config = ValidationConfig(
    early_stopping_source="both"  # Default: both can trigger
    # Options:
    # - "both": Either mechanism can stop training
    # - "hook": Only EarlyStoppingHook is checked
    # - "legacy": Only legacy EarlyStopping is checked
)
```

### DDP Synchronization

Early stopping decisions are synchronized across all ranks in distributed training to prevent deadlocks. When any rank triggers early stopping, all ranks will stop together.

## Best Practices

### For Large Models

1. **Gradient Monitoring**: Use parameter filtering to reduce overhead

   ```python
   gradient_monitor_config={
       "log_frequency": 100,
       "param_filter": ["attention", "mlp"]  # Monitor key layers only
   }
   ```

2. **Cross-rank Aggregation**: Enable only when needed for accuracy

   ```python
   all_reduce_metrics=True  # Has communication overhead
   ```

### For Production

1. **Composite Logging**: Use multiple backends for redundancy

   ```python
   composite_loggers=["console", "wandb", "tensorboard"]
   ```

2. **Early Stopping**: Prefer hook-based for consistency

   ```python
   early_stopping_source="hook"  # Deterministic behavior
   ```

3. **Hook Registration**: Register critical hooks first

   ```python
   hook_classes=["mypackage.hooks.CriticalHook", "mypackage.hooks.OptionalHook"]  # Order matters
   ```

## Examples

### Complete Configuration

```python
from model_training_framework.config.schemas import (
    GenericTrainerConfig,
    LoggingConfig,
    HooksConfig,
    ValidationConfig
)

config = GenericTrainerConfig(
    logging=LoggingConfig(
        logger_type="composite",
        composite_loggers=["console", "wandb"],
        wandb_project="my-experiment",
        all_reduce_metrics=True,
        log_per_loader_metrics=True,
        log_loader_proportions=True
    ),
    hooks=HooksConfig(
        enable_gradient_monitor=True,
        gradient_monitor_config={
            "log_frequency": 50,
            "param_filter": ["transformer"]
        },
        enable_early_stopping_hook=True,
        early_stopping_config={
            "monitor": "val/loss",
            "patience": 5
        }
    ),
    validation=ValidationConfig(
        early_stopping_source="hook"
    )
)
```

### Multi-DataLoader with Proportions

```python
# Training with multiple dataloaders using WEIGHTED strategy
from model_training_framework.trainer import SamplingStrategy, MultiDataLoaderConfig

config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.WEIGHTED,
        dataloader_weights=[0.7, 0.3],  # 70% loader1, 30% loader2
    ),

    logging=LoggingConfig(
        log_loader_proportions=True,  # Track realized proportions
        all_reduce_metrics=True       # Accurate global proportions
    )
)

# Metrics will include:
# - loader_proportions/loader1_batches: ~0.7
# - loader_proportions/loader2_batches: ~0.3
# - loader_proportions/loader1_divergence: |actual - expected|
```
