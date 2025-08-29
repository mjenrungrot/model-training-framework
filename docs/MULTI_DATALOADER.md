# Multi-DataLoader Training Guide

This guide covers training with multiple dataloaders, including sampling strategies, aggregation methods, and common patterns.

## Table of Contents

1. [Overview](#overview)
2. [Sampling Strategies](#sampling-strategies)
3. [Validation Aggregation](#validation-aggregation)
4. [Common Patterns](#common-patterns)
5. [Advanced Configurations](#advanced-configurations)
6. [Best Practices](#best-practices)

## Overview

The Model Training Framework is designed exclusively for multi-dataloader training. This unified approach provides:

- **Consistent API**: Same interface for 1 or N dataloaders
- **Deterministic Scheduling**: Reproducible training across runs
- **Flexible Aggregation**: Multiple validation strategies
- **Fault Tolerance**: Exact resume from any point

## Sampling Strategies

### ROUND_ROBIN - Fair Alternation

Alternates between dataloaders in order: A, B, C, A, B, C...

```python
from model_training_framework.trainer import (
    MultiDataLoaderConfig,
    SamplingStrategy,
    EpochLengthPolicy,
)

config = MultiDataLoaderConfig(
    sampling_strategy=SamplingStrategy.ROUND_ROBIN,
    epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
    dataloader_names=["dataset_a", "dataset_b", "dataset_c"],
)

# Each dataloader gets equal representation
# Good for: balanced multi-domain training
```

### WEIGHTED - Importance-Based Sampling

Sample based on specified weights:

```python
config = MultiDataLoaderConfig(
    sampling_strategy=SamplingStrategy.WEIGHTED,
    dataloader_weights=[0.5, 0.3, 0.2],  # 50%, 30%, 20%
    epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
    steps_per_epoch=1000,
    dataloader_names=["primary", "auxiliary", "synthetic"],
)

# Probabilistic sampling based on weights
# Good for: imbalanced datasets, importance weighting
```

### ALTERNATING - Custom Pattern

Define explicit sampling pattern:

```python
config = MultiDataLoaderConfig(
    sampling_strategy=SamplingStrategy.ALTERNATING,
    alternating_pattern=[0, 0, 1, 2],  # 2x A, 1x B, 1x C, repeat
    burst_size=3,  # Take 3 batches at a time from each
    dataloader_names=["main", "augmented", "hard_negatives"],
)

# Deterministic custom pattern
# Good for: curriculum learning, specific ratios
```

### SEQUENTIAL - Process in Order

Process dataloaders one after another:

```python
config = MultiDataLoaderConfig(
    sampling_strategy=SamplingStrategy.SEQUENTIAL,
    dataloader_names=["pretrain", "finetune", "adapt"],
    epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
)

# Complete one loader before moving to next
# Good for: staged training, transfer learning
```

## Validation Aggregation

### Aggregation Strategies

```python
from model_training_framework.trainer import ValidationConfig, ValAggregation

# Micro-average: Weight by number of samples
config = ValidationConfig(
    aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
    per_loader_metrics=True,
    global_metrics=True,
)

# Macro-average: Equal weight to each loader
config = ValidationConfig(
    aggregation=ValAggregation.MACRO_AVG_EQUAL_LOADERS,
)

# Primary metric per loader (for multi-task)
config = ValidationConfig(
    aggregation=ValAggregation.PRIMARY_METRIC_PER_LOADER,
)
```

### Custom Aggregation

```python
def custom_aggregation(metrics_dict):
    """Custom aggregation function."""
    # metrics_dict: {loader_name: {metric_name: value}}

    # Example: weighted by dataset importance
    weights = {"task_a": 0.6, "task_b": 0.4}
    total_loss = sum(
        metrics_dict[name]["loss"] * weights[name]
        for name in metrics_dict
    )
    return {"loss": total_loss}

config = ValidationConfig(
    aggregation_fn=custom_aggregation
)
```

## Common Patterns

### Multi-Task Learning

Different tasks with different optimizers:

```python
# Model with task-specific heads
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Linear(784, 256)
        self.task_a_head = nn.Linear(256, 10)  # Classification
        self.task_b_head = nn.Linear(256, 1)   # Regression

    def forward(self, x, task):
        features = self.shared_encoder(x)
        if task == "classification":
            return self.task_a_head(features)
        else:
            return self.task_b_head(features)

# Task-specific optimizers
model = MultiTaskModel()
optimizer_a = torch.optim.Adam(model.task_a_head.parameters(), lr=0.001)
optimizer_b = torch.optim.SGD(model.task_b_head.parameters(), lr=0.01)
optimizer_shared = torch.optim.Adam(model.shared_encoder.parameters(), lr=0.0001)

# Configuration
config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.WEIGHTED,
        dataloader_weights=[0.6, 0.4],
        dataloader_names=["classification", "regression"],
    ),
    per_loader_optimizers={
        "classification": {"optimizer_idx": [0, 2]},  # Use optimizer_a and shared
        "regression": {"optimizer_idx": [1, 2]},      # Use optimizer_b and shared
    },
)

# Training step
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch
    output = trainer.model(x, task=dataloader_name)

    if dataloader_name == "classification":
        loss = F.cross_entropy(output, y)
    else:
        loss = F.mse_loss(output, y)

    return {"loss": loss}

trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=[optimizer_a, optimizer_b, optimizer_shared]
)
```

### Curriculum Learning

Gradual difficulty increase:

```python
# Sequential processing with increasing difficulty
config = MultiDataLoaderConfig(
    sampling_strategy=SamplingStrategy.SEQUENTIAL,
    dataloader_names=["easy", "medium", "hard"],
    epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
)

# Or alternating pattern for gradual transition
config = MultiDataLoaderConfig(
    sampling_strategy=SamplingStrategy.ALTERNATING,
    alternating_pattern=[0, 0, 0, 0, 1, 1, 2],  # More easy, fewer hard
    dataloader_names=["easy", "medium", "hard"],
)

# Dynamic curriculum (adjust weights during training)
class CurriculumCallback(TrainerHooks):
    def on_epoch_end(self, trainer, epoch):
        if epoch < 10:
            weights = [0.7, 0.2, 0.1]  # Mostly easy
        elif epoch < 20:
            weights = [0.4, 0.4, 0.2]  # Balanced easy/medium
        else:
            weights = [0.2, 0.3, 0.5]  # Focus on hard

        trainer.multi_loader_manager.update_weights(weights)
```

### Domain Adaptation

Training on multiple domains:

```python
# Equal representation from each domain
config = MultiDataLoaderConfig(
    sampling_strategy=SamplingStrategy.ROUND_ROBIN,
    dataloader_names=["source_domain", "target_domain", "augmented"],
    cycle_short_loaders=True,  # Restart shorter loaders
)

# Domain-specific processing
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch

    # Apply domain-specific augmentation
    if dataloader_name == "target_domain":
        x = apply_target_augmentation(x)

    # Domain adversarial training
    features = trainer.model.encoder(x)
    task_output = trainer.model.task_head(features)

    if trainer.config.use_domain_adversarial:
        domain_output = trainer.model.domain_head(features)
        domain_label = torch.tensor([dataloader_idx]).to(x.device)
        domain_loss = F.cross_entropy(domain_output, domain_label)
        task_loss = F.cross_entropy(task_output, y)
        loss = task_loss - 0.1 * domain_loss  # Adversarial
    else:
        loss = F.cross_entropy(task_output, y)

    return {"loss": loss, f"{dataloader_name}_loss": loss}
```

### Handling Imbalanced Datasets

Oversampling minority classes:

```python
# Dataset sizes: A=10000, B=1000, C=100
# Use inverse weights for balanced sampling
total_samples = 10000 + 1000 + 100
weights = [
    100 / 10000,  # Undersample majority
    1.0,           # Normal for medium
    10.0,          # Oversample minority
]

# Normalize weights
weights = [w / sum(weights) for w in weights]

config = MultiDataLoaderConfig(
    sampling_strategy=SamplingStrategy.WEIGHTED,
    dataloader_weights=weights,
    dataloader_names=["majority", "medium", "minority"],
    epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
    steps_per_epoch=1000,  # Fixed epoch length
)
```

## Advanced Configurations

### Mixed Strategies

Combine different strategies for complex scenarios:

```python
# Start with sequential, then switch to weighted
class AdaptiveStrategy(TrainerHooks):
    def __init__(self, switch_epoch=10):
        self.switch_epoch = switch_epoch

    def on_epoch_start(self, trainer, epoch):
        if epoch == self.switch_epoch:
            # Switch from sequential to weighted
            trainer.multi_loader_config.sampling_strategy = SamplingStrategy.WEIGHTED
            trainer.multi_loader_config.dataloader_weights = [0.5, 0.3, 0.2]
            trainer.multi_loader_manager.rebuild_schedule()
```

### Dynamic Dataloader Addition

Add dataloaders during training:

```python
class DynamicLoaderCallback(TrainerHooks):
    def on_epoch_end(self, trainer, epoch):
        if epoch == 20 and not hasattr(self, "added_hard"):
            # Add hard negatives after 20 epochs
            hard_loader = create_hard_negatives_loader()
            trainer.add_dataloader(
                loader=hard_loader,
                name="hard_negatives",
                weight=0.2
            )
            self.added_hard = True
```

### Per-Loader Learning Rates

Different learning rates for different datasets:

```python
# Custom scheduler per loader
class PerLoaderScheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers  # {loader_name: scheduler}

    def step(self, loader_name):
        if loader_name in self.schedulers:
            self.schedulers[loader_name].step()

# In training step
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    # Adjust learning rate based on loader
    lr_scale = {"easy": 1.0, "medium": 0.5, "hard": 0.1}
    for param_group in trainer.optimizers[0].param_groups:
        param_group['lr'] *= lr_scale[dataloader_name]

    # ... rest of training

    # Reset learning rate
    for param_group in trainer.optimizers[0].param_groups:
        param_group['lr'] /= lr_scale[dataloader_name]
```

## Best Practices

### 1. Choose the Right Strategy

| Use Case | Recommended Strategy | Why |
|----------|---------------------|-----|
| Balanced multi-domain | ROUND_ROBIN | Fair representation |
| Imbalanced datasets | WEIGHTED | Control sampling ratio |
| Curriculum learning | ALTERNATING or SEQUENTIAL | Controlled progression |
| Pre-training + Fine-tuning | SEQUENTIAL | Clear stages |
| Multi-task learning | WEIGHTED | Task importance |

### 2. Monitor Loader Proportions

```python
config = GenericTrainerConfig(
    logging=LoggingConfig(
        log_loader_proportions=True,  # Track actual vs expected
    )
)

# Metrics will include:
# - loader_proportions/loader1_batches: actual proportion
# - loader_proportions/loader1_divergence: |actual - expected|
```

### 3. Handle Exhaustion Properly

```python
config = MultiDataLoaderConfig(
    cycle_short_loaders=True,  # Restart exhausted loaders
    drop_last=False,  # Use all data
    epoch_length_policy=EpochLengthPolicy.MAX_LENGTH,  # Continue until all exhausted
)
```

### 4. Validation Strategy Should Match Training

```python
# If training uses WEIGHTED, validation should too
train_config = MultiDataLoaderConfig(
    sampling_strategy=SamplingStrategy.WEIGHTED,
    dataloader_weights=[0.7, 0.3],
)

val_config = ValidationConfig(
    aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
    # Or use same weights as training
    validation_weights=[0.7, 0.3],
)
```

### 5. Test Resume Behavior

```python
# Always test that resume works correctly
def test_deterministic_resume():
    # Train for 5 steps
    trainer.fit(loaders, max_steps=5)
    state1 = trainer.get_state()

    # Save checkpoint
    trainer.save_checkpoint("test.ckpt")

    # Continue for 5 more steps
    trainer.fit(loaders, max_steps=10)
    state2 = trainer.get_state()

    # Load checkpoint and train same 5 steps
    trainer.load_checkpoint("test.ckpt")
    trainer.fit(loaders, max_steps=10)
    state3 = trainer.get_state()

    # Should be identical
    assert state2 == state3
```

## Troubleshooting

### Common Issues

1. **Loader exhaustion with ROUND_ROBIN**
   - Set `cycle_short_loaders=True` to restart exhausted loaders
   - Or use `EpochLengthPolicy.MIN_LENGTH` to stop when first exhausts

2. **Unbalanced validation metrics**
   - Check `ValAggregation` strategy matches your needs
   - Consider per-loader metrics: `per_loader_metrics=True`

3. **Non-deterministic scheduling**
   - Ensure all loaders have deterministic samplers
   - Set seeds properly: `torch.manual_seed(42)`
   - Use `fault_tolerance.verify_deterministic_resume=True`

4. **Memory issues with multiple loaders**
   - Reduce `num_workers` per loader
   - Use `persistent_workers=False` to free memory
   - Consider gradient accumulation to reduce batch size

## See Also

- [Migration Guide](MIGRATION.md) - Migrating to multi-loader API
- [API Reference](API.md#multi-dataloader) - Complete API docs
- [Example Code](../demo/example3_production/) - Working examples
