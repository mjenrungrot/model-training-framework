# Migration Guide

## Migrating from Single-Loader to Multi-Loader API

The Model Training Framework uses a **multi-dataloader-only architecture**. Even single dataloader scenarios must use the multi-dataloader API with a list containing one loader. This unified design enables seamless scaling and consistent behavior across all use cases.

## Why Multi-Loader Only?

- **Consistency**: Same API whether using 1 or 10 dataloaders
- **Scalability**: Easy to add more dataloaders later
- **Features**: All multi-loader features available even for single loaders
- **Simplicity**: One API to learn and maintain

## Migration Examples

### Old Pattern (No Longer Supported)

```python
# ❌ Old single-loader pattern - will not work
trainer = Trainer(model, optimizer)
trainer.fit(train_loader, val_loader)

def training_step(batch):
    x, y = batch
    # ...
```

### New Pattern (Required)

```python
# ✅ New multi-loader pattern - required for all training
from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
)

config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["main"],  # Single name in list
    ),
    val_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["validation"],
    )
)

trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=[optimizer],  # Always use list
)

def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    # New signature with dataloader information
    x, y = batch
    # dataloader_idx will be 0, dataloader_name will be "main"
    # ...

trainer.fit(
    train_loaders=[train_loader],  # Always use list
    val_loaders=[val_loader],      # Always use list
    max_epochs=10
)
```

## Key Changes Summary

### 1. Always Use Lists

```python
# Before
trainer.fit(train_loader, val_loader)

# After
trainer.fit(
    train_loaders=[train_loader],  # List required
    val_loaders=[val_loader]       # List required
)
```

### 2. Optimizer as List

```python
# Before
trainer = Trainer(model, optimizer)

# After
trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=[optimizer]  # List required
)
```

### 3. Training Step Signature

```python
# Before
def training_step(batch):
    pass

# After
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    # batch_idx: 0-based index within this dataloader for the epoch
    # dataloader_idx: index of the current dataloader (0 for single loader)
    # dataloader_name: name of the current dataloader ("main" for single loader)
    pass
```

### 4. Validation Step Signature

```python
# Before
def validation_step(batch):
    pass

# After
def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    pass
```

### 5. MultiDataLoaderConfig Required

```python
# Always required, even for single loader
config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["main"],  # Define names for your loaders
    ),
    val_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["validation"],
    )
)
```

## Common Migration Patterns

### Simple Training Script

**Before:**

```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
trainer = Trainer()

for epoch in range(epochs):
    for batch in train_loader:
        loss = trainer.training_step(batch)
        # ...
```

**After:**

```python
from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
)

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())

config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["train"],
    ),
    val_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["validation"],
    )
)

trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=[optimizer]
)

def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    # Your training logic here
    return {"loss": loss}

trainer.set_training_step(training_step)
trainer.fit(
    train_loaders=[train_loader],
    max_epochs=epochs
)
```

### Custom Training Loop

**Before:**

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
```

**After:**

```python
config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["main"],
    ),
    val_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["validation"],
    )
)

trainer = GenericTrainer(config, model, [optimizer])

def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    loss = compute_loss(trainer.model, batch)
    return {"loss": loss}

trainer.set_training_step(training_step)
trainer.fit(train_loaders=[train_loader], max_epochs=num_epochs)
```

## Benefits After Migration

Once migrated, you can easily:

1. **Add More Dataloaders**: Simply extend the lists

   ```python
   trainer.fit(
       train_loaders=[loader1, loader2, loader3],
       val_loaders=[val1, val2, val3]
   )
   ```

2. **Use Advanced Sampling**: Change strategy without code changes

   ```python
   config.train_loader_config.sampling_strategy = SamplingStrategy.WEIGHTED
   config.train_loader_config.dataloader_weights = [0.5, 0.3, 0.2]
   ```

3. **Enable Multi-Task Learning**: Use different optimizers per loader

   ```python
   trainer = GenericTrainer(
       config=config,
       model=model,
       optimizers=[optimizer1, optimizer2]
   )
   ```

## Troubleshooting

### Common Issues

1. **TypeError: Expected list, got DataLoader**
   - Solution: Wrap your dataloader in a list: `[dataloader]`

2. **Missing dataloader_idx parameter**
   - Solution: Update your training_step signature to include all required parameters

3. **MultiDataLoaderConfig not provided**
   - Solution: Always include MultiDataLoaderConfig in your GenericTrainerConfig

4. **Optimizer not in list**
   - Solution: Always pass optimizers as a list, even for single optimizer

## Configuration Key Renames

When upgrading to this version, update the following configuration keys:

- `config.multi` → `config.train_loader_config` (for training loaders)
- `config.multi` → `config.val_loader_config` (for validation loaders)
- `performance.mixed_precision: "16-mixed"` → `performance.use_amp: true`
- `data.num_workers` → `performance.dataloader_num_workers`

These changes align naming with runtime behavior and clarify the separation between training and validation configurations.

## Performance Profiling Default Change (v2.0+)

**Breaking Change**: Performance profiling is now **enabled by default** (`profile_training: true`).

### What Changed

- **Before v2.0**: Profiling was opt-in (`profile_training: false` by default)
- **After v2.0**: Profiling is opt-out (`profile_training: true` by default)

### Why This Changed

- Most users benefit from profiling insights during development
- Overhead is minimal (< 2% even with GPU synchronization)
- Profiling metrics are automatically throttled to prevent log spam

### How to Disable

If you don't need profiling metrics (e.g., for production runs):

```yaml
# In your config.yaml
profile_training: false
```

Or programmatically:

```python
config = GenericTrainerConfig(
    profile_training=False,  # Explicitly disable
    # ... other config
)
```

### What Gets Profiled

When enabled, the following metrics are tracked:

- Data loading time (`profile/*/dl_*/time_data_ms`)
- Forward pass time (`profile/*/dl_*/time_forward_ms`)
- Backward pass time (`profile/train/dl_*/time_backward_ms`)

These metrics follow your `log_scalars_every_n_steps` frequency to prevent excessive logging.

## Need Help?

- Check the [examples](../demo/example3_production/) for working code
- Review the [API documentation](API.md)
- See the [Multi-DataLoader Guide](MULTI_DATALOADER.md) for advanced patterns
