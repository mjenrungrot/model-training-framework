# Hooks System Documentation

The Model Training Framework provides a comprehensive hooks system for injecting custom behavior at various points in the training lifecycle.

## Table of Contents

1. [Overview](#overview)
2. [Built-in Hooks](#built-in-hooks)
3. [Custom Hooks](#custom-hooks)
4. [Hook Lifecycle](#hook-lifecycle)
5. [Advanced Usage](#advanced-usage)
6. [Best Practices](#best-practices)

## Overview

Hooks allow you to:

- Monitor training progress
- Implement custom logging
- Add early stopping conditions
- Modify training behavior dynamically
- Integrate external tools
- Debug training issues

## Built-in Hooks

### Logging Hook

Basic logging of training progress:

```python
from model_training_framework.config.schemas import HooksConfig

config = HooksConfig(
    enable_logging_hook=True
)
```

### Gradient Monitor Hook

Monitor gradient statistics during training:

```python
config = HooksConfig(
    enable_gradient_monitor=True,
    gradient_monitor_config={
        "log_frequency": 100,  # Log every N steps
        "log_histogram": True,  # Log gradient histograms
        "log_norm": True,  # Log gradient norms
        "param_filter": ["encoder", "decoder"],  # Monitor specific layers
        "percentiles": [10, 50, 90],  # Gradient percentiles to track
    }
)
```

The gradient monitor provides insights into:

- Gradient magnitudes
- Gradient flow through network
- Vanishing/exploding gradients
- Layer-wise gradient statistics

### Model Checkpoint Hook

Advanced checkpointing behavior:

```python
config = HooksConfig(
    enable_model_checkpoint_hook=True,
    model_checkpoint_config={
        "save_top_k": 3,  # Keep best 3 checkpoints
        "monitor": "val/loss",  # Metric to monitor
        "mode": "min",  # min for loss, max for accuracy
        "save_last": True,  # Always keep last checkpoint
        "save_on_train_epoch_end": False,  # Only save after validation
        "filename": "epoch_{epoch}_loss_{val/loss:.2f}",
        "auto_insert_metric_name": False,
    }
)
```

### Early Stopping Hook

Stop training when metrics stop improving:

```python
config = HooksConfig(
    enable_early_stopping_hook=True,
    early_stopping_config={
        "monitor": "val/loss",
        "patience": 10,  # Epochs without improvement
        "mode": "min",  # min for loss, max for accuracy
        "min_delta": 0.0001,  # Minimum change to qualify as improvement
        "check_on_train_epoch_end": False,  # Check after validation
        "strict": True,  # Crash if monitor metric not found
        "verbose": True,  # Print early stopping messages
    }
)
```

### Learning Rate Monitor Hook

Track learning rate changes:

```python
config = HooksConfig(
    enable_lr_monitor=True,
    lr_monitor_config={
        "log_momentum": True,  # Also log momentum if available
        "log_weight_decay": True,  # Log weight decay
        "frequency": "epoch",  # "step" or "epoch"
    }
)
```

## Custom Hooks

### Basic Custom Hook

Create custom hooks by subclassing `TrainerHooks`:

```python
from model_training_framework.trainer.hooks import TrainerHooks

class MyCustomHook(TrainerHooks):
    def __init__(self, threshold=10.0):
        super().__init__()
        self.threshold = threshold
        self.high_loss_count = 0

    def on_train_start(self, trainer):
        """Called once at the beginning of training."""
        print("Training started!")
        self.start_time = time.time()

    def on_epoch_start(self, trainer, epoch):
        """Called at the start of each epoch."""
        print(f"Starting epoch {epoch}")
        self.epoch_start_time = time.time()

    def on_train_batch_end(self, trainer, batch, loader_idx, loader_name, metrics):
        """Called after each training batch."""
        if metrics.get("loss", 0) > self.threshold:
            self.high_loss_count += 1
            print(f"High loss detected: {metrics['loss']:.4f}")

            if self.high_loss_count > 10:
                print("Too many high loss batches, reducing learning rate")
                for optimizer in trainer.optimizers:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5

    def on_validation_end(self, trainer, epoch, metrics):
        """Called after validation."""
        print(f"Validation metrics: {metrics}")

        # Custom metric computation
        if "val/accuracy" in metrics:
            improvement = metrics["val/accuracy"] - trainer.best_accuracy
            print(f"Accuracy improvement: {improvement:.4f}")
            trainer.best_accuracy = max(trainer.best_accuracy, metrics["val/accuracy"])

    def on_training_end(self, trainer):
        """Called once at the end of training."""
        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.2f} seconds")
```

### Registering Custom Hooks

Three ways to register hooks:

```python
# 1. Via configuration (using class path)
config = HooksConfig(
    hook_classes=["mypackage.hooks.MyCustomHook"],
    hook_configs={
        "mypackage.hooks.MyCustomHook": {"threshold": 5.0}
    }
)

# 2. Direct registration after trainer creation
trainer = GenericTrainer(config, model, optimizers)
trainer.hook_manager.register_hook(MyCustomHook(threshold=5.0))

# 3. Register a class explicitly
class AutoRegisteredHook(TrainerHooks):
    def on_epoch_end(self, trainer, epoch):
        print(f"Epoch {epoch} completed")

trainer.hook_manager.register_hook(AutoRegisteredHook())
```

## Hook Lifecycle

### Complete Hook Interface

```python
class TrainerHooks:
    """Base class for all trainer hooks."""

    # Training lifecycle
    def on_train_start(self, trainer):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass

    # Epoch boundaries
    def on_epoch_start(self, trainer, epoch):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer, epoch):
        """Called at the end of each epoch."""
        pass

    # Training batches
    def on_train_batch_start(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name):
        """Called before processing a training batch."""
        pass

    def on_train_batch_end(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name, metrics):
        """Called after processing a training batch."""
        pass

    # Validation
    def on_validation_start(self, trainer, epoch):
        """Called before validation."""
        pass

    def on_validation_end(self, trainer, epoch, metrics):
        """Called after validation."""
        pass

    def on_validation_batch_start(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name):
        """Called before processing a validation batch."""
        pass

    def on_validation_batch_end(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name, metrics):
        """Called after processing a validation batch."""
        pass

    # Gradient computation
    def on_before_backward(self, trainer, loss):
        """Called before backward pass."""
        pass

    def on_after_backward(self, trainer):
        """Called after backward pass."""
        pass

    # Optimization
    def on_before_optimizer_step(self, trainer, optimizer_idx):
        """Called before optimizer step."""
        pass

    def on_after_optimizer_step(self, trainer, optimizer_idx):
        """Called after optimizer step."""
        pass

    # Gradient clipping
    def on_gradient_clip(self, trainer, grad_norm):
        """Called after gradient clipping."""
        pass

    # Checkpointing
    def on_checkpoint_save(self, trainer, checkpoint_path):
        """Called after saving a checkpoint."""
        pass

    def on_checkpoint_load(self, trainer, checkpoint_path):
        """Called after loading a checkpoint."""
        pass

    # Exception handling
    def on_exception(self, trainer, exception):
        """Called when an exception occurs."""
        pass
```

### Execution Order

Hooks are executed in registration order:

1. User-provided custom hooks (first)
2. Built-in hooks (last)

This ensures user hooks can override or enhance built-in behavior.

```python
# Execution order example
trainer.hook_manager.register_hook(CustomHook1())  # Executes first
trainer.hook_manager.register_hook(CustomHook2())  # Executes second
# Built-in hooks execute last
```

## Advanced Usage

### Stateful Hooks

Hooks can maintain state across calls:

```python
class StatefulHook(TrainerHooks):
    def __init__(self):
        super().__init__()
        self.batch_losses = []
        self.epoch_losses = []

    def on_train_batch_end(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name, metrics):
        self.batch_losses.append(metrics.get("loss", 0))

    def on_epoch_end(self, trainer, epoch):
        avg_loss = sum(self.batch_losses) / len(self.batch_losses)
        self.epoch_losses.append(avg_loss)
        self.batch_losses = []  # Reset for next epoch

        # Detect training plateau
        if len(self.epoch_losses) > 5:
            recent_losses = self.epoch_losses[-5:]
            if max(recent_losses) - min(recent_losses) < 0.001:
                print("Training plateau detected!")
```

### Hook Communication

Hooks can communicate through the trainer:

```python
class ProducerHook(TrainerHooks):
    def on_epoch_end(self, trainer, epoch):
        # Store data for other hooks
        trainer.hook_data = {"important_metric": 0.95}

class ConsumerHook(TrainerHooks):
    def on_epoch_end(self, trainer, epoch):
        # Read data from other hooks
        if hasattr(trainer, "hook_data"):
            metric = trainer.hook_data.get("important_metric")
            print(f"Received metric: {metric}")
```

### Conditional Hooks

Enable/disable hooks based on conditions:

```python
class ConditionalHook(TrainerHooks):
    def __init__(self, enable_after_epoch=10):
        super().__init__()
        self.enable_after_epoch = enable_after_epoch
        self.enabled = False

    def on_epoch_start(self, trainer, epoch):
        if epoch >= self.enable_after_epoch:
            self.enabled = True

    def on_train_batch_end(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name, metrics):
        if not self.enabled:
            return

        # Hook logic only runs after specified epoch
        self._analyze_metrics(metrics)
```

### External Integration Hooks

Integrate with external services:

```python
class SlackNotificationHook(TrainerHooks):
    def __init__(self, webhook_url, notify_on=["training_end", "best_model"]):
        super().__init__()
        self.webhook_url = webhook_url
        self.notify_on = notify_on
        self.best_metric = float('inf')

    def on_validation_end(self, trainer, epoch, metrics):
        val_loss = metrics.get("val/loss", float('inf'))
        if val_loss < self.best_metric:
            self.best_metric = val_loss
            if "best_model" in self.notify_on:
                self._send_notification(
                    f"New best model! Epoch {epoch}, Loss: {val_loss:.4f}"
                )

    def on_training_end(self, trainer):
        if "training_end" in self.notify_on:
            self._send_notification(
                f"Training completed! Best loss: {self.best_metric:.4f}"
            )

    def _send_notification(self, message):
        import requests
        try:
            # Add timeout to prevent hanging
            requests.post(self.webhook_url, json={"text": message}, timeout=10)
        except requests.exceptions.Timeout:
            print(f"Warning: Webhook notification timed out")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to send webhook notification: {e}")
```

### Debugging Hooks

Hooks for debugging training issues:

```python
class DebugHook(TrainerHooks):
    def __init__(self, break_on_nan=True, log_shapes=False):
        super().__init__()
        self.break_on_nan = break_on_nan
        self.log_shapes = log_shapes

    def on_train_batch_start(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name):
        if self.log_shapes:
            for i, item in enumerate(batch):
                if torch.is_tensor(item):
                    print(f"Batch item {i} shape: {item.shape}")

    def on_after_backward(self, trainer):
        # Check for NaN gradients
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                    if self.break_on_nan:
                        import pdb; pdb.set_trace()

    def on_train_batch_end(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name, metrics):
        # Check for NaN loss
        if torch.isnan(torch.tensor(metrics.get("loss", 0))):
            print(f"NaN loss at batch {batch_idx}")
            if self.break_on_nan:
                import pdb; pdb.set_trace()
```

## Best Practices

### 1. Hook Organization

```python
# Organize hooks by functionality
hooks/
├── __init__.py
├── monitoring.py      # Monitoring and logging hooks
├── optimization.py    # Learning rate, gradient hooks
├── checkpointing.py   # Custom checkpoint hooks
├── debugging.py       # Debug and profiling hooks
└── integration.py     # External service hooks
```

### 2. Hook Configuration

```python
# Use configuration for flexibility
class ConfigurableHook(TrainerHooks):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.frequency = config.get("frequency", 100)
        self.verbose = config.get("verbose", False)

    def should_run(self, step):
        return self.enabled and step % self.frequency == 0
```

### 3. Error Handling

```python
class RobustHook(TrainerHooks):
    def on_train_batch_end(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name, metrics):
        try:
            # Hook logic that might fail
            self._process_metrics(metrics)
        except Exception as e:
            # Log error but don't crash training
            print(f"Hook error (non-fatal): {e}")
            if trainer.config.debug:
                import traceback
                traceback.print_exc()
```

### 4. Performance Considerations

```python
class EfficientHook(TrainerHooks):
    def __init__(self, sample_rate=0.1):
        super().__init__()
        self.sample_rate = sample_rate

    def on_train_batch_end(self, trainer, batch, batch_idx, dataloader_idx, dataloader_name, metrics):
        # Only run for a sample of batches
        if random.random() > self.sample_rate:
            return

        # Expensive computation only on sampled batches
        self._expensive_analysis(metrics)
```

### 5. Testing Hooks

```python
def test_custom_hook():
    # Create mock trainer
    mock_trainer = Mock()
    mock_trainer.model = create_test_model()
    mock_trainer.optimizers = [torch.optim.Adam(mock_trainer.model.parameters())]

    # Test hook
    hook = MyCustomHook(threshold=5.0)

    # Test lifecycle methods
    hook.on_train_start(mock_trainer)
    assert hasattr(hook, 'start_time')

    # Test batch processing
    metrics = {"loss": 15.0}
    hook.on_train_batch_end(mock_trainer, None, 0, 0, "test", metrics)
    assert hook.high_loss_count == 1

    # Test threshold behavior
    for _ in range(10):
        hook.on_train_batch_end(mock_trainer, None, 0, 0, "test", metrics)

    # Check if learning rate was reduced
    assert mock_trainer.optimizers[0].param_groups[0]['lr'] < 0.001
```

## See Also

- [Observability Guide](OBSERVABILITY.md) - Logging and monitoring
- [API Reference](API.md#hooks) - Complete hooks API
- [Example Hooks](../demo/example3_production/) - Production examples
