"""
Tests for Hooks System

This module tests the hooks functionality including:
- Hook registration and execution
- Hook call order
- Error handling in hooks
- Built-in hook implementations
"""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from model_training_framework.trainer.hooks import (
    EarlyStoppingHook,
    GradientMonitorHook,
    HookManager,
    LoggingHook,
    ModelCheckpointHook,
    TrainerHooks,
)


class HookForTests(TrainerHooks):
    """Test hook implementation for testing."""

    def __init__(self):
        self.calls = []

    def on_train_start(self, trainer):
        self.calls.append(("on_train_start", trainer))

    def on_train_end(self, trainer):
        self.calls.append(("on_train_end", trainer))

    def on_epoch_start(self, trainer, epoch):
        self.calls.append(("on_epoch_start", trainer, epoch))

    def on_epoch_end(self, trainer, epoch, metrics):
        self.calls.append(("on_epoch_end", trainer, epoch, metrics))

    def on_train_batch_start(self, trainer, batch, loader_idx, loader_name):
        self.calls.append(
            ("on_train_batch_start", trainer, batch, loader_idx, loader_name)
        )

    def on_train_batch_end(self, trainer, batch, loader_idx, loader_name, metrics):
        self.calls.append(
            ("on_train_batch_end", trainer, batch, loader_idx, loader_name, metrics)
        )


class ErrorHook(TrainerHooks):
    """Hook that raises errors for testing error handling."""

    def on_train_start(self, trainer):
        raise RuntimeError("Test error in hook")


class TestHookManager:
    """Test HookManager class."""

    def test_hook_registration(self):
        """Test registering hooks."""
        manager = HookManager()
        hook = HookForTests()

        manager.register_hook(hook)

        assert len(manager.hooks) == 1
        assert manager.hooks[0] is hook

    def test_register_multiple_hooks(self):
        """Test registering multiple hooks."""
        manager = HookManager()
        hook1 = HookForTests()
        hook2 = HookForTests()

        manager.register_hooks([hook1, hook2])

        assert len(manager.hooks) == 2
        assert manager.hooks[0] is hook1
        assert manager.hooks[1] is hook2

    def test_register_invalid_hook_raises(self):
        """Test registering invalid hook raises error."""
        manager = HookManager()

        with pytest.raises(TypeError, match="must inherit from TrainerHooks"):
            manager.register_hook(cast("Any", "not a hook"))

    def test_call_hook(self):
        """Test calling hooks."""
        manager = HookManager()
        hook = HookForTests()
        manager.register_hook(hook)

        trainer = MagicMock()
        manager.call_hook("on_train_start", trainer)

        assert len(hook.calls) == 1
        assert hook.calls[0] == ("on_train_start", trainer)

    def test_call_hook_with_multiple_hooks(self):
        """Test calling multiple hooks in order."""
        manager = HookManager()
        hook1 = HookForTests()
        hook2 = HookForTests()
        manager.register_hooks([hook1, hook2])

        trainer = MagicMock()
        manager.call_hook("on_train_start", trainer)

        # Both hooks should be called
        assert len(hook1.calls) == 1
        assert len(hook2.calls) == 1

        # Verify order - hook1 before hook2
        assert hook1.calls[0] == ("on_train_start", trainer)
        assert hook2.calls[0] == ("on_train_start", trainer)

    def test_call_nonexistent_hook_method(self):
        """Test calling non-existent hook method is safe."""
        manager = HookManager()
        hook = HookForTests()
        manager.register_hook(hook)

        # Should not raise
        manager.call_hook("nonexistent_method", MagicMock())

        # Hook should not have been called
        assert len(hook.calls) == 0

    def test_error_handling_in_hooks(self, caplog):
        """Test error handling when hook raises exception."""
        manager = HookManager()
        error_hook = ErrorHook()
        good_hook = HookForTests()

        manager.register_hooks([error_hook, good_hook])

        trainer = MagicMock()
        # Should not raise despite error in first hook
        manager.call_hook("on_train_start", trainer)

        # Good hook should still be called
        assert len(good_hook.calls) == 1

        # Error should be logged
        assert "Error in ErrorHook.on_train_start" in caplog.text

    def test_clear_hooks(self):
        """Test clearing all hooks."""
        manager = HookManager()
        manager.register_hooks([HookForTests(), HookForTests()])

        assert len(manager.hooks) == 2

        manager.clear_hooks()

        assert len(manager.hooks) == 0


class TestLoggingHook:
    """Test LoggingHook implementation."""

    def test_logging_hook_creation(self):
        """Test LoggingHook can be created."""
        hook = LoggingHook(log_level="DEBUG")
        assert hook is not None

    def test_logging_hook_on_train_start(self, caplog):
        """Test LoggingHook logs training start."""
        hook = LoggingHook(log_level="INFO")

        trainer = MagicMock()
        trainer.config.max_epochs = 10

        hook.on_train_start(trainer)

        assert "Training started" in caplog.text

    def test_logging_hook_on_epoch_start(self, caplog):
        """Test LoggingHook logs epoch start."""
        hook = LoggingHook(log_level="INFO")

        trainer = MagicMock()
        trainer.config.max_epochs = 10

        hook.on_epoch_start(trainer, epoch=3)

        assert "Starting epoch 4" in caplog.text

    def test_logging_hook_on_epoch_end(self, caplog):
        """Test LoggingHook logs epoch end."""
        hook = LoggingHook(log_level="INFO")

        trainer = MagicMock()
        metrics = {"train/loss": 0.5}

        hook.on_epoch_end(trainer, epoch=3, metrics=metrics)

        assert "Completed epoch 4" in caplog.text
        assert "loss: 0.5" in caplog.text


class TestGradientMonitorHook:
    """Test GradientMonitorHook implementation."""

    def test_gradient_monitor_creation(self):
        """Test GradientMonitorHook can be created."""
        hook = GradientMonitorHook(log_frequency=50)
        assert hook is not None
        assert hook.log_frequency == 50

    def test_gradient_monitor_logging(self, caplog):
        """Test GradientMonitorHook logs gradient stats."""
        hook = GradientMonitorHook(log_frequency=2)

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 1),
        )

        # Simulate gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 0.01

        trainer = MagicMock()
        trainer.model = model

        # Call hook multiple times
        hook.on_after_backward(trainer)  # Step 1
        hook.on_after_backward(trainer)  # Step 2 - should log

        assert "Step 2 - Gradient stats" in caplog.text
        assert "max=" in caplog.text
        assert "min=" in caplog.text
        assert "avg=" in caplog.text


class TestModelCheckpointHook:
    """Test ModelCheckpointHook implementation."""

    def test_model_checkpoint_creation(self):
        """Test ModelCheckpointHook can be created."""
        hook = ModelCheckpointHook(save_top_k=5, monitor="val/loss")
        assert hook is not None
        assert hook.save_top_k == 5
        assert hook.monitor == "val/loss"

    def test_checkpoint_save_logging(self, caplog):
        """Test ModelCheckpointHook logs checkpoint saves."""
        hook = ModelCheckpointHook()

        trainer = MagicMock()
        hook.on_checkpoint_save(trainer, "/path/to/checkpoint.pt")

        assert "Checkpoint saved: /path/to/checkpoint.pt" in caplog.text

    def test_epoch_end_monitoring(self, caplog):
        """Test ModelCheckpointHook monitors metrics."""
        hook = ModelCheckpointHook(monitor="val/loss")

        trainer = MagicMock()
        metrics = {"val/loss": 0.25}

        hook.on_epoch_end(trainer, epoch=5, metrics=metrics)

        assert "Epoch 6: val/loss=0.2500" in caplog.text


class TestEarlyStoppingHook:
    """Test EarlyStoppingHook implementation."""

    def test_early_stopping_creation(self):
        """Test EarlyStoppingHook can be created."""
        hook = EarlyStoppingHook(
            monitor="val/loss",
            patience=5,
            mode="min",
            min_delta=0.001,
        )
        assert hook is not None
        assert hook.monitor == "val/loss"
        assert hook.patience == 5
        assert hook.mode == "min"
        assert hook.min_delta == 0.001

    def test_early_stopping_improvement_min_mode(self, caplog):
        """Test early stopping detects improvement in min mode."""
        hook = EarlyStoppingHook(
            monitor="val/loss",
            patience=3,
            mode="min",
            min_delta=0.01,
        )

        trainer = MagicMock()

        # Initial score
        hook.on_epoch_end(trainer, epoch=0, metrics={"val/loss": 1.0})
        assert hook.counter == 0

        # Improvement
        hook.on_epoch_end(trainer, epoch=1, metrics={"val/loss": 0.8})
        assert hook.counter == 0
        assert "Improvement detected" in caplog.text

        # No improvement (not enough delta)
        hook.on_epoch_end(trainer, epoch=2, metrics={"val/loss": 0.795})
        assert hook.counter == 1
        assert "No improvement for 1 epochs" in caplog.text

    def test_early_stopping_improvement_max_mode(self):
        """Test early stopping detects improvement in max mode."""
        hook = EarlyStoppingHook(
            monitor="val/accuracy",
            patience=3,
            mode="max",
            min_delta=0.01,
        )

        trainer = MagicMock()

        # Initial score
        hook.on_epoch_end(trainer, epoch=0, metrics={"val/accuracy": 0.8})
        assert hook.counter == 0

        # Improvement
        hook.on_epoch_end(trainer, epoch=1, metrics={"val/accuracy": 0.85})
        assert hook.counter == 0

        # No improvement
        hook.on_epoch_end(trainer, epoch=2, metrics={"val/accuracy": 0.84})
        assert hook.counter == 1

    def test_early_stopping_triggered(self, caplog):
        """Test early stopping is triggered after patience."""
        hook = EarlyStoppingHook(
            monitor="val/loss",
            patience=2,
            mode="min",
        )

        trainer = MagicMock()

        # Initial score
        hook.on_epoch_end(trainer, epoch=0, metrics={"val/loss": 1.0})

        # No improvements
        hook.on_epoch_end(trainer, epoch=1, metrics={"val/loss": 1.1})
        assert hook.counter == 1
        assert not hook.should_stop

        hook.on_epoch_end(trainer, epoch=2, metrics={"val/loss": 1.2})
        assert hook.counter == 2
        assert hook.should_stop
        assert "Early stopping triggered after epoch 3" in caplog.text

    def test_early_stopping_missing_metric(self):
        """Test early stopping handles missing metric gracefully."""
        hook = EarlyStoppingHook(monitor="val/loss")

        trainer = MagicMock()

        # Missing metric - should not update
        hook.on_epoch_end(trainer, epoch=0, metrics={"train/loss": 0.5})

        assert hook.best_score is None
        assert hook.counter == 0
        assert not hook.should_stop
