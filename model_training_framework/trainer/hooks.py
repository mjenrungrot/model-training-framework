"""
Hooks System for Training Framework

This module provides a flexible hook system for injecting custom behavior
at various points in the training lifecycle:
- Training start/end
- Epoch boundaries
- Batch processing
- Optimizer steps
- Checkpoint operations
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from model_training_framework.trainer.core import GenericTrainer

logger = logging.getLogger(__name__)


class TrainerHooks:
    """
    Base class for implementing training hooks.

    Subclass this to inject custom behavior at various training lifecycle points.
    All hook methods are optional - implement only the ones you need.
    """

    def on_train_start(self, trainer: GenericTrainer) -> None:
        """
        Called at the beginning of training.

        Args:
            trainer: The trainer instance
        """

    def on_train_end(self, trainer: GenericTrainer) -> None:
        """
        Called at the end of training.

        Args:
            trainer: The trainer instance
        """

    def on_epoch_start(self, trainer: GenericTrainer, epoch: int) -> None:
        """
        Called at the beginning of each epoch.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number (0-based)
        """

    def on_epoch_end(
        self,
        trainer: GenericTrainer,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        """
        Called at the end of each epoch.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number (0-based)
            metrics: Metrics for the completed epoch
        """

    def on_train_batch_start(
        self,
        trainer: GenericTrainer,
        batch: Any,
        loader_idx: int,
        loader_name: str,
    ) -> None:
        """
        Called before processing a training batch.

        Args:
            trainer: The trainer instance
            batch: The input batch
            loader_idx: Index of the current dataloader
            loader_name: Name of the current dataloader
        """

    def on_train_batch_end(
        self,
        trainer: GenericTrainer,
        batch: Any,
        loader_idx: int,
        loader_name: str,
        metrics: dict[str, Any],
    ) -> None:
        """
        Called after processing a training batch.

        Args:
            trainer: The trainer instance
            batch: The input batch
            loader_idx: Index of the current dataloader
            loader_name: Name of the current dataloader
            metrics: Metrics from the batch
        """

    def on_validation_start(self, trainer: GenericTrainer, epoch: int) -> None:
        """
        Called before validation begins.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
        """

    def on_validation_end(
        self,
        trainer: GenericTrainer,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        """
        Called after validation completes.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            metrics: Validation metrics
        """

    def on_validation_batch_start(
        self,
        trainer: GenericTrainer,
        batch: Any,
        loader_idx: int,
        loader_name: str,
    ) -> None:
        """
        Called before processing a validation batch.

        Args:
            trainer: The trainer instance
            batch: The input batch
            loader_idx: Index of the current dataloader
            loader_name: Name of the current dataloader
        """

    def on_validation_batch_end(
        self,
        trainer: GenericTrainer,
        batch: Any,
        loader_idx: int,
        loader_name: str,
        metrics: dict[str, Any],
    ) -> None:
        """
        Called after processing a validation batch.

        Args:
            trainer: The trainer instance
            batch: The input batch
            loader_idx: Index of the current dataloader
            loader_name: Name of the current dataloader
            metrics: Metrics from the batch
        """

    def on_before_optimizer_step(
        self,
        trainer: GenericTrainer,
        optimizer_idx: int,
    ) -> None:
        """
        Called before an optimizer step.

        Args:
            trainer: The trainer instance
            optimizer_idx: Index of the optimizer being stepped
        """

    def on_after_optimizer_step(
        self,
        trainer: GenericTrainer,
        optimizer_idx: int,
    ) -> None:
        """
        Called after an optimizer step.

        Args:
            trainer: The trainer instance
            optimizer_idx: Index of the optimizer that was stepped
        """

    def on_before_backward(
        self,
        trainer: GenericTrainer,
        loss: Tensor,
    ) -> None:
        """
        Called before the backward pass.

        Args:
            trainer: The trainer instance
            loss: The loss tensor
        """

    def on_after_backward(self, trainer: GenericTrainer) -> None:
        """
        Called after the backward pass.

        Args:
            trainer: The trainer instance
        """

    def on_checkpoint_save(
        self,
        trainer: GenericTrainer,
        checkpoint_path: str,
    ) -> None:
        """
        Called after saving a checkpoint.

        Args:
            trainer: The trainer instance
            checkpoint_path: Path where checkpoint was saved
        """

    def on_checkpoint_load(
        self,
        trainer: GenericTrainer,
        checkpoint_path: str,
    ) -> None:
        """
        Called after loading a checkpoint.

        Args:
            trainer: The trainer instance
            checkpoint_path: Path from which checkpoint was loaded
        """

    def on_gradient_clip(
        self,
        trainer: GenericTrainer,
        grad_norm: float,
    ) -> None:
        """
        Called after gradient clipping.

        Args:
            trainer: The trainer instance
            grad_norm: The computed gradient norm
        """


class HookManager:
    """
    Manages the execution of hooks during training.

    Ensures hooks are called in order and handles exceptions gracefully
    to prevent training disruption.
    """

    def __init__(self):
        """Initialize the hook manager."""
        self.hooks: list[TrainerHooks] = []

    def register_hook(self, hook: TrainerHooks) -> None:
        """
        Register a hook instance.

        Args:
            hook: Hook instance to register
        """
        if not isinstance(hook, TrainerHooks):
            raise TypeError(f"Hook must inherit from TrainerHooks, got {type(hook)}")
        self.hooks.append(hook)
        logger.info(f"Registered hook: {hook.__class__.__name__}")

    def register_hooks(self, hooks: list[TrainerHooks]) -> None:
        """
        Register multiple hook instances.

        Args:
            hooks: List of hook instances to register
        """
        for hook in hooks:
            self.register_hook(hook)

    def call_hook(self, hook_name: str, *args, **kwargs) -> None:
        """
        Call a specific hook on all registered hook instances.

        Args:
            hook_name: Name of the hook method to call
            *args: Positional arguments to pass to the hook
            **kwargs: Keyword arguments to pass to the hook
        """
        for hook in self.hooks:
            try:
                method = getattr(hook, hook_name, None)
                if method is not None and callable(method):
                    method(*args, **kwargs)
            except Exception:
                logger.exception(
                    f"Error in {hook.__class__.__name__}.{hook_name}, continuing..."
                )

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        self.hooks.clear()
        logger.info("Cleared all hooks")


# Example hook implementations


class LoggingHook(TrainerHooks):
    """Example hook that adds detailed logging."""

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize logging hook.

        Args:
            log_level: Logging level to use
        """
        self.logger = logging.getLogger(f"{__name__}.LoggingHook")
        self.logger.setLevel(getattr(logging, log_level))

    def on_train_start(self, trainer: GenericTrainer) -> None:
        """Log training start."""
        self.logger.info(f"Training started with {trainer.config.max_epochs} epochs")

    def on_epoch_start(self, trainer: GenericTrainer, epoch: int) -> None:
        """Log epoch start."""
        self.logger.info(f"Starting epoch {epoch + 1}/{trainer.config.max_epochs}")

    def on_epoch_end(
        self,
        trainer: GenericTrainer,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        """Log epoch completion."""
        loss = metrics.get("train/loss", "N/A")
        self.logger.info(f"Completed epoch {epoch + 1}, loss: {loss}")


class GradientMonitorHook(TrainerHooks):
    """Hook for monitoring gradient statistics."""

    def __init__(self, log_frequency: int = 100, param_filter: list[str] | None = None):
        """
        Initialize gradient monitor.

        Args:
            log_frequency: How often to log gradient stats (in steps)
            param_filter: Optional list of parameter name patterns to monitor
        """
        self.log_frequency = log_frequency
        self.param_filter = param_filter
        self.step_count = 0

    def on_after_backward(self, trainer: GenericTrainer) -> None:
        """Monitor gradients after backward pass."""
        self.step_count += 1

        if self.step_count % self.log_frequency == 0:
            grad_norms = []
            for name, param in trainer.model.named_parameters():
                if param.grad is not None:
                    # Apply filter if specified
                    if self.param_filter and not any(
                        pattern in name for pattern in self.param_filter
                    ):
                        continue
                    grad_norms.append((name, param.grad.norm().item()))

            if grad_norms:
                max_grad = max(grad_norms, key=lambda x: x[1])
                min_grad = min(grad_norms, key=lambda x: x[1])
                avg_grad = sum(g[1] for g in grad_norms) / len(grad_norms)

                logger.info(
                    f"Step {self.step_count} - Gradient stats: "
                    f"max={max_grad[0]}:{max_grad[1]:.4f}, "
                    f"min={min_grad[0]}:{min_grad[1]:.4f}, "
                    f"avg={avg_grad:.4f}"
                )


class ModelCheckpointHook(TrainerHooks):
    """Hook for custom checkpoint behavior."""

    def __init__(self, save_top_k: int = 3, monitor: str = "val/loss"):
        """
        Initialize checkpoint hook.

        Args:
            save_top_k: Number of best checkpoints to keep
            monitor: Metric to monitor for best checkpoint
        """
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.best_scores: list[tuple[float, str]] = []

    def on_checkpoint_save(
        self,
        trainer: GenericTrainer,
        checkpoint_path: str,
    ) -> None:
        """Track saved checkpoints."""
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def on_epoch_end(
        self,
        trainer: GenericTrainer,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        """Check if this is a top-k checkpoint."""
        if self.monitor in metrics:
            score = metrics[self.monitor]
            logger.info(f"Epoch {epoch + 1}: {self.monitor}={score:.4f}")


class EarlyStoppingHook(TrainerHooks):
    """Hook for early stopping based on validation metrics."""

    def __init__(
        self,
        monitor: str = "val/loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0001,
    ):
        """
        Initialize early stopping hook.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            mode: "min" for lower is better, "max" for higher is better
            min_delta: Minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(
        self,
        trainer: GenericTrainer,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        """Check for early stopping condition."""
        if self.monitor not in metrics:
            return

        score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = score
        else:
            improved = False
            if self.mode == "min":
                improved = score < (self.best_score - self.min_delta)
            else:
                improved = score > (self.best_score + self.min_delta)

            if improved:
                self.best_score = score
                self.counter = 0
                logger.info(f"Improvement detected: {self.monitor}={score:.4f}")
            else:
                self.counter += 1
                logger.info(
                    f"No improvement for {self.counter} epochs "
                    f"(best: {self.best_score:.4f}, current: {score:.4f})"
                )

                if self.counter >= self.patience:
                    self.should_stop = True
                    logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                    # Note: Actual stopping should be handled by trainer
