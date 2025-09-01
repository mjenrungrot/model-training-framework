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

from contextlib import suppress
import logging
import time as _time
from typing import TYPE_CHECKING, Any

import torch

from .utils import ddp_is_primary, sanitize_metric_key_component

if TYPE_CHECKING:
    from torch import Tensor

    from .core import GenericTrainer

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

    def on_before_forward(self, trainer: GenericTrainer) -> None:
        """
        Called immediately before model forward (+ loss) computation.

        Args:
            trainer: The trainer instance
        """

    def on_after_forward(self, trainer: GenericTrainer, duration_ms: float) -> None:
        """
        Called immediately after model forward (+ loss) computation.

        Args:
            trainer: The trainer instance
            duration_ms: Duration of forward computation in milliseconds
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
        self.logger.info("Training started")

    def on_epoch_start(self, trainer: GenericTrainer, epoch: int) -> None:
        """Log epoch start."""
        self.logger.info(f"Starting epoch {epoch + 1}")

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
        self.monitor: str = monitor
        self.patience: int = patience
        self.mode: str = mode
        self.min_delta: float = min_delta

        self.best_score: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

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
            # Initialize best score on first observation
            try:
                self.best_score = float(score)
            except Exception:
                # If score isn't numeric, don't update
                return
            return

        # Compare according to mode (case-insensitive)
        mode = self.mode.lower()
        try:
            current = float(score)
            best = float(self.best_score)
        except Exception:
            return

        is_improved = False
        if mode == "min":
            is_improved = current < (best - self.min_delta)
        elif mode == "max":
            is_improved = current > (best + self.min_delta)

        if is_improved:
            self.best_score = current
            self.counter = 0
            logger.info(f"Improvement detected: {self.monitor}={current:.4f}")
        else:
            self.counter += 1
            logger.info(
                f"No improvement for {self.counter} epochs "
                f"(best: {best:.4f}, current: {current:.4f})"
            )

        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after epoch {epoch + 1}")
            # Note: Actual stopping should be handled by trainer


class RuntimeProfilingHook(TrainerHooks):
    """Lightweight runtime profiling hook for optimizer and batch wait timing.

    - Times optimizer steps via on_before/after_optimizer_step
    - Approximates dataloader wait time between batches
    - Respects log frequency and logs only on primary rank
    """

    def __init__(
        self,
        cuda_sync: bool = True,
        log_frequency: int | None = None,
        enable_batch_wait: bool = True,
    ) -> None:
        self.cuda_sync = cuda_sync
        self.log_frequency = log_frequency  # None -> use trainer.config.logging.log_scalars_every_n_steps
        self.enable_batch_wait = enable_batch_wait

        self._optim_t0: float | None = None
        self._last_batch_end_time: float | None = None
        self._last_logged_batch_wait_step: int | None = None

    def _should_log(self, trainer: GenericTrainer, step: int) -> bool:
        # Only primary rank logs
        if not ddp_is_primary(getattr(trainer, "fabric", None)):
            return False

        freq = self.log_frequency
        if freq is None:
            try:
                freq = trainer.config.logging.log_scalars_every_n_steps
            except Exception:
                freq = None

        return freq is None or (step % freq == 0)

    def on_before_optimizer_step(
        self, trainer: GenericTrainer, optimizer_idx: int
    ) -> None:
        # Optional CUDA sync before timing
        if self.cuda_sync and hasattr(trainer, "model") and torch.cuda.is_available():
            with suppress(Exception):
                torch.cuda.synchronize()

        self._optim_t0 = _time.perf_counter()

    def on_after_optimizer_step(
        self, trainer: GenericTrainer, optimizer_idx: int
    ) -> None:
        t0 = self._optim_t0
        self._optim_t0 = None

        if self.cuda_sync and torch.cuda.is_available():
            with suppress(Exception):
                torch.cuda.synchronize()

        if t0 is None:
            return

        dt_ms = (_time.perf_counter() - t0) * 1000.0

        # Log metric
        step = getattr(trainer, "global_step", 0)
        if not self._should_log(trainer, step):
            return

        loader_name = getattr(trainer, "current_dataloader_name", None) or "unknown"
        key = f"profile/train/dl_{sanitize_metric_key_component(loader_name)}/time_optimizer_ms"

        logger_inst = getattr(trainer, "logger", None)
        if logger_inst is not None:
            try:
                logger_inst.log_metrics({key: float(dt_ms)}, step)
            except Exception:
                logger.debug("Optimizer timing log failed", exc_info=True)

    def on_train_batch_end(
        self,
        trainer: GenericTrainer,
        batch: Any,
        loader_idx: int,
        loader_name: str,
        metrics: dict[str, Any],
    ) -> None:
        if not self.enable_batch_wait:
            return
        sanitized = sanitize_metric_key_component(loader_name)
        self._last_batch_end_time[sanitized] = _time.perf_counter()

    def on_train_batch_start(
        self,
        trainer: GenericTrainer,
        batch: Any,
        loader_idx: int,
        loader_name: str,
    ) -> None:
        if not self.enable_batch_wait:
            return
        sanitized = sanitize_metric_key_component(loader_name)
        last_end = self._last_batch_end_time.get(sanitized)
        if last_end is None:
            return
        now = _time.perf_counter()
        wait_ms = (now - last_end) * 1000.0

        # Gate to one log per loader per optimizer step
        step = getattr(trainer, "global_step", 0)
        if self._last_logged_batch_wait_step.get(sanitized) == step:
            return
        if not self._should_log(trainer, step):
            return

        self._last_logged_batch_wait_step[sanitized] = step

        key = f"profile/train/dl_{sanitize_metric_key_component(loader_name)}/time_batch_wait_ms"
        logger_inst = getattr(trainer, "logger", None)
        if logger_inst is not None:
            try:
                logger_inst.log_metrics({key: float(wait_ms)}, step)
            except Exception:
                logger.debug("Batch-wait timing log failed", exc_info=True)
