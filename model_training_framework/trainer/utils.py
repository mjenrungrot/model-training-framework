"""
Trainer Utilities

This module provides utility functions for the training engine:
- Timeout context manager for operations with time limits
- RNG state management functions
- Signal handling utilities
- Performance monitoring helpers
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import logging
import random
import signal
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class TrainerError(Exception):
    """Base exception class for trainer-specific errors."""


class PreemptionTimeoutError(TrainerError):
    """Raised when checkpoint save exceeds time limit during preemption."""


class CheckpointTimeoutError(TrainerError):
    """Raised when checkpoint save exceeds configured time limit."""


@contextlib.contextmanager
def timeout(seconds: float) -> Iterator[None]:
    """
    Context manager that raises TimeoutError after specified seconds.

    Args:
        seconds: Timeout duration in seconds

    Raises:
        TimeoutError: If operation exceeds timeout

    Example:
        with timeout(30.0):
            # Operation that should complete within 30 seconds
            long_running_operation()
    """

    def _raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("Operation exceeded allotted time")

    # Save original handler
    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)

    # Set alarm
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        # Cancel alarm and restore handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class SignalHandler:
    """Handle training-related signals like SIGUSR1 for preemption."""

    def __init__(self):
        self.preemption_requested = False
        self.original_handlers: dict[int, Any] = {}
        self.callbacks: dict[int, list[Callable]] = {}

    def register_preemption_handler(self, signal_num: int = signal.SIGUSR1) -> None:
        """
        Register handler for preemption signal.

        Args:
            signal_num: Signal number to handle (default: SIGUSR1)
        """

        def preemption_handler(signum: int, frame: Any) -> None:
            logger.warning(f"Received preemption signal {signum}")
            self.preemption_requested = True

            # Execute registered callbacks
            for callback in self.callbacks.get(signum, []):
                try:
                    callback(signum, frame)
                except Exception:
                    logger.exception("Error in signal callback: ")

        # Save original handler and install new one
        self.original_handlers[signal_num] = signal.signal(
            signal_num, preemption_handler
        )
        logger.info(f"Registered preemption handler for signal {signal_num}")

    def add_callback(self, signal_num: int, callback: Callable) -> None:
        """
        Add callback function to be called when signal is received.

        Args:
            signal_num: Signal number
            callback: Function to call when signal is received
        """
        if signal_num not in self.callbacks:
            self.callbacks[signal_num] = []
        self.callbacks[signal_num].append(callback)

    def restore_handlers(self) -> None:
        """Restore original signal handlers."""
        for signal_num, original_handler in self.original_handlers.items():
            signal.signal(signal_num, original_handler)

        self.original_handlers.clear()
        self.callbacks.clear()
        logger.debug("Restored original signal handlers")

    def is_preemption_requested(self) -> bool:
        """Check if preemption has been requested."""
        return self.preemption_requested

    def reset_preemption_flag(self) -> None:
        """Reset the preemption flag."""
        self.preemption_requested = False


class PerformanceMonitor:
    """Monitor training performance metrics."""

    def __init__(self):
        self.step_times: list[float] = []
        self.memory_usage: list[float] = []
        self.gpu_utilization: list[float] = []
        self.start_time = time.time()
        self.last_step_time = time.time()

    def record_step_time(self) -> float:
        """
        Record time for current step.

        Returns:
            Step duration in seconds
        """
        current_time = time.time()
        step_duration = current_time - self.last_step_time
        self.step_times.append(step_duration)
        self.last_step_time = current_time
        return step_duration

    def record_memory_usage(self) -> float | None:
        """
        Record current GPU memory usage.

        Returns:
            Memory usage in GB if CUDA available, None otherwise
        """
        if not torch.cuda.is_available():
            return None

        memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        memory_gb = memory_mb / 1024
        self.memory_usage.append(memory_gb)
        return memory_gb

    def get_average_step_time(self, last_n: int | None = None) -> float:
        """
        Get average step time.

        Args:
            last_n: Only consider last N steps (None for all)

        Returns:
            Average step time in seconds
        """
        if not self.step_times:
            return 0.0

        times = self.step_times[-last_n:] if last_n else self.step_times
        return sum(times) / len(times)

    def get_steps_per_second(self, last_n: int | None = None) -> float:
        """
        Get training steps per second.

        Args:
            last_n: Only consider last N steps (None for all)

        Returns:
            Steps per second
        """
        avg_time = self.get_average_step_time(last_n)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_total_training_time(self) -> float:
        """Get total training time in seconds."""
        return time.time() - self.start_time

    def get_memory_stats(self) -> dict[str, float]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        if not self.memory_usage:
            return {"current_gb": 0.0, "max_gb": 0.0, "avg_gb": 0.0}

        current_gb = self.record_memory_usage() or 0.0
        max_gb = max(self.memory_usage)
        avg_gb = sum(self.memory_usage) / len(self.memory_usage)

        return {"current_gb": current_gb, "max_gb": max_gb, "avg_gb": avg_gb}

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "total_steps": len(self.step_times),
            "total_time_sec": self.get_total_training_time(),
            "avg_step_time_sec": self.get_average_step_time(),
            "steps_per_sec": self.get_steps_per_second(),
            "memory_stats": self.get_memory_stats(),
        }


# Constants for formatting
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024**2
BYTES_PER_GB = 1024**3


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    if seconds < SECONDS_PER_HOUR:
        minutes = seconds / SECONDS_PER_MINUTE
        return f"{minutes:.1f}m"
    hours = seconds / SECONDS_PER_HOUR
    minutes = (seconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE
    return f"{hours:.0f}h {minutes:.0f}m"


def format_memory(bytes_val: float) -> str:
    """
    Format bytes into human-readable memory string.

    Args:
        bytes_val: Memory in bytes

    Returns:
        Formatted memory string (e.g., "1.5GB")
    """
    if bytes_val < BYTES_PER_KB:
        return f"{bytes_val:.0f}B"
    if bytes_val < BYTES_PER_MB:
        return f"{bytes_val / BYTES_PER_KB:.1f}KB"
    if bytes_val < BYTES_PER_GB:
        return f"{bytes_val / BYTES_PER_MB:.1f}MB"
    return f"{bytes_val / BYTES_PER_GB:.1f}GB"


def get_device_info() -> dict[str, Any]:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    info = {
        "cpu_count": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0
        if not torch.cuda.is_available()
        else torch.cuda.device_count(),
        "cuda_devices": [],
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append(
                {
                    "device_id": i,
                    "name": device_props.name,
                    "memory_gb": device_props.total_memory / (1024**3),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                }
            )

    return info


def log_system_info() -> None:
    """Log system and device information."""
    device_info = get_device_info()

    logger.info(f"CPU threads: {device_info['cpu_count']}")
    logger.info(f"CUDA available: {device_info['cuda_available']}")

    if device_info["cuda_available"]:
        logger.info(f"CUDA devices: {device_info['cuda_device_count']}")
        for device in device_info["cuda_devices"]:
            logger.info(
                f"  Device {device['device_id']}: {device['name']} "
                f"({device['memory_gb']:.1f}GB, compute {device['compute_capability']})"
            )


def ensure_reproducible(
    seed: int, deterministic: bool = True, benchmark: bool = False
) -> None:
    """
    Set random seeds and configure PyTorch for reproducible training.

    Args:
        seed: Random seed to use
        deterministic: Whether to use deterministic algorithms
        benchmark: Whether to enable cudnn benchmark mode
    """
    # Set seeds for all random number generators
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - using legacy API for compatibility
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configure PyTorch behavior
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark

    logger.info(
        f"Set random seed to {seed}, deterministic={deterministic}, benchmark={benchmark}"
    )


def calculate_model_size(model: torch.nn.Module) -> dict[str, Any]:
    """
    Calculate model size and parameter count.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory usage (parameters + gradients + optimizer states)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_memory = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "parameter_memory_mb": param_memory / (1024**2),
        "gradient_memory_mb": grad_memory / (1024**2),
        "estimated_total_memory_mb": (param_memory + grad_memory * 2)
        / (1024**2),  # params + grads + optimizer
    }


def log_model_info(model: torch.nn.Module) -> None:
    """Log information about the model."""
    model_info = calculate_model_size(model)

    logger.info(
        f"Model parameters: {model_info['total_parameters']:,} total, "
        f"{model_info['trainable_parameters']:,} trainable"
    )
    logger.info(
        f"Estimated memory usage: {model_info['estimated_total_memory_mb']:.1f}MB"
    )


class EarlyStopping:
    """Early stopping utility for training."""

    def __init__(
        self,
        patience: int,
        metric_name: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            metric_name: Name of metric to monitor
            mode: "min" to minimize metric, "max" to maximize
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.min_delta = min_delta

        self.best_value: float | None = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def __call__(self, metrics: dict[str, float]) -> bool:
        """
        Check if training should stop early.

        Args:
            metrics: Dictionary of current metrics

        Returns:
            True if training should stop
        """
        if self.metric_name not in metrics:
            logger.warning(
                f"Early stopping metric '{self.metric_name}' not found in metrics"
            )
            return False

        current_value = metrics[self.metric_name]

        # Check for improvement
        improved = False
        if self.best_value is None:
            improved = True
        elif self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        elif self.mode == "max":
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Check if we should stop
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            logger.info(
                f"Early stopping triggered after {self.epochs_without_improvement} epochs "
                f"without improvement in {self.metric_name}"
            )

        return self.should_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_value = None
        self.epochs_without_improvement = 0
        self.should_stop = False
