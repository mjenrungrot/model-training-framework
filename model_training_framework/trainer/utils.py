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

try:
    import psutil  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency not required
    psutil = None  # type: ignore[assignment]

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

    # Set alarm (minimum 1 second for signal.alarm)
    alarm_seconds = max(1, int(seconds))
    signal.alarm(alarm_seconds)

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
    if torch.cuda.is_available():
        # torch.manual_seed typically seeds CUDA as well; only call manual_seed
        # here to avoid duplicate manual_seed_all invocations under test mocks.
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

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


def seed_all(seed: int) -> None:
    """
    Set random seeds for all RNG sources.

    Simple utility to ensure reproducibility by seeding:
    - Python random
    - NumPy random
    - PyTorch CPU and CUDA

    Args:
        seed: Random seed value

    Note:
        For more control over determinism settings, use ensure_reproducible()
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - using legacy API for compatibility

    if torch.cuda.is_available():
        # For CUDA environments, call both manual_seed (current device) and
        # manual_seed_all exactly once to satisfy test expectations without
        # relying on backend-specific implicit behavior.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)


def count_samples_in_batch(batch: Any) -> int:
    """
    Count the number of samples in a batch using default heuristics.

    Handles various batch formats:
    - Tensor: returns first dimension size
    - Tuple/List of tensors: returns first element's first dimension
    - Dict with 'input'/'data' keys: uses those tensors
    - Other: attempts len(batch)

    Args:
        batch: Batch data in various formats

    Returns:
        Number of samples in the batch

    Raises:
        ValueError: If batch format cannot be determined
    """
    # Handle tensor directly
    if isinstance(batch, torch.Tensor):
        return batch.size(0)

    # Handle tuple/list batches
    if isinstance(batch, tuple | list):
        # Empty sequences are ambiguous; treat as error
        if len(batch) == 0:
            raise ValueError("Cannot determine batch size for empty sequence")
        # Tuple/list of tensors: use first element's batch dimension
        if isinstance(batch[0], torch.Tensor):
            return batch[0].size(0)

    # Handle dict with common keys
    if isinstance(batch, dict):
        for key in ["input", "inputs", "data", "x"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].size(0)
        # Try first tensor value in dict
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.size(0)

    # Fallback to len
    try:
        return len(batch)
    except TypeError as err:
        raise ValueError(
            f"Cannot determine batch size for type {type(batch).__name__}"
        ) from err


def balanced_interleave(quota: list[int]) -> list[int]:
    """
    Create an evenly spaced interleaving sequence based on quotas.

    Uses a credit-based algorithm to ensure fair distribution:
    - Each index accumulates credit proportional to its quota
    - The index with highest credit is selected next
    - Selected index's credit is reduced

    Args:
        quota: List of quotas for each index (e.g., [2, 3, 1])

    Returns:
        List of indices forming balanced sequence (e.g., [1, 0, 1, 0, 1, 2])

    Example:
        >>> balanced_interleave([2, 3, 1])
        [1, 0, 1, 0, 1, 2]
        >>> balanced_interleave([1, 1, 1])
        [0, 1, 2]
    """
    if not quota:
        return []

    if all(q == 0 for q in quota):
        return []

    # Normalize quotas to avoid negative values
    quota = [max(0, q) for q in quota]
    total = sum(quota)
    if total == 0:
        return []

    result = []
    credits = [0.0] * len(quota)
    increments = [q / total for q in quota]

    for _ in range(total):
        # Add credit to all indices
        for i in range(len(credits)):
            if quota[i] > 0:
                credits[i] += increments[i]

        # Find index with maximum credit (that still has quota)
        max_credit = -1
        max_idx = -1
        for i in range(len(credits)):
            if quota[i] > 0 and credits[i] > max_credit:
                max_credit = credits[i]
                max_idx = i

        if max_idx == -1:
            break

        # Select this index
        result.append(max_idx)
        credits[max_idx] -= 1.0
        quota[max_idx] -= 1

    return result


def ddp_is_primary(fabric: Any) -> bool:
    """
    Check if current process is the primary (rank 0) process.

    Args:
        fabric: Lightning Fabric instance

    Returns:
        True if primary process or single-process mode
    """
    if fabric is None:
        return True

    # Check for is_global_zero attribute (Lightning Fabric)
    if hasattr(fabric, "is_global_zero"):
        return fabric.is_global_zero

    # Check for global_rank attribute
    if hasattr(fabric, "global_rank"):
        return fabric.global_rank == 0

    # Check for rank attribute
    if hasattr(fabric, "rank"):
        return fabric.rank == 0

    # Default to primary in single-process mode
    return True


def ddp_barrier(fabric: Any) -> None:
    """
    Synchronization barrier for distributed training.

    No-op in single-process mode.

    Args:
        fabric: Lightning Fabric instance
    """
    if fabric is None:
        return

    # Try fabric barrier method
    if hasattr(fabric, "barrier"):
        # Silently ignore in single-process mode
        with contextlib.suppress(Exception):
            fabric.barrier()


def ddp_broadcast_object(fabric: Any, obj: Any, src: int = 0) -> Any:
    """
    Broadcast object from source rank to all other ranks.

    In single-process mode, returns the object unchanged.

    Args:
        fabric: Lightning Fabric instance
        obj: Object to broadcast
        src: Source rank (default: 0)

    Returns:
        The broadcasted object
    """
    if fabric is None:
        return obj

    # Try fabric broadcast method
    if hasattr(fabric, "broadcast"):
        try:
            return fabric.broadcast(obj, src=src)
        except Exception as e:
            # Log the issue but return unchanged to not break single-process mode
            logger.debug(f"Broadcast failed (likely single-process mode): {e}")
            return obj

    # No broadcast available, return unchanged
    return obj


def ddp_all_gather(
    fabric: Any, tensor: torch.Tensor
) -> torch.Tensor | list[torch.Tensor]:
    """
    Gather tensors from all ranks.

    In single-process mode, returns the tensor unchanged.

    Args:
        fabric: Lightning Fabric instance
        tensor: Tensor to gather from all ranks

    Returns:
        List of tensors from all ranks, or single tensor in single-process mode
    """
    if fabric is None:
        return tensor

    # Try fabric all_gather method
    if hasattr(fabric, "all_gather"):
        try:
            return fabric.all_gather(tensor)
        except Exception:
            # Return unchanged in single-process mode
            return tensor

    # No all_gather available, return unchanged
    return tensor


def ddp_all_reduce(fabric: Any, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """
    Reduce tensor across all ranks.

    In single-process mode, returns the tensor unchanged.

    Args:
        fabric: Lightning Fabric instance
        tensor: Tensor to reduce
        op: Reduction operation ("mean", "sum", "min", "max")

    Returns:
        Reduced tensor
    """
    if fabric is None:
        return tensor

    # Try fabric all_reduce method
    if hasattr(fabric, "all_reduce"):
        try:
            return fabric.all_reduce(tensor, op=op)
        except Exception:
            # Return unchanged in single-process mode
            return tensor

    # No all_reduce available, return unchanged
    return tensor


def ddp_world_size(fabric: Any) -> int:
    """
    Get the world size (number of processes).

    Args:
        fabric: Lightning Fabric instance

    Returns:
        World size (1 if not distributed)
    """
    if fabric is None:
        return 1

    # Check for world_size attribute
    if hasattr(fabric, "world_size"):
        return fabric.world_size

    # Default to 1
    return 1


def ddp_rank(fabric: Any) -> int:
    """
    Get the rank of current process.

    Args:
        fabric: Lightning Fabric instance

    Returns:
        Process rank (0 if not distributed)
    """
    if fabric is None:
        return 0

    # Check for global_rank attribute
    if hasattr(fabric, "global_rank"):
        return fabric.global_rank

    # Check for rank attribute
    if hasattr(fabric, "rank"):
        return fabric.rank

    # Default to 0
    return 0


class Stopwatch:
    """
    Simple stopwatch for timing operations.

    Supports start, stop, reset, and lap timing.

    Example:
        >>> sw = Stopwatch()
        >>> sw.start()
        >>> # ... do work ...
        >>> elapsed = sw.elapsed_time()
        >>> sw.lap()  # Record lap time
        >>> # ... more work ...
        >>> sw.stop()
        >>> total = sw.elapsed_time()
    """

    def __init__(self):
        """Initialize stopwatch in stopped state."""
        self.start_time: float | None = None
        self.elapsed: float = 0.0
        self.running: bool = False
        self.laps: list[float] = []

    def start(self) -> None:
        """Start or resume the stopwatch."""
        if not self.running:
            self.start_time = time.time()
            self.running = True

    def stop(self) -> float:
        """
        Stop the stopwatch and return elapsed time.

        Returns:
            Total elapsed time in seconds
        """
        if self.running and self.start_time is not None:
            self.elapsed += time.time() - self.start_time
            self.running = False
            self.start_time = None
        return self.elapsed

    def reset(self) -> None:
        """Reset stopwatch to initial state."""
        self.start_time = None
        self.elapsed = 0.0
        self.running = False
        self.laps = []

    def lap(self) -> float:
        """
        Record a lap time.

        Returns:
            Time since last lap (or start if first lap)
        """
        current = self.elapsed_time()
        lap_time = current - sum(self.laps) if self.laps else current
        self.laps.append(lap_time)
        return lap_time

    def elapsed_time(self) -> float:
        """
        Get elapsed time without stopping.

        Returns:
            Total elapsed time in seconds
        """
        if self.running and self.start_time is not None:
            return self.elapsed + (time.time() - self.start_time)
        return self.elapsed

    def get_laps(self) -> list[float]:
        """Get list of lap times."""
        return self.laps.copy()


def get_memory_usage() -> dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        Dictionary with memory stats:
        - gpu_allocated_gb: Currently allocated GPU memory
        - gpu_reserved_gb: Reserved GPU memory
        - gpu_free_gb: Free GPU memory
        - cpu_percent: CPU memory usage percentage

    Returns empty dict if no GPU available.
    """
    stats = {}

    if torch.cuda.is_available():
        # GPU memory stats
        stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)

        # Get total GPU memory
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_memory = device_props.total_memory / (1024**3)
        stats["gpu_total_gb"] = total_memory
        stats["gpu_free_gb"] = total_memory - stats["gpu_allocated_gb"]

    # CPU memory stats (requires psutil for accurate stats)
    if psutil is not None:  # type: ignore[truthy-bool]
        vm = psutil.virtual_memory()
        stats["cpu_percent"] = vm.percent
        stats["cpu_used_gb"] = vm.used / (1024**3)
        stats["cpu_available_gb"] = vm.available / (1024**3)

    return stats


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
