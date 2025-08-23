"""
Training Engine Component

This module provides the core training functionality:
- GenericTrainer with fault-tolerant training loops
- Checkpoint management and preemption handling
- Training state management and resume capabilities
- Training callbacks and hooks
"""

from .callbacks import (
    CheckpointCallback,
    LoggingCallback,
    TrainingCallback,
    WandbCallback,
)
from .checkpoints import (
    CheckpointConfig,
    CheckpointManager,
)
from .config import (
    LoggingConfig,
    PerformanceConfig,
    PreemptionConfig,
)
from .core import (
    CheckpointTimeoutError,
    GenericTrainer,
    GenericTrainerConfig,
    PreemptionTimeoutError,
    TrainerError,
)
from .states import (
    ResumeState,
    RNGState,
    TrainerPhase,
    TrainMicroState,
    ValMicroState,
)
from .utils import (
    capture_rng_state,
    restore_rng_state,
    timeout,
)

__all__ = [
    # Core trainer
    "GenericTrainer",
    "GenericTrainerConfig",
    "TrainerError",
    "PreemptionTimeoutError",
    "CheckpointTimeoutError",
    # States
    "TrainerPhase",
    "ResumeState",
    "RNGState",
    "TrainMicroState",
    "ValMicroState",
    # Checkpoints
    "CheckpointConfig",
    "CheckpointManager",
    # Configuration
    "PreemptionConfig",
    "PerformanceConfig",
    "LoggingConfig",
    # Callbacks
    "TrainingCallback",
    "CheckpointCallback",
    "LoggingCallback",
    "WandbCallback",
    # Utilities
    "timeout",
    "capture_rng_state",
    "restore_rng_state",
]
