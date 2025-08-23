"""
Training Engine Component

This module provides the core training functionality:
- GenericTrainer with fault-tolerant training loops
- Checkpoint management and preemption handling
- Training state management and resume capabilities
- Training callbacks and hooks
"""

from .checkpoints import (
    CheckpointManager,
)
from .config import (
    CheckpointConfig,
    GenericTrainerConfig,
    LoggingConfig,
    PerformanceConfig,
    PreemptionConfig,
)
from .core import (
    GenericTrainer,
)
from .states import (
    ResumeState,
    RNGState,
    TrainerPhase,
    TrainMicroState,
    ValMicroState,
)
from .utils import (
    CheckpointTimeoutError,
    PreemptionTimeoutError,
    TrainerError,
    timeout,
)

__all__ = [
    # Checkpoints
    "CheckpointConfig",
    "CheckpointManager",
    "CheckpointTimeoutError",
    # Core trainer
    "GenericTrainer",
    "GenericTrainerConfig",
    "LoggingConfig",
    "PerformanceConfig",
    # Configuration
    "PreemptionConfig",
    "PreemptionTimeoutError",
    "RNGState",
    "ResumeState",
    "TrainMicroState",
    "TrainerError",
    # States
    "TrainerPhase",
    "ValMicroState",
    # Utilities
    "timeout",
]
