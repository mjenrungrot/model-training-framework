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
    DDPConfig,
    EpochLengthPolicy,
    FaultToleranceConfig,
    GenericTrainerConfig,
    LoggingConfig,
    MultiDataLoaderConfig,
    PerformanceConfig,
    PreemptionConfig,
    SamplingStrategy,
    ValAggregation,
    ValidationConfig,
    ValidationFrequency,
    validate_infinite_loader_constraints,
    validate_trainer_config,
)
from .core import (
    GenericTrainer,
)
from .states import (
    ResumeState,
    RNGState,
    TrainerPhase,
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
    # Configuration
    "DDPConfig",
    "EpochLengthPolicy",
    "FaultToleranceConfig",
    "GenericTrainer",
    "GenericTrainerConfig",
    "LoggingConfig",
    "MultiDataLoaderConfig",
    "PerformanceConfig",
    "PreemptionConfig",
    "PreemptionTimeoutError",
    "RNGState",
    "ResumeState",
    "SamplingStrategy",
    # Core trainer
    "TrainerError",
    "TrainerPhase",
    "ValAggregation",
    "ValidationConfig",
    "ValidationFrequency",
    # Utilities
    "timeout",
    "validate_infinite_loader_constraints",
    "validate_trainer_config",
]
