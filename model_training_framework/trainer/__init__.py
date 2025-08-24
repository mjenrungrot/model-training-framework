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
    ChoiceRNGState,
    DataLoaderState,
    MultiTrainMicroState,
    MultiValMicroState,
    ResumeState,
    RNGState,
    TrainerPhase,
    capture_choice_rng_state,
    capture_rng_state,
    create_initial_resume_state,
    restore_choice_rng_state,
    restore_rng_state,
    update_resume_state,
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
    "ChoiceRNGState",
    # Configuration
    "DDPConfig",
    "DataLoaderState",
    "EpochLengthPolicy",
    "FaultToleranceConfig",
    "GenericTrainer",
    "GenericTrainerConfig",
    "LoggingConfig",
    "MultiDataLoaderConfig",
    "MultiTrainMicroState",
    "MultiValMicroState",
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
    # State helpers
    "capture_choice_rng_state",
    "capture_rng_state",
    "create_initial_resume_state",
    "restore_choice_rng_state",
    "restore_rng_state",
    # Utilities
    "timeout",
    "update_resume_state",
    "validate_infinite_loader_constraints",
    "validate_trainer_config",
]
