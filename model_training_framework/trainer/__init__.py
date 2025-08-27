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
from .multi_dataloader import (
    CheckpointableIterable,
    DataLoaderManager,
    MultiDataLoaderIterator,
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
    Stopwatch,
    TrainerError,
    balanced_interleave,
    count_samples_in_batch,
    ddp_barrier,
    ddp_broadcast_object,
    ddp_is_primary,
    get_memory_usage,
    seed_all,
    timeout,
)

__all__ = [
    # Checkpoints
    "CheckpointConfig",
    "CheckpointManager",
    "CheckpointTimeoutError",
    "CheckpointableIterable",
    "ChoiceRNGState",
    # Configuration
    "DDPConfig",
    "DataLoaderManager",
    "DataLoaderState",
    "EpochLengthPolicy",
    "FaultToleranceConfig",
    "GenericTrainer",
    "GenericTrainerConfig",
    "LoggingConfig",
    "MultiDataLoaderConfig",
    "MultiDataLoaderIterator",
    "MultiTrainMicroState",
    "MultiValMicroState",
    "PerformanceConfig",
    "PreemptionConfig",
    "PreemptionTimeoutError",
    "RNGState",
    "ResumeState",
    "SamplingStrategy",
    "Stopwatch",
    # Core trainer
    "TrainerError",
    "TrainerPhase",
    "ValAggregation",
    "ValidationConfig",
    "ValidationFrequency",
    # Utilities
    "balanced_interleave",
    # State helpers
    "capture_choice_rng_state",
    "capture_rng_state",
    "count_samples_in_batch",
    "create_initial_resume_state",
    "ddp_barrier",
    "ddp_broadcast_object",
    "ddp_is_primary",
    "get_memory_usage",
    "restore_choice_rng_state",
    "restore_rng_state",
    "seed_all",
    "timeout",
    "update_resume_state",
    "validate_infinite_loader_constraints",
    "validate_trainer_config",
]
