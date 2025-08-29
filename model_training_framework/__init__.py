"""
Model Training Framework - A comprehensive Python package for ML model training, launching, and configuration management.

This framework provides:
- Fault-tolerant training with SLURM integration
- Sophisticated parameter grid search capabilities
- Robust experiment tracking and configuration management
- Distributed training support via Lightning Fabric

Main Components:
- config: Configuration management, parameter grids, and validation
- trainer: Training engine with checkpointing and preemption handling
- slurm: SLURM job launcher and batch management
- utils: Utility functions and helper classes

Usage:
    from model_training_framework import ModelTrainingFramework

    # Initialize framework
    framework = ModelTrainingFramework(project_root=Path("/path/to/project"))

    # Create experiment configuration
    config = framework.create_experiment({
        "experiment_name": "resnet_baseline",
        "model": {"type": "resnet50", "num_classes": 1000},
        "training": {"max_epochs": 100, "batch_size": 64},
        "optimizer": {"type": "adamw", "lr": 1e-3}
    })

    # Execute experiment
    result = framework.run_single_experiment(config)
"""

__version__ = "1.0.0"
__author__ = "Model Training Framework Team"
__email__ = "team@example.com"

# Import main framework class
# Import configuration classes
from .config import (
    ConfigValidator,
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    ParameterGrid,
    ParameterGridSearch,
    SchedulerConfig,
    SLURMConfig,
    TrainingConfig,
    ValidationResult,
)

# Execution modes
from .config.schemas import ExecutionMode, NamingStrategy
from .core import ModelTrainingFramework

# Import SLURM classes
from .slurm import (
    BatchSubmissionResult,
    GitOperationLock,
    JobStatus,
    SLURMJobMonitor,
    SLURMLauncher,
)

# Import trainer classes
from .trainer import (
    CheckpointableIterable,
    CheckpointConfig,
    ChoiceRNGState,
    DataLoaderManager,
    DataLoaderState,
    DDPConfig,
    EpochLengthPolicy,
    FaultToleranceConfig,
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    MultiDataLoaderIterator,
    MultiTrainMicroState,
    MultiValMicroState,
    PerformanceConfig,
    PreemptionConfig,
    ResumeState,
    RNGState,
    SamplingStrategy,
    Stopwatch,
    TrainerPhase,
    ValAggregation,
    ValidationConfig,
    ValidationFrequency,
    balanced_interleave,
    capture_choice_rng_state,
    capture_rng_state,
    count_samples_in_batch,
    create_initial_resume_state,
    ddp_barrier,
    ddp_broadcast_object,
    ddp_is_primary,
    get_memory_usage,
    restore_choice_rng_state,
    restore_rng_state,
    seed_all,
    update_resume_state,
    validate_infinite_loader_constraints,
    validate_trainer_config,
)

# Import utility functions
from .utils import (
    get_project_root,
    setup_logging,
    validate_project_structure,
)

# Export all public APIs
__all__ = [
    "BatchSubmissionResult",
    "CheckpointConfig",
    "CheckpointableIterable",
    "ChoiceRNGState",
    "ConfigValidator",
    "DDPConfig",
    "DataConfig",
    "DataLoaderManager",
    "DataLoaderState",
    "EpochLengthPolicy",
    "ExecutionMode",
    # Configuration
    "ExperimentConfig",
    "FaultToleranceConfig",
    # Training
    "GenericTrainer",
    "GenericTrainerConfig",
    "GitOperationLock",
    "JobStatus",
    "LoggingConfig",
    "ModelConfig",
    # Main framework
    "ModelTrainingFramework",
    "MultiDataLoaderConfig",
    "MultiDataLoaderIterator",
    "MultiTrainMicroState",
    "MultiValMicroState",
    "NamingStrategy",
    "OptimizerConfig",
    "ParameterGrid",
    "ParameterGridSearch",
    "PerformanceConfig",
    "PreemptionConfig",
    "RNGState",
    "ResumeState",
    "SLURMConfig",
    "SLURMJobMonitor",
    # SLURM
    "SLURMLauncher",
    "SamplingStrategy",
    "SchedulerConfig",
    "Stopwatch",
    "TrainerPhase",
    "TrainingConfig",
    "ValAggregation",
    "ValidationConfig",
    "ValidationFrequency",
    "ValidationResult",
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
    "get_project_root",
    "restore_choice_rng_state",
    "restore_rng_state",
    "seed_all",
    "setup_logging",
    "update_resume_state",
    "validate_infinite_loader_constraints",
    "validate_project_structure",
    "validate_trainer_config",
]
