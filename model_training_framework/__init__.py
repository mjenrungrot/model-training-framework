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
    ResourceCheck,
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
    CheckpointConfig,
    DDPConfig,
    EpochLengthPolicy,
    FaultToleranceConfig,
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    PerformanceConfig,
    PreemptionConfig,
    ResumeState,
    RNGState,
    SamplingStrategy,
    TrainerPhase,
    ValAggregation,
    ValidationConfig,
    ValidationFrequency,
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
    "ConfigValidator",
    "DDPConfig",
    "DataConfig",
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
    "NamingStrategy",
    "OptimizerConfig",
    "ParameterGrid",
    "ParameterGridSearch",
    "PerformanceConfig",
    "PreemptionConfig",
    "RNGState",
    "ResourceCheck",
    "ResumeState",
    "SLURMConfig",
    "SLURMJobMonitor",
    # SLURM
    "SLURMLauncher",
    "SamplingStrategy",
    "SchedulerConfig",
    "TrainerPhase",
    "TrainingConfig",
    "ValAggregation",
    "ValidationConfig",
    "ValidationFrequency",
    "ValidationResult",
    "get_project_root",
    # Utilities
    "setup_logging",
    "validate_infinite_loader_constraints",
    "validate_project_structure",
    "validate_trainer_config",
]
