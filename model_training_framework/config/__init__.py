"""
Configuration Management Component

This module provides comprehensive configuration management for the training framework:
- Configuration schemas and validation
- Parameter grid search and experiment generation
- Configuration composition and management
- Experiment naming strategies
"""

from .grid_search import (
    GridSearchResult,
    ParameterGrid,
    ParameterGridSearch,
)
from .manager import (
    ConfigurationManager,
)
from .naming import (
    ExperimentNaming,
)
from .schemas import (
    # Trainer configuration classes
    CheckpointConfig,
    # Core configuration classes
    DataConfig,
    DDPConfig,
    EpochLengthPolicy,
    # Enums
    ExecutionMode,
    ExperimentConfig,
    FaultToleranceConfig,
    GenericTrainerConfig,
    GridSearchConfig,
    HooksConfig,
    LoggingConfig,
    ModelConfig,
    MultiDataLoaderConfig,
    NamingStrategy,
    OptimizerConfig,
    PerformanceConfig,
    PreemptionConfig,
    SamplingStrategy,
    SchedulerConfig,
    SLURMConfig,
    TrainingConfig,
    ValAggregation,
    ValidationConfig,
    ValidationFrequency,
    WarmStartConfig,
    # Validation functions
    validate_infinite_loader_constraints,
    validate_trainer_config,
)
from .validators import (
    ConfigValidator,
    ValidationResult,
)

__all__ = [  # noqa: RUF022
    # Validation
    "ConfigValidator",
    "ValidationResult",
    "validate_infinite_loader_constraints",
    "validate_trainer_config",
    # Management
    "ConfigurationManager",
    # Enums
    "ExecutionMode",
    "NamingStrategy",
    "SamplingStrategy",
    "EpochLengthPolicy",
    "ValidationFrequency",
    "ValAggregation",
    # Core configuration schemas
    "DataConfig",
    "ExperimentConfig",
    "GridSearchConfig",
    "LoggingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "SLURMConfig",
    "TrainingConfig",
    # Trainer configuration schemas
    "CheckpointConfig",
    "WarmStartConfig",
    "PreemptionConfig",
    "PerformanceConfig",
    "MultiDataLoaderConfig",
    "ValidationConfig",
    "FaultToleranceConfig",
    "DDPConfig",
    "HooksConfig",
    "GenericTrainerConfig",
    # Naming
    "ExperimentNaming",
    # Grid search
    "GridSearchResult",
    "ParameterGrid",
    "ParameterGridSearch",
]
