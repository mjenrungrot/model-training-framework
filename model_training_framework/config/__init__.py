"""
Configuration Management Component

This module provides comprehensive configuration management for the training framework:
- Configuration schemas and validation
- Parameter grid search and experiment generation
- Configuration composition and management
- Experiment naming strategies
"""

from .grid_search import (
    GridSearchExecutor,
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
    DataConfig,
    ExecutionMode,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    NamingStrategy,
    OptimizerConfig,
    SchedulerConfig,
    SLURMConfig,
    TrainingConfig,
)
from .validators import (
    ConfigValidator,
    ResourceCheck,
    ValidationResult,
)

__all__ = [
    # Schemas
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "SLURMConfig",
    "LoggingConfig",
    "ExecutionMode",
    "NamingStrategy",
    # Grid search
    "ParameterGrid",
    "ParameterGridSearch",
    "GridSearchResult",
    "GridSearchExecutor",
    # Validation
    "ConfigValidator",
    "ValidationResult",
    "ResourceCheck",
    # Management
    "ConfigurationManager",
    # Naming
    "ExperimentNaming",
]
