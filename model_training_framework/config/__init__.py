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
    # Validation
    "ConfigValidator",
    # Management
    "ConfigurationManager",
    "DataConfig",
    "ExecutionMode",
    # Schemas
    "ExperimentConfig",
    # Naming
    "ExperimentNaming",
    "GridSearchExecutor",
    "GridSearchResult",
    "LoggingConfig",
    "ModelConfig",
    "NamingStrategy",
    "OptimizerConfig",
    # Grid search
    "ParameterGrid",
    "ParameterGridSearch",
    "ResourceCheck",
    "SLURMConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "ValidationResult",
]
