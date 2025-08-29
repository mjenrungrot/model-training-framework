"""
Trainer Configuration Classes

This module re-exports configuration classes from model_training_framework.config.schemas
for backward compatibility. New code should import directly from config.schemas.
"""

from model_training_framework.config.schemas import (
    CheckpointConfig,
    DDPConfig,
    EpochLengthPolicy,
    FaultToleranceConfig,
    GenericTrainerConfig,
    HooksConfig,
    LoggingConfig,
    MultiDataLoaderConfig,
    PerformanceConfig,
    PreemptionConfig,
    SamplingStrategy,
    ValAggregation,
    ValidationConfig,
    ValidationFrequency,
    _validate_multi_config,
    validate_infinite_loader_constraints,
    validate_trainer_config,
)

__all__ = [
    "CheckpointConfig",
    "DDPConfig",
    "EpochLengthPolicy",
    "FaultToleranceConfig",
    "GenericTrainerConfig",
    "HooksConfig",
    "LoggingConfig",
    "MultiDataLoaderConfig",
    "PerformanceConfig",
    "PreemptionConfig",
    "SamplingStrategy",
    "ValAggregation",
    "ValidationConfig",
    "ValidationFrequency",
    "_validate_multi_config",
    "validate_infinite_loader_constraints",
    "validate_trainer_config",
]
