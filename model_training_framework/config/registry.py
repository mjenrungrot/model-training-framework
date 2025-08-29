"""
Configuration Registry System

This module provides a registry for custom configuration classes,
allowing users to register their own typed configurations while
maintaining backward compatibility with the flexible default configs.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from .schemas import DataConfig, ModelConfig

logger = logging.getLogger(__name__)


class ConfigRegistry:
    """Registry for custom configuration classes.

    This allows users to register their own typed configuration classes
    for specific model or dataset types, providing better type safety
    and custom validation while maintaining flexibility.
    """

    # Class-level registries
    _model_configs: ClassVar[dict[str, type[ModelConfig]]] = {}
    _data_configs: ClassVar[dict[str, type[DataConfig]]] = {}

    @classmethod
    def register_model_config(
        cls, model_type: str, config_class: type[ModelConfig]
    ) -> None:
        """Register a custom model configuration class.

        Args:
            model_type: The model type identifier (e.g., "resnet18", "bert")
            config_class: The configuration class to use for this model type
        """
        cls._model_configs[model_type] = config_class
        logger.debug(
            f"Registered model config {config_class.__name__} for type '{model_type}'"
        )

    @classmethod
    def register_data_config(
        cls, dataset_name: str, config_class: type[DataConfig]
    ) -> None:
        """Register a custom data configuration class.

        Args:
            dataset_name: The dataset identifier
            config_class: The configuration class to use for this dataset
        """
        cls._data_configs[dataset_name] = config_class
        logger.debug(
            f"Registered data config {config_class.__name__} for dataset '{dataset_name}'"
        )

    @classmethod
    def create_model_config(cls, data: dict[str, Any]) -> ModelConfig:
        """Create appropriate model config based on type.

        Args:
            data: Dictionary containing model configuration

        Returns:
            ModelConfig instance (either custom registered class or default)
        """
        model_type = data.get("type", "custom")
        config_class = cls._model_configs.get(model_type, ModelConfig)

        # Use from_dict method if available
        if hasattr(config_class, "from_dict"):
            return config_class.from_dict(data)
        return config_class(**data)

    @classmethod
    def create_data_config(cls, data: dict[str, Any]) -> DataConfig:
        """Create appropriate data config based on dataset name.

        Args:
            data: Dictionary containing data configuration

        Returns:
            DataConfig instance (either custom registered class or default)
        """
        dataset_name = data.get("dataset_name", "custom")
        config_class = cls._data_configs.get(dataset_name, DataConfig)

        # Use from_dict method if available
        if hasattr(config_class, "from_dict"):
            return config_class.from_dict(data)
        return config_class(**data)

    @classmethod
    def unregister_model_config(cls, model_type: str) -> None:
        """Unregister a model configuration class.

        Args:
            model_type: The model type to unregister
        """
        if model_type in cls._model_configs:
            del cls._model_configs[model_type]
            logger.debug(f"Unregistered model config for type '{model_type}'")

    @classmethod
    def unregister_data_config(cls, dataset_name: str) -> None:
        """Unregister a data configuration class.

        Args:
            dataset_name: The dataset to unregister
        """
        if dataset_name in cls._data_configs:
            del cls._data_configs[dataset_name]
            logger.debug(f"Unregistered data config for dataset '{dataset_name}'")

    @classmethod
    def list_registered_models(cls) -> list[str]:
        """Get list of registered model types.

        Returns:
            List of registered model type identifiers
        """
        return list(cls._model_configs.keys())

    @classmethod
    def list_registered_datasets(cls) -> list[str]:
        """Get list of registered dataset names.

        Returns:
            List of registered dataset identifiers
        """
        return list(cls._data_configs.keys())

    @classmethod
    def clear_all(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._model_configs.clear()
        cls._data_configs.clear()
        logger.debug("Cleared all config registrations")


def register_model(model_type: str):
    """Decorator for registering model configuration classes.

    Usage:
        @register_model("resnet18")
        class ResNet18Config(ModelConfig):
            num_layers: int = 18
            ...
    """

    def decorator(config_class: type[ModelConfig]) -> type[ModelConfig]:
        ConfigRegistry.register_model_config(model_type, config_class)
        return config_class

    return decorator


def register_dataset(dataset_name: str):
    """Decorator for registering data configuration classes.

    Usage:
        @register_dataset("imagenet")
        class ImageNetConfig(DataConfig):
            num_classes: int = 1000
            ...
    """

    def decorator(config_class: type[DataConfig]) -> type[DataConfig]:
        ConfigRegistry.register_data_config(dataset_name, config_class)
        return config_class

    return decorator
