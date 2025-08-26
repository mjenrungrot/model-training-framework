"""
Configuration Manager

This module provides centralized configuration management:
- Loading and saving configurations from various formats
- Configuration composition and inheritance
- Environment variable substitution
- Configuration validation and transformation
"""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, TypeVar, cast, overload

import yaml

from ..utils.path_utils import resolve_config_path
from .schemas import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    PerformanceConfig,
    PreemptionConfig,
    SchedulerConfig,
    SLURMConfig,
    TrainingConfig,
)
from .validators import ConfigValidator, ValidationResult

logger = logging.getLogger(__name__)


T = TypeVar("T")


class ConfigurationManager:
    """Manages configuration loading, validation, and composition."""

    def __init__(self, project_root: Path, config_dir: Path | None = None):
        """Initialize configuration manager."""
        self.project_root = Path(project_root)
        self.config_dir = config_dir or self.project_root / "configs"
        self.config_cache: dict[str, dict[str, Any]] = {}

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"Initialized ConfigurationManager with project_root={project_root}, config_dir={self.config_dir}"
        )

    def load_config(
        self,
        config_path: str | Path,
        validate: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Load configuration from file with optional validation."""

        # Resolve path relative to project root or config directory
        resolved_path = resolve_config_path(
            config_path, self.project_root, self.config_dir
        )

        if not resolved_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

        # Check cache first
        cache_key = str(resolved_path)
        if use_cache and cache_key in self.config_cache:
            logger.debug(f"Using cached config for {resolved_path}")
            return self.config_cache[cache_key].copy()

        # Load configuration based on file extension
        config_data = self._load_config_file(resolved_path)

        # Perform environment variable substitution
        config_data = cast(
            dict[str, Any], self._substitute_environment_variables(config_data)
        )

        # Cache the loaded config
        if use_cache:
            self.config_cache[cache_key] = config_data.copy()

        # Validate if requested
        if validate:
            try:
                experiment_config = self._dict_to_experiment_config(config_data)
                validation_result = ConfigValidator.validate_config(experiment_config)

                if validation_result.has_errors:
                    error_messages = [
                        issue.message for issue in validation_result.get_errors()
                    ]
                    raise ValueError(
                        f"Configuration validation failed: {error_messages}"
                    )

                if validation_result.has_warnings:
                    warning_messages = [
                        issue.message for issue in validation_result.get_warnings()
                    ]
                    logger.warning(f"Configuration warnings: {warning_messages}")

            except Exception as e:
                logger.warning(f"Could not validate config as ExperimentConfig: {e}")

        logger.info(f"Loaded configuration from {resolved_path}")
        return config_data

    def save_config(
        self,
        config: dict[str, Any] | ExperimentConfig,
        output_path: str | Path,
        format: str = "yaml",
    ) -> None:
        """Save configuration to file."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert ExperimentConfig to dict if needed
        if isinstance(config, ExperimentConfig):
            config_data = self._experiment_config_to_dict(config)
        else:
            config_data = config

        # Save based on format
        if format.lower() in ["yaml", "yml"]:
            with output_path.open("w") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with output_path.open("w") as f:
                json.dump(config_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved configuration to {output_path}")

    def compose_configs(
        self,
        base_config: str | Path | dict[str, Any],
        overrides: list[str | Path | dict[str, Any]] | None = None,
        parameter_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compose configuration from base config and overrides."""

        # Load base config
        if isinstance(base_config, str | Path):
            result = self.load_config(base_config, validate=False)
        else:
            result = base_config.copy()

        # Apply config overrides
        if overrides:
            for override in overrides:
                if isinstance(override, str | Path):
                    override_data = self.load_config(override, validate=False)
                else:
                    override_data = override

                result = self._deep_merge_dict(result, override_data)

        # Apply parameter overrides
        if parameter_overrides:
            for key_path, value in parameter_overrides.items():
                self._set_nested_value(result, key_path, value)

        return result

    def create_experiment_config(
        self, config_data: dict[str, Any], validate: bool = True
    ) -> ExperimentConfig:
        """Create ExperimentConfig from dictionary."""

        experiment_config = self._dict_to_experiment_config(config_data)

        if validate:
            validation_result = ConfigValidator.validate_config(experiment_config)

            if validation_result.has_errors:
                error_messages = [
                    issue.message for issue in validation_result.get_errors()
                ]
                raise ValueError(f"Configuration validation failed: {error_messages}")

            if validation_result.has_warnings:
                warning_messages = [
                    issue.message for issue in validation_result.get_warnings()
                ]
                logger.warning(f"Configuration warnings: {warning_messages}")

        return experiment_config

    def list_configs(self, pattern: str = "*.yaml") -> list[Path]:
        """List available configuration files."""
        return list(self.config_dir.glob(pattern))

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self.config_cache.clear()
        logger.debug("Cleared configuration cache")

    def _load_config_file(self, config_path: Path) -> dict[str, Any]:
        """Load configuration file based on extension."""

        suffix = config_path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            with config_path.open() as f:
                return cast(dict[str, Any], yaml.safe_load(f) or {})

        elif suffix == ".json":
            with config_path.open() as f:
                return cast(dict[str, Any], json.load(f))

        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}")

    def _substitute_environment_variables(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration."""

        if isinstance(config, dict):
            return {
                key: self._substitute_environment_variables(value)
                for key, value in config.items()
            }

        if isinstance(config, list):
            return [self._substitute_environment_variables(item) for item in config]

        if isinstance(config, str):
            # Simple environment variable substitution: ${VAR_NAME} or $VAR_NAME
            if config.startswith("${") and config.endswith("}"):
                var_name = config[2:-1]
                return os.environ.get(var_name, config)
            if config.startswith("$"):
                var_name = config[1:]
                return os.environ.get(var_name, config)
            return config

        return config

    def _deep_merge_dict(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value

        return result

    def _set_nested_value(
        self, config: dict[str, Any], key_path: str, value: Any
    ) -> None:
        """Set a nested value in config using dot notation."""
        keys = key_path.split(".")
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    def _dict_to_experiment_config(
        self, config_data: dict[str, Any]
    ) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig with proper type handling."""

        # Helper function to safely extract nested config with precise typing

        @overload
        def extract_config(
            key: str, config_class: type[T], required: Literal[True] = ...
        ) -> T: ...

        @overload
        def extract_config(
            key: str, config_class: type[T], required: Literal[False]
        ) -> T | None: ...

        def extract_config(
            key: str, config_class: type[T], required: bool = True
        ) -> T | None:
            if key in config_data:
                return config_class(**config_data[key])
            if required:
                return config_class()
            return None

        # Extract all nested configurations
        model = extract_config("model", ModelConfig)
        training = extract_config("training", TrainingConfig)
        data = extract_config("data", DataConfig)
        optimizer = extract_config("optimizer", OptimizerConfig)
        scheduler = extract_config("scheduler", SchedulerConfig, required=False)
        slurm = extract_config("slurm", SLURMConfig, required=False)
        logging_config = extract_config("logging", LoggingConfig)
        checkpoint = extract_config("checkpoint", CheckpointConfig)
        preemption = extract_config("preemption", PreemptionConfig)
        performance = extract_config("performance", PerformanceConfig)

        # Create main config
        return ExperimentConfig(
            experiment_name=config_data["experiment_name"],
            model=model,
            training=training,
            data=data,
            optimizer=optimizer,
            scheduler=scheduler,
            slurm=slurm,
            logging=logging_config,
            checkpoint=checkpoint,
            preemption=preemption,
            performance=performance,
            description=config_data.get("description"),
            tags=config_data.get("tags", []),
            created_by=config_data.get("created_by"),
            created_at=config_data.get("created_at"),
            version=config_data.get("version", "1.0"),
            seed=config_data.get("seed"),
            deterministic=config_data.get("deterministic", True),
            benchmark=config_data.get("benchmark", False),
            custom_params=config_data.get("custom_params", {}),
        )

    def _experiment_config_to_dict(self, config: ExperimentConfig) -> dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        return asdict(config)

    def validate_project_structure(self) -> ValidationResult:
        """Validate project structure for configuration management."""
        result = ValidationResult(is_valid=True)

        # Check if project root exists
        if not self.project_root.exists():
            result.add_error(
                "structure", f"Project root does not exist: {self.project_root}"
            )
            return result

        # Check if config directory exists
        if not self.config_dir.exists():
            result.add_warning(
                "structure", f"Config directory does not exist: {self.config_dir}"
            )

        # Check for common configuration files
        common_configs = ["default.yaml", "base.yaml", "config.yaml"]
        found_configs = []

        for config_name in common_configs:
            config_path = self.config_dir / config_name
            if config_path.exists():
                found_configs.append(config_name)

        if not found_configs:
            result.add_info(
                "structure", f"No common config files found in {self.config_dir}"
            )

        # Check for required directories
        required_dirs = ["experiments", "scripts"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                result.add_warning(
                    "structure", f"Recommended directory missing: {dir_path}"
                )

        return result
