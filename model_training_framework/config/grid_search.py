"""
Parameter Grid Search Engine

This module provides sophisticated parameter grid search capabilities:
- Definition of parameter search spaces
- Generation of all parameter permutations
- Experiment naming and organization
- Grid search execution and management
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import itertools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .naming import ExperimentNaming
from .schemas import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    GridSearchConfig,
    LoggingConfig,
    ModelConfig,
    NamingStrategy,
    OptimizerConfig,
    PerformanceConfig,
    PreemptionConfig,
    SchedulerConfig,
    SLURMConfig,
    TrainingConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Constants for grid search limits
MAX_GRID_COMBINATIONS = 10000


@dataclass
class ParameterGrid:
    """Defines a parameter search space with improved API.

    Attributes:
        name: Name of the parameter grid
        description: Optional description of what this grid explores
        parameters: Dictionary mapping parameter paths to lists of values
    """

    name: str
    description: str = ""
    parameters: dict[str, list[Any]] = field(default_factory=dict)

    def add_parameter(
        self, key: str, values: list[Any], validate: Callable[[Any], bool] | None = None
    ) -> ParameterGrid:
        """Add a parameter with possible values.

        Args:
            key: Parameter path (e.g., 'model.lr' or 'training.batch_size')
            values: List of possible values for this parameter
            validate: Optional validation function for values

        Returns:
            Self for method chaining

        Raises:
            ValueError: If values is empty or validation fails
        """
        if not values:
            raise ValueError(f"Parameter '{key}' must have at least one value")

        if validate:
            invalid_values = [v for v in values if not validate(v)]
            if invalid_values:
                raise ValueError(f"Invalid values for '{key}': {invalid_values}")

        self.parameters[key] = values
        logger.debug(
            f"Added parameter '{key}' with {len(values)} values to grid '{self.name}'"
        )
        return self

    def add_nested_parameter(self, key_path: str, values: list[Any]) -> ParameterGrid:
        """Add nested parameter (e.g., 'model.lr', 'optimizer.weight_decay').

        Args:
            key_path: Dot-separated path to parameter
            values: List of possible values

        Returns:
            Self for method chaining
        """
        return self.add_parameter(key_path, values)

    def remove_parameter(self, key: str) -> ParameterGrid:
        """Remove a parameter from the grid.

        Args:
            key: Parameter key to remove

        Returns:
            Self for method chaining
        """
        if key in self.parameters:
            del self.parameters[key]
            logger.debug(f"Removed parameter '{key}' from grid '{self.name}'")
        return self

    def get_parameter_count(self) -> int:
        """Get total number of parameter combinations.

        Returns:
            Number of unique parameter combinations
        """
        if not self.parameters:
            return 0

        count = 1
        for values in self.parameters.values():
            count *= len(values)
        return count

    def get_parameter_names(self) -> set[str]:
        """Get set of all parameter names.

        Returns:
            Set of parameter keys
        """
        return set(self.parameters.keys())

    def generate_permutations(self) -> Iterator[dict[str, Any]]:
        """Generate all parameter combinations.

        Yields:
            Dictionary mapping parameter keys to values
        """
        if not self.parameters:
            yield {}
            return

        keys = list(self.parameters.keys())
        value_lists = [self.parameters[key] for key in keys]

        for combination in itertools.product(*value_lists):
            yield dict(zip(keys, combination))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterGrid:
        """Create from dictionary.

        Args:
            data: Dictionary containing grid configuration

        Returns:
            ParameterGrid instance
        """
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
        )

    def validate(self) -> list[str]:
        """Validate the parameter grid.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if not self.name:
            issues.append("Grid name is required")

        if self.get_parameter_count() == 0:
            issues.append(f"Grid '{self.name}' has no parameter combinations")
        elif self.get_parameter_count() > MAX_GRID_COMBINATIONS:
            issues.append(
                f"Grid '{self.name}' has too many combinations: {self.get_parameter_count()}"
            )

        return issues


@dataclass
class GridSearchResult:
    """Result of a parameter grid search execution with improved API.

    Attributes:
        grid_config: Configuration used for the grid search
        total_experiments: Total number of experiments generated
        generated_experiments: List of generated experiment configurations
        submitted_jobs: List of successfully submitted job IDs
        failed_submissions: List of (experiment_name, error) tuples
        execution_time: Total execution time in seconds
        output_directory: Directory where results are stored
    """

    grid_config: GridSearchConfig
    total_experiments: int
    generated_experiments: list[ExperimentConfig] = field(default_factory=list)
    submitted_jobs: list[str] = field(default_factory=list)
    failed_submissions: list[tuple[str, str]] = field(default_factory=list)
    execution_time: float = 0.0
    output_directory: Path | None = None

    @property
    def success_rate(self) -> float:
        """Calculate submission success rate.

        Returns:
            Success rate as a fraction (0.0 to 1.0)
        """
        if self.total_experiments == 0:
            return 1.0
        return len(self.submitted_jobs) / self.total_experiments

    @property
    def failure_rate(self) -> float:
        """Calculate submission failure rate.

        Returns:
            Failure rate as a fraction (0.0 to 1.0)
        """
        return 1.0 - self.success_rate

    @property
    def is_complete(self) -> bool:
        """Check if all experiments were submitted successfully.

        Returns:
            True if all experiments succeeded
        """
        return len(self.failed_submissions) == 0

    def get_summary(self) -> dict[str, Any]:
        """Get execution summary.

        Returns:
            Dictionary with summary statistics
        """
        return {
            "grid_name": self.grid_config.name,
            "total_experiments": self.total_experiments,
            "successful_submissions": len(self.submitted_jobs),
            "failed_submissions": len(self.failed_submissions),
            "success_rate": self.success_rate,
            "execution_time_sec": self.execution_time,
            "output_directory": str(self.output_directory)
            if self.output_directory
            else None,
        }

    def save_summary(self, path: Path | None = None) -> Path:
        """Save summary to JSON file.

        Args:
            path: Optional path for summary file. If None, saves to output_directory.

        Returns:
            Path where summary was saved

        Raises:
            ValueError: If no path provided and no output_directory set
        """
        if path is None:
            if self.output_directory is None:
                raise ValueError("No path provided and output_directory not set")
            path = self.output_directory / "execution_summary.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.get_summary(), f, indent=2)

        logger.info(f"Saved grid search summary to {path}")
        return path


class ParameterGridSearch:
    """Handles parameter permutation and experiment naming with improved API."""

    def __init__(self, base_config: ExperimentConfig | dict[str, Any]):
        """Initialize with base configuration.

        Args:
            base_config: Either an ExperimentConfig object or a dictionary
        """
        # Convert ExperimentConfig to dict if needed, then deep copy
        self._base_experiment_config: ExperimentConfig | None
        if isinstance(base_config, ExperimentConfig):
            self.base_config = deepcopy(self._experiment_config_to_dict(base_config))
            self._base_experiment_config = base_config
        else:
            self.base_config = deepcopy(base_config)
            self._base_experiment_config = None

        self.parameter_grids: list[ParameterGrid] = []
        self.naming_strategy = NamingStrategy.HASH_BASED

    def add_grid(self, grid: ParameterGrid) -> ParameterGridSearch:
        """Add a parameter grid for exploration.

        Args:
            grid: ParameterGrid to add

        Returns:
            Self for method chaining
        """
        # Validate the grid first
        issues = grid.validate()
        if issues:
            logger.warning(f"Grid '{grid.name}' has validation issues: {issues}")

        self.parameter_grids.append(grid)
        logger.info(
            f"Added parameter grid '{grid.name}' with {grid.get_parameter_count()} combinations"
        )
        return self

    def create_grid(self, name: str, description: str = "") -> ParameterGrid:
        """Create and add a new parameter grid.

        Args:
            name: Name for the new grid
            description: Optional description

        Returns:
            The created ParameterGrid
        """
        grid = ParameterGrid(name=name, description=description)
        self.add_grid(grid)
        return grid

    def remove_grid(self, name: str) -> ParameterGridSearch:
        """Remove a parameter grid by name.

        Args:
            name: Name of grid to remove

        Returns:
            Self for method chaining
        """
        self.parameter_grids = [g for g in self.parameter_grids if g.name != name]
        return self

    def set_naming_strategy(self, strategy: NamingStrategy) -> ParameterGridSearch:
        """Set the experiment naming strategy.

        Args:
            strategy: NamingStrategy to use

        Returns:
            Self for method chaining
        """
        self.naming_strategy = strategy
        logger.debug(f"Set naming strategy to {strategy.value}")
        return self

    def get_total_experiments(self) -> int:
        """Get total number of experiments across all grids.

        Returns:
            Total number of experiments that will be generated
        """
        return sum(grid.get_parameter_count() for grid in self.parameter_grids)

    def validate_grids(self) -> list[str]:
        """Validate all parameter grids and check for conflicts.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if not self.parameter_grids:
            issues.append("No parameter grids defined")

        # Check each grid individually
        for grid in self.parameter_grids:
            grid_issues = grid.validate()
            issues.extend(f"Grid '{grid.name}': {issue}" for issue in grid_issues)

        # Check for conflicting parameter paths
        all_params: set[str] = set()
        for grid in self.parameter_grids:
            grid_params = grid.get_parameter_names()
            conflicts = all_params.intersection(grid_params)
            if conflicts:
                issues.append(f"Parameter conflicts between grids: {conflicts}")
            all_params.update(grid_params)

        # Check total experiment count
        total = self.get_total_experiments()
        if total > MAX_GRID_COMBINATIONS:
            issues.append(
                f"Total experiments ({total}) exceeds maximum ({MAX_GRID_COMBINATIONS})"
            )

        return issues

    def _apply_parameters_to_config(
        self, config: dict[str, Any], parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply parameter overrides to config using nested key paths.

        Args:
            config: Base configuration dictionary
            parameters: Parameters to apply

        Returns:
            New configuration with parameters applied
        """
        # Deep copy to isolate nested structures for each experiment
        result = deepcopy(config)

        for key_path, value in parameters.items():
            self._set_nested_value(result, key_path, value)

        return result

    def _set_nested_value(
        self, config: dict[str, Any], key_path: str, value: Any
    ) -> None:
        """Set a nested value in config using dot notation.

        Args:
            config: Configuration dictionary to modify
            key_path: Dot-separated path to value
            value: Value to set
        """
        keys = key_path.split(".")
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Convert to dict if it's not already
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    def generate_experiments(self) -> Iterator[ExperimentConfig]:
        """Generate all experiment configurations.

        Yields:
            ExperimentConfig for each parameter combination
        """
        for grid in self.parameter_grids:
            for parameters in grid.generate_permutations():
                # Apply parameters to base config
                experiment_config_dict = self._apply_parameters_to_config(
                    self.base_config, parameters
                )

                # Generate experiment name
                experiment_name = ExperimentNaming.generate_name(
                    base_name=experiment_config_dict.get(
                        "experiment_name", "experiment"
                    ),
                    parameters=parameters,
                    naming_strategy=self.naming_strategy,
                )
                experiment_config_dict["experiment_name"] = experiment_name

                # Convert to ExperimentConfig
                try:
                    experiment_config = self._dict_to_experiment_config(
                        experiment_config_dict
                    )
                    yield experiment_config
                except Exception:
                    logger.exception(
                        f"Failed to create experiment config for {experiment_name}"
                    )
                    continue

    def generate_experiment_dicts(self) -> Iterator[dict[str, Any]]:
        """Generate experiment dictionaries with overrides applied.

        Useful when callers want to handle ExperimentConfig conversion
        themselves (e.g., via ConfigurationManager) or serialize raw dicts.

        Yields:
            Dictionary for each parameter combination
        """
        for grid in self.parameter_grids:
            for parameters in grid.generate_permutations():
                # Apply parameters to base config
                cfg = self._apply_parameters_to_config(self.base_config, parameters)
                # Name using configured strategy
                cfg["experiment_name"] = ExperimentNaming.generate_name(
                    base_name=cfg.get("experiment_name", "experiment"),
                    parameters=parameters,
                    naming_strategy=self.naming_strategy,
                )
                yield cfg

    def _experiment_config_to_dict(self, config: ExperimentConfig) -> dict[str, Any]:
        """Convert ExperimentConfig to dictionary.

        Args:
            config: ExperimentConfig to convert

        Returns:
            Dictionary representation of the config
        """
        # Use asdict but handle custom configs specially
        result: dict[str, Any] = {}

        # Required fields
        result["experiment_name"] = config.experiment_name

        # Convert each config component
        result["model"] = (
            config.model.to_dict()
            if hasattr(config.model, "to_dict")
            else asdict(config.model)
        )
        result["data"] = (
            config.data.to_dict()
            if hasattr(config.data, "to_dict")
            else asdict(config.data)
        )
        result["training"] = asdict(config.training)
        result["optimizer"] = asdict(config.optimizer)

        # Optional fields
        if config.scheduler:
            result["scheduler"] = asdict(config.scheduler)
        if config.slurm:
            result["slurm"] = asdict(config.slurm)

        result["logging"] = asdict(config.logging)
        result["checkpoint"] = asdict(config.checkpoint)
        result["preemption"] = asdict(config.preemption)
        result["performance"] = asdict(config.performance)

        # Metadata
        if config.description:
            result["description"] = config.description
        result["tags"] = config.tags
        if config.created_by:
            result["created_by"] = config.created_by
        if config.created_at:
            result["created_at"] = config.created_at
        result["version"] = config.version

        # Runtime settings
        if config.seed is not None:
            result["seed"] = config.seed
        result["deterministic"] = config.deterministic
        result["benchmark"] = config.benchmark
        result["custom_params"] = config.custom_params

        return result

    def _dict_to_experiment_config(
        self, config_dict: dict[str, Any]
    ) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig (full fidelity).

        Mirrors ConfigurationManager's conversion so all sections are preserved
        (checkpoint, preemption, performance, slurm, custom_params, etc.).

        Args:
            config_dict: Dictionary to convert

        Returns:
            ExperimentConfig instance
        """
        # Nested configs (safe defaults when missing)
        # Use from_dict for flexible configs, direct instantiation for others
        model_config = ModelConfig.from_dict(config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig.from_dict(config_dict.get("data", {}))
        optimizer_config = OptimizerConfig(**config_dict.get("optimizer", {}))
        scheduler_config = (
            SchedulerConfig(**config_dict["scheduler"])
            if "scheduler" in config_dict and config_dict["scheduler"] is not None
            else None
        )
        slurm_config = (
            SLURMConfig(**config_dict["slurm"])
            if "slurm" in config_dict and config_dict["slurm"] is not None
            else None
        )
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        checkpoint_config = CheckpointConfig(**config_dict.get("checkpoint", {}))
        preemption_config = PreemptionConfig(**config_dict.get("preemption", {}))
        performance_config = PerformanceConfig(**config_dict.get("performance", {}))

        return ExperimentConfig(
            experiment_name=config_dict["experiment_name"],
            model=model_config,
            training=training_config,
            data=data_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            slurm=slurm_config,
            logging=logging_config,
            checkpoint=checkpoint_config,
            preemption=preemption_config,
            performance=performance_config,
            description=config_dict.get("description"),
            tags=config_dict.get("tags", []),
            created_by=config_dict.get("created_by"),
            created_at=config_dict.get("created_at"),
            version=config_dict.get("version", "1.0"),
            seed=config_dict.get("seed"),
            deterministic=config_dict.get("deterministic", True),
            benchmark=config_dict.get("benchmark", False),
            custom_params=config_dict.get("custom_params", {}),
        )

    def save_grid_config(self, output_path: Path) -> None:
        """Save grid search configuration to file.

        Args:
            output_path: Path where to save configuration
        """
        config_data = {
            "base_config": self.base_config,
            "naming_strategy": self.naming_strategy.value,
            "grids": [grid.to_dict() for grid in self.parameter_grids],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(config_data, f, indent=2, default=str)

        logger.info(f"Saved grid search configuration to {output_path}")

    @classmethod
    def load_grid_config(cls, config_path: Path) -> ParameterGridSearch:
        """Load grid search configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            ParameterGridSearch instance
        """
        with config_path.open() as f:
            config_data = json.load(f)

        grid_search = cls(base_config=config_data["base_config"])
        grid_search.naming_strategy = NamingStrategy(
            config_data.get("naming_strategy", "hash_based")
        )

        for grid_data in config_data.get("grids", []):
            grid = ParameterGrid.from_dict(grid_data)
            grid_search.add_grid(grid)

        logger.info(f"Loaded grid search configuration from {config_path}")
        return grid_search
