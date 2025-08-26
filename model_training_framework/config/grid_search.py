"""
Parameter Grid Search Engine

This module provides sophisticated parameter grid search capabilities:
- Definition of parameter search spaces
- Generation of all parameter permutations
- Experiment naming and organization
- Grid search execution and management
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import itertools
import json
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from .naming import ExperimentNaming
from .schemas import (
    CheckpointConfig,
    DataConfig,
    ExecutionMode,
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
    """Defines a parameter search space."""

    name: str
    description: str = ""
    parameters: dict[str, list[Any]] = field(default_factory=dict)

    def add_parameter(self, key: str, values: list[Any]) -> None:
        """Add a parameter with possible values."""
        if not values:
            raise ValueError(f"Parameter '{key}' must have at least one value")
        self.parameters[key] = values
        logger.debug(
            f"Added parameter '{key}' with {len(values)} values to grid '{self.name}'"
        )

    def add_nested_parameter(self, key_path: str, values: list[Any]) -> None:
        """Add nested parameter (e.g., 'model.lr', 'optimizer.weight_decay')."""
        self.add_parameter(key_path, values)

    def remove_parameter(self, key: str) -> None:
        """Remove a parameter from the grid."""
        if key in self.parameters:
            del self.parameters[key]
            logger.debug(f"Removed parameter '{key}' from grid '{self.name}'")

    def get_parameter_count(self) -> int:
        """Get total number of parameter combinations."""
        if not self.parameters:
            return 0

        count = 1
        for values in self.parameters.values():
            count *= len(values)
        return count

    def get_parameter_names(self) -> set[str]:
        """Get set of all parameter names."""
        return set(self.parameters.keys())

    def generate_permutations(self) -> Iterator[dict[str, Any]]:
        """Generate all parameter combinations."""
        if not self.parameters:
            yield {}
            return

        keys = list(self.parameters.keys())
        value_lists = [self.parameters[key] for key in keys]

        for combination in itertools.product(*value_lists):
            yield dict(zip(keys, combination))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterGrid:
        """Create from dictionary."""
        grid = cls(name=data["name"], description=data.get("description", ""))
        grid.parameters = data.get("parameters", {})
        return grid


@dataclass
class GridSearchResult:
    """Result of a parameter grid search execution."""

    grid_config: GridSearchConfig
    total_experiments: int
    generated_experiments: list[ExperimentConfig] = field(default_factory=list)
    submitted_jobs: list[str] = field(default_factory=list)
    failed_submissions: list[tuple[str, str]] = field(
        default_factory=list
    )  # (experiment_name, error)
    execution_time: float = 0.0
    output_directory: Path | None = None

    @property
    def success_rate(self) -> float:
        """Calculate submission success rate."""
        if self.total_experiments == 0:
            return 1.0
        return len(self.submitted_jobs) / self.total_experiments

    def get_summary(self) -> dict[str, Any]:
        """Get execution summary."""
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


class ParameterGridSearch:
    """Handles parameter permutation and experiment naming."""

    def __init__(self, base_config: dict[str, Any]):
        """Initialize with base configuration."""
        # Deep copy to avoid shared nested structures across experiments
        self.base_config = deepcopy(base_config)
        self.parameter_grids: list[ParameterGrid] = []
        self.naming_strategy = NamingStrategy.HASH_BASED

    def add_grid(self, grid: ParameterGrid) -> None:
        """Add a parameter grid for exploration."""
        self.parameter_grids.append(grid)
        logger.info(
            f"Added parameter grid '{grid.name}' with {grid.get_parameter_count()} combinations"
        )

    def create_grid(self, name: str, description: str = "") -> ParameterGrid:
        """Create and add a new parameter grid."""
        grid = ParameterGrid(name=name, description=description)
        self.add_grid(grid)
        return grid

    def set_naming_strategy(self, strategy: NamingStrategy) -> None:
        """Set the experiment naming strategy."""
        self.naming_strategy = strategy
        logger.debug(f"Set naming strategy to {strategy.value}")

    def get_total_experiments(self) -> int:
        """Get total number of experiments across all grids."""
        total = 0
        for grid in self.parameter_grids:
            total += grid.get_parameter_count()
        return total

    def validate_grids(self) -> list[str]:
        """Validate parameter grids and return any issues."""
        issues = []

        if not self.parameter_grids:
            issues.append("No parameter grids defined")

        # Check for conflicting parameter paths
        all_params: set[str] = set()
        for grid in self.parameter_grids:
            grid_params = grid.get_parameter_names()
            conflicts = all_params.intersection(grid_params)
            if conflicts:
                issues.append(f"Parameter conflicts between grids: {conflicts}")
            all_params.update(grid_params)

        # Check grid sizes
        for grid in self.parameter_grids:
            if grid.get_parameter_count() == 0:
                issues.append(f"Grid '{grid.name}' has no parameter combinations")
            elif grid.get_parameter_count() > MAX_GRID_COMBINATIONS:
                issues.append(
                    f"Grid '{grid.name}' has too many combinations: {grid.get_parameter_count()}"
                )

        return issues

    def _apply_parameters_to_config(
        self, config: dict[str, Any], parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply parameter overrides to config using nested key paths."""
        # Deep copy to isolate nested structures for each experiment
        result = deepcopy(config)

        for key_path, value in parameters.items():
            self._set_nested_value(result, key_path, value)

        return result

    def _set_nested_value(
        self, config: dict[str, Any], key_path: str, value: Any
    ) -> None:
        """Set a nested value in config using dot notation."""
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
        """Generate all experiment configurations."""
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
                    # This is a simplified conversion - in practice, you'd want more robust config parsing
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

    def _dict_to_experiment_config(
        self, config_dict: dict[str, Any]
    ) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig (full fidelity).

        Mirrors ConfigurationManager's conversion so all sections are preserved
        (checkpoint, preemption, performance, slurm, custom_params, etc.).
        """
        # Nested configs (safe defaults when missing)
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
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
        """Save grid search configuration to file."""
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
        """Load grid search configuration from file."""
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


class GridSearchExecutor:
    """Orchestrates parameter grid search execution."""

    def __init__(self, launcher=None, config_manager=None):
        """Initialize with launcher and config manager."""
        self.launcher = launcher
        self.config_manager = config_manager

    def execute_grid_search(
        self,
        grid_search: ParameterGridSearch,
        execution_mode: ExecutionMode = ExecutionMode.SLURM,
        output_dir: Path | None = None,
        max_concurrent_jobs: int | None = None,
    ) -> GridSearchResult:
        """Execute parameter grid search."""
        start_time = time.time()

        # Validate grids
        issues = grid_search.validate_grids()
        if issues:
            raise ValueError(f"Grid validation failed: {issues}")

        # Setup output directory
        if output_dir is None:
            output_dir = Path(f"grid_search_{int(start_time)}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save grid configuration
        grid_search.save_grid_config(output_dir / "grid_config.json")

        # Generate experiments
        experiments = list(grid_search.generate_experiments())
        total_experiments = len(experiments)

        logger.info(f"Generated {total_experiments} experiments for grid search")

        # Create result object
        grid_config = GridSearchConfig(
            name=f"grid_search_{int(start_time)}",
            base_config=grid_search.base_config,
            execution_mode=execution_mode,
            output_dir=str(output_dir),
        )

        result = GridSearchResult(
            grid_config=grid_config,
            total_experiments=total_experiments,
            output_directory=output_dir,
            generated_experiments=experiments,
        )

        # Execute based on mode
        if execution_mode == ExecutionMode.DRY_RUN:
            logger.info("Dry run mode - not executing experiments")
            result.submitted_jobs = [exp.experiment_name for exp in experiments]

        elif execution_mode == ExecutionMode.LOCAL:
            logger.info("Local execution not yet implemented")
            # TODO: Implement local execution

        elif execution_mode == ExecutionMode.SLURM:
            if self.launcher is None:
                raise ValueError("SLURM launcher required for SLURM execution mode")

            # Submit experiments to SLURM
            try:
                submission_result = self.launcher.submit_experiment_batch(experiments)
                result.submitted_jobs = submission_result.successful_jobs
                result.failed_submissions = [
                    (exp_name, error)
                    for exp_name, error in submission_result.failed_jobs.items()
                ]
            except Exception as e:
                logger.exception("Failed to submit experiments")
                result.failed_submissions = [
                    (exp.experiment_name, str(e)) for exp in experiments
                ]

        result.execution_time = time.time() - start_time

        # Save result summary
        summary_path = output_dir / "execution_summary.json"
        with summary_path.open("w") as f:
            json.dump(result.get_summary(), f, indent=2)

        logger.info(f"Grid search completed in {result.execution_time:.2f}s")
        logger.info(f"Success rate: {result.success_rate:.2%}")

        return result
