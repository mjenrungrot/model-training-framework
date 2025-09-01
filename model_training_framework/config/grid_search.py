"""
Parameter Grid Search Engine

This module provides sophisticated parameter grid search capabilities:
- Definition of parameter search spaces with multiple specification methods
- Linked parameters that vary together (inspired by pytest.mark.parametrize)
- Conditional parameters based on other parameter values
- Computed parameters derived from other parameters
- Parameter sampling from distributions
- Constraints and exclusion patterns for filtering combinations
- Generation of all parameter permutations
- Experiment naming and organization
- Grid search execution and management
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
import itertools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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


class ParameterType(Enum):
    """Types of parameter specifications."""

    SINGLE = "single"  # Traditional single parameter
    LINKED = "linked"  # Parameters that vary together
    CONDITIONAL = "conditional"  # Parameters that depend on conditions
    COMPUTED = "computed"  # Parameters computed from others
    DISTRIBUTION = "distribution"  # Parameters sampled from distribution


@dataclass
class ParameterSpec:
    """Base class for parameter specifications."""

    param_type: ParameterType
    keys: list[str] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)


@dataclass
class LinkedParameterSpec(ParameterSpec):
    """Specification for linked parameters that vary together."""

    def __init__(
        self, keys: list[str], value_sets: Sequence[tuple[Any, ...] | dict[str, Any]]
    ):
        super().__init__(param_type=ParameterType.LINKED, keys=keys)
        # Convert tuples to dicts for consistent handling
        self.values = []
        for vs in value_sets:
            if isinstance(vs, dict):
                self.values.append(vs)
            else:
                self.values.append(dict(zip(keys, vs)))


@dataclass
class ConditionalParameterSpec(ParameterSpec):
    """Specification for conditional parameters."""

    condition: dict[str, Any] = field(default_factory=dict)
    condition_func: Callable[[dict[str, Any]], bool] | None = None

    def __init__(
        self,
        key: str,
        values: list[Any],
        when: dict[str, Any] | None = None,
        when_func: Callable[[dict[str, Any]], bool] | None = None,
    ):
        super().__init__(
            param_type=ParameterType.CONDITIONAL, keys=[key], values=values
        )
        self.condition = when or {}
        self.condition_func = when_func

    def applies_to(self, params: dict[str, Any]) -> bool:
        """Check if this conditional parameter applies to given parameters."""
        if self.condition_func:
            return self.condition_func(params)
        return all(params.get(k) == v for k, v in self.condition.items())


@dataclass
class ComputedParameterSpec(ParameterSpec):
    """Specification for computed parameters."""

    compute_func: Callable[[dict[str, Any]], Any] = field(default=lambda x: None)

    def __init__(self, key: str, compute_func: Callable[[dict[str, Any]], Any]):
        super().__init__(param_type=ParameterType.COMPUTED, keys=[key])
        self.compute_func = compute_func


@dataclass
class DistributionParameterSpec(ParameterSpec):
    """Specification for parameters sampled from distributions."""

    distribution: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    n_samples: int = 10
    seed: int | None = None

    def __init__(
        self,
        key: str,
        distribution: str,
        n_samples: int = 10,
        seed: int | None = None,
        **dist_params: Any,
    ):
        super().__init__(param_type=ParameterType.DISTRIBUTION, keys=[key])
        self.distribution = distribution
        self.params = dist_params
        self.n_samples = n_samples
        self.seed = seed
        self._generate_samples()

    def _generate_samples(self) -> None:
        """Generate samples from the distribution."""
        # Set seed if provided for reproducibility
        rng = np.random.RandomState(self.seed) if self.seed is not None else np.random

        if self.distribution == "uniform":
            low = self.params.get("low", 0)
            high = self.params.get("high", 1)
            self.values = list(rng.uniform(low, high, self.n_samples))
        elif self.distribution == "loguniform":
            low = np.log10(self.params.get("low", 1e-5))
            high = np.log10(self.params.get("high", 1e-1))
            self.values = list(10 ** rng.uniform(low, high, self.n_samples))
        elif self.distribution == "normal":
            mean = self.params.get("mean", 0)
            std = self.params.get("std", 1)
            self.values = list(rng.normal(mean, std, self.n_samples))
        elif self.distribution == "choice":
            choices = self.params.get("choices", [])
            if not choices:
                raise ValueError(
                    "Choice distribution requires non-empty 'choices' parameter"
                )
            weights = self.params.get("weights")
            if weights is not None and len(weights) != len(choices):
                raise ValueError(
                    f"Length of 'weights' ({len(weights)}) must match length of 'choices' ({len(choices)})"
                )
            self.values = list(
                rng.choice(choices, size=self.n_samples, p=weights, replace=True)
            )
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class ParameterGrid:
    """Defines a parameter search space with improved API.

    Attributes:
        name: Name of the parameter grid
        description: Optional description of what this grid explores
        parameters: Dictionary mapping parameter paths to lists of values
        parameter_specs: List of advanced parameter specifications
        constraints: List of constraint functions
        exclude_patterns: List of patterns to exclude
    """

    name: str
    description: str = ""
    parameters: dict[str, list[Any]] = field(default_factory=dict)
    parameter_specs: list[ParameterSpec] = field(default_factory=list)
    constraints: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)
    exclude_patterns: list[dict[str, Any] | Callable[[dict[str, Any]], bool]] = field(
        default_factory=list
    )

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

    def add_linked_parameters(
        self, keys: list[str], value_sets: Sequence[tuple[Any, ...] | dict[str, Any]]
    ) -> ParameterGrid:
        """Add parameters that vary together (inspired by pytest.mark.parametrize).

        Args:
            keys: Parameter keys that vary together
            value_sets: List of value tuples or dicts for the parameters

        Example:
            grid.add_linked_parameters(
                ["lr", "weight_decay"],
                [(0.1, 0.1), (0.01, 0.01), (0.001, 0.001)]
            )

        Returns:
            Self for method chaining
        """
        spec = LinkedParameterSpec(keys, value_sets)
        self.parameter_specs.append(spec)
        logger.debug(
            f"Added linked parameters {keys} with {len(value_sets)} value sets to grid '{self.name}'"
        )
        return self

    def add_parameter_sets(self, parameter_sets: list[dict[str, Any]]) -> ParameterGrid:
        """Add complete parameter sets that vary together.

        Args:
            parameter_sets: List of parameter dictionaries

        Example:
            grid.add_parameter_sets([
                {"lr": 0.1, "weight_decay": 0.1, "warmup": 100},
                {"lr": 0.01, "weight_decay": 0.01, "warmup": 500}
            ])

        Returns:
            Self for method chaining
        """
        if not parameter_sets:
            raise ValueError("Parameter sets cannot be empty")

        # Extract keys from first set
        keys = list(parameter_sets[0].keys())
        # Pass dictionaries directly since LinkedParameterSpec accepts them
        return self.add_linked_parameters(keys, parameter_sets)

    def add_conditional_parameter(
        self,
        key: str,
        values: list[Any],
        when: dict[str, Any] | None = None,
        when_func: Callable[[dict[str, Any]], bool] | None = None,
    ) -> ParameterGrid:
        """Add a parameter that only applies under certain conditions.

        Args:
            key: Parameter key
            values: Possible values for this parameter
            when: Dictionary of conditions (all must match)
            when_func: Function that returns True when parameter should apply

        Example:
            grid.add_conditional_parameter(
                "momentum", [0.9, 0.95],
                when={"optimizer": "sgd"}
            )

        Returns:
            Self for method chaining
        """
        spec = ConditionalParameterSpec(key, values, when, when_func)
        self.parameter_specs.append(spec)
        logger.debug(
            f"Added conditional parameter '{key}' with {len(values)} values to grid '{self.name}'"
        )
        return self

    def add_computed_parameter(
        self, key: str, compute_func: Callable[[dict[str, Any]], Any]
    ) -> ParameterGrid:
        """Add a parameter computed from other parameters.

        Args:
            key: Parameter key
            compute_func: Function to compute value from other parameters

        Example:
            grid.add_computed_parameter(
                "warmup_steps",
                lambda params: int(params["num_epochs"] * 0.1)
            )

        Returns:
            Self for method chaining
        """
        spec = ComputedParameterSpec(key, compute_func)
        self.parameter_specs.append(spec)
        logger.debug(f"Added computed parameter '{key}' to grid '{self.name}'")
        return self

    def add_parameter_from_function(
        self, key: str, generator_func: Callable[[], list[Any]]
    ) -> ParameterGrid:
        """Add parameter values generated by a function.

        Args:
            key: Parameter key
            generator_func: Function that returns list of values

        Example:
            grid.add_parameter_from_function(
                "lr",
                lambda: np.logspace(-4, -1, num=10).tolist()
            )

        Returns:
            Self for method chaining
        """
        values = generator_func()
        return self.add_parameter(key, values)

    def add_parameter_distribution(
        self,
        key: str,
        distribution: str,
        n_samples: int = 10,
        seed: int | None = None,
        **dist_params: Any,
    ) -> ParameterGrid:
        """Add parameter values sampled from a distribution.

        Args:
            key: Parameter key
            distribution: Distribution name ("uniform", "loguniform", "normal", "choice")
            n_samples: Number of samples to generate
            seed: Optional random seed for reproducibility
            **dist_params: Distribution parameters

        Example:
            grid.add_parameter_distribution(
                "lr", "loguniform", n_samples=20, seed=42, low=1e-5, high=1e-1
            )

        Returns:
            Self for method chaining
        """
        spec = DistributionParameterSpec(
            key, distribution, n_samples, seed, **dist_params
        )
        self.parameter_specs.append(spec)
        logger.debug(
            f"Added distribution parameter '{key}' with {n_samples} samples to grid '{self.name}'"
        )
        return self

    def add_constraint(
        self, constraint_func: Callable[[dict[str, Any]], bool]
    ) -> ParameterGrid:
        """Add a constraint function to filter valid parameter combinations.

        Args:
            constraint_func: Function that returns True for valid combinations

        Example:
            grid.add_constraint(
                lambda p: p["batch_size"] * p["accumulation_steps"] <= 64
            )

        Returns:
            Self for method chaining
        """
        self.constraints.append(constraint_func)
        logger.debug(f"Added constraint to grid '{self.name}'")
        return self

    def exclude_combinations(
        self, patterns: list[dict[str, Any] | Callable[[dict[str, Any]], bool]]
    ) -> ParameterGrid:
        """Exclude specific parameter combinations.

        Args:
            patterns: List of patterns to exclude (dicts or functions)

        Example:
            grid.exclude_combinations([
                {"lr": 0.1, "batch_size": 64},  # Exclude specific combo
                lambda p: p["lr"] > 0.01 and p["batch_size"] < 32  # Exclude pattern
            ])

        Returns:
            Self for method chaining
        """
        self.exclude_patterns.extend(patterns)
        logger.debug(f"Added {len(patterns)} exclusion patterns to grid '{self.name}'")
        return self

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
        # Handle empty grid case
        if not self.parameters and not self.parameter_specs:
            return 0

        # Handle traditional parameters
        traditional_count = 1
        if self.parameters:
            for values in self.parameters.values():
                traditional_count *= len(values)
        else:
            traditional_count = 0

        # Handle advanced parameter specs
        linked_count = 0
        dist_count = 1

        # Count unconditional conditional parameters
        # These always apply and should be counted
        unconditional_cond_count = 1

        for spec in self.parameter_specs:
            if spec.param_type == ParameterType.LINKED:
                # Linked parameters are unioned (summed), not maxed
                linked_count += len(spec.values)
            elif spec.param_type == ParameterType.DISTRIBUTION:
                # Distribution parameters multiply with each other
                dist_count *= len(spec.values)
            elif (
                spec.param_type == ParameterType.CONDITIONAL
                and isinstance(spec, ConditionalParameterSpec)
                and not spec.condition
                and not spec.condition_func
            ):
                # Unconditional conditional (always applies)
                unconditional_cond_count *= len(spec.values)
            # Computed parameters don't add to base count

        # Calculate total based on what we have
        base_count = traditional_count if traditional_count > 0 else 1

        # Combine counts
        if linked_count > 0:
            # Linked specs define alternative base combinations
            base_count = (
                linked_count
                if traditional_count == 0
                else traditional_count * linked_count
            )

        # Apply multiplicative factors
        total = base_count * dist_count * unconditional_cond_count

        # Special case: if we only have traditional params set to 0 and nothing else
        if (
            traditional_count == 0
            and linked_count == 0
            and dist_count == 1
            and unconditional_cond_count == 1
        ):
            return 0

        return total

    def get_parameter_names(self) -> set[str]:
        """Get set of all parameter names.

        Returns:
            Set of parameter keys
        """
        names = set(self.parameters.keys())

        # Add keys from parameter specs
        for spec in self.parameter_specs:
            names.update(spec.keys)

        return names

    def generate_permutations(self) -> Iterator[dict[str, Any]]:
        """Generate all parameter combinations.

        Yields:
            Dictionary mapping parameter keys to values
        """
        # Check for empty grid
        if not self.parameters and not self.parameter_specs:
            return  # Empty grid yields nothing

        # Generate base combinations from traditional parameters
        base_combinations = []
        if self.parameters:
            keys = list(self.parameters.keys())
            value_lists = [self.parameters[key] for key in keys]
            for combination in itertools.product(*value_lists):
                base_combinations.append(dict(zip(keys, combination)))
        else:
            base_combinations = [{}]

        # Collect special parameter specs
        linked_specs = []
        distribution_specs = []
        for spec in self.parameter_specs:
            if spec.param_type == ParameterType.LINKED:
                linked_specs.append(spec)
            elif spec.param_type == ParameterType.DISTRIBUTION:
                distribution_specs.append(spec)

        # Handle linked parameters
        if linked_specs:
            linked_combinations = []
            for spec in linked_specs:
                linked_combinations.extend(spec.values)
        else:
            linked_combinations = []

        # Handle distribution parameters (treat them like traditional parameters)
        dist_combinations = []
        if distribution_specs:
            # Each distribution spec has a key and values
            dist_keys = []
            dist_value_lists = []
            for spec in distribution_specs:
                if spec.keys:  # Should have one key
                    dist_keys.append(spec.keys[0])
                    dist_value_lists.append(spec.values)

            if dist_keys and dist_value_lists:
                for combination in itertools.product(*dist_value_lists):
                    dist_combinations.append(dict(zip(dist_keys, combination)))

        # Combine all types of parameters
        # Treat empty lists as identity element [{}] for cleaner logic
        effective_linked = linked_combinations or [{}]
        effective_dist = dist_combinations or [{}]

        # Generate all combinations with clear precedence order
        for base in base_combinations:
            for linked in effective_linked:
                for dist in effective_dist:
                    # Order determines precedence: base < linked < dist
                    combined = {**base, **linked, **dist}
                    yield from self._process_combination(combined)

    def _process_combination(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Process a single parameter combination with conditionals, computed, constraints."""
        # Collect all applicable conditional parameters
        applicable_conditionals = []
        for spec in self.parameter_specs:
            if (
                spec.param_type == ParameterType.CONDITIONAL
                and isinstance(spec, ConditionalParameterSpec)
                and spec.applies_to(params)
            ):
                applicable_conditionals.append(spec)

        if applicable_conditionals:
            # Generate all combinations of applicable conditional parameters
            cond_keys = []
            cond_value_lists = []
            for spec in applicable_conditionals:
                if spec.keys:
                    cond_keys.append(spec.keys[0])
                    cond_value_lists.append(spec.values)

            # Generate all combinations
            for combination in itertools.product(*cond_value_lists):
                extended_params = {**params}
                for key, value in zip(cond_keys, combination):
                    extended_params[key] = value
                yield from self._finalize_combination(extended_params)
        else:
            # No conditional parameters apply
            yield from self._finalize_combination(params)

    def _finalize_combination(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Apply computed parameters, constraints, and exclusions."""
        # Apply computed parameters
        for spec in self.parameter_specs:
            if spec.param_type == ParameterType.COMPUTED and isinstance(
                spec, ComputedParameterSpec
            ):
                with suppress(KeyError, TypeError, ValueError):
                    # Skip if computation fails (dependent params might not exist)
                    params[spec.keys[0]] = spec.compute_func(params)

        # Check constraints
        for constraint in self.constraints:
            if not constraint(params):
                return  # Skip this combination

        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if callable(pattern):
                if pattern(params):
                    return  # Skip this combination
            elif isinstance(pattern, dict) and all(
                params.get(k) == v for k, v in pattern.items()
            ):
                return  # Skip this combination

        yield params

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        # Serialize parameter specs
        specs_data = []
        for spec in self.parameter_specs:
            spec_dict: dict[str, Any] = {
                "type": spec.param_type.value,
                "keys": spec.keys,
                "values": spec.values,
            }

            # Add type-specific fields
            if isinstance(spec, ConditionalParameterSpec):
                spec_dict["condition"] = spec.condition
                # Note: Can't serialize condition_func
            elif isinstance(spec, ComputedParameterSpec):
                # Note: Can't serialize compute_func - will need to be re-added manually
                spec_dict["compute_func_str"] = "# Function not serializable"
            elif isinstance(spec, DistributionParameterSpec):
                spec_dict["distribution"] = spec.distribution
                spec_dict["params"] = spec.params
                spec_dict["n_samples"] = spec.n_samples
                spec_dict["seed"] = spec.seed
                # Store actual values for reproducibility
                spec_dict["values"] = spec.values

            specs_data.append(spec_dict)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "parameter_specs": specs_data,
            # Note: Constraints and exclude_patterns with functions can't be serialized
            "has_constraints": len(self.constraints) > 0,
            "has_exclusions": len(self.exclude_patterns) > 0,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterGrid:
        """Create from dictionary.

        Args:
            data: Dictionary containing grid configuration

        Returns:
            ParameterGrid instance
        """
        grid = cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
        )

        # Restore parameter specs
        for spec_data in data.get("parameter_specs", []):
            spec_type = ParameterType(spec_data["type"])

            if spec_type == ParameterType.LINKED:
                # Reconstruct linked spec
                linked_spec = LinkedParameterSpec(
                    spec_data["keys"], spec_data["values"]
                )
                grid.parameter_specs.append(linked_spec)
            elif spec_type == ParameterType.CONDITIONAL:
                # Reconstruct conditional spec (without function)
                cond_keys = spec_data.get("keys")
                if not cond_keys or len(cond_keys) != 1:
                    logger.warning(
                        f"Conditional spec must have exactly one key, got {cond_keys}. Skipping."
                    )
                    continue
                cond_spec = ConditionalParameterSpec(
                    cond_keys[0],
                    spec_data.get("values", []),
                    when=spec_data.get("condition", {}),
                )
                grid.parameter_specs.append(cond_spec)
            elif spec_type == ParameterType.DISTRIBUTION:
                # Reconstruct distribution spec
                dist_keys = spec_data.get("keys")
                if not dist_keys or len(dist_keys) != 1:
                    logger.warning(
                        f"Distribution spec must have exactly one key, got {dist_keys}. Skipping."
                    )
                    continue
                key = dist_keys[0]
                if "values" in spec_data:
                    # Use persisted values for reproducibility
                    dist_spec = DistributionParameterSpec(
                        key,
                        spec_data.get("distribution", "uniform"),
                        spec_data.get("n_samples", 10),
                        spec_data.get("seed"),
                        **spec_data.get("params", {}),
                    )
                    # Override with persisted values
                    dist_spec.values = spec_data["values"]
                    grid.parameter_specs.append(dist_spec)
                else:
                    # Re-generate if no values persisted
                    dist_spec = DistributionParameterSpec(
                        key,
                        spec_data.get("distribution", "uniform"),
                        spec_data.get("n_samples", 10),
                        spec_data.get("seed"),
                        **spec_data.get("params", {}),
                    )
                    grid.parameter_specs.append(dist_spec)
            # Note: Computed specs can't be fully restored without the function

        # Log warnings about non-serializable features
        if data.get("has_constraints"):
            logger.warning(
                f"Grid '{data['name']}' had constraints that could not be restored"
            )
        if data.get("has_exclusions"):
            logger.warning(
                f"Grid '{data['name']}' had exclusion patterns that could not be restored"
            )

        return grid

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

        # Check for duplicate keys across different parameter sources
        all_keys = list(self.parameters.keys())
        seen_keys = set(all_keys)

        # Check for duplicates in traditional parameters
        if len(all_keys) != len(seen_keys):
            issues.append(f"Grid '{self.name}' has duplicate parameter keys")

        # Check specs for duplicate keys
        for spec in self.parameter_specs:
            for key in spec.keys:
                if key in seen_keys:
                    issues.append(
                        f"Grid '{self.name}': Parameter '{key}' defined in multiple places"
                    )
                seen_keys.add(key)

        # Warn about overlapping conditional parameters
        conditional_keys = {}
        for spec in self.parameter_specs:
            if spec.param_type == ParameterType.CONDITIONAL:
                for key in spec.keys:
                    if key in conditional_keys:
                        issues.append(
                            f"Grid '{self.name}': Multiple conditional specs target key '{key}'"
                        )
                    conditional_keys[key] = True

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
