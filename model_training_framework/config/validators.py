"""
Configuration Validation

This module provides comprehensive validation for experiment configurations:
- Schema validation against dataclass definitions
- Resource requirement validation for SLURM jobs
- Parameter compatibility checking
- Configuration completeness verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .schemas import (
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        OptimizerConfig,
        SchedulerConfig,
        SLURMConfig,
        TrainingConfig,
    )

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue."""

    level: str  # "error", "warning", "info"
    component: str  # Component where issue was found
    message: str
    field: str | None = None
    suggestion: str | None = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.level == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.level == "warning" for issue in self.issues + self.warnings)

    def get_errors(self) -> list[ValidationIssue]:
        """Get all error-level issues."""
        return [issue for issue in self.issues if issue.level == "error"]

    def get_warnings(self) -> list[ValidationIssue]:
        """Get all warning-level issues."""
        return [
            issue for issue in self.issues if issue.level == "warning"
        ] + self.warnings

    def add_error(
        self, component: str, message: str, field: str | None = None, suggestion: str | None = None
    ) -> None:
        """Add an error-level issue."""
        self.issues.append(
            ValidationIssue("error", component, message, field, suggestion)
        )
        self.is_valid = False

    def add_warning(
        self, component: str, message: str, field: str | None = None, suggestion: str | None = None
    ) -> None:
        """Add a warning-level issue."""
        self.warnings.append(
            ValidationIssue("warning", component, message, field, suggestion)
        )

    def add_info(self, component: str, message: str, field: str | None = None) -> None:
        """Add an info-level issue."""
        self.issues.append(ValidationIssue("info", component, message, field))


@dataclass
class ResourceCheck:
    """Result of resource requirement validation."""

    is_feasible: bool
    estimated_memory_gb: float
    estimated_time_hours: float
    recommended_partition: str | None = None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ConfigValidator:
    """Validates experiment configurations against schema and resource constraints."""

    # Constants for validation limits
    MAX_EXPERIMENT_NAME_LENGTH = 200
    MAX_EFFECTIVE_BATCH_SIZE = 10000
    PARTITION_LONG_HOURS = 72
    PARTITION_HIGH_MEM_GB = 500
    PARTITION_HIGH_MEM_GPUS = 4
    PARTITION_STANDARD_HOURS = 24

    # Constants for time parsing
    TIME_PARTS_HMS = 3  # Hours:Minutes:Seconds
    TIME_PARTS_HM = 2   # Hours:Minutes
    MINUTES_PER_HOUR = 60
    SECONDS_PER_HOUR = 3600
    HOURS_PER_DAY = 24

    # Constants for training time estimation
    BASE_TIME_PER_EPOCH = 0.5  # 30 minutes per epoch

    # Known model types and their approximate memory requirements (GB)
    MODEL_MEMORY_ESTIMATES: ClassVar[dict[str, float]] = {
        "resnet18": 0.5,
        "resnet50": 1.0,
        "resnet101": 2.0,
        "bert-base": 1.5,
        "bert-large": 3.0,
        "gpt2": 2.0,
        "gpt2-medium": 4.0,
        "gpt2-large": 6.0,
        "t5-small": 1.0,
        "t5-base": 2.0,
        "t5-large": 3.0,
    }

    # Valid optimizer types
    VALID_OPTIMIZERS: ClassVar[set[str]] = {
        "adam",
        "adamw",
        "sgd",
        "rmsprop",
        "adagrad",
        "adadelta",
        "adamax",
    }

    # Valid scheduler types
    VALID_SCHEDULERS: ClassVar[set[str]] = {
        "step",
        "multistep",
        "exponential",
        "cosine",
        "reduce_on_plateau",
        "cyclic",
        "one_cycle",
        "cosine_annealing_warm_restarts",
    }

    # Valid activation functions
    VALID_ACTIVATIONS: ClassVar[set[str]] = {
        "relu",
        "gelu",
        "swish",
        "tanh",
        "sigmoid",
        "leaky_relu",
        "elu",
        "selu",
    }

    @staticmethod
    def validate_config(config: ExperimentConfig) -> ValidationResult:
        """Comprehensive configuration validation."""
        result = ValidationResult(is_valid=True)

        # Validate basic required fields
        ConfigValidator._validate_basic_fields(config, result)

        # Validate model configuration
        ConfigValidator._validate_model_config(config.model, result)

        # Validate training configuration
        ConfigValidator._validate_training_config(config.training, result)

        # Validate optimizer configuration
        ConfigValidator._validate_optimizer_config(config.optimizer, result)

        # Validate scheduler configuration
        if config.scheduler:
            ConfigValidator._validate_scheduler_config(config.scheduler, result)

        # Validate SLURM configuration
        if config.slurm:
            ConfigValidator._validate_slurm_config(config.slurm, result)

        # Validate parameter compatibility
        ConfigValidator._validate_parameter_compatibility(config, result)

        return result

    @staticmethod
    def _validate_basic_fields(
        config: ExperimentConfig, result: ValidationResult
    ) -> None:
        """Validate basic required fields."""
        if not config.experiment_name:
            result.add_error("basic", "Experiment name is required")
        elif len(config.experiment_name) > ConfigValidator.MAX_EXPERIMENT_NAME_LENGTH:
            result.add_error("basic", f"Experiment name too long (max {ConfigValidator.MAX_EXPERIMENT_NAME_LENGTH} characters)")

        if not config.data.dataset_name:
            result.add_error("basic", "Dataset name is required", "data.dataset_name")

        # Validate seed if provided
        if config.seed is not None and (config.seed < 0 or config.seed > 2**32 - 1):
            result.add_error("basic", "Seed must be between 0 and 2^32-1", "seed")

    @staticmethod
    def _validate_model_config(model: ModelConfig, result: ValidationResult) -> None:
        """Validate model configuration."""
        if not model.type:
            result.add_error("model", "Model type is required", "model.type")

        # Validate dimensions
        if model.hidden_size <= 0:
            result.add_error(
                "model", "Hidden size must be positive", "model.hidden_size"
            )

        if model.num_layers <= 0:
            result.add_error(
                "model", "Number of layers must be positive", "model.num_layers"
            )

        # Validate dropout
        if not 0.0 <= model.dropout <= 1.0:
            result.add_error(
                "model", "Dropout must be between 0.0 and 1.0", "model.dropout"
            )

        # Validate activation
        if model.activation not in ConfigValidator.VALID_ACTIVATIONS:
            result.add_warning(
                "model",
                f"Unknown activation function: {model.activation}",
                "model.activation",
                f"Consider using one of: {', '.join(ConfigValidator.VALID_ACTIVATIONS)}",
            )

        # Validate num_classes if provided
        if model.num_classes is not None and model.num_classes <= 0:
            result.add_error(
                "model", "Number of classes must be positive", "model.num_classes"
            )

    @staticmethod
    def _validate_training_config(
        training: TrainingConfig, result: ValidationResult
    ) -> None:
        """Validate training configuration."""
        # Validate epochs and steps
        if training.max_epochs <= 0:
            result.add_error(
                "training", "Max epochs must be positive", "training.max_epochs"
            )

        if training.max_steps is not None and training.max_steps <= 0:
            result.add_error(
                "training", "Max steps must be positive", "training.max_steps"
            )

        # Validate gradient accumulation
        if training.gradient_accumulation_steps <= 0:
            result.add_error(
                "training",
                "Gradient accumulation steps must be positive",
                "training.gradient_accumulation_steps",
            )

        # Validate gradient clipping
        if training.max_grad_norm is not None and training.max_grad_norm <= 0:
            result.add_error(
                "training",
                "Max gradient norm must be positive",
                "training.max_grad_norm",
            )

        # Validate early stopping
        if training.early_stopping_patience is not None:
            if training.early_stopping_patience <= 0:
                result.add_error(
                    "training",
                    "Early stopping patience must be positive",
                    "training.early_stopping_patience",
                )

            if training.early_stopping_mode not in ["min", "max"]:
                result.add_error(
                    "training",
                    "Early stopping mode must be 'min' or 'max'",
                    "training.early_stopping_mode",
                )

        # Validate frequencies
        if training.validation_frequency <= 0:
            result.add_error(
                "training",
                "Validation frequency must be positive",
                "training.validation_frequency",
            )

        if training.log_frequency <= 0:
            result.add_error(
                "training", "Log frequency must be positive", "training.log_frequency"
            )

    @staticmethod
    def _validate_optimizer_config(
        optimizer: OptimizerConfig, result: ValidationResult
    ) -> None:
        """Validate optimizer configuration."""
        if optimizer.type not in ConfigValidator.VALID_OPTIMIZERS:
            result.add_warning(
                "optimizer",
                f"Unknown optimizer type: {optimizer.type}",
                "optimizer.type",
                f"Consider using one of: {', '.join(ConfigValidator.VALID_OPTIMIZERS)}",
            )

        # Validate learning rate
        if optimizer.lr <= 0:
            result.add_error(
                "optimizer", "Learning rate must be positive", "optimizer.lr"
            )
        elif optimizer.lr > 1.0:
            result.add_warning(
                "optimizer",
                f"Learning rate {optimizer.lr} is very high",
                "optimizer.lr",
                "Consider using a smaller learning rate (typically < 0.1)",
            )

        # Validate weight decay
        if optimizer.weight_decay < 0:
            result.add_error(
                "optimizer",
                "Weight decay must be non-negative",
                "optimizer.weight_decay",
            )

        # Validate beta parameters for Adam-family optimizers
        if optimizer.type in ["adam", "adamw"]:
            beta1, beta2 = optimizer.betas
            if not 0.0 <= beta1 < 1.0:
                result.add_error(
                    "optimizer", "Beta1 must be in [0, 1)", "optimizer.betas"
                )
            if not 0.0 <= beta2 < 1.0:
                result.add_error(
                    "optimizer", "Beta2 must be in [0, 1)", "optimizer.betas"
                )

        # Validate epsilon
        if optimizer.eps <= 0:
            result.add_error("optimizer", "Epsilon must be positive", "optimizer.eps")

    @staticmethod
    def _validate_scheduler_config(
        scheduler: SchedulerConfig, result: ValidationResult
    ) -> None:
        """Validate scheduler configuration."""
        if scheduler.type not in ConfigValidator.VALID_SCHEDULERS:
            result.add_warning(
                "scheduler",
                f"Unknown scheduler type: {scheduler.type}",
                "scheduler.type",
                f"Consider using one of: {', '.join(ConfigValidator.VALID_SCHEDULERS)}",
            )

        # Validate warmup steps
        if scheduler.warmup_steps < 0:
            result.add_error(
                "scheduler",
                "Warmup steps must be non-negative",
                "scheduler.warmup_steps",
            )

        # Validate min learning rate
        if scheduler.min_lr < 0:
            result.add_error(
                "scheduler",
                "Minimum learning rate must be non-negative",
                "scheduler.min_lr",
            )

        # Scheduler-specific validations
        if scheduler.type == "step":
            if scheduler.step_size <= 0:
                result.add_error(
                    "scheduler",
                    "Step size must be positive for step scheduler",
                    "scheduler.step_size",
                )
            if not 0.0 < scheduler.gamma <= 1.0:
                result.add_error(
                    "scheduler",
                    "Gamma must be in (0, 1] for step scheduler",
                    "scheduler.gamma",
                )

        elif scheduler.type == "multistep":
            if not scheduler.milestones:
                result.add_error(
                    "scheduler",
                    "Milestones required for multistep scheduler",
                    "scheduler.milestones",
                )
            elif len(scheduler.milestones) != len(set(scheduler.milestones)):
                result.add_error(
                    "scheduler", "Milestones must be unique", "scheduler.milestones"
                )
            elif scheduler.milestones != sorted(scheduler.milestones):
                result.add_error(
                    "scheduler", "Milestones must be sorted", "scheduler.milestones"
                )

    @staticmethod
    def _validate_slurm_config(slurm: SLURMConfig, result: ValidationResult) -> None:
        """Validate SLURM configuration."""
        # Validate resource requirements
        if slurm.nodes <= 0:
            result.add_error("slurm", "Number of nodes must be positive", "slurm.nodes")

        if slurm.ntasks_per_node <= 0:
            result.add_error(
                "slurm", "Tasks per node must be positive", "slurm.ntasks_per_node"
            )

        if slurm.gpus_per_node < 0:
            result.add_error(
                "slurm", "GPUs per node must be non-negative", "slurm.gpus_per_node"
            )

        if slurm.cpus_per_task <= 0:
            result.add_error(
                "slurm", "CPUs per task must be positive", "slurm.cpus_per_task"
            )

        # Validate memory format
        if not slurm.mem.endswith(("G", "M", "K")):
            result.add_error(
                "slurm",
                "Memory must end with G, M, or K",
                "slurm.mem",
                "Example: '256G', '4096M', '1048576K'",
            )

        # Validate time format (basic check)
        if not any(char in slurm.time for char in ["-", ":"]):
            result.add_warning(
                "slurm",
                "Time format should be HH:MM:SS or D-HH:MM:SS",
                "slurm.time",
                "Example: '1-00:00:00' for 1 day",
            )

        # Validate partition and account
        if not slurm.partition:
            result.add_error("slurm", "Partition is required", "slurm.partition")

        if not slurm.account:
            result.add_error("slurm", "Account is required", "slurm.account")

    @staticmethod
    def _validate_parameter_compatibility(
        config: ExperimentConfig, result: ValidationResult
    ) -> None:
        """Validate parameter compatibility across components."""
        # Check batch size vs gradient accumulation
        effective_batch_size = (
            config.data.batch_size * config.training.gradient_accumulation_steps
        )
        if config.slurm:
            effective_batch_size *= config.slurm.nodes * config.slurm.ntasks_per_node

        if effective_batch_size > ConfigValidator.MAX_EFFECTIVE_BATCH_SIZE:
            result.add_warning(
                "compatibility",
                f"Very large effective batch size: {effective_batch_size}",
                suggestion="Consider reducing batch_size or gradient_accumulation_steps",
            )

        # Check scheduler warmup vs total steps
        if config.scheduler and config.scheduler.warmup_steps > 0:
            if config.training.max_steps:
                if config.scheduler.warmup_steps >= config.training.max_steps:
                    result.add_error(
                        "compatibility",
                        "Warmup steps >= max training steps",
                        suggestion="Reduce warmup_steps or increase max_steps",
                    )
            else:
                # Estimate total steps from epochs
                estimated_steps = config.training.max_epochs * 1000  # Rough estimate
                if config.scheduler.warmup_steps > estimated_steps * 0.1:
                    result.add_warning(
                        "compatibility",
                        "Warmup steps may be too large relative to training length",
                        suggestion="Consider reducing warmup_steps",
                    )

        # Check early stopping vs validation frequency
        if (
            config.training.early_stopping_patience
            and config.training.early_stopping_patience
            < config.training.validation_frequency
        ):
            result.add_warning(
                "compatibility",
                "Early stopping patience < validation frequency",
                suggestion="Early stopping may trigger before seeing enough validation results",
            )

    @staticmethod
    def check_resource_requirements(config: ExperimentConfig) -> ResourceCheck:
        """Validate SLURM resource requirements and provide recommendations."""
        check = ResourceCheck(
            is_feasible=True, estimated_memory_gb=0.0, estimated_time_hours=0.0
        )

        if not config.slurm:
            check.issues.append("No SLURM configuration provided")
            check.is_feasible = False
            return check

        # Estimate memory requirements
        model_memory = ConfigValidator._estimate_model_memory(config.model)
        batch_memory = ConfigValidator._estimate_batch_memory(config.data, config.model)
        overhead_memory = 2.0  # OS and framework overhead

        check.estimated_memory_gb = model_memory + batch_memory + overhead_memory

        # Parse requested memory
        requested_memory = ConfigValidator._parse_memory_string(config.slurm.mem)

        if check.estimated_memory_gb > requested_memory:
            check.is_feasible = False
            check.issues.append(
                f"Estimated memory ({check.estimated_memory_gb:.1f}GB) > "
                f"requested memory ({requested_memory:.1f}GB)"
            )
        elif check.estimated_memory_gb > requested_memory * 0.8:
            check.warnings.append(
                f"Memory usage may be tight: {check.estimated_memory_gb:.1f}GB / "
                f"{requested_memory:.1f}GB requested"
            )

        # Estimate training time
        check.estimated_time_hours = ConfigValidator._estimate_training_time(config)

        # Parse requested time
        requested_time_hours = ConfigValidator._parse_time_string(config.slurm.time)

        if check.estimated_time_hours > requested_time_hours:
            check.warnings.append(
                f"Estimated time ({check.estimated_time_hours:.1f}h) > "
                f"requested time ({requested_time_hours:.1f}h)"
            )

        # Recommend partition based on requirements
        check.recommended_partition = ConfigValidator._recommend_partition(
            check.estimated_memory_gb,
            check.estimated_time_hours,
            config.slurm.gpus_per_node,
        )

        return check

    @staticmethod
    def _estimate_model_memory(model: ModelConfig) -> float:
        """Estimate model memory requirements in GB."""
        base_memory = ConfigValidator.MODEL_MEMORY_ESTIMATES.get(model.type, 2.0)

        # Scale by model size parameters
        size_factor = (model.hidden_size / 512) * (model.num_layers / 6)

        return base_memory * size_factor

    @staticmethod
    def _estimate_batch_memory(data: DataConfig, model: ModelConfig) -> float:
        """Estimate batch memory requirements in GB."""
        # Very rough estimate based on batch size and model complexity
        base_per_sample = 0.01  # 10MB per sample
        complexity_factor = model.hidden_size / 512

        return data.batch_size * base_per_sample * complexity_factor

    @staticmethod
    def _parse_memory_string(mem_str: str) -> float:
        """Parse memory string to GB."""
        if mem_str.endswith("G"):
            return float(mem_str[:-1])
        if mem_str.endswith("M"):
            return float(mem_str[:-1]) / 1024
        if mem_str.endswith("K"):
            return float(mem_str[:-1]) / (1024 * 1024)
        return float(mem_str) / (1024 * 1024 * 1024)  # Assume bytes

    @staticmethod
    def _parse_time_string(time_str: str) -> float:
        """Parse time string to hours."""
        # Handle format like "1-00:00:00" or "24:00:00"
        if "-" in time_str:
            days, time_part = time_str.split("-")
            hours_from_days = int(days) * ConfigValidator.HOURS_PER_DAY
        else:
            hours_from_days = 0
            time_part = time_str

        time_parts = time_part.split(":")
        if len(time_parts) == ConfigValidator.TIME_PARTS_HMS:
            hours, minutes, seconds = map(int, time_parts)
            return hours_from_days + hours + minutes / ConfigValidator.MINUTES_PER_HOUR + seconds / ConfigValidator.SECONDS_PER_HOUR
        if len(time_parts) == ConfigValidator.TIME_PARTS_HM:
            hours, minutes = map(int, time_parts)
            return hours_from_days + hours + minutes / ConfigValidator.MINUTES_PER_HOUR
        return hours_from_days + int(time_parts[0])

    @staticmethod
    def _estimate_training_time(config: ExperimentConfig) -> float:
        """Estimate training time in hours."""
        # Very rough estimate - in practice this would need more sophisticated modeling
        base_time_per_epoch = ConfigValidator.BASE_TIME_PER_EPOCH

        # Scale by model complexity
        complexity_factor = (config.model.hidden_size / 512) * (
            config.model.num_layers / 6
        )

        # Scale by batch size (inversely)
        batch_factor = 64 / config.data.batch_size

        return (
            config.training.max_epochs
            * base_time_per_epoch
            * complexity_factor
            * batch_factor
        )

    @staticmethod
    def _recommend_partition(memory_gb: float, time_hours: float, gpus: int) -> str:
        """Recommend SLURM partition based on requirements."""
        if time_hours > ConfigValidator.PARTITION_LONG_HOURS:
            return "gpu-long"
        if memory_gb > ConfigValidator.PARTITION_HIGH_MEM_GB or gpus > ConfigValidator.PARTITION_HIGH_MEM_GPUS:
            return "gpu-high-mem"
        if time_hours > ConfigValidator.PARTITION_STANDARD_HOURS:
            return "gpu"
        return "ckpt-all"
