"""
Configuration schemas for the model training framework.

This module defines all configuration dataclasses, enums, and validation schemas
used throughout the framework. It provides a hierarchical configuration system
that supports composition, validation, and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


class ExecutionMode(Enum):
    """Execution modes for experiments."""

    LOCAL = "local"
    SLURM = "slurm"
    DRY_RUN = "dry_run"


class NamingStrategy(Enum):
    """Strategies for generating experiment names."""

    HASH_BASED = "hash_based"
    PARAMETER_BASED = "parameter_based"
    TIMESTAMP_BASED = "timestamp_based"


@runtime_checkable
class ConfigProtocol(Protocol):
    """Protocol defining the interface for configuration objects."""

    def validate(self) -> list[str]:
        """Validate the configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        ...


@dataclass
class ModelConfig:
    """Flexible model architecture configuration.

    Supports both typed fields and dynamic attributes for maximum flexibility.
    Users can either use this class directly or subclass it for typed configs.
    """

    type: str = "custom"
    # Store extra fields that aren't defined as dataclass fields
    _extra_fields: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from dictionary.

        This method enables compatibility with grid search and config manager.
        """
        # Work with a copy to avoid mutating the input
        data_copy = data.copy()
        # Extract known fields
        type_val = data_copy.pop("type", "custom")
        # Create instance with known fields
        instance = cls(type=type_val)
        # Store remaining fields as extra
        instance._extra_fields = data_copy
        # Also set as attributes for backward compatibility
        for key, value in data_copy.items():
            setattr(instance, key, value)
        return instance

    def __init__(self, type: str = "custom", **kwargs: Any):
        """Initialize with type and arbitrary parameters.

        Args:
            type: Model type identifier
            **kwargs: Any additional model parameters
        """
        self.type = type
        self._extra_fields = kwargs
        # Set extra fields as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """Access extra fields as attributes."""
        if "_extra_fields" in self.__dict__ and name in self._extra_fields:
            return self._extra_fields[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attributes, storing extras in _extra_fields."""
        # Handle dataclass fields normally
        if name in ["type", "_extra_fields"] or name.startswith("_"):
            super().__setattr__(name, value)
        else:
            # Store in extra_fields if it exists
            if hasattr(self, "_extra_fields") and name not in ["type"]:
                self._extra_fields[name] = value
            super().__setattr__(name, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"type": self.type}
        result.update(self._extra_fields)
        return result

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        if not self.type:
            errors.append("Model type is required")
        return errors

    def __str__(self) -> str:
        """String representation matching dataclass style."""
        # Get all attributes including dynamic ones
        attrs = self.to_dict()
        attr_str = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{self.__class__.__name__}({attr_str})"

    def __repr__(self) -> str:
        """Representation string matching dataclass style."""
        return self.__str__()


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    type: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    type: str = "cosine"
    warmup_steps: int = 1000
    max_steps: int | None = None
    min_lr: float = 1e-6
    gamma: float = 0.1
    step_size: int = 10
    milestones: list[int] = field(default_factory=list)
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Flexible dataset configuration.

    Supports both typed fields and dynamic attributes for maximum flexibility.
    Users can either use this class directly or subclass it for typed configs.
    """

    dataset_name: str = "custom"
    batch_size: int = 32  # Common field for compatibility
    # Store extra fields that aren't defined as dataclass fields
    _extra_fields: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        """Create DataConfig from dictionary.

        This method enables compatibility with grid search and config manager.
        """
        # Work with a copy to avoid mutating the input
        data_copy = data.copy()
        # Extract known fields
        dataset_name = data_copy.pop("dataset_name", "custom")
        batch_size = data_copy.pop("batch_size", 32)
        # Create instance with known fields
        instance = cls(dataset_name=dataset_name, batch_size=batch_size)
        # Store remaining fields as extra
        instance._extra_fields = data_copy
        # Also set as attributes for backward compatibility
        for key, value in data_copy.items():
            setattr(instance, key, value)
        return instance

    def __init__(
        self, dataset_name: str = "custom", batch_size: int = 32, **kwargs: Any
    ):
        """Initialize with dataset name and arbitrary parameters.

        Args:
            dataset_name: Dataset identifier
            batch_size: Batch size for training
            **kwargs: Any additional data parameters
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self._extra_fields = kwargs
        # Set extra fields as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """Access extra fields as attributes."""
        if "_extra_fields" in self.__dict__ and name in self._extra_fields:
            return self._extra_fields[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attributes, storing extras in _extra_fields."""
        # Handle dataclass fields normally
        if name in ["dataset_name", "batch_size", "_extra_fields"] or name.startswith(
            "_"
        ):
            super().__setattr__(name, value)
        else:
            # Store in extra_fields if it exists
            if hasattr(self, "_extra_fields") and name not in [
                "dataset_name",
                "batch_size",
            ]:
                self._extra_fields[name] = value
            super().__setattr__(name, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"dataset_name": self.dataset_name, "batch_size": self.batch_size}
        result.update(self._extra_fields)
        return result

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        if not self.dataset_name:
            errors.append("Dataset name is required")
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        return errors

    def __str__(self) -> str:
        """String representation matching dataclass style."""
        # Get all attributes including dynamic ones
        attrs = self.to_dict()
        attr_str = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{self.__class__.__name__}({attr_str})"

    def __repr__(self) -> str:
        """Representation string matching dataclass style."""
        return self.__str__()


@dataclass
class TrainingConfig:
    """Training process configuration."""

    max_epochs: int = 100
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    use_amp: bool = True
    early_stopping_patience: int | None = None
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    validation_frequency: int = 1
    log_frequency: int = 100
    save_frequency: int = 1000


@dataclass
class LoggingConfig:
    """Experiment tracking and logging configuration."""

    use_wandb: bool = True
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str | None = None
    wandb_name: str | None = None
    wandb_mode: str | None = None
    wandb_id: str | None = None
    wandb_resume: str | None = None
    log_scalars_every_n_steps: int | None = 50
    log_images_every_n_steps: int | None = 500
    log_gradients: bool = False
    log_model_parameters: bool = False
    log_system_metrics: bool = True
    tensorboard_dir: str | None = None
    csv_log_dir: str | None = None


@dataclass
class SLURMConfig:
    """SLURM job configuration."""

    account: str = "realitylab"
    partition: str = "ckpt-all"
    nodes: int = 1
    ntasks_per_node: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    mem: str = "256G"
    time: str = "1-00:00:00"
    constraint: str | None = "a40|a100"
    requeue: bool = True
    job_name: str | None = None
    output: str | None = None
    error: str | None = None
    mail_type: str | None = None
    mail_user: str | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Checkpoint configuration from trainer module."""

    root_dir: str | Path = "checkpoints"
    save_every_n_steps: int | None = None
    save_every_n_epochs: int | None = 1
    max_checkpoints: int = 5
    save_rng: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True
    filename_template: str = "epoch_{epoch:03d}_step_{step:06d}.ckpt"
    monitor_metric: str | None = None
    monitor_mode: str = "min"


@dataclass
class PreemptionConfig:
    """Preemption handling configuration from trainer module."""

    signal: int = 10  # SIGUSR1
    max_checkpoint_sec: float = 300.0  # 5 minutes
    requeue_job: bool = True
    resume_from_latest_symlink: bool = True
    cleanup_on_exit: bool = True
    backup_checkpoints: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration from trainer module."""

    gradient_accumulation_steps: int = 1
    compile_model: bool = False
    use_amp: bool = True
    clip_grad_norm: float | None = 1.0
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass
class ExperimentConfig:
    """Master experiment configuration schema."""

    experiment_name: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None = None
    slurm: SLURMConfig | None = None
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Metadata
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    created_by: str | None = None
    created_at: str | None = None
    version: str = "1.0"

    # Runtime settings
    seed: int | None = None
    deterministic: bool = True
    benchmark: bool = False

    # Custom parameters for extensibility
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GridSearchConfig:
    """Configuration for parameter grid search."""

    name: str
    description: str = ""
    base_config: dict[str, Any] = field(default_factory=dict)
    parameter_grids: list[dict[str, Any]] = field(default_factory=list)
    naming_strategy: NamingStrategy = NamingStrategy.HASH_BASED
    max_concurrent_jobs: int | None = None
    execution_mode: ExecutionMode = ExecutionMode.SLURM
    output_dir: str | None = None


@dataclass
class ResourceRequirements:
    """Resource requirements for validation."""

    min_memory_gb: float = 1.0
    min_cpu_cores: int = 1
    min_gpu_memory_gb: float = 0.0
    max_time_hours: float = 24.0
    required_partitions: list[str] = field(default_factory=list)
    required_constraints: list[str] = field(default_factory=list)
