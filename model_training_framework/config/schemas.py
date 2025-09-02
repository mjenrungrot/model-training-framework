"""
Configuration schemas for the model training framework.

This module defines all configuration dataclasses, enums, and validation schemas
used throughout the framework. It provides a hierarchical configuration system
that supports composition, validation, and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import signal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
import warnings

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


# Multi-dataloader enums
class SamplingStrategy(Enum):
    """Strategy for sampling from multiple dataloaders."""

    SEQUENTIAL = "sequential"  # Process loaders one after another
    ROUND_ROBIN = "round_robin"  # Alternate between loaders
    WEIGHTED = "weighted"  # Sample based on weights
    ALTERNATING = "alternating"  # Follow explicit pattern


class EpochLengthPolicy(Enum):
    """Policy for determining epoch length with multiple dataloaders."""

    SUM_OF_LENGTHS = "sum_of_lengths"  # Sum of all loader lengths
    MAX_OF_LENGTHS = "max_of_lengths"  # Maximum loader length
    MIN_OF_LENGTHS = "min_of_lengths"  # Minimum loader length
    FIXED_NUM_STEPS = "fixed_num_steps"  # Fixed number of steps


class ValidationFrequency(Enum):
    """Frequency for running validation."""

    PER_EPOCH = "per_epoch"  # Validate at epoch boundaries
    EVERY_N_STEPS = "every_n_steps"  # Validate every N training steps


class ValAggregation(Enum):
    """Aggregation strategy for validation metrics across dataloaders."""

    MICRO_AVG_WEIGHTED_BY_SAMPLES = "micro_avg_weighted_by_samples"  # Weight by samples
    MACRO_AVG_EQUAL_LOADERS = "macro_avg_equal_loaders"  # Equal weight per loader
    PRIMARY_METRIC_PER_LOADER = "primary_metric_per_loader"  # Track per-loader metrics
    CUSTOM = "custom"  # User-defined aggregation


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
    """
    Configuration for experiment tracking and logging.

    Controls integration with various logging backends including Weights & Biases,
    TensorBoard, CSV files, and console output.
    """

    # Logger type selection (default to console; opt-in for external services)
    logger_type: str = (
        "console"  # Options: "console", "wandb", "tensorboard", "composite"
    )

    # Legacy W&B configuration (kept for backward compatibility)
    use_wandb: bool = False  # Whether to use Weights & Biases for experiment tracking
    wandb_project: str | None = None  # W&B project name (uses experiment name if None)
    wandb_entity: str | None = None  # W&B entity/team name
    wandb_tags: list[str] = field(default_factory=list)  # Tags for W&B experiment
    wandb_notes: str | None = None  # Notes for W&B experiment
    wandb_name: str | None = None  # Run name for W&B
    wandb_mode: str | None = None  # W&B mode (online, offline, disabled)
    wandb_id: str | None = None  # Unique run ID for W&B
    wandb_resume: str | None = None  # W&B resume strategy

    # TensorBoard configuration
    use_tensorboard: bool = False  # Whether to use TensorBoard logging
    tensorboard_dir: str | None = None  # TensorBoard log directory

    # CSV logging
    use_csv: bool = True  # Whether to log metrics to CSV files
    csv_log_dir: str | None = None  # CSV log directory (uses checkpoint dir if None)

    # Console logging
    console_log_level: str = "INFO"  # Console logging level
    console_log_format: str = (
        "[%(asctime)s] %(levelname)s: %(message)s"  # Console log format
    )

    # Composite logger configuration
    composite_loggers: list[str] | None = (
        None  # Explicit logger list for composite logger
    )

    # Metrics configuration
    log_per_loader_metrics: bool = True  # Log metrics per dataloader
    log_global_metrics: bool = True  # Log globally aggregated metrics
    log_loader_proportions: bool = True  # Log loader usage proportions (for WEIGHTED)
    all_reduce_metrics: bool = (
        False  # Aggregate metrics across ranks before logging (DDP)
    )

    # Logging frequencies
    log_scalars_every_n_steps: int | None = (
        50  # How often to log scalar metrics (None = every step)
    )
    log_images_every_n_steps: int | None = 500  # How often to log images (None = never)
    log_gradients: bool = False  # Whether to log gradient statistics
    log_model_parameters: bool = False  # Whether to log model parameter statistics
    log_system_metrics: bool = True  # Whether to log system metrics (GPU, memory, etc.)

    def __post_init__(self) -> None:
        """Validate WandB-related fields for early, actionable errors."""
        if self.wandb_mode is not None:
            valid_modes = {"online", "offline", "disabled"}
            if self.wandb_mode not in valid_modes:
                raise ValueError(
                    f"wandb_mode must be one of {valid_modes}, got {self.wandb_mode!r}"
                )
        if self.wandb_resume is not None:
            valid_resume = {"allow", "must", "never"}
            if self.wandb_resume not in valid_resume:
                raise ValueError(
                    f"wandb_resume must be one of {valid_resume}, got {self.wandb_resume!r}"
                )


@dataclass
class MultiDataLoaderConfig:
    """
    Configuration for multi-dataloader training.

    Defines how to sample from multiple dataloaders, determine epoch length,
    and manage iteration state for fault-tolerant resume.
    """

    sampling_strategy: SamplingStrategy = SamplingStrategy.ROUND_ROBIN
    dataloader_weights: list[float] | None = None  # Weights for WEIGHTED strategy
    alternating_pattern: list[int] | None = None  # Pattern for ALTERNATING strategy
    dataloader_names: list[str] = field(default_factory=list)  # Names for logging
    epoch_length_policy: EpochLengthPolicy = EpochLengthPolicy.SUM_OF_LENGTHS
    steps_per_epoch: int | None = None  # For FIXED_NUM_STEPS policy
    cycle_short_loaders: bool = True  # Whether to cycle shorter loaders
    burst_size: int = 1  # Number of batches per loader per switch
    choice_rng_seed: int | None = None  # Seed for weighted sampling
    prefetch_cap_total_batches: int | None = None  # Max batches to prefetch


@dataclass
class ValidationConfig:
    """
    Configuration for validation during training.

    Controls when validation runs and how metrics are aggregated
    across multiple validation dataloaders.
    """

    frequency: ValidationFrequency = ValidationFrequency.PER_EPOCH
    every_n_steps: int | None = None  # For EVERY_N_STEPS frequency
    aggregation: ValAggregation = ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES
    per_loader_metrics: bool = True  # Log per-loader metrics
    global_metrics: bool = True  # Log aggregated metrics
    early_stopping_source: str = "both"  # Options: "both", "hook", "legacy"

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_sources = {"both", "hook", "legacy"}
        if self.early_stopping_source not in valid_sources:
            raise ValueError(
                f"early_stopping_source must be one of {valid_sources}, "
                f"got '{self.early_stopping_source}'"
            )


@dataclass
class FaultToleranceConfig:
    """
    Configuration for fault-tolerant training.

    Enables deterministic resume after preemption by saving
    dataloader iteration state.
    """

    save_sampler_state: bool = True  # Save sampler state for resume
    save_dataset_state: bool = True  # Save dataset iteration state
    verify_deterministic_resume: bool = False  # Verify resume produces same batches
    checkpoint_timeout_sec: float = 300.0  # Max time for checkpoint save


@dataclass
class DDPConfig:
    """
    Configuration for Distributed Data Parallel training.

    Settings for DDP/Fabric distributed training synchronization.
    """

    backend: str = "nccl"  # DDP backend (nccl, gloo, mpi)
    init_method: str | None = None  # Process group init method
    find_unused_parameters: bool = False  # Find unused parameters in model
    gradient_as_bucket_view: bool = True  # Use gradient bucket view optimization

    # Schedule synchronization across ranks
    sync_schedules_across_ranks: bool = (
        True  # Ensure identical schedules across processes
    )
    validate_schedule_consistency: bool = (
        False  # Runtime validation of schedule consistency
    )

    # Additional DDP optimizations
    broadcast_buffers: bool = (
        True  # Broadcast buffers (e.g., BatchNorm stats) at forward
    )
    bucket_cap_mb: int = (
        25  # Gradient bucketing size in MB for communication efficiency
    )


@dataclass
class HooksConfig:
    """
    Configuration for training hooks system.

    Allows injection of custom behavior at various training lifecycle points.
    """

    # Hook class paths to load (e.g., ["mypackage.hooks.CustomHook"])
    hook_classes: list[str] = field(default_factory=list)

    # Hook-specific configuration passed to hook constructors
    hook_configs: dict[str, Any] = field(default_factory=dict)

    # Built-in hooks
    enable_logging_hook: bool = False  # Enable detailed logging hook
    enable_gradient_monitor: bool = False  # Enable gradient monitoring hook
    enable_model_checkpoint_hook: bool = False  # Enable model checkpoint tracking
    enable_early_stopping_hook: bool = False  # Enable early stopping hook

    # Built-in hook configurations
    early_stopping_config: dict[str, Any] = field(
        default_factory=lambda: {
            "monitor": "val/loss",
            "patience": 10,
            "mode": "min",
            "min_delta": 0.0001,
        }
    )

    gradient_monitor_config: dict[str, Any] = field(
        default_factory=lambda: {
            "log_frequency": 100,
            "param_filter": None,  # Optional list of param names to monitor
        }
    )

    model_checkpoint_config: dict[str, Any] = field(
        default_factory=lambda: {
            "save_top_k": 3,
            "monitor": "val/loss",
        }
    )

    # Hook execution settings
    continue_on_hook_error: bool = True  # Continue training if hook raises error
    log_hook_errors: bool = True  # Log hook errors to console


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
    """
    Controls checkpoint creation timing and storage behavior.

    Checkpoints can be triggered by step count, epoch boundaries, time intervals,
    or manually. Old checkpoints are automatically cleaned up based on max_checkpoints.
    """

    root_dir: str | Path = "checkpoints"  # Base directory for all checkpoints
    save_every_n_steps: int | None = None  # Save checkpoint every N optimizer steps
    save_every_n_epochs: int | None = 1  # Save checkpoint every N epochs
    save_every_n_minutes: float | None = None  # Save checkpoint every N minutes
    max_checkpoints: int = (
        5  # Maximum number of checkpoints to keep (oldest deleted first)
    )
    save_rng: bool = True  # Whether to save RNG states for perfect reproducibility
    save_optimizer: bool = True  # Whether to save optimizer state
    save_scheduler: bool = True  # Whether to save scheduler state
    filename_template: str = (
        "epoch_{epoch:03d}_step_{step:06d}.ckpt"  # Checkpoint filename pattern
    )
    monitor_metric: str | None = (
        None  # Metric to monitor for best checkpoint (e.g., "val_loss")
    )
    monitor_mode: str = (
        "min"  # Whether to minimize or maximize the monitored metric ("min" or "max")
    )
    save_best: bool = (
        True  # Whether to save the best checkpoint based on monitored metric
    )


@dataclass
class PreemptionConfig:
    """
    Configuration for handling job preemption (SIGUSR1 signals).

    When a preemption signal is received, the trainer will attempt to save a checkpoint
    within the specified time limit and optionally requeue the job.
    """

    signal: int = signal.SIGUSR1  # Signal number to listen for (default: SIGUSR1 = 10)
    max_checkpoint_sec: float = (
        300.0  # Maximum time allowed for checkpoint save (5 minutes)
    )
    requeue_job: bool = True  # Whether to requeue the SLURM job after preemption
    resume_from_latest_symlink: bool = (
        True  # Whether to resume from latest.ckpt symlink
    )
    cleanup_on_exit: bool = True  # Whether to cleanup temporary files on exit
    backup_checkpoints: bool = True  # Whether to create backup copies of checkpoints


@dataclass
class WarmStartConfig:
    """
    Configuration for user-provided warm-start loader.

    When provided, `loader_class` should be an import path to a callable or
    class implementing a `__call__(trainer, checkpoint_path) -> WarmStartResult`.
    """

    loader_class: str | None = None  # Dotted import path to a callable/loader class
    strict: bool = True  # Default strictness if loader does not specify


@dataclass
class PerformanceConfig:
    """
    Configuration for performance optimization features.

    These settings control various PyTorch performance optimizations including
    gradient accumulation, mixed precision, model compilation, and data loading.
    """

    gradient_accumulation_steps: int = (
        1  # Number of steps to accumulate gradients before update
    )
    compile_model: bool = False  # Whether to use torch.compile() for model optimization
    use_amp: bool = True  # Whether to use Automatic Mixed Precision (AMP)
    clip_grad_norm: float | None = (
        1.0  # Maximum gradient norm for clipping (None to disable)
    )
    dataloader_num_workers: int = 4  # Number of worker processes for data loading
    pin_memory: bool = True  # Whether to pin memory for faster GPU transfer
    persistent_workers: bool = (
        True  # Whether to keep dataloader workers alive between epochs
    )
    prefetch_factor: int = 2  # Number of batches to prefetch per worker


@dataclass
class GenericTrainerConfig:
    """
    Master configuration for the GenericTrainer.

    This combines all sub-configurations needed for training including checkpointing,
    preemption handling, performance optimization, and logging.
    """

    experiment_name: str = "experiment"  # Name of the experiment for checkpointing
    seed: int | None = None  # Random seed for reproducibility
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)  # Hooks configuration

    # Multi-dataloader configuration
    # Split configuration for train and val phases
    train_loader_config: MultiDataLoaderConfig = field(
        default_factory=MultiDataLoaderConfig
    )
    val_loader_config: MultiDataLoaderConfig = field(
        default_factory=MultiDataLoaderConfig
    )
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)
    ddp: DDPConfig | None = None  # Optional DDP configuration
    warm_start: WarmStartConfig | None = None  # Optional warm-start configuration

    # Per-loader configuration
    loss_weights_per_loader: list[float] | None = None  # Loss weights per dataloader
    per_loader_optimizer_id: list[int] | None = None  # Optimizer ID per dataloader

    # Training loop behavior
    log_loss_every_n_steps: int | None = 100  # How often to log training loss
    validate_every_n_epochs: int = 1  # How often to run validation
    early_stopping_patience: int | None = (
        None  # Early stopping patience (None = disabled)
    )
    early_stopping_metric: str = "val_loss"  # Metric to monitor for early stopping
    early_stopping_mode: str = (
        "min"  # Whether to minimize or maximize the early stopping metric
    )

    # Debugging and development
    debug_mode: bool = False  # Enable debug mode with additional checks and logging
    profile_training: bool = True  # Enable profiling of training loop by default
    dry_run: bool = False  # Run training loop without actual training (for debugging)

    # Optional full experiment configuration for comprehensive logging
    experiment_config: ExperimentConfig | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate early stopping configuration
        if self.early_stopping_patience is not None:
            if self.early_stopping_patience <= 0:
                raise ValueError("early_stopping_patience must be positive")
            if self.early_stopping_mode not in ["min", "max"]:
                raise ValueError("early_stopping_mode must be 'min' or 'max'")

        # Validate gradient accumulation
        if self.performance.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")

        # Validate logging frequencies
        if self.log_loss_every_n_steps is not None and self.log_loss_every_n_steps <= 0:
            raise ValueError("log_loss_every_n_steps must be positive")

        if self.validate_every_n_epochs <= 0:
            raise ValueError("validate_every_n_epochs must be positive")

        # Validate checkpoint configuration
        if self.checkpoint.max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be positive")

        if (
            self.checkpoint.monitor_metric is not None
            and self.checkpoint.monitor_mode not in ["min", "max"]
        ):
            raise ValueError("checkpoint monitor_mode must be 'min' or 'max'")

        # Validate multi-dataloader configuration (train/val)
        validate_trainer_config(self)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the configuration."""
        return {
            "checkpoint": {
                "root_dir": str(self.checkpoint.root_dir),
                "save_every_n_steps": self.checkpoint.save_every_n_steps,
                "save_every_n_epochs": self.checkpoint.save_every_n_epochs,
                "max_checkpoints": self.checkpoint.max_checkpoints,
                "save_rng": self.checkpoint.save_rng,
            },
            "preemption": {
                "signal": self.preemption.signal,
                "max_checkpoint_sec": self.preemption.max_checkpoint_sec,
                "requeue_job": self.preemption.requeue_job,
            },
            "performance": {
                "gradient_accumulation_steps": self.performance.gradient_accumulation_steps,
                "use_amp": self.performance.use_amp,
                "compile_model": self.performance.compile_model,
                "clip_grad_norm": self.performance.clip_grad_norm,
            },
            "logging": {
                "logger_type": self.logging.logger_type,
                "use_wandb": self.logging.use_wandb,
                "wandb_project": self.logging.wandb_project,
                "use_tensorboard": self.logging.use_tensorboard,
                "use_csv": self.logging.use_csv,
                "log_scalars_every_n_steps": self.logging.log_scalars_every_n_steps,
            },
            "training": {
                "log_loss_every_n_steps": self.log_loss_every_n_steps,
                "validate_every_n_epochs": self.validate_every_n_epochs,
                "early_stopping_patience": self.early_stopping_patience,
            },
            "train_multi_dataloader": {
                "sampling_strategy": self.train_loader_config.sampling_strategy.value,
                "epoch_length_policy": self.train_loader_config.epoch_length_policy.value,
                "dataloader_names": self.train_loader_config.dataloader_names,
            },
            "val_multi_dataloader": {
                "sampling_strategy": self.val_loader_config.sampling_strategy.value,
                "epoch_length_policy": self.val_loader_config.epoch_length_policy.value,
                "dataloader_names": self.val_loader_config.dataloader_names,
            },
            "validation": {
                "frequency": self.validation.frequency.value,
                "aggregation": self.validation.aggregation.value,
            },
        }


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

    # Multi-dataloader and validation configurations
    train_loader_config: MultiDataLoaderConfig | None = None
    val_loader_config: MultiDataLoaderConfig | None = None
    validation: ValidationConfig | None = None
    fault_tolerance: FaultToleranceConfig | None = None
    ddp: DDPConfig | None = None
    hooks: HooksConfig | None = None

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
    naming_strategy: NamingStrategy = NamingStrategy.PARAMETER_BASED
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


# Validation helper functions from trainer/config.py
def validate_infinite_loader_constraints(
    epoch_length_policy: EpochLengthPolicy, assume_finite_loaders: bool = True
) -> None:
    """
    Validate epoch length policy constraints for infinite loaders.

    Args:
        epoch_length_policy: The epoch length policy to validate
        assume_finite_loaders: Whether to assume all loaders are finite (default True)

    Raises:
        ValueError: If policy cannot handle infinite loaders and assumption is False
    """
    if not assume_finite_loaders:
        incompatible_policies = {
            EpochLengthPolicy.SUM_OF_LENGTHS,
            EpochLengthPolicy.MIN_OF_LENGTHS,
        }
        if epoch_length_policy in incompatible_policies:
            warnings.warn(
                f"EpochLengthPolicy.{epoch_length_policy.name} cannot handle infinite "
                f"dataloaders. Runtime validation will enforce finite loader requirement.",
                UserWarning,
                stacklevel=2,
            )


def _validate_multi_config(
    multi: MultiDataLoaderConfig,
    *,
    assume_finite_loaders: bool = True,
    context: str = "train",
) -> None:
    """Validate a single MultiDataLoaderConfig for internal consistency."""
    # Validate dataloader weights
    if multi.sampling_strategy == SamplingStrategy.WEIGHTED:
        if multi.dataloader_weights is None:
            raise ValueError(
                f"dataloader_weights must be provided for WEIGHTED sampling strategy ({context})"
            )
        if any(w <= 0 for w in multi.dataloader_weights):
            raise ValueError("All dataloader weights must be positive")
        # Validate weight count matches dataloader count
        if multi.dataloader_names and len(multi.dataloader_weights) != len(
            multi.dataloader_names
        ):
            raise ValueError(
                f"Number of dataloader_weights ({len(multi.dataloader_weights)}) must match "
                f"number of dataloaders ({len(multi.dataloader_names)}) for {context}"
            )

    # Validate alternating pattern
    if multi.sampling_strategy == SamplingStrategy.ALTERNATING:
        if multi.alternating_pattern is None:
            raise ValueError(
                f"alternating_pattern must be provided for ALTERNATING sampling strategy ({context})"
            )
        if not multi.alternating_pattern:
            raise ValueError("alternating_pattern cannot be empty")
        # Validate indices are non-negative
        if any(idx < 0 for idx in multi.alternating_pattern):
            raise ValueError("alternating_pattern indices must be non-negative")
        # If dataloader names provided, validate indices are in range
        if multi.dataloader_names:
            max_idx = len(multi.dataloader_names) - 1
            if any(idx > max_idx for idx in multi.alternating_pattern):
                raise ValueError(
                    f"alternating_pattern indices must be in range [0, {max_idx}] for {context}"
                )

    # Validate fixed steps policy
    if multi.epoch_length_policy == EpochLengthPolicy.FIXED_NUM_STEPS:
        if multi.steps_per_epoch is None:
            raise ValueError(
                f"steps_per_epoch must be provided for FIXED_NUM_STEPS epoch length policy ({context})"
            )
        if multi.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")

    # Validate burst size
    if multi.burst_size <= 0:
        raise ValueError("burst_size must be positive")

    # Validate prefetch cap
    if multi.prefetch_cap_total_batches is not None and (
        multi.prefetch_cap_total_batches <= 0
    ):
        raise ValueError("prefetch_cap_total_batches must be positive")

    # Validate dataloader names uniqueness (warning only)
    if multi.dataloader_names:
        unique_names = set(multi.dataloader_names)
        if len(unique_names) != len(multi.dataloader_names):
            warnings.warn(
                f"dataloader_names contains duplicates in {context} config; this may affect logging clarity",
                UserWarning,
                stacklevel=2,
            )

    # Validate infinite loader constraints (warning for incompatible policies)
    validate_infinite_loader_constraints(
        multi.epoch_length_policy, assume_finite_loaders
    )


def validate_trainer_config(
    config: GenericTrainerConfig, assume_finite_loaders: bool = True
) -> None:
    """
    Validate trainer configuration for consistency.

    Checks multi-dataloader configuration constraints and raises
    ValueError for invalid configurations.

    Args:
        config: Trainer configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate the train and val multi configs separately
    _validate_multi_config(
        config.train_loader_config,
        assume_finite_loaders=assume_finite_loaders,
        context="train",
    )
    _validate_multi_config(
        config.val_loader_config,
        assume_finite_loaders=assume_finite_loaders,
        context="val",
    )

    # Validate validation frequency
    if config.validation.frequency == ValidationFrequency.EVERY_N_STEPS:
        if config.validation.every_n_steps is None:
            raise ValueError(
                "every_n_steps must be provided for EVERY_N_STEPS validation frequency"
            )
        if config.validation.every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")

    # Validate loss weights
    if config.loss_weights_per_loader is not None:
        if any(w <= 0 for w in config.loss_weights_per_loader):
            raise ValueError("All loss weights must be positive")
        if config.train_loader_config.dataloader_names and len(
            config.loss_weights_per_loader
        ) != len(config.train_loader_config.dataloader_names):
            raise ValueError("Number of loss weights must match number of dataloaders")

    # Validate per-loader optimizer IDs
    if config.per_loader_optimizer_id is not None:
        if config.train_loader_config.dataloader_names and len(
            config.per_loader_optimizer_id
        ) != len(config.train_loader_config.dataloader_names):
            raise ValueError("Number of optimizer IDs must match number of dataloaders")
        if any(idx < 0 for idx in config.per_loader_optimizer_id):
            raise ValueError("Optimizer IDs must be non-negative")
    # Note: Full infinite loader validation happens at runtime when actual
    # dataloaders are provided. For policies that cannot handle infinite loaders
    # (SUM_OF_LENGTHS, MIN_OF_LENGTHS), runtime validation will enforce finite length
