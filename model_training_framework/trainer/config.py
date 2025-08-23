"""
Trainer Configuration Classes

This module defines configuration classes for the training engine:
- CheckpointConfig for checkpoint behavior
- PreemptionConfig for handling job preemption
- PerformanceConfig for optimization settings
- LoggingConfig for training logging
- GenericTrainerConfig for overall trainer configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


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
class LoggingConfig:
    """
    Configuration for experiment tracking and logging.

    Controls integration with various logging backends including Weights & Biases,
    TensorBoard, CSV files, and console output.
    """

    use_wandb: bool = True  # Whether to use Weights & Biases for experiment tracking
    wandb_project: str | None = (
        None  # W&B project name (uses experiment name if None)
    )
    wandb_entity: str | None = None  # W&B entity/team name
    wandb_tags: list[str] = field(default_factory=list)  # Tags for W&B experiment
    wandb_notes: str | None = None  # Notes for W&B experiment

    log_scalars_every_n_steps: int | None = (
        50  # How often to log scalar metrics (None = every step)
    )
    log_images_every_n_steps: int | None = (
        500  # How often to log images (None = never)
    )
    log_gradients: bool = False  # Whether to log gradient statistics
    log_model_parameters: bool = False  # Whether to log model parameter statistics
    log_system_metrics: bool = True  # Whether to log system metrics (GPU, memory, etc.)

    # Additional logging backends
    use_tensorboard: bool = False  # Whether to use TensorBoard logging
    tensorboard_dir: str | None = None  # TensorBoard log directory
    use_csv: bool = True  # Whether to log metrics to CSV files
    csv_log_dir: str | None = None  # CSV log directory (uses checkpoint dir if None)

    # Console logging
    console_log_level: str = "INFO"  # Console logging level
    console_log_format: str = (
        "[%(asctime)s] %(levelname)s: %(message)s"  # Console log format
    )


@dataclass
class GenericTrainerConfig:
    """
    Master configuration for the GenericTrainer.

    This combines all sub-configurations needed for training including checkpointing,
    preemption handling, performance optimization, and logging.
    """

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

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
    profile_training: bool = False  # Enable profiling of training loop
    dry_run: bool = False  # Run training loop without actual training (for debugging)

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

    def get_summary(self) -> dict[str, any]:
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
                "use_wandb": self.logging.use_wandb,
                "wandb_project": self.logging.wandb_project,
                "log_scalars_every_n_steps": self.logging.log_scalars_every_n_steps,
            },
            "training": {
                "log_loss_every_n_steps": self.log_loss_every_n_steps,
                "validate_every_n_epochs": self.validate_every_n_epochs,
                "early_stopping_patience": self.early_stopping_patience,
            },
        }
