"""
Configuration schemas for the model training framework.

This module defines all configuration dataclasses, enums, and validation schemas
used throughout the framework. It provides a hierarchical configuration system
that supports composition, validation, and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


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


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    type: str
    hidden_size: int = 512
    num_layers: int = 6
    dropout: float = 0.1
    activation: str = "relu"
    num_classes: Optional[int] = None
    pretrained: bool = False
    freeze_backbone: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    type: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    type: str = "cosine"
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    min_lr: float = 1e-6
    gamma: float = 0.1
    step_size: int = 10
    milestones: List[int] = field(default_factory=list)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Dataset configuration."""

    dataset_name: str
    dataset_path: Optional[str] = None
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    drop_last: bool = True
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    augmentations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training process configuration."""

    max_epochs: int = 100
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = 1.0
    use_amp: bool = True
    early_stopping_patience: Optional[int] = None
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    validation_frequency: int = 1
    log_frequency: int = 100
    save_frequency: int = 1000


@dataclass
class LoggingConfig:
    """Experiment tracking and logging configuration."""

    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    log_scalars_every_n_steps: Optional[int] = 50
    log_images_every_n_steps: Optional[int] = 500
    log_gradients: bool = False
    log_model_parameters: bool = False
    log_system_metrics: bool = True
    tensorboard_dir: Optional[str] = None
    csv_log_dir: Optional[str] = None


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
    constraint: Optional[str] = "a40|a100"
    requeue: bool = True
    job_name: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    mail_type: Optional[str] = None
    mail_user: Optional[str] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Checkpoint configuration from trainer module."""

    root_dir: Union[str, Path] = "checkpoints"
    save_every_n_steps: Optional[int] = None
    save_every_n_epochs: Optional[int] = 1
    max_checkpoints: int = 5
    save_rng: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True
    filename_template: str = "epoch_{epoch:03d}_step_{step:06d}.ckpt"
    monitor_metric: Optional[str] = None
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
    clip_grad_norm: Optional[float] = 1.0
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
    scheduler: Optional[SchedulerConfig] = None
    slurm: Optional[SLURMConfig] = None
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    version: str = "1.0"

    # Runtime settings
    seed: Optional[int] = None
    deterministic: bool = True
    benchmark: bool = False

    # Custom parameters for extensibility
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GridSearchConfig:
    """Configuration for parameter grid search."""

    name: str
    description: str = ""
    base_config: Dict[str, Any] = field(default_factory=dict)
    parameter_grids: List[Dict[str, Any]] = field(default_factory=list)
    naming_strategy: NamingStrategy = NamingStrategy.HASH_BASED
    max_concurrent_jobs: Optional[int] = None
    execution_mode: ExecutionMode = ExecutionMode.SLURM
    output_dir: Optional[str] = None


@dataclass
class ResourceRequirements:
    """Resource requirements for validation."""

    min_memory_gb: float = 1.0
    min_cpu_cores: int = 1
    min_gpu_memory_gb: float = 0.0
    max_time_hours: float = 24.0
    required_partitions: List[str] = field(default_factory=list)
    required_constraints: List[str] = field(default_factory=list)
