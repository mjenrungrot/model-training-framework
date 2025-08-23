# API Documentation

This document provides comprehensive API documentation for the Model Training Framework.

## Table of Contents

1. [Core Framework](#core-framework)
2. [Configuration Management](#configuration-management)
3. [Training Engine](#training-engine)
4. [SLURM Integration](#slurm-integration)
5. [Utilities](#utilities)

## Core Framework

### ModelTrainingFramework

The main entry point for the framework providing high-level orchestration.

```python
from model_training_framework import ModelTrainingFramework
```

#### Constructor

```python
ModelTrainingFramework(
    project_root: Union[str, Path],
    config_dir: Optional[Union[str, Path]] = None,
    slurm_template_path: Optional[Union[str, Path]] = None
)
```

**Parameters:**

- `project_root`: Root directory of your project
- `config_dir`: Directory containing configuration files (default: `{project_root}/configs`)
- `slurm_template_path`: Path to SLURM template file (optional)

#### Methods

##### `run_grid_search()`

Execute parameter grid search across multiple configurations.

```python
run_grid_search(
    parameter_grids: List[ParameterGrid],
    base_config: Optional[Union[Dict[str, Any], str, Path]] = None,
    base_config_path: Optional[Union[str, Path]] = None,
    execution_mode: ExecutionMode = ExecutionMode.SLURM,
    output_dir: Optional[Path] = None,
    max_concurrent_jobs: Optional[int] = None
) -> GridSearchResult
```

**Parameters:**

- `parameter_grids`: List of parameter grids to explore
- `base_config`: Base configuration as dictionary
- `base_config_path`: Path to base configuration file
- `execution_mode`: How to execute experiments (`SLURM`, `LOCAL`, `DRY_RUN`)
- `output_dir`: Directory for experiment outputs
- `max_concurrent_jobs`: Maximum number of concurrent SLURM jobs

**Returns:** `GridSearchResult` object with execution details

##### `get_config_manager()`

Get the configuration manager instance.

```python
get_config_manager() -> ConfigurationManager
```

##### `get_slurm_launcher()`

Get the SLURM launcher instance.

```python
get_slurm_launcher() -> SLURMLauncher
```

## Configuration Management

### ExperimentConfig

Main configuration schema for experiments.

```python
from model_training_framework.config import ExperimentConfig
```

#### Fields

```python
@dataclass
class ExperimentConfig:
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
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    version: str = "1.0"
    seed: Optional[int] = None
    deterministic: bool = True
    benchmark: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)
```

### ParameterGrid

Define parameter search spaces for grid search.

```python
from model_training_framework.config import ParameterGrid
```

#### Constructor

```python
ParameterGrid(
    name: str,
    description: str = ""
)
```

#### Methods

##### `add_parameter()`

Add a parameter with possible values.

```python
add_parameter(key: str, values: List[Any]) -> None
```

**Parameters:**

- `key`: Parameter key (supports dot notation, e.g., "model.hidden_size")
- `values`: List of possible values to explore

##### `add_nested_parameter()`

Add nested parameter using dot notation.

```python
add_nested_parameter(key_path: str, values: List[Any]) -> None
```

##### `get_parameter_count()`

Get total number of parameter combinations.

```python
get_parameter_count() -> int
```

##### `generate_permutations()`

Generate all parameter combinations.

```python
generate_permutations() -> Iterator[Dict[str, Any]]
```

### ParameterGridSearch

Orchestrate parameter grid search execution.

```python
from model_training_framework.config import ParameterGridSearch
```

#### Constructor

```python
ParameterGridSearch(base_config: Dict[str, Any])
```

#### Methods

##### `add_grid()`

Add a parameter grid for exploration.

```python
add_grid(grid: ParameterGrid) -> None
```

##### `create_grid()`

Create and add a new parameter grid.

```python
create_grid(name: str, description: str = "") -> ParameterGrid
```

##### `set_naming_strategy()`

Set the experiment naming strategy.

```python
set_naming_strategy(strategy: NamingStrategy) -> None
```

**Naming Strategies:**

- `NamingStrategy.HASH_BASED`: Use hash of parameters
- `NamingStrategy.DESCRIPTIVE`: Use parameter names and values
- `NamingStrategy.SEQUENTIAL`: Use sequential numbering

##### `generate_experiments()`

Generate all experiment configurations.

```python
generate_experiments() -> Iterator[ExperimentConfig]
```

### ConfigurationManager

Manage configuration loading, validation, and composition.

```python
from model_training_framework.config import ConfigurationManager
```

#### Constructor

```python
ConfigurationManager(
    project_root: Path,
    config_dir: Optional[Path] = None
)
```

#### Methods

##### `load_config()`

Load configuration from file with optional validation.

```python
load_config(
    config_path: Union[str, Path],
    validate: bool = True,
    use_cache: bool = True
) -> Dict[str, Any]
```

##### `save_config()`

Save configuration to file.

```python
save_config(
    config: Union[Dict[str, Any], ExperimentConfig],
    output_path: Union[str, Path],
    format: str = "yaml"
) -> None
```

##### `compose_configs()`

Compose configuration from base config and overrides.

```python
compose_configs(
    base_config: Union[str, Path, Dict[str, Any]],
    overrides: List[Union[str, Path, Dict[str, Any]]] = None,
    parameter_overrides: Dict[str, Any] = None
) -> Dict[str, Any]
```

## Training Engine

### GenericTrainer

Core trainer class with fault-tolerant capabilities.

```python
from model_training_framework.trainer import GenericTrainer
```

#### Constructor

```python
GenericTrainer(config: GenericTrainerConfig)
```

#### Methods

##### `fit()`

Main training loop with automatic preemption handling.

```python
fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    scheduler: Optional[Any] = None
) -> None
```

##### `training_step()`

Execute one training step. Override in subclasses.

```python
training_step(batch: Any, batch_idx: int) -> Dict[str, Any]
```

**Returns:** Dictionary with at least `'loss'` key

##### `validation_step()`

Execute one validation step. Override in subclasses.

```python
validation_step(batch: Any, batch_idx: int) -> Dict[str, Any]
```

**Returns:** Dictionary with validation metrics

##### `save_checkpoint()`

Save training checkpoint.

```python
save_checkpoint(emergency: bool = False) -> Path
```

##### `load_checkpoint()`

Load training checkpoint.

```python
load_checkpoint(checkpoint_path: Optional[Path] = None) -> ResumeState
```

##### `add_callback()`

Add training callback.

```python
add_callback(callback: TrainingCallback) -> None
```

### TrainingCallback

Base class for training callbacks.

```python
from model_training_framework.trainer import TrainingCallback
```

#### Methods to Override

```python
def on_training_start(self, trainer: GenericTrainer):
    """Called at the start of training."""
    pass

def on_training_end(self, trainer: GenericTrainer, logs: Dict[str, Any]):
    """Called at the end of training."""
    pass

def on_epoch_start(self, trainer: GenericTrainer, epoch: int):
    """Called at the start of each epoch."""
    pass

def on_epoch_end(self, trainer: GenericTrainer, epoch: int, logs: Dict[str, Any]):
    """Called at the end of each epoch."""
    pass

def on_batch_start(self, trainer: GenericTrainer, batch_idx: int):
    """Called at the start of each batch."""
    pass

def on_batch_end(self, trainer: GenericTrainer, batch_idx: int, logs: Dict[str, Any]):
    """Called at the end of each batch."""
    pass
```

### Training States

#### TrainerPhase

Enumeration of all training phases for instruction-level checkpointing.

```python
from model_training_framework.trainer import TrainerPhase

# Key phases
TrainerPhase.INIT
TrainerPhase.TRAIN_START_EPOCH
TrainerPhase.TRAIN_BATCH_FORWARD
TrainerPhase.TRAIN_BATCH_BACKWARD
TrainerPhase.TRAIN_BATCH_OPTIM_STEP
TrainerPhase.VAL_START_EPOCH
TrainerPhase.VAL_BATCH_FORWARD
TrainerPhase.EPOCH_END
TrainerPhase.TRAINING_COMPLETE
```

#### ResumeState

Complete checkpoint state for deterministic resume.

```python
from model_training_framework.trainer import ResumeState

@dataclass
class ResumeState:
    phase: TrainerPhase
    epoch: int
    global_step: int
    version: str = "v2.0"
    train: Optional[TrainMicroState] = None
    val: Optional[ValMicroState] = None
    rng: Optional[RNGState] = None
    timestamp: float = 0.0
```

## SLURM Integration

### SLURMLauncher

Handle SLURM job submission and management.

```python
from model_training_framework.slurm import SLURMLauncher
```

#### Constructor

```python
SLURMLauncher(
    slurm_template_path: str,
    output_dir: str,
    git_manager: Optional[GitManager] = None
)
```

#### Methods

##### `submit_experiment()`

Submit single experiment to SLURM.

```python
submit_experiment(
    experiment_config: ExperimentConfig,
    execution_mode: ExecutionMode = ExecutionMode.SLURM
) -> str
```

**Returns:** Job ID string

##### `submit_experiment_batch()`

Submit multiple experiments to SLURM.

```python
submit_experiment_batch(
    experiments: List[ExperimentConfig],
    execution_mode: ExecutionMode = ExecutionMode.SLURM,
    max_concurrent_jobs: Optional[int] = None
) -> BatchSubmissionResult
```

**Returns:** `BatchSubmissionResult` with successful and failed submissions

##### `get_job_status()`

Get status of a SLURM job.

```python
get_job_status(job_id: str) -> JobStatus
```

##### `cancel_job()`

Cancel a SLURM job.

```python
cancel_job(job_id: str) -> bool
```

### GitManager

Manage git operations for experiment isolation.

```python
from model_training_framework.slurm import GitManager
```

#### Constructor

```python
GitManager(repo_path: str)
```

#### Methods

##### `branch_context()`

Context manager for working on a specific branch.

```python
@contextmanager
branch_context(branch_name: str, create_if_not_exists: bool = True):
    # Work on the specified branch
    # Automatically returns to original branch
    pass
```

##### `create_branch()`

Create a new git branch.

```python
create_branch(branch_name: str, base_branch: Optional[str] = None) -> bool
```

##### `commit_changes()`

Commit changes to current branch.

```python
commit_changes(
    message: str,
    files: Optional[List[str]] = None,
    add_all: bool = False
) -> str
```

## Utilities

### Logging

```python
from model_training_framework.utils import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO", log_file="training.log")

# Get logger
logger = get_logger(__name__)
```

### Path Utilities

```python
from model_training_framework.utils import (
    get_project_root,
    validate_project_structure,
    ensure_directory_exists
)

# Get project root
root = get_project_root()

# Validate structure
is_valid = validate_project_structure(root)

# Ensure directory exists
ensure_directory_exists(Path("./experiments"))
```

### Data Structures

```python
from model_training_framework.utils import Result, Success, Error

# Result pattern
def some_operation() -> Result[str, str]:
    if success:
        return Success("operation completed")
    else:
        return Error("operation failed")

result = some_operation()
if isinstance(result, Success):
    print(result.value)
else:
    print(f"Error: {result.error}")
```

## Error Handling

The framework defines several custom exceptions:

```python
from model_training_framework.trainer import (
    TrainerError,
    PreemptionTimeoutError,
    CheckpointTimeoutError
)

from model_training_framework.config import (
    ConfigurationError,
    ValidationError
)

from model_training_framework.slurm import (
    SLURMError,
    JobSubmissionError,
    GitOperationError
)
```

## Type Hints

The framework provides comprehensive type hints. Import common types:

```python
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_training.py`: Simple training setup
- `grid_search.py`: Parameter grid search
- `slurm_batch.py`: Batch job submission
- `custom_trainer.py`: Custom trainer implementation
- `preemption_handling.py`: Preemption recovery

## Configuration Schema Reference

### Complete Configuration Example

```yaml
experiment_name: "example_experiment"
description: "Complete configuration example"

model:
  name: "transformer"
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0

data:
  dataset_name: "custom_dataset"
  train_split: "train"
  val_split: "validation"
  test_split: "test"
  preprocessing: {}

optimizer:
  name: "adamw"
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  name: "cosine"
  warmup_steps: 1000
  min_lr: 1e-6

slurm:
  job_name: "training_job"
  time_limit: "24:00:00"
  nodes: 1
  ntasks_per_node: 1
  cpus_per_task: 4
  mem: "32G"
  gres: "gpu:1"
  partition: "gpu"
  additional_args: []

logging:
  log_level: "INFO"
  use_wandb: true
  wandb_project: "my_project"
  wandb_entity: "my_team"
  log_every_n_steps: 10

checkpoint:
  save_every_n_epochs: 5
  save_top_k: 3
  monitor: "val_loss"
  mode: "min"
  checkpoint_dir: "./checkpoints"

preemption:
  timeout_minutes: 5
  grace_period_seconds: 60
  max_preemptions: 3

performance:
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true

# Custom parameters
seed: 42
deterministic: true
benchmark: false
tags: ["experiment", "baseline"]
version: "1.0"
```

This documentation covers the main APIs and usage patterns. For more detailed examples, see the example scripts and test files.
