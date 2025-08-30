# API Reference

This document provides comprehensive API documentation for the Model Training Framework's three main components.

## Quick Links

- [Config Search Sweep](#config-search-sweep) - Hyperparameter search and experiment generation
- [SLURM Launcher](#slurm-launcher) - HPC job submission and management
- [Trainer](#trainer) - Model training with fault tolerance

---

## Config Search Sweep

The configuration and parameter search system for generating and managing experiments.

### ParameterGridSearch

Main class for orchestrating parameter grid searches.

```python
from model_training_framework.config import ParameterGridSearch
```

#### Constructor

```python
ParameterGridSearch(base_config: Dict[str, Any])
```

**Parameters:**

- `base_config`: Base configuration dictionary or ExperimentConfig object

#### Methods

##### `add_grid(grid: ParameterGrid) -> None`

Add a parameter grid for exploration.

```python
gs = ParameterGridSearch(base_config)
grid = ParameterGrid("hyperparameter_search")
grid.add_parameter("optimizer.lr", [1e-3, 1e-4])
gs.add_grid(grid)
```

##### `set_naming_strategy(strategy: NamingStrategy) -> None`

Set the experiment naming strategy.

```python
from model_training_framework.config import NamingStrategy

gs.set_naming_strategy(NamingStrategy.PARAMETER_BASED)
# Options: PARAMETER_BASED, HASH_BASED, TIMESTAMP_BASED
```

##### `generate_experiments() -> Iterator[ExperimentConfig]`

Generate all experiment configurations.

```python
experiments = list(gs.generate_experiments())
print(f"Generated {len(experiments)} experiments")
```

##### `save_grid_config(path: Path) -> None`

Save grid configuration to file.

```python
gs.save_grid_config(Path("grid_config.json"))
```

##### `save_summary(path: Path) -> None`

Save human-readable summary.

```python
gs.save_summary(Path("summary.txt"))
```

### ParameterGrid

Define parameter search spaces.

```python
from model_training_framework.config import ParameterGrid
```

#### Constructor

```python
ParameterGrid(name: str, description: str = "")
```

#### Methods

##### `add_parameter(key: str, values: List[Any]) -> ParameterGrid`

Add a parameter with possible values (supports method chaining).

```python
grid = (
    ParameterGrid("search")
    .add_parameter("optimizer.lr", [1e-3, 1e-4, 1e-5])
    .add_parameter("model.hidden_size", [128, 256, 512])
)
```

##### `get_parameter_count() -> int`

Get total number of parameter combinations.

```python
count = grid.get_parameter_count()  # e.g., 9 for 3x3 grid
```

##### `generate_permutations() -> Iterator[Dict[str, Any]]`

Generate all parameter combinations.

```python
for params in grid.generate_permutations():
    print(params)  # {"optimizer.lr": 1e-3, "model.hidden_size": 128}
```

### ExperimentConfig

Configuration schema for experiments.

```python
from model_training_framework.config import ExperimentConfig
```

#### Key Fields

```python
@dataclass
class ExperimentConfig:
    experiment_name: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    optimizer: OptimizerConfig
    slurm: Optional[SLURMConfig] = None
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
```

### ConfigurationManager

Manage configuration loading and validation.

```python
from model_training_framework.config import ConfigurationManager
```

#### Constructor

```python
ConfigurationManager(project_root: Path, config_dir: Optional[Path] = None)
```

#### Methods

##### `load_config(config_path: Union[str, Path], validate: bool = True) -> Dict[str, Any]`

Load configuration from file.

```python
cm = ConfigurationManager(project_root=".")
config = cm.load_config("config.yaml")
```

##### `save_config(config: Union[Dict, ExperimentConfig], path: Path, format: str = "yaml") -> None`

Save configuration to file.

```python
cm.save_config(experiment_config, "experiment.yaml")
```

### Example: Complete Grid Search

```python
from model_training_framework.config import (
    ParameterGridSearch,
    ParameterGrid,
    NamingStrategy,
)

# Base configuration
base = {
    "experiment_name": "baseline",
    "model": {"type": "transformer", "hidden_size": 256},
    "optimizer": {"type": "adamw", "lr": 1e-3},
    "training": {"max_epochs": 50},
}

# Create grid search
gs = ParameterGridSearch(base)
gs.set_naming_strategy(NamingStrategy.PARAMETER_BASED)

# Define search space
grid = (
    ParameterGrid("hyperparameters")
    .add_parameter("optimizer.lr", [1e-4, 5e-4, 1e-3])
    .add_parameter("model.hidden_size", [256, 512, 1024])
    .add_parameter("training.gradient_accumulation_steps", [1, 2, 4])
)

gs.add_grid(grid)

# Generate and save
experiments = list(gs.generate_experiments())
gs.save_grid_config(Path("grid.json"))
gs.save_summary(Path("summary.txt"))

print(f"Generated {len(experiments)} experiments")
```

---

## SLURM Launcher

System for submitting and managing jobs on SLURM HPC clusters.

### SLURMLauncher

Main class for SLURM job submission.

```python
from model_training_framework.slurm import SLURMLauncher
```

#### Constructor

```python
SLURMLauncher(
    template_path: Union[str, Path],
    project_root: Union[str, Path],
    experiments_dir: Optional[Union[str, Path]] = None
)
```

**Parameters:**

- `template_path`: Path to SBATCH template file
- `project_root`: Root directory of project
- `experiments_dir`: Optional directory for experiment outputs (defaults to `{project_root}/experiments`)

#### Methods

##### `submit_single_experiment(config: ExperimentConfig, script_path: Path, use_git_branch: bool = False, dry_run: bool = False) -> SLURMJobResult`

Submit single experiment to SLURM.

```python
launcher = SLURMLauncher(
    template_path="slurm_template.txt",
    project_root=".",
    experiments_dir="./experiments"
)

result = launcher.submit_single_experiment(
    config=config,
    script_path="train.py",
    use_git_branch=False,
    dry_run=False  # Set True to preview without submitting
)

if result.success:
    print(f"Job ID: {result.job_id}")
```

##### `submit_experiment_batch(experiments: List[ExperimentConfig], ...) -> BatchSubmissionResult`

Submit multiple experiments.

```python
result = launcher.submit_experiment_batch(
    experiments=experiments,
    script_path="train.py",
    max_concurrent=10,
    use_git_branch=False,
    dry_run=False
)

print(f"Submitted {result.success_count}/{result.total_experiments} jobs")
```

### SBATCHTemplateEngine

Template engine for generating SLURM scripts.

```python
from model_training_framework.slurm.templates import SBATCHTemplateEngine
```

#### Constructor

```python
SBATCHTemplateEngine(
    template_path: Optional[Path] = None,
    template_string: Optional[str] = None
)
```

#### Methods

##### `generate_sbatch_script(context: TemplateContext, output_path: Path) -> Path`

Generate SBATCH script from template.

```python
engine = SBATCHTemplateEngine(template_path="template.txt")

context = TemplateContext(
    job_name="my_job",
    account="my_account",
    partition="gpu",
    nodes=1,
    gpus_per_node=1,
)

script_path = engine.generate_sbatch_script(context, Path("job.sbatch"))
```

##### `preview_rendered_script(context: TemplateContext) -> str`

Preview rendered script without saving.

```python
preview = engine.preview_rendered_script(context)
print(preview)
```

### Template Variables

Available template variables:

```bash
{{JOB_NAME}}         # Job name
{{ACCOUNT}}          # Account name
{{PARTITION}}        # Partition name
{{NODES}}            # Number of nodes
{{NTASKS_PER_NODE}}  # Tasks per node
{{GPUS_PER_NODE}}    # GPUs per node
{{CPUS_PER_TASK}}    # CPUs per task
{{MEM}}              # Memory allocation
{{TIME}}             # Time limit
{{EXPERIMENT_NAME}}  # Experiment name
{{SCRIPT_PATH}}      # Training script path
{{CONFIG_NAME}}      # Configuration file name
{{OUTPUT_FILE}}      # Output file path
{{ERROR_FILE}}       # Error file path
```

### Example SLURM Template

```bash
#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodes={{NODES}}
#SBATCH --gpus-per-node={{GPUS_PER_NODE}}
#SBATCH --cpus-per-task={{CPUS_PER_TASK}}
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}
#SBATCH --output=experiments/{{EXPERIMENT_NAME}}/slurm_%j.out
#SBATCH --error=experiments/{{EXPERIMENT_NAME}}/slurm_%j.err
#SBATCH --open-mode=append  # Preserve logs across requeues
#SBATCH --signal=USR1@60  # Signal 60s before timeout
#SBATCH --requeue

# Recommended environment settings for DDP stability/performance
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NCCL_ASYNC_ERROR_HANDLING=1
# Optional diagnostics: export NCCL_DEBUG=WARN
# Optional allocator tweak to reduce fragmentation on long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
python {{SCRIPT_PATH}} {{CONFIG_NAME}}
```

### Example: Complete SLURM Workflow

```python
from model_training_framework.slurm import SLURMLauncher
from model_training_framework.config import ParameterGridSearch

# Generate experiments
gs = ParameterGridSearch(base_config)
experiments = list(gs.generate_experiments())

# Create launcher
launcher = SLURMLauncher(
    template_path="slurm_template.txt",
    project_root=".",
    experiments_dir="./experiments"
)

# Submit batch
result = launcher.submit_experiment_batch(
    experiments=experiments,
    script_path="train.py",
    max_concurrent=5,
    dry_run=False
)

# Check results
for job in result.job_results:
    if job.success:
        print(f"{job.experiment_name}: Job {job.job_id}")
    else:
        print(f"{job.experiment_name}: Failed - {job.error}")
```

---

## Trainer

Core training engine with fault tolerance and multi-dataloader support.

### GenericTrainer

Main trainer class.

```python
from model_training_framework.trainer import GenericTrainer
```

#### Constructor

```python
GenericTrainer(
    config: GenericTrainerConfig,
    model: nn.Module,
    optimizers: List[Optimizer],
    fabric: Optional[Fabric] = None
)
```

**Parameters:**

- `config`: Trainer configuration
- `model`: PyTorch model
- `optimizers`: List of optimizers (always a list)
- `fabric`: Optional Lightning Fabric for distributed training

#### Methods

##### `fit(train_loaders: List[DataLoader], val_loaders: Optional[List[DataLoader]], max_epochs: int) -> None`

Main training loop.

```python
trainer.fit(
    train_loaders=[train_loader],  # Always a list
    val_loaders=[val_loader],      # Always a list
    max_epochs=100
)
```

##### `set_training_step(fn: Callable) -> None`

Set custom training step function.

```python
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch
    outputs = trainer.model(x)
    loss = F.cross_entropy(outputs, y)
    return {"loss": loss}

trainer.set_training_step(training_step)
```

##### `set_validation_step(fn: Callable) -> None`

Set custom validation step function.

```python
def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch
    with torch.no_grad():
        outputs = trainer.model(x)
        loss = F.cross_entropy(outputs, y)
    return {"loss": loss}

trainer.set_validation_step(validation_step)
```

##### `save_checkpoint(emergency: bool = False) -> Path`

Save training checkpoint.

```python
checkpoint_path = trainer.save_checkpoint()
# Emergency checkpoint for preemption
emergency_path = trainer.save_checkpoint(emergency=True)
```

##### `load_checkpoint(checkpoint_path: Path) -> None`

Load training checkpoint.

```python
trainer.load_checkpoint("checkpoints/best.ckpt")
```

### GenericTrainerConfig

Configuration for trainer.

```python
from model_training_framework.trainer import GenericTrainerConfig
```

#### Key Components

```python
@dataclass
class GenericTrainerConfig:
    train_loader_config: MultiDataLoaderConfig           # Training multi-loader configuration (required)
    val_loader_config: Optional[MultiDataLoaderConfig] = None  # Validation multi-loader configuration
    checkpoint: CheckpointConfig                          # Checkpointing settings
    logging: LoggingConfig                                # Logging configuration
    validation: ValidationConfig                          # Validation settings
    fault_tolerance: FaultToleranceConfig                 # Fault tolerance settings
    performance: PerformanceConfig                        # Performance optimizations
    hooks: HooksConfig                                    # Hooks configuration
```

### MultiDataLoaderConfig

Configuration for multi-dataloader training.

```python
from model_training_framework.trainer import MultiDataLoaderConfig, SamplingStrategy
```

#### Fields

```python
@dataclass
class MultiDataLoaderConfig:
    sampling_strategy: SamplingStrategy    # ROUND_ROBIN, WEIGHTED, ALTERNATING, SEQUENTIAL
    dataloader_names: List[str]           # Names for each dataloader
    dataloader_weights: Optional[List[float]] = None  # For WEIGHTED strategy
    alternating_pattern: Optional[List[int]] = None   # For ALTERNATING strategy
    epoch_length_policy: EpochLengthPolicy = EpochLengthPolicy.SUM_OF_LENGTHS
    steps_per_epoch: Optional[int] = None  # For FIXED_NUM_STEPS policy
```

### Sampling Strategies

```python
from model_training_framework.trainer import SamplingStrategy

SamplingStrategy.ROUND_ROBIN   # Fair alternation
SamplingStrategy.WEIGHTED      # Probability-based
SamplingStrategy.ALTERNATING   # Custom pattern
SamplingStrategy.SEQUENTIAL    # One after another
```

### CheckpointConfig

```python
@dataclass
class CheckpointConfig:
    save_every_n_epochs: Optional[int] = 1
    save_every_n_steps: Optional[int] = None
    max_checkpoints: int = 5
    root_dir: str | Path = "checkpoints"
    filename_template: str = "epoch_{epoch:03d}_step_{step:06d}.ckpt"
    monitor_metric: Optional[str] = None  # e.g., "val/loss"
    monitor_mode: str = "min"  # "min" or "max"
    save_best: bool = True
```

### Example: Complete Training Setup

```python
from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
    CheckpointConfig,
    LoggingConfig,
)
import torch.nn as nn

# Model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Configuration
config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.WEIGHTED,
        dataloader_weights=[0.7, 0.3],
        dataloader_names=["primary", "auxiliary"],
    ),
    val_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["validation"],
    ),
    checkpoint=CheckpointConfig(
        save_every_n_steps=500,
        max_checkpoints=3,
    ),
    logging=LoggingConfig(
        logger_type="wandb",
        wandb_project="my-project",
    )
)

# Trainer
trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=[optimizer]
)

# Step functions
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch
    outputs = trainer.model(x)
    loss = nn.functional.cross_entropy(outputs, y)
    return {"loss": loss, f"{dataloader_name}/loss": loss}

trainer.set_training_step(training_step)

# Train
trainer.fit(
    train_loaders=[primary_loader, auxiliary_loader],
    val_loaders=[val_primary, val_auxiliary],
    max_epochs=100
)
```

---

## Additional Components

### Hooks System

```python
from model_training_framework.trainer.hooks import TrainerHooks

class CustomHook(TrainerHooks):
    def on_epoch_end(self, trainer, epoch):
        print(f"Epoch {epoch} completed")

trainer.hook_manager.register_hook(CustomHook())
```

See [Hooks Documentation](HOOKS.md) for details.

### Logging

```python
from model_training_framework.config.schemas import LoggingConfig

config = LoggingConfig(
    logger_type="composite",
    composite_loggers=["console", "tensorboard", "wandb"],
    wandb_project="my-project",
    tensorboard_dir="./logs"
)
```

See [Observability Guide](OBSERVABILITY.md) for details.

### Metrics

```python
from model_training_framework.trainer import MetricsManager

metrics_manager = MetricsManager(
    aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE
)
```

---

## Error Handling

### Custom Exceptions

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
    JobSubmissionError
)

try:
    trainer.fit(train_loaders, val_loaders)
except PreemptionTimeoutError as e:
    # Handle preemption timeout
    trainer.save_emergency_checkpoint()
except TrainerError as e:
    # Handle general trainer errors
    logger.error(f"Training failed: {e}")
```

---

## Complete Workflow Example

```python
from model_training_framework.config import ParameterGridSearch, ParameterGrid
from model_training_framework.slurm import SLURMLauncher
from model_training_framework.trainer import GenericTrainer, GenericTrainerConfig

# 1. Configure experiments
base_config = {...}
gs = ParameterGridSearch(base_config)
grid = ParameterGrid("search").add_parameter("lr", [1e-3, 1e-4])
gs.add_grid(grid)
experiments = list(gs.generate_experiments())

# 2. Submit to SLURM
launcher = SLURMLauncher("template.txt", ".", "./experiments")
result = launcher.submit_experiment_batch(experiments, "train.py")

# 3. Training script (train.py)
config = load_config(sys.argv[1])
trainer_config = GenericTrainerConfig(...)
trainer = GenericTrainer(trainer_config, model, [optimizer])
trainer.fit([train_loader], [val_loader], max_epochs=100)
```

---

## See Also

- [Quick Start Guide](QUICKSTART.md)
- [Configuration Guide](CONFIGURATION.md)
- [Multi-DataLoader Guide](MULTI_DATALOADER.md)
- [Migration Guide](MIGRATION.md)
- [Example Code](../demo/example3_production/)
