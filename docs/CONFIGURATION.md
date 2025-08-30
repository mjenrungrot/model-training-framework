# Configuration Guide

This guide provides detailed information about configuring the Model Training Framework for various use cases.

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Base Configuration](#base-configuration)
3. [Parameter Grid Search](#parameter-grid-search)
4. [SLURM Configuration](#slurm-configuration)
5. [Environment Variables](#environment-variables)
6. [Configuration Composition](#configuration-composition)
7. [Validation](#validation)
8. [Best Practices](#best-practices)

## Configuration Overview

The framework uses a hierarchical configuration system based on YAML files with the following components:

- **ExperimentConfig**: Main configuration container
- **ModelConfig**: Model architecture parameters
- **TrainingConfig**: Training hyperparameters
- **DataConfig**: Dataset and data loading configuration
- **OptimizerConfig**: Optimizer settings
- **SchedulerConfig**: Learning rate scheduler settings
- **SLURMConfig**: SLURM job parameters
- **LoggingConfig**: Logging and monitoring settings

## Base Configuration

### Complete Configuration Template

```yaml
# Experiment metadata
experiment_name: "example_experiment"
description: "Detailed description of the experiment"
tags: ["baseline", "transformer", "large-scale"]
version: "1.0"
seed: 42
deterministic: true
benchmark: false

# Model configuration
model:
  name: "transformer"
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1
  activation: "gelu"
  max_sequence_length: 1024
  vocab_size: 50000

# Training configuration
training:
  max_epochs: 100
  batch_size: 32
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  warmup_steps: 1000
  validation_every_n_epochs: 1
  early_stopping_patience: 10

# Data configuration
data:
  dataset_name: "custom_dataset"
  data_dir: "/path/to/data"
  train_split: "train"
  val_split: "validation"
  test_split: "test"
  preprocessing:
    tokenizer: "bert-base-uncased"
    max_length: 512
    padding: "max_length"
    truncation: true

# Optimizer configuration
optimizer:
  name: "adamw"
  lr: 0.001
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8
  amsgrad: false

# Scheduler configuration (optional)
scheduler:
  name: "cosine"
  warmup_steps: 1000
  min_lr: 1e-6
  cycle_length: 10000
  restart_factor: 1.0

# SLURM configuration (optional)
slurm:
  job_name: "training_job"
  time_limit: "24:00:00"
  nodes: 1
  ntasks_per_node: 1
  cpus_per_task: 4
  mem: "32G"
  gres: "gpu:1"
  partition: "gpu"
  account: "my_account"
  qos: "normal"
  constraint: "v100"
  additional_args:
    - "--exclusive"
    - "--mail-type=END,FAIL"
    - "--mail-user=user@example.com"

# Logging configuration
logging:
  use_wandb: true
  wandb_project: "my_project"
  wandb_entity: "my_team"
  wandb_tags: ["experiment", "baseline"]
  wandb_name: "experiment-1"
  wandb_mode: "online"      # one of: online|offline|disabled
  wandb_id: "run-123"        # stable unique run id
  wandb_resume: "allow"      # one of: allow|must|never
  log_scalars_every_n_steps: 10
  log_images_every_n_steps: 500

# Checkpoint configuration
checkpoint:
  save_every_n_epochs: 5
  save_every_n_steps: null
  max_checkpoints: 3
  root_dir: "./checkpoints"
  filename_template: "epoch_{epoch:03d}_step_{step:06d}.ckpt"
  monitor_metric: "val/loss"
  monitor_mode: "min"
  save_best: true
  save_last: true

# Preemption configuration
preemption:
  # Time limit for saving a checkpoint on preemption (seconds)
  max_checkpoint_sec: 300
  # Whether to request SLURM requeue after checkpointing
  requeue_job: true
  # Resume from the latest symlink when restarting
  resume_from_latest_symlink: true
  # Optional: POSIX signal used for preemption handling (matches code examples)
  # signal: USR1  # corresponds to signal.SIGUSR1 in Python

# Performance configuration
performance:
  dataloader_num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true
  dataloader_drop_last: false
  use_amp: true  # Enable automatic mixed precision
  compile_model: false

# Custom parameters (project-specific)
custom_params:
  use_custom_loss: true
  loss_weight: 0.5
  augmentation:
    probability: 0.5
    methods: ["rotation", "scaling"]
  metrics:
    - "accuracy"
    - "f1_score"
    - "precision"
    - "recall"
```

### Minimal Configuration

For simple experiments, you can use a minimal configuration:

```yaml
experiment_name: "minimal_experiment"

model:
  name: "simple_mlp"
  hidden_size: 256

training:
  max_epochs: 10
  batch_size: 32

data:
  dataset_name: "mnist"

optimizer:
  name: "adam"
  lr: 0.001
```

## Parameter Grid Search

### Basic Grid Search

The framework provides a powerful and flexible grid search API for hyperparameter exploration:

```python
from model_training_framework.config import ParameterGridSearch, ParameterGrid
from pathlib import Path

# Define base configuration (dict or ExperimentConfig)
base_config = {
    "experiment_name": "transformer_baseline",
    "model": {
        "type": "transformer",
        "hidden_size": 256,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1
    },
    "data": {
        "dataset_name": "my_dataset",
        "batch_size": 32,
        "num_workers": 4
    },
    "training": {
        "max_epochs": 50,
        "gradient_accumulation_steps": 1,
        "validation_frequency": 100,
        "log_frequency": 10
    },
    "optimizer": {
        "type": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.01
    },
    "logging": {
        "use_wandb": True,
        "wandb_project": "my_project"
    }
}

# Create grid search with base config
gs = ParameterGridSearch(base_config)

# Define parameter grids with method chaining
optimization_grid = (
    ParameterGrid("optimization")
    .add_parameter("optimizer.lr", [1e-4, 5e-4, 1e-3])
    .add_parameter("optimizer.weight_decay", [0.01, 0.05, 0.1])
    .add_parameter("training.gradient_accumulation_steps", [1, 2, 4])
)

architecture_grid = (
    ParameterGrid("architecture")
    .add_parameter("model.hidden_size", [256, 512, 1024])
    .add_parameter("model.num_layers", [4, 6, 8])
    .add_parameter("model.dropout", [0.1, 0.2, 0.3])
)

# Add grids to search
gs.add_grid(optimization_grid)
gs.add_grid(architecture_grid)

# Set naming strategy
gs.set_naming_strategy("parameter_based")  # or "hash_based", "timestamp_based"

# Generate all experiment configurations
experiments = list(gs.generate_experiments())
print(f"Generated {len(experiments)} experiment configurations")

# Save configurations
output_dir = Path("experiments/grid_search")
gs.save_grid_config(output_dir / "grid_config.json")
gs.save_summary(output_dir / "summary.txt")

# Iterate through experiments
for exp in experiments:
    print(f"Experiment: {exp.experiment_name}")
    print(f"  Learning rate: {exp.optimizer.lr}")
    print(f"  Hidden size: {exp.model.hidden_size}")
```

### Advanced Grid Search Features

#### Custom Parameter Combinations

```python
# Create grid with custom combinations
grid = ParameterGrid("custom_combinations")

# Add linked parameters (will be varied together)
grid.add_linked_parameters([
    ("model.hidden_size", [256, 512, 1024]),
    ("model.num_heads", [8, 16, 32])  # Scales with hidden_size
])

# Add conditional parameters
grid.add_parameter_with_condition(
    "optimizer.lr",
    values=[1e-3, 5e-4, 1e-4],
    condition=lambda cfg: cfg["model"]["hidden_size"] <= 512
)
```

#### Using Typed Configurations

For better type safety and IDE support, use the configuration registry:

```python
from model_training_framework.config import ConfigRegistry, register_model, register_dataset
from model_training_framework.config.schemas import ModelConfig, DataConfig
from dataclasses import dataclass

# Register custom model configuration
@register_model("transformer")
@dataclass
class TransformerConfig(ModelConfig):
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_length: int = 512
    vocab_size: int = 30000

    def validate(self) -> list[str]:
        """Custom validation logic."""
        errors = []
        if self.hidden_size % self.num_heads != 0:
            errors.append(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
        return errors

# Register custom dataset configuration
@register_dataset("my_dataset")
@dataclass
class MyDatasetConfig(DataConfig):
    dataset_path: str = "/path/to/data"
    max_samples: int | None = None
    preprocessing: str = "standard"
    cache_dir: str | None = None

# Use typed configs in grid search
base_config = {
    "experiment_name": "typed_experiment",
    "model": {
        "type": "transformer",
        "hidden_size": 512,
        "num_heads": 8
    },
    "data": {
        "dataset_name": "my_dataset",
        "batch_size": 32,
        "preprocessing": "advanced"
    }
}

# Grid search will use registered configs for validation
gs = ParameterGridSearch(base_config)
grid = ParameterGrid("typed_search")
grid.add_parameter("model.num_layers", [6, 12, 24])
grid.add_parameter("data.max_samples", [1000, 5000, None])
gs.add_grid(grid)

# Generate experiments with type checking
experiments = list(gs.generate_experiments())
```

#### Grid Search Result Analysis

```python
# Load and analyze grid search results
from model_training_framework.config import GridSearchResult

# After running experiments, analyze results
result = GridSearchResult(
    grid_config=gs.get_config(),
    total_experiments=len(experiments),
    generated_experiments=experiments,
    submitted_jobs=["job1", "job2", "job3"],
    failed_submissions=[],
    execution_time=3600.0,
    output_directory=output_dir
)

print(f"Success rate: {result.success_rate:.1%}")
print(f"Total experiments: {result.total_experiments}")
print(f"Failed submissions: {len(result.failed_submissions)}")

# Save detailed results
result.save_summary(output_dir / "results.json")
```

## SLURM Configuration

### Basic SLURM Setup

```yaml
slurm:
  job_name: "my_training_job"
  time_limit: "24:00:00"
  nodes: 1
  ntasks_per_node: 1
  cpus_per_task: 4
  mem: "32G"
  gres: "gpu:1"
  partition: "gpu"
```

### Advanced SLURM Configuration

```yaml
slurm:
  # Job identification
  job_name: "large_model_training"
  account: "research_group"

  # Resource allocation
  nodes: 2
  ntasks_per_node: 4
  cpus_per_task: 8
  mem: "128G"
  mem_per_cpu: "4G"

  # GPU configuration
  gres: "gpu:v100:4"
  gpu_bind: "closest"

  # Time limits
  time_limit: "72:00:00"
  begin: "now+1hour"

  # Partitions and QoS
  partition: "gpu_large"
  qos: "high_priority"
  reservation: "special_event"

  # Constraints and features
  constraint: "v100&nvlink"
  exclude: "node001,node002"
  nodelist: "node[010-020]"

  # Notifications
  mail_type: "BEGIN,END,FAIL,TIME_LIMIT"
  mail_user: "researcher@university.edu"

  # Advanced options
  exclusive: true
  requeue: false
  array: "1-10%5"  # Array job with 10 tasks, max 5 concurrent
  dependency: "afterok:12345"

  # Output and error files
  output: "/path/to/logs/%j_%a.out"
  error: "/path/to/logs/%j_%a.err"

  # Additional SBATCH directives
  additional_args:
    - "--signal=USR1@300"  # Send signal 5 minutes before timeout
    - "--open-mode=append"
    - "--get-user-env=L"
```

### SLURM Template

```bash
#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --account={{account}}
#SBATCH --time={{time_limit}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --cpus-per-task={{cpus_per_task}}
#SBATCH --mem={{mem}}
#SBATCH --gres={{gres}}
#SBATCH --partition={{partition}}
#SBATCH --qos={{qos}}
#SBATCH --constraint={{constraint}}
#SBATCH --mail-type={{mail_type}}
#SBATCH --mail-user={{mail_user}}
#SBATCH --output={{output_dir}}/{{experiment_name}}_%j.out
#SBATCH --error={{output_dir}}/{{experiment_name}}_%j.err
{{#additional_args}}
#SBATCH {{.}}
{{/additional_args}}

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"

# Environment setup
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

source activate {{conda_env}}
cd {{project_root}}

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}  # Allow override via SLURM
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export SLURM_NNODES=${SLURM_NNODES:-1}
export SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-1}

# Print environment info
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

# Run training with torchrun for distributed training
# torchrun handles the distributed setup automatically
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    demo/example3_production/train_script.py \
    --config {{config_path}} \
    --experiment-name {{experiment_name}}

echo "Job completed at: $(date)"
```

## Environment Variables

### Supported Environment Variables

The framework supports environment variable substitution in configuration files:

```yaml
# Configuration with environment variables
data:
  data_dir: "${DATA_ROOT}/datasets"
  cache_dir: "${SCRATCH_DIR}/cache"

logging:
  wandb_project: "${WANDB_PROJECT}"
  wandb_entity: "${WANDB_ENTITY}"
  wandb_mode: "${WANDB_MODE}"       # online|offline|disabled
  wandb_id: "${WANDB_RUN_ID}"
  wandb_resume: "${WANDB_RESUME}"   # allow|must|never

slurm:
  account: "${SLURM_ACCOUNT}"
  partition: "${SLURM_PARTITION}"

custom_params:
  model_path: "${MODEL_CACHE}/pretrained_models"
  output_path: "${RESULTS_DIR}/${experiment_name}"
```

### Environment Variable Patterns

```bash
# Simple substitution
${VAR_NAME}

# With default values (not yet supported, but planned)
${VAR_NAME:-default_value}

# Nested substitution (not yet supported, but planned)
${BASE_DIR}/${SUB_DIR}
```

### Setting Environment Variables

```bash
# In your shell or job script
export DATA_ROOT="/scratch/datasets"
export WANDB_PROJECT="my_experiments"
export WANDB_ENTITY="my_team"
export WANDB_MODE="online"          # or offline|disabled
export WANDB_RUN_ID="run-123"
export WANDB_RESUME="allow"          # or must|never
export SLURM_ACCOUNT="research_group"

# Or in a .env file (if using python-dotenv)
DATA_ROOT=/scratch/datasets
WANDB_PROJECT=my_experiments
WANDB_ENTITY=my_team
WANDB_MODE=online
WANDB_RUN_ID=run-123
WANDB_RESUME=allow
SLURM_ACCOUNT=research_group
```

## Configuration Composition

### Configuration Inheritance

```python
# Load base configuration
base_config = config_manager.load_config("base_config.yaml")

# Compose with overrides
final_config = config_manager.compose_configs(
    base_config="base_config.yaml",
    overrides=[
        "overrides/large_model.yaml",
        "overrides/gpu_settings.yaml"
    ],
    parameter_overrides={
        "optimizer.lr": 0.0005,
        "model.hidden_size": 1024
    }
)
```

### Override Files

```yaml
# overrides/large_model.yaml
model:
  hidden_size: 1024
  num_layers: 24

training:
  batch_size: 16  # Smaller batch size for large model
  gradient_accumulation_steps: 4

performance:
  use_amp: true
```

```yaml
# overrides/gpu_settings.yaml
slurm:
  gres: "gpu:a100:2"
  mem: "64G"

performance:
  dataloader_num_workers: 8
  pin_memory: true
```

## Validation

### Configuration Validation

The framework automatically validates configurations:

```python
# Load and validate configuration
try:
    config = config_manager.load_config("my_config.yaml", validate=True)
    print("Configuration is valid")
except ValidationError as e:
    print(f"Configuration validation failed: {e}")
```

### Custom Validation Rules

```python
from model_training_framework.config.validators import ConfigValidator

# Add custom validation rules
validator = ConfigValidator()

@validator.add_rule("model.hidden_size")
def validate_hidden_size(value):
    if value % 64 != 0:
        return "Hidden size must be divisible by 64"
    return None

# Validate configuration
result = validator.validate_config(experiment_config)
if result.has_errors:
    for error in result.get_errors():
        print(f"Error: {error.message}")
```

## Best Practices

### 1. Configuration Organization

```text
configs/
├── base/
│   ├── model_base.yaml
│   ├── training_base.yaml
│   └── data_base.yaml
├── experiments/
│   ├── experiment_001.yaml
│   ├── experiment_002.yaml
│   └── ...
├── overrides/
│   ├── large_model.yaml
│   ├── small_gpu.yaml
│   └── debug.yaml
└── grid_searches/
    ├── lr_search.json
    ├── architecture_search.json
    └── ...
```

### 2. Naming Conventions

```yaml
# Use descriptive experiment names
experiment_name: "bert_large_squad_lr1e-5_bs16"

# Include important hyperparameters in the name
experiment_name: "transformer_h512_l6_lr0.001"

# Use version numbers for iterative experiments
experiment_name: "baseline_v1.0"
```

### 3. Parameter Organization

```yaml
# Group related parameters
model:
  # Architecture
  hidden_size: 512
  num_layers: 6
  num_heads: 8

  # Regularization
  dropout: 0.1
  layer_norm_eps: 1e-12

  # Initialization
  initializer_range: 0.02
```

### 4. Grid Search Best Practices

```python
# Start with coarse grids, then refine
coarse_grid = ParameterGrid("coarse_search")
coarse_grid.add_parameter("optimizer.lr", [1e-5, 1e-4, 1e-3])

# Then refine around best results
fine_grid = ParameterGrid("fine_search")
fine_grid.add_parameter("optimizer.lr", [5e-5, 8e-5, 1e-4, 2e-4])
```

### 5. SLURM Resource Planning

```yaml
# Plan resources based on model size
small_model:
  slurm:
    mem: "16G"
    gres: "gpu:1"
    time_limit: "8:00:00"

large_model:
  slurm:
    mem: "64G"
    gres: "gpu:4"
    time_limit: "48:00:00"
```

### 6. Configuration Validation

```python
# Always validate configurations before submission
try:
    config = config_manager.create_experiment_config(config_dict, validate=True)
except ValidationError as e:
    print(f"Fix configuration errors: {e}")
    return
```

## Backwards Compatibility

### Configuration Key Renames

For users upgrading from earlier versions, the following configuration keys have been renamed for clarity:

**Trainer Configuration:**

- `config.multi` → `config.train_loader_config` (for training loaders)
- `config.multi` → `config.val_loader_config` (for validation loaders, optional)

**Performance Configuration:**

- `performance.mixed_precision: "16-mixed"` → `performance.use_amp: true`
- `data.num_workers` → `performance.dataloader_num_workers`

**Example Migration:**

Old configuration:

```yaml
trainer:
  multi:
    sampling_strategy: "WEIGHTED"
    dataloader_weights: [0.7, 0.3]

performance:
  use_amp: false  # Was: mixed_precision: "16-mixed"

# Note: num_workers has moved to performance.dataloader_num_workers
```

New configuration:

```yaml
trainer:
  train_loader_config:
    sampling_strategy: "WEIGHTED"
    dataloader_weights: [0.7, 0.3]

  val_loader_config:  # Optional, separate validation config
    sampling_strategy: "SEQUENTIAL"

performance:
  use_amp: true
  dataloader_num_workers: 4
```

These changes align naming with runtime behavior and clarify the separation between training and validation configurations.

This configuration guide provides comprehensive information for effectively using the Model Training Framework's configuration system. For more examples, see the `examples/` directory.
