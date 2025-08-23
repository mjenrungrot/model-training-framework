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
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
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
  num_workers: 4
  preprocessing:
    tokenizer: "bert-base-uncased"
    max_length: 512
    padding: "max_length"
    truncation: true

# Optimizer configuration
optimizer:
  name: "adamw"
  learning_rate: 0.001
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
  log_level: "INFO"
  log_file: "training.log"
  console_log: true
  use_wandb: true
  wandb_project: "my_project"
  wandb_entity: "my_team"
  wandb_tags: ["experiment", "baseline"]
  log_every_n_steps: 10
  save_code: true

# Checkpoint configuration
checkpoint:
  save_every_n_epochs: 5
  save_top_k: 3
  monitor: "val_loss"
  mode: "min"
  checkpoint_dir: "./checkpoints"
  filename: "{epoch:02d}-{val_loss:.2f}"
  save_last: true
  auto_insert_metric_name: true

# Preemption configuration
preemption:
  timeout_minutes: 5
  grace_period_seconds: 60
  max_preemptions: 3
  checkpoint_on_preemption: true
  exit_on_max_preemptions: true

# Performance configuration
performance:
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true
  dataloader_drop_last: false
  mixed_precision: "16-mixed"
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
  epochs: 10
  batch_size: 32
  learning_rate: 0.001

data:
  dataset_name: "mnist"

optimizer:
  name: "adam"
```

## Parameter Grid Search

### Basic Grid Search

```yaml
# base_config.yaml
experiment_name: "grid_search_base"
model:
  name: "transformer"
  hidden_size: 256  # Will be overridden
training:
  epochs: 50
  learning_rate: 0.001  # Will be overridden
```

```python
# Python script for grid search
from model_training_framework.config import ParameterGridSearch, ParameterGrid

# Load base configuration
base_config = config_manager.load_config("base_config.yaml")

# Create grid search
grid_search = ParameterGridSearch(base_config)

# Add parameter grids
lr_grid = grid_search.create_grid("learning_rates")
lr_grid.add_parameter("training.learning_rate", [1e-4, 1e-3, 1e-2])
lr_grid.add_parameter("optimizer.weight_decay", [0.01, 0.1])

model_grid = grid_search.create_grid("model_sizes")
model_grid.add_parameter("model.hidden_size", [128, 256, 512])
model_grid.add_parameter("model.num_layers", [4, 6, 8])

# Generate experiments
experiments = list(grid_search.generate_experiments())
```

### Advanced Grid Search Configuration

```json
{
  "name": "advanced_grid_search",
  "description": "Complex parameter exploration",
  "base_config_path": "configs/base.yaml",
  "naming_strategy": "descriptive",
  "grids": [
    {
      "name": "architecture_search",
      "description": "Model architecture variations",
      "parameters": {
        "model.hidden_size": [256, 512, 1024],
        "model.num_layers": [6, 12, 24],
        "model.num_heads": [8, 16, 32]
      }
    },
    {
      "name": "optimization_search", 
      "description": "Optimization hyperparameters",
      "parameters": {
        "training.learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
        "optimizer.weight_decay": [0.01, 0.05, 0.1],
        "training.gradient_accumulation_steps": [1, 2, 4, 8]
      }
    },
    {
      "name": "regularization_search",
      "description": "Regularization techniques",
      "parameters": {
        "model.dropout": [0.0, 0.1, 0.2, 0.3],
        "training.max_grad_norm": [0.5, 1.0, 2.0],
        "scheduler.warmup_steps": [500, 1000, 2000]
      }
    }
  ],
  "constraints": {
    "max_combinations": 1000,
    "exclude_combinations": [
      {"model.hidden_size": 1024, "model.num_layers": 24}
    ]
  }
}
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

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Print environment info
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

# Run training
srun python -m model_training_framework.scripts.train \
    --config {{config_path}} \
    --experiment-name {{experiment_name}} \
    --output-dir {{output_dir}} \
    --distributed

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
export SLURM_ACCOUNT="research_group"

# Or in a .env file (if using python-dotenv)
DATA_ROOT=/scratch/datasets
WANDB_PROJECT=my_experiments
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
        "training.learning_rate": 0.0005,
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
  mixed_precision: "16-mixed"
```

```yaml
# overrides/gpu_settings.yaml
slurm:
  gres: "gpu:a100:2"
  mem: "64G"
  
performance:
  num_workers: 8
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

```
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
coarse_grid.add_parameter("training.learning_rate", [1e-5, 1e-4, 1e-3])

# Then refine around best results
fine_grid = ParameterGrid("fine_search")
fine_grid.add_parameter("training.learning_rate", [5e-5, 8e-5, 1e-4, 2e-4])
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

This configuration guide provides comprehensive information for effectively using the Model Training Framework's configuration system. For more examples, see the `examples/` directory.