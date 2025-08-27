# Model Training Framework

A comprehensive Python package for multi-dataloader machine learning model training with fault tolerance, SLURM integration, and deterministic scheduling.

## 🎯 Key Design: Multi-DataLoader-Only Architecture

**This framework is designed exclusively for multi-dataloader training.** Even single dataloader scenarios must use the multi-dataloader API with a list containing one loader. This unified design enables seamless scaling and consistent behavior across all use cases.

## Features

- **Multi-DataLoader Training**: Built-in support for training with multiple datasets simultaneously
- **Deterministic Scheduling**: ROUND_ROBIN, WEIGHTED, ALTERNATING, and SEQUENTIAL strategies
- **Fault-Tolerant Training**: Preemption-safe with instruction-level checkpointing and exact resume
- **SLURM Integration**: Seamless job launching and management on HPC clusters
- **Flexible Aggregation**: Multiple validation aggregation strategies for multi-loader scenarios
- **Distributed Training**: Multi-node and multi-GPU support via Lightning Fabric with DDP

## Installation

```bash
# Install from source
git clone https://github.com/example/model-training-framework.git
cd model-training-framework
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Or install specific extras
pip install -e ".[dev]"           # Development tools
pip install -e ".[tensorboard]"   # TensorBoard logging
pip install -e ".[wandb]"         # Weights & Biases integration
pip install -e ".[docs]"          # Documentation tools
```

### Core Dependencies

- `torch>=2.0.0` - PyTorch framework
- `lightning>=2.0.0` - PyTorch Lightning
- `lightning-fabric>=2.0.0` - Lightning Fabric for distributed training
- `tensorboard>=2.10.0` - TensorBoard logging
- `numpy>=1.21.0` - Numerical operations
- `pyyaml>=6.0` - Configuration files
- `gitpython>=3.1.0` - Git integration

## Quick Start

### 1. Single DataLoader (Using Multi-Loader API)

```python
from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
    EpochLengthPolicy,
)

# Configuration for single loader (still uses multi-loader config)
config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
        dataloader_names=["main"],  # Single name in list
    )
)

# Create trainer - note the list syntax
trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=[optimizer],  # Always a list
)

# Training step signature includes batch index and dataloader info
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    # batch_idx is 0-based within this dataloader for the epoch
    # dataloader_idx will be 0, dataloader_name will be "main"
    x, y = batch
    outputs = trainer.model(x)
    loss = torch.nn.functional.cross_entropy(outputs, y)
    return {"loss": loss}

trainer.set_training_step(training_step)

# Fit with single loader wrapped in list
trainer.fit(
    train_loaders=[train_loader],  # Single loader in list
    val_loaders=[val_loader],      # Single loader in list
    max_epochs=10
)
```

### 2. Multiple DataLoaders with Different Strategies

#### Round-Robin Strategy (Fair Alternation)

```python
from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
    EpochLengthPolicy,
    ValidationConfig,
    ValAggregation,
)

# Alternates between dataloaders: A, B, A, B, ...
config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.ROUND_ROBIN,
        epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
        dataloader_names=["dataset_a", "dataset_b", "dataset_c"],
    ),
    validation=ValidationConfig(
        aggregation=ValAggregation.MACRO_AVG_EQUAL_LOADERS,
    ),
)

trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=[optimizer],
)

# Training with multiple loaders
trainer.fit(
    train_loaders=[loader_a, loader_b, loader_c],
    val_loaders=[val_a, val_b, val_c],
    max_epochs=20,
)
```

#### Weighted Strategy (Importance-Based Sampling)

```python
# Minimal imports for copy‑paste
from model_training_framework.trainer import (
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
    EpochLengthPolicy,
    LoggingConfig,
)
# Sample based on dataset importance/size
config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.WEIGHTED,
        dataloader_weights=[0.5, 0.3, 0.2],  # 50%, 30%, 20%
        epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
        steps_per_epoch=1000,
        dataloader_names=["primary", "auxiliary", "synthetic"],
    ),
    logging=LoggingConfig(
        log_loader_proportions=True,  # Monitor actual sampling
    ),
)
```

#### Alternating Pattern (Custom Schedule)

```python
# Minimal imports for copy‑paste
from model_training_framework.trainer import (
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
)
# Define explicit pattern: 2x A, 1x B, 1x C, repeat
config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.ALTERNATING,
        alternating_pattern=[0, 0, 1, 2],  # Indices into loader list
        burst_size=3,  # Take 3 batches at a time
        dataloader_names=["main", "augmented", "hard_negatives"],
    ),
)
```

### 3. Validation Aggregation Strategies

```python
from model_training_framework.trainer import (
    GenericTrainerConfig,
    ValidationConfig,
    ValAggregation,
)

# Micro-average: Weight by number of samples
config = GenericTrainerConfig(
    validation=ValidationConfig(
        aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
        per_loader_metrics=True,  # Track per-loader metrics
        global_metrics=True,      # Also track aggregated
    ),
)

# Macro-average: Equal weight to each loader
config = GenericTrainerConfig(
    validation=ValidationConfig(
        aggregation=ValAggregation.MACRO_AVG_EQUAL_LOADERS,
    ),
)

# Per-loader tracking for multi-task
config = GenericTrainerConfig(
    validation=ValidationConfig(
        aggregation=ValAggregation.PRIMARY_METRIC_PER_LOADER,
    ),
)
```

### 4. Checkpoint and Resume

```python
# Minimal imports for copy‑paste
from model_training_framework.trainer import (
    GenericTrainerConfig,
    CheckpointConfig,
    FaultToleranceConfig,
)
# Checkpoint configuration
config = GenericTrainerConfig(
    checkpoint=CheckpointConfig(
        save_every_n_epochs=1,
        save_every_n_steps=500,
        max_checkpoints=5,
    ),
    fault_tolerance=FaultToleranceConfig(
        save_sampler_state=True,  # For exact resume
        save_dataset_state=True,
        verify_deterministic_resume=True,
    ),
)

# Resume from checkpoint
if checkpoint_path.exists():
    trainer.load_checkpoint(checkpoint_path)
    # Resumes from exact batch/sample
```

### 5. Advanced Multi-Loader Patterns

#### Multi-Task Learning

```python
# Different optimizers for different tasks
optimizers = [
    torch.optim.Adam(model.task_a_params(), lr=0.001),
    torch.optim.SGD(model.task_b_params(), lr=0.01),
]

config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.WEIGHTED,
        dataloader_weights=[0.6, 0.4],
        dataloader_names=["task_a", "task_b"],
    ),
    per_loader_optimizers={
        "task_a": {"optimizer_idx": 0, "loss_weight": 1.0},
        "task_b": {"optimizer_idx": 1, "loss_weight": 0.5},
    },
)

trainer = GenericTrainer(
    config=config,
    model=multi_task_model,
    optimizers=optimizers,  # Multiple optimizers
)
```

#### Curriculum Learning

```python
# Sequential processing with increasing difficulty
config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["easy", "medium", "hard"],
        epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
    ),
)

# Or use alternating pattern for gradual transition
config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.ALTERNATING,
        # More easy, fewer hard samples
        alternating_pattern=[0, 0, 0, 0, 1, 1, 2],
        dataloader_names=["easy", "medium", "hard"],
    ),
)
```

### 6. DDP with Multi-Loaders

```python
# DDP configuration for multi-loader training
from lightning.fabric import Fabric

from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
    DDPConfig,
)

fabric = Fabric(accelerator="gpu", devices=4, strategy="ddp")
fabric.launch()

config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.ROUND_ROBIN,
        dataloader_names=["shard_1", "shard_2"],
    ),
    ddp=DDPConfig(
        sync_schedules_across_ranks=True,
        validate_schedule_consistency=True,
    ),
)

# Fabric handles distributed setup
model, *optimizers = fabric.setup(model, *optimizers)
trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=optimizers,
    fabric=fabric,
)
```

## 📚 Examples

Comprehensive examples are available in the `demo/` directory:

### Beginner Examples (example1_beginner_local)

- **[basic_model_training.py](demo/example1_beginner_local/basic_model_training.py)**: Single dataloader with multi-loader API
- **[multi_loader_training.py](demo/example1_beginner_local/multi_loader_training.py)**: Multiple dataloaders with various strategies
- **[sample_dataset.py](demo/example1_beginner_local/data/sample_dataset.py)**: Example dataset implementation

### Intermediate HPC Examples (example2_intermediate_hpc)

- **[train_script.py](demo/example2_intermediate_hpc/train_script.py)**: Distributed training with DDP
- **[orchestrate.py](demo/example2_intermediate_hpc/orchestrate.py)**: SLURM job orchestration
- **[config.py](demo/example2_intermediate_hpc/config.py)**: Configuration management

### Production Examples (example3_production)

- **[train_script.py](demo/example3_production/train_script.py)**: Production training with fault tolerance
- **[orchestrate.py](demo/example3_production/orchestrate.py)**: Advanced job orchestration
- **[model.py](demo/example3_production/model.py)**: Model architecture examples
- **[data.py](demo/example3_production/data.py)**: Data pipeline implementation

### Configuration Examples

- **[multi_loader_config.yaml](demo/example1_beginner_local/config_examples/multi_loader_config.yaml)**: Complete multi-loader configuration
- **[simple_config.yaml](demo/example1_beginner_local/config_examples/simple_config.yaml)**: Basic configuration example

## 🔄 Migration Guide

### Migrating from Single-Loader to Multi-Loader API

**Old (Single-Loader) Pattern:**

```python
# ❌ Old pattern - no longer supported
trainer = Trainer(model, optimizer)
trainer.fit(train_loader, val_loader)
```

**New (Multi-Loader) Pattern:**

```python
# ✅ New pattern - required for all training
config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["main"],
    )
)
trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=[optimizer],  # List required
)
trainer.fit(
    train_loaders=[train_loader],  # List required
    val_loaders=[val_loader],      # List required
)
```

### Key Changes

1. **Always use lists for loaders**: `train_loaders=[loader]`
2. **Always use lists for optimizers**: `optimizers=[optimizer]`
3. **Require MultiDataLoaderConfig**: Even for single loader
4. **Training step signature changed**:

   ```python
   # Old: def training_step(batch)
   # New:
   def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
       pass
   ```

## 🎯 Use Cases

### When to Use Each Sampling Strategy

| Strategy | Use Case | Example |
|----------|----------|---------|
| **ROUND_ROBIN** | Fair representation from all datasets | Multi-domain training |
| **WEIGHTED** | Imbalanced datasets or importance-based | 70% main task, 30% auxiliary |
| **ALTERNATING** | Specific patterns needed | Curriculum learning patterns |
| **SEQUENTIAL** | Process datasets in order | Pretrain → Finetune → Adapt |

### Common Patterns

**Handling Imbalanced Datasets:**

```python
# Oversample minority class
config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.WEIGHTED,
        dataloader_weights=[0.2, 0.8],  # Inverse of actual sizes
        dataloader_names=["majority", "minority"],
    ),
)
```

**Multi-Domain Training:**

```python
# Equal representation from each domain
config = GenericTrainerConfig(
    multi=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.ROUND_ROBIN,
        dataloader_names=["domain_a", "domain_b", "domain_c"],
        cycle_short_loaders=True,  # Restart shorter loaders
    ),
)
```

## Project Structure

```text
your_project/
├── model_training_framework/    # This package
├── configs/                     # Configuration files
│   ├── base_config.yaml
│   └── experiment_configs/
├── experiments/                 # Experiment outputs
├── scripts/                     # Training scripts
│   └── train.py
├── slurm_template.txt          # SLURM batch template
└── requirements.txt
```

## Configuration System

### Experiment Configuration

```yaml
# configs/base_config.yaml
experiment_name: "my_experiment"

model:
  type: "resnet50"
  num_classes: 1000
  dropout: 0.1

training:
  max_epochs: 100
  batch_size: 32
  gradient_accumulation_steps: 1

optimizer:
  type: "adamw"
  lr: 1e-4
  weight_decay: 0.01

data:
  dataset_name: "imagenet"
  batch_size: 32
  num_workers: 4

slurm:
  account: "realitylab"
  partition: "gpu"
  nodes: 1
  gpus_per_node: 1
  time: "12:00:00"

logging:
  use_wandb: true
  wandb_project: "my_project"
```

### Parameter Grid Search

```python
from model_training_framework.config import ParameterGrid

# Define parameter search spaces
grid1 = ParameterGrid("optimization_search")
grid1.add_parameter("optimizer.lr", [1e-4, 5e-4, 1e-3])
grid1.add_parameter("optimizer.weight_decay", [0.0, 0.01])

grid2 = ParameterGrid("architecture_search")
grid2.add_parameter("model.dropout", [0.1, 0.2, 0.3])
grid2.add_parameter("model.hidden_size", [256, 512, 1024])

# Execute grid search
framework.run_grid_search(
    base_config="configs/base.yaml",
    parameter_grids=[grid1, grid2],
    script_path="scripts/train.py"
)
```

## Training Scripts

### Basic Training Script

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

from model_training_framework import ModelTrainingFramework
from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
)

import torch
import torch.nn as nn
from lightning.fabric import Fabric

def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    """Define training step logic."""
    x, y = batch
    pred = trainer.model(x)
    loss = nn.functional.cross_entropy(pred, y)
    return {"loss": loss}

def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    """Define validation step logic."""
    x, y = batch
    pred = trainer.model(x)
    loss = nn.functional.cross_entropy(pred, y)
    acc = (pred.argmax(dim=1) == y).float().mean()
    return {"loss": loss, "accuracy": acc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration file")
    args = parser.parse_args()

    # Load configuration
    framework = ModelTrainingFramework()
    config = framework.load_experiment_config(args.config)

    # Setup distributed training
    fabric = Fabric(devices="auto", accelerator="auto")

    # Create model, optimizer, data loaders
    model = create_model(config.model)
    optimizer = create_optimizer(model, config.optimizer)
    train_loader, val_loader = create_data_loaders(config.data)

    # Setup with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)

    # Create trainer with multi-loader config
    trainer_config = GenericTrainerConfig(
        multi=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            dataloader_names=["main"],
        )
    )
    trainer = GenericTrainer(
        config=trainer_config,
        fabric=fabric,
        model=model,
        optimizers=[optimizer]  # Always use list
    )

    # Set step functions
    trainer.set_training_step(training_step)
    trainer.set_validation_step(validation_step)

    # Train model with lists of loaders
    trainer.fit(
        train_loaders=[train_loader],  # Always use list
        val_loaders=[val_loader],      # Always use list
        max_epochs=config.training.max_epochs
    )

if __name__ == "__main__":
    main()
```

## SLURM Integration

### SLURM Template

Create `slurm_template.txt` in your project root:

```bash
#!/bin/bash

#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodes={{NODES}}
#SBATCH --ntasks-per-node={{NTASKS_PER_NODE}}
#SBATCH --gpus-per-node={{GPUS_PER_NODE}}
#SBATCH --cpus-per-task={{CPUS_PER_TASK}}
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}
#SBATCH --output={{OUTPUT_FILE}}
#SBATCH --error={{ERROR_FILE}}
#SBATCH --requeue={{REQUEUE}}

# Environment setup
module load python/3.9
module load cuda/11.8

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: {{EXPERIMENT_NAME}}"
echo "Node: $SLURM_NODEID"

# Run training
cd {{PROJECT_ROOT}}
python {{SCRIPT_PATH}} {{CONFIG_NAME}}
```

### Job Submission

```python
# Submit single experiment
result = framework.run_single_experiment(
    config=config,
    script_path="scripts/train.py",
    execution_mode=ExecutionMode.SLURM
)

print(f"Submitted job: {result.job_id}")

# Submit batch of experiments
results = framework.run_grid_search(
    base_config=base_config,
    parameter_grids=grids,
    script_path="scripts/train.py",
    max_concurrent_jobs=10
)

print(f"Submitted {results.success_count} jobs successfully")
```

## Advanced Features

### Hooks System

The framework provides a comprehensive hooks system for injecting custom behavior:

```python
from model_training_framework.trainer import TrainerHooks, GenericTrainer

class CustomHook(TrainerHooks):
    def on_epoch_start(self, trainer, epoch):
        print(f"Starting epoch {epoch}")

    def on_train_batch_end(self, trainer, batch, dataloader_idx, dataloader_name, metrics):
        if metrics["loss"] > 10:
            print(f"High loss detected: {metrics['loss']}")

    def on_validation_end(self, trainer, epoch, metrics):
        print(f"Validation metrics: {metrics}")

# Register hooks with trainer
trainer = GenericTrainer(config=config, model=model, optimizers=[optimizer])
trainer.hook_manager.register_hook(CustomHook())
```

Available hooks:

- `on_train_start/end` - Training lifecycle
- `on_epoch_start/end` - Epoch boundaries
- `on_train_batch_start/end` - Training batches
- `on_validation_start/end` - Validation phases
- `on_before/after_backward` - Gradient computation
- `on_before/after_optimizer_step` - Optimization
- `on_checkpoint_save/load` - Checkpointing
- `on_gradient_clip` - Gradient clipping

### Metrics Management

Advanced metrics tracking with per-loader and global aggregation:

```python
from model_training_framework.trainer import MetricsManager, AggregationStrategy

# Configure metrics aggregation
config = GenericTrainerConfig(
    metrics=MetricsConfig(
        aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        track_proportions=True,
        per_loader_metrics=True,
    )
)

# Access metrics during training
def on_epoch_end(trainer, epoch, metrics):
    # Per-loader metrics
    loader_a_loss = metrics.get("train/dl_loader_a/loss")

    # Global aggregated metrics
    global_loss = metrics.get("train/loss")

    # Loader proportions
    proportions = trainer.metrics_manager.get_loader_proportions()
```

### Enhanced Logging

Multiple logging backends with unified interface:

```python
from model_training_framework.trainer import (
    WandBLogger, TensorBoardLogger, ConsoleLogger, CompositeLogger
)

# Single logger
logger = TensorBoardLogger(log_dir="./tb_logs")

# Multiple loggers
loggers = CompositeLogger([
    ConsoleLogger(log_level="INFO"),
    TensorBoardLogger(log_dir="./tb_logs"),
    WandBLogger(project="my_project", entity="my_team")
])

config = GenericTrainerConfig(
    logging=LoggingConfig(
        logger=loggers,
        log_every_n_steps=10,
        log_loader_proportions=True,
    )
)
```

### Fault-Tolerant Training

The framework provides preemption-safe training with automatic checkpointing:

```python
# Training automatically handles SIGUSR1 signals
# Checkpoints are saved with instruction-level granularity
# Resume from latest checkpoint on restart

trainer_config = GenericTrainerConfig(
    checkpoint=CheckpointConfig(
        save_every_n_epochs=1,
        save_rng=True,  # For deterministic resume
        max_checkpoints=5
    ),
    preemption=PreemptionConfig(
        signal=signal.SIGUSR1,
        max_checkpoint_sec=300.0,
        requeue_job=True
    )
)
```

### Git Integration

Automatic git branch management for experiment isolation:

```python
# Creates temporary branches for each experiment
# Format: slurm-job/<experiment_name>/<timestamp>/<commit_hash>
result = framework.run_single_experiment(
    config=config,
    script_path="scripts/train.py",
    use_git_branch=True  # Enable git integration
)
```

### Experiment Tracking

Integration with Weights & Biases and other tracking systems:

```python
config = {
    "logging": {
        "use_wandb": True,
        "wandb_project": "my_project",
        "wandb_entity": "my_team",
        "log_scalars_every_n_steps": 50
    }
}
```

## API Reference

### Core Classes

- **GenericTrainer**: Multi-dataloader training engine with fault tolerance
- **MultiDataLoaderManager**: Manages multiple dataloaders with various sampling strategies
- **MetricsManager**: Advanced metrics tracking and aggregation
- **HookManager**: Training lifecycle hooks system
- **CheckpointManager**: Checkpoint saving and loading with exact resume

### Configuration Classes

- **GenericTrainerConfig**: Main trainer configuration
- **MultiDataLoaderConfig**: Multi-loader sampling configuration
- **ValidationConfig**: Validation aggregation settings
- **CheckpointConfig**: Checkpointing behavior
- **LoggingConfig**: Logging and monitoring settings

### Logging & Monitoring

- **WandBLogger**: Weights & Biases integration
- **TensorBoardLogger**: TensorBoard logging
- **ConsoleLogger**: Structured console output
- **CompositeLogger**: Multiple logger backends

### SLURM & Orchestration

- **ModelTrainingFramework**: Main orchestration class
- **SLURMLauncher**: Job submission and management
- **ParameterGrid**: Grid search configuration
- **GitManager**: Experiment branch management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- Documentation: <https://model-training-framework.readthedocs.io/>
- Issues: <https://github.com/example/model-training-framework/issues>
- Discussions: <https://github.com/example/model-training-framework/discussions>
