# Model Training Framework

A comprehensive Python package for machine learning model training, job launching, and configuration management with SLURM integration.

## Features

- **Fault-Tolerant Training**: Preemption-safe training with instruction-level checkpointing
- **SLURM Integration**: Seamless job launching and management on HPC clusters
- **Parameter Grid Search**: Automatic enumeration and naming of experiment configurations
- **Configuration Management**: Structured configuration system with validation
- **Experiment Tracking**: Comprehensive logging and reproducibility features
- **Distributed Training**: Multi-node and multi-GPU support via Lightning Fabric

## Installation

```bash
# Install from source
git clone https://github.com/example/model-training-framework.git
cd model-training-framework
pip install -e .

# Install with optional dependencies
pip install -e ".[dev,wandb,docs]"
```

## Quick Start

### 1. Basic Usage

```python
from model_training_framework import ModelTrainingFramework
from pathlib import Path

# Initialize framework
framework = ModelTrainingFramework(project_root=Path("/path/to/project"))

# Create experiment configuration
config = framework.create_experiment({
    "experiment_name": "resnet_baseline",
    "model": {
        "type": "resnet50",
        "num_classes": 1000
    },
    "training": {
        "max_epochs": 100,
        "batch_size": 64
    },
    "optimizer": {
        "type": "adamw",
        "lr": 1e-3
    }
})

# Execute experiment
result = framework.run_single_experiment(
    config=config,
    script_path="scripts/train.py"
)
```

### 2. Parameter Grid Search

```python
# Create parameter grid
grid = framework.create_parameter_grid(
    name="learning_rate_search",
    parameters={
        "optimizer.lr": [1e-4, 5e-4, 1e-3, 5e-3],
        "optimizer.weight_decay": [0.0, 0.01, 0.1],
        "training.batch_size": [32, 64, 128]
    }
)

# Execute grid search
results = framework.run_grid_search(
    base_config="configs/base_config.yaml",
    parameter_grids=[grid],
    script_path="scripts/train.py"
)
```

### 3. Job Monitoring

```python
# Monitor running jobs
status = framework.monitor_jobs()
print(f"Active jobs: {len(status['active_jobs'])}")

# Wait for specific experiments to complete
completed = framework.wait_for_experiments(
    job_ids=["12345", "12346", "12347"],
    timeout=3600  # 1 hour
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
from model_training_framework.trainer import GenericTrainer, GenericTrainerConfig

import torch
import torch.nn as nn
from lightning.fabric import Fabric

def training_step(trainer, batch, micro_step):
    """Define training step logic."""
    x, y = batch
    pred = trainer.model(x)
    loss = nn.functional.cross_entropy(pred, y)
    return {"loss": loss}

def validation_step(trainer, batch, batch_idx):
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

    # Create trainer
    trainer_config = GenericTrainerConfig()
    trainer = GenericTrainer(
        config=trainer_config,
        fabric=fabric,
        model=model,
        optimizer=optimizer
    )

    # Set step functions
    trainer.set_training_step(training_step)
    trainer.set_validation_step(validation_step)

    # Train model
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
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

- **ModelTrainingFramework**: Main orchestration class
- **ExperimentConfig**: Experiment configuration schema
- **ParameterGrid**: Parameter search space definition
- **GenericTrainer**: Fault-tolerant training engine
- **SLURMLauncher**: Job submission and management

### Configuration Management

- **ConfigurationManager**: Configuration loading and validation
- **ConfigValidator**: Configuration validation
- **ParameterGridSearch**: Grid search execution

### SLURM Integration — API

- **SLURMLauncher**: Job launcher and batch submission
- **SLURMJobMonitor**: Job status monitoring
- **GitManager**: Git operations and branch management

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
