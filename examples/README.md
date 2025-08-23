# Examples

This directory contains comprehensive examples demonstrating how to use the Model Training Framework in various scenarios.

## Overview

The examples are designed to be run independently and showcase different aspects of the framework:

1. **Basic Training** - Simple training setup and configuration
2. **Parameter Grid Search** - Automated hyperparameter exploration
3. **SLURM Batch Jobs** - Large-scale job submission and management
4. **Custom Trainers** - Extending the framework for specialized use cases
5. **Preemption Handling** - Fault-tolerant training with automatic recovery

## Running Examples

### Prerequisites

Make sure you have the framework installed and dependencies available:

```bash
# Install the framework
cd /path/to/model_training_framework
pip install -e .

# Install dependencies
pip install torch lightning-fabric pyyaml gitpython
```

### Example Descriptions

#### 1. Basic Training (`basic_training.py`)

**Purpose**: Demonstrates the simplest way to set up and run training with the framework.

**Features**:
- Configuration management
- Simple model and trainer setup
- Basic logging and checkpointing

**Run**:
```bash
python examples/basic_training.py
```

**What it does**:
- Creates a simple MLP model for MNIST-like data
- Sets up training configuration programmatically
- Runs training for 10 epochs with checkpointing
- Demonstrates basic framework usage patterns

#### 2. Parameter Grid Search (`grid_search.py`)

**Purpose**: Shows how to use the parameter grid search functionality for hyperparameter exploration.

**Features**:
- Multiple parameter grids
- Different naming strategies
- Experiment generation and validation
- Advanced grid search configuration

**Run**:
```bash
python examples/grid_search.py
```

**What it does**:
- Creates multiple parameter grids (learning rates, model sizes, training configs)
- Demonstrates different naming strategies (hash-based, descriptive)
- Generates all parameter combinations
- Shows validation and export functionality
- Runs in dry-run mode to show what would be submitted

#### 3. SLURM Batch Jobs (`slurm_batch.py`)

**Purpose**: Demonstrates submitting multiple training jobs to SLURM with automatic management.

**Features**:
- Batch job submission
- SLURM template generation
- Git branch management
- Job monitoring and status checking

**Run**:
```bash
python examples/slurm_batch.py
```

**What it does**:
- Creates multiple experiment configurations
- Sets up SLURM launcher with templates
- Demonstrates dry-run validation
- Shows git integration for experiment isolation
- Provides examples of job monitoring

**Note**: The actual SLURM submission is commented out for safety. Uncomment the relevant sections to submit real jobs.

#### 4. Custom Trainers (`custom_trainer.py`)

**Purpose**: Shows how to extend the GenericTrainer for specialized training scenarios.

**Features**:
- Multi-task learning implementation
- Custom training and validation steps
- Training callbacks (early stopping, LR scheduling)
- Advanced metric tracking

**Run**:
```bash
python examples/custom_trainer.py
```

**What it does**:
- Implements a multi-task model with two classification heads
- Creates a custom trainer with task-specific loss weighting
- Demonstrates training callbacks for early stopping and LR scheduling
- Shows advanced metrics tracking and logging

#### 5. Preemption Handling (`preemption_handling.py`)

**Purpose**: Demonstrates the framework's fault-tolerant training capabilities with SLURM preemption.

**Features**:
- Preemption signal handling
- Automatic checkpoint saving and loading
- Deterministic resume with RNG state
- Extended training simulation

**Run**:
```bash
python examples/preemption_handling.py
```

**What it does**:
- Sets up extended training that takes several minutes
- Simulates SLURM preemption signal (SIGUSR1) after 30 seconds
- Demonstrates automatic checkpoint saving on preemption
- Shows checkpoint inspection and resume capabilities
- Verifies deterministic resume with RNG state preservation

## Configuration Examples

### Base Configuration Template

All examples use similar base configurations. Here's a template you can adapt:

```yaml
experiment_name: "your_experiment"
description: "Description of your experiment"

model:
  name: "your_model_type"
  hidden_size: 256
  # ... other model parameters

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  # ... other training parameters

data:
  dataset_name: "your_dataset"
  train_split: "train"
  val_split: "validation"
  # ... other data parameters

optimizer:
  name: "adamw"
  weight_decay: 0.01
  # ... other optimizer parameters

scheduler:
  name: "cosine"
  warmup_steps: 1000
  # ... other scheduler parameters

slurm:
  job_name: "training_job"
  time_limit: "24:00:00"
  nodes: 1
  ntasks_per_node: 1
  cpus_per_task: 4
  mem: "32G"
  gres: "gpu:1"
  # ... other SLURM parameters

logging:
  log_level: "INFO"
  use_wandb: true
  wandb_project: "your_project"
  # ... other logging parameters
```

### SLURM Template

For SLURM examples, you'll need a template file (`scripts/slurm_template.txt`):

```bash
#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --time={{time_limit}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --cpus-per-task={{cpus_per_task}}
#SBATCH --mem={{mem}}
#SBATCH --gres={{gres}}
#SBATCH --output={{output_dir}}/{{experiment_name}}.out
#SBATCH --error={{output_dir}}/{{experiment_name}}.err

# Environment setup
source activate your_env
cd {{project_root}}

# Run training
python -m model_training_framework.scripts.train --config {{config_path}}
```

## Customization Guide

### Creating Your Own Examples

1. **Copy a base example** that's closest to your use case
2. **Modify the model** and data loading for your domain
3. **Adjust configuration** parameters for your requirements
4. **Add custom callbacks** if needed for specialized functionality
5. **Test locally** before submitting to SLURM

### Common Patterns

#### Configuration Management
```python
from model_training_framework import ModelTrainingFramework

framework = ModelTrainingFramework(
    project_root="/path/to/project",
    config_dir="/path/to/configs"
)

config_manager = framework.get_config_manager()
experiment_config = config_manager.create_experiment_config(config_dict)
```

#### Custom Training Steps
```python
class YourTrainer(GenericTrainer):
    def training_step(self, batch, batch_idx):
        # Your training logic here
        return {'loss': loss, 'accuracy': accuracy}
    
    def validation_step(self, batch, batch_idx):
        # Your validation logic here
        return {'val_loss': val_loss, 'val_accuracy': val_accuracy}
```

#### Parameter Grid Search
```python
from model_training_framework.config import ParameterGridSearch, ParameterGrid

grid_search = ParameterGridSearch(base_config)
grid = grid_search.create_grid("my_grid")
grid.add_parameter("training.learning_rate", [1e-4, 1e-3, 1e-2])

experiments = list(grid_search.generate_experiments())
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the framework is installed with `pip install -e .`
2. **Missing Dependencies**: Install with `pip install torch lightning-fabric pyyaml gitpython`
3. **SLURM Errors**: Check your SLURM template and cluster configuration
4. **Checkpoint Issues**: Ensure checkpoint directories have write permissions

### Getting Help

- Check the main README for detailed documentation
- Look at test files for additional usage patterns
- Run examples with the `--help` flag where available
- Check error messages and logs for specific issues

## Best Practices

1. **Start Simple**: Begin with `basic_training.py` and gradually add complexity
2. **Test Locally**: Always test configurations locally before submitting to SLURM
3. **Use Dry Runs**: Use `ExecutionMode.DRY_RUN` to validate configurations
4. **Monitor Resources**: Check memory and GPU usage to optimize resource allocation
5. **Save Frequently**: Use frequent checkpointing for long-running experiments
6. **Version Control**: Keep track of experiment configurations and results

## Next Steps

After running these examples:

1. Adapt them for your specific use case and data
2. Create your own custom trainers and callbacks
3. Set up proper configuration management for your project
4. Integrate with your existing training pipelines
5. Scale up to larger parameter searches and production workloads