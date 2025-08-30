# Extended Quick Start Guide

This guide provides comprehensive examples to get you started with the Model Training Framework.

## Table of Contents

1. [Basic Training](#basic-training)
2. [Multiple DataLoaders](#multiple-dataloaders)
3. [Grid Search](#grid-search)
4. [SLURM Submission](#slurm-submission)
5. [Checkpoint and Resume](#checkpoint-and-resume)
6. [Complete Example](#complete-example)

## Basic Training

### Minimal Example

```python
from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Create simple dataset
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32)
val_loader = DataLoader(dataset, batch_size=32)

# Create model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Configure trainer (multi-loader API required)
config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["main"],
    )
)

# Create trainer
trainer = GenericTrainer(
    config=config,
    model=model,
    optimizers=[optimizer]  # Always a list
)

# Define training step
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch
    outputs = trainer.model(x)
    loss = nn.functional.cross_entropy(outputs, y)
    return {"loss": loss}

# Define validation step
def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch
    with torch.no_grad():
        outputs = trainer.model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        acc = (outputs.argmax(1) == y).float().mean()
    return {"loss": loss, "accuracy": acc}

# Set step functions
trainer.set_training_step(training_step)
trainer.set_validation_step(validation_step)

# Train
trainer.fit(
    train_loaders=[train_loader],  # Always a list
    val_loaders=[val_loader],      # Always a list
    max_epochs=10
)
```

### With Logging and Checkpointing

```python
from model_training_framework.trainer import (
    GenericTrainerConfig,
    CheckpointConfig,
    LoggingConfig,
)

# Enhanced configuration
config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["main"],
    ),
    val_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["validation"],
    ),
    checkpoint=CheckpointConfig(
        save_every_n_epochs=1,
        save_every_n_steps=100,
        max_checkpoints=3,
        monitor_metric="val/loss",
        monitor_mode="min",
    ),
    logging=LoggingConfig(
        logger_type="tensorboard",
        tensorboard_dir="./logs",
        log_scalars_every_n_steps=10,
    )
)

trainer = GenericTrainer(config, model, [optimizer])
# ... rest of training code
```

## Multiple DataLoaders

### Round-Robin Strategy

```python
from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
    EpochLengthPolicy,
)

# Create multiple datasets
dataset_a = TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
dataset_b = TensorDataset(torch.randn(500, 10), torch.randint(0, 2, (500,)))
dataset_c = TensorDataset(torch.randn(200, 10), torch.randint(0, 2, (200,)))

loader_a = DataLoader(dataset_a, batch_size=32)
loader_b = DataLoader(dataset_b, batch_size=32)
loader_c = DataLoader(dataset_c, batch_size=32)

# Configure round-robin sampling
config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.ROUND_ROBIN,
        dataloader_names=["dataset_a", "dataset_b", "dataset_c"],
        epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
    )
)

trainer = GenericTrainer(config, model, [optimizer])

# Training step with dataloader awareness
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch

    # Optional: dataloader-specific processing
    if dataloader_name == "dataset_c":
        # Apply special augmentation for dataset_c
        x = x + torch.randn_like(x) * 0.1

    outputs = trainer.model(x)
    loss = nn.functional.cross_entropy(outputs, y)

    # Log per-dataloader metrics
    return {
        "loss": loss,
        f"{dataloader_name}/loss": loss,
    }

trainer.set_training_step(training_step)
trainer.fit(
    train_loaders=[loader_a, loader_b, loader_c],
    val_loaders=[loader_a, loader_b, loader_c],
    max_epochs=10
)
```

### Weighted Sampling

```python
from model_training_framework.trainer import (
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
    EpochLengthPolicy,
    LoggingConfig,
)

# Configure weighted sampling (70% A, 20% B, 10% C)
config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.WEIGHTED,
        dataloader_weights=[0.7, 0.2, 0.1],
        dataloader_names=["primary", "auxiliary", "rare"],
        epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
        steps_per_epoch=500,
    ),
    val_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["validation"],
    ),
    logging=LoggingConfig(
        log_loader_proportions=True,  # Monitor actual sampling
    )
)
```

## Grid Search

### Setting Up Parameter Search

```python
from model_training_framework.config import (
    ParameterGridSearch,
    ParameterGrid,
    NamingStrategy,
)

# Base configuration
base_config = {
    "experiment_name": "hyperparameter_search",
    "model": {
        "type": "mlp",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "training": {
        "max_epochs": 20,
        "gradient_accumulation_steps": 1,
    },
    "data": {
        "dataset_name": "my_dataset",
        "batch_size": 32,
    },
    "performance": {
        "dataloader_num_workers": 4,
    },
    "optimizer": {
        "type": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.01,
    },
    "logging": {
        "use_wandb": False,
    }
}

# Create grid search
gs = ParameterGridSearch(base_config)
gs.set_naming_strategy(NamingStrategy.PARAMETER_BASED)

# Define search grid
grid = ParameterGrid("hyperparameter_sweep")
grid.add_parameter("optimizer.lr", [1e-4, 5e-4, 1e-3, 5e-3])
grid.add_parameter("model.hidden_size", [64, 128, 256])
grid.add_parameter("model.dropout", [0.1, 0.2, 0.3])
grid.add_parameter("training.gradient_accumulation_steps", [1, 2, 4])

gs.add_grid(grid)

# Generate experiments
experiments = list(gs.generate_experiments())
print(f"Generated {len(experiments)} experiments")

# Save grid configuration
from pathlib import Path
output_dir = Path("experiments/grid_search")
output_dir.mkdir(parents=True, exist_ok=True)
gs.save_grid_config(output_dir / "grid_config.json")
gs.save_summary(output_dir / "summary.txt")
```

### Running Grid Search Locally

```python
import torch.nn as nn
import torch.optim as optim
from model_training_framework.config import ConfigurationManager

def create_model(config):
    """Create model from configuration."""
    model_type = config.get("type", "mlp")
    if model_type == "mlp":
        return nn.Sequential(
            nn.Linear(784, config["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(config["hidden_size"], 10)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_optimizer(model, config):
    """Create optimizer from configuration."""
    opt_type = config.get("type", "adamw")
    if opt_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.01)
        )
    elif opt_type == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config.get("momentum", 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

# Run all experiments
for exp in experiments:
    print(f"Running: {exp.experiment_name}")

    # Create model and optimizer based on config
    model = create_model(exp.model)
    optimizer = create_optimizer(model, exp.optimizer)

    # Create trainer
    trainer_config = GenericTrainerConfig(
        train_loader_config=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            dataloader_names=["main"],
        ),
        val_loader_config=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            dataloader_names=["validation"],
        ),
        checkpoint=CheckpointConfig(
            root_dir=f"./checkpoints/{exp.experiment_name}",
        )
    )

    trainer = GenericTrainer(trainer_config, model, [optimizer])
    trainer.set_training_step(training_step)
    trainer.set_validation_step(validation_step)

    # Train
    trainer.fit(
        train_loaders=[train_loader],
        val_loaders=[val_loader],
        max_epochs=exp.training.max_epochs
    )

    # Save results
    results = {
        "experiment": exp.experiment_name,
        "final_loss": trainer.current_val_loss,
        "best_loss": trainer.best_val_loss,
    }
    # Save results
    import json
    with open(output_dir / f"{exp.experiment_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
```

## SLURM Submission

### Basic SLURM Configuration

```python
from model_training_framework import ModelTrainingFramework
from model_training_framework.slurm import SLURMLauncher

# Initialize framework
framework = ModelTrainingFramework(
    project_root=".",
    slurm_template_path="slurm_template.txt"
)

# Configuration with SLURM settings
config = {
    "experiment_name": "slurm_experiment",
    "model": {...},
    "training": {...},
    "slurm": {
        "account": "my_account",
        "partition": "gpu",
        "nodes": 1,
        "gpus_per_node": 1,
        "cpus_per_task": 8,
        "mem": "32G",
        "time": "12:00:00",
    }
}

# Submit single experiment
result = framework.run_single_experiment(
    config=config,
    script_path="train.py",
    execution_mode="slurm"
)

print(f"Submitted job: {result.job_id}")
```

### Batch Submission with Grid Search

```python
# Submit all grid search experiments to SLURM
launcher = SLURMLauncher(
    template_path="slurm_template.txt",
    project_root=".",
    experiments_dir="./experiments"
)

result = launcher.submit_experiment_batch(
    experiments=experiments,  # From grid search
    script_path="train.py",
    max_concurrent=10,  # Limit concurrent jobs
    dry_run=False
)

print(f"Submitted {result.success_count}/{result.total_experiments} jobs")
for job in result.job_results:
    if job.success:
        print(f"  {job.experiment_name}: Job ID {job.job_id}")
    else:
        print(f"  {job.experiment_name}: Failed - {job.error}")
```

### SLURM Template

Create `slurm_template.txt`:

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
#SBATCH --signal=USR1@60
#SBATCH --requeue

# Load modules
module load python/3.9
module load cuda/11.8

# Activate environment
source .venv/bin/activate

# Run training
cd {{PROJECT_ROOT}}
python {{SCRIPT_PATH}} {{CONFIG_NAME}}
```

## Checkpoint and Resume

### Automatic Checkpointing

```python
config = GenericTrainerConfig(
    checkpoint=CheckpointConfig(
        save_every_n_epochs=1,
        save_every_n_steps=500,
        max_checkpoints=5,
        save_last=True,
        save_best=True,
        monitor="val/loss",
        mode="min",
    ),
    fault_tolerance=FaultToleranceConfig(
        save_sampler_state=True,
        save_dataset_state=True,
        verify_deterministic_resume=True,
    )
)

trainer = GenericTrainer(config, model, [optimizer])

# Automatic resume if checkpoint exists
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    latest = trainer.checkpoint_manager.get_latest_checkpoint()
    if latest:
        trainer.load_checkpoint(latest)
        print(f"Resumed from epoch {trainer.current_epoch}")
```

### Manual Checkpoint Management

```python
# Save checkpoint manually
checkpoint_path = trainer.save_checkpoint()
print(f"Saved checkpoint to {checkpoint_path}")

# Load specific checkpoint
trainer.load_checkpoint("checkpoints/epoch_5.ckpt")

# Save emergency checkpoint (on preemption)
import signal

def handle_preemption(signum, frame):
    print("Preemption signal received, saving checkpoint...")
    trainer.save_checkpoint(emergency=True)
    sys.exit(0)

signal.signal(signal.SIGUSR1, handle_preemption)
```

## Complete Example

### Production Training Script

```python
#!/usr/bin/env python3
"""
Complete training script with all features.
"""

import argparse
import logging
from pathlib import Path
import signal
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
    CheckpointConfig,
    LoggingConfig,
    FaultToleranceConfig,
    ValidationConfig,
    ValAggregation,
)
from model_training_framework.config import ConfigurationManager


def create_model(config):
    """Create model from config."""
    return nn.Sequential(
        nn.Linear(config["input_size"], config["hidden_size"]),
        nn.ReLU(),
        nn.Dropout(config.get("dropout", 0.1)),
        nn.Linear(config["hidden_size"], config["output_size"])
    )


def create_dataloaders(config):
    """Create train and validation dataloaders with optimal settings."""
    # Your dataset creation logic
    # Example with TensorDataset
    train_data = torch.randn(1000, 28 * 28)
    train_labels = torch.randint(0, 10, (1000,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    val_data = torch.randn(200, 28 * 28)
    val_labels = torch.randint(0, 10, (200,))
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)

    # Optimized DataLoader settings
    # Use performance.dataloader_num_workers for consistency with CONFIGURATION.md
    perf = config.get("performance", {})
    nw = int(perf.get("dataloader_num_workers", 4))

    tl_kwargs = dict(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if nw > 0:
        tl_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(**tl_kwargs)

    vl_kwargs = dict(
        dataset=val_dataset,
        batch_size=config["batch_size"] * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    if nw > 0:
        vl_kwargs.update(persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(**vl_kwargs)

    return train_loader, val_loader


def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    """Single training step."""
    device = next(trainer.model.parameters()).device
    x, y = batch

    # Move to device; use non_blocking if pin_memory is enabled
    nb = getattr(trainer.config.performance, "pin_memory", False) and device.type == "cuda"
    x = x.to(device, non_blocking=nb)
    y = y.to(device, non_blocking=nb)

    use_amp = getattr(trainer.config.performance, "use_amp", False) and device.type == "cuda"
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        outputs = trainer.model(x)
        loss = nn.functional.cross_entropy(outputs, y)

    # Compute additional metrics
    with torch.no_grad():
        acc = (outputs.argmax(1) == y).float().mean()

    return {
        "loss": loss,
        "accuracy": acc,
        f"{dataloader_name}/loss": loss,
        f"{dataloader_name}/accuracy": acc,
    }


def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    """Single validation step."""
    device = next(trainer.model.parameters()).device
    x, y = batch

    # Move to device; use non_blocking if pin_memory is enabled
    nb = getattr(trainer.config.performance, "pin_memory", False) and device.type == "cuda"
    x = x.to(device, non_blocking=nb)
    y = y.to(device, non_blocking=nb)

    with torch.no_grad():
        outputs = trainer.model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        acc = (outputs.argmax(1) == y).float().mean()

    return {
        "loss": loss,
        "accuracy": acc,
        f"{dataloader_name}/loss": loss,
        f"{dataloader_name}/accuracy": acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--resume", help="Resume from checkpoint", action="store_true")
    parser.add_argument("--checkpoint", help="Path to checkpoint file")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Load configuration
    config_manager = ConfigurationManager(project_root=".")
    config = config_manager.load_config(args.config)

    # Create model
    model = create_model(config["model"])

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"].get("weight_decay", 0.01)
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config["data"])

    # Trainer configuration
    trainer_config = GenericTrainerConfig(
        train_loader_config=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            dataloader_names=["main"],
        ),
        val_loader_config=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            dataloader_names=["validation"],
        ),
        checkpoint=CheckpointConfig(
            save_every_n_epochs=1,
            save_every_n_steps=500,
            max_checkpoints=3,
            root_dir=f"./checkpoints/{config['experiment_name']}",
        ),
        logging=LoggingConfig(
            logger_type="composite",
            composite_loggers=["console", "tensorboard"],
            tensorboard_dir=f"./logs/{config['experiment_name']}",
            log_scalars_every_n_steps=10,
        ),
        fault_tolerance=FaultToleranceConfig(
            save_sampler_state=True,
            verify_deterministic_resume=True,
        ),
        validation=ValidationConfig(
            aggregation=ValAggregation.MACRO_AVG_EQUAL_LOADERS,
            per_loader_metrics=True,
        )
    )

    # Create trainer
    trainer = GenericTrainer(
        config=trainer_config,
        model=model,
        optimizers=[optimizer]
    )

    # Set step functions
    trainer.set_training_step(training_step)
    trainer.set_validation_step(validation_step)

    # Handle preemption
    def handle_signal(signum, frame):
        logger.warning("Received preemption signal, saving checkpoint...")
        trainer.save_checkpoint(emergency=True)
        sys.exit(0)

    signal.signal(signal.SIGUSR1, handle_signal)

    # Resume from checkpoint if requested
    if args.resume:
        checkpoint_path = args.checkpoint or trainer.checkpoint_manager.get_latest_checkpoint()
        if checkpoint_path and Path(checkpoint_path).exists():
            trainer.load_checkpoint(checkpoint_path)
            logger.info(f"Resumed from {checkpoint_path}")
        else:
            logger.warning("No checkpoint found to resume from")

    # Train
    logger.info(f"Starting training for {config['experiment_name']}")
    trainer.fit(
        train_loaders=[train_loader],
        val_loaders=[val_loader],
        max_epochs=config["training"]["max_epochs"]
    )

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
```

## Next Steps

1. **Explore Examples**: Check `demo/example3_production/` for a complete production example
2. **Read Documentation**:
   - [Multi-DataLoader Guide](MULTI_DATALOADER.md) for advanced patterns
   - [SLURM Guide](CONFIGURATION.md#slurm-configuration) for cluster usage
   - [Hooks System](HOOKS.md) for customization
3. **Try Grid Search**: Set up hyperparameter optimization
4. **Enable Logging**: Add Weights & Biases or TensorBoard
5. **Test Fault Tolerance**: Simulate preemption and verify resume works
