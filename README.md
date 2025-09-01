# Model Training Framework

A comprehensive Python framework for deep learning research with three core components: **Config Search Sweep**, **SLURM Launcher**, and **Fault-Tolerant Trainer**. Built for researchers working on HPC clusters.

## 🚀 Quick Installation

```bash
git clone https://github.com/mjenrungrot/model-training-framework.git
cd model-training-framework
pip install -e ".[all]"
```

See [Installation Guide](docs/INSTALLATION.md) for detailed instructions.

## 🎯 Three Main Components

### 1. Config Search Sweep

Generate and manage hyperparameter search experiments programmatically.

```python
from model_training_framework.config import ParameterGridSearch, ParameterGrid

# Define base configuration
base_config = {
    "experiment_name": "baseline",
    "model": {"type": "transformer", "hidden_size": 256},
    "optimizer": {"type": "adamw", "lr": 1e-3},
    "training": {"max_epochs": 50}
}

# Create parameter search
gs = ParameterGridSearch(base_config)
grid = (
    ParameterGrid("hyperparameters")
    .add_parameter("optimizer.lr", [1e-4, 5e-4, 1e-3])
    .add_parameter("model.hidden_size", [256, 512, 1024])
)
gs.add_grid(grid)

# Generate experiments
experiments = list(gs.generate_experiments())
print(f"Generated {len(experiments)} experiments")
```

#### Advanced Grid Search

The framework supports sophisticated parameter specification methods:

```python
# Linked parameters - vary together as groups
grid.add_linked_parameters(
    ["model.size", "batch_size"],
    [("small", 64), ("medium", 32), ("large", 16)]
)

# Conditional parameters - apply only when conditions are met
grid.add_conditional_parameter(
    "optimizer.momentum",
    values=[0.9, 0.95],
    when={"optimizer.type": "sgd"}  # Only for SGD optimizer
)

# Computed parameters - derive from other parameters
grid.add_computed_parameter(
    "effective_batch_size",
    compute_func=lambda p: p["batch_size"] * p.get("gradient_accumulation_steps", 1)
)
```

See [Advanced ParameterGrid API](docs/API.md#advanced-parametergrid) for complete documentation.

## 🚀 Quick Start Example

For a complete, production-ready example that demonstrates all features:

📁 **[demo/example3_production/](demo/example3_production/README.md)** - Full implementation with:

- Grid search configuration
- SLURM job submission
- Fault-tolerant training
- Preemption handling
- Multi-GPU support

### 2. SLURM Launcher

Submit and manage jobs on HPC clusters with automatic requeue and preemption handling.

```python
from model_training_framework.slurm import SLURMLauncher

# Create launcher with SBATCH template
launcher = SLURMLauncher(
    template_path="slurm_template.txt",
    project_root=".",
    experiments_dir="./experiments"
)

# Submit batch of experiments
result = launcher.submit_experiment_batch(
    experiments=experiments,
    script_path="demo/example3_production/train_script.py",
    max_concurrent=10,
    dry_run=False  # Set True to preview
)

print(f"Submitted {result.success_count} jobs to SLURM")
```

### 3. Fault-Tolerant Trainer

Train models with automatic checkpointing, exact resume, and multi-dataloader support.

```python
import torch
import torch.nn.functional as F
from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    SamplingStrategy,
    CheckpointConfig,
)

# Configure trainer with multi-loader support
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
    )
)

# Assume model and optimizer are defined
# model = YourModel()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Create trainer
trainer = GenericTrainer(config, model, [optimizer])

# Define training step
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    device = next(trainer.model.parameters()).device
    x, y = batch

    # Move to device with non_blocking if using pinned memory
    if device.type == "cuda" and trainer.config.performance.pin_memory:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    # Use autocast for mixed precision if enabled
    if trainer.config.performance.use_amp and device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", enabled=True):
            outputs = trainer.model(x)
            loss = F.cross_entropy(outputs, y)
    else:
        outputs = trainer.model(x)
        loss = F.cross_entropy(outputs, y)

    return {"loss": loss}

trainer.set_training_step(training_step)

# Train with automatic resume on preemption
# Note: primary_loader, auxiliary_loader, and val_loader are placeholders.
# See docs/QUICKSTART.md for creating DataLoaders.
trainer.fit(
    train_loaders=[primary_loader, auxiliary_loader],
    val_loaders=[val_loader],
    max_epochs=100
)
```

## 📚 Complete Example

See [demo/example3_production/](demo/example3_production/) for a complete working example that demonstrates all three components working together.

For DataLoader optimization and best practices, see [DataLoader Best Practices](docs/DATALOADER_BEST_PRACTICES.md).

```bash
# Run locally
python demo/example3_production/orchestrate.py local

# Generate SLURM scripts (DRY RUN - does not submit)
python demo/example3_production/orchestrate.py slurm

# Actually submit jobs to SLURM (REAL SUBMISSION)
python demo/example3_production/orchestrate.py slurm submit
```

## 🔄 Key Features

### Multi-DataLoader Training

- **Sampling Strategies**: ROUND_ROBIN, WEIGHTED, ALTERNATING, SEQUENTIAL
- **Flexible Aggregation**: Multiple validation strategies
- **Per-Loader Metrics**: Track metrics for each dataset

### Fault Tolerance

- **Automatic Checkpointing**: Save at intervals or on preemption
- **Exact Resume**: Continue from exact batch/sample
- **SLURM Integration**: Handle preemption signals gracefully

### Resume & Warm-Start

- Auto-resume: When `preemption.resume_from_latest_symlink` is true (default), the trainer
  automatically resumes from the `latest.ckpt` if present (full state restore).
- Explicit resume: Pass `resume_from_checkpoint` to `fit()`. Use `'latest'` to resolve the latest
  checkpoint path.
- Warm-start (weights only): If no latest exists and a warm-start loader is registered (via
  configuration or `set_warm_start_loader()`), the trainer loads model weights only and starts a fresh run.

Examples:

```python
# Auto-resume (default behavior)
trainer.fit(train_loaders=[train_loader], val_loaders=[val_loader], max_epochs=10)

# Explicit full resume using the 'latest' alias
trainer.fit(train_loaders=[train_loader], val_loaders=[val_loader], max_epochs=10, resume_from_checkpoint="latest")

# Register a warm-start loader at runtime (weights-only)
from dataclasses import dataclass
from typing import Any

@dataclass
class WarmStartResult:
    model_state_dict: dict[str, Any]
    strict: bool = True

def my_loader(trainer, path: str) -> WarmStartResult:
    raw = torch.load(path, map_location="cpu")
    # Map or filter keys as needed
    state = raw.get("state_dict", raw)
    return WarmStartResult(model_state_dict=state, strict=False)

trainer.set_warm_start_loader(my_loader)
trainer.fit(train_loaders=[train_loader], max_epochs=10, resume_from_checkpoint="/path/to/custom.ckpt")

# Or via configuration (dotted import path)
from model_training_framework.config import WarmStartConfig

config.warm_start = WarmStartConfig(loader_class="mypackage.loaders.MyLoader", strict=False)
trainer = GenericTrainer(config, model, [optimizer])
trainer.fit([train_loader], max_epochs=10, resume_from_checkpoint="/path/to/custom.ckpt")
```

### Experiment Management

- **Grid Search**: Generate parameter combinations
- **Git Integration**: Isolate experiments in branches
- **Logging**: WandB, TensorBoard, Console support

## 📖 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Extended examples to get started
- **[API Reference](docs/API.md)** - Complete API documentation
- **[Configuration Guide](docs/CONFIGURATION.md)** - Detailed configuration options
- **[DataLoader Best Practices](docs/DATALOADER_BEST_PRACTICES.md)** - Performance optimization guide
- **[Multi-DataLoader Guide](docs/MULTI_DATALOADER.md)** - Multi-loader training patterns
- **[Migration Guide](docs/MIGRATION.md)** - Migrating existing code
- **[Advanced Features](docs/ADVANCED_FEATURES.md)** - Production features

## 🔄 Configuration Compatibility

### Key Mapping Changes (v0.2.0+)

| Old Key | New Key | Available Since |
|---------|---------|-----------------|
| `config.multi` | `config.train_loader_config` / `config.val_loader_config` | v0.2.0 |
| `data.num_workers` | `performance.dataloader_num_workers` | v0.2.0 |
| `training.epochs` | `training.max_epochs` | v0.2.0 |
| `optimizer.learning_rate` | `optimizer.lr` | v0.2.0 |
| `checkpoint.checkpoint_dir` | `checkpoint.root_dir` | v0.2.0 |
| `performance.mixed_precision: "16-mixed"` | `performance.use_amp: true` | v0.2.0 |

## 🏗️ Project Structure

```text
model_training_framework/
├── config/              # Configuration and grid search
├── slurm/               # SLURM job submission
├── trainer/             # Training engine
└── utils/               # Utilities

demo/
└── example3_production/ # Complete working example
    ├── config.py        # Configuration setup
    ├── orchestrate.py   # Job orchestration
    └── train_script.py  # Training script
```

## 🚦 Quick Start Workflow

### 1. Define Your Experiment

```python
base_config = {
    "experiment_name": "my_experiment",
    "model": {"type": "resnet", "num_layers": 18},
    "optimizer": {"type": "adam", "lr": 0.001},
    "training": {"max_epochs": 100}
}
```

### 2. Set Up Parameter Search

```python
gs = ParameterGridSearch(base_config)
grid = ParameterGrid("search").add_parameter("optimizer.lr", [1e-3, 1e-4])
gs.add_grid(grid)
experiments = list(gs.generate_experiments())
```

### 3. Submit to SLURM

```python
launcher = SLURMLauncher("template.txt", ".", "./experiments")
result = launcher.submit_experiment_batch(experiments, "train.py")
```

### 4. Train with Fault Tolerance

```python
trainer = GenericTrainer(config, model, [optimizer])
trainer.fit([train_loader], [val_loader], max_epochs=100)
# Automatically resumes from checkpoint if interrupted
```

## ⚙️ Requirements

- Python 3.12+
- PyTorch 2.0+
- Lightning Fabric 2.0+
- SLURM (for cluster submission)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE.md for details.

## 🆘 Support

- Issues: [GitHub Issues](https://github.com/mjenrungrot/model-training-framework/issues)
- Discussions: [GitHub Discussions](https://github.com/mjenrungrot/model-training-framework/discussions)
- Documentation: [Full Docs](docs/)
