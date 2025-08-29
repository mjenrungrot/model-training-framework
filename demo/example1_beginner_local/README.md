# Example 1: Beginner Local Development

Welcome to the beginner-friendly introduction to the Model Training Framework! This scenario is designed for new users who want to learn the framework basics and develop models locally before scaling to HPC environments.

## 🎯 Target Audience

- **New users** learning the framework
- **Researchers** starting with the framework
- **Local development** and prototyping
- **Students** and **educators** teaching ML concepts

## 📚 What You'll Learn

- Framework installation and setup
- Basic configuration file structure
- Local training execution
- Model checkpointing and recovery
- Results interpretation
- Configuration management best practices

## 🗂️ Directory Structure

```text
example1_beginner_local/
├── README.md                    # This file
├── basic_model_training.py      # Main training script
├── config_examples/             # Configuration templates
│   ├── simple_config.yaml       # Minimal single-loader config (new API)
│   └── multi_loader_config.yaml # A couple multi-loader presets (new API)
└── data/
    └── sample_dataset.py        # Synthetic datasets for learning
```

## 🚀 Quick Start

### Prerequisites

1. **Install the framework:**

   ```bash
   cd /path/to/model_training_framework
   pip install -e .
   ```

2. **Install dependencies:**

   ```bash
   pip install torch pyyaml
   ```

### Run Your First Training

1. **Navigate to the demo directory:**

   ```bash
   cd demo/example1_beginner_local
   ```

2. **Run basic training:**

   ```bash
   python basic_model_training.py
   ```

3. **Watch the magic happen!** ✨
   The script will:
   - Create a simple neural network
   - Generate synthetic MNIST-like data
   - Train for 5 epochs with checkpointing
   - Display progress and metrics

## 📖 Detailed Walkthrough

### Step 1: Understanding the Training Script

[`basic_model_training.py`](basic_model_training.py) uses the multi-dataloader-only API. Even a single dataloader is passed as a list, and the configuration is split per phase.

```python
from model_training_framework.trainer import (
    GenericTrainer, GenericTrainerConfig, MultiDataLoaderConfig,
    CheckpointConfig, LoggingConfig, ValidationConfig
)
from model_training_framework.config.schemas import (
    SamplingStrategy, EpochLengthPolicy, ValAggregation
)

# 1) Create config (split per phase)
config = GenericTrainerConfig(
    train_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
        dataloader_names=["main"],
    ),
    val_loader_config=MultiDataLoaderConfig(
        sampling_strategy=SamplingStrategy.SEQUENTIAL,
        dataloader_names=["main_val"],
    ),
    validation=ValidationConfig(
        aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
    ),
    checkpoint=CheckpointConfig(save_every_n_epochs=2),
    logging=LoggingConfig(logger_type="console"),
)

# 2) Create model and optimizer
model = SimpleModel(input_size=784, hidden_size=128, num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

# 3) Instantiate trainer (optimizers is always a list)
trainer = GenericTrainer(config=config, model=model, optimizers=[optimizer])

# 4) Define step functions (multi-loader signature)
def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch
    logits = trainer.model(x)
    loss = F.cross_entropy(logits, y)
    print(f"[TRAIN] {dataloader_name=} {dataloader_idx=} {batch_idx=}")
    return {"loss": loss, f"{dataloader_name}/loss": loss.item()}

def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    x, y = batch
    with torch.no_grad():
        logits = trainer.model(x)
        loss = F.cross_entropy(logits, y)
    print(f"[VALID] {dataloader_name=} {dataloader_idx=} {batch_idx=}")
    return {"val_loss": loss, f"val_{dataloader_name}/loss": loss.item()}

trainer.set_training_step(training_step)
trainer.set_validation_step(validation_step)

# 5) Always pass loaders as lists
train_loaders = [train_loader]
val_loaders = [val_loader]

trainer.fit(train_loaders=train_loaders, val_loaders=val_loaders, max_epochs=5)
```

**Key Learning Points:**

- Multi-loader-only API: pass loaders as lists, even for one loader
- Separate configs: `train_loader_config` and `val_loader_config`
- Step signature: `(trainer, batch, batch_idx, dataloader_idx, dataloader_name)`
- Default logging is console-only; no external services by default

### Step 2: Configuration Management

This example uses inline (code) config by default so it works out-of-the-box with `python basic_model_training.py`.

Included templates (new API, reference-only):

- `config_examples/simple_config.yaml`: shows single-dataloader settings using `train_loader_config`/`val_loader_config`.
- `config_examples/multi_loader_config.yaml`: a couple of multi-dataloader presets you can adapt.

The framework uses YAML configuration files for reproducible experiments:

#### Simple Configuration ([`simple_config.yaml`](config_examples/simple_config.yaml))

```yaml
experiment_name: "my_first_experiment"
model:
  hidden_size: 128
training:
  epochs: 5
  batch_size: 32
  learning_rate: 0.001
```

#### Multi-Loader Presets ([`multi_loader_config.yaml`](config_examples/multi_loader_config.yaml))

This file contains a couple of multi-loader presets that you can adapt. These use the new keys:

- `train_loader_config`: multi-loader settings for training
- `val_loader_config`: multi-loader settings for validation

Copy a block into your own YAML file and pass it with `--config`.

### Step 3: Working with Data

[`data/sample_dataset.py`](data/sample_dataset.py) provides utilities for:

```python
# Synthetic datasets for learning (see data/sample_dataset.py)
mnist_dataset = SyntheticMNIST(num_samples=1000)
train_loader, val_loader = create_train_val_split(mnist_dataset, val_ratio=0.2, batch_size=32)
```

**Key Learning Points:**

- Dataset creation and management
- Train/validation splitting
- DataLoader configuration
- Data preprocessing patterns

## 🔧 Customization Guide

### Modify the Model Architecture

Edit the `SimpleModel` class in [`basic_model_training.py`](basic_model_training.py):

```python
class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        # Add more layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Additional layer
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
```

### Experiment with Different Optimizers

```python
# Try different optimizers
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# or
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
```

### Modify Training Configuration

Create your own config file based on the examples:

```yaml
experiment_name: "my_custom_experiment"
description: "Experimenting with different settings"

model:
  hidden_size: 256           # Larger model
  dropout_rate: 0.2          # More regularization

training:
  epochs: 15                 # Longer training
  batch_size: 64             # Larger batches
  learning_rate: 0.0005      # Different learning rate
```

## 📊 Understanding the Output

When you run the training, you'll see output like:

```text
🚀 Starting Basic Model Training Example
==================================================
📊 Creating dummy MNIST-like dataset...
✓ Created dataloaders:
  - Training: 32 batches
  - Validation: 4 batches

🔧 Configuration created:
  - Train Sampling: sequential
  - Train Loader Names: ['main']
  - Val Loader Names: ['main_val']

🎯 Trainer initialized; starting fit() for 5 epochs...
...
✅ Training completed successfully!
💾 Checkpoints saved (see ./checkpoints or tmp dir)
```

**Key Metrics to Understand:**

- **Loss**: How far predictions are from true values (lower is better)
- **Accuracy**: Percentage of correct predictions (higher is better)
- **Val Loss/Accuracy**: Performance on unseen validation data
- **Learning Rate**: Step size for parameter updates

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **Import Errors**

   ```bash
   # Make sure framework is installed
   pip install -e /path/to/model_training_framework
   ```

2. **Configuration File Not Found**

   ```bash
   # Check file paths and current directory
   ls config_examples/
   ```

3. **CUDA Out of Memory**

   ```python
   # Reduce batch size in configuration
   training:
     batch_size: 16  # Instead of 32
   ```

4. **Training Too Slow**

   ```python
   # Reduce model size or data
   model:
     hidden_size: 64  # Instead of 128
   data:
     train_batches: 25  # Instead of 50
   ```

## 🎓 Next Steps

Congratulations! You've completed the beginner scenario. Here's what to explore next:

### 1. **Experiment Locally**

- Modify configurations and observe results
- Try different model architectures
- Experiment with learning rates and optimizers
- Add your own datasets

### 2. **Move to Intermediate HPC** → [`../example2_intermediate_hpc/`](../example2_intermediate_hpc/)

- Learn distributed training
- Scale to multiple GPUs
- Use SLURM job scheduling
- Run hyperparameter optimization

### 3. **Advance to Production** → [`../example3_advanced_production/`](../example3_advanced_production/)

- Implement fault-tolerant training
- Use advanced monitoring
- Deploy in production environments
- Handle enterprise-scale requirements

## 📚 Additional Resources

### Framework Documentation

- [Configuration System Guide](../../docs/configuration.md)
- [Trainer Architecture](../../docs/trainers.md)
- [Best Practices](../../docs/best_practices.md)

### Learning Materials

- [Machine Learning Fundamentals](https://ml-fundamentals.example.com)
- [PyTorch Tutorial](https://pytorch.org/tutorials/)
- [Deep Learning Course](https://course.fast.ai/)

### Community

- [GitHub Issues](https://github.com/model-training-framework/issues)
- [Discussion Forum](https://discussions.example.com)
- [Slack Community](https://slack.example.com)

## 💡 Tips for Success

1. **Start Small**: Begin with simple models and small datasets
2. **Understand Each Component**: Make sure you understand each part before moving on
3. **Experiment Freely**: The synthetic data is safe to experiment with
4. **Read the Logs**: Pay attention to training output and metrics
5. **Save Your Work**: Use meaningful experiment names and keep notes
6. **Ask Questions**: Use the community resources when you get stuck

---

**Happy Learning!** 🎉 You're on your way to mastering the Model Training Framework!
