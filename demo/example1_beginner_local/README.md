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
│   ├── simple_config.yaml       # Minimal configuration
│   └── mnist_config.yaml        # Complete MNIST example
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

[`basic_model_training.py`](basic_model_training.py) demonstrates:

```python
# 1. Model Definition
class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        # Simple MLP architecture

# 2. Trainer Implementation
class SimpleTrainer(GenericTrainer):
    def training_step(self, batch, batch_idx):
        # Custom training logic

# 3. Framework Integration
framework = ModelTrainingFramework(project_root=project_root)
trainer.fit(model, optimizer, train_loader, val_loader)
```

**Key Learning Points:**

- How to extend `GenericTrainer` for custom logic
- Framework initialization and configuration
- Model, optimizer, and data loader setup
- Training loop execution with monitoring

### Step 2: Configuration Management

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

#### Complete Configuration ([`mnist_config.yaml`](config_examples/mnist_config.yaml))

```yaml
experiment_name: "mnist_classification_example"
model:
  hidden_size: 256
  dropout_rate: 0.1
training:
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
scheduler:
  name: "cosine"
  warmup_steps: 50
early_stopping:
  enabled: true
  patience: 5
```

**Key Learning Points:**

- Configuration hierarchy and organization
- Parameter specification and validation
- Environment-specific overrides
- Best practices for reproducible experiments

### Step 3: Working with Data

[`data/sample_dataset.py`](data/sample_dataset.py) provides utilities for:

```python
# Synthetic datasets for learning
mnist_dataset = SyntheticMNIST(num_samples=1000)
regression_dataset = SyntheticRegression(num_samples=500)

# Data splitting and loading
train_loader, val_loader = create_train_val_split(
    dataset, val_ratio=0.2, batch_size=32
)
```python

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
📋 Using default configuration
🔧 Experiment: beginner_local_training
📊 Training for 5 epochs
✅ Framework initialized successfully
🏗️  Setting up model and training components...
📊 Created data loaders:
   - Training: 50 batches
   - Validation: 10 batches
🎯 Starting training...
------------------------------

📊 Epoch 1 completed:
   Loss: 2.1234
   Accuracy: 0.2345
   Val Loss: 1.9876
   Val Accuracy: 0.3456

[... training continues ...]

✅ Training completed successfully!
💾 Checkpoints saved to: ./checkpoints
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
