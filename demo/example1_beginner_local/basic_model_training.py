"""
Basic Model Training - Beginner Local Development

This example demonstrates the simplest way to get started with the model training
framework for local development and prototyping. Perfect for new users who want to:

- Learn the framework basics
- Develop models locally before scaling to HPC
- Understand configuration management
- Get familiar with training patterns

Target Audience: New users, researchers starting with the framework, local development
"""

from pathlib import Path

import torch
from torch import nn
import yaml

from model_training_framework import ModelTrainingFramework
from model_training_framework.trainer import GenericTrainer, GenericTrainerConfig


class SimpleModel(nn.Module):
    """
    Simple neural network for demonstration.

    This is a basic MLP (Multi-Layer Perceptron) that can be used for
    classification tasks like MNIST. It's intentionally simple to focus
    on framework usage rather than model complexity.
    """

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Flatten input for fully connected layers
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class SimpleTrainer(GenericTrainer):
    """
    Custom trainer with simple training logic.

    This trainer extends the GenericTrainer to provide basic training
    and validation steps. It's designed to be easy to understand and
    modify for learning purposes.
    """

    def __init__(self, config, model, optimizer, loss_fn):
        super().__init__(config)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        """
        Execute one training step.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the current batch

        Returns:
            Dictionary containing loss and metrics
        """
        x, y = batch

        # Forward pass
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)

        # Calculate accuracy for monitoring
        _, predicted = torch.max(y_pred.data, 1)
        accuracy = (predicted == y).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validation_step(self, batch, batch_idx):
        """
        Execute one validation step.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the current batch

        Returns:
            Dictionary containing validation loss and metrics
        """
        x, y = batch

        # No gradients needed for validation
        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)

            # Calculate accuracy
            _, predicted = torch.max(y_pred.data, 1)
            accuracy = (predicted == y).float().mean()

        return {"val_loss": loss, "val_accuracy": accuracy.item()}


def create_dummy_data_loader(batch_size=32, num_batches=100, data_type="train"):
    """
    Create dummy data loader for demonstration.

    This creates synthetic MNIST-like data for testing the framework
    without needing real datasets. Perfect for quick prototyping.

    Args:
        batch_size: Number of samples per batch
        num_batches: Number of batches to create
        data_type: Type of data ("train" or "val") for different distributions

    Returns:
        List of (inputs, targets) tuples
    """
    data = []

    # Add some variation between train and validation data
    noise_level = 0.1 if data_type == "train" else 0.05

    for _ in range(num_batches):
        # Create MNIST-like data (28x28 images)
        x = torch.randn(batch_size, 28, 28) * noise_level

        # Random labels for classification
        y = torch.randint(0, 10, (batch_size,))

        data.append((x, y))

    return data


def load_config_from_file(config_path: Path) -> dict:
    """
    Load configuration from YAML file if it exists.

    This function demonstrates how to load configuration from external
    files, which is useful for reproducible experiments.
    """
    if config_path.exists():
        with config_path.open() as f:
            return yaml.safe_load(f)
    return {}


def main():
    """
    Main training function.

    This is the entry point that demonstrates the complete workflow:
    1. Setup paths and configuration
    2. Initialize the framework
    3. Create model and training components
    4. Run training with proper error handling
    """

    print("🚀 Starting Basic Model Training Example")
    print("=" * 50)

    # Setup paths - using relative paths for local development
    project_root = Path.cwd()
    config_dir = project_root / "demo" / "example1_beginner_local" / "config_examples"

    # Try to load configuration from file, fall back to defaults
    config_file = config_dir / "simple_config.yaml"
    file_config = load_config_from_file(config_file)

    # Create basic configuration with sensible defaults
    base_config = {
        "experiment_name": "beginner_local_training",
        "description": "Local development training example for beginners",
        "model": {
            "name": "simple_mlp",
            "hidden_size": 128,
            "input_size": 784,
            "num_classes": 10,
        },
        "training": {
            "epochs": 5,  # Shorter for quick feedback
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "data": {
            "dataset_name": "dummy_mnist",
            "train_batches": 50,  # Smaller dataset for fast iteration
            "val_batches": 10,
        },
        "optimizer": {"name": "adam", "weight_decay": 0.01},
        "logging": {
            "log_level": "INFO",
            "use_wandb": False,  # Disabled for simplicity
        },
        "checkpoint": {"save_every_n_epochs": 2, "checkpoint_dir": "./checkpoints"},
        "preemption": {"timeout_minutes": 5, "grace_period_seconds": 60},
    }

    # Override with file config if available
    if file_config:
        print(f"📋 Loading configuration from {config_file}")
        # Simple merge - in production, you'd want deeper merging
        base_config.update(file_config)
    else:
        print("📋 Using default configuration")

    print(f"🔧 Experiment: {base_config['experiment_name']}")
    print(f"📊 Training for {base_config['training']['epochs']} epochs")

    # Initialize framework
    try:
        framework = ModelTrainingFramework(
            project_root=project_root, config_dir=config_dir
        )
        config_manager = framework.get_config_manager()
        experiment_config = config_manager.create_experiment_config(base_config)

        print("✅ Framework initialized successfully")

    except Exception as e:
        print(f"❌ Failed to initialize framework: {e}")
        print(
            "💡 Make sure you're in the correct directory and have installed the framework"
        )
        return

    # Setup model and training components
    print("🏗️  Setting up model and training components...")

    model = SimpleModel(
        input_size=experiment_config.model.input_size,
        hidden_size=experiment_config.model.hidden_size,
        num_classes=experiment_config.model.num_classes,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment_config.training.learning_rate,
        weight_decay=experiment_config.optimizer.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss()

    # Create data loaders with configuration
    train_loader = create_dummy_data_loader(
        batch_size=experiment_config.training.batch_size,
        num_batches=experiment_config.data.train_batches,
        data_type="train",
    )

    val_loader = create_dummy_data_loader(
        batch_size=experiment_config.training.batch_size,
        num_batches=experiment_config.data.val_batches,
        data_type="val",
    )

    print("📊 Created data loaders:")
    print(f"   - Training: {len(train_loader)} batches")
    print(f"   - Validation: {len(val_loader)} batches")

    # Initialize trainer
    trainer_config = GenericTrainerConfig(
        max_epochs=experiment_config.training.epochs,
        checkpoint_dir=experiment_config.checkpoint.checkpoint_dir,
        save_every_n_epochs=experiment_config.checkpoint.save_every_n_epochs,
        preemption_timeout=experiment_config.preemption.timeout_minutes * 60,
    )

    trainer = SimpleTrainer(
        config=trainer_config, model=model, optimizer=optimizer, loss_fn=loss_fn
    )

    print("🎯 Starting training...")
    print("-" * 30)

    # Start training with comprehensive error handling
    try:
        trainer.fit(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        print("-" * 30)
        print("✅ Training completed successfully!")
        print(f"💾 Checkpoints saved to: {experiment_config.checkpoint.checkpoint_dir}")

        # Provide next steps for users
        print("\n🎉 Congratulations! You've completed your first training run.")
        print("Next steps you can try:")
        print("  1. Modify the configuration in config_examples/")
        print("  2. Experiment with different model architectures")
        print("  3. Try the intermediate HPC example for distributed training")
        print("  4. Explore the advanced production example for fault tolerance")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("💡 This is normal - you can resume from the last checkpoint")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("💡 Check the configuration and make sure all dependencies are installed")

        # Provide debugging help
        print("\nDebugging tips:")
        print(
            "  1. Check if PyTorch is installed: python -c 'import torch; print(torch.__version__)'"
        )
        print(
            "  2. Verify the framework is installed: pip list | grep model-training-framework"
        )
        print("  3. Check the configuration file format")

        raise


if __name__ == "__main__":
    main()
