"""
Basic Training Example

This example demonstrates how to use the model training framework
for a simple training setup with configuration management.
"""

from pathlib import Path

import torch
from torch import nn

from model_training_framework import ModelTrainingFramework
from model_training_framework.trainer import GenericTrainer, GenericTrainerConfig


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class SimpleTrainer(GenericTrainer):
    """Custom trainer with simple training logic."""

    def __init__(self, config, model, optimizer, loss_fn):
        super().__init__(config)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        """Execute one training step."""
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)

        # Calculate accuracy
        _, predicted = torch.max(y_pred.data, 1)
        accuracy = (predicted == y).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validation_step(self, batch, batch_idx):
        """Execute one validation step."""
        x, y = batch
        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)

            # Calculate accuracy
            _, predicted = torch.max(y_pred.data, 1)
            accuracy = (predicted == y).float().mean()

        return {"val_loss": loss, "val_accuracy": accuracy.item()}


def create_dummy_data_loader(batch_size=32, num_batches=100):
    """Create dummy data loader for demonstration."""
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, 28, 28)  # MNIST-like data
        y = torch.randint(0, 10, (batch_size,))  # Random labels
        data.append((x, y))
    return data


def main():
    """Main training function."""

    # Setup paths
    project_root = Path.cwd()
    config_dir = project_root / "configs"

    # Create basic configuration
    base_config = {
        "experiment_name": "basic_training_example",
        "description": "Simple training example",
        "model": {"name": "simple_mlp", "hidden_size": 128},
        "training": {"epochs": 10, "batch_size": 32, "learning_rate": 0.001},
        "data": {"dataset_name": "dummy_mnist"},
        "optimizer": {"name": "adam", "weight_decay": 0.01},
        "logging": {"log_level": "INFO", "use_wandb": False},
        "checkpoint": {"save_every_n_epochs": 5, "checkpoint_dir": "./checkpoints"},
        "preemption": {"timeout_minutes": 5, "grace_period_seconds": 60},
        "performance": {"num_workers": 4, "pin_memory": True},
    }

    print("🚀 Initializing Model Training Framework...")

    # Initialize framework
    framework = ModelTrainingFramework(project_root=project_root, config_dir=config_dir)

    # Create experiment config
    config_manager = framework.get_config_manager()
    experiment_config = config_manager.create_experiment_config(base_config)

    print(f"📋 Created experiment: {experiment_config.experiment_name}")

    # Setup model and training components
    model = SimpleModel(
        input_size=784, hidden_size=experiment_config.model.hidden_size, num_classes=10
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment_config.training.learning_rate,
        weight_decay=experiment_config.optimizer.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss()

    # Create data loaders
    train_loader = create_dummy_data_loader(
        batch_size=experiment_config.training.batch_size, num_batches=100
    )
    val_loader = create_dummy_data_loader(
        batch_size=experiment_config.training.batch_size, num_batches=20
    )

    print("📊 Created dummy data loaders")

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

    # Start training
    try:
        trainer.fit(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        print("✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
