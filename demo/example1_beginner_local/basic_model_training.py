"""
Basic Model Training - Single DataLoader with Multi-Loader API

This example demonstrates the simplest way to get started with the model training
framework. Even though we're using a single dataloader, the framework requires
using the multi-dataloader API (with a list containing one loader).

Target Audience: New users learning the framework basics with single dataset
"""

from pathlib import Path
import tempfile

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model_training_framework.trainer import (
    CheckpointConfig,
    GenericTrainer,
    GenericTrainerConfig,
    LoggingConfig,
    MultiDataLoaderConfig,
    ValidationConfig,
)
from model_training_framework.trainer.config import (
    EpochLengthPolicy,
    SamplingStrategy,
    ValAggregation,
)


class SimpleModel(nn.Module):
    """
    Simple neural network for demonstration.

    This is a basic MLP (Multi-Layer Perceptron) that can be used for
    classification tasks like MNIST.
    """

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Flatten input if needed (for MNIST-like data)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


def create_dummy_mnist_data(num_samples=1000):
    """
    Create dummy MNIST-like data for demonstration.

    Args:
        num_samples: Number of samples to generate

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    torch.manual_seed(42)

    # Training data
    train_images = torch.randn(num_samples, 28, 28)  # 28x28 images
    train_labels = torch.randint(0, 10, (num_samples,))  # 10 classes
    train_dataset = TensorDataset(train_images, train_labels)

    # Validation data (smaller)
    val_images = torch.randn(num_samples // 5, 28, 28)
    val_labels = torch.randint(0, 10, (num_samples // 5,))
    val_dataset = TensorDataset(val_images, val_labels)

    return train_dataset, val_dataset


def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    """
    Execute one training step.

    Note: Even with a single dataloader, this function receives dataloader_idx
    and dataloader_name parameters (will be 0 and "main" respectively).

    Args:
        trainer: The GenericTrainer instance
        batch: Tuple of (inputs, targets)
        dataloader_idx: Index of the current dataloader (always 0 for single loader)
        dataloader_name: Name of the current dataloader (e.g., "main")

    Returns:
        Dictionary containing loss and metrics
    """
    inputs, targets = batch

    # Forward pass
    outputs = trainer.model(inputs)
    loss = F.cross_entropy(outputs, targets)

    # Calculate accuracy for monitoring
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == targets).float().mean()

    # Get current learning rate
    lr = trainer.optimizers[0].param_groups[0]["lr"]

    return {
        "loss": loss,
        "accuracy": accuracy.item(),
        "lr": lr,
    }


def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    """
    Execute one validation step.

    Args:
        trainer: The GenericTrainer instance
        batch: Tuple of (inputs, targets)
        dataloader_idx: Index of the current dataloader
        dataloader_name: Name of the current dataloader

    Returns:
        Dictionary containing validation metrics
    """
    inputs, targets = batch

    # No gradients needed for validation
    with torch.no_grad():
        outputs = trainer.model(inputs)
        loss = F.cross_entropy(outputs, targets)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).float().mean()

    return {
        "val_loss": loss,
        "val_accuracy": accuracy.item(),
    }


def main():
    """
    Main training function demonstrating single dataloader with multi-loader API.
    """

    print("🚀 Starting Basic Model Training Example")
    print("   (Single DataLoader with Multi-Loader API)")
    print("=" * 50)

    # Create datasets
    print("📊 Creating dummy MNIST-like dataset...")
    train_dataset, val_dataset = create_dummy_mnist_data(num_samples=1000)

    # Create dataloaders
    # IMPORTANT: Even with single loader, we'll wrap it in a list later
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Use 0 for simplicity
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    print("✓ Created dataloaders:")
    print(f"  - Training: {len(train_loader)} batches")
    print(f"  - Validation: {len(val_loader)} batches")

    # Create configuration
    # NOTE: Even for single loader, we must use MultiDataLoaderConfig
    config = GenericTrainerConfig(
        # Multi-dataloader configs (split for train/val)
        train_loader_config=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,  # For single loader
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,  # Use all data
            dataloader_names=["main"],  # Single name in a list
        ),
        val_loader_config=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            dataloader_names=["main_val"],
        ),
        # Validation configuration
        validation=ValidationConfig(
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
        ),
        # Checkpoint configuration
        checkpoint=CheckpointConfig(
            root_dir=Path(tempfile.mkdtemp()) / "checkpoints",
            save_every_n_epochs=2,
            max_checkpoints=3,
        ),
        # Logging configuration
        logging=LoggingConfig(
            logger_type="console",
            log_scalars_every_n_steps=10,
        ),
    )

    print("\n🔧 Configuration created:")
    print(f"  - Train Sampling: {config.train_loader_config.sampling_strategy.value}")
    print(f"  - Train Loader Names: {config.train_loader_config.dataloader_names}")
    print(f"  - Val Loader Names: {config.val_loader_config.dataloader_names}")
    print(f"  - Checkpoint Dir: {config.checkpoint.root_dir}")

    # Create model
    model = SimpleModel(
        input_size=784,  # 28x28 flattened
        hidden_size=128,
        num_classes=10,
    )

    # Create optimizer
    # IMPORTANT: Must be a list, even for single optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
    )

    print("\n🏗️ Model and optimizer created:")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  - Optimizer: Adam (lr=0.001)")

    # Create trainer
    # IMPORTANT: Note the API differences from single-loader design
    trainer = GenericTrainer(
        config=config,
        model=model,
        optimizers=[optimizer],  # Always a list, even for single optimizer
        fabric=None,  # No distributed training for this example
    )

    # Set training and validation step functions
    trainer.set_training_step(training_step)
    trainer.set_validation_step(validation_step)

    print("\n🎯 Trainer initialized with multi-loader API")
    print("   Key differences from traditional single-loader:")
    print("   • optimizers=[optimizer] - always a list")
    print("   • train_loaders=[loader] - always a list")
    print("   • Requires MultiDataLoaderConfig")
    print("   • Training step gets dataloader_idx and name")

    # Prepare for training
    # IMPORTANT: Loaders must be provided as lists
    train_loaders = [train_loader]  # Single loader in a list
    val_loaders = [val_loader]  # Single loader in a list

    return trainer, train_loaders, val_loaders


if __name__ == "__main__":
    trainer, train_loaders, val_loaders = main()
    print("\n📚 Ready to train! Running fit() for 5 epochs...")
    trainer.fit(
        train_loaders=train_loaders,  # List with one loader
        val_loaders=val_loaders,  # List with one loader
        max_epochs=5,
    )
    print("\n✅ Training run completed!")
