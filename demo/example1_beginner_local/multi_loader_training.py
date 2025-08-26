"""
Multi-DataLoader Training Example

This example demonstrates the multi-dataloader training architecture with:
- Two different dataloaders with different characteristics
- Various sampling strategies (ROUND_ROBIN, WEIGHTED, ALTERNATING)
- Validation with multiple loaders and aggregation strategies
- Checkpoint and resume functionality

This is the recommended way to use the training framework, as it's designed
as a multi-dataloader-only engine.
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
    PerformanceConfig,
    ValidationConfig,
)
from model_training_framework.trainer.config import (
    EpochLengthPolicy,
    SamplingStrategy,
    ValAggregation,
    ValidationFrequency,
)


class ToyModel(nn.Module):
    """
    Simple toy model for demonstration.

    This model can handle inputs of different sizes from different dataloaders,
    showing the flexibility of multi-dataloader training.
    """

    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def create_synthetic_datasets():
    """
    Create two synthetic datasets with different characteristics.

    Returns:
        tuple: (dataset_a, dataset_b) - Two datasets with different properties
    """
    # Dataset A: Larger, more samples
    torch.manual_seed(42)
    X_a = torch.randn(1000, 10)  # 1000 samples, 10 features
    y_a = torch.randint(0, 3, (1000,))  # 3 classes
    dataset_a = TensorDataset(X_a, y_a)

    # Dataset B: Smaller, different distribution
    torch.manual_seed(123)
    X_b = torch.randn(600, 10) * 2.0  # Different scale
    y_b = torch.randint(0, 3, (600,))
    # Add some bias to class distribution
    y_b[y_b == 2] = 1  # Make class 2 less frequent
    dataset_b = TensorDataset(X_b, y_b)

    return dataset_a, dataset_b


def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    """
    Training step function for multi-dataloader training.

    This function is called for each batch from any dataloader.
    The dataloader_idx and dataloader_name help identify the source.

    Args:
        trainer: The GenericTrainer instance
        batch: Current batch (inputs, targets)
        dataloader_idx: Index of the current dataloader (0, 1, etc.)
        dataloader_name: Name of the current dataloader

    Returns:
        dict: Metrics including loss
    """
    inputs, targets = batch

    # Forward pass
    outputs = trainer.model(inputs)
    loss = F.cross_entropy(outputs, targets)

    # Calculate accuracy
    _, predicted = outputs.max(1)
    accuracy = (predicted == targets).float().mean()

    # Return metrics with dataloader prefix for clarity
    return {
        "loss": loss,
        f"{dataloader_name}/accuracy": accuracy,
        f"{dataloader_name}/loss": loss.item(),
    }


def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    """
    Validation step function for multi-dataloader validation.

    Args:
        trainer: The GenericTrainer instance
        batch: Current batch (inputs, targets)
        dataloader_idx: Index of the current dataloader
        dataloader_name: Name of the current dataloader

    Returns:
        dict: Validation metrics
    """
    inputs, targets = batch

    with torch.no_grad():
        outputs = trainer.model(inputs)
        loss = F.cross_entropy(outputs, targets)

        _, predicted = outputs.max(1)
        accuracy = (predicted == targets).float().mean()

    return {
        "val_loss": loss,
        f"val_{dataloader_name}/accuracy": accuracy,
        f"val_{dataloader_name}/loss": loss.item(),
    }


def demonstrate_round_robin():
    """Demonstrate ROUND_ROBIN sampling strategy."""
    print("\n" + "=" * 60)
    print("ROUND_ROBIN Strategy: Alternates between dataloaders")
    print("=" * 60)

    # Create datasets and loaders
    dataset_a, dataset_b = create_synthetic_datasets()

    train_loader_a = DataLoader(dataset_a, batch_size=32, shuffle=True)
    train_loader_b = DataLoader(dataset_b, batch_size=32, shuffle=True)

    # Validation loaders (smaller batches)
    val_loader_a = DataLoader(dataset_a, batch_size=64, shuffle=False)
    val_loader_b = DataLoader(dataset_b, batch_size=64, shuffle=False)

    # Configuration with ROUND_ROBIN
    config = GenericTrainerConfig(
        multi=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.ROUND_ROBIN,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            dataloader_names=["dataset_a", "dataset_b"],
        ),
        validation=ValidationConfig(
            frequency=ValidationFrequency.PER_EPOCH,
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
        ),
        checkpoint=CheckpointConfig(
            root_dir=Path(tempfile.mkdtemp()),
            save_every_n_epochs=1,
        ),
        performance=PerformanceConfig(
            gradient_accumulation_steps=1,
            use_amp=False,  # Disable for simplicity
        ),
        logging=LoggingConfig(
            logger_type="console",
            log_per_loader_metrics=True,
        ),
    )

    # Create model and optimizer
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create trainer (note: optimizers is always a list)
    trainer = GenericTrainer(
        config=config,
        model=model,
        optimizers=[optimizer],  # Always a list, even for single optimizer
    )

    # Set training and validation functions
    trainer.set_training_step(training_step)
    trainer.set_validation_step(validation_step)

    print("Training with ROUND_ROBIN strategy...")
    print("This will alternate: dataset_a batch, dataset_b batch, repeat...")

    # Note: In a real implementation, you would call trainer.fit()
    # For this example, we're just demonstrating the setup
    print("✓ Trainer configured successfully with ROUND_ROBIN strategy")

    return trainer, [train_loader_a, train_loader_b], [val_loader_a, val_loader_b]


def demonstrate_weighted_sampling():
    """Demonstrate WEIGHTED sampling strategy."""
    print("\n" + "=" * 60)
    print("WEIGHTED Strategy: Sample based on weights")
    print("=" * 60)

    # Create datasets and loaders
    dataset_a, dataset_b = create_synthetic_datasets()

    train_loader_a = DataLoader(dataset_a, batch_size=32, shuffle=True)
    train_loader_b = DataLoader(dataset_b, batch_size=32, shuffle=True)

    # Configuration with WEIGHTED sampling
    config = GenericTrainerConfig(
        multi=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            dataloader_weights=[0.7, 0.3],  # 70% from dataset_a, 30% from dataset_b
            epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
            steps_per_epoch=100,  # Fixed 100 steps per epoch
            dataloader_names=["dataset_a", "dataset_b"],
        ),
        validation=ValidationConfig(
            frequency=ValidationFrequency.EVERY_N_STEPS,
            every_n_steps=50,
            aggregation=ValAggregation.MACRO_AVG_EQUAL_LOADERS,  # Equal weight to each loader
        ),
        performance=PerformanceConfig(
            gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
        ),
        logging=LoggingConfig(
            logger_type="console",
            log_loader_proportions=True,  # Log actual sampling proportions
        ),
    )

    # Create model and optimizer
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)

    # Create trainer
    trainer = GenericTrainer(
        config=config,
        model=model,
        optimizers=[optimizer],
    )

    trainer.set_training_step(training_step)

    print("Training with WEIGHTED strategy...")
    print("Dataset A will be sampled ~70% of the time")
    print("Dataset B will be sampled ~30% of the time")
    print("✓ Trainer configured successfully with WEIGHTED strategy")

    return trainer, [train_loader_a, train_loader_b]


def demonstrate_alternating_pattern():
    """Demonstrate ALTERNATING pattern strategy."""
    print("\n" + "=" * 60)
    print("ALTERNATING Strategy: Follow explicit pattern")
    print("=" * 60)

    # Create datasets and loaders
    dataset_a, dataset_b = create_synthetic_datasets()

    train_loader_a = DataLoader(dataset_a, batch_size=32, shuffle=True)
    train_loader_b = DataLoader(dataset_b, batch_size=32, shuffle=True)

    # Configuration with ALTERNATING pattern
    config = GenericTrainerConfig(
        multi=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.ALTERNATING,
            alternating_pattern=[0, 0, 1, 0, 0, 1],  # A, A, B, A, A, B, repeat...
            epoch_length_policy=EpochLengthPolicy.MAX_OF_LENGTHS,
            dataloader_names=["dataset_a", "dataset_b"],
            burst_size=2,  # Take 2 batches at a time from each loader
        ),
        logging=LoggingConfig(
            logger_type="console",
        ),
    )

    # Create model and optimizer
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create trainer
    trainer = GenericTrainer(
        config=config,
        model=model,
        optimizers=[optimizer],
    )

    trainer.set_training_step(training_step)

    print("Training with ALTERNATING pattern...")
    print("Pattern: [A, A, B, A, A, B] with burst_size=2")
    print("This means: 2 batches from A, 2 batches from A, 2 batches from B, repeat...")
    print("✓ Trainer configured successfully with ALTERNATING strategy")

    return trainer, [train_loader_a, train_loader_b]


def demonstrate_single_loader_as_multi():
    """Demonstrate how to use a single dataloader with the multi-loader API."""
    print("\n" + "=" * 60)
    print("Single DataLoader with Multi-Loader API")
    print("=" * 60)

    # Create single dataset
    torch.manual_seed(42)
    X = torch.randn(500, 10)
    y = torch.randint(0, 3, (500,))
    dataset = TensorDataset(X, y)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Configuration for single loader (still uses multi-loader config)
    config = GenericTrainerConfig(
        multi=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,  # Only one loader anyway
            dataloader_names=["main"],  # Single name in list
        ),
        validation=ValidationConfig(
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
        ),
    )

    # Create model and optimizer
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create trainer - same API as multi-loader
    trainer = GenericTrainer(
        config=config,
        model=model,
        optimizers=[optimizer],  # Still a list
    )

    trainer.set_training_step(training_step)
    trainer.set_validation_step(validation_step)

    print("Single dataloader configured with multi-loader API:")
    print("- train_loaders=[train_loader]  # Single loader in list")
    print("- val_loaders=[val_loader]      # Single loader in list")
    print("- dataloader_names=['main']     # Single name in list")
    print("✓ Single loader works seamlessly with multi-loader architecture")

    return trainer, [train_loader], [val_loader]


def main():
    """Main function demonstrating various multi-dataloader configurations."""
    print("\n" + "=" * 80)
    print(" Multi-DataLoader Training Example")
    print(" Demonstrating the multi-dataloader-only architecture")
    print("=" * 80)

    # Demonstrate different strategies
    round_robin_trainer, rr_train, rr_val = demonstrate_round_robin()
    weighted_trainer, w_train = demonstrate_weighted_sampling()
    alternating_trainer, a_train = demonstrate_alternating_pattern()
    single_trainer, s_train, s_val = demonstrate_single_loader_as_multi()

    print("\n" + "=" * 80)
    print(" Summary")
    print("=" * 80)
    print("\n✅ All configurations created successfully!")
    print("\nKey Takeaways:")
    print("1. The trainer is multi-dataloader-only by design")
    print("2. Single dataloaders must be wrapped in lists")
    print("3. Various sampling strategies available for different use cases:")
    print("   - ROUND_ROBIN: Fair alternation between loaders")
    print("   - WEIGHTED: Probabilistic sampling based on importance")
    print("   - ALTERNATING: Custom patterns for specific requirements")
    print("   - SEQUENTIAL: Process loaders one after another")
    print("4. Validation supports multiple loaders with aggregation strategies")
    print("5. Everything is deterministic and resumable")

    print("\n📚 Next Steps:")
    print("- Try running actual training with trainer.fit()")
    print("- Experiment with different sampling strategies")
    print("- Test checkpoint/resume functionality")
    print("- Explore validation aggregation options")


if __name__ == "__main__":
    main()
