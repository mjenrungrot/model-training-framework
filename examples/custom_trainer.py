"""
Custom Trainer Implementation Example

This example demonstrates how to create custom trainers by extending
the GenericTrainer class with domain-specific functionality.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    TrainingCallback,
)


class MultiTaskModel(nn.Module):
    """Example multi-task model for demonstration."""

    def __init__(
        self, input_size=784, hidden_size=256, num_classes_task1=10, num_classes_task2=5
    ):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Task-specific heads
        self.task1_head = nn.Linear(hidden_size, num_classes_task1)
        self.task2_head = nn.Linear(hidden_size, num_classes_task2)

    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Shared encoding
        encoded = self.encoder(x)

        # Task outputs
        task1_logits = self.task1_head(encoded)
        task2_logits = self.task2_head(encoded)

        return {"task1": task1_logits, "task2": task2_logits, "encoded": encoded}


class MultiTaskTrainer(GenericTrainer):
    """Custom trainer for multi-task learning."""

    def __init__(
        self,
        config: GenericTrainerConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(config)
        self.model = model
        self.optimizer = optimizer
        self.task_weights = task_weights or {"task1": 1.0, "task2": 1.0}

        # Loss functions for each task
        self.task1_loss_fn = nn.CrossEntropyLoss()
        self.task2_loss_fn = nn.CrossEntropyLoss()

        # Metrics tracking
        self.running_metrics = {
            "task1_loss": 0.0,
            "task2_loss": 0.0,
            "task1_acc": 0.0,
            "task2_acc": 0.0,
            "total_loss": 0.0,
        }

    def training_step(
        self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int
    ) -> Dict[str, Any]:
        """Execute one training step with multi-task loss."""

        # Unpack batch
        x, targets = batch
        task1_targets = targets["task1"]
        task2_targets = targets["task2"]

        # Forward pass
        outputs = self.model(x)
        task1_logits = outputs["task1"]
        task2_logits = outputs["task2"]

        # Compute losses
        task1_loss = self.task1_loss_fn(task1_logits, task1_targets)
        task2_loss = self.task2_loss_fn(task2_logits, task2_targets)

        # Weighted total loss
        total_loss = (
            self.task_weights["task1"] * task1_loss
            + self.task_weights["task2"] * task2_loss
        )

        # Compute accuracies
        task1_acc = self._compute_accuracy(task1_logits, task1_targets)
        task2_acc = self._compute_accuracy(task2_logits, task2_targets)

        # Update running metrics
        self.running_metrics["task1_loss"] += task1_loss.item()
        self.running_metrics["task2_loss"] += task2_loss.item()
        self.running_metrics["task1_acc"] += task1_acc
        self.running_metrics["task2_acc"] += task2_acc
        self.running_metrics["total_loss"] += total_loss.item()

        return {
            "loss": total_loss,
            "task1_loss": task1_loss.item(),
            "task2_loss": task2_loss.item(),
            "task1_accuracy": task1_acc,
            "task2_accuracy": task2_acc,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validation_step(
        self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int
    ) -> Dict[str, Any]:
        """Execute one validation step."""

        with torch.no_grad():
            # Unpack batch
            x, targets = batch
            task1_targets = targets["task1"]
            task2_targets = targets["task2"]

            # Forward pass
            outputs = self.model(x)
            task1_logits = outputs["task1"]
            task2_logits = outputs["task2"]

            # Compute losses
            task1_loss = self.task1_loss_fn(task1_logits, task1_targets)
            task2_loss = self.task2_loss_fn(task2_logits, task2_targets)

            # Weighted total loss
            total_loss = (
                self.task_weights["task1"] * task1_loss
                + self.task_weights["task2"] * task2_loss
            )

            # Compute accuracies
            task1_acc = self._compute_accuracy(task1_logits, task1_targets)
            task2_acc = self._compute_accuracy(task2_logits, task2_targets)

        return {
            "val_loss": total_loss,
            "val_task1_loss": task1_loss.item(),
            "val_task2_loss": task2_loss.item(),
            "val_task1_accuracy": task1_acc,
            "val_task2_accuracy": task2_acc,
        }

    def _compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute classification accuracy."""
        _, predicted = torch.max(logits, 1)
        correct = (predicted == targets).float().sum()
        accuracy = correct / targets.size(0)
        return accuracy.item()

    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        super().on_epoch_start(epoch)

        # Reset running metrics
        for key in self.running_metrics:
            self.running_metrics[key] = 0.0

        print(f"🎯 Starting epoch {epoch + 1}")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Called at the end of each epoch."""
        super().on_epoch_end(epoch, logs)

        # Print epoch summary
        print(f"📊 Epoch {epoch + 1} Summary:")
        print(f"   Total Loss: {logs.get('loss', 0):.4f}")
        print(
            f"   Task 1 - Loss: {logs.get('task1_loss', 0):.4f}, Acc: {logs.get('task1_accuracy', 0):.4f}"
        )
        print(
            f"   Task 2 - Loss: {logs.get('task2_loss', 0):.4f}, Acc: {logs.get('task2_accuracy', 0):.4f}"
        )

        if "val_loss" in logs:
            print(f"   Validation - Total Loss: {logs['val_loss']:.4f}")
            print(
                f"   Val Task 1 - Loss: {logs.get('val_task1_loss', 0):.4f}, Acc: {logs.get('val_task1_accuracy', 0):.4f}"
            )
            print(
                f"   Val Task 2 - Loss: {logs.get('val_task2_loss', 0):.4f}, Acc: {logs.get('val_task2_accuracy', 0):.4f}"
            )


class EarlyStopping(TrainingCallback):
    """Early stopping callback for custom trainer."""

    def __init__(
        self, patience: int = 5, min_delta: float = 0.001, monitor: str = "val_loss"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer: GenericTrainer, epoch: int, logs: Dict[str, Any]):
        """Check for early stopping condition."""
        current_score = logs.get(self.monitor)

        if current_score is None:
            return

        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                print(f"   Best {self.monitor}: {self.best_score:.4f}")


class LearningRateScheduler(TrainingCallback):
    """Learning rate scheduling callback."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, trainer: GenericTrainer, epoch: int, logs: Dict[str, Any]):
        """Step the learning rate scheduler."""
        if hasattr(self.scheduler, "step"):
            # For schedulers that take a metric
            if hasattr(self.scheduler, "mode"):
                val_loss = logs.get("val_loss")
                if val_loss is not None:
                    self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

        # Log current learning rate
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"📈 Learning rate: {current_lr:.2e}")


def create_dummy_multitask_data(batch_size=32, num_batches=50):
    """Create dummy multi-task data."""
    data = []

    for _ in range(num_batches):
        # Input features
        x = torch.randn(batch_size, 28, 28)  # MNIST-like

        # Multi-task targets
        task1_targets = torch.randint(0, 10, (batch_size,))  # 10-class classification
        task2_targets = torch.randint(0, 5, (batch_size,))  # 5-class classification

        targets = {"task1": task1_targets, "task2": task2_targets}
        data.append((x, targets))

    return data


def main():
    """Main custom trainer example."""

    print("🎯 Custom Multi-Task Trainer Example")

    # Model configuration
    model = MultiTaskModel(
        input_size=784, hidden_size=256, num_classes_task1=10, num_classes_task2=5
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Trainer configuration
    trainer_config = GenericTrainerConfig(
        max_epochs=20,
        checkpoint_dir="./checkpoints/multitask",
        save_every_n_epochs=5,
        preemption_timeout=300,
    )

    # Create custom trainer
    trainer = MultiTaskTrainer(
        config=trainer_config,
        model=model,
        optimizer=optimizer,
        task_weights={"task1": 0.7, "task2": 0.3},  # Task 1 is more important
    )

    # Add callbacks
    early_stopping = EarlyStopping(patience=5, monitor="val_loss")
    lr_scheduler = LearningRateScheduler(scheduler)

    trainer.add_callback(early_stopping)
    trainer.add_callback(lr_scheduler)

    # Create data
    print("📊 Creating multi-task data...")
    train_data = create_dummy_multitask_data(batch_size=32, num_batches=100)
    val_data = create_dummy_multitask_data(batch_size=32, num_batches=20)

    print(f"   Training batches: {len(train_data)}")
    print(f"   Validation batches: {len(val_data)}")

    # Start training
    print("🚀 Starting multi-task training...")

    try:
        trainer.fit(
            model=model,
            optimizer=optimizer,
            train_loader=train_data,
            val_loader=val_data,
        )

        print("✅ Training completed successfully!")

        # Final model evaluation
        print("\n📈 Final Model Performance:")

        # Run a few validation batches for final metrics
        model.eval()
        total_metrics = {"task1_acc": 0.0, "task2_acc": 0.0, "total_loss": 0.0}

        with torch.no_grad():
            for i, (x, targets) in enumerate(val_data[:5]):  # Sample 5 batches
                outputs = model(x)

                task1_acc = trainer._compute_accuracy(
                    outputs["task1"], targets["task1"]
                )
                task2_acc = trainer._compute_accuracy(
                    outputs["task2"], targets["task2"]
                )

                total_metrics["task1_acc"] += task1_acc
                total_metrics["task2_acc"] += task2_acc

        # Average metrics
        num_samples = 5
        print(f"   Task 1 Accuracy: {total_metrics['task1_acc'] / num_samples:.4f}")
        print(f"   Task 2 Accuracy: {total_metrics['task2_acc'] / num_samples:.4f}")

    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def demonstrate_advanced_trainer_features():
    """Demonstrate advanced trainer customization features."""

    print("\n🔬 Advanced Trainer Features Demo:")

    # 1. Custom training phases
    print("   1. Custom Training Phases:")
    print("      - Override phase transitions for custom training loops")
    print("      - Add custom phases for specific operations")

    # 2. State management
    print("   2. Advanced State Management:")
    print("      - Custom resume states for complex training scenarios")
    print("      - Model-specific checkpointing strategies")

    # 3. Distributed training integration
    print("   3. Distributed Training:")
    print("      - Use Lightning Fabric for multi-GPU training")
    print("      - Custom data parallel strategies")

    # 4. Custom metrics and logging
    print("   4. Custom Metrics:")
    print("      - Implement domain-specific metrics")
    print("      - Custom logging and visualization")

    print("✅ Advanced features overview completed")


if __name__ == "__main__":
    main()
    demonstrate_advanced_trainer_features()
