"""
Production Custom Trainer - Advanced Multi-Task Learning

This module demonstrates production-grade custom trainers with multi-task learning,
advanced metrics, and enterprise-level features for production deployments.
"""

from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Any

import torch
from torch import nn

from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
)


@dataclass
class TaskConfig:
    """Configuration for individual tasks in multi-task learning."""

    name: str
    weight: float = 1.0
    num_classes: int | None = None
    task_type: str = "classification"  # "classification", "regression"


class MultiTaskModel(nn.Module):
    """Production-grade multi-task model with shared encoder."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] | None = None,
        task_configs: list[TaskConfig] | None = None,
        dropout_rate: float = 0.1,
    ):
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        super().__init__()

        self.task_configs = task_configs or []

        # Build shared encoder
        encoder_layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            encoder_layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_size = hidden_size

        self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last dropout

        # Build task-specific heads
        self.task_heads = nn.ModuleDict()
        encoder_output_size = hidden_sizes[-1]

        for task_config in self.task_configs:
            if task_config.task_type == "classification":
                head = nn.Sequential(
                    nn.Linear(encoder_output_size, encoder_output_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(encoder_output_size // 2, task_config.num_classes),
                )
            else:  # regression
                head = nn.Sequential(
                    nn.Linear(encoder_output_size, encoder_output_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(encoder_output_size // 2, 1),
                )

            self.task_heads[task_config.name] = head

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using best practices."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the model."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Shared encoding
        encoded = self.encoder(x)

        # Task-specific outputs
        outputs = {"encoded": encoded}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(encoded)

        return outputs


class ProductionMultiTaskTrainer(GenericTrainer):
    """Production-grade multi-task trainer with advanced features."""

    def __init__(
        self,
        config: GenericTrainerConfig,
        model: MultiTaskModel,
        optimizer: torch.optim.Optimizer,
        task_configs: list[TaskConfig],
        adaptive_weighting: bool = True,
    ):
        super().__init__(config)

        self.model = model
        self.optimizer = optimizer
        self.task_configs = {tc.name: tc for tc in task_configs}
        self.adaptive_weighting = adaptive_weighting

        # Initialize loss functions
        self.loss_functions = {}
        for task_config in task_configs:
            if task_config.task_type == "classification":
                self.loss_functions[task_config.name] = nn.CrossEntropyLoss()
            else:  # regression
                self.loss_functions[task_config.name] = nn.MSELoss()

        # Task weights (can be adapted during training)
        self.task_weights = {tc.name: tc.weight for tc in task_configs}
        self.task_losses_history = defaultdict(list)

        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, Any]:
        """Enhanced training step with multi-task learning."""
        # Extract inputs and targets
        inputs = batch["inputs"]
        targets = {
            task_name: batch[f"{task_name}_targets"] for task_name in self.task_configs
        }

        # Forward pass
        outputs = self.model(inputs)

        # Compute task-specific losses
        task_losses = {}
        for task_name in self.task_configs:
            if task_name in targets and task_name in outputs:
                loss_fn = self.loss_functions[task_name]
                task_loss = loss_fn(outputs[task_name], targets[task_name])
                task_losses[task_name] = task_loss

        # Compute weighted total loss
        total_loss = self._compute_weighted_loss(task_losses)

        # Compute metrics
        metrics = self._compute_training_metrics(
            outputs, targets, task_losses, total_loss
        )

        # Update task weights if adaptive weighting is enabled
        if self.adaptive_weighting:
            self._update_adaptive_weights(task_losses)

        return metrics

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, Any]:
        """Enhanced validation step with multi-task evaluation."""
        with torch.no_grad():
            inputs = batch["inputs"]
            targets = {
                task_name: batch[f"{task_name}_targets"]
                for task_name in self.task_configs
            }

            outputs = self.model(inputs)

            # Compute losses
            task_losses = {}
            for task_name in self.task_configs:
                if task_name in targets and task_name in outputs:
                    loss_fn = self.loss_functions[task_name]
                    task_loss = loss_fn(outputs[task_name], targets[task_name])
                    task_losses[task_name] = task_loss

            total_loss = self._compute_weighted_loss(task_losses)
            return self._compute_validation_metrics(
                outputs, targets, task_losses, total_loss
            )

    def _compute_weighted_loss(
        self, task_losses: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute weighted total loss across tasks."""
        total_loss = torch.tensor(0.0, device=next(iter(task_losses.values())).device)

        for task_name, loss in task_losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * loss

        return total_loss

    def _compute_training_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        task_losses: dict[str, torch.Tensor],
        total_loss: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute comprehensive training metrics."""
        metrics = {
            "loss": total_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        # Task-specific metrics
        for task_name in self.task_configs:
            if task_name in task_losses:
                metrics[f"{task_name}_loss"] = task_losses[task_name].item()

                # Compute accuracy for classification tasks
                if (
                    task_name in outputs
                    and task_name in targets
                    and self.task_configs[task_name].task_type == "classification"
                ):
                    with torch.no_grad():
                        _, predicted = torch.max(outputs[task_name], 1)
                        accuracy = (
                            (predicted == targets[task_name]).float().mean().item()
                        )
                        metrics[f"{task_name}_accuracy"] = accuracy

        return metrics

    def _compute_validation_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        task_losses: dict[str, torch.Tensor],
        total_loss: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute validation metrics."""
        val_metrics = {"val_loss": total_loss.item()}

        for task_name in self.task_configs:
            if task_name in task_losses:
                val_metrics[f"val_{task_name}_loss"] = task_losses[task_name].item()

                if (
                    task_name in outputs
                    and task_name in targets
                    and self.task_configs[task_name].task_type == "classification"
                ):
                    with torch.no_grad():
                        _, predicted = torch.max(outputs[task_name], 1)
                        accuracy = (
                            (predicted == targets[task_name]).float().mean().item()
                        )
                        val_metrics[f"val_{task_name}_accuracy"] = accuracy

        return val_metrics

    def _update_adaptive_weights(self, task_losses: dict[str, torch.Tensor]) -> None:
        """Update task weights based on relative performance."""
        # Store loss history
        for task_name, loss in task_losses.items():
            self.task_losses_history[task_name].append(loss.item())

            # Keep only recent history
            if len(self.task_losses_history[task_name]) > 50:
                self.task_losses_history[task_name] = self.task_losses_history[
                    task_name
                ][-50:]

        # Update weights every 20 steps
        if self.global_step % 20 == 0 and len(self.task_losses_history) > 0:
            self._recompute_adaptive_weights()

    def _recompute_adaptive_weights(self) -> None:
        """Recompute adaptive task weights based on performance trends."""
        # Compute recent average losses
        recent_losses = {}
        for task_name, loss_history in self.task_losses_history.items():
            if len(loss_history) >= 10:
                recent_losses[task_name] = sum(loss_history[-10:]) / 10

        if not recent_losses:
            return

        # Update weights (tasks with higher loss get higher weight)
        for task_name, recent_loss in recent_losses.items():
            new_weight = (recent_loss / sum(recent_losses.values())) * len(
                recent_losses
            )

            # Apply momentum
            momentum = 0.9
            current_weight = self.task_weights.get(task_name, 1.0)
            self.task_weights[task_name] = (
                momentum * current_weight + (1 - momentum) * new_weight
            )

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """Enhanced epoch end with multi-task analysis."""
        super().on_epoch_end(epoch, logs)

        self.logger.info(f"Epoch {epoch + 1} Summary:")
        self.logger.info(f"  Total Loss: {logs.get('loss', 0):.4f}")

        # Task-specific summaries
        for task_name in self.task_configs:
            task_loss = logs.get(f"{task_name}_loss", 0)
            task_acc = logs.get(f"{task_name}_accuracy", None)
            weight = self.task_weights[task_name]

            self.logger.info(
                f"  {task_name}: loss={task_loss:.4f}, weight={weight:.3f}"
            )
            if task_acc is not None:
                self.logger.info(f"    accuracy={task_acc:.4f}")


def create_demo_multi_task_data(
    batch_size: int = 32, num_batches: int = 100
) -> list[dict[str, torch.Tensor]]:
    """Create demo multi-task dataset."""
    data = []

    for _ in range(num_batches):
        inputs = torch.randn(batch_size, 28, 28)
        task1_targets = torch.randint(0, 10, (batch_size,))  # 10-class classification
        task2_targets = torch.randint(0, 2, (batch_size,))  # Binary classification
        task3_targets = torch.randn(batch_size)  # Regression

        batch = {
            "inputs": inputs,
            "classification_targets": task1_targets,
            "binary_targets": task2_targets,
            "regression_targets": task3_targets,
        }
        data.append(batch)

    return data


def main():
    """Main function demonstrating advanced multi-task learning."""
    print("🏭 Advanced Multi-Task Learning - Production Setup")
    print("=" * 60)

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Define task configurations
    task_configs = [
        TaskConfig(
            name="classification",
            weight=1.0,
            num_classes=10,
            task_type="classification",
        ),
        TaskConfig(
            name="binary", weight=0.5, num_classes=2, task_type="classification"
        ),
        TaskConfig(name="regression", weight=0.8, task_type="regression"),
    ]

    # Create model
    print("🔧 Creating multi-task model...")
    model = MultiTaskModel(
        input_size=784,
        hidden_sizes=[512, 256, 128],
        task_configs=task_configs,
        dropout_rate=0.1,
    )

    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Configure trainer
    trainer_config = GenericTrainerConfig(
        max_epochs=10,
        checkpoint_dir="./production_multitask_checkpoints",
        save_every_n_epochs=2,
    )

    # Create trainer
    trainer = ProductionMultiTaskTrainer(
        config=trainer_config,
        model=model,
        optimizer=optimizer,
        task_configs=task_configs,
        adaptive_weighting=True,
    )

    # Create dataset
    print("📊 Creating multi-task dataset...")
    train_data = create_demo_multi_task_data(batch_size=64, num_batches=100)
    val_data = create_demo_multi_task_data(batch_size=64, num_batches=20)

    print("\n🚀 Starting advanced multi-task training...")

    try:
        trainer.fit(
            model=model,
            optimizer=optimizer,
            train_loader=train_data,
            val_loader=val_data,
        )

        print("✅ Multi-task training completed successfully!")

        # Display final task weights
        print("\n📊 Final Task Weights:")
        for task_name, weight in trainer.task_weights.items():
            print(f"   {task_name}: {weight:.3f}")

    except Exception as e:
        print(f"❌ Training failed: {e}")


if __name__ == "__main__":
    main()
