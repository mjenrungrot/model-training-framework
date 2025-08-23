"""
Preemption Handling Example

This example demonstrates how the framework handles SLURM preemption
signals and automatically resumes training from exact checkpoint states.
"""

import os
from pathlib import Path
import signal
import threading
import time
from typing import Any

import torch
from torch import nn

from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
)
from model_training_framework.trainer.states import (
    ResumeState,
    capture_rng_state,
    restore_rng_state,
)


class PreemptionDemoModel(nn.Module):
    """Simple model for preemption demonstration."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))


class PreemptionAwareTrainer(GenericTrainer):
    """Trainer with enhanced preemption handling demonstration."""

    def __init__(
        self,
        config: GenericTrainerConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__(config)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss()

        # Enhanced preemption tracking
        self.preemption_count = 0
        self.last_preemption_time = None
        self.resume_history = []

        # Demonstrate manual signal handling (framework handles this automatically)
        self.original_signal_handler = None
        self.preemption_received = False

    def training_step(self, batch, batch_idx: int) -> dict[str, Any]:
        """Training step with preemption awareness."""

        # Check for preemption signal (normally handled by framework)
        if self.preemption_received:
            print("⚠️  Preemption signal received during training step!")
            self.preemption_received = False

        # Normal training step
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)

        # Calculate accuracy
        _, predicted = torch.max(y_pred, 1)
        accuracy = (predicted == y).float().mean()

        # Simulate some processing time
        time.sleep(0.01)  # 10ms per batch

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "batch_idx": batch_idx,
        }

    def validation_step(self, batch, batch_idx: int) -> dict[str, Any]:
        """Validation step with preemption awareness."""

        with torch.no_grad():
            x, y = batch
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)

            _, predicted = torch.max(y_pred, 1)
            accuracy = (predicted == y).float().mean()

        return {"val_loss": loss, "val_accuracy": accuracy.item()}

    def on_preemption_signal(self, signum: int, frame):
        """Handle preemption signal (demonstration - framework handles this)."""
        print(f"\n🚨 Preemption signal received: {signum}")
        print(f"   Current phase: {self.current_phase}")
        print(f"   Current epoch: {self.current_epoch}")
        print(f"   Global step: {self.global_step}")

        self.preemption_received = True
        self.preemption_count += 1
        self.last_preemption_time = time.time()

        # Trigger checkpoint save
        self.save_checkpoint(emergency=True)

        print("💾 Emergency checkpoint saved")
        print("🔄 Preparing for graceful shutdown...")

    def on_resume_from_checkpoint(self, resume_state: ResumeState):
        """Called when resuming from checkpoint."""
        super().on_resume_from_checkpoint(resume_state)

        self.resume_history.append(
            {
                "timestamp": time.time(),
                "resumed_from_phase": resume_state.phase,
                "resumed_from_epoch": resume_state.epoch,
                "resumed_from_step": resume_state.global_step,
            }
        )

        print("🔄 Resumed from checkpoint:")
        print(f"   Phase: {resume_state.phase}")
        print(f"   Epoch: {resume_state.epoch}")
        print(f"   Global step: {resume_state.global_step}")

        if resume_state.rng:
            print("   ✅ RNG state restored for deterministic resume")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]):
        """Enhanced epoch end logging."""
        super().on_epoch_end(epoch, logs)

        print(f"📊 Epoch {epoch + 1} completed:")
        print(f"   Loss: {logs.get('loss', 0):.4f}")
        print(f"   Accuracy: {logs.get('accuracy', 0):.4f}")

        if logs.get("val_loss"):
            print(f"   Val Loss: {logs['val_loss']:.4f}")
            print(f"   Val Accuracy: {logs.get('val_accuracy', 0):.4f}")

        if self.preemption_count > 0:
            print(f"   Preemptions handled: {self.preemption_count}")

    def on_training_end(self, logs: dict[str, Any]):
        """Training end summary."""
        super().on_training_end(logs)

        print("\n🏁 Training Summary:")
        print(f"   Total preemptions handled: {self.preemption_count}")
        print(f"   Resume count: {len(self.resume_history)}")

        if self.resume_history:
            print("   Resume history:")
            for i, resume_info in enumerate(self.resume_history, 1):
                print(
                    f"      {i}. Phase: {resume_info['resumed_from_phase']}, "
                    f"Epoch: {resume_info['resumed_from_epoch']}, "
                    f"Step: {resume_info['resumed_from_step']}"
                )


def simulate_preemption_signal(pid: int, delay: float = 30.0):
    """Simulate SLURM preemption signal after delay."""

    def send_signal():
        time.sleep(delay)
        print(f"\n⚡ Simulating SLURM preemption signal to PID {pid}")
        try:
            os.kill(pid, signal.SIGUSR1)
        except ProcessLookupError:
            print("   Process not found (may have already completed)")

    thread = threading.Thread(target=send_signal, daemon=True)
    thread.start()
    return thread


def create_dummy_data(batch_size=32, num_batches=200):
    """Create dummy data for extended training."""
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, 28, 28)
        y = torch.randint(0, 10, (batch_size,))
        data.append((x, y))
    return data


def main():
    """Main preemption handling demonstration."""

    print("🚨 Preemption Handling Example")
    print("This example demonstrates automatic recovery from SLURM preemption signals")

    # Setup model and training components
    model = PreemptionDemoModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Configure trainer for preemption handling
    trainer_config = GenericTrainerConfig(
        max_epochs=50,  # Long training to increase preemption chance
        checkpoint_dir="./checkpoints/preemption_demo",
        save_every_n_epochs=2,  # Frequent checkpoints
        preemption_timeout=120,  # 2 minutes before SLURM kills job
        save_rng_state=True,  # Enable deterministic resume
    )

    # Create trainer
    trainer = PreemptionAwareTrainer(
        config=trainer_config, model=model, optimizer=optimizer
    )

    # Create extended dataset to ensure training takes time
    print("📊 Creating extended dataset for demonstration...")
    train_data = create_dummy_data(batch_size=32, num_batches=200)  # ~6-7 minutes
    val_data = create_dummy_data(batch_size=32, num_batches=40)

    print(f"   Training batches: {len(train_data)} (estimated time: ~10 minutes)")
    print(f"   Validation batches: {len(val_data)}")

    # Simulate preemption signal during training
    current_pid = os.getpid()

    print(f"🔧 Setting up simulated preemption (PID: {current_pid})")
    print("   Will send SIGUSR1 signal after 30 seconds")

    # Start background thread to simulate preemption
    simulate_preemption_signal(current_pid, delay=30.0)

    # Demonstrate checkpoint loading if exists
    checkpoint_dir = Path(trainer_config.checkpoint_dir)
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.ckpt")):
        print("🔄 Found existing checkpoints - training will resume automatically")
    else:
        print("🆕 No existing checkpoints - starting fresh training")

    print("\n🚀 Starting training with preemption handling...")

    try:
        # This training will be interrupted by the simulated preemption signal
        trainer.fit(
            model=model,
            optimizer=optimizer,
            train_loader=train_data,
            val_loader=val_data,
        )

        print("✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("💾 Checkpoint should be saved automatically")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")

        # Show checkpoint status
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoints:
                print(f"💾 Found {len(checkpoints)} checkpoint(s) for recovery:")
                for ckpt in checkpoints:
                    print(f"   - {ckpt.name}")

    # Demonstrate manual checkpoint inspection
    demonstrate_checkpoint_inspection(checkpoint_dir)


def demonstrate_checkpoint_inspection(checkpoint_dir: Path):
    """Demonstrate how to inspect and load checkpoints manually."""

    print("\n🔍 Checkpoint Inspection Demo:")

    if not checkpoint_dir.exists():
        print("   No checkpoint directory found")
        return

    checkpoints = list(checkpoint_dir.glob("*.ckpt"))

    if not checkpoints:
        print("   No checkpoints found")
        return

    print(f"   Found {len(checkpoints)} checkpoint(s):")

    # Load and inspect the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"   Latest: {latest_checkpoint.name}")

    try:
        # Load checkpoint data
        checkpoint_data = torch.load(latest_checkpoint, map_location="cpu")

        print("   Checkpoint contents:")
        for key in checkpoint_data:
            if key == "resume_state":
                resume_state = checkpoint_data[key]
                print(f"      {key}:")
                print(f"         Phase: {resume_state.phase}")
                print(f"         Epoch: {resume_state.epoch}")
                print(f"         Global step: {resume_state.global_step}")
                print(f"         Has RNG state: {resume_state.rng is not None}")
            else:
                value = checkpoint_data[key]
                if hasattr(value, "shape"):
                    print(f"      {key}: {type(value).__name__} {value.shape}")
                else:
                    print(f"      {key}: {type(value).__name__}")

    except Exception as e:
        print(f"      Error loading checkpoint: {e}")


def demonstrate_deterministic_resume():
    """Demonstrate deterministic resume with RNG state."""

    print("\n🎲 Deterministic Resume Demo:")

    # Set initial seeds
    torch.manual_seed(42)

    # Capture initial RNG state
    initial_rng = capture_rng_state()
    print("   ✅ Captured initial RNG state")

    # Generate some random numbers
    random_numbers_1 = torch.randn(5)
    print(f"   Random sequence 1: {random_numbers_1}")

    # Restore RNG state and generate again
    restore_rng_state(initial_rng)
    random_numbers_2 = torch.randn(5)
    print(f"   Random sequence 2: {random_numbers_2}")

    # Check if they're identical
    if torch.allclose(random_numbers_1, random_numbers_2):
        print("   ✅ Deterministic resume verified - sequences are identical")
    else:
        print("   ❌ Deterministic resume failed - sequences differ")


if __name__ == "__main__":
    main()
    demonstrate_deterministic_resume()
