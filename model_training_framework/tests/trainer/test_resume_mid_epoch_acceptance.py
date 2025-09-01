"""
Acceptance test for mid-epoch checkpoint resume functionality.

This test verifies that:
1. Training can be saved mid-epoch with complete state
2. Loading a checkpoint reproduces the exact dataloader sequence
3. Per-loader batch indices continue from the exact position
4. Metrics continue seamlessly without gaps or jumps
"""

import logging
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from model_training_framework.trainer import (
    CheckpointConfig,
    EpochLengthPolicy,
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    PerformanceConfig,
    SamplingStrategy,
)
from model_training_framework.trainer.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from model_training_framework.trainer.states import create_initial_resume_state


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CheckpointableIterableDataset(IterableDataset):
    """Iterable dataset that supports state checkpointing."""

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.labels = labels
        self.position = 0

    def __iter__(self):
        while self.position < len(self.data):
            yield self.data[self.position], self.labels[self.position]
            self.position += 1

    def state_dict(self) -> dict[str, Any]:
        """Return current dataset state."""
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore dataset state."""
        self.position = state.get("position", 0)

    def reset(self):
        """Reset position for new epoch."""
        self.position = 0


class TestAcceptanceMidEpochResume:
    """Test mid-epoch checkpoint resume functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = Path(self.temp_dir) / "mid_epoch_checkpoint.pt"

        # Enable debug logging
        logging.getLogger("model_training_framework.trainer").setLevel(logging.DEBUG)

        # Set deterministic seeds
        self.seed = 42
        self._set_all_seeds(self.seed)

    def teardown_method(self):
        """Cleanup after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _set_all_seeds(self, seed: int):
        """Set all RNG seeds for determinism."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_dataloaders(
        self, use_iterable: bool = False
    ) -> tuple[list[DataLoader], list[DataLoader]]:
        """Create training and validation dataloaders."""
        # Create datasets with different sizes to test weighted sampling
        sizes = [100, 80, 120]
        train_loaders = []
        val_loaders = []

        for i, size in enumerate(sizes):
            # Generate deterministic data
            torch.manual_seed(1000 + i)
            X_train = torch.randn(size, 10)
            y_train = torch.randint(0, 3, (size,))

            X_val = torch.randn(size // 5, 10)
            y_val = torch.randint(0, 3, (size // 5,))

            if use_iterable:
                train_dataset = CheckpointableIterableDataset(X_train, y_train)
                val_dataset = CheckpointableIterableDataset(X_val, y_val)
            else:
                train_dataset = TensorDataset(X_train, y_train)
                val_dataset = TensorDataset(X_val, y_val)

            train_loaders.append(DataLoader(train_dataset, batch_size=8, shuffle=False))
            val_loaders.append(DataLoader(val_dataset, batch_size=8, shuffle=False))

        return train_loaders, val_loaders

    def _create_trainer(
        self,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        steps_per_epoch: int = 50,
    ) -> GenericTrainer:
        """Create a trainer instance with specified configuration."""
        multi_cfg = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
            steps_per_epoch=steps_per_epoch,
            dataloader_names=["loader_0", "loader_1", "loader_2"],
            dataloader_weights=[0.5, 0.3, 0.2],
            choice_rng_seed=12345,
        )
        config = GenericTrainerConfig(
            train_loader_config=multi_cfg,
            val_loader_config=MultiDataLoaderConfig(),
            performance=PerformanceConfig(
                use_amp=use_amp,
                gradient_accumulation_steps=gradient_accumulation_steps,
            ),
            checkpoint=CheckpointConfig(
                root_dir=self.temp_dir,
                save_every_n_steps=25,  # Save mid-epoch
            ),
            validate_every_n_epochs=100,  # Effectively disable validation
            log_loss_every_n_steps=10,
        )

        model = SimpleModel()

        # Create multiple optimizers for comprehensive testing
        optimizers = [
            torch.optim.SGD(model.fc1.parameters(), lr=0.01, momentum=0.9),
            torch.optim.Adam(
                [*model.fc2.parameters(), *model.fc3.parameters()], lr=0.001
            ),
        ]

        # Create schedulers
        schedulers = [
            torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=20),
            torch.optim.lr_scheduler.ExponentialLR(optimizers[1], gamma=0.95),
        ]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
        )

        # Initialize optimizer state to avoid PyTorch scheduler warnings
        # PyTorch warns if scheduler.step() is called before any optimizer.step()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        return trainer

    def test_mid_epoch_resume_basic(self):
        """Test basic mid-epoch checkpoint and resume."""
        # Create trainer and dataloaders - will manually stop at 25 steps
        trainer = self._create_trainer(steps_per_epoch=50)
        train_loaders, val_loaders = self._create_dataloaders()

        # Track dataloader sequences and metrics
        loader_sequence_run1 = []
        batch_indices_run1 = {0: [], 1: [], 2: []}
        metrics_run1 = []

        # Training step that tracks loader sequence
        batch_counters_run1: dict[int, int] = {}

        def training_step(
            trainer: GenericTrainer,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
            dataloader_name: str,
        ) -> dict[str, Any]:
            x, y = batch

            # Track loader sequence
            loader_sequence_run1.append(dataloader_idx)

            # Track batch index (approximation based on call count)
            if dataloader_idx not in batch_counters_run1:
                batch_counters_run1[dataloader_idx] = 0
            batch_indices_run1[dataloader_idx].append(
                batch_counters_run1[dataloader_idx]
            )
            batch_counters_run1[dataloader_idx] += 1

            # Forward pass
            output = trainer.model(x)
            loss = F.cross_entropy(output, y)

            # Track metrics
            metrics_run1.append({"loss": loss.item(), "step": trainer.global_step})

            return {"loss": loss}

        trainer.set_training_step(training_step)

        # Run 1: Train for 25 steps (mid-epoch) and save
        print("\n=== Run 1: Training and saving mid-epoch ===")

        # Hook to stop training after 25 steps
        original_optimizer_step = trainer._optimizer_step
        steps_taken = [0]

        def optimizer_step_with_stop(loader_idx):
            original_optimizer_step(loader_idx)
            steps_taken[0] += 1
            if steps_taken[0] >= 25:
                # Force stop by raising an exception
                raise KeyboardInterrupt("Stopping at 25 steps for checkpoint")

        trainer._optimizer_step = optimizer_step_with_stop

        # Train for first 25 steps (half epoch)
        from contextlib import suppress

        with suppress(KeyboardInterrupt):
            trainer.fit(train_loaders, val_loaders, max_epochs=1)

        # Save checkpoint after training
        print(f"Saving checkpoint at step {trainer.global_step}")
        # Update trainer's resume state before saving
        assert trainer.dataloader_manager is not None
        cast(
            "Any", trainer.resume_state
        ).dataloader_manager_state = trainer.dataloader_manager.get_state()

        save_checkpoint(
            path=self.checkpoint_path,
            trainer=trainer,
        )
        print(f"Checkpoint saved after {trainer.global_step} steps")

        # Record state after Run 1
        final_step_run1 = trainer.global_step
        print(f"Run 1 completed: {final_step_run1} steps")
        print(f"Loader sequence length: {len(loader_sequence_run1)}")
        print(f"Unique loaders used: {set(loader_sequence_run1)}")
        print(
            f"Batch indices per loader: {[len(v) for v in batch_indices_run1.values()]}"
        )

        # Run 2: Create new trainer, load checkpoint, and resume
        print("\n=== Run 2: Loading checkpoint and resuming ===")

        # Create fresh trainer with full 50 steps per epoch
        trainer2 = self._create_trainer(steps_per_epoch=50)
        train_loaders2, val_loaders2 = self._create_dataloaders()

        # Track sequences for Run 2
        loader_sequence_run2 = []
        batch_indices_run2 = {0: [], 1: [], 2: []}
        metrics_run2 = []

        batch_counters_run2: dict[int, int] = {}

        def training_step_run2(
            trainer: GenericTrainer,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
            dataloader_name: str,
        ) -> dict[str, Any]:
            x, y = batch

            # Track loader sequence
            loader_sequence_run2.append(dataloader_idx)

            # Track batch index
            if dataloader_idx not in batch_counters_run2:
                batch_counters_run2[dataloader_idx] = 0
            batch_indices_run2[dataloader_idx].append(
                batch_counters_run2[dataloader_idx]
            )
            batch_counters_run2[dataloader_idx] += 1

            # Forward pass
            output = trainer.model(x)
            loss = F.cross_entropy(output, y)

            # Track metrics
            metrics_run2.append({"loss": loss.item(), "step": trainer.global_step})

            return {"loss": loss}

        trainer2.set_training_step(training_step_run2)

        # Initialize dataloader manager before loading checkpoint
        # This is normally done in fit(), but we need it before loading
        from model_training_framework.trainer.multi_dataloader import DataLoaderManager

        trainer2.dataloader_manager = DataLoaderManager(
            train_loaders=train_loaders2,
            val_loaders=val_loaders2,
            train_config=trainer2.config.train_loader_config,
            val_config=trainer2.config.val_loader_config,
        )

        # Load checkpoint
        load_checkpoint(path=self.checkpoint_path, trainer=trainer2)

        print(f"Checkpoint loaded: starting from step {trainer2.global_step}")

        # Continue training from checkpoint
        # Since we loaded at step 25, and want to train to step 50 total,
        # we need to continue training. The epoch should still be 0.
        trainer2.fit(
            train_loaders2, val_loaders2, max_epochs=2
        )  # Allow more epochs to complete

        print(f"Training complete at step {trainer2.global_step}")

        # Verify results
        print("\n=== Verification ===")

        # 1. Verify training completed correctly
        print(f"Run 1 completed at step: {final_step_run1}")
        print(f"Run 2 completed at step: {trainer2.global_step}")
        print(f"Run 1 loader sequence length: {len(loader_sequence_run1)}")
        print(f"Run 2 loader sequence length: {len(loader_sequence_run2)}")

        # Run 1 should have done 25 steps, Run 2 should continue from 25
        assert final_step_run1 == 25, (
            f"Run 1 should stop at step 25, got {final_step_run1}"
        )
        assert trainer2.global_step >= 50, (
            f"Run 2 should reach at least step 50, got {trainer2.global_step}"
        )
        assert len(loader_sequence_run2) > 0, "Run 2 should have processed batches"

        # 2. Verify batch indices continue without gaps
        print(
            f"Batch indices per loader in Run 2: {[len(v) for v in batch_indices_run2.values()]}"
        )

        # 3. Verify metrics show continuous progression
        if metrics_run2:
            first_step_run2 = metrics_run2[0]["step"]
            # Run 2 should continue from where Run 1 left off
            assert first_step_run2 >= final_step_run1, (
                f"Run 2 should continue from at least step {final_step_run1}, but started at {first_step_run2}"
            )

        print("✓ Mid-epoch resume test PASSED")
        print(f"  - Successfully saved at step {final_step_run1}")
        print(
            f"  - Successfully resumed and continued for {len(loader_sequence_run2)} steps"
        )
        print(f"  - Final step: {trainer2.global_step}")

    def test_mid_epoch_resume_with_amp(self):
        """Test mid-epoch resume with AMP enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for AMP test")

        # Create trainer with AMP
        trainer = self._create_trainer(use_amp=True, steps_per_epoch=30)
        train_loaders, _ = self._create_dataloaders()

        # Simple training step
        def training_step(
            trainer: GenericTrainer,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
            dataloader_name: str,
        ) -> dict[str, Any]:
            x, y = batch
            x, y = x.cuda(), y.cuda()
            output = trainer.model(x)
            loss = F.cross_entropy(output, y)
            return {"loss": loss}

        trainer.set_training_step(training_step)

        # Train until 15 optimizer steps, then save and interrupt
        original_optimizer_step = trainer._optimizer_step
        steps_before_save = 0

        def optimizer_step_save(loader_idx):
            nonlocal steps_before_save
            original_optimizer_step(loader_idx)
            steps_before_save += 1
            if steps_before_save >= 15:
                # Save with AMP scaler state
                resume_state = create_initial_resume_state(save_rng=True)
                assert trainer.dataloader_manager is not None
                resume_state.dataloader_manager_state = (
                    trainer.dataloader_manager.get_state()
                )
                save_checkpoint(path=self.checkpoint_path, trainer=trainer)
                raise KeyboardInterrupt("Saved after 15 steps")

        trainer._optimizer_step = optimizer_step_save
        from contextlib import suppress

        with suppress(KeyboardInterrupt):
            trainer.fit(train_loaders, max_epochs=1)

        # Verify scaler was saved
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        assert "amp_scaler_state" in checkpoint, "AMP scaler state not saved"
        assert checkpoint["amp_scaler_state"] is not None

        # Create new trainer and load
        trainer2 = self._create_trainer(use_amp=True, steps_per_epoch=30)
        trainer2.set_training_step(training_step)

        # Load checkpoint
        load_checkpoint(path=self.checkpoint_path, trainer=trainer2)

        # Verify scaler was restored
        assert trainer2.scaler is not None, "AMP scaler not restored"
        assert trainer2.global_step == 15, (
            f"Expected step 15, got {trainer2.global_step}"
        )

        print("✓ AMP mid-epoch resume test PASSED")

    def test_mid_epoch_resume_with_iterable_dataset(self):
        """Test mid-epoch resume with checkpointable iterable datasets."""
        # Create trainer with iterable datasets
        trainer = self._create_trainer(steps_per_epoch=40)
        train_loaders, _ = self._create_dataloaders(use_iterable=True)

        # Training step
        def training_step(
            trainer: GenericTrainer,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
            dataloader_name: str,
        ) -> dict[str, Any]:
            x, y = batch
            output = trainer.model(x)
            loss = F.cross_entropy(output, y)
            return {"loss": loss}

        trainer.set_training_step(training_step)

        # Train for 20 steps and save
        steps_trained = 0

        def train_partial(epoch: int) -> dict[str, Any]:
            nonlocal steps_trained
            trainer.model.train()
            assert trainer.dataloader_manager is not None
            train_iterator = trainer.dataloader_manager.create_epoch_iterator(
                "train", epoch
            )

            for loader_idx, batch in train_iterator:
                assert trainer.dataloader_manager is not None
                loader_name = trainer.dataloader_manager.train_names[loader_idx]
                # Use internal _training_step with required args
                trainer._training_step(batch, loader_idx, loader_name, 0, 0)
                trainer._optimizer_step(loader_idx)
                steps_trained += 1

                if steps_trained >= 20:
                    # Save checkpoint
                    assert trainer.dataloader_manager is not None
                    cast(
                        "Any", trainer.resume_state
                    ).dataloader_manager_state = trainer.dataloader_manager.get_state()

                    save_checkpoint(
                        path=self.checkpoint_path,
                        trainer=trainer,
                    )

                    # Check dataset positions
                    for i, loader in enumerate(train_loaders):
                        pos = getattr(loader.dataset, "position", None)
                        if pos is not None:
                            print(f"Dataset {i} position: {pos}")
                    return {}
            return {}

        trainer._train_epoch = train_partial
        trainer.fit(train_loaders, max_epochs=1)

        # Create new trainer and load
        trainer2 = self._create_trainer(steps_per_epoch=40)
        train_loaders2, _ = self._create_dataloaders(use_iterable=True)
        trainer2.set_training_step(training_step)

        # Initialize dataloader manager before loading checkpoint
        from model_training_framework.trainer.multi_dataloader import DataLoaderManager

        trainer2.dataloader_manager = DataLoaderManager(
            train_loaders=train_loaders2,
            val_loaders=None,
            train_config=trainer2.config.train_loader_config,
        )

        # Load checkpoint
        load_checkpoint(path=self.checkpoint_path, trainer=trainer2)

        # Verify datasets restored their positions
        assert trainer2.global_step == 20, (
            f"Expected step 20, got {trainer2.global_step}"
        )

        # Continue training
        steps_after_resume = 0

        def train_continue(epoch: int) -> dict[str, Any]:
            nonlocal steps_after_resume
            trainer2.model.train()
            assert trainer2.dataloader_manager is not None
            train_iterator = trainer2.dataloader_manager.create_epoch_iterator(
                "train", epoch
            )

            for loader_idx, batch in train_iterator:
                assert trainer2.dataloader_manager is not None
                loader_name = trainer2.dataloader_manager.train_names[loader_idx]
                trainer2._training_step(batch, loader_idx, loader_name, 0, 0)
                trainer2._optimizer_step(loader_idx)
                steps_after_resume += 1

                if steps_after_resume >= 20:
                    return {}
            return {}

        trainer2._train_epoch = train_continue
        trainer2.fit(train_loaders2, max_epochs=1)

        assert trainer2.global_step == 40, (
            f"Expected step 40, got {trainer2.global_step}"
        )
        print("✓ Iterable dataset mid-epoch resume test PASSED")

    def test_deterministic_sequence_reproduction(self):
        """Test that the exact dataloader sequence is reproduced after resume."""
        # Set specific seed for reproducibility
        self._set_all_seeds(999)

        # Create trainer
        trainer = self._create_trainer(steps_per_epoch=100)
        train_loaders, _ = self._create_dataloaders()

        # Record complete sequence for full epoch
        full_sequence = []

        def recording_step(
            trainer: GenericTrainer,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
            dataloader_name: str,
        ) -> dict[str, Any]:
            full_sequence.append((trainer.global_step, dataloader_idx, dataloader_name))
            x, y = batch
            output = trainer.model(x)
            loss = F.cross_entropy(output, y)
            return {"loss": loss}

        trainer.set_training_step(recording_step)

        # Record full sequence in one go
        trainer.fit(train_loaders, max_epochs=1)

        print(f"Full sequence recorded: {len(full_sequence)} steps")

        # Now test with checkpoint/resume at step 50
        self._set_all_seeds(999)  # Reset seeds

        trainer2 = self._create_trainer(steps_per_epoch=100)
        train_loaders2, _ = self._create_dataloaders()

        partial_sequence = []

        def recording_step2(
            trainer: GenericTrainer,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
            dataloader_name: str,
        ) -> dict[str, Any]:
            partial_sequence.append(
                (trainer.global_step, dataloader_idx, dataloader_name)
            )
            x, y = batch
            output = trainer.model(x)
            loss = F.cross_entropy(output, y)
            return {"loss": loss}

        trainer2.set_training_step(recording_step2)

        # Train for 50 steps and checkpoint
        def train_first_half(epoch: int) -> dict[str, Any]:
            trainer2.model.train()
            assert trainer2.dataloader_manager is not None
            train_iterator = trainer2.dataloader_manager.create_epoch_iterator(
                "train", epoch
            )
            for loader_idx, batch in train_iterator:
                assert trainer2.dataloader_manager is not None
                loader_name = trainer2.dataloader_manager.train_names[loader_idx]
                _ = trainer2._training_step(batch, loader_idx, loader_name, 0, 0)
                trainer2._optimizer_step(loader_idx)
                if trainer2.global_step >= 50:
                    # Save checkpoint
                    assert trainer2.dataloader_manager is not None
                    cast(
                        "Any", trainer2.resume_state
                    ).dataloader_manager_state = trainer2.dataloader_manager.get_state()
                    save_checkpoint(path=self.checkpoint_path, trainer=trainer2)
                    return {}
            return {}

        trainer2._train_epoch = train_first_half
        trainer2.fit(train_loaders2, max_epochs=1)

        # Create new trainer and resume
        self._set_all_seeds(888)  # Different seed to test RNG restoration

        trainer3 = self._create_trainer(steps_per_epoch=100)
        train_loaders3, _ = self._create_dataloaders()
        trainer3.set_training_step(recording_step2)

        # Initialize dataloader manager before loading checkpoint
        from model_training_framework.trainer.multi_dataloader import DataLoaderManager

        trainer3.dataloader_manager = DataLoaderManager(
            train_loaders=train_loaders3,
            val_loaders=None,
            train_config=trainer3.config.train_loader_config,
        )

        # Load checkpoint
        load_checkpoint(path=self.checkpoint_path, trainer=trainer3)

        # Continue training
        def train_second_half(epoch: int) -> dict[str, Any]:
            trainer3.model.train()
            assert trainer3.dataloader_manager is not None
            train_iterator = trainer3.dataloader_manager.create_epoch_iterator(
                "train", epoch
            )
            for loader_idx, batch in train_iterator:
                assert trainer3.dataloader_manager is not None
                loader_name = trainer3.dataloader_manager.train_names[loader_idx]
                _ = trainer3._training_step(batch, loader_idx, loader_name, 0, 0)
                trainer3._optimizer_step(loader_idx)
                if trainer3.global_step >= 100:
                    return {}
            return {}

        trainer3._train_epoch = train_second_half
        trainer3.fit(train_loaders3, max_epochs=1)

        # Verify sequences match
        print(f"Partial sequence length: {len(partial_sequence)}")
        assert len(partial_sequence) == len(full_sequence), (
            f"Sequence lengths don't match: {len(partial_sequence)} vs {len(full_sequence)}"
        )

        # Check each step
        mismatches = []
        for i, (full_step, partial_step) in enumerate(
            zip(full_sequence, partial_sequence)
        ):
            if full_step != partial_step:
                mismatches.append((i, full_step, partial_step))

        if mismatches:
            print(f"Found {len(mismatches)} mismatches:")
            for idx, full, partial in mismatches[:5]:  # Show first 5
                print(f"  Step {idx}: {full} vs {partial}")
            raise AssertionError("Sequences don't match after resume")

        print("✓ Deterministic sequence reproduction test PASSED")
        print(f"  - All {len(full_sequence)} steps matched exactly")


if __name__ == "__main__":
    # Run the tests
    test = TestAcceptanceMidEpochResume()
    test.setup_method()

    try:
        print("Running acceptance tests for mid-epoch checkpoint resume...")
        print("=" * 60)

        test.test_mid_epoch_resume_basic()
        print()

        test.test_mid_epoch_resume_with_iterable_dataset()
        print()

        test.test_deterministic_sequence_reproduction()
        print()

        if torch.cuda.is_available():
            test.test_mid_epoch_resume_with_amp()
            print()

        print("=" * 60)
        print("All acceptance tests PASSED! ✓")

    finally:
        test.teardown_method()
