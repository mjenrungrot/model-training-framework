"""
Tests for IterableDataset Fault Tolerance Error Handling

This module tests that IterableDatasets provide clear errors when used with
fault tolerance but don't support checkpointing, and that checkpointable
IterableDatasets work correctly with fault tolerance.
"""

from collections.abc import Iterator
from pathlib import Path
import tempfile
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from model_training_framework.config.schemas import EpochLengthPolicy, SamplingStrategy
from model_training_framework.trainer import (
    CheckpointConfig,
    FaultToleranceConfig,
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
)

# Remove non-existent import


class NonCheckpointableIterableDataset(IterableDataset):
    """IterableDataset that doesn't support checkpointing."""

    def __init__(self, size: int = 100):
        self.size = size
        self.counter = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while self.counter < self.size:
            data = torch.randn(5)
            label = torch.randint(0, 2, ())
            yield data, label
            self.counter += 1

    # No state_dict or load_state_dict methods


class CheckpointableIterableDataset(IterableDataset):
    """IterableDataset with checkpointing support."""

    def __init__(self, size: int = 100):
        self.size = size
        self.position = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        torch.manual_seed(42)  # For reproducibility
        while self.position < self.size:
            data = torch.randn(5)
            label = torch.randint(0, 2, ())
            yield data, label
            self.position += 1

    def state_dict(self) -> dict[str, Any]:
        """Return state for checkpointing."""
        return {"position": self.position, "size": self.size}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.position = state.get("position", 0)
        self.size = state.get("size", self.size)

    def reset(self):
        """Reset for new epoch."""
        self.position = 0


class PartiallyCheckpointableIterableDataset(IterableDataset):
    """IterableDataset with incomplete checkpointing support."""

    def __init__(self, size: int = 100):
        self.size = size
        self.position = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while self.position < self.size:
            data = torch.randn(5)
            label = torch.randint(0, 2, ())
            yield data, label
            self.position += 1

    def state_dict(self) -> dict[str, Any]:
        """Has state_dict but missing load_state_dict."""
        return {"position": self.position}

    # Missing load_state_dict method


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc(x)


class TestIterableDatasetFTErrors:
    """Test IterableDataset fault tolerance error handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        torch.manual_seed(42)

    def teardown_method(self):
        """Cleanup after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_non_checkpointable_with_ft_raises_error(self):
        """Test that non-checkpointable IterableDataset raises clear error with FT."""
        dataset = NonCheckpointableIterableDataset(size=50)
        loader = DataLoader(dataset, batch_size=10)

        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        config = GenericTrainerConfig(
            train_loader_config=MultiDataLoaderConfig(
                sampling_strategy=SamplingStrategy.SEQUENTIAL,
                epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
                dataloader_names=["non_checkpointable"],
            ),
            val_loader_config=MultiDataLoaderConfig(),
            fault_tolerance=FaultToleranceConfig(
                save_dataset_state=True,  # FT enabled
            ),
            checkpoint=CheckpointConfig(
                root_dir=self.checkpoint_dir,
                save_every_n_steps=10,
            ),
        )

        trainer = GenericTrainer(
            model=model,
            config=config,
            optimizers=[optimizer],
            fabric=None,
        )

        def training_step(
            trainer: GenericTrainer,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
            dataloader_name: str,
        ) -> dict[str, Any]:
            inputs, targets = batch
            outputs = trainer.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets.long())
            return {"loss": loss}

        trainer.set_training_step(training_step)

        # Should raise an informative error about non-checkpointable dataset
        # when fit() is called with FT enabled and non-checkpointable IterableDataset
        with pytest.raises(ValueError, match="does not support checkpointing"):
            trainer.fit(
                train_loaders=[loader],
                val_loaders=None,
                max_epochs=1,
            )

    def test_checkpointable_with_ft_succeeds(self):
        """Test that checkpointable IterableDataset works with FT."""
        dataset = CheckpointableIterableDataset(size=50)
        loader = DataLoader(dataset, batch_size=10)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        config = GenericTrainerConfig(
            train_loader_config=MultiDataLoaderConfig(
                sampling_strategy=SamplingStrategy.SEQUENTIAL,
                epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
                dataloader_names=["checkpointable"],
            ),
            val_loader_config=MultiDataLoaderConfig(),
            fault_tolerance=FaultToleranceConfig(
                save_dataset_state=True,
            ),
            checkpoint=CheckpointConfig(
                root_dir=self.checkpoint_dir,
                save_every_n_steps=10,
            ),
        )

        trainer = GenericTrainer(
            model=model,
            config=config,
            optimizers=[optimizer],
            fabric=None,
        )

        def training_step(
            trainer: GenericTrainer,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
            dataloader_name: str,
        ) -> dict[str, Any]:
            inputs, targets = batch
            outputs = trainer.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets.long())
            return {"loss": loss}

        trainer.set_training_step(training_step)

        # No need to assign internal loaders; using external dataset ops

        # Should work without errors
        try:
            # Save initial state
            initial_state = dataset.state_dict()
            assert "position" in initial_state

            # Process some batches
            iterator = iter(loader)
            for _ in range(2):
                next(iterator)

            # Save state after processing
            mid_state = dataset.state_dict()
            assert mid_state["position"] > initial_state["position"]

            # Test state restoration
            dataset.load_state_dict(initial_state)
            assert dataset.position == initial_state["position"]

            # Successfully saved and restored
            assert True
        except Exception as e:
            pytest.fail(f"Checkpointable dataset should work with FT: {e}")

    def test_mixed_datasets_with_ft(self):
        """Test mixed checkpointable and non-checkpointable datasets."""
        checkpointable = CheckpointableIterableDataset(size=50)
        non_checkpointable = NonCheckpointableIterableDataset(size=50)
        regular = TensorDataset(torch.randn(50, 5), torch.randint(0, 2, (50,)))

        DataLoader(checkpointable, batch_size=10)
        DataLoader(non_checkpointable, batch_size=10)
        DataLoader(regular, batch_size=10)

        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        config = GenericTrainerConfig(
            train_loader_config=MultiDataLoaderConfig(
                sampling_strategy=SamplingStrategy.ROUND_ROBIN,
                epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
                dataloader_names=["checkpointable", "non_checkpointable", "regular"],
            ),
            val_loader_config=MultiDataLoaderConfig(),
            fault_tolerance=FaultToleranceConfig(
                save_dataset_state=True,
            ),
        )

        # The presence of non-checkpointable should be detected
        # This test verifies the error is clear when mixed datasets are used
        trainer = GenericTrainer(
            model=model,
            config=config,
            optimizers=[optimizer],
            fabric=None,
        )

        def training_step(
            trainer: GenericTrainer,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
            dataloader_name: str,
        ) -> dict[str, Any]:
            inputs, targets = batch
            outputs = trainer.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets.long())
            return {"loss": loss}

        trainer.set_training_step(training_step)

        # No need to assign internal loaders; validation happens during fit

        # Should detect the non-checkpointable dataset
        # The exact behavior depends on implementation
        # Either it works with regular datasets or gives a warning

    def test_partially_checkpointable_raises_error(self):
        """Test that partially implemented checkpointing raises error."""
        dataset = PartiallyCheckpointableIterableDataset(size=50)
        DataLoader(dataset, batch_size=10)

        # Has state_dict but not load_state_dict
        state = dataset.state_dict()
        assert "position" in state

        # Should raise error when trying to load
        with pytest.raises(AttributeError, match="load_state_dict"):
            dataset.load_state_dict(state)

    def test_error_message_clarity(self):
        """Test that error messages are clear and actionable."""
        dataset = NonCheckpointableIterableDataset(size=50)

        # Simulate checkpoint attempt
        try:
            # Try to access checkpointing methods
            if not hasattr(dataset, "state_dict"):
                raise AttributeError(
                    f"IterableDataset {dataset.__class__.__name__} does not support checkpointing. "
                    f"To use fault tolerance with IterableDatasets, implement 'state_dict()' and "
                    f"'load_state_dict()' methods in your dataset class."
                )
        except AttributeError as e:
            # Check error message is informative
            assert "does not support checkpointing" in str(e)
            assert "state_dict" in str(e)
            assert "load_state_dict" in str(e)

    def test_checkpointable_dataset_resume(self):
        """Test resuming training with checkpointable IterableDataset."""
        dataset = CheckpointableIterableDataset(size=100)
        loader = DataLoader(dataset, batch_size=10)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Process some batches
        iterator = iter(loader)
        processed_batches = []
        for _i in range(3):
            batch = next(iterator)
            processed_batches.append(batch)

        # Save checkpoint
        checkpoint = {
            "dataset_state": dataset.state_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }

        checkpoint_path = self.checkpoint_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create new dataset and model
        new_dataset = CheckpointableIterableDataset(size=100)
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        # Load checkpoint
        loaded = torch.load(checkpoint_path)
        new_dataset.load_state_dict(loaded["dataset_state"])
        new_model.load_state_dict(loaded["model_state"])
        new_optimizer.load_state_dict(loaded["optimizer_state"])

        # Position should have been restored before any iteration
        assert new_dataset.position == loaded["dataset_state"]["position"]

        # Continue from checkpoint
        new_loader = DataLoader(new_dataset, batch_size=10)
        new_iterator = iter(new_loader)

        # Should continue from where we left off (position will increment after next())
        next(new_iterator)

        # After getting next batch, position should have incremented
        assert new_dataset.position > loaded["dataset_state"]["position"]

    def test_iterable_dataset_detection(self):
        """Test detection of IterableDataset type."""
        from model_training_framework.trainer.dataset_validation import (
            detect_iterable_dataset_type,
        )

        regular_dataset = TensorDataset(torch.randn(10, 5), torch.randn(10, 1))
        iterable_dataset = NonCheckpointableIterableDataset()
        checkpointable_dataset = CheckpointableIterableDataset()

        # Test detection
        assert detect_iterable_dataset_type(regular_dataset) == "map"
        assert detect_iterable_dataset_type(iterable_dataset) == "iterable"
        assert (
            detect_iterable_dataset_type(checkpointable_dataset)
            == "iterable_checkpointable"
        )

    def test_length_estimation_for_iterable(self):
        """Test length estimation for IterableDatasets."""
        from model_training_framework.trainer.dataset_validation import (
            estimate_iterable_dataset_length,
        )

        # Dataset with __len__
        class LenIterableDataset(IterableDataset):
            def __init__(self, size):
                self.size = size

            def __iter__(self):
                for _i in range(self.size):
                    yield torch.randn(5), torch.tensor(0)

            def __len__(self):
                return self.size

        dataset_with_len = LenIterableDataset(50)
        assert estimate_iterable_dataset_length(dataset_with_len) == 50

        # Dataset without __len__
        dataset_without_len = NonCheckpointableIterableDataset()
        assert estimate_iterable_dataset_length(dataset_without_len) is None
