"""
Simple Tests for IterableDataset FT Validation

This module tests the validation logic for IterableDataset checkpointing support.
"""

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from model_training_framework.trainer.dataset_validation import (
    detect_iterable_dataset_type,
    validate_iterable_dataset_checkpointing,
)


class NonCheckpointableIterableDataset(IterableDataset):
    """IterableDataset without checkpointing support."""

    def __init__(self, total_items=100):
        self.total_items = total_items
        self.position = 0

    def __iter__(self):
        while self.position < self.total_items:
            yield torch.randn(5)
            self.position += 1


class CheckpointableIterableDataset(IterableDataset):
    """IterableDataset with checkpointing support."""

    def __init__(self, total_items=100):
        self.total_items = total_items
        self.position = 0

    def __iter__(self):
        while self.position < self.total_items:
            yield torch.randn(5)
            self.position += 1

    def state_dict(self):
        """Save state for checkpointing."""
        return {"position": self.position}

    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.position = state_dict["position"]


class TestIterableDatasetValidation:
    """Test IterableDataset validation for FT."""

    def test_non_checkpointable_with_ft_raises_error(self):
        """Test that non-checkpointable IterableDataset raises error with FT."""
        dataset = NonCheckpointableIterableDataset()
        loader = DataLoader(dataset, batch_size=10)

        # Should raise error when require_checkpointing=True
        with pytest.raises(ValueError, match="does not support checkpointing"):
            validate_iterable_dataset_checkpointing(
                [loader], require_checkpointing=True
            )

    def test_checkpointable_with_ft_succeeds(self):
        """Test that checkpointable IterableDataset works with FT."""
        dataset = CheckpointableIterableDataset()
        loader = DataLoader(dataset, batch_size=10)

        # Should not raise error
        validate_iterable_dataset_checkpointing([loader], require_checkpointing=True)

    def test_regular_dataset_works(self):
        """Test that regular Dataset works with FT."""
        dataset = TensorDataset(torch.randn(100, 5), torch.randn(100, 1))
        loader = DataLoader(dataset, batch_size=10)

        # Regular datasets should always work
        validate_iterable_dataset_checkpointing([loader], require_checkpointing=True)

    def test_mixed_datasets_with_ft(self):
        """Test mixed checkpointable and non-checkpointable datasets."""
        checkpointable = CheckpointableIterableDataset()
        non_checkpointable = NonCheckpointableIterableDataset()
        regular = TensorDataset(torch.randn(100, 5), torch.randn(100, 1))

        loaders = [
            DataLoader(checkpointable, batch_size=10),
            DataLoader(non_checkpointable, batch_size=10),
            DataLoader(regular, batch_size=10),
        ]

        # Should raise error due to non-checkpointable IterableDataset
        with pytest.raises(ValueError, match="does not support checkpointing"):
            validate_iterable_dataset_checkpointing(loaders, require_checkpointing=True)

    def test_detect_iterable_dataset_type(self):
        """Test detection of IterableDataset type."""
        iterable_dataset = NonCheckpointableIterableDataset()
        regular_dataset = TensorDataset(torch.randn(10, 5), torch.randn(10, 1))

        assert detect_iterable_dataset_type(iterable_dataset) == "iterable"
        assert detect_iterable_dataset_type(regular_dataset) == "map"

    def test_checkpointable_dataset_state_preservation(self):
        """Test that checkpointable dataset state is preserved."""
        dataset = CheckpointableIterableDataset(total_items=100)

        # Consume some items
        iterator = iter(dataset)
        for _ in range(10):
            next(iterator)

        # Save state
        state = dataset.state_dict()
        original_position = dataset.position

        # Create new dataset and restore state
        new_dataset = CheckpointableIterableDataset(total_items=100)
        new_dataset.load_state_dict(state)

        # Position should be restored
        assert new_dataset.position == original_position

    def test_validation_without_ft_requirement(self):
        """Test that validation passes when FT is not required."""
        non_checkpointable = NonCheckpointableIterableDataset()
        loader = DataLoader(non_checkpointable, batch_size=10)

        # Should not raise error when require_checkpointing=False
        validate_iterable_dataset_checkpointing([loader], require_checkpointing=False)
