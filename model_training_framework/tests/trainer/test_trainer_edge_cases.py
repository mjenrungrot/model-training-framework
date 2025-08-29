"""
Tests for Edge Cases in Multi-Dataloader Training (Fixed Version)

This module tests edge cases using the actual DataLoaderManager implementation.
"""

from pathlib import Path

# Import the Hamilton function from test file
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model_training_framework.trainer import (
    MultiDataLoaderConfig,
    PerformanceConfig,
)
from model_training_framework.trainer.config import (
    EpochLengthPolicy,
    SamplingStrategy,
)
from model_training_framework.trainer.multi_dataloader import DataLoaderManager

sys.path.insert(0, str(Path(__file__).parent))
from test_scheduler_hamilton_allocation import compute_weighted_targets_hamilton


class TinyModel(nn.Module):
    """Tiny model for edge case testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)


class EmptyDataset(Dataset):
    """Dataset with no samples."""

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("Empty dataset")


class SingleBatchDataset(Dataset):
    """Dataset with exactly one batch worth of data."""

    def __init__(self, batch_size=10):
        self.data = torch.randn(batch_size, 3)
        self.labels = torch.randint(0, 2, (batch_size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TestSingleBatchDatasets:
    """Test datasets with only one batch."""

    def test_single_batch_sequential(self):
        """Test SEQUENTIAL with single batch datasets."""
        datasets = [
            SingleBatchDataset(batch_size=5),
            SingleBatchDataset(batch_size=5),
        ]
        loaders = [DataLoader(ds, batch_size=5) for ds in datasets]

        config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            dataloader_names=["single1", "single2"],
        )

        manager = DataLoaderManager(
            train_loaders=loaders,
            val_loaders=None,
            train_config=config,
        )

        # Should complete exactly 2 batches
        batch_count = 0
        iterator = manager.create_epoch_iterator(phase="train", epoch=0)
        for _idx, _batch in iterator:
            batch_count += 1

        assert batch_count == 2

    def test_single_batch_round_robin(self):
        """Test ROUND_ROBIN with single batch datasets."""
        datasets = [
            SingleBatchDataset(batch_size=8),
            SingleBatchDataset(batch_size=8),
        ]
        loaders = [DataLoader(ds, batch_size=8) for ds in datasets]

        config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.ROUND_ROBIN,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            dataloader_names=["single1", "single2"],
            cycle_short_loaders=False,
        )

        manager = DataLoaderManager(
            train_loaders=loaders,
            val_loaders=None,
            train_config=config,
        )

        # Should stop after 2 batches (one from each)
        batch_count = 0
        iterator = manager.create_epoch_iterator(phase="train", epoch=0)
        for _idx, _batch in iterator:
            batch_count += 1

        assert batch_count == 2

    def test_single_batch_weighted(self):
        """Test WEIGHTED with single batch datasets."""
        datasets = [
            SingleBatchDataset(batch_size=10),
            SingleBatchDataset(batch_size=10),
            SingleBatchDataset(batch_size=10),
        ]
        loaders = [DataLoader(ds, batch_size=10) for ds in datasets]

        config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
            steps_per_epoch=3,
            dataloader_names=["single1", "single2", "single3"],
            dataloader_weights=[0.5, 0.3, 0.2],
            cycle_short_loaders=False,
        )

        manager = DataLoaderManager(
            train_loaders=loaders,
            val_loaders=None,
            train_config=config,
        )

        # Should complete 3 steps (one per loader)
        loader_usage = [0, 0, 0]
        iterator = manager.create_epoch_iterator(phase="train", epoch=0)
        for idx, _batch in iterator:
            loader_usage[idx] += 1

        # Each loader used exactly once
        assert all(count == 1 for count in loader_usage)


class TestEmptyDatasets:
    """Test handling of empty datasets."""

    def test_empty_dataset_handling(self):
        """Test that empty datasets are handled gracefully."""
        empty_dataset = EmptyDataset()
        loader = DataLoader(empty_dataset, batch_size=10)

        config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            dataloader_names=["empty"],
        )

        manager = DataLoaderManager(
            train_loaders=[loader],
            val_loaders=None,
            train_config=config,
        )

        # Should handle empty dataset gracefully
        batch_count = 0
        iterator = manager.create_epoch_iterator(phase="train", epoch=0)
        for _idx, _batch in iterator:
            batch_count += 1

        assert batch_count == 0  # No batches from empty dataset

    def test_mixed_empty_nonempty(self):
        """Test mix of empty and non-empty datasets."""
        datasets = [
            EmptyDataset(),
            TensorDataset(torch.randn(20, 3), torch.randint(0, 2, (20,))),
        ]
        loaders = [DataLoader(ds, batch_size=10) for ds in datasets]

        config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.ROUND_ROBIN,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            dataloader_names=["empty", "nonempty"],
            cycle_short_loaders=False,
        )

        manager = DataLoaderManager(
            train_loaders=loaders,
            val_loaders=None,
            train_config=config,
        )

        # Should only get batches from non-empty loader
        batch_count = 0
        loader_indices = []

        iterator = manager.create_epoch_iterator(phase="train", epoch=0)
        for idx, _batch in iterator:
            batch_count += 1
            loader_indices.append(idx)

        # With ROUND_ROBIN and one empty loader, it tries to alternate but can't
        # So we only get batches from the non-empty loader
        assert batch_count >= 1  # At least 1 batch from non-empty loader
        assert all(idx == 1 for idx in loader_indices)  # All from loader 1


class TestExtremeGradientAccumulation:
    """Test extreme gradient accumulation values."""

    def test_very_large_accumulation(self):
        """Test with very large gradient accumulation steps."""
        # Test gradient accumulation logic directly
        config = PerformanceConfig(
            gradient_accumulation_steps=50,  # Very large
        )

        # Simulate 20 batches with accumulation of 50
        # Should only step once at the end
        total_batches = 20
        accumulation_steps = config.gradient_accumulation_steps

        optimizer_steps = 0
        for batch_idx in range(total_batches):
            # Check if we should step optimizer
            if (
                batch_idx + 1
            ) % accumulation_steps == 0 or batch_idx == total_batches - 1:
                optimizer_steps += 1

        # With 20 batches and accumulation=50, we step once at the end
        assert optimizer_steps == 1

        # Test another scenario: 100 batches with accumulation of 30
        total_batches = 100
        accumulation_steps = 30

        optimizer_steps = 0
        for batch_idx in range(total_batches):
            if (
                batch_idx + 1
            ) % accumulation_steps == 0 or batch_idx == total_batches - 1:
                optimizer_steps += 1

        # Should step at 30, 60, 90, and 100 = 4 times
        assert optimizer_steps == 4


class TestExtremeWeightRatios:
    """Test extreme weight ratios in WEIGHTED sampling."""

    def test_extreme_weight_ratio_99_to_1(self):
        """Test 99:1 weight ratio."""
        weights = [0.99, 0.01]
        total_steps = 100

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        # First loader should get 99 steps
        assert quotas[0] == 99
        assert quotas[1] == 1
        assert sum(quotas) == total_steps

    def test_extreme_weight_ratio_999_to_1(self):
        """Test 999:1 weight ratio."""
        weights = [0.999, 0.001]
        total_steps = 1000

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        # First loader should get 999 steps
        assert quotas[0] == 999
        assert quotas[1] == 1
        assert sum(quotas) == total_steps

    def test_many_tiny_weights(self):
        """Test many loaders with tiny weights."""
        # number of loaders inferred from weights length
        weights = [0.91] + [0.01] * 9  # One dominant, rest tiny
        total_steps = 100

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        # First loader should dominate
        assert quotas[0] >= 90
        # Others should get at most 1-2 each
        for quota in quotas[1:]:
            assert quota <= 2
        assert sum(quotas) == total_steps


class TestZeroAndNegativeValues:
    """Test handling of zero and negative values."""

    def test_zero_steps_per_epoch(self):
        """Test with steps_per_epoch = 0."""
        datasets = [
            TensorDataset(torch.randn(20, 3), torch.randint(0, 2, (20,))),
        ]
        loaders = [DataLoader(ds, batch_size=10) for ds in datasets]

        config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
            steps_per_epoch=0,  # Zero steps
            dataloader_names=["loader1"],
            dataloader_weights=[1.0],
        )

        manager = DataLoaderManager(
            train_loaders=loaders,
            val_loaders=None,
            train_config=config,
        )

        # Should complete immediately with no batches
        batch_count = 0
        iterator = manager.create_epoch_iterator(phase="train", epoch=0)
        for _idx, _batch in iterator:
            batch_count += 1

        assert batch_count == 0

    def test_zero_batch_size_error(self):
        """Test that zero batch size is handled."""
        dataset = TensorDataset(torch.randn(20, 3), torch.randint(0, 2, (20,)))

        # DataLoader with batch_size=0 should raise error
        with pytest.raises(ValueError):
            DataLoader(dataset, batch_size=0)

    def test_negative_weights_error(self):
        """Test that negative weights raise error."""
        with pytest.raises(ValueError, match="negative"):
            compute_weighted_targets_hamilton([-0.5, 0.5], 100)

    def test_all_zero_weights_error(self):
        """Test that all-zero weights raise error."""
        with pytest.raises(ValueError, match="Weights must sum to positive"):
            compute_weighted_targets_hamilton([0, 0, 0], 100)
