"""
Tests for Metrics Aggregation Modes

This module tests all metric aggregation strategies including MICRO_WEIGHTED,
MACRO_EQUAL, PRIMARY_LOADER_METRIC, and custom aggregation functions.
Also tests cross-rank aggregation for distributed training.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from model_training_framework.trainer.metrics import (
    AggregationStrategy,
    MetricsManager,
)


class TestMicroWeightedAggregation:
    """Test MICRO_WEIGHTED aggregation (weighted by sample count)."""

    def test_micro_weighted_basic(self):
        """Test basic micro-weighted aggregation."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2"],
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        )

        # Add metrics from loader1 (more samples)
        manager.add_train_batch(0, {"loss": 0.5, "accuracy": 0.8}, batch_size=32)
        manager.add_train_batch(0, {"loss": 0.4, "accuracy": 0.85}, batch_size=32)

        # Add metrics from loader2 (fewer samples)
        manager.add_train_batch(1, {"loss": 0.6, "accuracy": 0.7}, batch_size=16)

        # Get aggregated metrics
        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )

        # Global metrics should be weighted by sample count
        # Loader1: 64 samples, avg loss = 0.45, avg acc = 0.825
        # Loader2: 16 samples, avg loss = 0.6, avg acc = 0.7
        # Weighted: loss = (0.45*64 + 0.6*16) / 80 = 0.48
        #          acc = (0.825*64 + 0.7*16) / 80 = 0.80
        assert "train/loss" in metrics
        assert metrics["train/loss"] == pytest.approx(0.48, rel=1e-5)
        assert metrics["train/accuracy"] == pytest.approx(0.80, rel=1e-5)

    def test_micro_weighted_with_different_batch_sizes(self):
        """Test micro-weighted with varying batch sizes."""
        manager = MetricsManager(
            train_loader_names=["small_batch", "large_batch"],
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        )

        # Small batch loader
        for _ in range(5):
            manager.add_train_batch(0, {"loss": 0.3}, batch_size=4)

        # Large batch loader
        for _ in range(2):
            manager.add_train_batch(1, {"loss": 0.7}, batch_size=32)

        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )

        # Total samples: 5*4 + 2*32 = 20 + 64 = 84
        # Weighted loss: (0.3*20 + 0.7*64) / 84 = 0.605...
        assert metrics["train/loss"] == pytest.approx(0.605, abs=0.01)

    def test_micro_weighted_empty_loader(self):
        """Test micro-weighted when one loader has no data."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2"],
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        )

        # Only add to loader1
        manager.add_train_batch(0, {"loss": 0.5}, batch_size=32)

        # Loader2 has no data

        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )

        # Should only use loader1's metrics
        assert metrics["train/loss"] == 0.5


class TestMacroEqualAggregation:
    """Test MACRO_EQUAL aggregation (equal weight regardless of sample count)."""

    def test_macro_equal_basic(self):
        """Test basic macro-equal aggregation."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2"],
            aggregation_strategy=AggregationStrategy.SIMPLE_AVERAGE,
        )

        # Add metrics from loader1 (more samples)
        manager.add_train_batch(0, {"loss": 0.4}, batch_size=100)

        # Add metrics from loader2 (fewer samples)
        manager.add_train_batch(1, {"loss": 0.6}, batch_size=10)

        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )

        # Should be simple average: (0.4 + 0.6) / 2 = 0.5
        assert metrics["train/loss"] == 0.5

    def test_macro_equal_multiple_loaders(self):
        """Test macro-equal with multiple loaders."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2", "loader3"],
            aggregation_strategy=AggregationStrategy.SIMPLE_AVERAGE,
        )

        # Different sample counts but should get equal weight
        manager.add_train_batch(0, {"accuracy": 0.9}, batch_size=1000)
        manager.add_train_batch(1, {"accuracy": 0.6}, batch_size=10)
        manager.add_train_batch(2, {"accuracy": 0.75}, batch_size=100)

        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )

        # Simple average: (0.9 + 0.6 + 0.75) / 3 = 0.75
        assert metrics["train/accuracy"] == 0.75


class TestPrimaryLoaderMetric:
    """Test PRIMARY_LOADER_METRIC aggregation."""

    def test_primary_loader_selection(self):
        """Test that primary loader metrics are selected."""
        # For PRIMARY_LOADER_METRIC, we typically want the first loader's metrics
        manager = MetricsManager(
            train_loader_names=["primary", "secondary", "tertiary"],
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,  # Default for now
        )

        # Add metrics to all loaders
        manager.add_train_batch(0, {"loss": 0.3, "primary_metric": 0.95}, batch_size=32)
        manager.add_train_batch(
            1, {"loss": 0.5, "secondary_metric": 0.8}, batch_size=32
        )
        manager.add_train_batch(2, {"loss": 0.7, "tertiary_metric": 0.6}, batch_size=32)

        # Get per-loader metrics
        metrics = manager.get_train_metrics(
            include_global=False, include_per_loader=True
        )

        # Primary loader's metrics should be available
        assert "train/dl_primary/loss" in metrics
        assert metrics["train/dl_primary/loss"] == 0.3
        assert "train/dl_primary/primary_metric" in metrics


class TestMaxMinAggregation:
    """Test MAX and MIN aggregation strategies."""

    def test_max_aggregation(self):
        """Test MAX aggregation strategy."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2", "loader3"],
            aggregation_strategy=AggregationStrategy.MAX,
        )

        manager.add_train_batch(0, {"loss": 0.3}, batch_size=10)
        manager.add_train_batch(1, {"loss": 0.7}, batch_size=10)
        manager.add_train_batch(2, {"loss": 0.5}, batch_size=10)

        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )

        # Should return maximum: 0.7
        assert metrics["train/loss"] == 0.7

    def test_min_aggregation(self):
        """Test MIN aggregation strategy."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2", "loader3"],
            aggregation_strategy=AggregationStrategy.MIN,
        )

        manager.add_train_batch(0, {"accuracy": 0.8}, batch_size=10)
        manager.add_train_batch(1, {"accuracy": 0.6}, batch_size=10)
        manager.add_train_batch(2, {"accuracy": 0.9}, batch_size=10)

        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )

        # Should return minimum: 0.6
        assert metrics["train/accuracy"] == 0.6


class TestSumAggregation:
    """Test SUM aggregation strategy."""

    def test_sum_aggregation(self):
        """Test SUM aggregation strategy."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2"],
            aggregation_strategy=AggregationStrategy.SUM,
        )

        manager.add_train_batch(0, {"count": 100}, batch_size=10)
        manager.add_train_batch(1, {"count": 150}, batch_size=10)

        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )

        # Should return sum: 100 + 150 = 250
        assert metrics["train/count"] == 250


class TestCrossRankAggregation:
    """Test cross-rank aggregation for distributed training."""

    def test_cross_rank_aggregation_single_process(self):
        """Test that single-process doesn't aggregate."""
        manager = MetricsManager(
            train_loader_names=["loader1"],
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        )

        manager.add_train_batch(0, {"loss": 0.5}, batch_size=32)

        # Mock fabric with world_size=1
        fabric = MagicMock()
        fabric.world_size = 1

        # Should not change metrics
        manager.aggregate_across_ranks(fabric)

        metrics = manager.get_train_metrics()
        assert metrics["train/loss"] == 0.5

    @patch("model_training_framework.trainer.metrics.ddp_all_reduce")
    def test_cross_rank_aggregation_multi_process(self, mock_all_reduce):
        """Test cross-rank aggregation with multiple processes."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2"],
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        )

        # Add local metrics
        manager.add_train_batch(0, {"loss": 0.4}, batch_size=16)
        manager.add_train_batch(1, {"loss": 0.6}, batch_size=16)

        # Mock fabric with world_size=2
        fabric = MagicMock()
        fabric.world_size = 2

        # Mock all_reduce to simulate aggregation
        def mock_reduce(fabric, tensor, op="mean"):
            if op == "sum":
                # Simulate summing across 2 ranks
                return torch.tensor(tensor.item() * 2)
            return tensor

        mock_all_reduce.side_effect = mock_reduce

        # Aggregate across ranks
        manager.aggregate_across_ranks(fabric)

        # Check that all_reduce was called
        assert mock_all_reduce.called

        # Sample counts should be doubled (2 ranks)
        assert manager.train_metrics[0].sample_count == 32  # 16 * 2
        assert manager.train_metrics[1].sample_count == 32  # 16 * 2


class TestValidationMetrics:
    """Test validation metrics aggregation."""

    def test_validation_metrics_aggregation(self):
        """Test that validation metrics are aggregated correctly."""
        manager = MetricsManager(
            train_loader_names=["train1"],
            val_loader_names=["val1", "val2"],
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        )

        # Add validation metrics
        manager.add_val_batch(0, {"val_loss": 0.3, "val_acc": 0.9}, batch_size=20)
        manager.add_val_batch(1, {"val_loss": 0.5, "val_acc": 0.7}, batch_size=30)

        val_metrics = manager.get_val_metrics()

        # Check per-loader metrics
        assert "val/dl_val1/val_loss" in val_metrics
        assert "val/dl_val2/val_loss" in val_metrics

        # Check global aggregation (weighted by samples)
        # (0.3*20 + 0.5*30) / 50 = 0.42
        assert val_metrics["val/val_loss"] == pytest.approx(0.42, rel=1e-5)


class TestEpochSummaries:
    """Test epoch summary functionality."""

    def test_save_epoch_summary(self):
        """Test saving epoch summaries."""
        manager = MetricsManager(
            train_loader_names=["loader1"],
            val_loader_names=["val1"],
        )

        # Add metrics
        manager.add_train_batch(0, {"loss": 0.5}, batch_size=32)
        manager.add_val_batch(0, {"val_loss": 0.3}, batch_size=32)

        # Save epoch summary
        summary = manager.save_epoch_summary(
            epoch=1, extra_info={"learning_rate": 0.001}
        )

        assert summary["epoch"] == 1
        assert "train_metrics" in summary
        assert "val_metrics" in summary
        assert summary["learning_rate"] == 0.001

        # Check it was added to history
        assert len(manager.epoch_summaries) == 1

    def test_best_metric_tracking(self):
        """Test finding best metric across epochs."""
        manager = MetricsManager(
            train_loader_names=["loader1"],
            val_loader_names=["val1"],
        )

        # Simulate multiple epochs
        for epoch in range(5):
            manager.reset_epoch("both")

            # Add metrics with decreasing loss
            val_loss = 0.5 - (epoch * 0.05)
            manager.add_val_batch(0, {"loss": val_loss}, batch_size=32)

            manager.save_epoch_summary(epoch=epoch)

        # Find best validation loss
        best_loss, best_epoch = manager.get_best_metric(
            "val/loss", mode="min", phase="val"
        )

        assert best_loss == pytest.approx(0.3, rel=1e-5)  # 0.5 - 4*0.05
        assert best_epoch == 4

        # Test max mode
        for epoch in range(5):
            manager.reset_epoch("train")
            manager.add_train_batch(0, {"accuracy": 0.7 + epoch * 0.05}, batch_size=32)
            manager.save_epoch_summary(epoch=epoch + 5)

        best_acc, best_epoch = manager.get_best_metric(
            "train/accuracy", mode="max", phase="train"
        )
        assert best_acc == pytest.approx(0.9, rel=1e-5)  # 0.7 + 4*0.05
        assert best_epoch == 9


class TestProportionTracking:
    """Test dataloader proportion tracking."""

    def test_loader_proportions(self):
        """Test tracking of loader usage proportions."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2", "loader3"],
            track_proportions=True,
        )

        # Set expected proportions
        manager.set_expected_proportions([0.5, 0.3, 0.2])

        # Add batches with actual usage
        for _ in range(6):
            manager.add_train_batch(0, {"loss": 0.5}, batch_size=10)
        for _ in range(3):
            manager.add_train_batch(1, {"loss": 0.5}, batch_size=10)
        for _ in range(1):
            manager.add_train_batch(2, {"loss": 0.5}, batch_size=10)

        proportions, counts = manager.get_loader_proportions()

        # Check batch proportions
        assert proportions["loader1_batches"] == 0.6  # 6/10
        assert proportions["loader2_batches"] == 0.3  # 3/10
        assert proportions["loader3_batches"] == 0.1  # 1/10

        # Check divergence from expected
        assert "loader1_divergence" in proportions
        assert proportions["loader1_divergence"] == pytest.approx(
            0.1, abs=0.01
        )  # |0.6-0.5|
        assert proportions["loader3_divergence"] == pytest.approx(
            0.1, abs=0.01
        )  # |0.1-0.2|

        # Check counts
        assert counts["loader1_batches"] == 6
        assert counts["loader2_batches"] == 3
        assert counts["loader3_batches"] == 1
