"""
Tests for Metrics Management

This module tests the metrics functionality including:
- Per-loader metric tracking
- Global aggregation strategies
- Proportion tracking for weighted sampling
- Epoch summaries and best metric tracking
"""

import pytest
import torch

from model_training_framework.trainer.metrics import (
    AggregationStrategy,
    LoaderMetrics,
    MetricsManager,
)


class TestLoaderMetrics:
    """Test LoaderMetrics class."""

    def test_loader_metrics_creation(self):
        """Test LoaderMetrics can be created."""
        metrics = LoaderMetrics(name="train_loader", loader_idx=0)
        assert metrics.name == "train_loader"
        assert metrics.loader_idx == 0
        assert metrics.batch_count == 0
        assert metrics.sample_count == 0

    def test_add_batch(self):
        """Test adding batch metrics."""
        metrics = LoaderMetrics(name="loader1", loader_idx=0)

        batch_metrics = {
            "loss": 0.5,
            "accuracy": torch.tensor(0.95),
            "text": "ignored",  # Non-numeric should be ignored
        }

        metrics.add_batch(batch_metrics, batch_size=32)

        assert metrics.batch_count == 1
        assert metrics.sample_count == 32
        assert "loss" in metrics.metrics
        assert "accuracy" in metrics.metrics
        assert "text" not in metrics.metrics
        assert metrics.metrics["loss"] == [0.5]
        assert metrics.metrics["accuracy"][0] == pytest.approx(0.95, rel=1e-5)

    def test_get_average(self):
        """Test getting average of metrics."""
        metrics = LoaderMetrics(name="loader1", loader_idx=0)

        # Add multiple batches
        metrics.add_batch({"loss": 0.5}, batch_size=32)
        metrics.add_batch({"loss": 0.3}, batch_size=32)
        metrics.add_batch({"loss": 0.4}, batch_size=32)

        avg = metrics.get_average("loss")
        assert avg == pytest.approx(0.4, rel=1e-5)

        # Non-existent metric
        assert metrics.get_average("nonexistent") is None

    def test_get_last(self):
        """Test getting last value of metrics."""
        metrics = LoaderMetrics(name="loader1", loader_idx=0)

        metrics.add_batch({"loss": 0.5}, batch_size=32)
        metrics.add_batch({"loss": 0.3}, batch_size=32)
        metrics.add_batch({"loss": 0.2}, batch_size=32)

        last = metrics.get_last("loss")
        assert last == 0.2

        # Non-existent metric
        assert metrics.get_last("nonexistent") is None

    def test_reset(self):
        """Test resetting metrics."""
        metrics = LoaderMetrics(name="loader1", loader_idx=0)

        metrics.add_batch({"loss": 0.5}, batch_size=32)
        assert metrics.batch_count == 1
        assert metrics.sample_count == 32

        metrics.reset()

        assert metrics.batch_count == 0
        assert metrics.sample_count == 0
        assert len(metrics.metrics) == 0


class TestMetricsManager:
    """Test MetricsManager class."""

    def test_metrics_manager_creation(self):
        """Test MetricsManager can be created."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2"],
            val_loader_names=["val_loader1"],
        )

        assert len(manager.train_metrics) == 2
        assert len(manager.val_metrics) == 1
        assert manager.train_metrics[0].name == "loader1"
        assert manager.train_metrics[1].name == "loader2"

    def test_add_train_batch(self):
        """Test adding training batch metrics."""
        manager = MetricsManager(train_loader_names=["loader1", "loader2"])

        metrics = {"loss": 0.5, "accuracy": 0.95}
        manager.add_train_batch(0, metrics, batch_size=32)

        assert manager.train_metrics[0].batch_count == 1
        assert manager.train_metrics[0].sample_count == 32

    def test_add_val_batch(self):
        """Test adding validation batch metrics."""
        manager = MetricsManager(
            train_loader_names=["loader1"],
            val_loader_names=["val_loader1"],
        )

        metrics = {"loss": 0.3, "accuracy": 0.97}
        manager.add_val_batch(0, metrics, batch_size=16)

        assert manager.val_metrics[0].batch_count == 1
        assert manager.val_metrics[0].sample_count == 16

    def test_get_train_metrics(self):
        """Test getting aggregated training metrics."""
        manager = MetricsManager(train_loader_names=["loader1", "loader2"])

        # Add metrics for both loaders
        manager.add_train_batch(0, {"loss": 0.5}, batch_size=32)
        manager.add_train_batch(0, {"loss": 0.4}, batch_size=32)
        manager.add_train_batch(1, {"loss": 0.6}, batch_size=16)
        manager.add_train_batch(1, {"loss": 0.7}, batch_size=16)

        # Get metrics with per-loader and global
        metrics = manager.get_train_metrics(
            include_per_loader=True,
            include_global=True,
        )

        # Check per-loader metrics
        assert "train/dl_loader1/loss" in metrics
        assert "train/dl_loader2/loss" in metrics
        assert metrics["train/dl_loader1/loss"] == pytest.approx(0.45, rel=1e-5)
        assert metrics["train/dl_loader2/loss"] == pytest.approx(0.65, rel=1e-5)

        # Check global metric (weighted average by default)
        assert "train/loss" in metrics

    def test_get_val_metrics(self):
        """Test getting aggregated validation metrics."""
        manager = MetricsManager(
            train_loader_names=["loader1"],
            val_loader_names=["val_loader1", "val_loader2"],
        )

        # Add metrics
        manager.add_val_batch(0, {"loss": 0.2}, batch_size=32)
        manager.add_val_batch(1, {"loss": 0.3}, batch_size=32)

        metrics = manager.get_val_metrics()

        assert "val/dl_val_loader1/loss" in metrics
        assert "val/dl_val_loader2/loss" in metrics
        assert "val/loss" in metrics

    def test_aggregation_strategies(self):
        """Test different aggregation strategies."""
        # Test weighted average
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2"],
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        )

        manager.add_train_batch(0, {"loss": 0.4}, batch_size=30)
        manager.add_train_batch(1, {"loss": 0.6}, batch_size=10)

        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )
        assert metrics["train/loss"] == pytest.approx(0.45, rel=1e-5)

        # Test simple average
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2"],
            aggregation_strategy=AggregationStrategy.SIMPLE_AVERAGE,
        )

        manager.add_train_batch(0, {"loss": 0.4}, batch_size=30)
        manager.add_train_batch(1, {"loss": 0.6}, batch_size=10)

        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )
        assert metrics["train/loss"] == pytest.approx(0.5, rel=1e-5)

        # Test sum
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2"],
            aggregation_strategy=AggregationStrategy.SUM,
        )

        manager.add_train_batch(0, {"loss": 0.4}, batch_size=30)
        manager.add_train_batch(1, {"loss": 0.6}, batch_size=10)

        metrics = manager.get_train_metrics(
            include_global=True, include_per_loader=False
        )
        assert metrics["train/loss"] == pytest.approx(1.0, rel=1e-5)

    def test_loader_proportions(self):
        """Test getting loader proportions."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2", "loader3"],
            track_proportions=True,
        )

        # Add different numbers of batches
        for _ in range(6):
            manager.add_train_batch(0, {"loss": 0.5}, batch_size=32)
        for _ in range(3):
            manager.add_train_batch(1, {"loss": 0.5}, batch_size=32)
        for _ in range(1):
            manager.add_train_batch(2, {"loss": 0.5}, batch_size=32)

        proportions, counts = manager.get_loader_proportions()

        # Check batch proportions
        assert proportions["loader1_batches"] == pytest.approx(0.6, rel=1e-5)
        assert proportions["loader2_batches"] == pytest.approx(0.3, rel=1e-5)
        assert proportions["loader3_batches"] == pytest.approx(0.1, rel=1e-5)

        # Check counts
        assert counts["loader1_batches"] == 6
        assert counts["loader2_batches"] == 3
        assert counts["loader3_batches"] == 1

    def test_expected_proportions_divergence(self):
        """Test divergence from expected proportions."""
        manager = MetricsManager(
            train_loader_names=["loader1", "loader2"],
            track_proportions=True,
        )

        # Set expected proportions (60/40 split)
        manager.set_expected_proportions([0.6, 0.4])

        # Add batches with 50/50 split
        for _ in range(5):
            manager.add_train_batch(0, {"loss": 0.5}, batch_size=32)
        for _ in range(5):
            manager.add_train_batch(1, {"loss": 0.5}, batch_size=32)

        proportions, _ = manager.get_loader_proportions()

        # Check divergence
        assert "loader1_divergence" in proportions
        assert "loader2_divergence" in proportions
        # Actual is 0.5, expected is 0.6, divergence is 0.1
        assert proportions["loader1_divergence"] == pytest.approx(0.1, rel=1e-5)
        # Actual is 0.5, expected is 0.4, divergence is 0.1
        assert proportions["loader2_divergence"] == pytest.approx(0.1, rel=1e-5)

    def test_reset_epoch(self):
        """Test resetting metrics for new epoch."""
        manager = MetricsManager(
            train_loader_names=["loader1"],
            val_loader_names=["val_loader1"],
        )

        # Add metrics
        manager.add_train_batch(0, {"loss": 0.5}, batch_size=32)
        manager.add_val_batch(0, {"loss": 0.3}, batch_size=16)

        # Reset training only
        manager.reset_epoch("train")
        assert manager.train_metrics[0].batch_count == 0
        assert manager.val_metrics[0].batch_count == 1

        # Reset validation only
        manager.reset_epoch("val")
        assert manager.val_metrics[0].batch_count == 0

        # Add again and reset both
        manager.add_train_batch(0, {"loss": 0.5}, batch_size=32)
        manager.add_val_batch(0, {"loss": 0.3}, batch_size=16)

        manager.reset_epoch("both")
        assert manager.train_metrics[0].batch_count == 0
        assert manager.val_metrics[0].batch_count == 0

    def test_epoch_summary(self):
        """Test saving epoch summary."""
        manager = MetricsManager(
            train_loader_names=["loader1"],
            val_loader_names=["val_loader1"],
        )

        # Add metrics
        manager.add_train_batch(0, {"loss": 0.5}, batch_size=32)
        manager.add_val_batch(0, {"loss": 0.3}, batch_size=16)

        # Save summary
        summary = manager.save_epoch_summary(
            epoch=5,
            extra_info={"learning_rate": 0.001},
        )

        assert summary["epoch"] == 5
        assert "train_metrics" in summary
        assert "val_metrics" in summary
        assert summary["learning_rate"] == 0.001

        # Check it was saved
        assert len(manager.epoch_summaries) == 1
        assert manager.epoch_summaries[0]["epoch"] == 5

    def test_get_best_metric(self):
        """Test getting best metric across epochs."""
        manager = MetricsManager(
            train_loader_names=["loader1"],
            val_loader_names=["val_loader1"],
        )

        # Simulate multiple epochs
        for epoch in range(5):
            manager.reset_epoch("both")

            # Add decreasing loss
            manager.add_val_batch(0, {"loss": 1.0 - epoch * 0.1}, batch_size=32)

            manager.save_epoch_summary(epoch)

        # Get best (minimum) validation loss
        best_value, best_epoch = manager.get_best_metric(
            "val/loss", mode="min", phase="val"
        )

        assert best_value == pytest.approx(0.6, rel=1e-5)
        assert best_epoch == 4

        # Get best (maximum) - should be first epoch
        best_value, best_epoch = manager.get_best_metric(
            "val/loss", mode="max", phase="val"
        )

        assert best_value == pytest.approx(1.0, rel=1e-5)
        assert best_epoch == 0

        # Non-existent metric
        best_value, best_epoch = manager.get_best_metric(
            "nonexistent", mode="min", phase="val"
        )

        assert best_value is None
        assert best_epoch is None
