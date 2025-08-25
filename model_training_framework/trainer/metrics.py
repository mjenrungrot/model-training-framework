"""
Metrics Management for Training Framework

This module provides advanced metrics tracking and aggregation:
- Per-loader metric accumulation with running averages
- Realized proportions tracking for weighted sampling
- Batch and sample counting per loader
- Global aggregation with configurable strategies
- Consistent metric naming conventions
- Cross-rank aggregation for distributed training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any

import torch

from .utils import ddp_all_reduce

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategies for aggregating metrics across dataloaders."""

    WEIGHTED_AVERAGE = "weighted_average"  # Weight by sample count
    SIMPLE_AVERAGE = "simple_average"  # Equal weight for all loaders
    SUM = "sum"  # Sum across loaders
    MAX = "max"  # Maximum value across loaders
    MIN = "min"  # Minimum value across loaders


@dataclass
class LoaderMetrics:
    """Metrics tracked for a single dataloader."""

    name: str
    loader_idx: int
    batch_count: int = 0
    sample_count: int = 0
    metrics: dict[str, list[float]] = field(default_factory=dict)

    def add_batch(self, metrics: dict[str, Any], batch_size: int) -> None:
        """Add metrics from a single batch."""
        self.batch_count += 1
        self.sample_count += batch_size

        for key, value in metrics.items():
            # Normalize to a numeric Python scalar
            if isinstance(value, torch.Tensor):
                numeric = value.item()
            elif isinstance(value, int | float):
                numeric = value
            else:
                continue  # Skip non-numeric values

            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(float(numeric))

    def get_average(self, metric_key: str) -> float | None:
        """Get average value for a metric."""
        if metric_key not in self.metrics or not self.metrics[metric_key]:
            return None
        return sum(self.metrics[metric_key]) / len(self.metrics[metric_key])

    def get_last(self, metric_key: str) -> float | None:
        """Get last value for a metric."""
        if metric_key not in self.metrics or not self.metrics[metric_key]:
            return None
        return self.metrics[metric_key][-1]

    def reset(self) -> None:
        """Reset metrics for new epoch."""
        self.batch_count = 0
        self.sample_count = 0
        self.metrics.clear()


class MetricsManager:
    """
    Manages metric collection and aggregation across multiple dataloaders.

    Handles per-loader tracking, global aggregation, and proportion calculation
    for weighted sampling strategies.
    """

    def __init__(
        self,
        train_loader_names: list[str] | None = None,
        val_loader_names: list[str] | None = None,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_AVERAGE,
        track_proportions: bool = True,
    ):
        """
        Initialize metrics manager.

        Args:
            train_loader_names: Names of training dataloaders
            val_loader_names: Names of validation dataloaders
            aggregation_strategy: How to aggregate metrics across loaders
            track_proportions: Whether to track loader usage proportions
        """
        self.aggregation_strategy = aggregation_strategy
        self.track_proportions = track_proportions

        # Initialize per-loader metrics
        self.train_metrics: dict[int, LoaderMetrics] = {}
        if train_loader_names:
            for idx, name in enumerate(train_loader_names):
                self.train_metrics[idx] = LoaderMetrics(name, idx)

        self.val_metrics: dict[int, LoaderMetrics] = {}
        if val_loader_names:
            for idx, name in enumerate(val_loader_names):
                self.val_metrics[idx] = LoaderMetrics(name, idx)

        # Track expected proportions for comparison (WEIGHTED strategy)
        self.expected_proportions: dict[str, float] | None = None

        # Epoch-level summaries
        self.epoch_summaries: list[dict[str, Any]] = []

    def set_expected_proportions(self, weights: list[float]) -> None:
        """
        Set expected proportions for weighted sampling.

        Args:
            weights: Dataloader weights for WEIGHTED strategy
        """
        if not weights:
            return

        total = sum(weights)
        self.expected_proportions = {}
        for idx, weight in enumerate(weights):
            if idx in self.train_metrics:
                name = self.train_metrics[idx].name
                self.expected_proportions[name] = weight / total

    def add_train_batch(
        self,
        loader_idx: int,
        metrics: dict[str, Any],
        batch_size: int,
    ) -> None:
        """
        Add metrics from a training batch.

        Args:
            loader_idx: Index of the dataloader
            metrics: Metrics dictionary from training step
            batch_size: Number of samples in batch
        """
        if loader_idx not in self.train_metrics:
            logger.warning(f"Unknown train loader index: {loader_idx}")
            return

        self.train_metrics[loader_idx].add_batch(metrics, batch_size)

    def add_val_batch(
        self,
        loader_idx: int,
        metrics: dict[str, Any],
        batch_size: int,
    ) -> None:
        """
        Add metrics from a validation batch.

        Args:
            loader_idx: Index of the dataloader
            metrics: Metrics dictionary from validation step
            batch_size: Number of samples in batch
        """
        if loader_idx not in self.val_metrics:
            logger.warning(f"Unknown val loader index: {loader_idx}")
            return

        self.val_metrics[loader_idx].add_batch(metrics, batch_size)

    def get_train_metrics(
        self,
        include_global: bool = True,
        include_per_loader: bool = True,
    ) -> dict[str, float]:
        """
        Get aggregated training metrics.

        Args:
            include_global: Include globally aggregated metrics
            include_per_loader: Include per-loader metrics

        Returns:
            Dictionary of metric names to values
        """
        result = {}

        # Per-loader metrics
        if include_per_loader:
            for loader_metrics in self.train_metrics.values():
                prefix = f"train/dl_{loader_metrics.name}"
                for metric_key in loader_metrics.metrics:
                    avg = loader_metrics.get_average(metric_key)
                    if avg is not None:
                        result[f"{prefix}/{metric_key}"] = avg

        # Global aggregation
        if include_global and len(self.train_metrics) > 0:
            global_metrics = self._aggregate_metrics(self.train_metrics, prefix="train")
            result.update(global_metrics)

        return result

    def get_val_metrics(
        self,
        include_global: bool = True,
        include_per_loader: bool = True,
    ) -> dict[str, float]:
        """
        Get aggregated validation metrics.

        Args:
            include_global: Include globally aggregated metrics
            include_per_loader: Include per-loader metrics

        Returns:
            Dictionary of metric names to values
        """
        result = {}

        # Per-loader metrics
        if include_per_loader:
            for loader_metrics in self.val_metrics.values():
                prefix = f"val/dl_{loader_metrics.name}"
                for metric_key in loader_metrics.metrics:
                    avg = loader_metrics.get_average(metric_key)
                    if avg is not None:
                        result[f"{prefix}/{metric_key}"] = avg

        # Global aggregation
        if include_global and len(self.val_metrics) > 0:
            global_metrics = self._aggregate_metrics(self.val_metrics, prefix="val")
            result.update(global_metrics)

        return result

    def get_loader_proportions(self) -> tuple[dict[str, float], dict[str, int]]:
        """
        Get realized proportions and counts for training loaders.

        Returns:
            Tuple of (proportions_dict, counts_dict)
        """
        proportions = {}
        counts = {}

        total_batches = sum(m.batch_count for m in self.train_metrics.values())
        total_samples = sum(m.sample_count for m in self.train_metrics.values())

        for loader_metrics in self.train_metrics.values():
            name = loader_metrics.name

            # Batch-based proportion
            if total_batches > 0:
                proportions[f"{name}_batches"] = (
                    loader_metrics.batch_count / total_batches
                )

            # Sample-based proportion
            if total_samples > 0:
                proportions[f"{name}_samples"] = (
                    loader_metrics.sample_count / total_samples
                )

            # Counts
            counts[f"{name}_batches"] = loader_metrics.batch_count
            counts[f"{name}_samples"] = loader_metrics.sample_count

        # Add divergence from expected if available
        if self.expected_proportions and total_batches > 0:
            for name, expected in self.expected_proportions.items():
                actual = proportions.get(f"{name}_batches", 0)
                proportions[f"{name}_divergence"] = abs(actual - expected)

        return proportions, counts

    def aggregate_across_ranks(self, fabric: Any) -> None:
        """
        Aggregate metrics across all ranks using all_reduce.

        This modifies the metrics in-place to contain globally aggregated values.
        Should only be called on metrics that haven't been aggregated yet.

        Args:
            fabric: Lightning Fabric instance for distributed operations
        """
        if (
            fabric is None
            or not hasattr(fabric, "world_size")
            or fabric.world_size == 1
        ):
            # Single process, no aggregation needed
            return

        # ddp_all_reduce is imported at module level

        # Aggregate training metrics
        for loader_metrics in self.train_metrics.values():
            # Aggregate counts
            loader_metrics.sample_count = int(
                ddp_all_reduce(
                    fabric,
                    torch.tensor(loader_metrics.sample_count, dtype=torch.float32),
                    op="sum",
                ).item()
            )
            loader_metrics.batch_count = int(
                ddp_all_reduce(
                    fabric,
                    torch.tensor(loader_metrics.batch_count, dtype=torch.float32),
                    op="sum",
                ).item()
            )

            # Aggregate metric sums and counts for proper averaging
            for metric_key, values in loader_metrics.metrics.items():
                if values:
                    # Sum of all values on this rank
                    local_sum = sum(values)
                    local_count = len(values)

                    # All-reduce sum and count
                    global_sum = ddp_all_reduce(
                        fabric, torch.tensor(local_sum, dtype=torch.float32), op="sum"
                    ).item()
                    global_count = int(
                        ddp_all_reduce(
                            fabric,
                            torch.tensor(local_count, dtype=torch.float32),
                            op="sum",
                        ).item()
                    )

                    # Replace with globally averaged value
                    if global_count > 0:
                        global_avg = global_sum / global_count
                        loader_metrics.metrics[metric_key] = [global_avg] * local_count

        # Aggregate validation metrics
        for loader_metrics in self.val_metrics.values():
            # Aggregate counts
            loader_metrics.sample_count = int(
                ddp_all_reduce(
                    fabric,
                    torch.tensor(loader_metrics.sample_count, dtype=torch.float32),
                    op="sum",
                ).item()
            )
            loader_metrics.batch_count = int(
                ddp_all_reduce(
                    fabric,
                    torch.tensor(loader_metrics.batch_count, dtype=torch.float32),
                    op="sum",
                ).item()
            )

            # Aggregate metric sums and counts
            for metric_key, values in loader_metrics.metrics.items():
                if values:
                    local_sum = sum(values)
                    local_count = len(values)

                    global_sum = ddp_all_reduce(
                        fabric, torch.tensor(local_sum, dtype=torch.float32), op="sum"
                    ).item()
                    global_count = int(
                        ddp_all_reduce(
                            fabric,
                            torch.tensor(local_count, dtype=torch.float32),
                            op="sum",
                        ).item()
                    )

                    if global_count > 0:
                        global_avg = global_sum / global_count
                        loader_metrics.metrics[metric_key] = [global_avg] * local_count

    def reset_epoch(self, phase: str = "train") -> None:
        """
        Reset metrics for a new epoch.

        Args:
            phase: Phase to reset ("train", "val", or "both")
        """
        if phase in ("train", "both"):
            for loader_metrics in self.train_metrics.values():
                loader_metrics.reset()

        if phase in ("val", "both"):
            for loader_metrics in self.val_metrics.values():
                loader_metrics.reset()

    def save_epoch_summary(
        self, epoch: int, extra_info: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Save summary for the completed epoch.

        Args:
            epoch: Epoch number
            extra_info: Additional information to include

        Returns:
            Epoch summary dictionary
        """
        summary = {
            "epoch": epoch,
            "train_metrics": self.get_train_metrics(),
            "val_metrics": self.get_val_metrics(),
        }

        if self.track_proportions:
            proportions, counts = self.get_loader_proportions()
            summary["loader_proportions"] = proportions
            summary["loader_counts"] = counts

        if extra_info:
            summary.update(extra_info)

        self.epoch_summaries.append(summary)
        return summary

    def _aggregate_metrics(
        self,
        loader_metrics_dict: dict[int, LoaderMetrics],
        prefix: str,
    ) -> dict[str, float]:
        """
        Aggregate metrics across multiple loaders.

        Args:
            loader_metrics_dict: Dictionary of LoaderMetrics by index
            prefix: Prefix for aggregated metric names

        Returns:
            Dictionary of aggregated metrics
        """
        if not loader_metrics_dict:
            return {}

        # Collect all metric keys
        all_keys: set[str] = set()
        for loader_metrics in loader_metrics_dict.values():
            all_keys.update(loader_metrics.metrics.keys())

        result = {}

        for key in all_keys:
            values = []
            weights = []

            for loader_metrics in loader_metrics_dict.values():
                avg = loader_metrics.get_average(key)
                if avg is not None:
                    values.append(avg)
                    weights.append(loader_metrics.sample_count)

            if not values:
                continue

            # Apply aggregation strategy
            if self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
                if sum(weights) > 0:
                    weighted_sum = sum(v * w for v, w in zip(values, weights))
                    result[f"{prefix}/{key}"] = weighted_sum / sum(weights)
            elif self.aggregation_strategy == AggregationStrategy.SIMPLE_AVERAGE:
                result[f"{prefix}/{key}"] = sum(values) / len(values)
            elif self.aggregation_strategy == AggregationStrategy.SUM:
                result[f"{prefix}/{key}"] = sum(values)
            elif self.aggregation_strategy == AggregationStrategy.MAX:
                result[f"{prefix}/{key}"] = max(values)
            elif self.aggregation_strategy == AggregationStrategy.MIN:
                result[f"{prefix}/{key}"] = min(values)

        return result

    def get_best_metric(
        self,
        metric_name: str,
        mode: str = "min",
        phase: str = "val",
    ) -> tuple[float | None, int | None]:
        """
        Get best value of a metric across all epochs.

        Args:
            metric_name: Name of the metric to check
            mode: "min" for lower is better, "max" for higher is better
            phase: "train" or "val"

        Returns:
            Tuple of (best_value, epoch_number) or (None, None) if not found
        """
        best_value = None
        best_epoch = None

        for summary in self.epoch_summaries:
            metrics_key = f"{phase}_metrics"
            if metrics_key not in summary:
                continue

            metrics = summary[metrics_key]

            # Look for exact match or partial match
            value = None
            for key, val in metrics.items():
                if key == metric_name or key.endswith(f"/{metric_name}"):
                    value = val
                    break

            if value is None:
                continue

            if (
                best_value is None
                or (mode == "min" and value < best_value)
                or (mode == "max" and value > best_value)
            ):
                best_value = value
                best_epoch = summary["epoch"]

        return best_value, best_epoch
