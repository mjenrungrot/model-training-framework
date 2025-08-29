"""
Test to verify aggregation strategy wiring issues identified in code review.

This test file documents the gaps between configuration and implementation
for metrics aggregation that need to be fixed.
"""

from model_training_framework.config.schemas import ValAggregation, ValidationConfig
from model_training_framework.trainer.metrics import AggregationStrategy, MetricsManager


class TestAggregationWiringGaps:
    """Document and test the aggregation wiring gaps."""

    def test_validation_aggregation_not_wired(self):
        """Test that ValidationConfig.aggregation is not actually used."""
        # This documents the issue: ValidationConfig has aggregation_policy
        # but it's not used by the training loop

        val_config = ValidationConfig(
            aggregation=ValAggregation.MACRO_AVG_EQUAL_LOADERS
        )

        # The MetricsManager doesn't receive this configuration
        manager = MetricsManager(
            train_loader_names=["loader1"],
            val_loader_names=["val1"],
            # No way to pass val_config.aggregation here
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,  # Hardcoded
        )

        # Document the mismatch
        assert val_config.aggregation == ValAggregation.MACRO_AVG_EQUAL_LOADERS
        assert manager.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE
        # These should be connected but aren't

    def test_aggregation_enum_mismatch(self):
        """Test that ValAggregation and AggregationStrategy enums don't align."""
        # ValAggregation has:
        _val_options = [
            ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
            ValAggregation.MACRO_AVG_EQUAL_LOADERS,
            ValAggregation.PRIMARY_METRIC_PER_LOADER,
            ValAggregation.CUSTOM,
        ]

        # AggregationStrategy has:
        _agg_options = [
            AggregationStrategy.WEIGHTED_AVERAGE,  # Maps to MICRO
            AggregationStrategy.SIMPLE_AVERAGE,  # Maps to MACRO
            AggregationStrategy.SUM,
            AggregationStrategy.MAX,
            AggregationStrategy.MIN,
            # No PRIMARY_METRIC equivalent
        ]

        # Document the mapping needed
        _mapping_needed = {
            ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES: AggregationStrategy.WEIGHTED_AVERAGE,
            ValAggregation.MACRO_AVG_EQUAL_LOADERS: AggregationStrategy.SIMPLE_AVERAGE,
            ValAggregation.PRIMARY_METRIC_PER_LOADER: None,  # Not supported
            ValAggregation.CUSTOM: None,  # Not supported
        }

    def test_metrics_manager_construction_missing_config(self):
        """Test that MetricsManager is not constructed with config values."""
        # In GenericTrainer.__init__, MetricsManager is created without
        # considering ValidationConfig.aggregation_policy

        # Should be wired via a mapping from ValidationConfig to MetricsManager.


def map_val_aggregation_to_strategy(val_agg: ValAggregation) -> AggregationStrategy:
    """
    Proposed mapping function to connect ValidationConfig to MetricsManager.

    This should be added to the codebase to wire the configurations.
    """
    mapping = {
        ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES: AggregationStrategy.WEIGHTED_AVERAGE,
        ValAggregation.MACRO_AVG_EQUAL_LOADERS: AggregationStrategy.SIMPLE_AVERAGE,
        ValAggregation.PRIMARY_METRIC_PER_LOADER: AggregationStrategy.WEIGHTED_AVERAGE,  # Default
        ValAggregation.CUSTOM: AggregationStrategy.WEIGHTED_AVERAGE,  # Default
    }
    return mapping.get(val_agg, AggregationStrategy.WEIGHTED_AVERAGE)
