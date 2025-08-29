"""
Tests for Hamilton's Method Quota Allocation

This module tests the Hamilton's method (largest remainder method) implementation
used for allocating batches to dataloaders in the WEIGHTED sampling strategy.
"""

import pytest

from model_training_framework.trainer.utils import balanced_interleave


def compute_weighted_targets_hamilton(
    weights: list[float], total_steps: int
) -> list[int]:
    """
    Compute quota allocation using Hamilton's method (largest remainder).

    This is extracted from WeightedIterator.build_weighted_schedule for testing.

    Args:
        weights: List of weights for each loader
        total_steps: Total number of steps to allocate

    Returns:
        List of integer quotas for each loader
    """
    n_loaders = len(weights)
    if n_loaders == 0:
        raise ValueError("Weights list cannot be empty")
    if total_steps < 0:
        raise ValueError("Total steps cannot be negative")
    if any(w < 0 for w in weights):
        raise ValueError("Weights cannot be negative")

    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError(f"Weights must sum to positive value, got {total_weight}")
    norm_weights = [w / total_weight for w in weights]

    # Calculate quotas using Hamilton's method (largest remainder)
    exact_quotas = [total_steps * w for w in norm_weights]
    integer_quotas = [int(q) for q in exact_quotas]
    remainders = [
        exact - integer for exact, integer in zip(exact_quotas, integer_quotas)
    ]

    # Distribute remaining steps to largest remainders
    remaining_steps = total_steps - sum(integer_quotas)
    if remaining_steps > 0:
        # Get indices sorted by remainder (descending)
        sorted_indices = sorted(
            range(n_loaders), key=lambda i: remainders[i], reverse=True
        )
        for i in range(remaining_steps):
            integer_quotas[sorted_indices[i]] += 1

    return integer_quotas


class TestHamiltonMethod:
    """Test Hamilton's method for quota allocation."""

    def test_basic_allocation(self):
        """Test basic quota allocation with simple weights."""
        weights = [0.5, 0.3, 0.2]
        total_steps = 100

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        # Should get [50, 30, 20]
        assert quotas == [50, 30, 20]
        assert sum(quotas) == total_steps

    def test_remainder_distribution(self):
        """Test that remainders are distributed to highest fractional parts."""
        weights = [0.33, 0.33, 0.34]
        total_steps = 10

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        # 0.33 * 10 = 3.3, 0.33 * 10 = 3.3, 0.34 * 10 = 3.4
        # Base: [3, 3, 3] = 9, need to distribute 1 more
        # 0.34 has highest remainder (0.4), so it gets the extra
        assert quotas == [3, 3, 4]
        assert sum(quotas) == total_steps

    def test_multiple_remainders(self):
        """Test distribution when multiple remainders need allocation."""
        weights = [0.25, 0.25, 0.25, 0.25]
        total_steps = 11

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        # 0.25 * 11 = 2.75 for each
        # Base: [2, 2, 2, 2] = 8, need to distribute 3 more
        # All have same remainder (0.75), so first 3 get extra
        assert sum(quotas) == total_steps
        assert quotas.count(3) == 3  # Three get 3
        assert quotas.count(2) == 1  # One gets 2

    def test_very_small_weights(self):
        """Test allocation with very small weights."""
        weights = [0.9, 0.05, 0.05]
        total_steps = 20

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        # 0.9 * 20 = 18, 0.05 * 20 = 1, 0.05 * 20 = 1
        assert quotas == [18, 1, 1]
        assert sum(quotas) == total_steps

    def test_extreme_weight_ratios(self):
        """Test with extreme weight ratios."""
        weights = [0.99, 0.005, 0.005]
        total_steps = 100

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        # 0.99 * 100 = 99, 0.005 * 100 = 0.5, 0.005 * 100 = 0.5
        # Base: [99, 0, 0] = 99, need to distribute 1 more
        # Both small weights have remainder 0.5, first gets it
        assert quotas[0] == 99
        assert sum(quotas) == total_steps
        assert sum(quotas[1:]) == 1  # Small weights share 1 batch

    def test_single_loader(self):
        """Test with a single loader."""
        weights = [1.0]
        total_steps = 50

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        assert quotas == [50]
        assert sum(quotas) == total_steps

    def test_many_loaders(self):
        """Test with many loaders."""
        num_loaders = 10
        weights = [1.0 / num_loaders] * num_loaders
        total_steps = 100

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        assert sum(quotas) == total_steps
        assert all(q == 10 for q in quotas)  # All should get equal share

    def test_uneven_many_loaders(self):
        """Test with many loaders and uneven total."""
        num_loaders = 7
        weights = [1.0 / num_loaders] * num_loaders
        total_steps = 100

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        assert sum(quotas) == total_steps
        # 100 / 7 = 14.285...
        # Base: 14 * 7 = 98, need to distribute 2 more
        assert quotas.count(15) == 2  # Two get 15
        assert quotas.count(14) == 5  # Five get 14

    def test_deterministic_allocation(self):
        """Test that allocation is deterministic."""
        weights = [0.33, 0.33, 0.34]
        total_steps = 100

        # Run multiple times
        results = []
        for _ in range(10):
            quotas = compute_weighted_targets_hamilton(weights, total_steps)
            results.append(quotas)

        # All should be identical
        for result in results[1:]:
            assert result == results[0], "Hamilton's method should be deterministic"

    def test_fairness_over_time(self):
        """Test that allocation maintains fairness as total steps increase."""
        weights = [0.6, 0.3, 0.1]

        cumulative_error = [0.0, 0.0, 0.0]
        for total_steps in range(10, 101, 10):
            quotas = compute_weighted_targets_hamilton(weights, total_steps)

            # Check deviation from ideal
            for i, (quota, weight) in enumerate(zip(quotas, weights)):
                ideal = weight * total_steps
                error = abs(quota - ideal)
                cumulative_error[i] += error

                # Error should be at most 1 (due to integer rounding)
                assert error <= 1.0, f"Error too large: {error} for weight {weight}"

        # Average error should be small
        avg_errors = [err / 10 for err in cumulative_error]
        for avg_err in avg_errors:
            assert avg_err < 0.5, f"Average error too large: {avg_err}"

    def test_zero_total_steps(self):
        """Test edge case with zero total steps."""
        weights = [0.5, 0.5]
        total_steps = 0

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        assert quotas == [0, 0]
        assert sum(quotas) == 0

    def test_one_total_step(self):
        """Test edge case with one total step."""
        weights = [0.5, 0.5]
        total_steps = 1

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        # One loader should get the single step
        assert sum(quotas) == 1
        assert quotas.count(1) == 1
        assert quotas.count(0) == 1

    def test_weights_normalization(self):
        """Test that weights don't need to sum to 1."""
        # Weights that don't sum to 1 should still work
        weights = [2.0, 3.0, 5.0]  # Sum = 10
        total_steps = 100

        quotas = compute_weighted_targets_hamilton(weights, total_steps)

        # Should normalize: [0.2, 0.3, 0.5]
        assert quotas == [20, 30, 50]
        assert sum(quotas) == total_steps

    def test_negative_weights_error(self):
        """Test that negative weights raise an error."""
        weights = [0.5, -0.3, 0.8]
        total_steps = 100

        with pytest.raises(ValueError, match="negative"):
            compute_weighted_targets_hamilton(weights, total_steps)

    def test_zero_weights_error(self):
        """Test that all-zero weights raise an error."""
        weights = [0.0, 0.0, 0.0]
        total_steps = 100

        with pytest.raises(ValueError, match="Weights must sum to positive value"):
            compute_weighted_targets_hamilton(weights, total_steps)

    def test_empty_weights_error(self):
        """Test that empty weights raise an error."""
        weights = []
        total_steps = 100

        with pytest.raises(ValueError, match="empty"):
            compute_weighted_targets_hamilton(weights, total_steps)


class TestBalancedInterleave:
    """Test balanced interleave function used with Hamilton's method."""

    def test_balanced_interleave_basic(self):
        """Test basic balanced interleaving."""
        quotas = [2, 3, 1]
        result = balanced_interleave(quotas)

        # Should create balanced distribution
        assert len(result) == sum(quotas)  # Total length matches
        assert result.count(0) == 2
        assert result.count(1) == 3
        assert result.count(2) == 1

    def test_balanced_interleave_equal(self):
        """Test interleaving with equal quotas."""
        quotas = [2, 2, 2]
        result = balanced_interleave(quotas)

        assert len(result) == 6
        assert result.count(0) == 2
        assert result.count(1) == 2
        assert result.count(2) == 2

    def test_balanced_interleave_single(self):
        """Test interleaving with single loader."""
        quotas = [5]
        result = balanced_interleave(quotas)

        assert result == [0, 0, 0, 0, 0]

    def test_balanced_interleave_empty(self):
        """Test interleaving with empty quotas."""
        quotas = []
        result = balanced_interleave(quotas)

        assert result == []
