"""
Tests for trainer utility functions.

This module tests trainer-specific utilities including:
- Deterministic seeding (seed_all)
- Batch counting heuristics
- Balanced interleaving for scheduling
- DDP helper functions
- Stopwatch timing utility
- Memory usage sampling
"""

import random
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from model_training_framework.trainer.utils import (
    Stopwatch,
    balanced_interleave,
    count_samples_in_batch,
    ddp_barrier,
    ddp_broadcast_object,
    ddp_is_primary,
    get_memory_usage,
    seed_all,
)


class TestSeedAll:
    """Test seed_all function for deterministic seeding."""

    def test_seed_all_changes_random_state(self):
        """Test that seed_all changes RNG states."""
        # Get initial random values
        initial_python = random.random()
        initial_numpy = np.random.random()
        initial_torch = torch.rand(1).item()

        # Seed with value 42
        seed_all(42)

        # Get new random values
        python_42_1 = random.random()
        numpy_42_1 = np.random.random()
        torch_42_1 = torch.rand(1).item()

        # Values should be different from initial
        assert python_42_1 != initial_python
        assert numpy_42_1 != initial_numpy
        assert torch_42_1 != initial_torch

        # Seed again with same value
        seed_all(42)

        # Should get same values again
        python_42_2 = random.random()
        numpy_42_2 = np.random.random()
        torch_42_2 = torch.rand(1).item()

        assert python_42_1 == python_42_2
        assert numpy_42_1 == numpy_42_2
        assert torch_42_1 == torch_42_2

    def test_seed_all_different_seeds(self):
        """Test that different seeds produce different sequences."""
        seed_all(123)
        vals_123 = [random.random(), np.random.random(), torch.rand(1).item()]

        seed_all(456)
        vals_456 = [random.random(), np.random.random(), torch.rand(1).item()]

        # Different seeds should produce different values
        assert vals_123 != vals_456

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.manual_seed")
    @patch("torch.cuda.manual_seed_all")
    def test_seed_all_with_cuda(self, mock_seed_all, mock_seed, mock_available):
        """Test CUDA seeding when available."""
        mock_available.return_value = True

        seed_all(999)

        # Check CUDA seeding was called
        mock_seed.assert_called_once_with(999)
        mock_seed_all.assert_called_once_with(999)


class TestCountSamplesInBatch:
    """Test count_samples_in_batch function."""

    def test_count_tensor_batch(self):
        """Test counting samples in tensor batch."""
        batch = torch.randn(32, 10)
        assert count_samples_in_batch(batch) == 32

        batch = torch.randn(16, 5, 5)
        assert count_samples_in_batch(batch) == 16

    def test_count_tuple_batch(self):
        """Test counting samples in tuple batch."""
        batch = (torch.randn(8, 10), torch.randn(8, 5))
        assert count_samples_in_batch(batch) == 8

        # Empty tuple
        batch = ()
        with pytest.raises(ValueError):
            count_samples_in_batch(batch)

    def test_count_list_batch(self):
        """Test counting samples in list batch."""
        batch = [torch.randn(4, 10), torch.randn(4, 5)]
        assert count_samples_in_batch(batch) == 4

    def test_count_dict_batch(self):
        """Test counting samples in dict batch."""
        # With 'input' key
        batch = {"input": torch.randn(16, 10), "target": torch.randn(16)}
        assert count_samples_in_batch(batch) == 16

        # With 'data' key
        batch = {"data": torch.randn(8, 10), "label": torch.randn(8)}
        assert count_samples_in_batch(batch) == 8

        # With 'x' key
        batch = {"x": torch.randn(4, 10), "y": torch.randn(4)}
        assert count_samples_in_batch(batch) == 4

        # No standard keys but has tensor values
        batch = {"features": torch.randn(12, 10), "labels": torch.randn(12)}
        assert count_samples_in_batch(batch) == 12

    def test_count_fallback_to_len(self):
        """Test fallback to len() for other types."""
        batch = list(range(25))
        assert count_samples_in_batch(batch) == 25

    def test_count_unsupported_type(self):
        """Test error for unsupported batch type."""
        batch = 42  # Not iterable
        with pytest.raises(ValueError, match="Cannot determine batch size"):
            count_samples_in_batch(batch)


class TestBalancedInterleave:
    """Test balanced_interleave function."""

    def test_balanced_interleave_basic(self):
        """Test basic balanced interleaving."""
        # Simple equal quotas
        result = balanced_interleave([1, 1, 1])
        assert len(result) == 3
        assert set(result) == {0, 1, 2}

        # Different quotas
        result = balanced_interleave([2, 3, 1])
        assert len(result) == 6
        assert result.count(0) == 2
        assert result.count(1) == 3
        assert result.count(2) == 1

    def test_balanced_interleave_spacing(self):
        """Test that interleaving is well-spaced."""
        result = balanced_interleave([3, 3])

        # Check alternation pattern
        transitions = 0
        for i in range(len(result) - 1):
            if result[i] != result[i + 1]:
                transitions += 1

        # Should have good mixing (5 transitions for perfect alternation)
        assert transitions >= 4

    def test_balanced_interleave_edge_cases(self):
        """Test edge cases for balanced interleaving."""
        # Empty quota
        assert balanced_interleave([]) == []

        # All zeros
        assert balanced_interleave([0, 0, 0]) == []

        # Single non-zero
        result = balanced_interleave([0, 5, 0])
        assert result == [1] * 5

        # Negative values (should be treated as 0)
        result = balanced_interleave([-1, 2, -3])
        assert result == [1] * 2

    def test_balanced_interleave_large_quotas(self):
        """Test with larger quotas."""
        result = balanced_interleave([10, 15, 5])
        assert len(result) == 30
        assert result.count(0) == 10
        assert result.count(1) == 15
        assert result.count(2) == 5

        # Check reasonable distribution
        # First third should have mix of all
        first_third = result[:10]
        assert 0 in first_third
        assert 1 in first_third
        assert 2 in first_third

    def test_balanced_interleave_proportions(self):
        """Test that proportions are maintained."""
        # 2:1 ratio
        result = balanced_interleave([20, 10])
        assert result.count(0) == 20
        assert result.count(1) == 10

        # Check that 0 appears roughly twice as often in any window
        window_size = 6
        for i in range(len(result) - window_size):
            window = result[i : i + window_size]
            count_0 = window.count(0)
            count_1 = window.count(1)
            # Ratio should be approximately 2:1
            if count_1 > 0:
                ratio = count_0 / count_1
                assert 1.0 <= ratio <= 3.0  # Allow some variance


class TestDDPHelpers:
    """Test DDP helper functions."""

    def test_ddp_is_primary_none_fabric(self):
        """Test is_primary with None fabric."""
        assert ddp_is_primary(None) is True

    def test_ddp_is_primary_with_fabric(self):
        """Test is_primary with fabric mock."""
        # Test with is_global_zero attribute
        fabric = MagicMock()
        fabric.is_global_zero = True
        assert ddp_is_primary(fabric) is True

        fabric.is_global_zero = False
        assert ddp_is_primary(fabric) is False

        # Test with global_rank attribute
        fabric = MagicMock(spec=[])  # No is_global_zero
        fabric.global_rank = 0
        assert ddp_is_primary(fabric) is True

        fabric.global_rank = 1
        assert ddp_is_primary(fabric) is False

        # Test with rank attribute
        fabric = MagicMock(spec=[])
        fabric.rank = 0
        assert ddp_is_primary(fabric) is True

        fabric.rank = 2
        assert ddp_is_primary(fabric) is False

    def test_ddp_barrier_none_fabric(self):
        """Test barrier with None fabric."""
        # Should not raise
        ddp_barrier(None)

    def test_ddp_barrier_with_fabric(self):
        """Test barrier with fabric mock."""
        fabric = MagicMock()
        fabric.barrier = MagicMock()

        ddp_barrier(fabric)
        fabric.barrier.assert_called_once()

        # Test with exception (single-process mode)
        fabric.barrier.side_effect = RuntimeError("Not in DDP")
        ddp_barrier(fabric)  # Should not raise

    def test_ddp_broadcast_object_none_fabric(self):
        """Test broadcast with None fabric."""
        obj = {"test": "data"}
        result = ddp_broadcast_object(None, obj)
        assert result is obj

    def test_ddp_broadcast_object_with_fabric(self):
        """Test broadcast with fabric mock."""
        fabric = MagicMock()
        obj = {"test": "data"}
        fabric.broadcast = MagicMock(return_value={"broadcasted": "data"})

        result = ddp_broadcast_object(fabric, obj, src=0)
        fabric.broadcast.assert_called_once_with(obj, src=0)
        assert result == {"broadcasted": "data"}

        # Test with exception (single-process mode)
        fabric.broadcast.side_effect = RuntimeError("Not in DDP")
        result = ddp_broadcast_object(fabric, obj)
        assert result is obj


class TestStopwatch:
    """Test Stopwatch class."""

    def test_stopwatch_basic(self):
        """Test basic stopwatch functionality."""
        sw = Stopwatch()

        # Initially stopped
        assert sw.elapsed_time() == 0.0
        assert not sw.running

        # Start and check running
        sw.start()
        assert sw.running
        time.sleep(0.01)  # Small delay

        elapsed = sw.elapsed_time()
        assert elapsed > 0.0

        # Stop and check
        total = sw.stop()
        assert not sw.running
        assert total > 0.0
        assert sw.elapsed_time() == total

    def test_stopwatch_reset(self):
        """Test stopwatch reset."""
        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)
        sw.stop()

        sw.reset()
        assert sw.elapsed_time() == 0.0
        assert not sw.running
        assert sw.laps == []

    def test_stopwatch_laps(self):
        """Test lap timing."""
        sw = Stopwatch()
        sw.start()

        time.sleep(0.01)
        lap1 = sw.lap()
        assert lap1 > 0.0

        time.sleep(0.01)
        lap2 = sw.lap()
        assert lap2 > 0.0

        laps = sw.get_laps()
        assert len(laps) == 2
        assert laps[0] == lap1
        assert laps[1] == lap2

    def test_stopwatch_resume(self):
        """Test stopwatch resume after stop."""
        sw = Stopwatch()

        # First period
        sw.start()
        time.sleep(0.01)
        first_elapsed = sw.stop()

        # Resume
        sw.start()
        time.sleep(0.01)
        total = sw.stop()

        assert total > first_elapsed


class TestMemoryUsage:
    """Test get_memory_usage function."""

    @patch("torch.cuda.is_available")
    def test_memory_usage_no_gpu(self, mock_available):
        """Test memory usage without GPU."""
        mock_available.return_value = False

        stats = get_memory_usage()

        # Should not have GPU stats
        assert "gpu_allocated_gb" not in stats
        assert "gpu_reserved_gb" not in stats

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.current_device")
    def test_memory_usage_with_gpu(
        self, mock_device, mock_props, mock_reserved, mock_allocated, mock_available
    ):
        """Test memory usage with GPU."""
        mock_available.return_value = True
        mock_allocated.return_value = 2 * (1024**3)  # 2 GB
        mock_reserved.return_value = 3 * (1024**3)  # 3 GB
        mock_device.return_value = 0

        # Mock device properties
        props = MagicMock()
        props.total_memory = 8 * (1024**3)  # 8 GB
        mock_props.return_value = props

        stats = get_memory_usage()

        assert stats["gpu_allocated_gb"] == pytest.approx(2.0)
        assert stats["gpu_reserved_gb"] == pytest.approx(3.0)
        assert stats["gpu_total_gb"] == pytest.approx(8.0)
        assert stats["gpu_free_gb"] == pytest.approx(6.0)

    @patch("model_training_framework.trainer.utils.psutil")
    @patch("torch.cuda.is_available")
    def test_memory_usage_with_cpu_stats(self, mock_cuda, mock_psutil):
        """Test CPU memory stats with psutil available."""
        mock_cuda.return_value = False

        # Mock psutil
        mock_memory = MagicMock()
        mock_memory.percent = 45.5
        mock_memory.used = 16 * (1024**3)
        mock_memory.available = 20 * (1024**3)
        mock_psutil.virtual_memory.return_value = mock_memory

        stats = get_memory_usage()

        assert stats["cpu_percent"] == 45.5
        assert stats["cpu_used_gb"] == pytest.approx(16.0)
        assert stats["cpu_available_gb"] == pytest.approx(20.0)
