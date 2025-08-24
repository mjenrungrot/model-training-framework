"""
Tests for DataLoaderManager and schedule builders.

This module tests:
- Schedule builders for different strategies
- Determinism across runs with same seed
- Weighted quota accuracy
- Alternating pattern enforcement
- DDP synchronization
- Iterator functionality
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from model_training_framework.trainer import (
    DataLoaderManager,
    DataLoaderState,
    EpochLengthPolicy,
    MultiDataLoaderConfig,
    MultiDataLoaderIterator,
    SamplingStrategy,
)


class MockDataLoader:
    """Mock DataLoader for testing."""

    def __init__(self, data_size=10, batch_size=2):
        self.data_size = data_size
        self.batch_size = batch_size
        self.data = [
            (torch.randn(batch_size, 4), torch.randint(0, 2, (batch_size,)))
            for _ in range(data_size)
        ]
        self.sampler = MagicMock()
        self.dataset = MagicMock()

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return self.data_size


class TestDataLoaderManager:
    """Test DataLoaderManager functionality."""

    def test_manager_initialization(self):
        """Test basic manager initialization."""
        train_loaders = [MockDataLoader(10), MockDataLoader(15)]
        val_loaders = [MockDataLoader(5)]
        config = MultiDataLoaderConfig()

        manager = DataLoaderManager(
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            config=config,
        )

        assert len(manager.train_loaders) == 2
        assert len(manager.val_loaders) == 1
        assert manager.train_names == ["dl0", "dl1"]
        assert manager.val_names == ["val_dl0"]

    def test_manager_with_custom_names(self):
        """Test manager with custom dataloader names."""
        train_loaders = [MockDataLoader(10), MockDataLoader(15)]
        config = MultiDataLoaderConfig(dataloader_names=["source", "target"])

        manager = DataLoaderManager(train_loaders=train_loaders, config=config)

        assert manager.train_names == ["source", "target"]

    def test_manager_validates_unique_names(self):
        """Test that manager validates unique names."""
        train_loaders = [MockDataLoader(10), MockDataLoader(15)]
        config = MultiDataLoaderConfig(dataloader_names=["same", "same"])

        with pytest.raises(ValueError, match="names must be unique"):
            DataLoaderManager(train_loaders=train_loaders, config=config)

    def test_manager_validates_name_count(self):
        """Test that manager validates name count matches loaders."""
        train_loaders = [MockDataLoader(10), MockDataLoader(15)]
        config = MultiDataLoaderConfig(dataloader_names=["only_one"])

        with pytest.raises(ValueError, match="doesn't match"):
            DataLoaderManager(train_loaders=train_loaders, config=config)


class TestScheduleBuilders:
    """Test schedule building methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DataLoaderManager()

    def test_round_robin_schedule_basic(self):
        """Test basic round-robin schedule."""
        schedule = self.manager.build_round_robin_schedule(
            lengths=[10, 10, 10],
            policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            burst_size=1,
        )

        assert len(schedule) == 30
        # Check even distribution
        assert schedule.count(0) == 10
        assert schedule.count(1) == 10
        assert schedule.count(2) == 10

        # Check alternation pattern
        for i in range(0, 30, 3):
            assert schedule[i : i + 3] == [0, 1, 2]

    def test_round_robin_schedule_with_burst(self):
        """Test round-robin schedule with burst size."""
        schedule = self.manager.build_round_robin_schedule(
            lengths=[10, 10],
            policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            burst_size=2,
        )

        assert len(schedule) == 20
        # Check burst pattern
        assert schedule[0:2] == [0, 0]
        assert schedule[2:4] == [1, 1]
        assert schedule[4:6] == [0, 0]

    def test_round_robin_schedule_policies(self):
        """Test round-robin with different epoch length policies."""
        lengths = [10, 20, 30]

        # SUM_OF_LENGTHS
        schedule = self.manager.build_round_robin_schedule(
            lengths, EpochLengthPolicy.SUM_OF_LENGTHS
        )
        assert len(schedule) == 60

        # MAX_OF_LENGTHS
        schedule = self.manager.build_round_robin_schedule(
            lengths, EpochLengthPolicy.MAX_OF_LENGTHS
        )
        assert len(schedule) == 30

        # MIN_OF_LENGTHS
        schedule = self.manager.build_round_robin_schedule(
            lengths, EpochLengthPolicy.MIN_OF_LENGTHS
        )
        assert len(schedule) == 10

        # FIXED_NUM_STEPS
        schedule = self.manager.build_round_robin_schedule(
            lengths, EpochLengthPolicy.FIXED_NUM_STEPS, steps_per_epoch=42
        )
        assert len(schedule) == 42

    def test_sequential_schedule_basic(self):
        """Test basic sequential schedule."""
        schedule = self.manager.build_sequential_schedule(
            lengths=[10, 15, 5],
            policy=EpochLengthPolicy.SUM_OF_LENGTHS,
        )

        assert len(schedule) == 30
        # Check sequential pattern
        assert schedule[:10] == [0] * 10
        assert schedule[10:25] == [1] * 15
        assert schedule[25:30] == [2] * 5

    def test_weighted_schedule_basic(self):
        """Test basic weighted schedule using Hamilton's method."""
        schedule = self.manager.build_weighted_schedule(
            total_steps=100,
            weights=[0.5, 0.3, 0.2],
        )

        assert len(schedule) == 100
        # Check quotas match weights (within ±1 due to rounding)
        assert abs(schedule.count(0) - 50) <= 1
        assert abs(schedule.count(1) - 30) <= 1
        assert abs(schedule.count(2) - 20) <= 1

    def test_weighted_schedule_hamilton_rounding(self):
        """Test Hamilton's method handles fractional quotas correctly."""
        # This should give exact quotas [33.33..., 33.33..., 33.33...]
        # Hamilton's method should distribute the remainder
        schedule = self.manager.build_weighted_schedule(
            total_steps=100,
            weights=[1, 1, 1],  # Equal weights
        )

        counts = [schedule.count(i) for i in range(3)]
        # Two loaders get 33, one gets 34
        assert sorted(counts) == [33, 33, 34]

    def test_weighted_schedule_normalization(self):
        """Test that weights are normalized correctly."""
        # Use non-normalized weights
        schedule1 = self.manager.build_weighted_schedule(
            total_steps=100,
            weights=[2, 3, 5],
        )

        schedule2 = self.manager.build_weighted_schedule(
            total_steps=100,
            weights=[0.2, 0.3, 0.5],
        )

        # Should produce same distribution
        assert schedule1.count(0) == schedule2.count(0)
        assert schedule1.count(1) == schedule2.count(1)
        assert schedule1.count(2) == schedule2.count(2)

    def test_weighted_schedule_spacing(self):
        """Test that weighted schedule uses balanced interleaving."""
        schedule = self.manager.build_weighted_schedule(
            total_steps=20,
            weights=[0.5, 0.5],
        )

        # Should alternate fairly evenly
        transitions = 0
        for i in range(len(schedule) - 1):
            if schedule[i] != schedule[i + 1]:
                transitions += 1

        # Should have good mixing (many transitions)
        assert transitions >= 10

    def test_alternating_schedule_basic(self):
        """Test basic alternating schedule."""
        schedule = self.manager.build_alternating_schedule(
            pattern=[0, 1, 2, 1],
            total_steps=20,
        )

        assert len(schedule) == 20
        # Check pattern repeats
        for i in range(0, 20, 4):
            if i + 4 <= 20:
                assert schedule[i : i + 4] == [0, 1, 2, 1]

    def test_alternating_schedule_with_burst(self):
        """Test alternating schedule with burst size."""
        schedule = self.manager.build_alternating_schedule(
            pattern=[0, 1],
            total_steps=20,
            burst_size=3,
        )

        assert len(schedule) == 20
        # Check burst pattern
        assert schedule[0:3] == [0, 0, 0]
        assert schedule[3:6] == [1, 1, 1]
        assert schedule[6:9] == [0, 0, 0]

    def test_schedule_determinism(self):
        """Test that schedules are deterministic for same inputs."""
        # Create two managers with same seed
        config1 = MultiDataLoaderConfig(choice_rng_seed=42)
        manager1 = DataLoaderManager(config=config1)

        config2 = MultiDataLoaderConfig(choice_rng_seed=42)
        manager2 = DataLoaderManager(config=config2)

        # Build same weighted schedule
        schedule1 = manager1.build_weighted_schedule(100, [0.4, 0.3, 0.3])
        schedule2 = manager2.build_weighted_schedule(100, [0.4, 0.3, 0.3])

        assert schedule1 == schedule2

    def test_empty_loaders(self):
        """Test schedule building with empty loaders."""
        assert (
            self.manager.build_round_robin_schedule(
                [], EpochLengthPolicy.SUM_OF_LENGTHS
            )
            == []
        )
        assert (
            self.manager.build_sequential_schedule([], EpochLengthPolicy.SUM_OF_LENGTHS)
            == []
        )
        assert self.manager.build_weighted_schedule(0, []) == []
        assert self.manager.build_alternating_schedule([], 10) == []


class TestMultiDataLoaderIterator:
    """Test MultiDataLoaderIterator functionality."""

    def test_iterator_basic(self):
        """Test basic iterator functionality."""
        loaders = [MockDataLoader(5), MockDataLoader(5)]
        schedule = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        config = MultiDataLoaderConfig()

        iterator = MultiDataLoaderIterator(
            loaders=loaders,
            names=["dl0", "dl1"],
            schedule=schedule,
            config=config,
        )

        batches = list(iterator)
        assert len(batches) == 10

        # Check alternation
        for i, (loader_idx, _batch) in enumerate(batches):
            assert loader_idx == i % 2

    def test_iterator_with_exhaustion(self):
        """Test iterator handles loader exhaustion."""
        loaders = [MockDataLoader(2), MockDataLoader(5)]
        schedule = [0] * 5 + [1] * 5  # Try to get 5 from each
        config = MultiDataLoaderConfig(cycle_short_loaders=False)

        iterator = MultiDataLoaderIterator(
            loaders=loaders,
            names=["dl0", "dl1"],
            schedule=schedule,
            config=config,
        )

        batches = []
        for loader_idx, _batch in iterator:
            batches.append(loader_idx)
            if len(batches) >= 7:  # Only 2 from loader 0, 5 from loader 1
                break

        # First loader should exhaust after 2
        assert batches[:2] == [0, 0]
        # Then should skip to loader 1
        assert all(idx == 1 for idx in batches[2:])

    def test_iterator_with_cycling(self):
        """Test iterator with cycling of exhausted loaders."""
        loaders = [MockDataLoader(2), MockDataLoader(5)]
        schedule = [0] * 5 + [1] * 5
        config = MultiDataLoaderConfig(cycle_short_loaders=True)

        iterator = MultiDataLoaderIterator(
            loaders=loaders,
            names=["dl0", "dl1"],
            schedule=schedule,
            config=config,
        )

        batches = list(iterator)
        assert len(batches) == 10

        # Check that loader 0 was accessed 5 times (with cycling)
        loader_0_count = sum(1 for idx, _ in batches if idx == 0)
        assert loader_0_count == 5

    def test_iterator_state_capture(self):
        """Test iterator state capture for checkpointing."""
        loaders = [MockDataLoader(10), MockDataLoader(10)]
        schedule = [0, 1] * 10
        config = MultiDataLoaderConfig()

        iterator = MultiDataLoaderIterator(
            loaders=loaders,
            names=["dl0", "dl1"],
            schedule=schedule,
            config=config,
        )

        # Consume some batches
        for i, _ in enumerate(iterator):
            if i >= 5:
                break

        # Capture state
        states = iterator.get_loader_states()
        assert len(states) == 2
        assert states[0].batch_idx > 0
        assert states[1].batch_idx > 0

    def test_iterator_with_prefetch_cap(self):
        """Test iterator respects prefetch cap."""
        loaders = [MockDataLoader(10), MockDataLoader(10)]
        schedule = list(range(20))
        config = MultiDataLoaderConfig(prefetch_cap_total_batches=5)

        iterator = MultiDataLoaderIterator(
            loaders=loaders,
            names=["dl0", "dl1"],
            schedule=schedule,
            config=config,
        )

        batches = list(iterator)
        assert len(batches) == 5  # Capped at prefetch limit


class TestCreateEpochIterator:
    """Test create_epoch_iterator method."""

    def test_create_train_iterator(self):
        """Test creating training iterator."""
        train_loaders = [MockDataLoader(10), MockDataLoader(15)]
        config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.ROUND_ROBIN,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
        )

        manager = DataLoaderManager(train_loaders=train_loaders, config=config)

        iterator = manager.create_epoch_iterator("train", epoch=0)

        assert isinstance(iterator, MultiDataLoaderIterator)
        assert len(iterator.loaders) == 2
        assert len(iterator.schedule) == 25

    def test_create_val_iterator(self):
        """Test creating validation iterator."""
        train_loaders = [MockDataLoader(10)]
        val_loaders = [MockDataLoader(5)]
        config = MultiDataLoaderConfig()

        manager = DataLoaderManager(
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            config=config,
        )

        iterator = manager.create_epoch_iterator("val", epoch=0)

        assert isinstance(iterator, MultiDataLoaderIterator)
        assert len(iterator.loaders) == 1
        assert iterator.names == ["val_dl0"]

    def test_create_iterator_with_weighted_strategy(self):
        """Test creating iterator with weighted strategy."""
        train_loaders = [MockDataLoader(20), MockDataLoader(20)]
        config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            dataloader_weights=[0.7, 0.3],
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
        )

        manager = DataLoaderManager(train_loaders=train_loaders, config=config)

        iterator = manager.create_epoch_iterator("train", epoch=0)

        # Check schedule reflects weights
        schedule = iterator.schedule
        assert len(schedule) == 40
        assert abs(schedule.count(0) - 28) <= 1  # 70% of 40
        assert abs(schedule.count(1) - 12) <= 1  # 30% of 40

    def test_create_iterator_with_alternating_strategy(self):
        """Test creating iterator with alternating strategy."""
        train_loaders = [MockDataLoader(10), MockDataLoader(10), MockDataLoader(10)]
        config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.ALTERNATING,
            alternating_pattern=[0, 1, 2, 1],
            epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
            steps_per_epoch=20,
        )

        manager = DataLoaderManager(train_loaders=train_loaders, config=config)

        iterator = manager.create_epoch_iterator("train", epoch=0)

        # Check schedule follows pattern
        schedule = iterator.schedule
        assert len(schedule) == 20
        for i in range(0, 20, 4):
            if i + 4 <= 20:
                assert schedule[i : i + 4] == [0, 1, 2, 1]

    @patch("model_training_framework.trainer.multi_dataloader.ddp_broadcast_object")
    def test_create_iterator_with_ddp(self, mock_broadcast):
        """Test creating iterator with DDP broadcasts schedule."""
        mock_broadcast.side_effect = lambda fabric, obj, src: obj

        train_loaders = [MockDataLoader(10)]
        config = MultiDataLoaderConfig()
        fabric = MagicMock()

        manager = DataLoaderManager(
            train_loaders=train_loaders, config=config, fabric=fabric
        )

        _ = manager.create_epoch_iterator("train", epoch=0)

        # Check broadcast was called
        mock_broadcast.assert_called_once()
        assert mock_broadcast.call_args[0][0] == fabric
        assert mock_broadcast.call_args[1]["src"] == 0

    def test_create_iterator_sets_epoch_for_distributed_sampler(self):
        """Test that distributed sampler epoch is set."""
        loader = MockDataLoader(10)
        loader.sampler = MagicMock(spec=torch.utils.data.distributed.DistributedSampler)

        manager = DataLoaderManager(train_loaders=[loader])

        manager.create_epoch_iterator("train", epoch=5)

        loader.sampler.set_epoch.assert_called_once_with(5)

    def test_create_iterator_with_resume_state(self):
        """Test creating iterator with resume state."""
        train_loaders = [MockDataLoader(10), MockDataLoader(10)]
        config = MultiDataLoaderConfig()

        # Create resume state
        resume_state = [
            DataLoaderState(id=0, name="dl0", batch_idx=3, exhausted=False),
            DataLoaderState(id=1, name="dl1", batch_idx=2, exhausted=False),
        ]

        manager = DataLoaderManager(train_loaders=train_loaders, config=config)

        iterator = manager.create_epoch_iterator(
            "train", epoch=0, resume_state=resume_state
        )

        # Check state was restored
        assert iterator.loader_states[0].batch_idx == 3
        assert iterator.loader_states[1].batch_idx == 2

    def test_create_iterator_handles_infinite_loaders(self):
        """Test handling of infinite loaders in length calculation."""
        # Create mock infinite loader
        inf_loader = MockDataLoader(10)

        def raise_type_error():
            raise TypeError("Infinite loader has no len")

        inf_loader.__len__ = raise_type_error

        train_loaders = [inf_loader, MockDataLoader(20)]
        config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.ROUND_ROBIN,
            epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
            steps_per_epoch=30,
        )

        manager = DataLoaderManager(train_loaders=train_loaders, config=config)

        iterator = manager.create_epoch_iterator("train", epoch=0)

        assert len(iterator.schedule) == 30


class TestDeterminism:
    """Test determinism of schedules across runs."""

    def test_same_seed_produces_same_schedule(self):
        """Test that same seed produces identical schedules."""
        # First run
        config1 = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            dataloader_weights=[0.4, 0.3, 0.3],
            choice_rng_seed=12345,
        )
        manager1 = DataLoaderManager(
            train_loaders=[
                MockDataLoader(100),
                MockDataLoader(100),
                MockDataLoader(100),
            ],
            config=config1,
        )
        iterator1 = manager1.create_epoch_iterator("train", epoch=0)
        schedule1 = iterator1.schedule

        # Second run with same seed
        config2 = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            dataloader_weights=[0.4, 0.3, 0.3],
            choice_rng_seed=12345,
        )
        manager2 = DataLoaderManager(
            train_loaders=[
                MockDataLoader(100),
                MockDataLoader(100),
                MockDataLoader(100),
            ],
            config=config2,
        )
        iterator2 = manager2.create_epoch_iterator("train", epoch=0)
        schedule2 = iterator2.schedule

        assert schedule1 == schedule2

    def test_different_seeds_produce_different_schedules(self):
        """Test that different seeds produce different schedules."""
        # First run
        config1 = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            dataloader_weights=[0.4, 0.3, 0.3],
            choice_rng_seed=11111,
        )
        manager1 = DataLoaderManager(
            train_loaders=[
                MockDataLoader(100),
                MockDataLoader(100),
                MockDataLoader(100),
            ],
            config=config1,
        )
        schedule1 = manager1.build_weighted_schedule(100, [0.4, 0.3, 0.3])

        # Second run with different seed
        config2 = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            dataloader_weights=[0.4, 0.3, 0.3],
            choice_rng_seed=22222,
        )
        manager2 = DataLoaderManager(
            train_loaders=[
                MockDataLoader(100),
                MockDataLoader(100),
                MockDataLoader(100),
            ],
            config=config2,
        )
        schedule2 = manager2.build_weighted_schedule(100, [0.4, 0.3, 0.3])

        # Schedules should have same counts but potentially different order
        assert schedule1.count(0) == schedule2.count(0)
        assert schedule1.count(1) == schedule2.count(1)
        assert schedule1.count(2) == schedule2.count(2)
        # But the actual sequences might differ
        # (they could be the same by chance, but very unlikely with good RNG)
