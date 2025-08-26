"""
Multi-DataLoader Training Support

This module provides the foundation for the multi-dataloader-only training engine.
All training, including single dataloader scenarios, must use the multi-dataloader API.

Features:
- DataLoaderManager for managing one or more dataloaders
- Multi-dataloader iterators with deterministic scheduling
- Sampling strategies (SEQUENTIAL, ROUND_ROBIN, WEIGHTED, ALTERNATING)
- Epoch length policies (SUM_OF_LENGTHS, MAX_OF_LENGTHS, MIN_OF_LENGTHS, FIXED_NUM_STEPS)
- Checkpointable iteration state for fault tolerance
- DDP synchronization support

Single Dataloader Usage:
    # Even with one dataloader, use the multi-dataloader API
    manager = DataLoaderManager(
        train_loaders=[single_loader],  # Wrap in list
        config=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            dataloader_names=["main"]
        )
    )

Multi-Dataloader Usage:
    # With multiple dataloaders
    manager = DataLoaderManager(
        train_loaders=[loader_a, loader_b, loader_c],
        config=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            dataloader_weights=[0.5, 0.3, 0.2],
            dataloader_names=["dataset_a", "dataset_b", "dataset_c"]
        )
    )
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
import math
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .config import (
    EpochLengthPolicy,
    MultiDataLoaderConfig,
    SamplingStrategy,
)
from .states import ChoiceRNGState, DataLoaderState
from .utils import (
    balanced_interleave,
    ddp_barrier,
    ddp_broadcast_object,
    ddp_is_primary,
)

if TYPE_CHECKING:
    # Imported only for type checking to avoid runtime dependency per TC002
    from torch.utils.data import DataLoader


@runtime_checkable
class CheckpointableIterable(Protocol):
    """Protocol for checkpointable iterable datasets."""

    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary for checkpointing."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        ...


class MultiDataLoaderIterator:
    """
    Iterator for multiple dataloaders with deterministic scheduling.

    Manages iteration over multiple dataloaders according to a pre-computed
    schedule, handling state management, exhaustion, and cycling.
    """

    def __init__(
        self,
        loaders: list[DataLoader],
        names: list[str],
        schedule: list[int],
        config: MultiDataLoaderConfig,
        loader_states: list[DataLoaderState] | None = None,
        fabric: Any = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize multi-dataloader iterator.

        Args:
            loaders: List of DataLoader objects
            names: Names for each loader
            schedule: Pre-computed schedule of loader indices
            config: Multi-dataloader configuration
            loader_states: Optional restored states for resume
            fabric: Optional Lightning Fabric for DDP
            logger: Optional logger
        """
        self.loaders = loaders
        self.names = names
        self.schedule = schedule
        self.config = config
        self.fabric = fabric
        self.logger = logger or logging.getLogger(__name__)

        # Initialize loader states
        if loader_states is None:
            self.loader_states = [
                DataLoaderState(
                    id=i,
                    name=names[i],
                    batch_idx=0,
                    exhausted=False,
                    sampler_state=None,
                    dataset_state=None,
                )
                for i in range(len(loaders))
            ]
        else:
            self.loader_states = loader_states

        # Current position in schedule
        self.schedule_position = 0

        # Loader iterators (created lazily)
        self.loader_iterators: list[Iterator | None] = [None] * len(loaders)

        # Track cycles for exhausted loaders
        self.loader_cycles = [0] * len(loaders)

        # Total batches yielded
        self.total_batches = 0

        # Prefetch tracking
        self.prefetched_batches = 0

    def _create_loader_iterator(self, loader_idx: int) -> Iterator:
        """Create iterator for a specific loader, restoring state if needed."""
        loader = self.loaders[loader_idx]
        state = self.loader_states[loader_idx]

        # Restore dataset/sampler state prior to iterator creation where possible
        # to avoid consuming elements during skip.
        dataset_restored = False
        if (
            hasattr(loader.dataset, "load_state_dict")
            and state.dataset_state is not None
        ):
            try:
                loader.dataset.load_state_dict(state.dataset_state)
                dataset_restored = True
                self.logger.debug(
                    f"Successfully restored dataset state for loader {loader_idx} "
                    f"at batch {state.batch_idx}"
                )
            except Exception:
                # Log and fall through to skip-based restoration if dataset restoration fails
                self.logger.debug(
                    "Dataset state restore failed; falling back to skip-based restoration",
                    exc_info=True,
                )

        sampler_restored = False
        if state.sampler_state is not None:
            try:
                # Prefer explicit load_state_dict if available
                batch_sampler = getattr(loader, "batch_sampler", None)
                sampler = getattr(loader, "sampler", None)
                if (
                    batch_sampler is not None
                    and hasattr(batch_sampler, "sampler")
                    and hasattr(batch_sampler.sampler, "load_state_dict")
                ):
                    batch_sampler.sampler.load_state_dict(state.sampler_state)
                    sampler_restored = True
                    self.logger.debug(
                        f"Successfully restored batch sampler state for loader {loader_idx} "
                        f"at batch {state.batch_idx}"
                    )
                elif sampler is not None and hasattr(sampler, "load_state_dict"):
                    sampler.load_state_dict(state.sampler_state)
                    sampler_restored = True
                    self.logger.debug(
                        f"Successfully restored sampler state for loader {loader_idx} "
                        f"at batch {state.batch_idx}"
                    )
                elif sampler is not None and hasattr(sampler, "set_state"):
                    # Support custom samplers exposing set_state
                    sampler.set_state(state.sampler_state)
                    sampler_restored = True
                    self.logger.debug(
                        f"Successfully restored sampler state via set_state for loader {loader_idx} "
                        f"at batch {state.batch_idx}"
                    )
            except Exception:
                # If sampler restoration fails, we'll fall back to skip-based approach
                sampler_restored = False

        # Create base iterator after attempting to restore state
        iterator = iter(loader)

        # Skip forward only when we cannot restore via dataset/sampler APIs
        if state.batch_idx > 0 and not (dataset_restored or sampler_restored):
            self.logger.warning(
                f"Falling back to skip-based restoration for loader {loader_idx} "
                f"(skipping {state.batch_idx} batches). "
                f"Consider implementing state_dict/load_state_dict methods on your "
                f"dataset or sampler for more efficient deterministic resume."
            )
            for _ in range(state.batch_idx):
                try:
                    next(iterator)
                except StopIteration:
                    # Loader exhausted during skip
                    state.exhausted = True
                    return iterator

        return iterator

    def _get_next_batch_from_loader(self, loader_idx: int) -> Any:
        """Get next batch from a specific loader, handling exhaustion and cycling."""
        state = self.loader_states[loader_idx]

        # Create iterator if needed
        if self.loader_iterators[loader_idx] is None:
            self.loader_iterators[loader_idx] = self._create_loader_iterator(loader_idx)

        iterator = self.loader_iterators[loader_idx]

        try:
            assert iterator is not None
            batch = next(iterator)
            state.batch_idx += 1
            return batch
        except StopIteration:
            # Loader exhausted
            state.exhausted = True

            # Handle cycling if enabled
            if self.config.cycle_short_loaders:
                # Reset loader for cycling
                self.logger.debug(
                    f"Cycling loader {self.names[loader_idx]} (cycle {self.loader_cycles[loader_idx] + 1})"
                )
                state.batch_idx = 0
                state.exhausted = False
                state.sampler_state = None
                state.dataset_state = None
                self.loader_cycles[loader_idx] += 1

                # Create new iterator
                self.loader_iterators[loader_idx] = self._create_loader_iterator(
                    loader_idx
                )

                # Try again
                try:
                    it2 = self.loader_iterators[loader_idx]
                    assert it2 is not None
                    batch = next(it2)
                    state.batch_idx += 1
                    return batch
                except StopIteration as err:
                    # Still exhausted after cycling (shouldn't happen with proper datasets)
                    raise RuntimeError(
                        f"Loader {self.names[loader_idx]} exhausted even after cycling"
                    ) from err
            else:
                raise

    def __iter__(self):
        """Return iterator object."""
        return self

    def __next__(self) -> tuple[int, Any]:
        """
        Get next batch according to schedule.

        Returns:
            Tuple of (loader_index, batch)
        """
        # Check if schedule exhausted
        if self.schedule_position >= len(self.schedule):
            raise StopIteration

        # Check prefetch cap
        if (
            self.config.prefetch_cap_total_batches is not None
            and self.prefetched_batches >= self.config.prefetch_cap_total_batches
        ):
            raise StopIteration

        # Get loader index from schedule (tolerate arbitrary integers)
        raw_idx = self.schedule[self.schedule_position]
        n_loaders = len(self.loaders)
        loader_idx = raw_idx % n_loaders if n_loaders > 0 else raw_idx

        # Handle exhausted loaders in WEIGHTED strategy with redistribution
        if (
            self.config.sampling_strategy == SamplingStrategy.WEIGHTED
            and self.loader_states[loader_idx].exhausted
            and not self.config.cycle_short_loaders
        ):
            # Find non-exhausted loaders for redistribution
            non_exhausted = [
                i
                for i in range(len(self.loaders))
                if not self.loader_states[i].exhausted
            ]
            if non_exhausted:
                # Redistribute to non-exhausted loaders
                # (In practice, this would rebuild remaining schedule)
                # For now, skip to next non-exhausted loader
                for next_pos in range(self.schedule_position + 1, len(self.schedule)):
                    next_idx = self.schedule[next_pos]
                    if not self.loader_states[next_idx].exhausted:
                        loader_idx = next_idx
                        self.schedule_position = next_pos
                        break
                else:
                    # No more non-exhausted loaders in schedule
                    raise StopIteration
            else:
                # All loaders exhausted
                raise StopIteration

        # Get batch from loader
        try:
            batch = self._get_next_batch_from_loader(loader_idx)
        except StopIteration:
            # Handle exhaustion without cycling
            if not self.config.cycle_short_loaders:
                # Try to continue with remaining loaders
                remaining_schedule = self.schedule[self.schedule_position + 1 :]
                if remaining_schedule:
                    self.schedule_position += 1
                    return self.__next__()
                raise
            # Should have been handled in _get_next_batch_from_loader
            raise

        # Update position and counters
        self.schedule_position += 1
        self.total_batches += 1
        self.prefetched_batches += 1

        return loader_idx, batch

    def get_loader_states(self) -> list[DataLoaderState]:
        """Get current states of all loaders for checkpointing."""
        # Capture current sampler/dataset states
        for i, loader in enumerate(self.loaders):
            state = self.loader_states[i]

            # Capture sampler state for map-style datasets
            sampler = getattr(loader, "sampler", None)
            batch_sampler = getattr(loader, "batch_sampler", None)
            if sampler is not None and hasattr(sampler, "state_dict"):
                state.sampler_state = sampler.state_dict()
            elif (
                batch_sampler is not None
                and hasattr(batch_sampler, "sampler")
                and hasattr(batch_sampler.sampler, "state_dict")
            ):
                state.sampler_state = batch_sampler.sampler.state_dict()

            # Capture dataset state for iterable datasets
            if isinstance(loader.dataset, CheckpointableIterable):
                state.dataset_state = loader.dataset.state_dict()

        return self.loader_states

    def get_state(self) -> dict[str, Any]:
        """Get complete iterator state for checkpointing."""
        return {
            "schedule_position": self.schedule_position,
            "loader_states": [state.to_dict() for state in self.get_loader_states()],
            "total_batches": self.total_batches,
            "loader_cycles": self.loader_cycles,
            "prefetched_batches": self.prefetched_batches,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load iterator state from checkpoint."""
        self.schedule_position = state["schedule_position"]
        self.total_batches = state["total_batches"]
        self.loader_cycles = state["loader_cycles"]
        self.prefetched_batches = state.get("prefetched_batches", 0)

        # Restore loader states
        self.loader_states = [
            DataLoaderState.from_dict(s) for s in state["loader_states"]
        ]


class DataLoaderManager:
    """
    Manages one or more dataloaders with deterministic scheduling.

    This is the core component of the multi-dataloader-only training engine.
    Even single dataloader scenarios are managed through this class.

    Responsibilities:
    - Building deterministic schedules for each epoch
    - Managing dataloader lifecycle and state
    - Handling DDP synchronization
    - Supporting fault-tolerant resume

    Note: Always provide dataloaders as lists, even for single loader scenarios.
    """

    def __init__(
        self,
        train_loaders: list[DataLoader] | None = None,
        val_loaders: list[DataLoader] | None = None,
        config: MultiDataLoaderConfig | None = None,
        fabric: Any = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize DataLoader manager.

        Args:
            train_loaders: List of training dataloaders
            val_loaders: List of validation dataloaders
            config: Multi-dataloader configuration
            fabric: Optional Lightning Fabric for DDP
            logger: Optional logger
        """
        self.train_loaders = train_loaders or []
        self.val_loaders = val_loaders or []
        self.config = config or MultiDataLoaderConfig()
        self.fabric = fabric
        self.logger = logger or logging.getLogger(__name__)

        # Validate and set names
        self._validate_and_set_names()

        # Initialize choice RNG for weighted sampling
        self.choice_rng_state = ChoiceRNGState(seed=self.config.choice_rng_seed)
        # Always initialize choice_rng for reproducible scheduling
        if self.choice_rng_state.seed is not None:
            self.choice_rng = np.random.RandomState(self.choice_rng_state.seed)
        else:
            self.choice_rng = np.random.RandomState()

        # Store iterator references for state management
        self._train_iterator: MultiDataLoaderIterator | None = None
        self._val_iterator: MultiDataLoaderIterator | None = None
        self._stored_train_state: dict[str, Any] | None = None
        self._stored_val_state: dict[str, Any] | None = None

    def _validate_and_set_names(self) -> None:
        """Validate loaders and set names if not provided."""
        # Set train loader names
        if self.config.dataloader_names:
            if len(self.config.dataloader_names) != len(self.train_loaders):
                raise ValueError(
                    f"Number of dataloader names ({len(self.config.dataloader_names)}) "
                    f"doesn't match number of train loaders ({len(self.train_loaders)})"
                )
            self.train_names = self.config.dataloader_names
        else:
            # Auto-generate names
            self.train_names = [f"dl{i}" for i in range(len(self.train_loaders))]

        # Validate unique names
        if len(set(self.train_names)) != len(self.train_names):
            raise ValueError(f"Dataloader names must be unique: {self.train_names}")

        # Set validation loader names
        self.val_names = [
            f"val_{name}" for name in self.train_names[: len(self.val_loaders)]
        ]

    def build_round_robin_schedule(
        self,
        lengths: list[int],
        policy: EpochLengthPolicy,
        burst_size: int = 1,
        steps_per_epoch: int | None = None,
    ) -> list[int]:
        """
        Build round-robin schedule.

        Args:
            lengths: Length of each dataloader
            policy: Epoch length policy
            burst_size: Number of consecutive batches per loader
            steps_per_epoch: Fixed steps for FIXED_NUM_STEPS policy

        Returns:
            List of loader indices representing the schedule
        """
        n_loaders = len(lengths)
        if n_loaders == 0:
            return []

        # Determine total steps based on policy
        if policy == EpochLengthPolicy.SUM_OF_LENGTHS:
            total_steps = sum(lengths)
        elif policy == EpochLengthPolicy.MAX_OF_LENGTHS:
            total_steps = max(lengths)
        elif policy == EpochLengthPolicy.MIN_OF_LENGTHS:
            total_steps = min(lengths)
        elif policy == EpochLengthPolicy.FIXED_NUM_STEPS:
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch required for FIXED_NUM_STEPS policy")
            total_steps = steps_per_epoch
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Build round-robin schedule with burst size
        schedule: list[int] = []
        loader_idx = 0
        while len(schedule) < total_steps:
            # Add burst_size batches from current loader
            for _ in range(min(burst_size, total_steps - len(schedule))):
                schedule.append(loader_idx)
            loader_idx = (loader_idx + 1) % n_loaders

        return schedule

    def build_sequential_schedule(
        self,
        lengths: list[int],
        policy: EpochLengthPolicy,
        burst_size: int = 1,
        steps_per_epoch: int | None = None,
    ) -> list[int]:
        """
        Build sequential schedule.

        Args:
            lengths: Length of each dataloader
            policy: Epoch length policy
            burst_size: Number of consecutive batches per loader
            steps_per_epoch: Fixed steps for FIXED_NUM_STEPS policy

        Returns:
            List of loader indices representing the schedule
        """
        n_loaders = len(lengths)
        if n_loaders == 0:
            return []

        # Determine total steps based on policy
        if policy == EpochLengthPolicy.SUM_OF_LENGTHS:
            total_steps = sum(lengths)
        elif policy == EpochLengthPolicy.MAX_OF_LENGTHS:
            total_steps = max(lengths) * n_loaders  # Process each loader fully
        elif policy == EpochLengthPolicy.MIN_OF_LENGTHS:
            total_steps = min(lengths) * n_loaders  # Stop when shortest exhausted
        elif policy == EpochLengthPolicy.FIXED_NUM_STEPS:
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch required for FIXED_NUM_STEPS policy")
            total_steps = steps_per_epoch
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Build sequential schedule
        schedule: list[int] = []
        for i in range(n_loaders):
            if policy == EpochLengthPolicy.SUM_OF_LENGTHS:
                # Use actual length for each loader
                loader_steps = lengths[i]
            else:
                # Use equal steps per loader
                loader_steps = total_steps // n_loaders
                if i < total_steps % n_loaders:
                    loader_steps += 1

            for _ in range(loader_steps):
                schedule.append(i)
                if len(schedule) >= total_steps:
                    break

        return schedule[:total_steps]

    def build_weighted_schedule(
        self,
        total_steps: int,
        weights: list[float],
        burst_size: int = 1,
    ) -> list[int]:
        """
        Build weighted schedule using Hamilton's method.

        Uses largest remainder method to allocate integer quotas,
        then balanced_interleave for even distribution.

        Args:
            total_steps: Total number of steps in epoch
            weights: Weights for each dataloader (will be normalized)
            burst_size: Number of consecutive batches per loader

        Returns:
            List of loader indices representing the schedule
        """
        n_loaders = len(weights)
        if n_loaders == 0 or total_steps == 0:
            return []

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

        # Use balanced_interleave to create evenly spaced schedule
        base_schedule = balanced_interleave(integer_quotas)

        # Apply burst size if needed
        if burst_size > 1:
            schedule: list[int] = []
            for idx in base_schedule:
                for _ in range(burst_size):
                    schedule.append(idx)
                    if len(schedule) >= total_steps:
                        break
            return schedule[:total_steps]
        return base_schedule

    def build_alternating_schedule(
        self,
        pattern: list[int],
        total_steps: int,
        burst_size: int = 1,
    ) -> list[int]:
        """
        Build alternating schedule following explicit pattern.

        Args:
            pattern: List of loader indices defining the pattern
            total_steps: Total number of steps in epoch
            burst_size: Number of consecutive batches per loader

        Returns:
            List of loader indices representing the schedule
        """
        if not pattern:
            return []

        schedule: list[int] = []
        pattern_idx = 0

        while len(schedule) < total_steps:
            loader_idx = pattern[pattern_idx]

            # Add burst_size batches from current loader
            for _ in range(min(burst_size, total_steps - len(schedule))):
                schedule.append(loader_idx)

            pattern_idx = (pattern_idx + 1) % len(pattern)

        return schedule

    def create_epoch_iterator(
        self,
        phase: str,
        epoch: int,
        resume_state: list[DataLoaderState] | None = None,
    ) -> MultiDataLoaderIterator:
        """
        Create iterator for an epoch with deterministic schedule.

        Args:
            phase: 'train' or 'val'
            epoch: Current epoch number
            resume_state: Optional loader states for resume

        Returns:
            MultiDataLoaderIterator configured for the epoch
        """
        # Select loaders based on phase
        if phase == "train":
            loaders = self.train_loaders
            names = self.train_names
            # Check for stored state to restore later
            stored_state = self._stored_train_state if resume_state is None else None
        elif phase == "val":
            loaders = self.val_loaders
            names = self.val_names
            # Check for stored state to restore later
            stored_state = self._stored_val_state if resume_state is None else None
        else:
            raise ValueError(f"Unknown phase: {phase}")

        if not loaders:
            raise ValueError(f"No {phase} loaders available")

        # Set epoch for distributed samplers (duck-typed to support mocks)
        for idx, loader in enumerate(loaders):
            sampler = getattr(loader, "sampler", None)
            batch_sampler = getattr(loader, "batch_sampler", None)

            # Check for DistributedSampler
            sampler_set = False
            if sampler is not None:
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                    sampler_set = True
                    self.logger.debug(
                        f"Set epoch {epoch} for DistributedSampler on loader {idx} ({names[idx]})"
                    )
                elif "DistributedSampler" in str(type(sampler)):
                    self.logger.warning(
                        f"Loader {idx} ({names[idx]}) appears to use DistributedSampler "
                        f"but doesn't have set_epoch method. This may cause issues in DDP."
                    )

            # Check batch sampler's inner sampler
            if not sampler_set and batch_sampler is not None:
                inner_sampler = getattr(batch_sampler, "sampler", None)
                if inner_sampler is not None and hasattr(inner_sampler, "set_epoch"):
                    inner_sampler.set_epoch(epoch)
                    self.logger.debug(
                        f"Set epoch {epoch} for DistributedSampler (via batch_sampler) "
                        f"on loader {idx} ({names[idx]})"
                    )

        # Get loader lengths (support infinite loaders explicitly)
        lengths: list[float] = []
        for loader in loaders:
            try:
                lengths.append(float(len(loader)))
            except TypeError:
                # Infinite loader
                lengths.append(math.inf)

        # Build schedule based on strategy
        if self.config.sampling_strategy == SamplingStrategy.ROUND_ROBIN:
            # Disallow infinite loaders unless a fixed number of steps is provided
            if any(math.isinf(L) for L in lengths) and (
                self.config.epoch_length_policy != EpochLengthPolicy.FIXED_NUM_STEPS
            ):
                raise ValueError(
                    "ROUND_ROBIN with infinite loaders requires FIXED_NUM_STEPS; set steps_per_epoch."
                )
            lengths_int = [int(L) for L in lengths]
            schedule = self.build_round_robin_schedule(
                lengths_int,
                self.config.epoch_length_policy,
                self.config.burst_size,
                self.config.steps_per_epoch,
            )
        elif self.config.sampling_strategy == SamplingStrategy.SEQUENTIAL:
            # Disallow infinite loaders unless a fixed number of steps is provided
            if any(math.isinf(L) for L in lengths) and (
                self.config.epoch_length_policy != EpochLengthPolicy.FIXED_NUM_STEPS
            ):
                raise ValueError(
                    "SEQUENTIAL with infinite loaders requires FIXED_NUM_STEPS; set steps_per_epoch."
                )
            lengths_int = [int(L) for L in lengths]
            schedule = self.build_sequential_schedule(
                lengths_int,
                self.config.epoch_length_policy,
                self.config.burst_size,
                self.config.steps_per_epoch,
            )
        elif self.config.sampling_strategy == SamplingStrategy.WEIGHTED:
            if self.config.dataloader_weights is None:
                raise ValueError("Weights required for WEIGHTED strategy")
            if len(self.config.dataloader_weights) != len(loaders):
                raise ValueError(
                    "Number of dataloader_weights must match number of loaders: "
                    f"got {len(self.config.dataloader_weights)} weights for {len(loaders)} loaders"
                )

            # Determine total steps
            if self.config.epoch_length_policy == EpochLengthPolicy.FIXED_NUM_STEPS:
                if self.config.steps_per_epoch is None:
                    raise ValueError("steps_per_epoch required for FIXED_NUM_STEPS")
                total_steps = self.config.steps_per_epoch
            elif self.config.epoch_length_policy == EpochLengthPolicy.SUM_OF_LENGTHS:
                if any(math.isinf(L) for L in lengths):
                    raise ValueError(
                        "SUM_OF_LENGTHS cannot be used with infinite loaders; use FIXED_NUM_STEPS."
                    )
                total_steps = int(sum(lengths))
            elif self.config.epoch_length_policy == EpochLengthPolicy.MAX_OF_LENGTHS:
                finite_lengths = [L for L in lengths if not math.isinf(L)]
                total_steps = int(max(finite_lengths)) if finite_lengths else 0
            elif self.config.epoch_length_policy == EpochLengthPolicy.MIN_OF_LENGTHS:
                if any(math.isinf(L) for L in lengths):
                    raise ValueError(
                        "MIN_OF_LENGTHS requires all loaders to be finite; use FIXED_NUM_STEPS."
                    )
                total_steps = int(min(lengths))
            else:
                raise ValueError(f"Unknown policy: {self.config.epoch_length_policy}")

            schedule = self.build_weighted_schedule(
                total_steps,
                self.config.dataloader_weights,
                self.config.burst_size,
            )
        elif self.config.sampling_strategy == SamplingStrategy.ALTERNATING:
            if self.config.alternating_pattern is None:
                raise ValueError("Pattern required for ALTERNATING strategy")

            # Determine total steps
            if self.config.epoch_length_policy == EpochLengthPolicy.FIXED_NUM_STEPS:
                if self.config.steps_per_epoch is None:
                    raise ValueError("steps_per_epoch required for FIXED_NUM_STEPS")
                total_steps = self.config.steps_per_epoch
            elif self.config.epoch_length_policy == EpochLengthPolicy.SUM_OF_LENGTHS:
                if any(math.isinf(L) for L in lengths):
                    raise ValueError(
                        "SUM_OF_LENGTHS cannot be used with infinite loaders; use FIXED_NUM_STEPS."
                    )
                total_steps = int(sum(lengths))
            else:
                # For alternating, default to sum
                if any(math.isinf(L) for L in lengths):
                    raise ValueError(
                        "ALTERNATING with infinite loaders requires FIXED_NUM_STEPS; set steps_per_epoch."
                    )
                total_steps = int(sum(lengths))

            schedule = self.build_alternating_schedule(
                self.config.alternating_pattern,
                total_steps,
                self.config.burst_size,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.config.sampling_strategy}")

        # Build schedule deterministically or broadcast from rank 0
        if self.fabric is not None:
            # For DDP: Build on rank 0 and broadcast to ensure consistency
            if ddp_is_primary(self.fabric):
                self.logger.debug(
                    f"Rank 0 broadcasting schedule of length {len(schedule)} for {phase} epoch {epoch}"
                )

            # Broadcast the schedule and ensure all ranks get the same one
            schedule = ddp_broadcast_object(self.fabric, schedule, src=0)

            # Also broadcast the choice RNG state for weighted sampling consistency
            if self.config.sampling_strategy == SamplingStrategy.WEIGHTED:
                if ddp_is_primary(self.fabric):
                    choice_rng_state = self.choice_rng.get_state()
                else:
                    choice_rng_state = None
                choice_rng_state = ddp_broadcast_object(
                    self.fabric, choice_rng_state, src=0
                )
                if choice_rng_state is not None:
                    self.choice_rng.set_state(choice_rng_state)

            # Synchronize all ranks after schedule broadcast
            ddp_barrier(self.fabric)

            self.logger.debug(
                f"All ranks synchronized with schedule length {len(schedule)} for {phase} epoch {epoch}"
            )

        # Create iterator
        iterator = MultiDataLoaderIterator(
            loaders=loaders,
            names=names,
            schedule=schedule,
            config=self.config,
            loader_states=resume_state,
            fabric=self.fabric,
            logger=self.logger,
        )

        # Store iterator reference for state management
        if phase == "train":
            self._train_iterator = iterator
            # Restore stored state if available
            if stored_state:
                iterator.load_state(stored_state)
                self._stored_train_state = None  # Clear after use
        else:
            self._val_iterator = iterator
            # Restore stored state if available
            if stored_state:
                iterator.load_state(stored_state)
                self._stored_val_state = None  # Clear after use

        return iterator

    def get_state(self) -> dict[str, Any]:
        """Get complete manager state for checkpointing."""
        state = {
            "choice_rng_state": self.choice_rng_state.to_dict(),
            "choice_rng": self.choice_rng.get_state() if self.choice_rng else None,
        }

        # Include iterator states if they exist
        if self._train_iterator is not None:
            state["train_iterator_state"] = self._train_iterator.get_state()
        elif self._stored_train_state is not None:
            # Include stored state if no active iterator
            state["train_iterator_state"] = self._stored_train_state

        if self._val_iterator is not None:
            state["val_iterator_state"] = self._val_iterator.get_state()
        elif self._stored_val_state is not None:
            # Include stored state if no active iterator
            state["val_iterator_state"] = self._stored_val_state

        return state

    def load_state(self, state: dict[str, Any], skip_broadcast: bool = False) -> None:
        """Load manager state from checkpoint.

        Args:
            state: State dictionary to load
            skip_broadcast: If True, skip broadcasting state (already broadcast by caller)
        """
        # Restore choice RNG state
        if "choice_rng_state" in state:
            self.choice_rng_state = ChoiceRNGState.from_dict(state["choice_rng_state"])

        # Restore choice RNG
        if state.get("choice_rng") is not None:
            # choice_rng exists; update its state
            self.choice_rng.set_state(state["choice_rng"])

        # Store iterator states for later restoration when iterators are created
        self._stored_train_state = state.get("train_iterator_state")
        self._stored_val_state = state.get("val_iterator_state")

        # In DDP, broadcast the loaded state from rank 0 to ensure consistency
        # Skip if already broadcast by caller (e.g., from core.py)
        if self.fabric is not None and not skip_broadcast:
            # Broadcast the entire state from rank 0
            broadcasted_state = ddp_broadcast_object(self.fabric, state, src=0)

            # If not rank 0, update with broadcasted state
            if not ddp_is_primary(self.fabric):
                # Re-apply the broadcasted state
                if "choice_rng_state" in broadcasted_state:
                    self.choice_rng_state = ChoiceRNGState.from_dict(
                        broadcasted_state["choice_rng_state"]
                    )

                if broadcasted_state.get("choice_rng") is not None:
                    self.choice_rng.set_state(broadcasted_state["choice_rng"])

                self._stored_train_state = broadcasted_state.get("train_iterator_state")
                self._stored_val_state = broadcasted_state.get("val_iterator_state")

            # Synchronize all ranks
            ddp_barrier(self.fabric)

            self.logger.debug("DataLoader manager state synchronized across all ranks")
