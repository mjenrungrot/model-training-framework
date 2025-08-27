"""
Dataset validation utilities for fault-tolerant training.

This module provides validation for IterableDataset checkpointing support.
"""

from torch.utils.data import DataLoader, IterableDataset


def validate_iterable_dataset_checkpointing(
    loaders: list[DataLoader],
    require_checkpointing: bool = False,
) -> None:
    """
    Validate that IterableDatasets support checkpointing if required.

    Args:
        loaders: List of DataLoaders to validate
        require_checkpointing: If True, raise error for non-checkpointable IterableDatasets

    Raises:
        ValueError: If an IterableDataset doesn't support checkpointing when required
    """
    if not require_checkpointing:
        return

    for i, loader in enumerate(loaders):
        dataset = loader.dataset

        # Check if it's an IterableDataset
        if isinstance(dataset, IterableDataset):
            # Check for checkpointing support
            has_state_dict = hasattr(dataset, "state_dict") and callable(
                getattr(dataset, "state_dict", None)
            )
            has_load_state_dict = hasattr(dataset, "load_state_dict") and callable(
                getattr(dataset, "load_state_dict", None)
            )

            if not (has_state_dict and has_load_state_dict):
                raise ValueError(
                    f"IterableDataset at loader index {i} does not support checkpointing. "
                    f"To use fault tolerance with IterableDatasets, your dataset class must implement:\n"
                    f"  1. state_dict() -> dict - Returns current iteration state\n"
                    f"  2. load_state_dict(state: dict) -> None - Restores iteration state\n"
                    f"\nAlternatively, you can:\n"
                    f"  - Use a map-style Dataset instead of IterableDataset\n"
                    f"  - Disable dataset state saving: FaultToleranceConfig(save_dataset_state=False)\n"
                    f"  - Implement the CheckpointableIterable protocol in your dataset"
                )


def detect_iterable_dataset_type(dataset) -> str:
    """
    Detect the type of dataset and its checkpointing capabilities.

    Args:
        dataset: Dataset to check

    Returns:
        One of: "map", "iterable", "iterable_checkpointable"
    """
    if not isinstance(dataset, IterableDataset):
        return "map"

    # Check for checkpointing support
    has_state_dict = hasattr(dataset, "state_dict") and callable(
        getattr(dataset, "state_dict", None)
    )
    has_load_state_dict = hasattr(dataset, "load_state_dict") and callable(
        getattr(dataset, "load_state_dict", None)
    )

    if has_state_dict and has_load_state_dict:
        return "iterable_checkpointable"

    return "iterable"


def estimate_iterable_dataset_length(dataset) -> int | None:
    """
    Try to estimate the length of an IterableDataset.

    Args:
        dataset: IterableDataset to estimate

    Returns:
        Estimated length if available, None otherwise
    """
    # First try __len__ if available
    if hasattr(dataset, "__len__"):
        try:
            return len(dataset)
        except (TypeError, NotImplementedError):
            pass

    # Check for custom length hint
    if hasattr(dataset, "length_hint"):
        length_hint = dataset.length_hint
        if isinstance(length_hint, int):
            return length_hint
        return None

    # No length information available
    return None
