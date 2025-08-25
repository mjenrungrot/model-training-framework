"""
Training States

This module defines all state classes for the training engine:
- TrainerPhase enum for instruction-level checkpointing
- RNG state management for deterministic resume
- Training and validation micro-states
- Complete resume state management
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import pickle
import random
import time
from typing import Any

import numpy as np
import torch


class TrainerPhase(Enum):
    """Every **instruction-level** location at which the trainer can legally receive SIGUSR1 and later resume **deterministically**."""

    # --- Setup ----------------------------------------------------------------
    INIT = "INIT"  # Before first training step

    # --- Training micro-steps -------------------------------------------------
    TRAIN_START_EPOCH = "TRAIN_START_EPOCH"
    TRAIN_BATCH_LOAD = "TRAIN_BATCH_LOAD"
    TRAIN_BATCH_FORWARD = "TRAIN_BATCH_FORWARD"
    TRAIN_BATCH_BACKWARD = "TRAIN_BATCH_BACKWARD"
    TRAIN_BATCH_OPTIM_STEP = "TRAIN_BATCH_OPTIM_STEP"
    TRAIN_BATCH_END = "TRAIN_BATCH_END"

    # --- Validation micro-steps ----------------------------------------------
    VAL_START_EPOCH = "VAL_START_EPOCH"
    VAL_BATCH_LOAD = "VAL_BATCH_LOAD"
    VAL_BATCH_FORWARD = "VAL_BATCH_FORWARD"
    VAL_BATCH_END = "VAL_BATCH_END"

    # --- Scheduler interactions ----------------------------------------------
    SCHEDULER_EPOCH_STEP = "SCHEDULER_EPOCH_STEP"
    SCHEDULER_BATCH_STEP = "SCHEDULER_BATCH_STEP"

    # --- Checkpoint I/O --------------------------------------------------------
    CHECKPOINT_SAVE_START = "CHECKPOINT_SAVE_START"
    CHECKPOINT_SAVE_END = "CHECKPOINT_SAVE_END"

    # --- Clean / terminal ------------------------------------------------------
    EPOCH_END = "EPOCH_END"
    TRAINING_COMPLETE = "TRAINING_COMPLETE"

    # --- Error handling --------------------------------------------------------
    ERROR_RECOVERABLE = "ERROR_RECOVERABLE"
    ERROR_FATAL = "ERROR_FATAL"


@dataclass
class RNGState:
    """
    Portable container for PyTorch, NumPy, and Python RNG seeds/states.

    Captures the complete random number generator state across all libraries
    to ensure perfect deterministic resume after preemption. Saved at every
    micro-step boundary to guarantee bit-for-bit reproducibility.
    """

    torch_state: bytes  # Serialized torch.get_rng_state() including CUDA states
    numpy_state: bytes  # Serialized numpy.random.get_state()
    python_state: tuple[int, ...]  # Python's random.getstate() tuple


@dataclass
class DataLoaderState:
    """
    State for a single dataloader in multi-dataloader training.

    Tracks the iteration state of an individual dataloader, including
    position, exhaustion status, and checkpointable sampler/dataset state.
    """

    id: int  # Unique ID for this loader
    name: str  # Human-readable name for logging
    batch_idx: int  # Current batch index within this loader
    exhausted: bool  # Whether this loader has been exhausted
    sampler_state: dict[str, Any] | None = (
        None  # Sampler state for deterministic resume
    )
    dataset_state: dict[str, Any] | None = None  # Dataset state for IterableDatasets

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataLoaderState:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ChoiceRNGState:
    """
    RNG state for deterministic weighted sampling.

    Captures the state of the random number generator used for
    weighted sampling strategies to ensure reproducible scheduling.
    """

    seed: int | None = None  # Initial seed if provided
    state: Any | None = None  # Current numpy RandomState or Generator state

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data: dict[str, Any] = {"seed": self.seed, "state": None}
        if self.state is not None:
            # Pickle the numpy RNG state
            data["state"] = pickle.dumps(self.state).hex()
        else:
            data["state"] = None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChoiceRNGState:
        """Create from dictionary."""
        seed = data.get("seed")
        state = None
        if data.get("state") is not None:
            state = pickle.loads(bytes.fromhex(data["state"]))
        return cls(seed=seed, state=state)


@dataclass
class MultiTrainMicroState:
    """
    Training micro-state for multi-dataloader training.

    Captures the complete training iteration state across multiple dataloaders,
    including the active loader, schedule position, and individual loader states.
    """

    active_loader_id: int  # Currently active dataloader ID
    micro_step: int  # Micro-step within current batch (for grad accumulation)
    loader_states: list[DataLoaderState]  # State for each dataloader
    schedule_position: int  # Position in the deterministic schedule
    total_steps_completed: int  # Total steps completed across all loaders

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "active_loader_id": self.active_loader_id,
            "micro_step": self.micro_step,
            "loader_states": [state.to_dict() for state in self.loader_states],
            "schedule_position": self.schedule_position,
            "total_steps_completed": self.total_steps_completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiTrainMicroState:
        """Create from dictionary."""
        return cls(
            active_loader_id=data["active_loader_id"],
            micro_step=data["micro_step"],
            loader_states=[DataLoaderState.from_dict(s) for s in data["loader_states"]],
            schedule_position=data["schedule_position"],
            total_steps_completed=data["total_steps_completed"],
        )


@dataclass
class MultiValMicroState:
    """
    Validation micro-state for multi-dataloader training.

    Captures the validation iteration state across multiple validation dataloaders,
    including accumulated metrics for proper aggregation.
    """

    active_loader_id: int  # Currently active validation loader ID
    loader_states: list[DataLoaderState]  # State for each validation loader
    accumulated_metrics: dict[str, Any]  # Accumulated metrics per loader

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "active_loader_id": self.active_loader_id,
            "loader_states": [state.to_dict() for state in self.loader_states],
            "accumulated_metrics": self.accumulated_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiValMicroState:
        """Create from dictionary."""
        return cls(
            active_loader_id=data["active_loader_id"],
            loader_states=[DataLoaderState.from_dict(s) for s in data["loader_states"]],
            accumulated_metrics=data["accumulated_metrics"],
        )


@dataclass
class ResumeState:
    """
    Complete snapshot enabling deterministic resume after SIGUSR1 preemption.

    This is the core checkpoint format that captures the trainer's exact state
    at any instruction boundary. The 'phase' field acts as an instruction pointer,
    while the optional multi_train/multi_val/rng fields provide phase-specific context.

    Fields are populated based on the current phase:
    - multi_train: Only populated during TRAIN_* phases
    - multi_val: Only populated during VAL_* phases
    - dataloader_manager_state: State of the DataLoaderManager
    - rng: Always populated when save_rng=True in config
    - choice_rng: RNG state for weighted sampling
    """

    phase: TrainerPhase  # Current execution phase (instruction pointer)
    epoch: int  # Current epoch number (0-based)
    global_step: int  # Total optimizer steps taken across all epochs
    version: int = 1  # Checkpoint format version for migration
    multi_train: MultiTrainMicroState | None = (
        None  # Training state for multi-dataloader
    )
    multi_val: MultiValMicroState | None = None  # Validation state for multi-dataloader
    dataloader_manager_state: dict[str, Any] | None = None  # DataLoaderManager state
    rng: RNGState | None = None  # RNG states for deterministic resume
    choice_rng: ChoiceRNGState | None = None  # RNG state for weighted sampling
    timestamp: float = 0.0  # Unix timestamp when checkpoint was created

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data: dict[str, Any] = {
            "phase": self.phase.value,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "version": self.version,
            "timestamp": self.timestamp,
        }

        if self.multi_train is not None:
            data["multi_train"] = self.multi_train.to_dict()
        else:
            data["multi_train"] = None

        if self.multi_val is not None:
            data["multi_val"] = self.multi_val.to_dict()
        else:
            data["multi_val"] = None

        data["dataloader_manager_state"] = self.dataloader_manager_state

        if self.rng is not None:
            data["rng"] = {
                "torch_state": self.rng.torch_state.hex(),
                "numpy_state": self.rng.numpy_state.hex(),
                "python_state": self.rng.python_state,
            }
        else:
            data["rng"] = None

        if self.choice_rng is not None:
            data["choice_rng"] = self.choice_rng.to_dict()
        else:
            data["choice_rng"] = None

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResumeState:
        """Create from dictionary."""
        phase = TrainerPhase(data["phase"])

        multi_train = None
        if data.get("multi_train") is not None:
            multi_train = MultiTrainMicroState.from_dict(data["multi_train"])

        multi_val = None
        if data.get("multi_val") is not None:
            multi_val = MultiValMicroState.from_dict(data["multi_val"])

        rng = None
        if data.get("rng") is not None:
            rng = RNGState(
                torch_state=bytes.fromhex(data["rng"]["torch_state"]),
                numpy_state=bytes.fromhex(data["rng"]["numpy_state"]),
                python_state=tuple(data["rng"]["python_state"]),
            )

        choice_rng = None
        if data.get("choice_rng") is not None:
            choice_rng = ChoiceRNGState.from_dict(data["choice_rng"])

        return cls(
            phase=phase,
            epoch=data["epoch"],
            global_step=data["global_step"],
            version=data.get("version", 1),
            multi_train=multi_train,
            multi_val=multi_val,
            dataloader_manager_state=data.get("dataloader_manager_state"),
            rng=rng,
            choice_rng=choice_rng,
            timestamp=data.get("timestamp", 0.0),
        )


def capture_rng_state() -> RNGState:
    """
    Capture complete RNG state from all libraries.

    Returns:
        RNGState containing serialized states from PyTorch, NumPy, and Python
    """
    # Capture PyTorch state (including CUDA if available)
    torch_state_dict: dict[str, Any] = {
        "cpu": torch.get_rng_state(),
    }

    # Add CUDA states if available
    if torch.cuda.is_available():
        torch_state_dict["cuda_states"] = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch_state_dict["cuda_states"].append(torch.cuda.get_rng_state())

    torch_state_bytes = pickle.dumps(torch_state_dict)

    # Capture NumPy state (using legacy API for compatibility)
    numpy_state_bytes = pickle.dumps(np.random.get_state())  # noqa: NPY002

    # Capture Python random state
    python_state = random.getstate()

    return RNGState(
        torch_state=torch_state_bytes,
        numpy_state=numpy_state_bytes,
        python_state=python_state,
    )


def restore_rng_state(rng_state: RNGState) -> None:
    """
    Restore complete RNG state to all libraries.

    Args:
        rng_state: Previously captured RNG state
    """
    # Restore PyTorch state
    torch_state_dict = pickle.loads(rng_state.torch_state)
    torch.set_rng_state(torch_state_dict["cpu"])

    # Restore CUDA states if they exist
    if "cuda_states" in torch_state_dict and torch.cuda.is_available():
        for i, cuda_state in enumerate(torch_state_dict["cuda_states"]):
            if i < torch.cuda.device_count():
                with torch.cuda.device(i):
                    torch.cuda.set_rng_state(cuda_state)

    # Restore NumPy state (using legacy API for compatibility)
    numpy_state = pickle.loads(rng_state.numpy_state)
    np.random.set_state(numpy_state)  # noqa: NPY002

    # Restore Python random state
    random.setstate(rng_state.python_state)


def capture_choice_rng_state(rng: Any | None) -> ChoiceRNGState | None:
    """
    Capture the state of a numpy random number generator.

    Args:
        rng: numpy.random.RandomState or numpy.random.Generator instance

    Returns:
        ChoiceRNGState if rng is provided, None otherwise
    """
    if rng is None:
        return None

    # Get the state based on the type of RNG
    if hasattr(rng, "get_state"):  # RandomState
        state = rng.get_state()
    elif hasattr(rng, "bit_generator"):  # Generator
        state = rng.bit_generator.state
    else:
        state = None

    return ChoiceRNGState(seed=None, state=state)


def restore_choice_rng_state(
    choice_state: ChoiceRNGState | None, rng: Any | None
) -> None:
    """
    Restore the state of a numpy random number generator.

    Args:
        choice_state: Previously captured ChoiceRNGState
        rng: numpy.random.RandomState or Generator to restore state to
    """
    if choice_state is None or rng is None or choice_state.state is None:
        return

    # Restore the state based on the type of RNG
    if hasattr(rng, "set_state"):  # RandomState
        rng.set_state(choice_state.state)
    elif hasattr(rng, "bit_generator"):  # Generator
        rng.bit_generator.state = choice_state.state


def create_initial_resume_state(
    save_rng: bool = True, choice_rng: Any | None = None
) -> ResumeState:
    """
    Create initial resume state at the beginning of training.

    Args:
        save_rng: Whether to capture RNG state
        choice_rng: Optional numpy RNG for weighted sampling

    Returns:
        Initial ResumeState
    """
    return ResumeState(
        phase=TrainerPhase.INIT,
        epoch=0,
        global_step=0,
        version=1,
        multi_train=None,
        multi_val=None,
        dataloader_manager_state=None,
        rng=capture_rng_state() if save_rng else None,
        choice_rng=capture_choice_rng_state(choice_rng) if save_rng else None,
        timestamp=time.time(),
    )


def update_resume_state(
    current_state: ResumeState,
    phase: TrainerPhase,
    epoch: int | None = None,
    global_step: int | None = None,
    multi_train: MultiTrainMicroState | None = None,
    multi_val: MultiValMicroState | None = None,
    dataloader_manager_state: dict[str, Any] | None = None,
    save_rng: bool = True,
    choice_rng: Any | None = None,
) -> ResumeState:
    """
    Update resume state with new phase and context information.

    Args:
        current_state: Current resume state
        phase: New trainer phase
        epoch: New epoch number (if changed)
        global_step: New global step (if changed)
        multi_train: Multi-dataloader training micro-state (for training phases)
        multi_val: Multi-dataloader validation micro-state (for validation phases)
        dataloader_manager_state: DataLoaderManager state dict
        save_rng: Whether to update RNG state
        choice_rng: Optional numpy RNG for weighted sampling

    Returns:
        Updated ResumeState
    """
    return ResumeState(
        phase=phase,
        epoch=epoch if epoch is not None else current_state.epoch,
        global_step=global_step
        if global_step is not None
        else current_state.global_step,
        version=current_state.version,
        multi_train=multi_train,
        multi_val=multi_val,
        dataloader_manager_state=dataloader_manager_state
        if dataloader_manager_state is not None
        else current_state.dataloader_manager_state,
        rng=capture_rng_state() if save_rng else current_state.rng,
        choice_rng=capture_choice_rng_state(choice_rng)
        if save_rng and choice_rng is not None
        else current_state.choice_rng,
        timestamp=time.time(),
    )


def is_training_phase(phase: TrainerPhase) -> bool:
    """Check if phase is a training-related phase."""
    training_phases = {
        TrainerPhase.TRAIN_START_EPOCH,
        TrainerPhase.TRAIN_BATCH_LOAD,
        TrainerPhase.TRAIN_BATCH_FORWARD,
        TrainerPhase.TRAIN_BATCH_BACKWARD,
        TrainerPhase.TRAIN_BATCH_OPTIM_STEP,
        TrainerPhase.TRAIN_BATCH_END,
        TrainerPhase.SCHEDULER_BATCH_STEP,
    }
    return phase in training_phases


def is_validation_phase(phase: TrainerPhase) -> bool:
    """Check if phase is a validation-related phase."""
    validation_phases = {
        TrainerPhase.VAL_START_EPOCH,
        TrainerPhase.VAL_BATCH_LOAD,
        TrainerPhase.VAL_BATCH_FORWARD,
        TrainerPhase.VAL_BATCH_END,
    }
    return phase in validation_phases


def is_scheduler_phase(phase: TrainerPhase) -> bool:
    """Check if phase is a scheduler-related phase."""
    scheduler_phases = {
        TrainerPhase.SCHEDULER_EPOCH_STEP,
        TrainerPhase.SCHEDULER_BATCH_STEP,
    }
    return phase in scheduler_phases


def is_checkpoint_phase(phase: TrainerPhase) -> bool:
    """Check if phase is a checkpoint-related phase."""
    checkpoint_phases = {
        TrainerPhase.CHECKPOINT_SAVE_START,
        TrainerPhase.CHECKPOINT_SAVE_END,
    }
    return phase in checkpoint_phases


def is_terminal_phase(phase: TrainerPhase) -> bool:
    """Check if phase indicates training completion or error."""
    terminal_phases = {
        TrainerPhase.TRAINING_COMPLETE,
        TrainerPhase.ERROR_FATAL,
    }
    return phase in terminal_phases


def validate_phase_transition(from_phase: TrainerPhase, to_phase: TrainerPhase) -> bool:
    """
    Validate that a phase transition is legal.

    Args:
        from_phase: Current phase
        to_phase: Target phase

    Returns:
        True if transition is valid
    """
    # Define valid transitions (simplified - could be more comprehensive)
    valid_transitions = {
        TrainerPhase.INIT: {TrainerPhase.TRAIN_START_EPOCH},
        TrainerPhase.TRAIN_START_EPOCH: {
            TrainerPhase.TRAIN_BATCH_LOAD,
            TrainerPhase.VAL_START_EPOCH,
            TrainerPhase.EPOCH_END,
        },
        TrainerPhase.TRAIN_BATCH_LOAD: {TrainerPhase.TRAIN_BATCH_FORWARD},
        TrainerPhase.TRAIN_BATCH_FORWARD: {TrainerPhase.TRAIN_BATCH_BACKWARD},
        TrainerPhase.TRAIN_BATCH_BACKWARD: {TrainerPhase.TRAIN_BATCH_OPTIM_STEP},
        TrainerPhase.TRAIN_BATCH_OPTIM_STEP: {
            TrainerPhase.SCHEDULER_BATCH_STEP,
            TrainerPhase.TRAIN_BATCH_END,
        },
        TrainerPhase.SCHEDULER_BATCH_STEP: {TrainerPhase.TRAIN_BATCH_END},
        TrainerPhase.TRAIN_BATCH_END: {
            TrainerPhase.TRAIN_BATCH_LOAD,
            TrainerPhase.VAL_START_EPOCH,
            TrainerPhase.SCHEDULER_EPOCH_STEP,
        },
        TrainerPhase.VAL_START_EPOCH: {
            TrainerPhase.VAL_BATCH_LOAD,
            TrainerPhase.EPOCH_END,
        },
        TrainerPhase.VAL_BATCH_LOAD: {TrainerPhase.VAL_BATCH_FORWARD},
        TrainerPhase.VAL_BATCH_FORWARD: {TrainerPhase.VAL_BATCH_END},
        TrainerPhase.VAL_BATCH_END: {
            TrainerPhase.VAL_BATCH_LOAD,
            TrainerPhase.EPOCH_END,
        },
        TrainerPhase.SCHEDULER_EPOCH_STEP: {TrainerPhase.EPOCH_END},
        TrainerPhase.EPOCH_END: {
            TrainerPhase.TRAIN_START_EPOCH,
            TrainerPhase.TRAINING_COMPLETE,
        },
        TrainerPhase.CHECKPOINT_SAVE_START: {TrainerPhase.CHECKPOINT_SAVE_END},
        TrainerPhase.CHECKPOINT_SAVE_END: {
            TrainerPhase.TRAIN_START_EPOCH,
            TrainerPhase.TRAINING_COMPLETE,
        },
        TrainerPhase.TRAINING_COMPLETE: set(),  # Terminal state
        TrainerPhase.ERROR_RECOVERABLE: {
            TrainerPhase.TRAIN_START_EPOCH,
            TrainerPhase.VAL_START_EPOCH,
        },
        TrainerPhase.ERROR_FATAL: set(),  # Terminal state
    }

    return to_phase in valid_transitions.get(from_phase, set())


def get_phase_description(phase: TrainerPhase) -> str:
    """Get human-readable description of a training phase."""
    descriptions = {
        TrainerPhase.INIT: "Initializing training",
        TrainerPhase.TRAIN_START_EPOCH: "Starting training epoch",
        TrainerPhase.TRAIN_BATCH_LOAD: "Loading training batch",
        TrainerPhase.TRAIN_BATCH_FORWARD: "Forward pass",
        TrainerPhase.TRAIN_BATCH_BACKWARD: "Backward pass",
        TrainerPhase.TRAIN_BATCH_OPTIM_STEP: "Optimizer step",
        TrainerPhase.TRAIN_BATCH_END: "Training batch complete",
        TrainerPhase.VAL_START_EPOCH: "Starting validation epoch",
        TrainerPhase.VAL_BATCH_LOAD: "Loading validation batch",
        TrainerPhase.VAL_BATCH_FORWARD: "Validation forward pass",
        TrainerPhase.VAL_BATCH_END: "Validation batch complete",
        TrainerPhase.SCHEDULER_EPOCH_STEP: "Scheduler epoch step",
        TrainerPhase.SCHEDULER_BATCH_STEP: "Scheduler batch step",
        TrainerPhase.CHECKPOINT_SAVE_START: "Starting checkpoint save",
        TrainerPhase.CHECKPOINT_SAVE_END: "Checkpoint save complete",
        TrainerPhase.EPOCH_END: "Epoch complete",
        TrainerPhase.TRAINING_COMPLETE: "Training complete",
        TrainerPhase.ERROR_RECOVERABLE: "Recoverable error occurred",
        TrainerPhase.ERROR_FATAL: "Fatal error occurred",
    }
    return descriptions.get(phase, f"Unknown phase: {phase}")
