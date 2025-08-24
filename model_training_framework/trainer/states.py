"""
Training States

This module defines all state classes for the training engine:
- TrainerPhase enum for instruction-level checkpointing
- RNG state management for deterministic resume
- Training and validation micro-states
- Complete resume state management
"""

from __future__ import annotations

from dataclasses import dataclass
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


# Single-dataloader micro states removed - will be replaced with multi-dataloader versions


@dataclass
class ResumeState:
    """
    Complete snapshot enabling deterministic resume after SIGUSR1 preemption.

    This is the core checkpoint format that captures the trainer's exact state
    at any instruction boundary. The 'phase' field acts as an instruction pointer,
    while the optional train/val/rng fields provide phase-specific context.

    Fields are populated based on the current phase:
    - train: Only populated during TRAIN_* phases
    - val: Only populated during VAL_* phases
    - rng: Always populated when save_rng=True in config
    """

    phase: TrainerPhase  # Current execution phase (instruction pointer)
    epoch: int  # Current epoch number (0-based)
    global_step: int  # Total optimizer steps taken across all epochs
    version: str = "v2.0"  # Checkpoint format version for migration
    train: Any | None = (
        None  # Training state (if in training phase) - will be MultiTrainMicroState
    )
    val: Any | None = (
        None  # Validation state (if in validation phase) - will be MultiValMicroState
    )
    rng: RNGState | None = None  # RNG states for deterministic resume
    timestamp: float = 0.0  # Unix timestamp when checkpoint was created


def capture_rng_state() -> RNGState:
    """
    Capture complete RNG state from all libraries.

    Returns:
        RNGState containing serialized states from PyTorch, NumPy, and Python
    """
    # Capture PyTorch state (including CUDA if available)
    torch_state_dict = {
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


def create_initial_resume_state(save_rng: bool = True) -> ResumeState:
    """
    Create initial resume state at the beginning of training.

    Args:
        save_rng: Whether to capture RNG state

    Returns:
        Initial ResumeState
    """
    return ResumeState(
        phase=TrainerPhase.INIT,
        epoch=0,
        global_step=0,
        rng=capture_rng_state() if save_rng else None,
        timestamp=time.time(),
    )


def update_resume_state(
    current_state: ResumeState,
    phase: TrainerPhase,
    epoch: int | None = None,
    global_step: int | None = None,
    train_state: Any | None = None,  # Will be MultiTrainMicroState
    val_state: Any | None = None,  # Will be MultiValMicroState
    save_rng: bool = True,
) -> ResumeState:
    """
    Update resume state with new phase and context information.

    Args:
        current_state: Current resume state
        phase: New trainer phase
        epoch: New epoch number (if changed)
        global_step: New global step (if changed)
        train_state: Training micro-state (for training phases)
        val_state: Validation micro-state (for validation phases)
        save_rng: Whether to update RNG state

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
        train=train_state,
        val=val_state,
        rng=capture_rng_state() if save_rng else current_state.rng,
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
