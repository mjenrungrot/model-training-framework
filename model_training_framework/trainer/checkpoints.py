"""
Checkpoint Management

This module handles all checkpoint-related operations:
- Saving and loading model checkpoints
- Managing checkpoint storage and cleanup
- Creating symlinks for latest checkpoints
- Monitoring metrics for best checkpoint selection
"""

from __future__ import annotations

import logging
from pathlib import Path
import random
import shutil
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from .config import CheckpointConfig
    from .core import GenericTrainer
    from .states import ResumeState

from .utils import timeout

logger = logging.getLogger(__name__)


def save_checkpoint(path: Path, trainer: GenericTrainer) -> None:
    """
    Save checkpoint in format version 1.

    Args:
        path: Path to save checkpoint
        trainer: GenericTrainer instance to save state from
    """
    dmanager = getattr(trainer, "dataloader_manager", None)
    checkpoint_data: dict[str, Any] = {
        # Format metadata
        "format_version": 1,
        "is_multi_dataloader_only": True,
        "save_timestamp": time.time(),
        # Basic training state
        "epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        # Model state
        "model_state_dict": trainer.model.state_dict(),
        # Multi-optimizer support
        "optimizer_state_dicts": [opt.state_dict() for opt in trainer.optimizers],
        # Multi-scheduler support
        "scheduler_state_dicts": [sched.state_dict() for sched in trainer.schedulers]
        if trainer.schedulers
        else [],
        # AMP scaler state
        "amp_scaler_state": trainer.scaler.state_dict()
        if hasattr(trainer, "scaler") and trainer.scaler
        else None,
        # RNG states
        "rng_states": {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),  # noqa: NPY002
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [
                torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else None,
        },
        # DataLoaderManager state
        "dataloader_manager_state": (
            dmanager.get_state() if dmanager is not None else None
        ),
        # Optional explicit choice RNG state (for weighted sampling reproducibility)
        "choice_rng_state": (
            dmanager.choice_rng.get_state()
            if (
                dmanager is not None
                and getattr(dmanager, "choice_rng", None) is not None
            )
            else None
        ),
        # Resume state
        "resume_state": trainer.resume_state,
        # Optional: metrics history
        "metrics_history": getattr(trainer, "metrics_history", None),
        # Optional: config snapshot
        "config_snapshot": trainer.config if hasattr(trainer, "config") else None,
    }

    # Save checkpoint
    torch.save(checkpoint_data, path)
    logger.info(f"Saved checkpoint format v1 to {path}")


def load_checkpoint(path: Path, trainer: GenericTrainer) -> None:
    """
    Load checkpoint in format version 1.

    Args:
        path: Path to load checkpoint from
        trainer: GenericTrainer instance to restore state to
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint_data = torch.load(
        path,
        map_location=trainer.device if hasattr(trainer, "device") else None,
        weights_only=False,
    )

    # Validate format version
    if checkpoint_data.get("format_version") != 1:
        raise ValueError(
            f"Unsupported checkpoint format version: {checkpoint_data.get('format_version')}"
        )

    # Restore basic training state
    trainer.current_epoch = checkpoint_data["epoch"]
    trainer.global_step = checkpoint_data["global_step"]

    # Restore model state
    trainer.model.load_state_dict(checkpoint_data["model_state_dict"])

    # Restore all optimizer states
    optimizer_states = checkpoint_data.get("optimizer_state_dicts", [])
    for i, opt_state in enumerate(optimizer_states):
        if i < len(trainer.optimizers):
            trainer.optimizers[i].load_state_dict(opt_state)

    # Restore all scheduler states
    scheduler_states = checkpoint_data.get("scheduler_state_dicts", [])
    if trainer.schedulers:
        for i, sched_state in enumerate(scheduler_states):
            if i < len(trainer.schedulers):
                trainer.schedulers[i].load_state_dict(sched_state)

    # Restore AMP scaler state
    if (
        checkpoint_data.get("amp_scaler_state")
        and hasattr(trainer, "scaler")
        and trainer.scaler
    ):
        trainer.scaler.load_state_dict(checkpoint_data["amp_scaler_state"])

    # Restore RNG states
    rng_states = checkpoint_data.get("rng_states", {})
    if rng_states:
        if "python_random" in rng_states:
            random.setstate(rng_states["python_random"])
        if "numpy_random" in rng_states:
            np.random.set_state(rng_states["numpy_random"])  # noqa: NPY002
        if "torch_cpu" in rng_states:
            torch.set_rng_state(rng_states["torch_cpu"])
        if rng_states.get("torch_cuda") and torch.cuda.is_available():
            for i, cuda_state in enumerate(rng_states["torch_cuda"]):
                if i < torch.cuda.device_count():
                    torch.cuda.set_rng_state(cuda_state, i)

    # Restore DataLoaderManager state
    dmanager = getattr(trainer, "dataloader_manager", None)
    if checkpoint_data.get("dataloader_manager_state") and dmanager is not None:
        dmanager.load_state(checkpoint_data["dataloader_manager_state"])
        # Optionally restore explicit choice RNG state
        if checkpoint_data.get("choice_rng_state") is not None:
            try:
                rng_state = checkpoint_data["choice_rng_state"]
                if getattr(dmanager, "choice_rng", None) is not None:
                    dmanager.choice_rng.set_state(rng_state)
            except Exception:
                logger.debug(
                    "Failed to restore explicit choice RNG state", exc_info=True
                )

    # Restore resume state
    if "resume_state" in checkpoint_data:
        trainer.resume_state = checkpoint_data["resume_state"]

    # Restore optional fields
    if "metrics_history" in checkpoint_data:
        from typing import cast as _cast

        _trainer_any = _cast(Any, trainer)
        _trainer_any.metrics_history = checkpoint_data["metrics_history"]

    logger.info(
        f"Loaded checkpoint format v1 from {path} (epoch={trainer.current_epoch}, step={trainer.global_step})"
    )


class CheckpointManager:
    """Manages model checkpoint saving, loading, and cleanup."""

    def __init__(self, config: CheckpointConfig, experiment_name: str = "experiment"):
        """
        Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration
            experiment_name: Name of the experiment (used for directory naming)
        """
        self.config = config
        self.experiment_name = experiment_name

        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.root_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Paths for special checkpoint files
        self.latest_path = self.checkpoint_dir / "latest.ckpt"
        self.best_path = self.checkpoint_dir / "best.ckpt"

        # Track checkpoint history and metrics
        self.checkpoint_history: list[Path] = []
        self.best_metric_value: float | None = None
        self.last_save_time = time.time()

        logger.info(
            f"Initialized CheckpointManager for {experiment_name} at {self.checkpoint_dir}"
        )

    def should_save_checkpoint(
        self, epoch: int, global_step: int, force: bool = False
    ) -> bool:
        """
        Determine if a checkpoint should be saved based on configuration.

        Args:
            epoch: Current epoch number
            global_step: Current global step
            force: Force checkpoint save regardless of schedule

        Returns:
            True if checkpoint should be saved
        """
        if force:
            return True

        # Check step-based saving
        if (
            self.config.save_every_n_steps is not None
            and global_step > 0
            and global_step % self.config.save_every_n_steps == 0
        ):
            return True

        # Check epoch-based saving
        if (
            self.config.save_every_n_epochs is not None
            and epoch > 0
            and epoch % self.config.save_every_n_epochs == 0
        ):
            return True

        # Check time-based saving
        if self.config.save_every_n_minutes is not None:
            time_since_last_save = (time.time() - self.last_save_time) / 60.0
            if time_since_last_save >= self.config.save_every_n_minutes:
                return True

        return False

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer | None = None,
        schedulers: list[Any] | Any | None = None,
        resume_state: ResumeState | None = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: dict[str, float] | None = None,
        timeout_seconds: float | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> Path:
        """
        Save a checkpoint with model, optimizers, and training state.

        Args:
            model: Model to save
            optimizers: Optimizer(s) to save (single or list)
            schedulers: Learning rate scheduler(s) to save (single or list)
            resume_state: Training resume state
            epoch: Current epoch
            global_step: Current global step
            metrics: Current metrics for best checkpoint tracking
            timeout_seconds: Timeout for save operation
            scaler: AMP GradScaler for mixed precision training

        Returns:
            Path to saved checkpoint file

        Raises:
            TimeoutError: If save operation exceeds timeout
        """
        # Handle backward compatibility: convert single optimizer/scheduler to list
        if optimizers is not None and not isinstance(optimizers, list):
            optimizers = [optimizers]
        if schedulers is not None and not isinstance(schedulers, list):
            schedulers = [schedulers] if schedulers else []
        # Generate checkpoint filename
        checkpoint_filename = self.config.filename_template.format(
            epoch=epoch, step=global_step, timestamp=int(time.time())
        )
        checkpoint_path = self.checkpoint_dir / checkpoint_filename

        # Prepare checkpoint data with format v1
        checkpoint_data = {
            "format_version": 1,
            "is_multi_dataloader_only": True,
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "experiment_name": self.experiment_name,
            "save_timestamp": time.time(),
            "config": self.config,
        }

        # Add optimizer states if requested (multi-optimizer support)
        if self.config.save_optimizer and optimizers:
            checkpoint_data["optimizer_state_dicts"] = [
                opt.state_dict() for opt in optimizers
            ]

        # Add scheduler states if requested and available (multi-scheduler support)
        if self.config.save_scheduler and schedulers:
            checkpoint_data["scheduler_state_dicts"] = [
                sched.state_dict() for sched in schedulers
            ]

        # Add RNG states if requested
        if self.config.save_rng:
            checkpoint_data["rng_states"] = {
                "python_random": random.getstate(),
                "numpy_random": np.random.get_state(),  # noqa: NPY002
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": [
                    torch.cuda.get_rng_state(i)
                    for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else None,
            }

        # Add AMP scaler state if provided
        if scaler is not None:
            checkpoint_data["amp_scaler_state"] = scaler.state_dict()

        # Add resume state if provided
        if resume_state is not None:
            checkpoint_data["resume_state"] = resume_state

        # Add metrics if provided
        if metrics is not None:
            checkpoint_data["metrics"] = metrics

        # Save checkpoint with optional timeout
        def _save_checkpoint():
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        if timeout_seconds is not None:
            with timeout(timeout_seconds):
                _save_checkpoint()
        else:
            _save_checkpoint()

        # Update checkpoint tracking
        self.checkpoint_history.append(checkpoint_path)
        self.last_save_time = time.time()

        # Update latest symlink
        self._update_latest_symlink(checkpoint_path)

        # Check if this is the best checkpoint
        if self.config.save_best and metrics is not None:
            self._maybe_update_best_checkpoint(checkpoint_path, metrics)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path | None = None,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """
        Load checkpoint data from file.

        Args:
            checkpoint_path: Path to checkpoint file (uses latest if None)
            map_location: Device to map tensors to

        Returns:
            Dictionary containing checkpoint data

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No checkpoints found in {self.checkpoint_dir}"
                )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        return cast(
            dict[str, Any],
            torch.load(checkpoint_path, map_location=map_location, weights_only=False),
        )

    def restore_from_checkpoint(
        self,
        model: torch.nn.Module,
        optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer | None = None,
        schedulers: list[Any] | Any | None = None,
        checkpoint_path: str | Path | None = None,
        strict: bool = True,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> tuple[int, int, ResumeState | None]:
        """
        Restore model, optimizers, and schedulers from checkpoint.

        Args:
            model: Model to restore state to
            optimizers: Optimizer(s) to restore state to (single or list)
            schedulers: Scheduler(s) to restore state to (single or list)
            checkpoint_path: Path to checkpoint (uses latest if None)
            strict: Whether to strictly enforce state dict matching
            scaler: AMP GradScaler to restore state to

        Returns:
            Tuple of (epoch, global_step, resume_state)
        """
        checkpoint_data = self.load_checkpoint(checkpoint_path)

        # Handle backward compatibility: convert single optimizer/scheduler to list
        if optimizers is not None and not isinstance(optimizers, list):
            optimizers = [optimizers]
        if schedulers is not None and not isinstance(schedulers, list):
            schedulers = [schedulers] if schedulers else []

        # Restore model state
        model.load_state_dict(checkpoint_data["model_state_dict"], strict=strict)

        # Restore optimizer states (multi-optimizer support)
        if (
            "optimizer_state_dicts" in checkpoint_data
            and self.config.save_optimizer
            and optimizers
        ):
            for i, opt_state in enumerate(checkpoint_data["optimizer_state_dicts"]):
                if i < len(optimizers):
                    optimizers[i].load_state_dict(opt_state)
        # Fallback for old format
        elif (
            "optimizer_state_dict" in checkpoint_data
            and self.config.save_optimizer
            and optimizers
        ):
            optimizers[0].load_state_dict(checkpoint_data["optimizer_state_dict"])

        # Restore scheduler states (multi-scheduler support)
        if (
            "scheduler_state_dicts" in checkpoint_data
            and self.config.save_scheduler
            and schedulers
        ):
            for i, sched_state in enumerate(checkpoint_data["scheduler_state_dicts"]):
                if i < len(schedulers):
                    schedulers[i].load_state_dict(sched_state)
        # Fallback for old format
        elif (
            "scheduler_state_dict" in checkpoint_data
            and self.config.save_scheduler
            and schedulers
        ):
            schedulers[0].load_state_dict(checkpoint_data["scheduler_state_dict"])

        # Restore RNG states if saved and requested
        if "rng_states" in checkpoint_data and self.config.save_rng:
            rng_states = checkpoint_data["rng_states"]
            if "python_random" in rng_states:
                random.setstate(rng_states["python_random"])
            if "numpy_random" in rng_states:
                np.random.set_state(rng_states["numpy_random"])  # noqa: NPY002
            if "torch_cpu" in rng_states:
                torch.set_rng_state(rng_states["torch_cpu"])
            if rng_states.get("torch_cuda") and torch.cuda.is_available():
                for i, cuda_state in enumerate(rng_states["torch_cuda"]):
                    if i < torch.cuda.device_count():
                        torch.cuda.set_rng_state(cuda_state, i)

        # Restore AMP scaler state if available
        if "amp_scaler_state" in checkpoint_data and scaler is not None:
            scaler.load_state_dict(checkpoint_data["amp_scaler_state"])

        # Extract training state
        epoch = checkpoint_data.get("epoch", 0)
        global_step = checkpoint_data.get("global_step", 0)
        resume_state = checkpoint_data.get("resume_state")

        logger.info(f"Restored checkpoint from epoch {epoch}, step {global_step}")

        return epoch, global_step, resume_state

    def list_checkpoints(self) -> list[Path]:
        """List all available checkpoint files."""
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []
        for path in self.checkpoint_dir.glob("*.ckpt"):
            if path.name not in ["latest.ckpt", "best.ckpt"]:
                checkpoints.append(path)

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return checkpoints

    def get_latest_checkpoint(self) -> Path | None:
        """Get path to latest checkpoint."""
        return self._find_latest_checkpoint()

    def get_best_checkpoint(self) -> Path | None:
        """Get path to best checkpoint."""
        if self.best_path.exists() and self.best_path.is_symlink():
            return self.best_path.resolve()
        return None

    def cleanup_all_checkpoints(self) -> None:
        """Remove all checkpoint files."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        logger.info(f"Removed all checkpoints from {self.checkpoint_dir}")

    def _find_latest_checkpoint(self) -> Path | None:
        """Find the most recent checkpoint."""
        # First try the latest symlink
        if self.latest_path.exists() and self.latest_path.is_symlink():
            target = self.latest_path.resolve()
            if target.exists():
                return target

        # Fall back to finding the newest checkpoint file
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None

    def _update_latest_symlink(self, checkpoint_path: Path) -> None:
        """Update the latest.ckpt symlink to point to the newest checkpoint."""
        try:
            # Remove existing symlink
            if self.latest_path.exists() or self.latest_path.is_symlink():
                self.latest_path.unlink()

            # Create new symlink
            self.latest_path.symlink_to(checkpoint_path.name)
            logger.debug(f"Updated latest symlink to {checkpoint_path.name}")

        except OSError as e:
            logger.warning(f"Failed to update latest symlink: {e}")
            # Fallback: copy file for platforms without symlink support (e.g., Windows)
            try:
                shutil.copy2(checkpoint_path, self.latest_path)
                logger.debug(
                    f"Copied latest checkpoint to {self.latest_path} as fallback"
                )
            except Exception as copy_err:
                logger.warning(f"Failed to copy latest checkpoint fallback: {copy_err}")

    def _maybe_update_best_checkpoint(
        self, checkpoint_path: Path, metrics: dict[str, float]
    ) -> None:
        """Update best checkpoint if current metrics are better."""
        if self.config.monitor_metric is None:
            return

        if self.config.monitor_metric not in metrics:
            logger.warning(
                f"Monitor metric '{self.config.monitor_metric}' not found in metrics"
            )
            return

        current_value = metrics[self.config.monitor_metric]

        # Check if this is the best value so far
        is_best = False
        if self.best_metric_value is None:
            is_best = True
        elif self.config.monitor_mode == "min":
            is_best = current_value < self.best_metric_value
        elif self.config.monitor_mode == "max":
            is_best = current_value > self.best_metric_value

        if is_best:
            try:
                # Remove existing best symlink
                if self.best_path.exists() or self.best_path.is_symlink():
                    self.best_path.unlink()

                # Create new best symlink
                self.best_path.symlink_to(checkpoint_path.name)
                self.best_metric_value = current_value

                logger.info(
                    f"New best checkpoint: {checkpoint_path.name} "
                    f"({self.config.monitor_metric}={current_value:.6f})"
                )

            except OSError as e:
                logger.warning(f"Failed to update best checkpoint symlink: {e}")
                # Fallback: copy file
                try:
                    shutil.copy2(checkpoint_path, self.best_path)
                    self.best_metric_value = current_value
                    logger.info(
                        f"Copied best checkpoint to {self.best_path} as fallback"
                    )
                except Exception as copy_err:
                    logger.warning(
                        f"Failed to copy best checkpoint fallback: {copy_err}"
                    )

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond the maximum count."""
        if self.config.max_checkpoints <= 0:
            return

        checkpoints = self.list_checkpoints()

        # Keep only the most recent max_checkpoints
        if len(checkpoints) > self.config.max_checkpoints:
            checkpoints_to_remove = checkpoints[self.config.max_checkpoints :]

            for checkpoint_path in checkpoints_to_remove:
                try:
                    checkpoint_path.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint_path}")
                    if checkpoint_path in self.checkpoint_history:
                        self.checkpoint_history.remove(checkpoint_path)
                except OSError as e:
                    logger.warning(
                        f"Failed to remove checkpoint {checkpoint_path}: {e}"
                    )

    def get_storage_info(self) -> dict[str, Any]:
        """Get information about checkpoint storage usage."""
        if not self.checkpoint_dir.exists():
            return {"total_size_mb": 0, "num_checkpoints": 0, "checkpoints": []}

        checkpoints = self.list_checkpoints()
        checkpoint_info = []
        total_size = 0

        for checkpoint_path in checkpoints:
            try:
                size = checkpoint_path.stat().st_size
                total_size += size
                checkpoint_info.append(
                    {
                        "path": str(checkpoint_path),
                        "size_mb": size / (1024 * 1024),
                        "modified": checkpoint_path.stat().st_mtime,
                    }
                )
            except OSError:
                continue

        return {
            "total_size_mb": total_size / (1024 * 1024),
            "num_checkpoints": len(checkpoints),
            "checkpoints": checkpoint_info,
            "latest_checkpoint": str(self.get_latest_checkpoint())
            if self.get_latest_checkpoint()
            else None,
            "best_checkpoint": str(self.get_best_checkpoint())
            if self.get_best_checkpoint()
            else None,
        }
