"""
Checkpoint Management

This module handles all checkpoint-related operations:
- Saving and loading model checkpoints
- Managing checkpoint storage and cleanup
- Creating symlinks for latest checkpoints
- Monitoring metrics for best checkpoint selection
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from .states import update_resume_state  # unify resume embedding
from .utils import timeout

logger = logging.getLogger(__name__)

# Runtime reference to ResumeState for isinstance/attribute access in functions
# Use a module import to avoid assigning to a type alias
_ResumeStateRuntime: Any | None = None
try:  # Guard to avoid hard import errors in unusual import orders
    from . import states as _states_mod

    _ResumeStateRuntime = _states_mod.ResumeState
except Exception:  # pragma: no cover - defensive fallback
    _ResumeStateRuntime = None


@dataclass
class CheckpointPayload:
    """
    Serializable checkpoint payload (format v1).

    This dataclass defines exactly what we save in a checkpoint. The to_dict()
    method produces the stable on-disk format used by both the free
    save_checkpoint() helper and CheckpointManager.save_checkpoint().
    """

    # Format metadata
    format_version: int
    save_timestamp: float

    # Basic training state
    epoch: int
    global_step: int

    # Model and optimizer/scheduler state
    model_state_dict: dict[str, Any]
    optimizer_state_dicts: list[dict[str, Any]] | None = None
    scheduler_state_dicts: list[dict[str, Any]] | None = None

    # AMP / RNG
    amp_scaler_state: dict[str, Any] | None = None
    rng_states: dict[str, Any] | None = None

    # Multi-dataloader state and resume
    dataloader_manager_state: dict[str, Any] | None = None
    choice_rng_state: Any | None = None
    resume_state: Any | None = None

    # Optional extras
    metrics: dict[str, float] | None = None
    metrics_history: Any | None = None
    config_snapshot: Any | None = None
    # Manager-provided context
    experiment_name: str | None = None
    config: Any | None = None  # CheckpointConfig (kept as Any for serialization)

    def to_dict(self) -> dict[str, Any]:
        # Preserve None entries to keep stable keys in saved files
        data = asdict(self)
        try:
            if _ResumeStateRuntime is not None and isinstance(
                self.resume_state, _ResumeStateRuntime
            ):
                data["resume_state"] = self.resume_state.to_dict()
        except Exception:
            logger.debug(
                "Failed to serialize resume_state in checkpoint payload", exc_info=True
            )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointPayload:
        """Reconstruct payload from a raw dict, handling legacy fields when present."""
        # Legacy single optimizer/scheduler support
        opt_states = data.get("optimizer_state_dicts")
        if opt_states is None and "optimizer_state_dict" in data:
            opt_states = [data["optimizer_state_dict"]]

        sched_states = data.get("scheduler_state_dicts")
        if sched_states is None and "scheduler_state_dict" in data:
            sched_states = [data["scheduler_state_dict"]]

        return cls(
            format_version=int(data.get("format_version", 1)),
            save_timestamp=float(data.get("save_timestamp", time.time())),
            epoch=int(data.get("epoch", 0)),
            global_step=int(data.get("global_step", 0)),
            model_state_dict=data.get("model_state_dict", {}),
            optimizer_state_dicts=opt_states,
            scheduler_state_dicts=sched_states,
            amp_scaler_state=data.get("amp_scaler_state"),
            rng_states=data.get("rng_states"),
            dataloader_manager_state=data.get("dataloader_manager_state"),
            choice_rng_state=data.get("choice_rng_state"),
            resume_state=data.get("resume_state"),
            metrics=data.get("metrics"),
            metrics_history=data.get("metrics_history"),
            config_snapshot=data.get("config_snapshot"),
            experiment_name=data.get("experiment_name"),
            config=data.get("config"),
        )


def _capture_rng_states_dict() -> dict[str, Any]:
    """Capture RNG states in a plain dict for serialization."""
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),  # noqa: NPY002
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [
            torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else None,
    }


def build_checkpoint_payload(
    *,
    trainer: GenericTrainer,
    include_optimizer: bool = True,
    include_scheduler: bool = True,
    include_rng: bool = True,
    include_dataloader_state: bool = False,
    include_choice_rng: bool = False,
    include_metrics_history: bool = True,
    include_config_snapshot: bool = False,
    experiment_name: str | None = None,
    config: CheckpointConfig | None = None,
    metrics: dict[str, float] | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> CheckpointPayload:
    return CheckpointPayload(
        format_version=1,
        save_timestamp=time.time(),
        epoch=trainer.current_epoch,
        global_step=trainer.global_step,
        model_state_dict=trainer.model.state_dict(),
        optimizer_state_dicts=[opt.state_dict() for opt in trainer.optimizers]
        if include_optimizer
        else None,
        scheduler_state_dicts=(
            [sched.state_dict() for sched in trainer.schedulers]
            if trainer.schedulers
            else []
        )
        if include_scheduler
        else None,
        amp_scaler_state=(scaler.state_dict() if scaler is not None else None),
        rng_states=_capture_rng_states_dict() if include_rng else None,
        dataloader_manager_state=None,  # Always use resume_state as the single source
        choice_rng_state=None,  # Always use resume_state as the single source
        resume_state=getattr(trainer, "resume_state", None),
        metrics=metrics,
        metrics_history=(
            getattr(trainer, "metrics_history", None)
            if include_metrics_history
            else None
        ),
        config_snapshot=(
            trainer.config
            if include_config_snapshot and hasattr(trainer, "config")
            else None
        ),
        experiment_name=experiment_name,
        config=config,
    )


def save_checkpoint(path: Path, trainer: GenericTrainer) -> None:
    """Compatibility helper that saves a checkpoint to an explicit path using the unified payload builder.

    Ensures the trainer.resume_state embeds the latest dataloader manager state and choice RNG so that
    only resume_state is needed to restore multi-loader iteration state.
    """
    # Refresh resume_state with latest dataloader manager/choice RNG
    try:
        dmanager = getattr(trainer, "dataloader_manager", None)
        current = getattr(trainer, "resume_state", None)
        if current is not None and dmanager is not None:
            trainer.resume_state = update_resume_state(
                current_state=current,
                phase=current.phase,
                dataloader_manager_state=dmanager.get_state(),
                save_rng=True,
                choice_rng=getattr(dmanager, "choice_rng", None),
            )
    except Exception:
        logger.debug("Could not refresh resume_state before save", exc_info=True)

    payload = build_checkpoint_payload(
        trainer=trainer,
        include_optimizer=True,
        include_scheduler=True,
        include_rng=True,
        include_metrics_history=True,
        include_config_snapshot=True,
        metrics=None,
        scaler=getattr(trainer, "scaler", None),
    )
    torch.save(payload.to_dict(), path)
    logger.info(f"Saved checkpoint format v1 to {path}")


def load_checkpoint(path: Path, trainer: GenericTrainer) -> CheckpointPayload:
    """
    Load checkpoint in format version 1.

    Args:
        path: Path to load checkpoint from
        trainer: GenericTrainer instance to restore state to
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Map to CPU to avoid device-coupled payloads and satisfy static typing
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)

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

    # Restore resume state (convert dict to dataclass if needed)
    if "resume_state" in checkpoint_data:
        rs = checkpoint_data["resume_state"]
        try:
            if isinstance(rs, dict) and _ResumeStateRuntime is not None:
                trainer.resume_state = _ResumeStateRuntime.from_dict(rs)
            else:
                trainer.resume_state = rs
        except Exception:
            trainer.resume_state = rs

    # Restore optional fields
    if "metrics_history" in checkpoint_data:
        _trainer_any = cast("Any", trainer)
        _trainer_any.metrics_history = checkpoint_data["metrics_history"]

    logger.info(
        f"Loaded checkpoint format v1 from {path} (epoch={trainer.current_epoch}, step={trainer.global_step})"
    )

    # Return typed payload for compatibility with CheckpointManager.load_checkpoint
    return CheckpointPayload.from_dict(checkpoint_data)


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

        # Check epoch-based saving (epochs are 0-based internally)
        if (
            self.config.save_every_n_epochs is not None
            and (epoch + 1) % self.config.save_every_n_epochs == 0
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

        # Build unified payload
        include_optimizer = self.config.save_optimizer and bool(optimizers)
        include_scheduler = self.config.save_scheduler and bool(schedulers)
        include_rng = self.config.save_rng

        # Create a temporary trainer-like shim for the builder
        class _Shim:
            current_epoch: int
            global_step: int
            model: torch.nn.Module
            optimizers: list[torch.optim.Optimizer]
            schedulers: list[Any]
            resume_state: ResumeState | None
            dataloader_manager: Any | None
            config: CheckpointConfig
            scaler: torch.cuda.amp.GradScaler | None

        shim = _Shim()
        shim.current_epoch = epoch
        shim.global_step = global_step
        shim.model = model
        shim.optimizers = optimizers or []
        shim.schedulers = schedulers or []
        shim.resume_state = resume_state
        # Best effort: capture dataloader manager via attribute if caller provided a real trainer later
        # For manager-based saving, dataloader manager state is typically embedded via trainer._save_checkpoint
        # so we leave it None here.
        shim.dataloader_manager = None
        shim.config = self.config
        shim.scaler = scaler

        payload = build_checkpoint_payload(
            trainer=cast("GenericTrainer", shim),
            include_optimizer=include_optimizer,
            include_scheduler=include_scheduler,
            include_rng=include_rng,
            include_dataloader_state=False,
            include_choice_rng=False,
            include_metrics_history=False,
            include_config_snapshot=False,
            experiment_name=self.experiment_name,
            config=self.config,
            metrics=metrics,
            scaler=scaler,
        )

        # Save checkpoint with optional timeout
        def _save_checkpoint():
            torch.save(payload.to_dict(), checkpoint_path)
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
    ) -> CheckpointPayload:
        """
        Load checkpoint data from file.

        Args:
            checkpoint_path: Path to checkpoint file (uses latest if None)
            map_location: Device to map tensors to

        Returns:
            CheckpointPayload containing checkpoint data

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
        raw = cast(
            "dict[str, Any]",
            torch.load(checkpoint_path, map_location=map_location, weights_only=False),
        )
        return CheckpointPayload.from_dict(raw)

    def is_framework_checkpoint(self, path: str | Path) -> bool:
        """Detect whether a file is a framework checkpoint (format v1).

        Returns True if the file can be loaded and contains required keys
        like 'format_version' and 'model_state_dict'. Returns False for
        corrupted files or unknown formats.
        """
        try:
            p = Path(path)
            if not p.exists() or not p.is_file():
                return False
            raw = cast(
                "dict[str, Any]", torch.load(p, map_location="cpu", weights_only=False)
            )
            return (
                isinstance(raw, dict)
                and "format_version" in raw
                and "model_state_dict" in raw
            )
        except Exception:
            return False

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
        payload = self.load_checkpoint(checkpoint_path)

        # Handle backward compatibility: convert single optimizer/scheduler to list
        if optimizers is not None and not isinstance(optimizers, list):
            optimizers = [optimizers]
        if schedulers is not None and not isinstance(schedulers, list):
            schedulers = [schedulers] if schedulers else []

        # Restore model state
        model.load_state_dict(payload.model_state_dict, strict=strict)

        # Restore optimizer states (multi-optimizer support)
        if self.config.save_optimizer and optimizers and payload.optimizer_state_dicts:
            for i, opt_state in enumerate(payload.optimizer_state_dicts):
                if i < len(optimizers):
                    optimizers[i].load_state_dict(opt_state)

        # Restore scheduler states (multi-scheduler support)
        if self.config.save_scheduler and schedulers and payload.scheduler_state_dicts:
            for i, sched_state in enumerate(payload.scheduler_state_dicts):
                if i < len(schedulers):
                    schedulers[i].load_state_dict(sched_state)

        # Restore RNG states if saved and requested
        if payload.rng_states and self.config.save_rng:
            rng_states = payload.rng_states
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
        if scaler is not None and payload.amp_scaler_state is not None:
            scaler.load_state_dict(payload.amp_scaler_state)

        # Extract training state
        epoch = payload.epoch
        global_step = payload.global_step
        resume_state = payload.resume_state

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
