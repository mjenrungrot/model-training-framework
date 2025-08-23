"""
Core Training Engine

This module provides the GenericTrainer class - the main training engine with:
- Fault-tolerant training with instruction-level checkpointing
- Preemption handling and automatic resume
- Distributed training support via Lightning Fabric
- Comprehensive logging and monitoring
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import torch

from .checkpoints import CheckpointManager
from .states import (
    TrainerPhase,
    TrainMicroState,
    ValMicroState,
    create_initial_resume_state,
    restore_rng_state,
    update_resume_state,
)
from .utils import EarlyStopping, PerformanceMonitor, SignalHandler, ensure_reproducible

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from .config import GenericTrainerConfig

# Type checking imports
import importlib.util

HAS_FABRIC = importlib.util.find_spec("lightning.fabric") is not None

logger = logging.getLogger(__name__)


class TrainingStep(Protocol):
    """Protocol for training step function."""

    def __call__(
        self, trainer: GenericTrainer, batch: Any, micro_step: int
    ) -> dict[str, Any]:
        """Execute training step and return metrics."""
        ...


class ValidationStep(Protocol):
    """Protocol for validation step function."""

    def __call__(
        self, trainer: GenericTrainer, batch: Any, batch_idx: int
    ) -> dict[str, Any]:
        """Execute validation step and return metrics."""
        ...


class GenericTrainer:
    """
    Preemption-safe trainer with instruction-level checkpoint granularity.

    This trainer provides:
    - Fault-tolerant training with SLURM preemption handling
    - Instruction-level resume capability
    - Distributed training support
    - Comprehensive monitoring and logging
    """

    def __init__(
        self,
        config: GenericTrainerConfig,
        fabric: Any,  # Fabric type
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Any | None = None,  # _LRScheduler type
        wandb_run: Any | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Trainer configuration
            fabric: Lightning Fabric instance for distributed training
            model: Model to train
            optimizer: Optimizer for training
            scheduler: Optional learning rate scheduler
            wandb_run: Optional Weights & Biases run for logging
        """
        self.config = config
        self.fabric = fabric
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.wandb_run = wandb_run

        # Initialize components
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint,
            experiment_name=getattr(config, "experiment_name", "experiment"),
        )
        self.signal_handler = SignalHandler()
        self.performance_monitor = PerformanceMonitor()

        # Training state
        self.resume_state = create_initial_resume_state(config.checkpoint.save_rng)
        self.global_step = 0
        self.current_epoch = 0

        # User-defined step functions
        self.training_step_fn: TrainingStep | None = None
        self.validation_step_fn: ValidationStep | None = None

        # Early stopping
        self.early_stopping: EarlyStopping | None = None
        if config.early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                metric_name=config.early_stopping_metric,
                mode=config.early_stopping_mode,
            )

        # Setup signal handling
        self.signal_handler.register_preemption_handler(config.preemption.signal)

        logger.info("Initialized GenericTrainer")

    def set_training_step(self, training_step_fn: TrainingStep) -> None:
        """Set the training step function."""
        self.training_step_fn = training_step_fn

    def set_validation_step(self, validation_step_fn: ValidationStep) -> None:
        """Set the validation step function."""
        self.validation_step_fn = validation_step_fn

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int = 1,
        resume_from_checkpoint: str | None = None,
    ) -> None:
        """
        Main training loop with preemption safety.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            max_epochs: Maximum number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        if self.training_step_fn is None:
            raise ValueError(
                "Training step function not set. Call set_training_step() first."
            )

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)

        # Setup reproducibility if configured
        if hasattr(self.config, "seed") and self.config.seed is not None:
            ensure_reproducible(self.config.seed, deterministic=True)

        logger.info(f"Starting training for {max_epochs} epochs")

        try:
            self._training_loop(train_loader, val_loader, max_epochs)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint(force=True)
        except Exception:
            logger.exception("Training failed with error: ")
            self._save_checkpoint(force=True)
            raise
        finally:
            self.signal_handler.restore_handlers()

        logger.info("Training completed")

    def _training_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        max_epochs: int,
    ) -> None:
        """Main training loop implementation."""

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch

            # Update resume state
            self.resume_state = update_resume_state(
                self.resume_state,
                TrainerPhase.TRAIN_START_EPOCH,
                epoch=epoch,
                global_step=self.global_step,
                save_rng=self.config.checkpoint.save_rng,
            )

            # Check for preemption
            if self.signal_handler.is_preemption_requested():
                logger.info("Preemption requested, saving checkpoint and exiting")
                self._save_checkpoint(force=True)
                return

            # Training epoch
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validation epoch
            val_metrics = {}
            if (
                val_loader is not None
                and epoch % self.config.validate_every_n_epochs == 0
            ):
                val_metrics = self._validation_epoch(val_loader, epoch)

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Log epoch metrics
            self._log_epoch_metrics(epoch, epoch_metrics)

            # Scheduler step (epoch-based)
            if self.scheduler is not None:
                self.resume_state = update_resume_state(
                    self.resume_state,
                    TrainerPhase.SCHEDULER_EPOCH_STEP,
                    save_rng=self.config.checkpoint.save_rng,
                )
                self.scheduler.step()

            # Save checkpoint if needed
            if self.checkpoint_manager.should_save_checkpoint(epoch, self.global_step):
                self._save_checkpoint(epoch_metrics)

            # Early stopping check
            if self.early_stopping is not None and self.early_stopping(epoch_metrics):
                logger.info("Early stopping triggered")
                break

            # Update resume state
            self.resume_state = update_resume_state(
                self.resume_state,
                TrainerPhase.EPOCH_END,
                save_rng=self.config.checkpoint.save_rng,
            )

        # Final checkpoint
        self.resume_state = update_resume_state(
            self.resume_state,
            TrainerPhase.TRAINING_COMPLETE,
            save_rng=self.config.checkpoint.save_rng,
        )
        self._save_checkpoint(force=True)

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, Any]:
        """Execute one training epoch."""
        self.model.train()

        epoch_loss = 0.0
        epoch_metrics = {}
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Check for preemption
            if self.signal_handler.is_preemption_requested():
                logger.info("Preemption requested during training")
                self._save_checkpoint(force=True)
                return {"train_loss": epoch_loss / max(num_batches, 1)}

            # Update resume state
            train_state = TrainMicroState(batch_idx=batch_idx, micro_step=0)
            self.resume_state = update_resume_state(
                self.resume_state,
                TrainerPhase.TRAIN_BATCH_LOAD,
                train_state=train_state,
                save_rng=self.config.checkpoint.save_rng,
            )

            # Training step
            step_metrics = self._training_step(batch, batch_idx)

            # Update metrics
            if "loss" in step_metrics:
                epoch_loss += float(step_metrics["loss"])

            # Accumulate other metrics
            for key, value in step_metrics.items():
                if key != "loss" and isinstance(value, (int, float)):
                    epoch_metrics[f"train_{key}"] = (
                        epoch_metrics.get(f"train_{key}", 0) + value
                    )

            num_batches += 1

            # Log step metrics
            if (
                self.config.log_loss_every_n_steps is not None
                and self.global_step % self.config.log_loss_every_n_steps == 0
            ):
                self._log_step_metrics(step_metrics, self.global_step, "train")

        # Average metrics
        final_metrics = {"train_loss": epoch_loss / max(num_batches, 1)}
        for key, value in epoch_metrics.items():
            final_metrics[key] = value / max(num_batches, 1)

        return final_metrics

    def _validation_epoch(self, val_loader: DataLoader, epoch: int) -> dict[str, Any]:
        """Execute one validation epoch."""
        if self.validation_step_fn is None:
            return {}

        self.model.eval()

        epoch_loss = 0.0
        epoch_metrics = {}
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Update resume state
                val_state = ValMicroState(batch_idx=batch_idx)
                self.resume_state = update_resume_state(
                    self.resume_state,
                    TrainerPhase.VAL_BATCH_LOAD,
                    val_state=val_state,
                    save_rng=self.config.checkpoint.save_rng,
                )

                # Validation step
                step_metrics = self.validation_step_fn(self, batch, batch_idx)

                # Update metrics
                if "loss" in step_metrics:
                    epoch_loss += float(step_metrics["loss"])

                # Accumulate other metrics
                for key, value in step_metrics.items():
                    if key != "loss" and isinstance(value, (int, float)):
                        epoch_metrics[f"val_{key}"] = (
                            epoch_metrics.get(f"val_{key}", 0) + value
                        )

                num_batches += 1

        # Average metrics
        final_metrics = {"val_loss": epoch_loss / max(num_batches, 1)}
        for key, value in epoch_metrics.items():
            final_metrics[key] = value / max(num_batches, 1)

        return final_metrics

    def _training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Execute single training step with gradient accumulation."""

        # Forward pass
        self.resume_state = update_resume_state(
            self.resume_state,
            TrainerPhase.TRAIN_BATCH_FORWARD,
            save_rng=self.config.checkpoint.save_rng,
        )

        step_metrics = self.training_step_fn(
            self, batch, 0
        )  # micro_step = 0 for simplicity
        loss = step_metrics["loss"]

        # Backward pass
        self.resume_state = update_resume_state(
            self.resume_state,
            TrainerPhase.TRAIN_BATCH_BACKWARD,
            save_rng=self.config.checkpoint.save_rng,
        )

        loss = loss / self.config.performance.gradient_accumulation_steps
        self.fabric.backward(loss)

        # Optimizer step (simplified - no gradient accumulation for now)
        self.resume_state = update_resume_state(
            self.resume_state,
            TrainerPhase.TRAIN_BATCH_OPTIM_STEP,
            save_rng=self.config.checkpoint.save_rng,
        )

        # Gradient clipping
        if self.config.performance.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.performance.clip_grad_norm
            )

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1

        # Scheduler step (batch-based)
        if self.scheduler is not None:
            self.resume_state = update_resume_state(
                self.resume_state,
                TrainerPhase.SCHEDULER_BATCH_STEP,
                save_rng=self.config.checkpoint.save_rng,
            )
            self.scheduler.step()

        # Record performance
        self.performance_monitor.record_step_time()
        self.performance_monitor.record_memory_usage()

        return step_metrics

    def _save_checkpoint(
        self, metrics: dict[str, Any] | None = None, force: bool = False
    ) -> None:
        """Save checkpoint with timeout handling."""
        try:
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                resume_state=self.resume_state,
                epoch=self.current_epoch,
                global_step=self.global_step,
                metrics=metrics,
                timeout_seconds=self.config.preemption.max_checkpoint_sec
                if not force
                else None,
            )
        except Exception:
            logger.exception("Failed to save checkpoint: ")
            if force:
                raise

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        try:
            epoch, global_step, resume_state = (
                self.checkpoint_manager.restore_from_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    checkpoint_path=checkpoint_path,
                )
            )

            self.current_epoch = epoch
            self.global_step = global_step

            if resume_state is not None:
                self.resume_state = resume_state

                # Restore RNG state if available
                if resume_state.rng is not None:
                    restore_rng_state(resume_state.rng)

            logger.info(f"Resumed from checkpoint: epoch={epoch}, step={global_step}")

        except Exception:
            logger.exception("Failed to resume from checkpoint: ")
            raise

    def _log_step_metrics(
        self, metrics: dict[str, Any], step: int, prefix: str = ""
    ) -> None:
        """Log step-level metrics."""
        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, torch.Tensor)):
                log_key = f"{prefix}_{key}" if prefix else key
                log_dict[log_key] = float(value)

        if self.wandb_run is not None:
            self.wandb_run.log(log_dict, step=step)

        # Console logging
        loss_str = f"loss={log_dict.get(f'{prefix}_loss', 'N/A'):.4f}"
        logger.info(f"Step {step}: {loss_str}")

    def _log_epoch_metrics(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Log epoch-level metrics."""
        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, torch.Tensor)):
                log_dict[key] = float(value)

        if self.wandb_run is not None:
            self.wandb_run.log(log_dict, step=self.global_step)

        # Console logging
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in log_dict.items()])
        logger.info(f"Epoch {epoch}: {metrics_str}")

    def get_training_state(self) -> dict[str, Any]:
        """Get current training state."""
        return {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "resume_state": self.resume_state,
            "performance_summary": self.performance_monitor.get_performance_summary(),
        }
