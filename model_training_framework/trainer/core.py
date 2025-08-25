"""
Core Training Engine

This module provides the GenericTrainer class - the main training engine with:
- Multi-dataloader training with deterministic scheduling
- Fault-tolerant training with instruction-level checkpointing
- Preemption handling and automatic resume
- Distributed training support via Lightning Fabric
- Per-loader optimization and metrics tracking
- Comprehensive logging and monitoring
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch.amp import autocast

from .checkpoints import CheckpointManager
from .config import (
    GenericTrainerConfig,
    ValAggregation,
    ValidationFrequency,
    validate_trainer_config,
)
from .multi_dataloader import DataLoaderManager
from .states import (
    MultiTrainMicroState,
    MultiValMicroState,
    TrainerPhase,
    create_initial_resume_state,
    restore_rng_state,
    update_resume_state,
)
from .utils import (
    EarlyStopping,
    PerformanceMonitor,
    SignalHandler,
    seed_all,
)

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TrainingStep(Protocol):
    """Protocol for training step function."""

    def __call__(
        self,
        trainer: GenericTrainer,
        batch: Any,
        dataloader_idx: int,
        dataloader_name: str,
    ) -> dict[str, Any]:
        """
        Execute training step and return metrics.

        Args:
            trainer: The trainer instance
            batch: Input batch
            dataloader_idx: Index of the current dataloader
            dataloader_name: Name of the current dataloader

        Returns:
            Dictionary with at least 'loss' key
        """
        ...


class ValidationStep(Protocol):
    """Protocol for validation step function."""

    def __call__(
        self,
        trainer: GenericTrainer,
        batch: Any,
        dataloader_idx: int,
        dataloader_name: str,
    ) -> dict[str, Any]:
        """
        Execute validation step and return metrics.

        Args:
            trainer: The trainer instance
            batch: Input batch
            dataloader_idx: Index of the current dataloader
            dataloader_name: Name of the current dataloader

        Returns:
            Dictionary with metrics (typically including 'loss')
        """
        ...


class GenericTrainer:
    """
    Multi-dataloader trainer with preemption safety and fault tolerance.

    This trainer provides:
    - Multi-dataloader training with deterministic scheduling
    - Per-loader optimizer routing and loss weighting
    - Fault-tolerant training with SLURM preemption handling
    - Instruction-level resume capability
    - Distributed training support via Lightning Fabric
    - Comprehensive monitoring and logging
    """

    def __init__(
        self,
        config: GenericTrainerConfig,
        model: Module,
        optimizers: list[Optimizer],
        schedulers: list[_LRScheduler] | None = None,
        fabric: Any = None,  # Lightning Fabric
        wandb_run: Any = None,
    ):
        """
        Initialize the multi-dataloader trainer.

        Args:
            config: Trainer configuration
            model: Model to train
            optimizers: List of optimizers (can be one or multiple)
            schedulers: Optional list of learning rate schedulers
            fabric: Optional Lightning Fabric instance for distributed training
            wandb_run: Optional Weights & Biases run for logging
        """
        self.config = config
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers or []
        self.fabric = fabric
        self.wandb_run = wandb_run

        # Validate configuration
        validate_trainer_config(config)

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

        # DataLoader manager (initialized in fit())
        self.dataloader_manager: DataLoaderManager | None = None

        # AMP scaler for mixed precision training
        self.scaler: torch.cuda.amp.GradScaler | None = None
        if config.performance.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()

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

        # Validation step counter
        self.steps_since_validation = 0

        logger.info(f"Initialized GenericTrainer with {len(optimizers)} optimizer(s)")

    def set_training_step(self, training_step_fn: TrainingStep) -> None:
        """Set the training step function with multi-dataloader signature."""
        self.training_step_fn = training_step_fn

    def set_validation_step(self, validation_step_fn: ValidationStep) -> None:
        """Set the validation step function with multi-dataloader signature."""
        self.validation_step_fn = validation_step_fn

    def fit(
        self,
        train_loaders: list[DataLoader],
        val_loaders: list[DataLoader] | None = None,
        max_epochs: int = 1,
        resume_from_checkpoint: str | None = None,
    ) -> None:
        """
        Main training loop with multi-dataloader support.

        Args:
            train_loaders: List of training dataloaders
            val_loaders: Optional list of validation dataloaders
            max_epochs: Maximum number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        if self.training_step_fn is None:
            raise ValueError(
                "Training step function not set. Call set_training_step() first."
            )

        if not train_loaders:
            raise ValueError("At least one training dataloader required")

        # Initialize DataLoader manager
        self.dataloader_manager = DataLoaderManager(
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            config=self.config.multi,
            fabric=self.fabric,
            logger=logger,
        )

        # Validate optimizer configuration
        if self.config.per_loader_optimizer_id is not None:
            max_optimizer_id = max(self.config.per_loader_optimizer_id)
            if max_optimizer_id >= len(self.optimizers):
                raise ValueError(
                    f"Optimizer ID {max_optimizer_id} referenced but only "
                    f"{len(self.optimizers)} optimizer(s) provided"
                )

        # Resume from checkpoint if specified
        resumed = False
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
            resumed = True

        # Setup reproducibility if configured and not resuming
        if (
            not resumed
            and hasattr(self.config, "seed")
            and self.config.seed is not None
        ):
            seed_all(self.config.seed)

        logger.info(
            f"Starting training for {max_epochs} epochs with "
            f"{len(train_loaders)} training loader(s)"
        )

        try:
            self._training_loop(train_loaders, val_loaders, max_epochs)
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
        train_loaders: list[DataLoader],
        val_loaders: list[DataLoader] | None,
        max_epochs: int,
    ) -> None:
        """Main training loop implementation."""

        for epoch in range(self.current_epoch, max_epochs):
            # Note: epoch is the 0-based epoch index
            # current_epoch will be updated to epoch+1 after successful completion

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
            train_metrics = self._train_epoch(epoch)

            # Validation based on frequency configuration
            val_metrics = {}
            should_validate = False

            if val_loaders is not None:
                if self.config.validation.frequency == ValidationFrequency.PER_EPOCH:
                    should_validate = (
                        epoch + 1
                    ) % self.config.validate_every_n_epochs == 0
                elif (
                    self.config.validation.frequency
                    == ValidationFrequency.EVERY_N_STEPS
                ):
                    # Validation frequency handled within training loop
                    pass  # Validation already triggered during training if needed

                if should_validate:
                    val_metrics = self._validation_epoch(epoch)

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Log epoch metrics
            self._log_epoch_metrics(epoch, epoch_metrics)

            # Scheduler step (epoch-based)
            for scheduler in self.schedulers:
                self.resume_state = update_resume_state(
                    self.resume_state,
                    TrainerPhase.SCHEDULER_EPOCH_STEP,
                    save_rng=self.config.checkpoint.save_rng,
                )
                scheduler.step()

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

            # Update current_epoch to reflect completed epochs
            self.current_epoch = epoch + 1

        # Final checkpoint
        self.resume_state = update_resume_state(
            self.resume_state,
            TrainerPhase.TRAINING_COMPLETE,
            save_rng=self.config.checkpoint.save_rng,
        )
        self._save_checkpoint(force=True)

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        """Execute one training epoch with multi-dataloader support."""
        self.model.train()

        # Create epoch iterator
        iterator = self.dataloader_manager.create_epoch_iterator("train", epoch)

        # Metrics tracking
        per_loader_metrics: dict[int, dict[str, list[float]]] = {}
        per_loader_samples: dict[int, int] = {}

        # Accumulation state
        accumulation_counter = 0
        accumulated_loss = 0.0

        # Get validation loaders for step-based validation
        val_loaders = self.dataloader_manager.val_loaders

        for loader_idx, batch in iterator:
            # Check for preemption
            if self.signal_handler.is_preemption_requested():
                logger.info("Preemption requested during training")
                self._save_checkpoint(force=True)
                return self._aggregate_metrics(per_loader_metrics, per_loader_samples)

            # Get loader name
            loader_name = self.dataloader_manager.train_names[loader_idx]

            # Update resume state with multi-dataloader state
            loader_states = iterator.get_loader_states()
            multi_train = MultiTrainMicroState(
                active_loader_id=loader_idx,
                micro_step=accumulation_counter
                % self.config.performance.gradient_accumulation_steps,
                loader_states=loader_states,
                schedule_position=iterator.schedule_position,
                total_steps_completed=iterator.total_batches,
            )
            self.resume_state = update_resume_state(
                self.resume_state,
                TrainerPhase.TRAIN_BATCH_LOAD,
                multi_train=multi_train,
                save_rng=self.config.checkpoint.save_rng,
            )

            # Training step with AMP
            step_metrics = self._training_step(
                batch, loader_idx, loader_name, accumulation_counter
            )

            # Apply per-loader loss weight if configured
            loss = step_metrics["loss"]
            if self.config.loss_weights_per_loader is not None:
                loss = loss * self.config.loss_weights_per_loader[loader_idx]

            # Accumulate loss
            accumulated_loss += loss

            # Track per-loader metrics
            if loader_idx not in per_loader_metrics:
                per_loader_metrics[loader_idx] = {}
                per_loader_samples[loader_idx] = 0

            for key, value in step_metrics.items():
                if isinstance(value, int | float | torch.Tensor):
                    if key not in per_loader_metrics[loader_idx]:
                        per_loader_metrics[loader_idx][key] = []
                    per_loader_metrics[loader_idx][key].append(float(value))

            # Count samples (assuming first dimension is batch size)
            batch_size = self._get_batch_size(batch)
            per_loader_samples[loader_idx] += batch_size

            # Gradient accumulation boundary
            accumulation_counter += 1
            if (
                accumulation_counter
                % self.config.performance.gradient_accumulation_steps
                == 0
            ):
                # Optimizer step
                self._optimizer_step(loader_idx)
                accumulated_loss = 0.0

                # Step-based validation (only count after optimizer steps)
                if (
                    val_loaders  # only if non-empty list
                    and self.config.validation.frequency
                    == ValidationFrequency.EVERY_N_STEPS
                    and self.config.validation.every_n_steps is not None
                ):
                    self.steps_since_validation += 1
                    if (
                        self.steps_since_validation
                        >= self.config.validation.every_n_steps
                    ):
                        val_metrics = self._validation_epoch(epoch)
                        self._log_epoch_metrics(epoch, val_metrics, prefix="step_val")
                        self.steps_since_validation = 0

            # Log step metrics
            if (
                self.config.log_loss_every_n_steps is not None
                and self.global_step % self.config.log_loss_every_n_steps == 0
            ):
                self._log_step_metrics(
                    step_metrics, self.global_step, f"train/{loader_name}"
                )

        # Final optimizer step if there's remaining gradients
        if (
            accumulation_counter % self.config.performance.gradient_accumulation_steps
            != 0
        ):
            # Need to step optimizer for remaining accumulated gradients
            # Use the last loader's optimizer
            last_loader_idx = loader_idx  # From the last iteration
            self._optimizer_step(last_loader_idx)

            # Check for step-based validation after final optimizer step
            if (
                val_loaders  # only if non-empty list
                and self.config.validation.frequency
                == ValidationFrequency.EVERY_N_STEPS
                and self.config.validation.every_n_steps is not None
            ):
                self.steps_since_validation += 1
                if self.steps_since_validation >= self.config.validation.every_n_steps:
                    val_metrics = self._validation_epoch(epoch)
                    self._log_epoch_metrics(epoch, val_metrics, prefix="step_val")
                    self.steps_since_validation = 0

        # Aggregate metrics
        return self._aggregate_metrics(per_loader_metrics, per_loader_samples)

    def _validation_epoch(self, epoch: int) -> dict[str, Any]:
        """Execute one validation epoch with multi-dataloader support."""
        if (
            self.validation_step_fn is None
            or self.dataloader_manager.val_loaders is None
        ):
            return {}

        self.model.eval()

        # Create epoch iterator
        iterator = self.dataloader_manager.create_epoch_iterator("val", epoch)

        # Metrics tracking
        per_loader_metrics: dict[int, dict[str, list[float]]] = {}
        per_loader_samples: dict[int, int] = {}

        with torch.no_grad():
            for loader_idx, batch in iterator:
                # Get loader name
                loader_name = self.dataloader_manager.val_names[loader_idx]

                # Update resume state with multi-dataloader state
                loader_states = iterator.get_loader_states()
                multi_val = MultiValMicroState(
                    active_loader_id=loader_idx,
                    loader_states=loader_states,
                    accumulated_metrics={},  # Will be used for aggregation
                )
                self.resume_state = update_resume_state(
                    self.resume_state,
                    TrainerPhase.VAL_BATCH_LOAD,
                    multi_val=multi_val,
                    save_rng=self.config.checkpoint.save_rng,
                )

                # Validation step
                step_metrics = self.validation_step_fn(
                    self, batch, loader_idx, loader_name
                )

                # Track per-loader metrics
                if loader_idx not in per_loader_metrics:
                    per_loader_metrics[loader_idx] = {}
                    per_loader_samples[loader_idx] = 0

                for key, value in step_metrics.items():
                    if isinstance(value, int | float | torch.Tensor):
                        if key not in per_loader_metrics[loader_idx]:
                            per_loader_metrics[loader_idx][key] = []
                        per_loader_metrics[loader_idx][key].append(float(value))

                # Count samples
                batch_size = self._get_batch_size(batch)
                per_loader_samples[loader_idx] += batch_size

        # Aggregate validation metrics based on configuration
        return self._aggregate_validation_metrics(
            per_loader_metrics, per_loader_samples
        )

    def _training_step(
        self,
        batch: Any,
        loader_idx: int,
        loader_name: str,
        accumulation_step: int,
    ) -> dict[str, Any]:
        """Execute single training step with gradient accumulation and AMP."""

        # Forward pass
        self.resume_state = update_resume_state(
            self.resume_state,
            TrainerPhase.TRAIN_BATCH_FORWARD,
            save_rng=self.config.checkpoint.save_rng,
        )

        # Execute training step with AMP if configured and CUDA is available
        if self.config.performance.use_amp and torch.cuda.is_available():
            with autocast("cuda"):
                step_metrics = self.training_step_fn(
                    self, batch, loader_idx, loader_name
                )
        else:
            step_metrics = self.training_step_fn(self, batch, loader_idx, loader_name)

        loss = step_metrics["loss"]
        # Coerce numeric losses to tensors and ensure requires_grad for backward
        if not isinstance(loss, torch.Tensor):
            try:
                device = next(self.model.parameters()).device  # type: ignore[attr-defined]
            except Exception:
                device = None
            loss = torch.tensor(float(loss), device=device, requires_grad=True)
            step_metrics["loss"] = loss
        elif not loss.requires_grad:
            loss = loss.clone().detach().requires_grad_(True)
            step_metrics["loss"] = loss

        # Scale loss for gradient accumulation
        loss = loss / self.config.performance.gradient_accumulation_steps

        # Backward pass
        self.resume_state = update_resume_state(
            self.resume_state,
            TrainerPhase.TRAIN_BATCH_BACKWARD,
            save_rng=self.config.checkpoint.save_rng,
        )

        # Use scaler for backward pass if available
        if self.scaler is not None:
            # Scale loss and backward
            scaled_loss = self.scaler.scale(loss)
            if self.fabric is not None:
                self.fabric.backward(scaled_loss)
            else:
                scaled_loss.backward()
        # Regular backward
        elif self.fabric is not None:
            self.fabric.backward(loss)
        else:
            loss.backward()

        return step_metrics

    def _optimizer_step(self, loader_idx: int) -> None:
        """Execute optimizer step for the specified loader."""
        # Determine which optimizer to use
        if self.config.per_loader_optimizer_id is not None:
            optimizer_idx = self.config.per_loader_optimizer_id[loader_idx]
        else:
            optimizer_idx = 0  # Default to first optimizer

        if optimizer_idx >= len(self.optimizers):
            optimizer_idx = 0  # Fallback to first optimizer

        optimizer = self.optimizers[optimizer_idx]

        # Update resume state
        self.resume_state = update_resume_state(
            self.resume_state,
            TrainerPhase.TRAIN_BATCH_OPTIM_STEP,
            save_rng=self.config.checkpoint.save_rng,
        )

        # Handle optimizer step with or without scaler
        if self.scaler is not None:
            # Unscale gradients before clipping
            self.scaler.unscale_(optimizer)

            # Gradient clipping
            if self.config.performance.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.performance.clip_grad_norm,
                )

            # Optimizer step with scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Gradient clipping without scaler
            if self.config.performance.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.performance.clip_grad_norm,
                )

            # Regular optimizer step
            optimizer.step()

        optimizer.zero_grad()
        self.global_step += 1

        # Scheduler step (step-based) if exists for this optimizer
        if optimizer_idx < len(self.schedulers):
            scheduler = self.schedulers[optimizer_idx]
            if scheduler is not None:
                self.resume_state = update_resume_state(
                    self.resume_state,
                    TrainerPhase.SCHEDULER_BATCH_STEP,
                    save_rng=self.config.checkpoint.save_rng,
                )
                scheduler.step()

        # Record performance
        self.performance_monitor.record_step_time()
        self.performance_monitor.record_memory_usage()

        # Optional step-based checkpoint save
        try:
            if self.checkpoint_manager.should_save_checkpoint(
                self.current_epoch, self.global_step
            ):
                self._save_checkpoint()
        except Exception:
            # Don't interrupt training if periodic save fails; it will retry on next opportunity
            logger.debug(
                "Step-based checkpoint save skipped due to error", exc_info=True
            )

    def _aggregate_metrics(
        self,
        per_loader_metrics: dict[int, dict[str, list[float]]],
        per_loader_samples: dict[int, int],
    ) -> dict[str, Any]:
        """Aggregate metrics across dataloaders."""
        final_metrics = {}

        # Aggregate per-loader metrics
        if self.config.log_per_loader_metrics:
            for loader_idx, metrics in per_loader_metrics.items():
                loader_name = self.dataloader_manager.train_names[loader_idx]
                for key, values in metrics.items():
                    # Average within loader
                    avg_value = sum(values) / len(values)
                    final_metrics[f"train/{loader_name}/{key}"] = avg_value

        # Aggregate global metrics (weighted by samples)
        if self.config.log_global_metrics:
            total_samples = sum(per_loader_samples.values())
            global_metrics: dict[str, float] = {}

            # Collect all metric keys
            all_keys = set()
            for metrics in per_loader_metrics.values():
                all_keys.update(metrics.keys())

            for key in all_keys:
                weighted_sum = 0.0
                for loader_idx, metrics in per_loader_metrics.items():
                    if key in metrics:
                        loader_avg = sum(metrics[key]) / len(metrics[key])
                        weight = per_loader_samples[loader_idx] / total_samples
                        weighted_sum += loader_avg * weight

                global_metrics[f"train/{key}"] = weighted_sum

            final_metrics.update(global_metrics)

        return final_metrics

    def _aggregate_validation_metrics(
        self,
        per_loader_metrics: dict[int, dict[str, list[float]]],
        per_loader_samples: dict[int, int],
    ) -> dict[str, Any]:
        """Aggregate validation metrics based on configuration."""
        final_metrics = {}

        # Per-loader metrics
        if self.config.validation.per_loader_metrics:
            for loader_idx, metrics in per_loader_metrics.items():
                loader_name = self.dataloader_manager.val_names[loader_idx]
                for key, values in metrics.items():
                    avg_value = sum(values) / len(values)
                    final_metrics[f"val/{loader_name}/{key}"] = avg_value

        # Global aggregation based on strategy
        if self.config.validation.global_metrics:
            aggregation = self.config.validation.aggregation

            if aggregation == ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES:
                # Weight by sample count
                total_samples = sum(per_loader_samples.values())

                all_keys = set()
                for metrics in per_loader_metrics.values():
                    all_keys.update(metrics.keys())

                for key in all_keys:
                    weighted_sum = 0.0
                    for loader_idx, metrics in per_loader_metrics.items():
                        if key in metrics:
                            loader_avg = sum(metrics[key]) / len(metrics[key])
                            weight = per_loader_samples[loader_idx] / total_samples
                            weighted_sum += loader_avg * weight

                    final_metrics[f"val/{key}"] = weighted_sum

            elif aggregation == ValAggregation.MACRO_AVG_EQUAL_LOADERS:
                # Equal weight per loader
                # num_loaders intentionally not needed; average across present keys

                all_keys = set()
                for metrics in per_loader_metrics.values():
                    all_keys.update(metrics.keys())

                for key in all_keys:
                    total = 0.0
                    count = 0
                    for _loader_idx, metrics in per_loader_metrics.items():
                        if key in metrics:
                            loader_avg = sum(metrics[key]) / len(metrics[key])
                            total += loader_avg
                            count += 1

                    if count > 0:
                        final_metrics[f"val/{key}"] = total / count

            elif aggregation == ValAggregation.PRIMARY_METRIC_PER_LOADER:
                # Just keep per-loader metrics (already added above)
                pass

            elif aggregation == ValAggregation.CUSTOM:
                # User should override this method for custom aggregation
                logger.warning("CUSTOM aggregation selected but not implemented")

        return final_metrics

    def _get_batch_size(self, batch: Any) -> int:
        """Get batch size from various batch formats."""
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        if isinstance(batch, list | tuple) and len(batch) > 0:
            return self._get_batch_size(batch[0])
        if isinstance(batch, dict):
            # Try common keys
            for key in ["input", "inputs", "x", "data"]:
                if key in batch:
                    return self._get_batch_size(batch[key])
            # Fallback to first value
            if batch:
                first_value = next(iter(batch.values()))
                return self._get_batch_size(first_value)
        return 1  # Default to 1 if can't determine

    def _save_checkpoint(
        self, metrics: dict[str, Any] | None = None, force: bool = False
    ) -> None:
        """Save checkpoint with timeout handling."""
        try:
            # Update resume state with dataloader manager state
            if self.dataloader_manager:
                self.resume_state = update_resume_state(
                    self.resume_state,
                    self.resume_state.phase,
                    dataloader_manager_state=self.dataloader_manager.get_state(),
                    save_rng=self.config.checkpoint.save_rng,
                    choice_rng=getattr(self.dataloader_manager, "choice_rng", None),
                )

            # Save checkpoint with all optimizers and schedulers
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizers=self.optimizers,  # Save all optimizers
                schedulers=self.schedulers,  # Save all schedulers
                resume_state=self.resume_state,
                epoch=self.current_epoch,
                global_step=self.global_step,
                metrics=metrics,
                timeout_seconds=self.config.preemption.max_checkpoint_sec
                if not force
                else None,
                scaler=self.scaler,  # Save AMP scaler state
            )
        except Exception:
            logger.exception("Failed to save checkpoint: ")
            if force:
                raise

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        try:
            # Restore checkpoint with all optimizers and schedulers
            epoch, global_step, resume_state = (
                self.checkpoint_manager.restore_from_checkpoint(
                    model=self.model,
                    optimizers=self.optimizers,  # Restore all optimizers
                    schedulers=self.schedulers,  # Restore all schedulers
                    checkpoint_path=checkpoint_path,
                    scaler=self.scaler,  # Restore AMP scaler state
                )
            )

            self.current_epoch = epoch
            self.global_step = global_step

            if resume_state is not None:
                self.resume_state = resume_state

                # Restore RNG state if available
                if resume_state.rng is not None:
                    restore_rng_state(resume_state.rng)

                # Restore DataLoaderManager state if available
                if resume_state.dataloader_manager_state and self.dataloader_manager:
                    self.dataloader_manager.load_state(
                        resume_state.dataloader_manager_state
                    )

                # Restore choice RNG if available
                if resume_state.choice_rng and hasattr(
                    self.dataloader_manager, "choice_rng"
                ):
                    from .states import restore_choice_rng_state

                    restore_choice_rng_state(
                        resume_state.choice_rng, self.dataloader_manager.choice_rng
                    )

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
            if isinstance(value, int | float | torch.Tensor):
                log_key = f"{prefix}/{key}" if prefix else key
                log_dict[log_key] = float(value)

        if self.wandb_run is not None:
            self.wandb_run.log(log_dict, step=step)

        # Console logging
        # Log a compact loss line if present
        loss_key = f"{prefix}/loss" if prefix else "loss"
        if loss_key in log_dict:
            logger.info(f"Step {step}: loss={log_dict[loss_key]:.4f}")
        else:
            logger.info(f"Step {step}")

    def _log_epoch_metrics(
        self, epoch: int, metrics: dict[str, Any], prefix: str = ""
    ) -> None:
        """Log epoch-level metrics."""
        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, int | float | torch.Tensor):
                log_key = f"{prefix}/{key}" if prefix else key
                log_dict[log_key] = float(value)

        if self.wandb_run is not None:
            self.wandb_run.log(log_dict, step=self.global_step)

        # Console logging
        if log_dict:
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
