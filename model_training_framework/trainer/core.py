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
import os
from pathlib import Path
import subprocess
import time
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch.amp.autocast_mode import autocast

from .checkpoints import CheckpointManager
from .config import (
    GenericTrainerConfig,
    ValAggregation,
    ValidationFrequency,
    validate_trainer_config,
)
from .dataset_validation import validate_iterable_dataset_checkpointing
from .hooks import (
    EarlyStoppingHook,
    GradientMonitorHook,
    HookManager,
    LoggingHook,
    ModelCheckpointHook,
)
from .loggers import LoggerProtocol, create_logger
from .metrics import AggregationStrategy, MetricsManager
from .multi_dataloader import DataLoaderManager, SamplingStrategy
from .states import (
    MultiTrainMicroState,
    MultiValMicroState,
    TrainerPhase,
    create_initial_resume_state,
    restore_choice_rng_state,
    restore_rng_state,
    update_resume_state,
)
from .utils import (
    EarlyStopping,
    PerformanceMonitor,
    SignalHandler,
    ddp_all_reduce,
    ddp_barrier,
    ddp_broadcast_object,
    ddp_is_primary,
    seed_all,
)

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data import DataLoader

    # No direct import of ResumeState needed here; runtime-safe alias is used below

logger = logging.getLogger(__name__)

# Runtime reference to ResumeState for isinstance/attribute access without triggering import cycles
# Use a module import to avoid assigning to a type alias
_ResumeStateRuntime: Any | None = None
try:  # Guard to avoid hard import issues in unusual import orders
    from . import states as _states_mod

    _ResumeStateRuntime = _states_mod.ResumeState
except Exception:  # pragma: no cover - defensive fallback
    _ResumeStateRuntime = None


class TrainingStep(Protocol):
    """Protocol for training step function."""

    def __call__(
        self,
        trainer: GenericTrainer,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        dataloader_name: str,
    ) -> dict[str, Any]:
        """
        Execute training step and return metrics.

        Args:
            trainer: The trainer instance
            batch: Input batch
            batch_idx: 0-based batch index within this dataloader for the epoch
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
        batch_idx: int,
        dataloader_idx: int,
        dataloader_name: str,
    ) -> dict[str, Any]:
        """
        Execute validation step and return metrics.

        Args:
            trainer: The trainer instance
            batch: Input batch
            batch_idx: 0-based batch index within this dataloader for the epoch
            dataloader_idx: Index of the current dataloader
            dataloader_name: Name of the current dataloader

        Returns:
            Dictionary with metrics (typically including 'loss')
        """
        ...


class GenericTrainer:
    """
    Multi-dataloader trainer with preemption safety and fault tolerance.

    This trainer is designed as a multi-dataloader-only engine. Even single dataloader
    scenarios must use the multi-dataloader API with a list containing one loader.

    Features:
    - Multi-dataloader training with deterministic scheduling
    - Per-loader optimizer routing and loss weighting
    - Fault-tolerant training with SLURM preemption handling
    - Instruction-level resume capability
    - Distributed training support via Lightning Fabric
    - Comprehensive monitoring and logging

    Example:
        # Single dataloader (must still use list)
        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=[optimizer],  # Always a list
            fabric=fabric
        )
        trainer.fit(
            train_loaders=[train_loader],  # Single loader in list
            val_loaders=[val_loader]       # Single loader in list
        )

        # Multiple dataloaders
        trainer.fit(
            train_loaders=[loader_a, loader_b, loader_c],
            val_loaders=[val_a, val_b]
        )
    """

    def __init__(
        self,
        config: GenericTrainerConfig,
        model: Module,
        optimizers: list[Optimizer],
        schedulers: list[_LRScheduler] | None = None,
        fabric: Any = None,  # Lightning Fabric
    ):
        """
        Initialize the multi-dataloader trainer.

        Note: This trainer operates exclusively in multi-dataloader mode.
        Single dataloader usage requires wrapping the loader in a list.

        Args:
            config: Trainer configuration with MultiDataLoaderConfig
            model: Model to train
            optimizers: List of optimizers (always a list, even for single optimizer)
            schedulers: Optional list of learning rate schedulers
            fabric: Optional Lightning Fabric instance for distributed training

        Example:
            config = GenericTrainerConfig(
                train_loader_config=MultiDataLoaderConfig(
                    sampling_strategy=SamplingStrategy.ROUND_ROBIN,
                    dataloader_names=["main"]  # Even for single loader
                ),
                val_loader_config=MultiDataLoaderConfig(
                    sampling_strategy=SamplingStrategy.SEQUENTIAL,
                    dataloader_names=["val_main"]
                ),
            )
            trainer = GenericTrainer(
                config=config,
                model=model,
                optimizers=[optimizer],  # Single optimizer in list
            )
        """
        self.config = config
        self.fabric = fabric

        # Setup model and optimizers with Fabric if available
        if fabric is not None:
            # Fabric setup for model and optimizers
            # Note: Fabric.setup() handles both model and optimizers
            setup_result = fabric.setup(model, *optimizers)
            if isinstance(setup_result, tuple):
                self.model = setup_result[0]
                self.optimizers = list(setup_result[1:])
            else:
                # Single model case
                self.model = setup_result
                self.optimizers = optimizers
        else:
            self.model = model
            self.optimizers = optimizers

        self.schedulers = schedulers or []

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

        # Initialize logger based on configuration
        self.logger: LoggerProtocol | None = None
        if ddp_is_primary(fabric):  # Only create logger on primary rank
            log_dir = (
                Path(config.logging.tensorboard_dir)
                if config.logging.tensorboard_dir
                else None
            )
            self.logger = create_logger(
                logger_type=config.logging.logger_type,
                project=config.logging.wandb_project,
                log_dir=log_dir,
                entity=config.logging.wandb_entity,
                tags=config.logging.wandb_tags,
                notes=config.logging.wandb_notes,
                loggers_list=config.logging.composite_loggers,
            )

        # Run outcome flags to distinguish "current run finished" vs "training fully complete"
        self._was_preempted: bool = False
        self._was_interrupted: bool = False
        self._was_failed: bool = False

        # Initialize metrics manager
        self.metrics_manager: MetricsManager | None = None

        # Initialize hook manager
        self.hook_manager = HookManager()

        # Load custom hooks from class paths first
        for hook_class_path in config.hooks.hook_classes:
            try:
                # Dynamic import of hook class
                module_path, class_name = hook_class_path.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                hook_class = getattr(module, class_name)

                # Get hook-specific config if available
                hook_config = config.hooks.hook_configs.get(class_name, {})

                # Instantiate and register hook
                hook_instance = hook_class(**hook_config)
                self.hook_manager.register_hook(hook_instance)
            except Exception:
                logger.exception(f"Failed to load hook: {hook_class_path}")
                if not config.hooks.continue_on_hook_error:
                    raise

        # Register built-in hooks last so user hooks run first
        if config.hooks.enable_logging_hook:
            self.hook_manager.register_hook(
                LoggingHook(config.logging.console_log_level)
            )
        if config.hooks.enable_gradient_monitor:
            self.hook_manager.register_hook(
                GradientMonitorHook(**config.hooks.gradient_monitor_config)
            )
        if config.hooks.enable_model_checkpoint_hook:
            self.hook_manager.register_hook(
                ModelCheckpointHook(**config.hooks.model_checkpoint_config)
            )
        if config.hooks.enable_early_stopping_hook:
            self.hook_manager.register_hook(
                EarlyStoppingHook(**config.hooks.early_stopping_config)
            )

        logger.info(f"Initialized GenericTrainer with {len(optimizers)} optimizer(s)")

    def _maybe_requeue_job(self) -> None:
        """Request Slurm to requeue the current job after preemption.

        Respects config.preemption.requeue_job, only runs on primary rank,
        and requires SLURM_JOB_ID in environment.
        """
        try:
            requeue_enabled = getattr(self.config.preemption, "requeue_job", True)
        except Exception:
            requeue_enabled = True
        if not requeue_enabled:
            return

        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return
        if not ddp_is_primary(self.fabric):
            return
        try:
            logger.info(f"Requesting Slurm requeue for job {job_id}")
            subprocess.run(["scontrol", "requeue", job_id], check=True)
            logger.info("Requeue request sent successfully")
        except Exception:
            logger.exception("Failed to requeue job: ")

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

        # Ensure resume_state is of correct type (convert dicts from external loaders)
        try:
            if isinstance(self.resume_state, dict) and _ResumeStateRuntime is not None:
                self.resume_state = _ResumeStateRuntime.from_dict(self.resume_state)
        except Exception:
            logger.debug(
                "resume_state conversion failed; continuing without conversion",
                exc_info=True,
            )

        # Validate IterableDataset checkpointing support if fault tolerance is enabled
        if (
            self.config.fault_tolerance
            and self.config.fault_tolerance.save_dataset_state
        ):
            validate_iterable_dataset_checkpointing(
                train_loaders, require_checkpointing=True
            )
            if val_loaders:
                validate_iterable_dataset_checkpointing(
                    val_loaders, require_checkpointing=True
                )

        # Initialize DataLoader manager
        self.dataloader_manager = DataLoaderManager(
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            train_config=self.config.train_loader_config,
            val_config=self.config.val_loader_config,
            fabric=self.fabric,
            logger=logger,
        )

        # Initialize metrics manager with loader names and aggregation strategy
        # Map ValidationConfig.aggregation (ValAggregation) to AggregationStrategy
        aggregation_strategy = AggregationStrategy.WEIGHTED_AVERAGE  # Default
        if self.config.validation:
            val_agg = self.config.validation.aggregation
            if val_agg == ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES:
                aggregation_strategy = AggregationStrategy.WEIGHTED_AVERAGE
            elif val_agg == ValAggregation.MACRO_AVG_EQUAL_LOADERS:
                aggregation_strategy = AggregationStrategy.SIMPLE_AVERAGE
            # PRIMARY_METRIC_PER_LOADER and CUSTOM default to WEIGHTED_AVERAGE for now

        self.metrics_manager = MetricsManager(
            train_loader_names=self.dataloader_manager.train_names,
            val_loader_names=self.dataloader_manager.val_names if val_loaders else None,
            aggregation_strategy=aggregation_strategy,
            track_proportions=self.config.logging.log_loader_proportions,
        )

        # Set expected proportions for WEIGHTED strategy
        if (
            self.config.train_loader_config.sampling_strategy
            == SamplingStrategy.WEIGHTED
            and self.config.train_loader_config.dataloader_weights
        ):
            assert self.metrics_manager is not None
            self.metrics_manager.set_expected_proportions(
                self.config.train_loader_config.dataloader_weights
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
        if not resumed:
            seed_value = getattr(self.config, "seed", None)
            if seed_value is not None:
                seed_all(seed_value)

        logger.info(
            f"Starting training for {max_epochs} epochs with "
            f"{len(train_loaders)} training loader(s)"
        )

        try:
            self._training_loop(train_loaders, val_loaders, max_epochs)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint(force=True)
            self._was_interrupted = True
        except Exception:
            logger.exception("Training failed with error: ")
            self._save_checkpoint(force=True)
            self._was_failed = True
            raise
        finally:
            # Fire training end hooks and cleanup logger/resources
            try:
                self.hook_manager.call_hook("on_train_end", self)
            except Exception:
                logger.debug("on_train_end hook error", exc_info=True)
            # Close logger (primary rank only)
            try:
                if self.logger and ddp_is_primary(self.fabric):
                    self.logger.close()
            except Exception:
                logger.debug("Logger close error", exc_info=True)
            self.signal_handler.restore_handlers()

        # Distinguish current run completion from full training completion
        if self._was_preempted:
            logger.info(
                "Run completed (preempted). Checkpoint saved; resume to continue training."
            )
        elif self._was_interrupted:
            logger.info(
                "Run completed (user interrupt). Checkpoint saved; resume to continue training."
            )
        elif self._was_failed:
            logger.info("Run completed (error). See logs above; checkpoint saved.")
        else:
            logger.info("Training fully completed")

    def _training_loop(
        self,
        train_loaders: list[DataLoader],
        val_loaders: list[DataLoader] | None,
        max_epochs: int,
    ) -> None:
        """Main training loop implementation."""

        # Call on_train_start hook
        self.hook_manager.call_hook("on_train_start", self)

        for epoch in range(self.current_epoch, max_epochs):
            # Note: epoch is the 0-based epoch index
            # current_epoch will be updated to epoch+1 after successful completion

            # Synchronize at epoch start for DDP
            ddp_barrier(self.fabric)

            # Call on_epoch_start hook
            self.hook_manager.call_hook("on_epoch_start", self, epoch)

            # Reset metrics for new epoch
            assert self.metrics_manager is not None
            self.metrics_manager.reset_epoch("both")

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
                t0 = time.time()
                self._save_checkpoint(force=True)
                dt = time.time() - t0
                logger.info(f"Preemption checkpoint saved in {dt:.2f}s")
                self._was_preempted = True
                # Ask Slurm to requeue this job before exiting
                self._maybe_requeue_job()
                return

            # Training epoch
            self._train_epoch(epoch)

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
                    # Call validation hooks around per-epoch validation
                    self.hook_manager.call_hook("on_validation_start", self, epoch)
                    val_metrics = self._validation_epoch(epoch)
                    self.hook_manager.call_hook(
                        "on_validation_end", self, epoch, val_metrics
                    )

            # Optionally aggregate metrics across ranks for DDP
            if self.config.logging.all_reduce_metrics and self.fabric is not None:
                assert self.metrics_manager is not None
                self.metrics_manager.aggregate_across_ranks(self.fabric)

            # Get combined metrics from MetricsManager
            epoch_metrics = {
                **self.metrics_manager.get_train_metrics(
                    include_global=self.config.logging.log_global_metrics,
                    include_per_loader=self.config.logging.log_per_loader_metrics,
                ),
                **self.metrics_manager.get_val_metrics(
                    include_global=self.config.logging.log_global_metrics,
                    include_per_loader=self.config.logging.log_per_loader_metrics,
                ),
            }

            # Log loader proportions if using WEIGHTED strategy
            if (
                self.config.logging.log_loader_proportions
                and self.config.train_loader_config.sampling_strategy
                == SamplingStrategy.WEIGHTED
            ):
                proportions, counts = self.metrics_manager.get_loader_proportions()
                if self.logger and ddp_is_primary(self.fabric):
                    self.logger.log_loader_proportions(epoch, proportions, counts)

            # Save epoch summary
            epoch_summary = self.metrics_manager.save_epoch_summary(
                epoch, {"global_step": self.global_step}
            )

            # Log epoch metrics through logger
            if self.logger and ddp_is_primary(self.fabric):
                self.logger.log_metrics(epoch_metrics, self.global_step)
                self.logger.log_epoch_summary(epoch, epoch_summary)

            # Call on_epoch_end hook
            self.hook_manager.call_hook("on_epoch_end", self, epoch, epoch_metrics)

            # Scheduler step (epoch-based)
            for scheduler in self.schedulers:
                self.resume_state = update_resume_state(
                    self.resume_state,
                    TrainerPhase.SCHEDULER_EPOCH_STEP,
                    save_rng=self.config.checkpoint.save_rng,
                )
                scheduler.step()

            # Save checkpoint if needed (only on primary rank)
            if self.checkpoint_manager.should_save_checkpoint(epoch, self.global_step):
                if ddp_is_primary(self.fabric):
                    self._save_checkpoint(epoch_metrics)
                # All ranks wait for checkpoint save to complete
                ddp_barrier(self.fabric)

            # Early stopping (legacy utility + hook), synchronized across ranks
            local_should_stop = False
            source = self.config.validation.early_stopping_source

            # Check legacy early stopping based on source configuration
            if (
                source in ("both", "legacy")
                and self.early_stopping is not None
                and self.early_stopping(epoch_metrics)
            ):
                local_should_stop = True

            # Check early stopping hook based on source configuration
            if source in ("both", "hook"):
                for hook in self.hook_manager.hooks:
                    if isinstance(hook, EarlyStoppingHook) and getattr(
                        hook, "should_stop", False
                    ):
                        local_should_stop = True
                        break

            global_should_stop = local_should_stop
            if self.fabric is not None:
                try:
                    flag = torch.tensor(
                        1.0 if local_should_stop else 0.0, dtype=torch.float32
                    )
                    reduced = ddp_all_reduce(self.fabric, flag, op="sum")
                    global_should_stop = reduced.item() > 0.5
                except Exception:
                    # Fallback to local decision if reduction fails
                    global_should_stop = local_should_stop

            if global_should_stop:
                logger.info("Early stopping condition met; terminating training")
                break

            # Update resume state
            self.resume_state = update_resume_state(
                self.resume_state,
                TrainerPhase.EPOCH_END,
                save_rng=self.config.checkpoint.save_rng,
            )

            # Synchronize at epoch end for DDP
            ddp_barrier(self.fabric)

            # Update current_epoch to reflect completed epochs
            self.current_epoch = epoch + 1

        # Final checkpoint (only on primary rank)
        self.resume_state = update_resume_state(
            self.resume_state,
            TrainerPhase.TRAINING_COMPLETE,
            save_rng=self.config.checkpoint.save_rng,
        )
        if ddp_is_primary(self.fabric):
            self._save_checkpoint(force=True)
        # All ranks wait for final checkpoint
        ddp_barrier(self.fabric)

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        """Execute one training epoch with multi-dataloader support."""
        self.model.train()

        # Create epoch iterator
        assert self.dataloader_manager is not None
        iterator = self.dataloader_manager.create_epoch_iterator("train", epoch)

        # Log a short schedule preview for demo/debugging
        try:
            names = self.dataloader_manager.train_names
            preview_len = min(20, len(getattr(iterator, "schedule", [])))
            if preview_len > 0 and self.logger and ddp_is_primary(self.fabric):
                schedule = getattr(iterator, "schedule", [])[:preview_len]
                preview_names = ", ".join(names[i] for i in schedule)
                self.logger.log_text(
                    "schedule_preview",
                    f"epoch={epoch} next loaders: {preview_names}",
                    step=self.global_step,
                )
        except Exception:
            logger.debug("Failed to log schedule preview", exc_info=True)

        # Accumulation state
        accumulation_counter = 0
        accumulated_loss = 0.0

        # Get validation loaders for step-based validation
        assert self.dataloader_manager is not None
        val_loaders = self.dataloader_manager.val_loaders

        last_loader_idx: int | None = None
        for loader_idx, batch in iterator:
            last_loader_idx = loader_idx
            # Check for preemption
            if self.signal_handler.is_preemption_requested():
                logger.info("Preemption requested during training")
                if ddp_is_primary(self.fabric):
                    t0 = time.time()
                    self._save_checkpoint(force=True)
                    dt = time.time() - t0
                    logger.info(f"Preemption checkpoint saved in {dt:.2f}s")
                # All ranks wait for checkpoint save
                ddp_barrier(self.fabric)
                self._was_preempted = True
                # Ask Slurm to requeue this job before exiting
                self._maybe_requeue_job()
                assert self.metrics_manager is not None
                return self.metrics_manager.get_train_metrics(
                    include_global=self.config.logging.log_global_metrics,
                    include_per_loader=self.config.logging.log_per_loader_metrics,
                )

            # Get loader name
            assert self.dataloader_manager is not None
            loader_name = self.dataloader_manager.train_names[loader_idx]

            # Call on_train_batch_start hook
            self.hook_manager.call_hook(
                "on_train_batch_start", self, batch, loader_idx, loader_name
            )

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

            # Determine 0-based batch index for this loader
            try:
                state_for_loader = loader_states[loader_idx]
                current_batch_idx = max(0, state_for_loader.batch_idx - 1)
            except Exception:
                current_batch_idx = 0

            # Training step with AMP
            step_metrics = self._training_step(
                batch, loader_idx, loader_name, accumulation_counter, current_batch_idx
            )

            # Apply per-loader loss weight if configured
            loss = step_metrics["loss"]
            if self.config.loss_weights_per_loader is not None:
                loss = loss * self.config.loss_weights_per_loader[loader_idx]

            # Accumulate loss
            accumulated_loss += loss

            # Get batch size for metrics tracking
            batch_size = self._get_batch_size(batch)

            # Add metrics to MetricsManager
            assert self.metrics_manager is not None
            self.metrics_manager.add_train_batch(loader_idx, step_metrics, batch_size)

            # Call on_train_batch_end hook
            self.hook_manager.call_hook(
                "on_train_batch_end", self, batch, loader_idx, loader_name, step_metrics
            )

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
                        # Log validation start with aggregator info
                        try:
                            if self.logger and ddp_is_primary(self.fabric):
                                agg = self.config.validation.aggregation.value
                                nload = (
                                    len(self.dataloader_manager.val_loaders)
                                    if self.dataloader_manager.val_loaders
                                    else 0
                                )
                                self.logger.log_text(
                                    "validation_start",
                                    f"step-based validation start (agg={agg}, loaders={nload})",
                                    step=self.global_step,
                                )
                        except Exception:
                            logger.debug("validation_start log failed", exc_info=True)
                        # Call validation hooks
                        self.hook_manager.call_hook("on_validation_start", self, epoch)
                        val_metrics = self._validation_epoch(epoch)
                        self.hook_manager.call_hook(
                            "on_validation_end", self, epoch, val_metrics
                        )

                        # Log validation metrics
                        if self.logger and ddp_is_primary(self.fabric):
                            self.logger.log_metrics(val_metrics, self.global_step)
                            # Log validation done with aggregator info
                            try:
                                agg = self.config.validation.aggregation.value
                                self.logger.log_text(
                                    "validation_done",
                                    f"validation done (agg={agg})",
                                    step=self.global_step,
                                )
                            except Exception:
                                logger.debug(
                                    "validation_done log failed", exc_info=True
                                )
                        self.steps_since_validation = 0

            # Log step metrics through logger
            if (
                self.config.log_loss_every_n_steps is not None
                and self.global_step % self.config.log_loss_every_n_steps == 0
                and self.logger
                and ddp_is_primary(self.fabric)
            ):
                # Format metrics with proper prefix
                formatted_metrics = {
                    f"train/dl_{loader_name}/{k}": v
                    for k, v in step_metrics.items()
                    if isinstance(v, int | float | torch.Tensor)
                }
                self.logger.log_metrics(formatted_metrics, self.global_step)

        # Final optimizer step if there's remaining gradients
        if (
            accumulation_counter % self.config.performance.gradient_accumulation_steps
            != 0
        ) and (last_loader_idx is not None):
            # Need to step optimizer for remaining accumulated gradients
            # Use the last loader's optimizer
            # Use the last observed loader index for selecting optimizer
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
                    try:
                        if self.logger and ddp_is_primary(self.fabric):
                            agg = self.config.validation.aggregation.value
                            nload = (
                                len(self.dataloader_manager.val_loaders)
                                if self.dataloader_manager.val_loaders
                                else 0
                            )
                            self.logger.log_text(
                                "validation_start",
                                f"step-based validation start (agg={agg}, loaders={nload})",
                                step=self.global_step,
                            )
                    except Exception:
                        logger.debug("validation_start log failed", exc_info=True)
                    # Call validation hooks
                    self.hook_manager.call_hook("on_validation_start", self, epoch)
                    val_metrics = self._validation_epoch(epoch)
                    self.hook_manager.call_hook(
                        "on_validation_end", self, epoch, val_metrics
                    )

                    # Log validation metrics
                    if self.logger and ddp_is_primary(self.fabric):
                        self.logger.log_metrics(val_metrics, self.global_step)
                        try:
                            agg = self.config.validation.aggregation.value
                            self.logger.log_text(
                                "validation_done",
                                f"validation done (agg={agg})",
                                step=self.global_step,
                            )
                        except Exception:
                            logger.debug("validation_done log failed", exc_info=True)
                    self.steps_since_validation = 0

        # Return aggregated metrics from MetricsManager
        assert self.metrics_manager is not None
        return self.metrics_manager.get_train_metrics(
            include_global=self.config.logging.log_global_metrics,
            include_per_loader=self.config.logging.log_per_loader_metrics,
        )

    def _validation_epoch(self, epoch: int) -> dict[str, Any]:
        """Execute one validation epoch with multi-dataloader support."""
        if (
            self.validation_step_fn is None
            or self.dataloader_manager is None
            or self.dataloader_manager.val_loaders is None
        ):
            return {}

        self.model.eval()

        # Create epoch iterator
        assert self.dataloader_manager is not None
        iterator = self.dataloader_manager.create_epoch_iterator("val", epoch)

        with torch.no_grad():
            for loader_idx, batch in iterator:
                # Get loader name
                assert self.dataloader_manager is not None
                assert self.dataloader_manager is not None
                loader_name = self.dataloader_manager.val_names[loader_idx]

                # Call on_validation_batch_start hook
                self.hook_manager.call_hook(
                    "on_validation_batch_start", self, batch, loader_idx, loader_name
                )

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
                assert self.validation_step_fn is not None
                try:
                    state_for_loader = loader_states[loader_idx]
                    current_batch_idx = max(0, state_for_loader.batch_idx - 1)
                except Exception:
                    current_batch_idx = 0

                step_metrics = self.validation_step_fn(
                    self, batch, current_batch_idx, loader_idx, loader_name
                )

                # Get batch size for metrics tracking
                batch_size = self._get_batch_size(batch)

                # Add metrics to MetricsManager
                assert self.metrics_manager is not None
                self.metrics_manager.add_val_batch(loader_idx, step_metrics, batch_size)

                # Call on_validation_batch_end hook
                self.hook_manager.call_hook(
                    "on_validation_batch_end",
                    self,
                    batch,
                    loader_idx,
                    loader_name,
                    step_metrics,
                )

        # Return aggregated validation metrics from MetricsManager
        assert self.metrics_manager is not None
        return self.metrics_manager.get_val_metrics(
            include_global=self.config.logging.log_global_metrics,
            include_per_loader=self.config.logging.log_per_loader_metrics,
        )

    def _training_step(
        self,
        batch: Any,
        loader_idx: int,
        loader_name: str,
        accumulation_step: int,
        batch_idx: int,
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
                assert self.training_step_fn is not None
                step_metrics = self.training_step_fn(
                    self, batch, batch_idx, loader_idx, loader_name
                )
        else:
            assert self.training_step_fn is not None
            step_metrics = self.training_step_fn(
                self, batch, batch_idx, loader_idx, loader_name
            )

        loss = step_metrics["loss"]
        # Coerce numeric losses to tensors and ensure requires_grad for backward
        if not isinstance(loss, torch.Tensor):
            try:
                device = next(self.model.parameters()).device
            except Exception:
                device = None
            loss = torch.tensor(float(loss), device=device, requires_grad=True)
            step_metrics["loss"] = loss
        elif not loss.requires_grad:
            loss = loss.clone().detach().requires_grad_(True)
            step_metrics["loss"] = loss

        # Scale loss for gradient accumulation
        loss = loss / self.config.performance.gradient_accumulation_steps

        # Call on_before_backward hook
        self.hook_manager.call_hook("on_before_backward", self, loss)

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

        # Call on_after_backward hook
        self.hook_manager.call_hook("on_after_backward", self)

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

        # Call on_before_optimizer_step hook
        self.hook_manager.call_hook("on_before_optimizer_step", self, optimizer_idx)

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
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.performance.clip_grad_norm,
                )
                # Call on_gradient_clip hook
                self.hook_manager.call_hook("on_gradient_clip", self, grad_norm)

            # Optimizer step with scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Gradient clipping without scaler
            if self.config.performance.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.performance.clip_grad_norm,
                )
                # Call on_gradient_clip hook
                self.hook_manager.call_hook("on_gradient_clip", self, grad_norm)

            # Regular optimizer step
            optimizer.step()

        optimizer.zero_grad()
        self.global_step += 1

        # Call on_after_optimizer_step hook
        self.hook_manager.call_hook("on_after_optimizer_step", self, optimizer_idx)

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

        # Optional step-based checkpoint save (only on primary rank)
        try:
            if self.checkpoint_manager.should_save_checkpoint(
                self.current_epoch, self.global_step
            ):
                if ddp_is_primary(self.fabric):
                    self._save_checkpoint()
                # All ranks wait for checkpoint save
                ddp_barrier(self.fabric)
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
        """Deprecated: superseded by MetricsManager; retained for compatibility."""
        return {}

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
                assert self.dataloader_manager is not None
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

                all_keys: set[str] = set()
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
                metric_keys: set[str] = set()
                for metrics in per_loader_metrics.values():
                    metric_keys.update(metrics.keys())

                for key in metric_keys:
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
            # Ensure resume_state is a proper ResumeState object (not a raw dict)
            try:
                if (
                    isinstance(self.resume_state, dict)
                    and _ResumeStateRuntime is not None
                ):
                    self.resume_state = _ResumeStateRuntime.from_dict(self.resume_state)
            except Exception:
                logger.debug(
                    "resume_state conversion prior to checkpoint save failed",
                    exc_info=True,
                )

            # Update resume state with dataloader manager state
            if self.dataloader_manager:
                self.resume_state = update_resume_state(
                    self.resume_state,
                    cast("Any", self.resume_state).phase,
                    dataloader_manager_state=self.dataloader_manager.get_state(),
                    save_rng=self.config.checkpoint.save_rng,
                    choice_rng=getattr(self.dataloader_manager, "choice_rng", None),
                )

            # Save checkpoint with all optimizers and schedulers
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
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

            # Call on_checkpoint_save hook
            if checkpoint_path:
                self.hook_manager.call_hook(
                    "on_checkpoint_save", self, str(checkpoint_path)
                )
        except Exception:
            logger.exception("Failed to save checkpoint: ")
            if force:
                raise

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        # Synchronize before loading checkpoint
        ddp_barrier(self.fabric)

        try:
            # Call on_checkpoint_load hook (before loading)
            self.hook_manager.call_hook("on_checkpoint_load", self, checkpoint_path)

            # Load checkpoint only on rank 0 and broadcast to other ranks
            if ddp_is_primary(self.fabric):
                # Load checkpoint data on rank 0
                # Map to CPU to avoid device-coupled payloads and reduce GPU memory spikes
                payload = self.checkpoint_manager.load_checkpoint(
                    checkpoint_path, map_location="cpu"
                )

                # Extract necessary data for broadcast
                optimizer_states = payload.optimizer_state_dicts or []
                scheduler_states = payload.scheduler_state_dicts or []

                broadcast_data = {
                    "epoch": payload.epoch,
                    "global_step": payload.global_step,
                    "resume_state": payload.resume_state,
                    "model_state_dict": payload.model_state_dict,
                    "optimizer_state_dicts": optimizer_states,
                    "scheduler_state_dicts": scheduler_states,
                    "amp_scaler_state": payload.amp_scaler_state,
                    # For legacy compatibility; new payloads use resume_state only
                    "dataloader_manager_state": payload.dataloader_manager_state,
                    "choice_rng_state": payload.choice_rng_state,
                }
            else:
                broadcast_data = None

            # Broadcast checkpoint data from rank 0 to all ranks
            broadcast_data = ddp_broadcast_object(self.fabric, broadcast_data, src=0)

            # All ranks apply the checkpoint state
            if broadcast_data:
                # Restore basic training state
                self.current_epoch = broadcast_data["epoch"]
                self.global_step = broadcast_data["global_step"]

                # Restore model state
                self.model.load_state_dict(broadcast_data["model_state_dict"])

                # Restore optimizer states
                for i, opt_state in enumerate(broadcast_data["optimizer_state_dicts"]):
                    if i < len(self.optimizers):
                        self.optimizers[i].load_state_dict(opt_state)

                # Restore scheduler states
                for i, sched_state in enumerate(
                    broadcast_data["scheduler_state_dicts"]
                ):
                    if i < len(self.schedulers):
                        self.schedulers[i].load_state_dict(sched_state)

                # Restore AMP scaler state
                if broadcast_data["amp_scaler_state"] and self.scaler:
                    self.scaler.load_state_dict(broadcast_data["amp_scaler_state"])

                # Restore resume state
                if broadcast_data["resume_state"] is not None:
                    rs = broadcast_data["resume_state"]
                    try:
                        if isinstance(rs, dict) and _ResumeStateRuntime is not None:
                            self.resume_state = _ResumeStateRuntime.from_dict(rs)
                        else:
                            self.resume_state = rs
                    except Exception:
                        self.resume_state = rs

                    # Restore RNG state if available
                    try:
                        if cast("Any", self.resume_state).rng is not None:
                            restore_rng_state(cast("Any", self.resume_state).rng)
                    except Exception:
                        logger.debug("restore_rng_state failed", exc_info=True)

                # Restore DataLoaderManager state if available
                # Check both top-level (free save function) and resume_state (CheckpointManager)
                # Pass skip_broadcast=True since we already broadcast the entire checkpoint
                if (
                    broadcast_data["dataloader_manager_state"]
                    and self.dataloader_manager
                ):
                    # Top-level state (from free save function)
                    assert self.dataloader_manager is not None
                    self.dataloader_manager.load_state(
                        broadcast_data["dataloader_manager_state"], skip_broadcast=True
                    )
                elif (
                    self.resume_state
                    and cast("Any", self.resume_state).dataloader_manager_state
                    and self.dataloader_manager
                ):
                    # State inside resume_state (from CheckpointManager)
                    assert self.dataloader_manager is not None
                    try:
                        state = cast("Any", self.resume_state).dataloader_manager_state
                        if state is not None:
                            self.dataloader_manager.load_state(
                                state, skip_broadcast=True
                            )
                    except Exception:
                        logger.debug(
                            "Failed to load dataloader_manager state from resume_state",
                            exc_info=True,
                        )

                # Restore choice RNG if available
                # Check both top-level and resume_state locations
                if broadcast_data["choice_rng_state"] and hasattr(
                    self.dataloader_manager, "choice_rng"
                ):
                    assert self.dataloader_manager is not None
                    self.dataloader_manager.choice_rng.set_state(
                        broadcast_data["choice_rng_state"]
                    )
                elif (
                    self.resume_state
                    and cast("Any", self.resume_state).choice_rng
                    and hasattr(self.dataloader_manager, "choice_rng")
                ):
                    # Use the helper function for resume_state.choice_rng
                    assert self.dataloader_manager is not None
                    assert hasattr(self.dataloader_manager, "choice_rng")
                    try:
                        restore_choice_rng_state(
                            cast("Any", self.resume_state).choice_rng,
                            self.dataloader_manager.choice_rng,
                        )
                    except Exception:
                        logger.debug(
                            "Failed to restore choice RNG from resume_state",
                            exc_info=True,
                        )

                logger.info(
                    f"Resumed from checkpoint: epoch={self.current_epoch}, step={self.global_step}"
                )

            # Synchronize after loading checkpoint
            ddp_barrier(self.fabric)

        except Exception:
            logger.exception("Failed to resume from checkpoint: ")
            raise

    def _log_step_metrics(
        self, metrics: dict[str, Any], step: int, prefix: str = ""
    ) -> None:
        """Log step metrics (for backward compatibility with tests)."""
        # Only log on primary rank
        if not ddp_is_primary(self.fabric):
            return

        # If wandb_run exists (for test compatibility), use it
        wandb_run = getattr(self, "wandb_run", None)
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        # Otherwise use the logger if available
        elif self.logger:
            self.logger.log_metrics(metrics, step)

    def _log_epoch_metrics(
        self, epoch: int, metrics: dict[str, Any], prefix: str = ""
    ) -> None:
        """Log epoch metrics (for backward compatibility with tests)."""
        # Only log on primary rank
        if not ddp_is_primary(self.fabric):
            return

        # If wandb_run exists (for test compatibility), use it
        wandb_run = getattr(self, "wandb_run", None)
        if wandb_run is not None:
            wandb_run.log(metrics, epoch=epoch)
        # Otherwise use the logger if available
        elif self.logger:
            self.logger.log_epoch_summary(epoch, metrics)

    def get_training_state(self) -> dict[str, Any]:
        """Get current training state."""
        return {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "resume_state": self.resume_state,
            "performance_summary": self.performance_monitor.get_performance_summary(),
        }
