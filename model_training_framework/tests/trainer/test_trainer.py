"""
Tests for training engine component.

This module tests all aspects of the training system including:
- GenericTrainer functionality
- Checkpoint management
- Training state management
- Preemption handling
"""

import signal
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from model_training_framework.trainer import (
    CheckpointConfig,
    GenericTrainer,
    GenericTrainerConfig,
    LoggingConfig,
    PerformanceConfig,
    PreemptionConfig,
    ResumeState,
    RNGState,
    TrainerPhase,
    TrainMicroState,
)
from model_training_framework.trainer.checkpoints import CheckpointManager
from model_training_framework.trainer.states import (
    capture_rng_state,
    create_initial_resume_state,
    is_training_phase,
    is_validation_phase,
    restore_rng_state,
    update_resume_state,
)
from model_training_framework.trainer.utils import (
    EarlyStopping,
    PerformanceMonitor,
    SignalHandler,
    timeout,
)


class TestTrainerPhases:
    """Test trainer phase enumeration and utilities."""

    def test_training_phases(self):
        """Test training phase identification."""
        training_phases = [
            TrainerPhase.TRAIN_START_EPOCH,
            TrainerPhase.TRAIN_BATCH_LOAD,
            TrainerPhase.TRAIN_BATCH_FORWARD,
            TrainerPhase.TRAIN_BATCH_BACKWARD,
            TrainerPhase.TRAIN_BATCH_OPTIM_STEP,
            TrainerPhase.TRAIN_BATCH_END,
        ]

        for phase in training_phases:
            assert is_training_phase(phase)
            assert not is_validation_phase(phase)

    def test_validation_phases(self):
        """Test validation phase identification."""
        validation_phases = [
            TrainerPhase.VAL_START_EPOCH,
            TrainerPhase.VAL_BATCH_LOAD,
            TrainerPhase.VAL_BATCH_FORWARD,
            TrainerPhase.VAL_BATCH_END,
        ]

        for phase in validation_phases:
            assert is_validation_phase(phase)
            assert not is_training_phase(phase)


class TestRNGState:
    """Test RNG state management."""

    def test_capture_rng_state(self):
        """Test capturing RNG state."""
        rng_state = capture_rng_state()

        assert isinstance(rng_state, RNGState)
        assert isinstance(rng_state.torch_state, bytes)
        assert isinstance(rng_state.numpy_state, bytes)
        assert isinstance(rng_state.python_state, tuple)

    def test_rng_state_determinism(self):
        """Test that RNG state capture enables deterministic behavior."""
        import random

        import numpy as np
        import torch

        # Set initial seed
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Capture state
        rng_state = capture_rng_state()

        # Generate some random numbers
        torch_val1 = torch.rand(1).item()
        np_val1 = np.random.rand()
        py_val1 = random.random()

        # Restore state and generate again
        restore_rng_state(rng_state)
        torch_val2 = torch.rand(1).item()
        np_val2 = np.random.rand()
        py_val2 = random.random()

        # Values should be identical
        assert torch_val1 == torch_val2
        assert np_val1 == np_val2
        assert py_val1 == py_val2


class TestResumeState:
    """Test resume state management."""

    def test_create_initial_resume_state(self):
        """Test creating initial resume state."""
        resume_state = create_initial_resume_state(save_rng=True)

        assert isinstance(resume_state, ResumeState)
        assert resume_state.phase == TrainerPhase.INIT
        assert resume_state.epoch == 0
        assert resume_state.global_step == 0
        assert resume_state.rng is not None

    def test_update_resume_state(self):
        """Test updating resume state."""
        initial_state = create_initial_resume_state(save_rng=False)

        train_state = TrainMicroState(batch_idx=5, micro_step=2, loss_sum=1.5)

        updated_state = update_resume_state(
            initial_state,
            phase=TrainerPhase.TRAIN_BATCH_FORWARD,
            epoch=2,
            global_step=100,
            train_state=train_state,
            save_rng=False,
        )

        assert updated_state.phase == TrainerPhase.TRAIN_BATCH_FORWARD
        assert updated_state.epoch == 2
        assert updated_state.global_step == 100
        assert updated_state.train is not None
        assert updated_state.train.batch_idx == 5
        assert updated_state.train.micro_step == 2


class TestCheckpointConfig:
    """Test checkpoint configuration."""

    def test_checkpoint_config_creation(self, tmp_path):
        """Test checkpoint configuration creation."""
        config = CheckpointConfig(
            root_dir=tmp_path,
            save_every_n_epochs=5,
            max_checkpoints=10,
            save_rng=True,
        )

        assert str(config.root_dir) == str(tmp_path)
        assert config.save_every_n_epochs == 5
        assert config.max_checkpoints == 10
        assert config.save_rng

    def test_checkpoint_config_defaults(self):
        """Test checkpoint configuration defaults."""
        config = CheckpointConfig()

        assert config.max_checkpoints == 5
        assert config.save_rng
        assert config.save_optimizer


class TestCheckpointManager:
    """Test checkpoint management."""

    def test_checkpoint_manager_creation(self):
        """Test checkpoint manager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CheckpointConfig(root_dir=temp_dir)
            manager = CheckpointManager(config, "test_experiment")

            assert manager.experiment_name == "test_experiment"
            assert manager.checkpoint_dir.exists()

    def test_should_save_checkpoint(self):
        """Test checkpoint save scheduling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CheckpointConfig(
                root_dir=temp_dir, save_every_n_epochs=2, save_every_n_steps=100
            )
            manager = CheckpointManager(config, "test_experiment")

            # Should save on epoch boundaries
            assert manager.should_save_checkpoint(epoch=2, global_step=50)
            assert not manager.should_save_checkpoint(epoch=1, global_step=50)

            # Should save on step boundaries
            assert manager.should_save_checkpoint(epoch=1, global_step=100)
            assert not manager.should_save_checkpoint(epoch=1, global_step=99)

            # Should save when forced
            assert manager.should_save_checkpoint(epoch=1, global_step=50, force=True)

    @patch("torch.save")
    def test_save_checkpoint(self, mock_torch_save):
        """Test checkpoint saving."""

        def create_checkpoint_file(checkpoint, path):
            """Mock torch.save that creates an actual file."""
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        mock_torch_save.side_effect = create_checkpoint_file

        with tempfile.TemporaryDirectory() as temp_dir:
            config = CheckpointConfig(root_dir=temp_dir)
            manager = CheckpointManager(config, "test_experiment")

            # Mock objects
            mock_model = Mock()
            mock_model.state_dict.return_value = {"layer.weight": "mock_weight"}

            mock_optimizer = Mock()
            mock_optimizer.state_dict.return_value = {"param_groups": []}

            resume_state = ResumeState(
                phase=TrainerPhase.TRAIN_BATCH_END, epoch=1, global_step=100
            )

            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                resume_state=resume_state,
                epoch=1,
                global_step=100,
            )

            assert checkpoint_path.exists()
            mock_torch_save.assert_called_once()


class TestGenericTrainerConfig:
    """Test trainer configuration."""

    def test_trainer_config_creation(self):
        """Test trainer configuration creation."""
        config = GenericTrainerConfig(
            checkpoint=CheckpointConfig(max_checkpoints=3),
            preemption=PreemptionConfig(signal=signal.SIGUSR1),
            performance=PerformanceConfig(gradient_accumulation_steps=4),
            logging=LoggingConfig(use_wandb=False),
        )

        assert config.checkpoint.max_checkpoints == 3
        assert config.preemption.signal == signal.SIGUSR1
        assert config.performance.gradient_accumulation_steps == 4
        assert not config.logging.use_wandb

    def test_trainer_config_validation(self):
        """Test trainer configuration validation."""
        # Valid config should not raise
        config = GenericTrainerConfig()
        config.__post_init__()  # Should not raise

        # Invalid gradient accumulation should raise
        with pytest.raises(ValueError):
            config = GenericTrainerConfig(
                performance=PerformanceConfig(gradient_accumulation_steps=0)
            )
            config.__post_init__()


class TestSignalHandler:
    """Test signal handling functionality."""

    def test_signal_handler_creation(self):
        """Test signal handler creation."""
        handler = SignalHandler()

        assert not handler.preemption_requested
        assert len(handler.original_handlers) == 0
        assert len(handler.callbacks) == 0

    def test_signal_registration(self):
        """Test signal handler registration."""
        handler = SignalHandler()

        # Register handler (use SIGUSR2 to avoid conflicts)
        handler.register_preemption_handler(signal.SIGUSR2)

        assert signal.SIGUSR2 in handler.original_handlers

        # Clean up
        handler.restore_handlers()

    def test_preemption_flag(self):
        """Test preemption flag management."""
        handler = SignalHandler()

        assert not handler.is_preemption_requested()

        handler.preemption_requested = True
        assert handler.is_preemption_requested()

        handler.reset_preemption_flag()
        assert not handler.is_preemption_requested()


class TestPerformanceMonitor:
    """Test performance monitoring."""

    def test_performance_monitor_creation(self):
        """Test performance monitor creation."""
        monitor = PerformanceMonitor()

        assert len(monitor.step_times) == 0
        assert len(monitor.memory_usage) == 0
        assert monitor.start_time > 0

    def test_step_time_recording(self):
        """Test step time recording."""
        monitor = PerformanceMonitor()

        # Record a few steps
        time.sleep(0.01)
        duration1 = monitor.record_step_time()

        time.sleep(0.01)
        duration2 = monitor.record_step_time()

        assert len(monitor.step_times) == 2
        assert duration1 > 0
        assert duration2 > 0

        # Test averages
        avg_time = monitor.get_average_step_time()
        assert avg_time > 0

        steps_per_sec = monitor.get_steps_per_second()
        assert steps_per_sec > 0

    def test_performance_summary(self):
        """Test performance summary."""
        monitor = PerformanceMonitor()

        # Record some steps
        monitor.record_step_time()
        monitor.record_step_time()

        summary = monitor.get_performance_summary()

        assert "total_steps" in summary
        assert "total_time_sec" in summary
        assert "avg_step_time_sec" in summary
        assert "steps_per_sec" in summary
        assert summary["total_steps"] == 2


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_creation(self):
        """Test early stopping creation."""
        early_stopping = EarlyStopping(patience=5, metric_name="val_loss", mode="min")

        assert early_stopping.patience == 5
        assert early_stopping.metric_name == "val_loss"
        assert early_stopping.mode == "min"
        assert early_stopping.best_value is None
        assert not early_stopping.should_stop

    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        early_stopping = EarlyStopping(patience=2, metric_name="val_loss", mode="min")

        # First metric (should not stop)
        should_stop = early_stopping({"val_loss": 1.0})
        assert not should_stop
        assert early_stopping.best_value == 1.0

        # Improvement (should not stop)
        should_stop = early_stopping({"val_loss": 0.8})
        assert not should_stop
        assert early_stopping.best_value == 0.8
        assert early_stopping.epochs_without_improvement == 0

    def test_early_stopping_trigger(self):
        """Test early stopping trigger."""
        early_stopping = EarlyStopping(patience=2, metric_name="val_loss", mode="min")

        # Initial values
        early_stopping({"val_loss": 1.0})
        early_stopping({"val_loss": 1.1})  # No improvement

        assert early_stopping.epochs_without_improvement == 1
        assert not early_stopping.should_stop

        # Should trigger stopping
        should_stop = early_stopping({"val_loss": 1.2})
        assert should_stop
        assert early_stopping.should_stop


class TestTimeoutUtility:
    """Test timeout utility function."""

    def test_timeout_success(self):
        """Test timeout with operation that completes in time."""

        def quick_operation():
            time.sleep(0.01)
            return "success"

        with timeout(1.0):
            result = quick_operation()

        assert result == "success"

    def test_timeout_failure(self):
        """Test timeout with operation that exceeds limit."""

        def slow_operation():
            time.sleep(2.0)
            return "should not reach"

        with (
            pytest.raises(TimeoutError),
            timeout(1.0),
        ):  # Use 1 second timeout (minimum for signal.alarm)
            slow_operation()


class MockDataLoader:
    """Mock DataLoader for testing."""

    def __init__(self, data_size=10):
        self.data_size = data_size

    def __iter__(self):
        for _i in range(self.data_size):
            # Mock batch with tensors
            import torch

            yield (torch.randn(2, 10), torch.randn(2, 1))


class TestGenericTrainer:
    """Test GenericTrainer functionality."""

    def test_trainer_creation(self, mock_fabric, mock_model, mock_optimizer):
        """Test trainer creation."""
        config = GenericTrainerConfig()

        trainer = GenericTrainer(
            config=config,
            fabric=mock_fabric,
            model=mock_model,
            optimizer=mock_optimizer,
        )

        assert trainer.config == config
        assert trainer.fabric == mock_fabric
        assert trainer.model == mock_model
        assert trainer.optimizer == mock_optimizer
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_set_training_step(self, mock_fabric, mock_model, mock_optimizer):
        """Test setting training step function."""
        config = GenericTrainerConfig()
        trainer = GenericTrainer(config, mock_fabric, mock_model, mock_optimizer)

        def dummy_training_step(trainer, batch, micro_step):
            return {"loss": 0.5}

        trainer.set_training_step(dummy_training_step)
        assert trainer.training_step_fn == dummy_training_step

    def test_set_validation_step(self, mock_fabric, mock_model, mock_optimizer):
        """Test setting validation step function."""
        config = GenericTrainerConfig()
        trainer = GenericTrainer(config, mock_fabric, mock_model, mock_optimizer)

        def dummy_validation_step(trainer, batch, batch_idx):
            return {"loss": 0.3}

        trainer.set_validation_step(dummy_validation_step)
        assert trainer.validation_step_fn == dummy_validation_step

    @patch("torch.nn.utils.clip_grad_norm_")
    def test_training_step_execution(
        self, mock_clip_grad, mock_fabric, mock_model, mock_optimizer
    ):
        """Test training step execution."""
        config = GenericTrainerConfig()
        trainer = GenericTrainer(config, mock_fabric, mock_model, mock_optimizer)

        # Mock training step function
        def mock_training_step(trainer, batch, micro_step):
            import torch

            return {"loss": torch.tensor(0.5, requires_grad=True)}

        trainer.set_training_step(mock_training_step)

        # Mock batch
        import torch

        batch = (torch.randn(2, 10), torch.randn(2, 1))

        # Execute training step
        metrics = trainer._training_step(batch, 0)

        assert "loss" in metrics
        assert trainer.global_step == 1
        # Can't assert on MockOptimizer calls since it's not a Mock object
        # Just verify it runs without error

    def test_get_training_state(self, mock_fabric, mock_model, mock_optimizer):
        """Test getting training state."""
        config = GenericTrainerConfig()
        trainer = GenericTrainer(config, mock_fabric, mock_model, mock_optimizer)

        state = trainer.get_training_state()

        assert "current_epoch" in state
        assert "global_step" in state
        assert "resume_state" in state
        assert "performance_summary" in state

        assert state["current_epoch"] == 0
        assert state["global_step"] == 0
