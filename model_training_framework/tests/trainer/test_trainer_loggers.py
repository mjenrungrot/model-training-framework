"""
Tests for Logger Protocol and Implementations

This module tests the logging functionality including:
- LoggerProtocol interface compliance
- WandB, TensorBoard, and Console logger implementations
- Composite logger functionality
- Metrics formatting and logging
"""

import importlib.util
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from model_training_framework.trainer.loggers import (
    CompositeLogger,
    ConsoleLogger,
    TensorBoardLogger,
    WandBLogger,
    create_logger,
)

# Check if wandb is available using importlib
WANDB_AVAILABLE = importlib.util.find_spec("wandb") is not None


class TestConsoleLogger:
    """Test console logger implementation."""

    def test_console_logger_creation(self):
        """Test console logger can be created."""
        logger = ConsoleLogger(log_level="INFO")
        assert logger is not None

    def test_console_logger_metrics(self, caplog):
        """Test console logger logs metrics."""
        logger = ConsoleLogger(log_level="INFO")

        metrics = {
            "train/loss": 0.5,
            "train/accuracy": 0.95,
            "train/dl_loader1/loss": 0.45,
            "train/dl_loader2/loss": 0.55,
        }

        logger.log_metrics(metrics, step=100)

        # Check that something was logged
        assert len(caplog.records) > 0
        assert "Step 100" in caplog.text

    def test_console_logger_epoch_summary(self, caplog):
        """Test console logger logs epoch summary."""
        logger = ConsoleLogger(log_level="INFO")

        summary = {
            "loss": 0.3,
            "accuracy": 0.97,
            "learning_rate": 0.001,
        }

        logger.log_epoch_summary(epoch=5, summary=summary)

        assert "Epoch 5 Summary" in caplog.text

    def test_console_logger_proportions(self, caplog):
        """Test console logger logs loader proportions."""
        logger = ConsoleLogger(log_level="INFO")

        proportions = {
            "loader1": 0.6,
            "loader2": 0.4,
        }
        counts = {
            "loader1": 600,
            "loader2": 400,
        }

        logger.log_loader_proportions(epoch=3, proportions=proportions, counts=counts)

        assert "Loader Proportions" in caplog.text
        assert "Loader Counts" in caplog.text

    def test_console_logger_text(self, caplog):
        """Test console logger logs text."""
        logger = ConsoleLogger(log_level="INFO")

        logger.log_text("model_summary", "Model has 1M parameters", step=50)

        assert "model_summary" in caplog.text
        assert "Model has 1M parameters" in caplog.text
        assert "Step 50" in caplog.text

    def test_console_logger_matplotlib_figure(self, caplog):
        """Test console logger logs matplotlib figure info."""
        logger = ConsoleLogger(log_level="INFO")

        # Create a mock figure
        from unittest.mock import MagicMock

        mock_figure = MagicMock()
        mock_figure.get_size_inches.return_value = (8.0, 6.0)
        mock_figure._suptitle = MagicMock()
        mock_figure._suptitle.get_text.return_value = "Test Figure"

        logger.log_matplotlib_figure("test_plot", mock_figure, step=100)

        assert "Figure 'test_plot'" in caplog.text
        assert "title='Test Figure'" in caplog.text
        assert "8.0x6.0" in caplog.text
        assert "Step 100" in caplog.text

    def test_console_logger_image(self, caplog):
        """Test console logger logs image tensor info."""
        logger = ConsoleLogger(log_level="INFO")

        # Test 2D tensor (HW)
        image_2d = torch.randn(100, 100)
        logger.log_image("test_image_2d", image_2d, step=50)

        assert "Image 'test_image_2d'" in caplog.text
        assert "shape=100x100" in caplog.text
        assert "Step 50" in caplog.text

        # Test 3D tensor (HWC)
        image_3d = torch.randn(100, 100, 3)
        logger.log_image("test_image_3d", image_3d, step=60, channels_last=True)

        assert "Image 'test_image_3d'" in caplog.text
        assert "shape=100x100x3" in caplog.text


class TestTensorBoardLogger:
    """Test TensorBoard logger implementation."""

    def test_tensorboard_logger_creation(self):
        """Test TensorBoard logger can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            assert logger is not None
            assert logger.log_dir.exists()
            logger.close()

    def test_tensorboard_logger_metrics(self):
        """Test TensorBoard logger logs metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            metrics = {
                "train/loss": torch.tensor(0.5),
                "train/accuracy": 0.95,
            }

            logger.log_metrics(metrics, step=100)

            # Verify writer was called (would need to mock in real test)
            assert logger.writer is not None

            logger.close()

    def test_tensorboard_logger_epoch_summary(self):
        """Test TensorBoard logger logs epoch summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            summary = {"loss": 0.3, "accuracy": 0.97}
            logger.log_epoch_summary(epoch=5, summary=summary)

            logger.close()

    def test_tensorboard_logger_proportions(self):
        """Test TensorBoard logger logs proportions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            proportions = {"loader1": 0.6, "loader2": 0.4}
            counts = {"loader1": 600, "loader2": 400}

            logger.log_loader_proportions(
                epoch=3, proportions=proportions, counts=counts
            )

            logger.close()

    def test_tensorboard_logger_matplotlib_figure(self):
        """Test TensorBoard logger logs matplotlib figures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            # Create a mock figure
            from unittest.mock import MagicMock

            mock_figure = MagicMock()

            # Mock the writer's add_figure method
            with patch.object(logger.writer, "add_figure") as mock_add_figure:
                logger.log_matplotlib_figure("test_plot", mock_figure, step=100)
                mock_add_figure.assert_called_once_with(
                    "test_plot", mock_figure, 100, close=False
                )

            logger.close()

    def test_tensorboard_logger_image(self):
        """Test TensorBoard logger logs image tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            # Test 2D tensor (HW)
            image_2d = torch.randn(100, 100)

            with patch.object(logger.writer, "add_image") as mock_add_image:
                logger.log_image("test_image_2d", image_2d, step=50)
                # Should add channel dimension and call add_image
                assert mock_add_image.called
                call_args = mock_add_image.call_args
                assert call_args[0][0] == "test_image_2d"
                assert call_args[0][2] == 50
                assert call_args[1]["dataformats"] == "CHW"

            # Test 3D tensor (HWC)
            image_3d = torch.randn(100, 100, 3)

            with patch.object(logger.writer, "add_image") as mock_add_image:
                logger.log_image("test_image_3d", image_3d, step=60, channels_last=True)
                assert mock_add_image.called
                call_args = mock_add_image.call_args
                # Should have been converted to CHW
                logged_image = call_args[0][1]
                assert logged_image.shape == (3, 100, 100)

            logger.close()


@pytest.mark.skipif(not WANDB_AVAILABLE, reason="wandb not installed")
class TestWandBLogger:
    """Test WandB logger implementation."""

    @patch("model_training_framework.trainer.loggers.wandb_mod")
    def test_wandb_logger_creation(self, mock_wandb):
        """Test WandB logger can be created."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(project="test_project", entity="test_entity")

        assert logger is not None
        mock_wandb.init.assert_called_once_with(
            project="test_project",
            entity="test_entity",
            name=None,
            config=None,
        )

    @patch("model_training_framework.trainer.loggers.wandb_mod")
    def test_wandb_logger_metrics(self, mock_wandb):
        """Test WandB logger logs metrics."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(project="test_project")

        metrics = {
            "train/loss": torch.tensor(0.5),
            "train/accuracy": 0.95,
        }

        logger.log_metrics(metrics, step=100)

        mock_run.log.assert_called_once()
        call_args = mock_run.log.call_args
        assert call_args[1]["step"] == 100
        assert "train/loss" in call_args[0][0]
        assert "train/accuracy" in call_args[0][0]

    @patch("model_training_framework.trainer.loggers.wandb_mod")
    def test_wandb_logger_epoch_summary(self, mock_wandb):
        """Test WandB logger logs epoch summary."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(project="test_project")

        summary = {"loss": 0.3, "accuracy": 0.97}
        logger.log_epoch_summary(epoch=5, summary=summary)

        mock_run.log.assert_called_once()
        call_args = mock_run.log.call_args[0][0]
        assert "epoch" in call_args
        assert call_args["epoch"] == 5

    @patch("model_training_framework.trainer.loggers.wandb_mod")
    def test_wandb_logger_close(self, mock_wandb):
        """Test WandB logger closes properly."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(project="test_project")
        logger.close()

        mock_run.finish.assert_called_once()

    @patch("model_training_framework.trainer.loggers.wandb_mod")
    @patch("model_training_framework.trainer.loggers.Image")
    def test_wandb_logger_matplotlib_figure(self, mock_pil_image, mock_wandb):
        """Test WandB logger logs matplotlib figures."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Image = MagicMock(return_value="mock_image")

        # Mock PIL Image.open
        mock_pil_image.open.return_value = "mock_pil_image"

        logger = WandBLogger(project="test_project")

        # Create a mock figure
        mock_figure = MagicMock()
        mock_figure.savefig = MagicMock()

        logger.log_matplotlib_figure("test_plot", mock_figure, step=100, dpi=100)

        # Check that savefig was called with correct parameters
        mock_figure.savefig.assert_called_once()
        call_args = mock_figure.savefig.call_args
        assert call_args[1]["format"] == "jpeg"  # Default is jpg
        assert call_args[1]["dpi"] == 100
        assert call_args[1]["bbox_inches"] == "tight"

        # Check that PIL Image.open was called
        mock_pil_image.open.assert_called_once()

        # Check that wandb.Image was called with the PIL image
        mock_wandb.Image.assert_called_with("mock_pil_image")

        # Check that log was called
        mock_run.log.assert_called_once()
        assert mock_run.log.call_args[1]["step"] == 100

    @patch("model_training_framework.trainer.loggers.wandb_mod")
    def test_wandb_logger_image(self, mock_wandb):
        """Test WandB logger logs image tensors."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Image = MagicMock(return_value="mock_image")

        logger = WandBLogger(project="test_project")

        # Test 2D tensor (HW)
        image_2d = torch.randn(100, 100)
        logger.log_image("test_image_2d", image_2d, step=50)

        # Check Image was created and logged
        assert mock_wandb.Image.called
        mock_run.log.assert_called()

        # Test 3D tensor (HWC)
        image_3d = torch.randn(100, 100, 3).clamp(0, 1)
        logger.log_image("test_image_3d", image_3d, step=60, channels_last=True)

        # Check that image was processed correctly
        assert mock_wandb.Image.called
        # Get the numpy array that was passed to wandb.Image
        image_np = mock_wandb.Image.call_args[0][0]
        assert image_np.shape == (3, 100, 100)  # Should be CHW format


class TestCompositeLogger:
    """Test composite logger implementation."""

    def test_composite_logger_creation(self):
        """Test composite logger can be created."""
        console_logger = ConsoleLogger()
        composite = CompositeLogger([console_logger])

        assert composite is not None
        assert len(composite.loggers) == 1

    def test_composite_logger_forwards_metrics(self, caplog):
        """Test composite logger forwards to all loggers."""
        console_logger = ConsoleLogger(log_level="INFO")
        mock_logger = MagicMock()

        composite = CompositeLogger([console_logger, mock_logger])

        metrics = {"train/loss": 0.5}
        composite.log_metrics(metrics, step=100)

        # Check console logger received it
        assert "Step 100" in caplog.text

        # Check mock logger received it
        mock_logger.log_metrics.assert_called_once_with(metrics, 100)

    def test_composite_logger_handles_errors(self, caplog):
        """Test composite logger handles errors gracefully."""
        console_logger = ConsoleLogger(log_level="INFO")
        failing_logger = MagicMock()
        failing_logger.log_metrics.side_effect = Exception("Test error")

        composite = CompositeLogger([console_logger, failing_logger])

        metrics = {"train/loss": 0.5}
        # Should not raise even though one logger fails
        composite.log_metrics(metrics, step=100)

        # Console logger should still work
        assert "Step 100" in caplog.text

    def test_composite_logger_close(self):
        """Test composite logger closes all loggers."""
        mock_logger1 = MagicMock()
        mock_logger2 = MagicMock()

        composite = CompositeLogger([mock_logger1, mock_logger2])
        composite.close()

        mock_logger1.close.assert_called_once()
        mock_logger2.close.assert_called_once()

    def test_composite_logger_matplotlib_figure(self):
        """Test composite logger forwards matplotlib figures to all loggers."""
        mock_logger1 = MagicMock()
        mock_logger2 = MagicMock()

        composite = CompositeLogger([mock_logger1, mock_logger2])

        # Create a mock figure
        mock_figure = MagicMock()

        composite.log_matplotlib_figure("test_plot", mock_figure, step=100, dpi=100)

        # Check both loggers received the call
        mock_logger1.log_matplotlib_figure.assert_called_once_with(
            "test_plot", mock_figure, 100, 100, "jpg"
        )
        mock_logger2.log_matplotlib_figure.assert_called_once_with(
            "test_plot", mock_figure, 100, 100, "jpg"
        )

    def test_composite_logger_image(self):
        """Test composite logger forwards image tensors to all loggers."""
        mock_logger1 = MagicMock()
        mock_logger2 = MagicMock()

        composite = CompositeLogger([mock_logger1, mock_logger2])

        image = torch.randn(100, 100, 3)
        composite.log_image("test_image", image, step=50, channels_last=True)

        # Check both loggers received the call
        mock_logger1.log_image.assert_called_once_with(
            "test_image", image, 50, True, "jpg"
        )
        mock_logger2.log_image.assert_called_once_with(
            "test_image", image, 50, True, "jpg"
        )


class TestCreateLogger:
    """Test logger factory function."""

    def test_create_console_logger(self):
        """Test creating console logger."""
        logger = create_logger("console")
        assert isinstance(logger, ConsoleLogger)

    def test_create_tensorboard_logger(self):
        """Test creating TensorBoard logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            logger = create_logger("tensorboard", log_dir=Path(tmpdir))
            assert isinstance(logger, TensorBoardLogger)
            logger.close()

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="wandb not installed")
    @patch("model_training_framework.trainer.loggers.wandb_mod")
    def test_create_wandb_logger(self, mock_wandb):
        """Test creating WandB logger."""
        mock_wandb.init.return_value = MagicMock()

        logger = create_logger("wandb", project="test_project")
        assert isinstance(logger, WandBLogger)

    def test_create_composite_logger(self):
        """Test creating composite logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            logger = create_logger("composite", log_dir=Path(tmpdir))
            assert isinstance(logger, CompositeLogger)
            assert len(logger.loggers) >= 1  # At least console
            logger.close()

    def test_create_unknown_logger_raises(self):
        """Test creating unknown logger raises error."""
        with pytest.raises(ValueError, match="Unknown logger type"):
            create_logger("unknown_logger")
