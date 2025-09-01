"""
Logger Protocol and Implementations for Training Framework

This module provides a unified logging interface with multiple backend support:
- WandB (Weights & Biases) integration
- TensorBoard logging
- Console output with structured formatting
- Composite logger for multiple backends simultaneously
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from PIL import Image
import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
# Predeclare optional modules to keep mypy happy when reassigning
wandb_mod: Any | None = None
TB_SummaryWriter: Any = None

try:
    import wandb as _wandb

    wandb_mod = _wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter as TB_SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class LoggerProtocol(Protocol):
    """Protocol defining the interface for all training loggers."""

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """
        Log metrics at a given step.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step
        """
        ...

    def log_epoch_summary(self, epoch: int, summary: dict[str, Any]) -> None:
        """
        Log epoch-level summary information.

        Args:
            epoch: Current epoch number
            summary: Summary statistics for the epoch
        """
        ...

    def log_loader_proportions(
        self, epoch: int, proportions: dict[str, float], counts: dict[str, int]
    ) -> None:
        """
        Log dataloader usage proportions and counts.

        Args:
            epoch: Current epoch number
            proportions: Realized proportions per loader (for WEIGHTED strategy)
            counts: Batch/sample counts per loader
        """
        ...

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        """
        Log text information.

        Args:
            key: Identifier for the text
            text: Text content to log
            step: Optional step number
        """
        ...

    def log_matplotlib_figure(
        self,
        key: str,
        figure: Figure,
        step: int | None = None,
        dpi: int = 100,
        image_format: str = "jpg",
    ) -> None:
        """
        Log a matplotlib figure.

        Args:
            key: Identifier for the figure
            figure: Matplotlib Figure object
            step: Optional step number
            dpi: DPI for saving the figure (default: 100 for space efficiency)
            image_format: Image format to use ("jpg" or "png")
        """
        ...

    def log_image(
        self,
        key: str,
        image: torch.Tensor,
        step: int | None = None,
        channels_last: bool = True,
        image_format: str = "jpg",
    ) -> None:
        """
        Log an image tensor.

        Args:
            key: Identifier for the image
            image: Image tensor (HW, HWC, or CHW format)
            step: Optional step number
            channels_last: If True, assumes HWC format for 3D tensors; if False, assumes CHW
            image_format: Image format to use ("jpg" or "png")
        """
        ...

    def close(self) -> None:
        """Close the logger and clean up resources."""
        ...


class WandBLogger:
    """Weights & Biases logger implementation."""

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            entity: WandB entity/team name
            run_name: Name for this run
            config: Configuration to log
            **kwargs: Additional arguments for wandb.init()
        """
        if not WANDB_AVAILABLE:
            raise ImportError("WandB not installed. Install with: pip install wandb")
        # Ensure instance attributes are typed as Any
        self.wandb: Any = wandb_mod
        assert self.wandb is not None
        self.run: Any = self.wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            **kwargs,
        )

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to WandB."""
        # Convert tensors to Python scalars
        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                log_dict[key] = value.item()
            elif isinstance(value, int | float):
                log_dict[key] = value

        self.run.log(log_dict, step=step)

    def log_epoch_summary(self, epoch: int, summary: dict[str, Any]) -> None:
        """Log epoch summary to WandB."""
        summary_dict = {f"epoch_summary/{k}": v for k, v in summary.items()}
        summary_dict["epoch"] = epoch
        self.run.log(summary_dict)

    def log_loader_proportions(
        self, epoch: int, proportions: dict[str, float], counts: dict[str, int]
    ) -> None:
        """Log dataloader proportions to WandB."""
        log_dict: dict[str, Any] = {"epoch": epoch}

        for loader_name, proportion in proportions.items():
            log_dict[f"loader_proportions/{loader_name}"] = proportion

        for loader_name, count in counts.items():
            log_dict[f"loader_counts/{loader_name}"] = count

        self.run.log(log_dict)

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        """Log text to WandB."""
        log_dict: dict[str, Any] = {key: self.wandb.Html(f"<pre>{text}</pre>")}
        self.run.log(log_dict, step=step)

    def log_matplotlib_figure(
        self,
        key: str,
        figure: Figure,
        step: int | None = None,
        dpi: int = 100,
        image_format: str = "jpg",
    ) -> None:
        """Log matplotlib figure to WandB."""
        with io.BytesIO() as buf:
            # Use jpeg for jpg format
            save_format = (
                "jpeg" if image_format.lower() == "jpg" else image_format.lower()
            )
            figure.savefig(buf, format=save_format, dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            # Load buffer into PIL Image
            pil_image = Image.open(buf)
            image = self.wandb.Image(pil_image)
        log_dict: dict[str, Any] = {key: image}
        self.run.log(log_dict, step=step)

    def log_image(
        self,
        key: str,
        image: torch.Tensor,
        step: int | None = None,
        channels_last: bool = True,
        image_format: str = "jpg",
    ) -> None:
        """Log image tensor to WandB."""
        # Handle different tensor formats
        if image.dim() == 2:
            # HW format - add channel dimension
            image = image.unsqueeze(0)  # CHW
        elif image.dim() == 3:
            # CHW or HWC format
            if channels_last:
                # Assume HWC, convert to CHW
                image = image.permute(2, 0, 1)
            # else: already in CHW format
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {image.dim()}D")

        # Ensure values are in [0, 1] range
        if image.dtype != torch.uint8:
            image = torch.clamp(image, 0, 1)

        # Convert to numpy and log
        image_np = image.detach().cpu().numpy()
        # WandB Image class handles the format internally
        wandb_image = self.wandb.Image(
            image_np, mode="RGB" if image.shape[0] == 3 else None
        )
        log_dict: dict[str, Any] = {key: wandb_image}
        self.run.log(log_dict, step=step)

    def close(self) -> None:
        """Finish the WandB run."""
        self.run.finish()


class TensorBoardLogger:
    """TensorBoard logger implementation."""

    def __init__(self, log_dir: Path | str, comment: str = "", **_: Any):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment suffix for the run
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard not available. Install with: pip install tensorboard"
            )

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if TB_SummaryWriter is None:
            raise ImportError(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
        self.writer = TB_SummaryWriter(str(self.log_dir), comment=comment)

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to TensorBoard."""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(key, value.item(), step)
            elif isinstance(value, int | float):
                self.writer.add_scalar(key, value, step)

    def log_epoch_summary(self, epoch: int, summary: dict[str, Any]) -> None:
        """Log epoch summary to TensorBoard."""
        for key, value in summary.items():
            if isinstance(value, int | float | torch.Tensor):
                self.writer.add_scalar(f"epoch_summary/{key}", value, epoch)

    def log_loader_proportions(
        self, epoch: int, proportions: dict[str, float], counts: dict[str, int]
    ) -> None:
        """Log dataloader proportions to TensorBoard."""
        for loader_name, proportion in proportions.items():
            self.writer.add_scalar(
                f"loader_proportions/{loader_name}", proportion, epoch
            )

        for loader_name, count in counts.items():
            self.writer.add_scalar(f"loader_counts/{loader_name}", count, epoch)

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        """Log text to TensorBoard."""
        global_step = step if step is not None else 0
        self.writer.add_text(key, text, global_step)

    def log_matplotlib_figure(
        self,
        key: str,
        figure: Figure,
        step: int | None = None,
        dpi: int = 100,
        image_format: str = "jpg",
    ) -> None:
        """Log matplotlib figure to TensorBoard."""
        global_step = step if step is not None else 0
        self.writer.add_figure(key, figure, global_step, close=False)

    def log_image(
        self,
        key: str,
        image: torch.Tensor,
        step: int | None = None,
        channels_last: bool = True,
        image_format: str = "jpg",
    ) -> None:
        """Log image tensor to TensorBoard."""
        # Handle different tensor formats
        if image.dim() == 2:
            # HW format - add channel dimension
            image = image.unsqueeze(0)  # CHW
        elif image.dim() == 3:
            # CHW or HWC format
            if channels_last:
                # Assume HWC, convert to CHW
                image = image.permute(2, 0, 1)
            # else: already in CHW format
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {image.dim()}D")

        # Ensure values are in [0, 1] range
        if image.dtype != torch.uint8:
            image = torch.clamp(image, 0, 1)

        global_step = step if step is not None else 0
        # TensorBoard expects CHW format
        self.writer.add_image(key, image, global_step, dataformats="CHW")

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()


class ConsoleLogger:
    """Enhanced console logger with structured output."""

    def __init__(self, log_level: str = "INFO", **_: Any):
        """
        Initialize console logger.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(f"{__name__}.console")
        self.logger.setLevel(getattr(logging, log_level))

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to console."""
        # Group metrics by prefix for cleaner output
        grouped: dict[str, list[str]] = {}
        for key, value in metrics.items():
            parts = key.split("/")
            if len(parts) > 1:
                prefix = parts[0]
                if prefix not in grouped:
                    grouped[prefix] = []
                metric_name = "/".join(parts[1:])
                scalar_value = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
                grouped[prefix].append(f"{metric_name}={scalar_value:.4f}")
            else:
                if key not in grouped:
                    grouped[key] = []
                scalar_value = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
                grouped[key].append(f"{scalar_value:.4f}")

        # Format output
        output_parts = [f"Step {step}"]
        for prefix, values in grouped.items():
            if len(values) == 1 and "=" not in values[0]:
                output_parts.append(f"{prefix}={values[0]}")
            else:
                output_parts.append(f"{prefix}[{', '.join(values)}]")

        self.logger.info(" | ".join(output_parts))

    def log_epoch_summary(self, epoch: int, summary: dict[str, Any]) -> None:
        """Log epoch summary to console."""
        summary_str = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, int | float) else f"{k}={v}"
            for k, v in summary.items()
        )
        self.logger.info(f"Epoch {epoch} Summary: {summary_str}")

    def log_loader_proportions(
        self, epoch: int, proportions: dict[str, float], counts: dict[str, int]
    ) -> None:
        """Log dataloader proportions to console."""
        if proportions:
            prop_str = ", ".join(f"{k}: {v:.1%}" for k, v in proportions.items())
            self.logger.info(f"Epoch {epoch} Loader Proportions: {prop_str}")

        if counts:
            count_str = ", ".join(f"{k}: {v}" for k, v in counts.items())
            self.logger.info(f"Epoch {epoch} Loader Counts: {count_str}")

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        """Log text to console."""
        step_str = f"[Step {step}] " if step is not None else ""
        self.logger.info(f"{step_str}{key}: {text}")

    def log_matplotlib_figure(
        self,
        key: str,
        figure: Figure,
        step: int | None = None,
        dpi: int = 100,
        image_format: str = "jpg",
    ) -> None:
        """Log matplotlib figure info to console."""
        step_str = f"[Step {step}] " if step is not None else ""
        # Just log that a figure was generated - console can't display images
        fig_size = figure.get_size_inches()
        # Try to get figure title if available
        title = (
            figure._suptitle.get_text()
            if hasattr(figure, "_suptitle") and figure._suptitle
            else "untitled"
        )
        self.logger.info(
            f"{step_str}Figure '{key}': title='{title}', size={fig_size[0]:.1f}x{fig_size[1]:.1f} inches"
        )

    def log_image(
        self,
        key: str,
        image: torch.Tensor,
        step: int | None = None,
        channels_last: bool = True,
        image_format: str = "jpg",
    ) -> None:
        """Log image tensor info to console."""
        step_str = f"[Step {step}] " if step is not None else ""
        # Just log image dimensions - console can't display images
        shape_str = "x".join(str(s) for s in image.shape)
        min_val = image.min().item() if image.numel() > 0 else 0
        max_val = image.max().item() if image.numel() > 0 else 0
        self.logger.info(
            f"{step_str}Image '{key}': shape={shape_str}, range=[{min_val:.3f}, {max_val:.3f}]"
        )

    def close(self) -> None:
        """No cleanup needed for console logger."""


class CompositeLogger:
    """Logger that forwards to multiple backend loggers."""

    def __init__(self, loggers: list[LoggerProtocol]):
        """
        Initialize composite logger.

        Args:
            loggers: List of logger instances to forward to
        """
        self.loggers = loggers

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Forward metrics to all loggers."""
        for logger_inst in self.loggers:
            try:
                logger_inst.log_metrics(metrics, step)
            except Exception:
                logger.exception(
                    f"Error in {logger_inst.__class__.__name__}.log_metrics"
                )

    def log_epoch_summary(self, epoch: int, summary: dict[str, Any]) -> None:
        """Forward epoch summary to all loggers."""
        for logger_inst in self.loggers:
            try:
                logger_inst.log_epoch_summary(epoch, summary)
            except Exception:
                logger.exception(
                    f"Error in {logger_inst.__class__.__name__}.log_epoch_summary"
                )

    def log_loader_proportions(
        self, epoch: int, proportions: dict[str, float], counts: dict[str, int]
    ) -> None:
        """Forward loader proportions to all loggers."""
        for logger_inst in self.loggers:
            try:
                logger_inst.log_loader_proportions(epoch, proportions, counts)
            except Exception:
                logger.exception(
                    f"Error in {logger_inst.__class__.__name__}.log_loader_proportions"
                )

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        """Forward text to all loggers."""
        for logger_inst in self.loggers:
            try:
                logger_inst.log_text(key, text, step)
            except Exception:
                logger.exception(f"Error in {logger_inst.__class__.__name__}.log_text")

    def log_matplotlib_figure(
        self,
        key: str,
        figure: Figure,
        step: int | None = None,
        dpi: int = 100,
        image_format: str = "jpg",
    ) -> None:
        """Forward matplotlib figure to all loggers."""
        for logger_inst in self.loggers:
            try:
                logger_inst.log_matplotlib_figure(key, figure, step, dpi, image_format)
            except Exception:
                logger.exception(
                    f"Error in {logger_inst.__class__.__name__}.log_matplotlib_figure"
                )

    def log_image(
        self,
        key: str,
        image: torch.Tensor,
        step: int | None = None,
        channels_last: bool = True,
        image_format: str = "jpg",
    ) -> None:
        """Forward image tensor to all loggers."""
        for logger_inst in self.loggers:
            try:
                logger_inst.log_image(key, image, step, channels_last, image_format)
            except Exception:
                logger.exception(f"Error in {logger_inst.__class__.__name__}.log_image")

    def close(self) -> None:
        """Close all loggers."""
        for logger_inst in self.loggers:
            try:
                logger_inst.close()
            except Exception:
                logger.exception(f"Error closing {logger_inst.__class__.__name__}")


def create_logger(
    logger_type: str,
    project: str | None = None,
    log_dir: Path | None = None,
    loggers_list: list[str] | None = None,
    **kwargs: Any,
) -> LoggerProtocol:
    """
    Factory function to create logger instances.

    Args:
        logger_type: Type of logger ("wandb", "tensorboard", "console", "composite")
        project: Project name for WandB
        log_dir: Directory for TensorBoard logs
        loggers_list: Explicit list of logger types for composite logger
        **kwargs: Additional logger-specific arguments

    Returns:
        Logger instance implementing LoggerProtocol

    Raises:
        ValueError: If logger_type is not recognized
    """
    if logger_type == "wandb":
        if project is None:
            logger.warning(
                "WandB logger requested without a project. Falling back to console."
            )
            return ConsoleLogger(**kwargs)
        return WandBLogger(project=project, **kwargs)
    if logger_type == "tensorboard":
        if log_dir is None:
            log_dir = Path("tb_logs")
        return TensorBoardLogger(log_dir=log_dir, **kwargs)
    if logger_type == "console":
        return ConsoleLogger(**kwargs)
    if logger_type == "composite":
        # If explicit logger list provided, use it
        if loggers_list:
            valid_types = {"wandb", "tensorboard", "console"}
            loggers = []
            for logger_name in loggers_list:
                if logger_name == "composite":
                    # Avoid recursive composite
                    logger.warning(
                        "Skipping 'composite' in composite_loggers list to avoid recursion"
                    )
                    continue
                if logger_name not in valid_types:
                    logger.warning(
                        f"Unknown logger type '{logger_name}' in composite_loggers list. "
                        f"Valid types are: {valid_types}"
                    )
                    continue
                # Warn if wandb is used without project
                if logger_name == "wandb" and project is None:
                    logger.warning(
                        "Using WandB logger without a project. "
                        "Consider providing wandb_project in LoggingConfig"
                    )
                loggers.append(
                    create_logger(
                        logger_name, project=project, log_dir=log_dir, **kwargs
                    )
                )
            return CompositeLogger(loggers)

        # Otherwise, create composite with console + another backend
        loggers = [ConsoleLogger()]
        if project is not None:
            loggers.append(WandBLogger(project=project, **kwargs))  # pyright: ignore[reportArgumentType]
        elif log_dir is not None:
            loggers.append(TensorBoardLogger(log_dir=log_dir, **kwargs))  # pyright: ignore[reportArgumentType]
        return CompositeLogger(loggers)  # pyright: ignore[reportArgumentType]
    raise ValueError(f"Unknown logger type: {logger_type}")
