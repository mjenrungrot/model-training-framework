"""
Logging Utilities

This module provides centralized logging configuration and utilities:
- Colored console logging
- File logging with rotation
- Logger setup and configuration
- Custom formatters and handlers
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
import sys
from typing import Optional, Union

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


class ColoredFormatter(logging.Formatter):
    """Custom colored formatter that works with or without colorlog."""

    # Color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ):
        """Initialize formatter."""
        if fmt is None:
            fmt = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"

        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"

        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if not self.use_colors:
            return super().format(record)

        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
            record.levelname = colored_levelname

        # Format the message
        formatted = super().format(record)

        # Reset levelname for other formatters
        record.levelname = levelname

        return formatted


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    use_colors: bool = True,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file path
        log_dir: Directory for log files (used if log_file not specified)
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        use_colors: Whether to use colored console output
        format_string: Custom format string
        date_format: Custom date format
        logger_name: Name of logger to configure (None for root logger)

    Returns:
        Configured logger instance
    """

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)

    # Use colorlog if available and colors requested
    if HAS_COLORLOG and use_colors:
        console_formatter = colorlog.ColoredFormatter(
            format_string
            or "%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s %(name)s: %(message)s",
            datefmt=date_format or "%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        console_formatter = ColoredFormatter(
            fmt=format_string, datefmt=date_format, use_colors=use_colors
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Setup file handler if requested
    if log_file or log_dir:
        if log_file:
            log_path = Path(log_file)
        else:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "training_framework.log"

        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setLevel(level)

        # File formatter (no colors)
        file_formatter = logging.Formatter(
            format_string or "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
            datefmt=date_format or "%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to avoid double logging
    if logger_name:
        logger.propagate = False

    return logger


def get_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """
    Get a logger with optional level override.

    Args:
        name: Logger name
        level: Optional level override

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)

    return logger


def set_logger_level(logger_name: str, level: Union[str, int]) -> None:
    """
    Set level for a specific logger.

    Args:
        logger_name: Name of logger
        level: New logging level
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)


def disable_logger(logger_name: str) -> None:
    """
    Disable a specific logger.

    Args:
        logger_name: Name of logger to disable
    """
    logger = logging.getLogger(logger_name)
    logger.disabled = True


def enable_logger(logger_name: str) -> None:
    """
    Enable a previously disabled logger.

    Args:
        logger_name: Name of logger to enable
    """
    logger = logging.getLogger(logger_name)
    logger.disabled = False


def configure_third_party_loggers(level: Union[str, int] = logging.WARNING) -> None:
    """
    Configure common third-party library loggers to reduce noise.

    Args:
        level: Level to set for third-party loggers
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Common noisy loggers
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool",
        "matplotlib.font_manager",
        "PIL.PngImagePlugin",
        "numba.core.ssa",
        "numba.core.byteflow",
        "numba.core.interpreter",
        "transformers.tokenization_utils_base",
        "transformers.configuration_utils",
        "transformers.modeling_utils",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(level)


class LoggingContext:
    """Context manager for temporary logging level changes."""

    def __init__(self, logger_name: str, temp_level: Union[str, int]):
        """
        Initialize logging context.

        Args:
            logger_name: Name of logger to modify
            temp_level: Temporary level to set
        """
        self.logger_name = logger_name
        self.temp_level = temp_level
        self.original_level = None

    def __enter__(self) -> logging.Logger:
        """Enter context and change logging level."""
        logger = logging.getLogger(self.logger_name)
        self.original_level = logger.level

        if isinstance(self.temp_level, str):
            self.temp_level = getattr(logging, self.temp_level.upper())

        logger.setLevel(self.temp_level)
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore original logging level."""
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.original_level)


def log_function_call(func):
    """Decorator to log function calls with arguments."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise

    return wrapper


def create_experiment_logger(
    experiment_name: str,
    log_dir: Union[str, Path],
    level: Union[str, int] = logging.INFO,
) -> logging.Logger:
    """
    Create a dedicated logger for an experiment.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        level: Logging level

    Returns:
        Configured experiment logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f"experiment.{experiment_name}"
    log_file = log_dir / f"{experiment_name}.log"

    return setup_logging(
        level=level,
        log_file=log_file,
        logger_name=logger_name,
        use_colors=False,  # File logging doesn't need colors
    )


# Setup default logging for the package
_default_logger = None


def get_default_logger() -> logging.Logger:
    """Get the default logger for the package."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging(
            level=logging.INFO, logger_name="model_training_framework"
        )
        configure_third_party_loggers()
    return _default_logger
