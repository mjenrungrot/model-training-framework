"""
Utilities Component

This module provides common utilities and helper functions:
- Logging setup and configuration
- Path utilities and project structure validation
- Common data structures and enums
- Helper functions for various components
"""

from .data_structures import (
    Result,
    error,
    success,
)
from .logging import (
    ColoredFormatter,
    get_logger,
    setup_logging,
)
from .path_utils import (
    ensure_directory_exists,
    get_project_root,
    resolve_config_path,
    validate_project_structure,
)
from .validation import (
    validate_choices,
    validate_range,
    validate_type,
)

__all__ = [
    # Logging
    "ColoredFormatter",
    # Data structures
    "Result",
    # Path utilities
    "ensure_directory_exists",
    "error",
    "get_logger",
    "get_project_root",
    "resolve_config_path",
    "setup_logging",
    "success",
    # Validation
    "validate_choices",
    "validate_project_structure",
    "validate_range",
    "validate_type",
]
