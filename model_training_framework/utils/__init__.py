"""
Utilities Component

This module provides common utilities and helper functions:
- Logging setup and configuration
- Path utilities and project structure validation
- Common data structures and enums
- Helper functions for various components
"""

from .data_structures import (
    Error,
    Optional,
    Result,
    Success,
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
from .serialization import (
    config_to_dict,
    deserialize_config,
    dict_to_config,
    serialize_config,
)
from .validation import (
    validate_choices,
    validate_range,
    validate_type,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "ColoredFormatter",
    # Path utilities
    "get_project_root",
    "validate_project_structure",
    "ensure_directory_exists",
    "resolve_config_path",
    # Data structures
    "Result",
    "Success",
    "Error",
    "Optional",
    # Validation
    "validate_type",
    "validate_range",
    "validate_choices",
    # Serialization
    "serialize_config",
    "deserialize_config",
    "config_to_dict",
    "dict_to_config",
]
