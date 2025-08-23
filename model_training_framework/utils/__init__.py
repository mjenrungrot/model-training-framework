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
    "ColoredFormatter",
    "Error",
    "Optional",
    # Data structures
    "Result",
    "Success",
    "config_to_dict",
    "deserialize_config",
    "dict_to_config",
    "ensure_directory_exists",
    "get_logger",
    # Path utilities
    "get_project_root",
    "resolve_config_path",
    # Serialization
    "serialize_config",
    # Logging
    "setup_logging",
    "validate_choices",
    "validate_project_structure",
    "validate_range",
    # Validation
    "validate_type",
]
