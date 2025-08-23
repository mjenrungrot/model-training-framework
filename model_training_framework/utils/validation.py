"""
Validation Utilities

This module provides common validation functions:
- Type validation helpers
- Range and constraint checking
- Value validation utilities
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, List, Optional, Set, Union


def validate_type(value: Any, expected_type: type, name: str = "value") -> bool:
    """
    Validate that a value is of expected type.

    Args:
        value: Value to validate
        expected_type: Expected type
        name: Name of the value for error messages

    Returns:
        True if valid

    Raises:
        TypeError: If type is invalid
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}"
        )
    return True


def validate_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    name: str = "value",
) -> bool:
    """
    Validate that a numeric value is within specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the value for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If value is out of range
    """
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} ({value}) must be >= {min_val}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{name} ({value}) must be <= {max_val}")

    return True


def validate_choices(value: Any, choices: Set[Any], name: str = "value") -> bool:
    """
    Validate that a value is one of allowed choices.

    Args:
        value: Value to validate
        choices: Set of allowed values
        name: Name of the value for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If value is not in choices
    """
    if value not in choices:
        raise ValueError(f"{name} ({value}) must be one of {choices}")
    return True


def validate_string_format(value: str, pattern: str, name: str = "value") -> bool:
    """
    Validate that a string matches a regex pattern.

    Args:
        value: String value to validate
        pattern: Regex pattern to match
        name: Name of the value for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If string doesn't match pattern
    """
    if not re.match(pattern, value):
        raise ValueError(f"{name} ({value}) does not match required pattern: {pattern}")
    return True


def validate_path_exists(
    path: Union[str, Path],
    must_be_file: bool = False,
    must_be_dir: bool = False,
    name: str = "path",
) -> bool:
    """
    Validate that a path exists and optionally is a file or directory.

    Args:
        path: Path to validate
        must_be_file: If True, path must be a file
        must_be_dir: If True, path must be a directory
        name: Name of the path for error messages

    Returns:
        True if valid

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If path type is incorrect
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")

    if must_be_file and not path.is_file():
        raise ValueError(f"{name} must be a file: {path}")

    if must_be_dir and not path.is_dir():
        raise ValueError(f"{name} must be a directory: {path}")

    return True


def validate_email(email: str, name: str = "email") -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate
        name: Name of the field for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If email format is invalid
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return validate_string_format(email, pattern, name)


def validate_url(url: str, name: str = "url") -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate
        name: Name of the field for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If URL format is invalid
    """
    pattern = r'^https?://[^\s<>"{}|\\^`\[\]]+$'
    return validate_string_format(url, pattern, name)


def validate_list_not_empty(values: List[Any], name: str = "list") -> bool:
    """
    Validate that a list is not empty.

    Args:
        values: List to validate
        name: Name of the field for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If list is empty
    """
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return True


def validate_dict_has_keys(
    data: dict, required_keys: Set[str], name: str = "dictionary"
) -> bool:
    """
    Validate that a dictionary has required keys.

    Args:
        data: Dictionary to validate
        required_keys: Set of required keys
        name: Name of the field for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"{name} is missing required keys: {missing_keys}")
    return True


def validate_positive_number(
    value: Union[int, float], name: str = "value", allow_zero: bool = False
) -> bool:
    """
    Validate that a number is positive.

    Args:
        value: Number to validate
        name: Name of the field for error messages
        allow_zero: Whether to allow zero

    Returns:
        True if valid

    Raises:
        ValueError: If number is not positive
    """
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} ({value}) must be >= 0")
    elif value <= 0:
        raise ValueError(f"{name} ({value}) must be > 0")
    return True


def validate_memory_string(memory: str, name: str = "memory") -> bool:
    """
    Validate memory string format (e.g., "256G", "4096M").

    Args:
        memory: Memory string to validate
        name: Name of the field for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If memory format is invalid
    """
    pattern = r"^\d+[KMG]$"
    return validate_string_format(memory, pattern, name)


def validate_time_string(time_str: str, name: str = "time") -> bool:
    """
    Validate SLURM time string format (e.g., "1-00:00:00", "24:00:00").

    Args:
        time_str: Time string to validate
        name: Name of the field for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If time format is invalid
    """
    # Pattern for D-HH:MM:SS or HH:MM:SS format
    pattern = r"^(\d+-)?(\d{1,2}):(\d{2}):(\d{2})$"

    if not re.match(pattern, time_str):
        raise ValueError(
            f"{name} ({time_str}) must be in format 'D-HH:MM:SS' or 'HH:MM:SS'"
        )

    return True
