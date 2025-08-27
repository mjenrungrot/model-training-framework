"""
Data Structures and Common Types

This module provides common data structures and type definitions used throughout the framework:
- Result types for error handling
- Optional type handling
- Common enums and constants
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass
class Result[T, E]:
    """
    Result type for error handling, similar to Rust's Result.

    Can be either success(value) or error(error).
    """

    def __init__(self, value: T | None = None, error: E | None = None):
        if value is not None and error is not None:
            raise ValueError("Result cannot have both value and error")
        if value is None and error is None:
            raise ValueError("Result must have either value or error")

        # Internal storage (optional until validated by accessors)
        self._value: T | None = value
        self._error: E | None = error

    @property
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self._value is not None

    @property
    def is_error(self) -> bool:
        """Check if result is an error."""
        return self._error is not None

    @property
    def value(self) -> T:
        """Get the success value (raises if error)."""
        if self.is_error or self._value is None:
            raise ValueError(f"Cannot get value from error result: {self._error}")
        return self._value

    @property
    def error(self) -> E:
        """Get the error value (raises if success)."""
        if self.is_success or self._error is None:
            raise ValueError("Cannot get error from success result")
        return self._error

    def unwrap(self) -> T:
        """Get value or raise error."""
        if self.is_error:
            if isinstance(self._error, Exception):
                raise self._error
            raise RuntimeError(f"Result failed: {self._error}")
        # At this point _value must be present
        assert self._value is not None
        return self._value

    def unwrap_or(self, default: T) -> T:
        """Get value or return default."""
        if self.is_success and self._value is not None:
            return self._value
        return default

    def map(self, func) -> Result:
        """Apply function to value if success."""
        if self.is_success:
            try:
                return success(func(self._value))
            except Exception as e:
                return error(e)
        return self

    def map_error(self, func) -> Result:
        """Apply function to error if error."""
        if self.is_error:
            return error(func(self._error))
        return self


def success[T](value: T) -> Result[T, Any]:
    """Create successful result."""
    return Result(value=value)


def error[E](error: E) -> Result[Any, E]:
    """Create error result."""
    return Result(error=error)


class Priority(Enum):
    """Priority levels for various operations."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Type aliases for common types
PathLike = str | Path
ConfigDict = dict[str, Any]
MetricsDict = dict[str, float]
