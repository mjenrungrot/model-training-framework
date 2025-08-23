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
from typing import Any, Generic, Optional, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")


@dataclass
class Result(Generic[T, E]):
    """
    Result type for error handling, similar to Rust's Result.

    Can be either Success(value) or Error(error).
    """

    def __init__(self, value: Optional[T] = None, error: Optional[E] = None):
        if value is not None and error is not None:
            raise ValueError("Result cannot have both value and error")
        if value is None and error is None:
            raise ValueError("Result must have either value or error")

        self._value = value
        self._error = error

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
        if self.is_error:
            raise ValueError(f"Cannot get value from error result: {self._error}")
        return self._value

    @property
    def error(self) -> E:
        """Get the error value (raises if success)."""
        if self.is_success:
            raise ValueError("Cannot get error from success result")
        return self._error

    def unwrap(self) -> T:
        """Get value or raise error."""
        if self.is_error:
            if isinstance(self._error, Exception):
                raise self._error
            raise RuntimeError(f"Result failed: {self._error}")
        return self._value

    def unwrap_or(self, default: T) -> T:
        """Get value or return default."""
        return self._value if self.is_success else default

    def map(self, func) -> Result:
        """Apply function to value if success."""
        if self.is_success:
            try:
                return Success(func(self._value))
            except Exception as e:
                return Error(e)
        return self

    def map_error(self, func) -> Result:
        """Apply function to error if error."""
        if self.is_error:
            return Error(func(self._error))
        return self


def Success(value: T) -> Result[T, Any]:
    """Create successful result."""
    return Result(value=value)


def Error(error: E) -> Result[Any, E]:
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
PathLike = Union[str, "Path"]
ConfigDict = dict[str, Any]
MetricsDict = dict[str, float]
