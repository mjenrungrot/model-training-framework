"""
Tests for utility components.

This module tests all utility functions including:
- Path utilities
- Logging setup
- Validation helpers
- Data structures
"""

import logging
from pathlib import Path
import tempfile

import pytest

from model_training_framework.utils import (
    get_project_root,
    setup_logging,
    validate_project_structure,
)
from model_training_framework.utils.data_structures import (
    LogLevel,
    Priority,
    error,
    success,
)
from model_training_framework.utils.path_utils import (
    ensure_directory_exists,
    find_files_by_pattern,
    get_file_size,
    get_relative_path,
    is_subpath,
    resolve_config_path,
    safe_filename,
)
from model_training_framework.utils.validation import (
    validate_choices,
    validate_email,
    validate_memory_string,
    validate_path_exists,
    validate_positive_number,
    validate_range,
    validate_string_format,
    validate_time_string,
    validate_type,
    validate_url,
)


class TestPathUtilities:
    """Test path utility functions."""

    def test_get_project_root(self):
        """Test project root detection."""
        # This might find different roots depending on environment
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()

    def test_get_project_root_with_indicators(self):
        """Test project root detection with specific indicators."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir).resolve()

            # Create git directory
            (temp_path / ".git").mkdir()

            # Should find this as project root
            root = get_project_root(temp_path)
            assert root == temp_path

    def test_resolve_config_path_absolute(self):
        """Test resolving absolute config path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_file = temp_path / "config.yaml"
            config_file.write_text("test: config")

            resolved = resolve_config_path(
                config_file, project_root=temp_path, config_dir=temp_path / "configs"
            )

            assert resolved == config_file

    def test_resolve_config_path_relative(self):
        """Test resolving relative config path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_dir = temp_path / "configs"
            config_dir.mkdir()

            config_file = config_dir / "test_config.yaml"
            config_file.write_text("test: config")

            resolved = resolve_config_path(
                "test_config.yaml", project_root=temp_path, config_dir=config_dir
            )

            assert resolved == config_file

    def test_resolve_config_path_not_found(self):
        """Test resolving non-existent config path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with pytest.raises(FileNotFoundError):
                resolve_config_path(
                    "nonexistent.yaml",
                    project_root=temp_path,
                    config_dir=temp_path / "configs",
                )

    def test_ensure_directory_exists(self):
        """Test ensuring directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            new_dir = temp_path / "new_directory"

            assert not new_dir.exists()

            result = ensure_directory_exists(new_dir)

            assert result == new_dir
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_find_files_by_pattern(self):
        """Test finding files by pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "test1.py").write_text("content")
            (temp_path / "test2.py").write_text("content")
            (temp_path / "test.txt").write_text("content")

            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "test3.py").write_text("content")

            # Find Python files
            py_files = find_files_by_pattern(temp_path, "*.py", recursive=True)

            assert len(py_files) == 3
            assert all(f.suffix == ".py" for f in py_files)

    def test_get_relative_path(self):
        """Test getting relative path."""
        base = Path("/home/user/project")
        target = Path("/home/user/project/configs/test.yaml")

        relative = get_relative_path(target, base)

        assert relative == Path("configs/test.yaml")

    def test_is_subpath(self):
        """Test checking if path is subpath."""
        parent = Path("/home/user/project")
        child = Path("/home/user/project/configs")
        non_child = Path("/home/other/project")

        assert is_subpath(child, parent) == True
        assert is_subpath(non_child, parent) == False

    def test_safe_filename(self):
        """Test creating safe filename."""
        unsafe_name = "test<>file|name*.txt"
        safe_name = safe_filename(unsafe_name)

        assert "<" not in safe_name
        assert ">" not in safe_name
        assert "|" not in safe_name
        assert "*" not in safe_name

        # Test length limit
        long_name = "a" * 300
        safe_long = safe_filename(long_name, max_length=50)
        assert len(safe_long) <= 50

    def test_get_file_size(self):
        """Test getting file size."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            size = get_file_size(f.name)
            assert size > 0

            # Cleanup
            Path(f.name).unlink()

    def test_validate_project_structure(self):
        """Test project structure validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Empty directory should have issues
            issues = validate_project_structure(temp_path)
            assert len(issues) > 0

            # Add some structure
            (temp_path / "configs").mkdir()
            (temp_path / "experiments").mkdir()

            issues = validate_project_structure(temp_path)
            # Should have fewer issues now
            assert "configs" not in str(issues).lower()


class TestValidationUtilities:
    """Test validation utility functions."""

    def test_validate_type(self):
        """Test type validation."""
        # Valid type
        assert validate_type("test", str, "test_string") == True
        assert validate_type(42, int, "test_int") == True

        # Invalid type
        with pytest.raises(TypeError):
            validate_type("test", int, "test_string")

    def test_validate_range(self):
        """Test range validation."""
        # Valid ranges
        assert validate_range(5, min_val=0, max_val=10) == True
        assert validate_range(5.5, min_val=5.0, max_val=6.0) == True

        # Invalid ranges
        with pytest.raises(ValueError):
            validate_range(-1, min_val=0, max_val=10)

        with pytest.raises(ValueError):
            validate_range(11, min_val=0, max_val=10)

    def test_validate_choices(self):
        """Test choices validation."""
        choices = {"option1", "option2", "option3"}

        # Valid choice
        assert validate_choices("option1", choices) == True

        # Invalid choice
        with pytest.raises(ValueError):
            validate_choices("invalid", choices)

    def test_validate_string_format(self):
        """Test string format validation."""
        # Valid email pattern
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        assert validate_string_format("test@example.com", email_pattern) == True

        # Invalid format
        with pytest.raises(ValueError):
            validate_string_format("invalid-email", email_pattern)

    def test_validate_path_exists(self):
        """Test path existence validation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            f.flush()

            # Valid file
            assert validate_path_exists(f.name, must_be_file=True) == True

            # Invalid directory requirement
            with pytest.raises(ValueError):
                validate_path_exists(f.name, must_be_dir=True)

            # Cleanup
            Path(f.name).unlink()

        # Non-existent path
        with pytest.raises(FileNotFoundError):
            validate_path_exists("/nonexistent/path")

    def test_validate_email(self):
        """Test email validation."""
        # Valid emails
        assert validate_email("test@example.com") == True
        assert validate_email("user.name+tag@domain.co.uk") == True

        # Invalid emails
        with pytest.raises(ValueError):
            validate_email("invalid-email")

        with pytest.raises(ValueError):
            validate_email("@domain.com")

    def test_validate_url(self):
        """Test URL validation."""
        # Valid URLs
        assert validate_url("https://example.com") == True
        assert validate_url("http://subdomain.example.com/path") == True

        # Invalid URLs
        with pytest.raises(ValueError):
            validate_url("not-a-url")

        with pytest.raises(ValueError):
            validate_url("ftp://example.com")  # Only http/https allowed

    def test_validate_positive_number(self):
        """Test positive number validation."""
        # Valid positive numbers
        assert validate_positive_number(5) == True
        assert validate_positive_number(1.5) == True
        assert validate_positive_number(0, allow_zero=True) == True

        # Invalid numbers
        with pytest.raises(ValueError):
            validate_positive_number(-1)

        with pytest.raises(ValueError):
            validate_positive_number(0, allow_zero=False)

    def test_validate_memory_string(self):
        """Test memory string validation."""
        # Valid memory strings
        assert validate_memory_string("256G") == True
        assert validate_memory_string("4096M") == True
        assert validate_memory_string("1024K") == True

        # Invalid memory strings
        with pytest.raises(ValueError):
            validate_memory_string("256")  # No unit

        with pytest.raises(ValueError):
            validate_memory_string("256T")  # Invalid unit

    def test_validate_time_string(self):
        """Test time string validation."""
        # Valid time strings
        assert validate_time_string("1-00:00:00") == True
        assert validate_time_string("24:00:00") == True
        assert validate_time_string("01:30:45") == True

        # Invalid time strings
        with pytest.raises(ValueError):
            validate_time_string("25:00")  # Wrong format

        with pytest.raises(ValueError):
            validate_time_string("1:30")  # Missing seconds


class TestDataStructures:
    """Test data structure utilities."""

    def test_success_result(self):
        """Test successful result."""
        result = success("test_value")

        assert result.is_success == True
        assert result.is_error == False
        assert result.value == "test_value"
        assert result.unwrap() == "test_value"
        assert result.unwrap_or("default") == "test_value"

    def test_error_result(self):
        """Test error result."""
        result = error("test_error")

        assert result.is_success == False
        assert result.is_error == True
        assert result.error == "test_error"
        assert result.unwrap_or("default") == "default"

        with pytest.raises(RuntimeError):
            result.unwrap()

    def test_result_map(self):
        """Test result mapping."""
        success_result = success(5)
        mapped = success_result.map(lambda x: x * 2)

        assert mapped.is_success == True
        assert mapped.value == 10

        error_result = error("error")
        mapped_error = error_result.map(lambda x: x * 2)

        assert mapped_error.is_error == True
        assert mapped_error.error == "error"

    def test_result_map_error(self):
        """Test result error mapping."""
        error_result = error("original_error")
        mapped = error_result.map_error(lambda e: f"mapped_{e}")

        assert mapped.is_error == True
        assert mapped.error == "mapped_original_error"

        success_result = success("value")
        mapped_success = success_result.map_error(lambda e: f"mapped_{e}")

        assert mapped_success.is_success == True
        assert mapped_success.value == "value"

    def test_priority_enum(self):
        """Test Priority enumeration."""
        assert Priority.LOW.value == 1
        assert Priority.MEDIUM.value == 2
        assert Priority.HIGH.value == 3
        assert Priority.CRITICAL.value == 4

    def test_log_level_enum(self):
        """Test LogLevel enumeration."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestLoggingUtilities:
    """Test logging utility functions."""

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logger = setup_logging(level=logging.INFO, logger_name="test_logger")

        assert logger.level == logging.INFO
        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            logger = setup_logging(
                level=logging.DEBUG, log_file=log_file, logger_name="test_file_logger"
            )

            # Test logging
            logger.info("Test message")

            # Check file was created and has content
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

    def test_setup_logging_with_colors(self):
        """Test logging setup with colors."""
        logger = setup_logging(
            level=logging.INFO, use_colors=True, logger_name="test_color_logger"
        )

        assert logger.name == "test_color_logger"
        # Hard to test colors directly, but ensure setup succeeds
        assert len(logger.handlers) > 0

    def test_setup_logging_colorlog_fallback(self):
        """Test logging setup with colorlog unavailable."""
        # Test that logging still works even if colorlog is available
        # (we can't easily test the fallback since colorlog is installed)
        logger = setup_logging(
            level=logging.INFO, use_colors=False, logger_name="test_fallback_logger"
        )

        # Should still work with standard formatter
        assert logger.name == "test_fallback_logger"
        assert len(logger.handlers) > 0
