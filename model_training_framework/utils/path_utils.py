"""
Path Utilities

This module provides utility functions for path resolution and project structure management:
- Project root detection
- Configuration file resolution
- Directory validation and creation
- Path normalization and validation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def get_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root by looking for common project indicators.

    Args:
        start_path: Path to start searching from (defaults to current directory)

    Returns:
        Path to project root

    Raises:
        RuntimeError: If project root cannot be determined
    """
    if start_path is None:
        start_path = Path.cwd()

    current_path = Path(start_path).resolve()

    # Common indicators of project root
    root_indicators = [
        ".git",
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "Pipfile",
        "conda.yaml",
        "environment.yml",
        "slurm_template.txt",  # Framework-specific indicator
        "model_training_framework",  # Package directory
    ]

    # Search up the directory tree
    while current_path != current_path.parent:
        for indicator in root_indicators:
            if (current_path / indicator).exists():
                logger.debug(
                    f"Found project root at {current_path} (indicator: {indicator})"
                )
                return current_path
        current_path = current_path.parent

    # If no indicators found, use the starting path
    logger.warning(f"Could not find project root indicators, using {start_path}")
    return Path(start_path).resolve()


def resolve_config_path(
    config_path: Union[str, Path], project_root: Path, config_dir: Optional[Path] = None
) -> Path:
    """
    Resolve configuration file path with fallback search locations.

    Args:
        config_path: Configuration file path (can be relative or absolute)
        project_root: Project root directory
        config_dir: Configuration directory (defaults to project_root/configs)

    Returns:
        Resolved path to configuration file

    Raises:
        FileNotFoundError: If configuration file cannot be found
    """
    config_path = Path(config_path)
    project_root = Path(project_root)

    if config_dir is None:
        config_dir = project_root / "configs"

    # If absolute path, use as-is
    if config_path.is_absolute():
        if config_path.exists():
            return config_path
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Search locations in order of preference
    search_locations = [
        # Relative to current directory
        Path.cwd() / config_path,
        # Relative to project root
        project_root / config_path,
        # Relative to config directory
        config_dir / config_path,
        # With .yaml extension added
        config_dir / f"{config_path}.yaml",
        config_dir / f"{config_path}.yml",
        # With .json extension added
        config_dir / f"{config_path}.json",
    ]

    for location in search_locations:
        if location.exists():
            logger.debug(f"Resolved config path {config_path} to {location}")
            return location

    # If still not found, raise error with helpful message
    searched_paths = "\n".join(f"  - {loc}" for loc in search_locations)
    raise FileNotFoundError(
        f"Configuration file '{config_path}' not found. Searched locations:\n{searched_paths}"
    )


def ensure_directory_exists(directory: Union[str, Path], parents: bool = True) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        directory: Directory path to ensure exists
        parents: Whether to create parent directories

    Returns:
        Path to the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=parents, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")
    return directory


def validate_project_structure(project_root: Path) -> List[str]:
    """
    Validate project structure and return list of issues.

    Args:
        project_root: Path to project root

    Returns:
        List of validation issues (empty if no issues)
    """
    issues = []
    project_root = Path(project_root)

    if not project_root.exists():
        issues.append(f"Project root does not exist: {project_root}")
        return issues

    if not project_root.is_dir():
        issues.append(f"Project root is not a directory: {project_root}")
        return issues

    # Check for recommended directories
    recommended_dirs = {
        "configs": "Configuration files",
        "experiments": "Experiment outputs",
        "scripts": "Training scripts",
        "data": "Dataset storage (optional)",
        "notebooks": "Jupyter notebooks (optional)",
    }

    for dir_name, description in recommended_dirs.items():
        dir_path = project_root / dir_name
        if not dir_path.exists():
            if dir_name not in ["data", "notebooks"]:  # Optional directories
                issues.append(
                    f"Recommended directory missing: {dir_path} ({description})"
                )

    # Check for SLURM template
    slurm_template_paths = [
        project_root / "slurm_template.txt",
        project_root / "slurm" / "slurm_template.txt",
        project_root / "slurm" / "slurm_template.sbatch",
    ]

    if not any(path.exists() for path in slurm_template_paths):
        issues.append(
            "SLURM template file not found (slurm_template.txt or slurm_template.sbatch)"
        )

    # Check for Python package
    package_path = project_root / "model_training_framework"
    if not package_path.exists():
        issues.append(f"Package directory not found: {package_path}")

    return issues


def find_files_by_pattern(
    directory: Union[str, Path], pattern: str, recursive: bool = True
) -> List[Path]:
    """
    Find files matching a glob pattern.

    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        recursive: Whether to search recursively

    Returns:
        List of matching file paths
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    if recursive:
        matches = list(directory.rglob(pattern))
    else:
        matches = list(directory.glob(pattern))

    # Filter to only files (exclude directories)
    file_matches = [path for path in matches if path.is_file()]

    logger.debug(f"Found {len(file_matches)} files matching '{pattern}' in {directory}")
    return file_matches


def get_relative_path(file_path: Union[str, Path], base_path: Union[str, Path]) -> Path:
    """
    Get relative path from base_path to file_path.

    Args:
        file_path: Target file path
        base_path: Base path to calculate relative from

    Returns:
        Relative path from base to target
    """
    file_path = Path(file_path).resolve()
    base_path = Path(base_path).resolve()

    try:
        return file_path.relative_to(base_path)
    except ValueError:
        # If paths don't share a common base, return absolute path
        logger.warning(f"Cannot create relative path from {base_path} to {file_path}")
        return file_path


def is_subpath(child_path: Union[str, Path], parent_path: Union[str, Path]) -> bool:
    """
    Check if child_path is a subpath of parent_path.

    Args:
        child_path: Potential child path
        parent_path: Potential parent path

    Returns:
        True if child_path is under parent_path
    """
    try:
        child_path = Path(child_path).resolve()
        parent_path = Path(parent_path).resolve()
        child_path.relative_to(parent_path)
        return True
    except ValueError:
        return False


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalize a path by resolving symlinks and relative components.

    Args:
        path: Path to normalize

    Returns:
        Normalized path
    """
    return Path(path).resolve()


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create a safe filename by removing invalid characters.

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Safe filename
    """
    import re

    # Replace invalid characters with underscores
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Remove multiple consecutive underscores
    safe_name = re.sub(r"_+", "_", safe_name)

    # Remove leading/trailing underscores and spaces
    safe_name = safe_name.strip("_ ")

    # Ensure name is not empty
    if not safe_name:
        safe_name = "unnamed"

    # Truncate if too long
    if len(safe_name) > max_length:
        safe_name = safe_name[: max_length - 4] + "..."

    return safe_name


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path.stat().st_size


def get_directory_size(dir_path: Union[str, Path]) -> int:
    """
    Get total size of directory and all its contents in bytes.

    Args:
        dir_path: Path to directory

    Returns:
        Total size in bytes
    """
    dir_path = Path(dir_path)
    total_size = 0

    if not dir_path.exists():
        return 0

    for file_path in dir_path.rglob("*"):
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except (OSError, FileNotFoundError):
                # Skip files that can't be accessed
                continue

    return total_size


def create_backup_path(original_path: Union[str, Path], suffix: str = ".bak") -> Path:
    """
    Create a backup path for a file by adding a suffix.

    Args:
        original_path: Original file path
        suffix: Suffix to add for backup

    Returns:
        Backup file path
    """
    original_path = Path(original_path)
    backup_path = original_path.with_suffix(original_path.suffix + suffix)

    # If backup already exists, add a counter
    counter = 1
    while backup_path.exists():
        backup_path = original_path.with_suffix(
            f"{original_path.suffix}{suffix}.{counter}"
        )
        counter += 1

    return backup_path
