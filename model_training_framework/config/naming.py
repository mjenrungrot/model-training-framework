"""
Experiment Naming Strategies

This module provides standardized experiment naming conventions:
- Hash-based naming for uniqueness
- Parameter-based naming for readability
- Timestamp-based naming for chronological ordering
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
from typing import Any, Dict, Optional

from .schemas import NamingStrategy


class ExperimentNaming:
    """Standardized experiment naming system."""

    # Maximum length for experiment names to avoid filesystem issues
    MAX_NAME_LENGTH = 200

    # Characters to sanitize in experiment names
    INVALID_CHARS = r'[<>:"/\\|?*\s]'

    @staticmethod
    def generate_name(
        base_name: str,
        parameters: Dict[str, Any],
        naming_strategy: NamingStrategy = NamingStrategy.HASH_BASED,
    ) -> str:
        """Generate experiment name based on parameters."""

        if naming_strategy == NamingStrategy.HASH_BASED:
            return ExperimentNaming._generate_hash_based_name(base_name, parameters)
        if naming_strategy == NamingStrategy.PARAMETER_BASED:
            return ExperimentNaming._generate_parameter_based_name(
                base_name, parameters
            )
        if naming_strategy == NamingStrategy.TIMESTAMP_BASED:
            return ExperimentNaming._generate_timestamp_based_name(
                base_name, parameters
            )
        raise ValueError(f"Unknown naming strategy: {naming_strategy}")

    @staticmethod
    def _generate_hash_based_name(base_name: str, parameters: Dict[str, Any]) -> str:
        """Generate hash-based experiment name."""
        # Create deterministic hash from parameters
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:8]

        # Combine base name with hash
        sanitized_base = ExperimentNaming._sanitize_name(base_name)
        name = f"{sanitized_base}_{param_hash}"

        return ExperimentNaming._ensure_valid_length(name)

    @staticmethod
    def _generate_parameter_based_name(
        base_name: str, parameters: Dict[str, Any]
    ) -> str:
        """Generate parameter-based experiment name."""
        sanitized_base = ExperimentNaming._sanitize_name(base_name)

        # Create parameter string from key parameters
        param_parts = []

        # Sort parameters for consistency
        sorted_params = sorted(parameters.items())

        for key, value in sorted_params:
            # Shorten common parameter names
            short_key = ExperimentNaming._shorten_parameter_name(key)
            short_value = ExperimentNaming._format_parameter_value(value)
            param_parts.append(f"{short_key}_{short_value}")

        if param_parts:
            param_str = "_".join(param_parts)
            name = f"{sanitized_base}_{param_str}"
        else:
            name = sanitized_base

        return ExperimentNaming._ensure_valid_length(name)

    @staticmethod
    def _generate_timestamp_based_name(
        base_name: str, parameters: Dict[str, Any]
    ) -> str:
        """Generate timestamp-based experiment name."""
        sanitized_base = ExperimentNaming._sanitize_name(base_name)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Add parameter hash for uniqueness
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:6]

        name = f"{sanitized_base}_{timestamp}_{param_hash}"

        return ExperimentNaming._ensure_valid_length(name)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize experiment name by removing invalid characters."""
        # Replace invalid characters with underscores
        sanitized = re.sub(ExperimentNaming.INVALID_CHARS, "_", name)

        # Remove multiple consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        # Ensure name is not empty
        if not sanitized:
            sanitized = "experiment"

        return sanitized

    @staticmethod
    def _shorten_parameter_name(param_name: str) -> str:
        """Shorten common parameter names."""
        # Common abbreviations
        abbreviations = {
            "learning_rate": "lr",
            "batch_size": "bs",
            "weight_decay": "wd",
            "dropout": "drop",
            "hidden_size": "hs",
            "num_layers": "nl",
            "max_epochs": "ep",
            "optimizer": "opt",
            "scheduler": "sched",
            "gradient_accumulation_steps": "gas",
        }

        # Handle nested parameters (e.g., "optimizer.lr")
        if "." in param_name:
            parts = param_name.split(".")
            shortened_parts = [abbreviations.get(part, part) for part in parts]
            return ".".join(shortened_parts)

        return abbreviations.get(param_name, param_name)

    @staticmethod
    def _format_parameter_value(value: Any) -> str:
        """Format parameter value for inclusion in name."""
        if isinstance(value, float):
            # Format floats in scientific notation if small
            if abs(value) < 0.001 and value != 0:
                return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")
            return f"{value:.4g}".rstrip("0").rstrip(".")
        if isinstance(value, bool):
            return "T" if value else "F"
        if isinstance(value, str):
            # Take first 10 characters and sanitize
            return ExperimentNaming._sanitize_name(value[:10])
        return str(value)

    @staticmethod
    def _ensure_valid_length(name: str) -> str:
        """Ensure experiment name doesn't exceed maximum length."""
        if len(name) <= ExperimentNaming.MAX_NAME_LENGTH:
            return name

        # Truncate and add hash to maintain uniqueness
        truncated = name[: ExperimentNaming.MAX_NAME_LENGTH - 9]  # Leave space for hash
        name_hash = hashlib.sha256(name.encode()).hexdigest()[:8]

        return f"{truncated}_{name_hash}"

    @staticmethod
    def parse_experiment_name(name: str) -> Dict[str, Any]:
        """Extract parameter information from experiment name."""
        # This is a best-effort parsing - may not work for all naming strategies
        info = {
            "original_name": name,
            "base_name": None,
            "timestamp": None,
            "hash": None,
            "parameters": {},
        }

        # Try to extract timestamp (YYYYMMDD_HHMMSS pattern)
        timestamp_match = re.search(r"(\d{8}_\d{6})", name)
        if timestamp_match:
            info["timestamp"] = timestamp_match.group(1)

        # Try to extract hash (8 character hex at end)
        hash_match = re.search(r"_([a-f0-9]{8})$", name)
        if hash_match:
            info["hash"] = hash_match.group(1)

        # Extract base name (everything before first underscore or parameters)
        parts = name.split("_")
        if parts:
            info["base_name"] = parts[0]

        return info

    @staticmethod
    def validate_experiment_name(name: str) -> bool:
        """Validate experiment name format."""
        if not name:
            return False

        if len(name) > ExperimentNaming.MAX_NAME_LENGTH:
            return False

        # Check for invalid characters
        if re.search(ExperimentNaming.INVALID_CHARS, name):
            return False

        return True

    @staticmethod
    def suggest_experiment_name(
        base_name: str,
        existing_names: set[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Suggest a unique experiment name."""
        if parameters is None:
            parameters = {}

        # Try different naming strategies until we find a unique name
        strategies = [
            NamingStrategy.HASH_BASED,
            NamingStrategy.PARAMETER_BASED,
            NamingStrategy.TIMESTAMP_BASED,
        ]

        for strategy in strategies:
            candidate_name = ExperimentNaming.generate_name(
                base_name, parameters, strategy
            )

            if candidate_name not in existing_names:
                return candidate_name

        # If all strategies produce existing names, add a counter
        counter = 1
        while True:
            params_with_counter = parameters.copy()
            params_with_counter["_counter"] = counter

            candidate_name = ExperimentNaming.generate_name(
                base_name, params_with_counter, NamingStrategy.HASH_BASED
            )

            if candidate_name not in existing_names:
                return candidate_name

            counter += 1
