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
from typing import Any

from .schemas import NamingStrategy


class ExperimentNaming:
    """Standardized experiment naming system."""

    # Maximum length for experiment names. W&B limits run names to 128 chars,
    # so we keep a small safety buffer to avoid hitting the limit downstream.
    MAX_NAME_LENGTH = 120

    # Characters to sanitize in experiment names
    INVALID_CHARS = r'[<>:"/\\|?*\s]'

    @staticmethod
    def generate_name(
        base_name: str,
        parameters: dict[str, Any],
        naming_strategy: NamingStrategy = NamingStrategy.PARAMETER_BASED,
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
    def _generate_hash_based_name(base_name: str, parameters: dict[str, Any]) -> str:
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
        base_name: str, parameters: dict[str, Any]
    ) -> str:
        """Generate parameter-based experiment name with bracket formatting.

        Format: exp_[param1_val1][param2_val2]...
        """
        sanitized_base = ExperimentNaming._sanitize_name(base_name)

        # Allow flat dicts with dot-separated keys by converting them to nested dicts
        parameters = ExperimentNaming._unflatten_dict(parameters)

        # Create parameter string from key parameters
        param_parts: list[str] = []

        # Sort parameters for consistency
        sorted_params = sorted(parameters.items())

        for key, value in sorted_params:
            # Shorten common parameter names
            short_key = ExperimentNaming._shorten_parameter_name(key)
            short_value = ExperimentNaming._format_parameter_value(value)
            # Use bracket format for better readability
            param_parts.append(f"[{short_key}_{short_value}]")

        if not param_parts:
            return ExperimentNaming._ensure_valid_length(sanitized_base)

        # Build full name and check length before truncation
        param_str = "".join(param_parts)
        full_name = f"{sanitized_base}_{param_str}"
        if len(full_name) <= ExperimentNaming.MAX_NAME_LENGTH:
            return full_name

        # Name is too long; truncate at parameter boundaries and append hash for uniqueness
        name_hash = hashlib.sha256(full_name.encode()).hexdigest()[:8]
        max_prefix_len = (
            ExperimentNaming.MAX_NAME_LENGTH - len(name_hash) - 1
        )  # underscore before hash

        prefix = f"{sanitized_base}_"
        if len(prefix) > max_prefix_len:
            # Base name alone is too long; need to include parameters in hash for uniqueness
            # Use the full name (base + params) to generate a unique hash
            truncated_base = sanitized_base[
                : max_prefix_len - 9
            ]  # Leave room for underscore and hash
            full_name_hash = hashlib.sha256(full_name.encode()).hexdigest()[:8]
            return f"{truncated_base}_{full_name_hash}"

        truncated = prefix
        for part in param_parts:
            if len(truncated) + len(part) <= max_prefix_len:
                truncated += part
            else:
                break

        if truncated.endswith("_"):
            return f"{truncated}{name_hash}"
        return f"{truncated}_{name_hash}"

    @staticmethod
    def _unflatten_dict(flat_params: dict[str, Any]) -> dict[str, Any]:
        """Convert flat dict with dot-separated keys into nested dictionaries.

        Raises:
            ValueError: If there are conflicting keys where a key is used both as
                       a scalar value and as a prefix for nested keys.
        """
        nested: dict[str, Any] = {}

        # First pass: Check for conflicts between scalar and nested keys
        all_keys = set(flat_params.keys())
        for key in all_keys:
            if "." in key:
                parts = key.split(".")
                # Check if any prefix of this key exists as a scalar
                for i in range(1, len(parts)):
                    prefix = ".".join(parts[:i])
                    if prefix in all_keys:
                        # Check if the prefix key maps to a non-dict value
                        # (We need to handle it before unflattening to detect conflicts)
                        prefix_parts = prefix.split(".")
                        if len(prefix_parts) == 1:
                            # Direct conflict at root level
                            raise ValueError(
                                f"Conflicting key: '{prefix}' is used both as a scalar value "
                                f"and as a prefix for nested key '{key}'. This would result in "
                                f"data loss. Please rename one of these parameters."
                            )

        # Second pass: Build the nested structure
        for key, value in flat_params.items():
            if "." not in key:
                if (
                    key in nested
                    and isinstance(nested[key], dict)
                    and isinstance(value, dict)
                ):
                    nested[key].update(value)
                else:
                    nested[key] = value
                continue

            parts = key.split(".")
            current = nested
            for i, part in enumerate(parts[:-1]):
                if part in current:
                    if not isinstance(current[part], dict):
                        # Found a conflict during construction
                        partial_key = ".".join(parts[: i + 1])
                        raise ValueError(
                            f"Conflicting key: '{partial_key}' has a scalar value but "
                            f"'{key}' requires it to be a nested structure. This would "
                            f"result in data loss. Please rename one of these parameters."
                        )
                else:
                    current[part] = {}
                current = current[part]

            last = parts[-1]
            if (
                last in current
                and isinstance(current[last], dict)
                and isinstance(value, dict)
            ):
                current[last].update(value)
            else:
                current[last] = value
        return nested

    @staticmethod
    def _generate_timestamp_based_name(
        base_name: str, parameters: dict[str, Any]
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
        """Shorten common parameter names.

        For known parameters, use predefined abbreviations.
        For unknown parameters, split by underscore and take first 3 chars of each part.
        """
        # Common abbreviations - prioritizing readability
        abbreviations = {
            # Training parameters
            "learning_rate": "lr",
            "batch_size": "batch",
            "weight_decay": "wdecay",
            "dropout": "dropout",
            "hidden_size": "hidden",
            "num_layers": "layers",
            "max_epochs": "epochs",
            "max_steps": "steps",
            "optimizer": "optim",
            "scheduler": "sched",
            "gradient_accumulation_steps": "grad_accum",  # 10 chars
            "gradient_as_bucket_view": "grad_bkt",  # 8 chars
            "clip_grad_norm": "clip_grad",  # 9 chars
            "max_grad_norm": "max_grad",  # 8 chars
            # Model/architecture parameters
            "model": "model",
            "type": "typ",
            "compile_model": "compile",
            # Data parameters
            "dataset_name": "dataset",
            "dataloader_num_workers": "workers",
            "dataloader_names": "dl_names",  # 8 chars
            "dataloader_weights": "dl_weights",  # 10 chars
            "prefetch_factor": "prefetch",  # 8 chars
            "pin_memory": "pin_mem",
            "persistent_workers": "persist_w",  # 9 chars
            # Validation parameters
            "validation_frequency": "val_freq",
            "validate_every_n_epochs": "val_epochs",
            "early_stopping_patience": "es_pat",  # 6 chars
            "early_stopping_metric": "es_metric",  # 9 chars
            "early_stopping_mode": "es_mode",  # 7 chars
            "early_stopping_source": "es_src",  # 6 chars
            # Checkpointing parameters
            "checkpoint": "ckpt",
            "save_frequency": "save_freq",
            "save_every_n_steps": "save_steps",  # 10 chars
            "save_every_n_epochs": "save_epoch",
            "save_every_n_minutes": "save_mins",
            "max_checkpoints": "max_ckpts",
            "save_optimizer": "save_opt",
            "save_scheduler": "save_sched",
            "save_rng": "save_rng",
            "save_best": "save_best",
            "save_dataset_state": "save_dset",  # 9 chars
            "save_sampler_state": "save_sampl",  # 10 chars
            "monitor_metric": "mon_metric",  # 10 chars
            "monitor_mode": "mon_mode",  # 8 chars
            "root_dir": "root_dir",
            # Logging parameters
            "log_frequency": "log_freq",
            "log_loss_every_n_steps": "log_loss_n",  # 10 chars
            "log_scalars_every_n_steps": "log_scal_n",  # 10 chars
            "log_images_every_n_steps": "log_img_n",  # 9 chars
            "log_gradients": "log_grads",
            "log_model_parameters": "log_params",
            "log_system_metrics": "log_sys",  # 7 chars
            "log_per_loader_metrics": "log_per_dl",  # 10 chars
            "log_global_metrics": "log_global",  # 10 chars
            "log_loader_proportions": "log_props",  # 9 chars
            "logger_type": "logger",
            "use_wandb": "wandb",
            "use_tensorboard": "tboard",
            "use_csv": "csv",
            "wandb_project": "wandb_proj",
            "wandb_entity": "wandb_ent",  # 9 chars
            "wandb_name": "wandb_name",
            "wandb_mode": "wandb_mode",
            "wandb_tags": "wandb_tags",
            "tensorboard_dir": "tboard_dir",
            "csv_log_dir": "csv_dir",
            # Performance parameters
            "use_amp": "amp",
            "benchmark": "benchmark",
            "deterministic": "determ",  # 6 chars
            "profile_training": "profile",
            "debug_mode": "debug",
            "dry_run": "dry_run",
            # Multi-dataloader parameters
            "sampling_strategy": "sampling",
            "epoch_length_policy": "epoch_pol",  # 9 chars
            "steps_per_epoch": "steps_ep",  # 8 chars
            "alternating_pattern": "alt_patt",  # 8 chars
            "burst_size": "burst",
            "cycle_short_loaders": "cycle_shrt",  # 10 chars
            "choice_rng_seed": "rng_seed",
            "prefetch_cap_total_batches": "pref_cap",  # 8 chars
            # DDP parameters
            "backend": "backend",
            "find_unused_parameters": "find_unus",  # 9 chars
            "broadcast_buffers": "bcast_buf",  # 9 chars
            "bucket_cap_mb": "bucket_mb",
            "sync_schedules_across_ranks": "sync_ranks",
            "validate_schedule_consistency": "val_sched",  # 9 chars
            # Preemption parameters
            "preemption": "preempt",
            "signal": "signal",
            "max_checkpoint_sec": "max_ckpt_s",  # 10 chars
            "requeue_job": "req_job",  # 7 chars
            "resume_from_latest_symlink": "res_latest",  # 10 chars
            "cleanup_on_exit": "cleanup",
            "backup_checkpoints": "bkup_ckpt",  # 9 chars
            # Hook parameters
            "hooks": "hooks",
            "hook_classes": "hook_cls",  # 8 chars
            "hook_configs": "hook_cfg",  # 8 chars
            "enable_logging_hook": "en_log_hk",  # 9 chars
            "enable_gradient_monitor": "en_grad_mo",  # 10 chars
            "enable_model_checkpoint_hook": "en_ckpt_hk",  # 10 chars
            "enable_early_stopping_hook": "en_es_hk",  # 8 chars
            "continue_on_hook_error": "cont_err",  # 8 chars
            "log_hook_errors": "log_hk_err",  # 10 chars
            # SLURM parameters
            "account": "account",
            "partition": "partition",
            "nodes": "nodes",
            "ntasks_per_node": "tasks_node",  # 10 chars
            "gpus_per_node": "gpus_node",  # 9 chars
            "cpus_per_task": "cpus_task",  # 9 chars
            "mem": "memory",
            "time": "time",
            "constraint": "constraint",  # 10 chars
            "requeue": "slurm_rq",  # 8 chars, avoid collision
            "job_name": "job_name",
            # Optimizer parameters
            "betas": "betas",
            "eps": "eps",
            "amsgrad": "amsgrad",
            "warmup_steps": "warmup",
            "min_lr": "min_lr",
            "gamma": "gamma",
            "step_size": "step_size",
            "milestones": "milestones",
            # Other common parameters
            "experiment_name": "exp",
            "seed": "seed",
            "frequency": "freq",
            "aggregation": "agg",  # 3 chars
            "per_loader_metrics": "per_dl_met",  # 10 chars
            "global_metrics": "glob_met",  # 8 chars
            "loss_weights_per_loader": "loss_wts",  # 8 chars
            "per_loader_optimizer_id": "opt_per_dl",  # 10 chars
            "strict": "strict",
            "custom_params": "custom",
        }

        # Handle nested parameters (e.g., "optimizer.lr" or "model.num_heads")
        if "." in param_name:
            parts = param_name.split(".")
            shortened_parts = []
            for part in parts:
                if part in abbreviations:
                    shortened_parts.append(abbreviations[part])
                else:
                    # For unknown nested parts, use the 3-char algorithm
                    shortened_parts.append(ExperimentNaming._abbreviate_unknown(part))
            return ".".join(shortened_parts)

        # Check if we have a predefined abbreviation
        if param_name in abbreviations:
            return abbreviations[param_name]

        # For unknown parameters, use the 3-character algorithm
        return ExperimentNaming._abbreviate_unknown(param_name)

    @staticmethod
    def _abbreviate_unknown(param_name: str) -> str:
        """Abbreviate unknown parameter names by taking first 3 chars of each underscore-separated part.

        Examples:
            disable_positional_encoding -> dis_pos_enc
            learnable_temperature -> lea_tem
            pairwise_rank -> pai_ran
        """
        if "_" not in param_name:
            # Single word - take first 3 characters
            return param_name[:3] if len(param_name) > 3 else param_name

        # Split by underscore and take first 3 chars of each part
        parts = param_name.split("_")
        abbreviated_parts = []
        for part in parts:
            if part:  # Skip empty parts
                # Take first 3 characters (or less if the part is shorter)
                abbreviated_parts.append(part[:3])

        return "_".join(abbreviated_parts)

    # Constants for value formatting
    FLOAT_SCIENTIFIC_THRESHOLD = 0.001
    STRING_TRUNCATE_LENGTH = 10

    @staticmethod
    def _format_parameter_value(value: Any) -> str:  # noqa: PLR0911
        """Format parameter value for inclusion in name."""
        if isinstance(value, dict):
            # For nested dictionaries, recursively process each item with brackets
            # This handles cases like model={'num_heads': 8, 'mlp_dim': 512}
            formatted_parts = []
            for k, v in sorted(value.items()):
                short_key = ExperimentNaming._shorten_parameter_name(k)
                formatted_value = ExperimentNaming._format_parameter_value(v)
                # Use dot notation for nested params for clarity
                formatted_parts.append(f"{short_key}.{formatted_value}")
            # Join with brackets for nested structure
            return "{" + ",".join(formatted_parts) + "}" if formatted_parts else "empty"
        if isinstance(value, list | tuple):
            # For lists/tuples, join with underscore
            formatted_items = [
                ExperimentNaming._format_parameter_value(v) for v in value
            ]
            return "_".join(formatted_items) if formatted_items else "empty"
        if isinstance(value, float):
            # Format floats in scientific notation if small
            if abs(value) < ExperimentNaming.FLOAT_SCIENTIFIC_THRESHOLD and value != 0:
                return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")
            return f"{value:.4g}".rstrip("0").rstrip(".")
        if isinstance(value, bool):
            return "T" if value else "F"
        if isinstance(value, str):
            # Take first 10 characters and sanitize
            return ExperimentNaming._sanitize_name(
                value[: ExperimentNaming.STRING_TRUNCATE_LENGTH]
            )
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
    def validate_experiment_name(name: str) -> bool:
        """Validate experiment name format."""
        if not name:
            return False

        if len(name) > ExperimentNaming.MAX_NAME_LENGTH:
            return False

        # Check for invalid characters
        return not re.search(ExperimentNaming.INVALID_CHARS, name)

    @staticmethod
    def suggest_experiment_name(
        base_name: str,
        existing_names: set[str],
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Suggest a unique experiment name."""
        if parameters is None:
            parameters = {}

        # Try different naming strategies until we find a unique name
        strategies = [
            NamingStrategy.PARAMETER_BASED,
            NamingStrategy.HASH_BASED,
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
