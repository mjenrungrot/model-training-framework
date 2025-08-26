"""
Example 2 configuration: base ExperimentConfig and parameter grid search.

This module centralizes configuration for the single-file demo so the
training script focuses on orchestration and execution.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from model_training_framework.config import ParameterGrid, ParameterGridSearch

DEFAULT_EXPERIMENT_NAME = "example2_distributed"


def build_base_config() -> dict[str, Any]:
    """Return a base ExperimentConfig dictionary for the demo.

    Edit values here to tweak the default model, training setup, and
    multi-dataloader behavior used by the demo.
    """
    return {
        "experiment_name": DEFAULT_EXPERIMENT_NAME,
        "model": {"type": "transformer", "hidden_size": 384, "num_layers": 6},
        "training": {"max_epochs": 1, "gradient_accumulation_steps": 1},
        "data": {"dataset_name": "synthetic", "batch_size": 16, "num_workers": 2},
        "optimizer": {"type": "adamw", "lr": 3e-4, "weight_decay": 0.01},
        "logging": {"use_wandb": False},
        "custom_params": {
            "multi_loader": {"num_loaders": 2, "sampling_strategy": "round_robin"}
        },
    }


def build_parameter_grid_search(base_config: dict[str, Any]) -> ParameterGridSearch:
    """Create a small grid search over LR and batch size."""
    grid_search = ParameterGridSearch(base_config)
    grid = ParameterGrid(name="quick_demo")
    grid.add_parameter("optimizer.lr", [1e-4, 3e-4])
    grid.add_parameter("data.batch_size", [16, 32])
    grid_search.add_grid(grid)
    return grid_search


if __name__ == "__main__":
    # Preview composed configurations after applying the parameter grid
    base = build_base_config()
    grid_search = build_parameter_grid_search(base)
    experiments = list(grid_search.generate_experiments())

    print(f"Generated {len(experiments)} composed configurations:\n")
    for i, exp in enumerate(experiments, 1):
        exp_dict = asdict(exp)
        print(f"--- Experiment {i}: {exp.experiment_name} ---")
        print(json.dumps(exp_dict, indent=2))
        print()
