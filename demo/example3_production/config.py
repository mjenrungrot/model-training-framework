"""
Example 3 configuration: base ExperimentConfig and a tiny grid search.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from model_training_framework.config import (
    NamingStrategy,
    ParameterGrid,
    ParameterGridSearch,
)

DEFAULT_EXPERIMENT_NAME = "example3_production"


def build_base_config() -> dict[str, Any]:
    """Return the base ExperimentConfig dictionary for example3.

    Keep it small and fast, but realistic enough to demonstrate resume & SLURM.
    """
    return {
        "experiment_name": DEFAULT_EXPERIMENT_NAME,
        "model": {"type": "mlp", "hidden_size": 128, "num_layers": 2},
        # Keep training short for demo; adjust as needed
        "training": {"max_epochs": 2, "gradient_accumulation_steps": 2},
        "data": {"dataset_name": "synthetic", "batch_size": 16, "num_workers": 0},
        "optimizer": {"type": "adamw", "lr": 3e-4, "weight_decay": 0.01},
        "logging": {"use_wandb": False},
        # SLURM defaults tuned for Hyak-like ckpt partitions
        # Note: "realitylab-ckpt" is an account on the ckpt partition; the actual
        # partition name is typically "ckpt" or "ckpt-all". We set conservative
        # time/memory suitable for this demo; adjust if your site requires.
        "slurm": {
            # On this cluster, ckpt partition requires the -ckpt account variant
            "account": "realitylab-ckpt",
            "partition": "ckpt",
            "nodes": 1,
            "ntasks_per_node": 1,
            "gpus_per_node": 1,
            "cpus_per_task": 8,
            "mem": "32G",
            "time": "02:00:00",
            "constraint": "a40|a100",
            "requeue": True,
        },
        # Demo multi-dataloader behavior via custom params
        "custom_params": {
            "multi_loader": {
                "num_loaders": 2,
                "sampling_strategy": "round_robin",  # or "weighted"
                "dataloader_weights": [0.6, 0.4],  # used if weighted
            }
        },
    }


def build_parameter_grid_search(base_config: dict[str, Any]) -> ParameterGridSearch:
    grid_search = ParameterGridSearch(base_config)
    # Use human-readable parameter-based naming for demonstration
    grid_search.set_naming_strategy(NamingStrategy.PARAMETER_BASED)
    grid = ParameterGrid(name="prod_demo")
    grid.add_parameter("optimizer.lr", [1e-4, 3e-4])
    grid.add_parameter("data.batch_size", [16])
    grid.add_parameter("training.gradient_accumulation_steps", [1, 2])
    grid.add_parameter(
        "custom_params.multi_loader.sampling_strategy", ["round_robin", "weighted"]
    )
    grid_search.add_grid(grid)
    return grid_search


if __name__ == "__main__":
    base = build_base_config()
    grid = build_parameter_grid_search(base)
    exps = list(grid.generate_experiments())
    print(f"Generated {len(exps)} experiment configs\n")
    for i, exp in enumerate(exps, 1):
        print(f"--- Experiment {i}: {exp.experiment_name}")
        print(json.dumps(asdict(exp), indent=2))
        print()
