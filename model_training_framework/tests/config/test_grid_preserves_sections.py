"""
Tests that ParameterGridSearch preserves full configuration sections, including
custom_params, checkpoint, preemption, performance, and slurm, when generating
ExperimentConfig objects and dicts.
"""

from __future__ import annotations

from model_training_framework.config import (
    NamingStrategy,
    ParameterGrid,
    ParameterGridSearch,
)


def _make_base_config() -> dict:
    return {
        "experiment_name": "base_exp",
        "model": {"type": "mlp", "hidden_size": 64, "num_layers": 2},
        "training": {"max_epochs": 3, "gradient_accumulation_steps": 2},
        "data": {"dataset_name": "synthetic", "batch_size": 8, "num_workers": 0},
        "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01},
        "logging": {"use_wandb": False},
        "checkpoint": {
            "root_dir": "checkpoints",
            "save_every_n_steps": 10,
            "max_checkpoints": 2,
            "save_rng": True,
        },
        "preemption": {
            "signal": 10,
            "max_checkpoint_sec": 123.0,
            "resume_from_latest_symlink": True,
        },
        "performance": {
            "gradient_accumulation_steps": 2,
            "use_amp": False,
            "clip_grad_norm": 0.5,
        },
        "slurm": {
            "account": "acct",
            "partition": "part",
            "nodes": 1,
            "gpus_per_node": 0,
            "cpus_per_task": 2,
        },
        "custom_params": {
            "multi_loader": {
                "num_loaders": 2,
                "sampling_strategy": "weighted",
                "dataloader_weights": [0.6, 0.4],
            }
        },
    }


def test_generate_experiments_preserves_sections():
    base = _make_base_config()
    gs = ParameterGridSearch(base)
    gs.set_naming_strategy(NamingStrategy.PARAMETER_BASED)

    grid = ParameterGrid(name="demo")
    grid.add_parameter("optimizer.lr", [1e-4])
    grid.add_parameter("custom_params.multi_loader.sampling_strategy", ["round_robin"])
    gs.add_grid(grid)

    exps = list(gs.generate_experiments())
    assert len(exps) == 1
    exp = exps[0]

    # core sections present
    assert exp.model.hidden_size == 64
    assert exp.training.gradient_accumulation_steps == 2
    assert exp.data.batch_size == 8
    assert exp.optimizer.lr == 1e-4

    # preserved sections
    assert exp.checkpoint.max_checkpoints == 2
    assert exp.preemption.max_checkpoint_sec == 123.0
    assert exp.performance.clip_grad_norm == 0.5
    assert exp.slurm is not None
    assert exp.slurm.partition == "part"

    # custom_params preserved and overridden
    assert "multi_loader" in exp.custom_params
    ml = exp.custom_params["multi_loader"]
    assert ml["num_loaders"] == 2
    assert ml["sampling_strategy"] == "round_robin"
    assert ml["dataloader_weights"] == [0.6, 0.4]


def test_generate_dicts_matches_conversion():
    base = _make_base_config()
    gs = ParameterGridSearch(base)
    gs.set_naming_strategy(NamingStrategy.PARAMETER_BASED)

    grid = ParameterGrid(name="demo")
    grid.add_parameter("optimizer.lr", [1e-4, 3e-4])
    gs.add_grid(grid)

    dicts = list(gs.generate_experiment_dicts())
    exps = list(gs.generate_experiments())
    assert len(dicts) == len(exps) == 2

    # Dicts retain custom_params and config sections
    for d in dicts:
        assert d["custom_params"]["multi_loader"]["num_loaders"] == 2
        assert d["checkpoint"]["save_every_n_steps"] == 10
        assert d["preemption"]["signal"] == 10
        assert d["performance"]["gradient_accumulation_steps"] == 2
