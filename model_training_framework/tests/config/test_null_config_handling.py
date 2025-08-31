"""
Tests for handling null/None values in configuration loading.

This module tests the handling of optional configuration sections
when they are explicitly set to null in saved configuration files.
"""

import json

import pytest
import yaml

from model_training_framework.config import ConfigurationManager
from model_training_framework.config.schemas import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


class TestNullConfigHandling:
    """Test handling of null values in configuration files."""

    @pytest.fixture
    def config_with_null_scheduler(self):
        """Create a sample config with null scheduler."""
        return {
            "experiment_name": "test_null_scheduler",
            "model": {
                "type": "test_model",
                "hidden_size": 256,
                "num_layers": 3,
            },
            "training": {
                "max_epochs": 5,
                "gradient_accumulation_steps": 1,
            },
            "data": {
                "dataset_name": "test_dataset",
                "batch_size": 16,
                "num_workers": 2,
            },
            "optimizer": {
                "type": "adam",
                "lr": 0.001,
            },
            "scheduler": None,  # Explicitly null
            "slurm": None,  # Explicitly null
        }

    @pytest.fixture
    def config_with_all_nulls(self):
        """Create a config with all optional fields set to null."""
        return {
            "experiment_name": "test_all_nulls",
            "model": {
                "type": "test_model",
            },
            "training": {},
            "data": {
                "dataset_name": "test",
                "batch_size": 8,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
            },
            "scheduler": None,
            "slurm": None,
            "train_loader_config": None,
            "val_loader_config": None,
            "validation": None,
            "fault_tolerance": None,
            "ddp": None,
        }

    def test_load_json_with_null_scheduler(
        self, test_project_root, config_with_null_scheduler
    ):
        """Test loading JSON config with null scheduler."""
        config_manager = ConfigurationManager(project_root=test_project_root)

        # Save config as JSON
        config_file = test_project_root / "configs" / "test_null.json"
        with config_file.open("w") as f:
            json.dump(config_with_null_scheduler, f)

        # Load and create experiment config
        loaded_dict = config_manager.load_config(config_file, validate=False)
        experiment_config = config_manager.create_experiment_config(
            loaded_dict, validate=False
        )

        # Verify the config was created successfully
        assert experiment_config.experiment_name == "test_null_scheduler"
        assert experiment_config.scheduler is None
        assert experiment_config.slurm is None
        assert experiment_config.model.type == "test_model"

    def test_load_yaml_with_null_scheduler(
        self, test_project_root, config_with_null_scheduler
    ):
        """Test loading YAML config with null scheduler."""
        config_manager = ConfigurationManager(project_root=test_project_root)

        # Save config as YAML
        config_file = test_project_root / "configs" / "test_null.yaml"
        with config_file.open("w") as f:
            yaml.dump(config_with_null_scheduler, f)

        # Load and create experiment config
        loaded_dict = config_manager.load_config(config_file, validate=False)
        experiment_config = config_manager.create_experiment_config(
            loaded_dict, validate=False
        )

        # Verify the config was created successfully
        assert experiment_config.experiment_name == "test_null_scheduler"
        assert experiment_config.scheduler is None
        assert experiment_config.slurm is None

    def test_round_trip_with_null_values(self, test_project_root):
        """Test saving and loading configs with null values."""
        config_manager = ConfigurationManager(project_root=test_project_root)

        # Create an experiment config with None values
        original_config = ExperimentConfig(
            experiment_name="round_trip_test",
            model=ModelConfig(type="test_model"),
            training=TrainingConfig(),
            data=DataConfig(dataset_name="test", batch_size=32),
            optimizer=OptimizerConfig(type="adam", lr=0.001),
            scheduler=None,  # Explicitly None
            slurm=None,  # Explicitly None
        )

        # Save to JSON
        json_file = test_project_root / "configs" / "round_trip.json"
        config_manager.save_config(original_config, json_file, format="json")

        # Load back
        loaded_dict = config_manager.load_config(json_file, validate=False)
        loaded_config = config_manager.create_experiment_config(
            loaded_dict, validate=False
        )

        # Verify round-trip preservation
        assert loaded_config.experiment_name == original_config.experiment_name
        assert loaded_config.scheduler is None
        assert loaded_config.slurm is None
        assert loaded_config.model.type == original_config.model.type

    def test_multiple_null_optional_configs(
        self, test_project_root, config_with_all_nulls
    ):
        """Test handling multiple null optional configuration sections."""
        config_manager = ConfigurationManager(project_root=test_project_root)

        # Save config with multiple nulls
        config_file = test_project_root / "configs" / "all_nulls.json"
        with config_file.open("w") as f:
            json.dump(config_with_all_nulls, f)

        # Load and create experiment config
        loaded_dict = config_manager.load_config(config_file, validate=False)
        experiment_config = config_manager.create_experiment_config(
            loaded_dict, validate=False
        )

        # Verify all optional fields are None
        assert experiment_config.scheduler is None
        assert experiment_config.slurm is None
        assert experiment_config.train_loader_config is None
        assert experiment_config.val_loader_config is None
        assert experiment_config.validation is None
        assert experiment_config.fault_tolerance is None
        assert experiment_config.ddp is None

    def test_empty_dict_vs_null_handling(self, test_project_root):
        """Test difference between empty dict {} and null for optional configs."""
        config_manager = ConfigurationManager(project_root=test_project_root)

        # Config with empty dict for scheduler
        config_empty_dict = {
            "experiment_name": "test_empty_dict",
            "model": {"type": "test_model"},
            "training": {},
            "data": {"dataset_name": "test", "batch_size": 8},
            "optimizer": {"type": "sgd", "lr": 0.01},
            "scheduler": {},  # Empty dict, should create default SchedulerConfig
        }

        # Config with null scheduler
        config_null = {
            "experiment_name": "test_null",
            "model": {"type": "test_model"},
            "training": {},
            "data": {"dataset_name": "test", "batch_size": 8},
            "optimizer": {"type": "sgd", "lr": 0.01},
            "scheduler": None,  # Null, should remain None
        }

        # Test empty dict case
        empty_dict_file = test_project_root / "configs" / "empty_dict.json"
        with empty_dict_file.open("w") as f:
            json.dump(config_empty_dict, f)

        loaded_empty = config_manager.load_config(empty_dict_file, validate=False)
        config_empty = config_manager.create_experiment_config(
            loaded_empty, validate=False
        )

        # Empty dict should create a SchedulerConfig with defaults
        assert config_empty.scheduler is not None
        assert config_empty.scheduler.type == "cosine"  # Default value

        # Test null case
        null_file = test_project_root / "configs" / "null.json"
        with null_file.open("w") as f:
            json.dump(config_null, f)

        loaded_null = config_manager.load_config(null_file, validate=False)
        config_null_obj = config_manager.create_experiment_config(
            loaded_null, validate=False
        )

        # Null should remain None
        assert config_null_obj.scheduler is None

    def test_compose_configs_with_null_override(self, test_project_root):
        """Test config composition when override contains null values."""
        config_manager = ConfigurationManager(project_root=test_project_root)

        base_config = {
            "experiment_name": "base",
            "model": {"type": "base_model"},
            "training": {},
            "data": {"dataset_name": "test", "batch_size": 32},
            "optimizer": {"type": "adam", "lr": 0.001},
            "scheduler": {"type": "cosine", "warmup_steps": 100},
        }

        override_config = {
            "experiment_name": "override",
            "scheduler": None,  # Override with null
        }

        # Compose configs
        composed = config_manager.compose_configs(base_config, [override_config])

        # Create experiment config
        experiment_config = config_manager.create_experiment_config(
            composed, validate=False
        )

        # Scheduler should be None due to override
        assert experiment_config.experiment_name == "override"
        assert experiment_config.scheduler is None

    def test_real_world_config_file(self, test_project_root):
        """Test with a real-world config file structure."""
        config_manager = ConfigurationManager(project_root=test_project_root)

        # Simulate a real saved config from the framework
        real_config = {
            "experiment_name": "association_model_v1",
            "model": {
                "type": "transformer",
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "dropout": 0.1,
            },
            "training": {
                "max_epochs": 100,
                "gradient_accumulation_steps": 4,
                "early_stopping_patience": 10,
            },
            "data": {
                "dataset_name": "association_dataset",
                "batch_size": 64,
                "num_workers": 4,
                "pin_memory": True,
            },
            "optimizer": {
                "type": "adamw",
                "lr": 5e-5,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "scheduler": None,
            "slurm": None,
            "logging": {
                "use_wandb": True,
                "log_scalars_every_n_steps": 50,
            },
            "checkpoint": {
                "save_every_n_epochs": 5,
                "max_checkpoints": 3,
                "save_optimizer": True,
                "save_scheduler": True,
            },
            "preemption": {},
            "performance": {},
        }

        # Save and reload
        config_file = test_project_root / "configs" / "real_world.json"
        with config_file.open("w") as f:
            json.dump(real_config, f, indent=2)

        # This should not raise an error
        loaded_dict = config_manager.load_config(config_file, validate=False)
        experiment_config = config_manager.create_experiment_config(
            loaded_dict, validate=False
        )

        # Verify config loaded correctly
        assert experiment_config.experiment_name == "association_model_v1"
        assert experiment_config.scheduler is None
        assert experiment_config.slurm is None
        assert experiment_config.model.type == "transformer"
        assert experiment_config.optimizer.lr == 5e-5
