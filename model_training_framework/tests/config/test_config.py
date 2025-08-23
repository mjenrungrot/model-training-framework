"""
Tests for configuration management component.

This module tests all aspects of the configuration system including:
- Schema validation
- Parameter grid search
- Configuration loading and saving
- Validation logic
"""

import pytest

from model_training_framework.config import (
    ConfigurationManager,
    ConfigValidator,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    ParameterGrid,
    ParameterGridSearch,
    TrainingConfig,
    ValidationResult,
)
from model_training_framework.config.naming import ExperimentNaming
from model_training_framework.config.schemas import NamingStrategy


class TestConfigurationSchemas:
    """Test configuration schema classes."""

    def test_model_config_creation(self):
        """Test ModelConfig creation and validation."""
        config = ModelConfig(
            type="resnet18", hidden_size=512, num_layers=6, dropout=0.1
        )

        assert config.type == "resnet18"
        assert config.hidden_size == 512
        assert config.num_layers == 6
        assert config.dropout == 0.1

    def test_experiment_config_creation(self, sample_config_dict):
        """Test ExperimentConfig creation from dictionary."""
        model_config = ModelConfig(**sample_config_dict["model"])
        training_config = TrainingConfig(**sample_config_dict["training"])
        data_config = DataConfig(**sample_config_dict["data"])
        optimizer_config = OptimizerConfig(**sample_config_dict["optimizer"])

        config = ExperimentConfig(
            experiment_name=sample_config_dict["experiment_name"],
            model=model_config,
            training=training_config,
            data=data_config,
            optimizer=optimizer_config,
        )

        assert config.experiment_name == "test_experiment"
        assert config.model.type == "resnet18"
        assert config.training.max_epochs == 10
        assert config.data.dataset_name == "test_dataset"
        assert config.optimizer.type == "adamw"


class TestParameterGrid:
    """Test parameter grid functionality."""

    def test_parameter_grid_creation(self):
        """Test basic parameter grid creation."""
        grid = ParameterGrid("test_grid", "Test parameter grid")

        assert grid.name == "test_grid"
        assert grid.description == "Test parameter grid"
        assert len(grid.parameters) == 0

    def test_add_parameter(self):
        """Test adding parameters to grid."""
        grid = ParameterGrid("test_grid")
        grid.add_parameter("lr", [0.01, 0.001, 0.0001])
        grid.add_parameter("batch_size", [32, 64])

        assert "lr" in grid.parameters
        assert "batch_size" in grid.parameters
        assert grid.parameters["lr"] == [0.01, 0.001, 0.0001]
        assert grid.parameters["batch_size"] == [32, 64]

    def test_parameter_count(self):
        """Test parameter combination counting."""
        grid = ParameterGrid("test_grid")
        grid.add_parameter("lr", [0.01, 0.001])
        grid.add_parameter("batch_size", [32, 64])

        assert grid.get_parameter_count() == 4  # 2 * 2

    def test_generate_permutations(self):
        """Test parameter permutation generation."""
        grid = ParameterGrid("test_grid")
        grid.add_parameter("lr", [0.01, 0.001])
        grid.add_parameter("batch_size", [32, 64])

        permutations = list(grid.generate_permutations())

        assert len(permutations) == 4
        expected_combinations = [
            {"lr": 0.01, "batch_size": 32},
            {"lr": 0.01, "batch_size": 64},
            {"lr": 0.001, "batch_size": 32},
            {"lr": 0.001, "batch_size": 64},
        ]

        for combo in expected_combinations:
            assert combo in permutations

    def test_nested_parameters(self):
        """Test nested parameter handling."""
        grid = ParameterGrid("test_grid")
        grid.add_nested_parameter("optimizer.lr", [0.01, 0.001])
        grid.add_nested_parameter("model.dropout", [0.1, 0.2])

        permutations = list(grid.generate_permutations())
        assert len(permutations) == 4

        for perm in permutations:
            assert "optimizer.lr" in perm
            assert "model.dropout" in perm


class TestParameterGridSearch:
    """Test parameter grid search functionality."""

    def test_grid_search_creation(self, sample_config_dict):
        """Test parameter grid search creation."""
        grid_search = ParameterGridSearch(sample_config_dict)

        assert grid_search.base_config == sample_config_dict
        assert len(grid_search.parameter_grids) == 0

    def test_add_grid(self, sample_config_dict):
        """Test adding grids to search."""
        grid_search = ParameterGridSearch(sample_config_dict)

        grid = ParameterGrid("test_grid")
        grid.add_parameter("lr", [0.01, 0.001])

        grid_search.add_grid(grid)

        assert len(grid_search.parameter_grids) == 1
        assert grid_search.parameter_grids[0] == grid

    def test_total_experiments(self, sample_config_dict):
        """Test total experiment counting."""
        grid_search = ParameterGridSearch(sample_config_dict)

        grid1 = ParameterGrid("grid1")
        grid1.add_parameter("lr", [0.01, 0.001])

        grid2 = ParameterGrid("grid2")
        grid2.add_parameter("batch_size", [32, 64, 128])

        grid_search.add_grid(grid1)
        grid_search.add_grid(grid2)

        assert grid_search.get_total_experiments() == 5  # 2 + 3

    def test_validate_grids(self, sample_config_dict):
        """Test grid validation."""
        grid_search = ParameterGridSearch(sample_config_dict)

        # Empty grids should have issues
        issues = grid_search.validate_grids()
        assert "No parameter grids defined" in issues

        # Add valid grid
        grid = ParameterGrid("test_grid")
        grid.add_parameter("lr", [0.01, 0.001])
        grid_search.add_grid(grid)

        issues = grid_search.validate_grids()
        assert len(issues) == 0


class TestConfigValidator:
    """Test configuration validation."""

    def test_valid_config(self, sample_config_dict):
        """Test validation of valid configuration."""
        config = ExperimentConfig(
            experiment_name=sample_config_dict["experiment_name"],
            model=ModelConfig(**sample_config_dict["model"]),
            training=TrainingConfig(**sample_config_dict["training"]),
            data=DataConfig(**sample_config_dict["data"]),
            optimizer=OptimizerConfig(**sample_config_dict["optimizer"]),
        )

        result = ConfigValidator.validate_config(config)

        assert isinstance(result, ValidationResult)
        # Should have minimal issues for a reasonable config
        assert len(result.get_errors()) == 0

    def test_invalid_model_config(self, sample_config_dict):
        """Test validation with invalid model configuration."""
        # Create config with invalid values
        invalid_model = ModelConfig(
            type="",  # Empty type should cause error
            hidden_size=-1,  # Negative hidden size
            num_layers=0,  # Zero layers
            dropout=2.0,  # Dropout > 1.0
        )

        config = ExperimentConfig(
            experiment_name=sample_config_dict["experiment_name"],
            model=invalid_model,
            training=TrainingConfig(**sample_config_dict["training"]),
            data=DataConfig(**sample_config_dict["data"]),
            optimizer=OptimizerConfig(**sample_config_dict["optimizer"]),
        )

        result = ConfigValidator.validate_config(config)

        assert result.has_errors
        errors = result.get_errors()
        assert len(errors) > 0

    def test_invalid_optimizer_config(self, sample_config_dict):
        """Test validation with invalid optimizer configuration."""
        invalid_optimizer = OptimizerConfig(
            type="adamw",
            lr=-1.0,  # Negative learning rate
            weight_decay=-0.1,  # Negative weight decay
        )

        config = ExperimentConfig(
            experiment_name=sample_config_dict["experiment_name"],
            model=ModelConfig(**sample_config_dict["model"]),
            training=TrainingConfig(**sample_config_dict["training"]),
            data=DataConfig(**sample_config_dict["data"]),
            optimizer=invalid_optimizer,
        )

        result = ConfigValidator.validate_config(config)

        assert result.has_errors
        errors = result.get_errors()
        assert any(
            "Learning rate must be positive" in error.message for error in errors
        )


class TestExperimentNaming:
    """Test experiment naming functionality."""

    def test_hash_based_naming(self):
        """Test hash-based experiment naming."""
        parameters = {"lr": 0.01, "batch_size": 32}

        name1 = ExperimentNaming.generate_name(
            "experiment", parameters, NamingStrategy.HASH_BASED
        )
        name2 = ExperimentNaming.generate_name(
            "experiment", parameters, NamingStrategy.HASH_BASED
        )

        # Same parameters should generate same name
        assert name1 == name2
        assert "experiment" in name1
        assert len(name1.split("_")[-1]) == 8  # Hash should be 8 characters

    def test_parameter_based_naming(self):
        """Test parameter-based experiment naming."""
        parameters = {"lr": 0.01, "batch_size": 32}

        name = ExperimentNaming.generate_name(
            "experiment", parameters, NamingStrategy.PARAMETER_BASED
        )

        assert "experiment" in name
        assert "lr_0.01" in name
        # batch_size is shortened to bs in the implementation
        assert "bs_32" in name

    def test_timestamp_based_naming(self):
        """Test timestamp-based experiment naming."""
        parameters = {"lr": 0.01}

        name1 = ExperimentNaming.generate_name(
            "experiment", parameters, NamingStrategy.TIMESTAMP_BASED
        )

        import time

        time.sleep(
            1.1
        )  # Sleep longer to ensure different timestamp (seconds precision)

        name2 = ExperimentNaming.generate_name(
            "experiment", parameters, NamingStrategy.TIMESTAMP_BASED
        )

        # Different timestamps should generate different names
        assert name1 != name2
        assert "experiment" in name1
        assert "experiment" in name2

    def test_name_validation(self):
        """Test experiment name validation."""
        # Valid names
        assert ExperimentNaming.validate_experiment_name("valid_experiment") == True
        assert ExperimentNaming.validate_experiment_name("exp_123_abc") == True

        # Invalid names
        assert ExperimentNaming.validate_experiment_name("") == False
        assert ExperimentNaming.validate_experiment_name("invalid/name") == False
        assert ExperimentNaming.validate_experiment_name("invalid:name") == False


class TestConfigurationManager:
    """Test configuration manager functionality."""

    def test_config_manager_creation(self, test_project_root):
        """Test configuration manager creation."""
        config_manager = ConfigurationManager(
            project_root=test_project_root, config_dir=test_project_root / "configs"
        )

        assert config_manager.project_root == test_project_root
        assert config_manager.config_dir == test_project_root / "configs"

    def test_load_config_yaml(self, test_project_root, sample_config_dict):
        """Test loading YAML configuration."""
        config_manager = ConfigurationManager(project_root=test_project_root)

        # Create a test config file
        config_file = test_project_root / "configs" / "test.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        loaded_config = config_manager.load_config(config_file, validate=False)

        assert loaded_config["experiment_name"] == sample_config_dict["experiment_name"]
        assert loaded_config["model"]["type"] == sample_config_dict["model"]["type"]

    def test_create_experiment_config(self, test_project_root, sample_config_dict):
        """Test creating experiment configuration."""
        config_manager = ConfigurationManager(project_root=test_project_root)

        experiment_config = config_manager.create_experiment_config(
            sample_config_dict, validate=True
        )

        assert isinstance(experiment_config, ExperimentConfig)
        assert (
            experiment_config.experiment_name == sample_config_dict["experiment_name"]
        )

    def test_validation_failure(self, test_project_root):
        """Test configuration validation failure."""
        config_manager = ConfigurationManager(project_root=test_project_root)

        invalid_config = {
            "experiment_name": "",  # Empty name should fail
            "model": {"type": "", "hidden_size": 512, "num_layers": 6, "dropout": 0.1},
            "training": {"max_epochs": 10, "gradient_accumulation_steps": 1},
            "data": {"dataset_name": "test", "batch_size": 32, "num_workers": 2},
            "optimizer": {"type": "adamw", "lr": 0.001, "weight_decay": 0.01},
        }

        with pytest.raises(ValueError):
            config_manager.create_experiment_config(invalid_config, validate=True)
