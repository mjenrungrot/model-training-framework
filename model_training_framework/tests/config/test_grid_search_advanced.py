"""Tests for advanced ParameterGrid API features."""

from __future__ import annotations

import numpy as np
import pytest

from model_training_framework.config import ParameterGrid, ParameterGridSearch


class TestLinkedParameters:
    """Test linked/grouped parameter functionality."""

    def test_multiple_linked_groups_count(self):
        """Test that multiple linked groups are counted correctly."""
        grid = ParameterGrid("test")

        # Add first linked group with 2 value sets
        grid.add_linked_parameters(["param1", "param2"], [(1, 2), (3, 4)])

        # Add second linked group with 2 value sets
        grid.add_linked_parameters(["param3", "param4"], [(5, 6), (7, 8)])

        # Should be 4 total (2 + 2), not 2 (max(2, 2))
        assert grid.get_parameter_count() == 4

        # Verify generation matches count
        combinations = list(grid.generate_permutations())
        assert len(combinations) == 4

        # Verify all expected combinations exist
        expected = [
            {"param1": 1, "param2": 2},
            {"param1": 3, "param2": 4},
            {"param3": 5, "param4": 6},
            {"param3": 7, "param4": 8},
        ]
        for exp in expected:
            assert exp in combinations

    def test_add_linked_parameters_tuples(self):
        """Test adding linked parameters with tuples."""
        grid = ParameterGrid("test")
        grid.add_linked_parameters(
            ["lr", "weight_decay"], [(0.1, 0.01), (0.01, 0.001), (0.001, 0.0001)]
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 3

        # Check that parameters are linked correctly
        assert combinations[0] == {"lr": 0.1, "weight_decay": 0.01}
        assert combinations[1] == {"lr": 0.01, "weight_decay": 0.001}
        assert combinations[2] == {"lr": 0.001, "weight_decay": 0.0001}

    def test_add_linked_parameters_dicts(self):
        """Test adding linked parameters with dictionaries."""
        grid = ParameterGrid("test")
        grid.add_linked_parameters(
            ["lr", "weight_decay"],
            [{"lr": 0.1, "weight_decay": 0.01}, {"lr": 0.01, "weight_decay": 0.001}],
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 2
        assert combinations[0] == {"lr": 0.1, "weight_decay": 0.01}
        assert combinations[1] == {"lr": 0.01, "weight_decay": 0.001}

    def test_add_parameter_sets(self):
        """Test adding complete parameter sets."""
        grid = ParameterGrid("test")
        grid.add_parameter_sets(
            [
                {"lr": 0.1, "weight_decay": 0.1, "warmup": 100},
                {"lr": 0.01, "weight_decay": 0.01, "warmup": 500},
                {"lr": 0.001, "weight_decay": 0.001, "warmup": 1000},
            ]
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 3

        # Check first combination has all parameters
        assert combinations[0]["lr"] == 0.1
        assert combinations[0]["weight_decay"] == 0.1
        assert combinations[0]["warmup"] == 100

    def test_linked_with_traditional_parameters(self):
        """Test combining linked parameters with traditional ones."""
        grid = ParameterGrid("test")
        grid.add_parameter("batch_size", [32, 64])
        grid.add_linked_parameters(["lr", "weight_decay"], [(0.1, 0.01), (0.01, 0.001)])

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 4  # 2 batch_sizes * 2 linked sets

        # Check all combinations exist
        expected = [
            {"batch_size": 32, "lr": 0.1, "weight_decay": 0.01},
            {"batch_size": 32, "lr": 0.01, "weight_decay": 0.001},
            {"batch_size": 64, "lr": 0.1, "weight_decay": 0.01},
            {"batch_size": 64, "lr": 0.01, "weight_decay": 0.001},
        ]
        for exp in expected:
            assert exp in combinations


class TestConditionalParameters:
    """Test conditional parameter functionality."""

    def test_conditional_parameter_with_dict(self):
        """Test conditional parameter with dictionary condition."""
        grid = ParameterGrid("test")
        grid.add_parameter("optimizer", ["adam", "sgd"])
        grid.add_conditional_parameter(
            "momentum", [0.9, 0.95], when={"optimizer": "sgd"}
        )

        combinations = list(grid.generate_permutations())

        # Adam should have no momentum, SGD should have 2 momentum values
        adam_combos = [c for c in combinations if c["optimizer"] == "adam"]
        sgd_combos = [c for c in combinations if c["optimizer"] == "sgd"]

        assert len(adam_combos) == 1
        assert "momentum" not in adam_combos[0]

        assert len(sgd_combos) == 2
        assert all("momentum" in c for c in sgd_combos)
        assert {c["momentum"] for c in sgd_combos} == {0.9, 0.95}

    def test_conditional_parameter_with_function(self):
        """Test conditional parameter with function condition."""
        grid = ParameterGrid("test")
        grid.add_parameter("batch_size", [16, 32, 64])
        grid.add_conditional_parameter(
            "gradient_accumulation", [2, 4], when_func=lambda p: p["batch_size"] < 64
        )

        combinations = list(grid.generate_permutations())

        # Only batch_size < 64 should have gradient_accumulation
        small_batch = [c for c in combinations if c["batch_size"] < 64]
        large_batch = [c for c in combinations if c["batch_size"] == 64]

        assert all("gradient_accumulation" in c for c in small_batch)
        assert all("gradient_accumulation" not in c for c in large_batch)

    def test_multiple_conditional_parameters(self):
        """Test multiple conditional parameters."""
        grid = ParameterGrid("test")
        grid.add_parameter("optimizer", ["adam", "sgd"])
        grid.add_conditional_parameter("momentum", [0.9], when={"optimizer": "sgd"})
        grid.add_conditional_parameter(
            "betas", [(0.9, 0.999)], when={"optimizer": "adam"}
        )

        combinations = list(grid.generate_permutations())

        adam_combo = next(c for c in combinations if c["optimizer"] == "adam")
        sgd_combo = next(c for c in combinations if c["optimizer"] == "sgd")

        assert "betas" in adam_combo
        assert "momentum" not in adam_combo
        assert "momentum" in sgd_combo
        assert "betas" not in sgd_combo


class TestComputedParameters:
    """Test computed parameter functionality."""

    def test_computed_parameter_simple(self):
        """Test simple computed parameter."""
        grid = ParameterGrid("test")
        grid.add_parameter("num_epochs", [10, 20])
        grid.add_computed_parameter(
            "warmup_steps", lambda params: params["num_epochs"] * 10
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 2

        for combo in combinations:
            assert combo["warmup_steps"] == combo["num_epochs"] * 10

    def test_computed_parameter_complex(self):
        """Test computed parameter with complex logic."""
        grid = ParameterGrid("test")
        grid.add_parameter("batch_size", [32, 64])
        grid.add_parameter("accumulation_steps", [1, 2])
        grid.add_computed_parameter(
            "effective_batch_size", lambda p: p["batch_size"] * p["accumulation_steps"]
        )

        combinations = list(grid.generate_permutations())

        for combo in combinations:
            expected = combo["batch_size"] * combo["accumulation_steps"]
            assert combo["effective_batch_size"] == expected

    def test_computed_parameter_with_missing_dependency(self):
        """Test computed parameter handles missing dependencies gracefully."""
        grid = ParameterGrid("test")
        grid.add_parameter("lr", [0.01])
        grid.add_computed_parameter(
            "warmup_lr", lambda p: p.get("lr", 0) * p.get("warmup_factor", 0.1)
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 1
        assert combinations[0]["warmup_lr"] == 0.001


class TestParameterGenerators:
    """Test parameter generation functions."""

    def test_parameter_from_function(self):
        """Test adding parameters from a generator function."""
        grid = ParameterGrid("test")
        grid.add_parameter_from_function("lr", lambda: [0.1, 0.01, 0.001])

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 3
        assert {c["lr"] for c in combinations} == {0.1, 0.01, 0.001}

    def test_parameter_from_numpy_logspace(self):
        """Test using numpy to generate parameter values."""
        grid = ParameterGrid("test")
        grid.add_parameter_from_function(
            "lr", lambda: np.logspace(-4, -1, num=4).tolist()
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 4

        # Check values are in expected range
        lrs = [c["lr"] for c in combinations]
        assert all(1e-4 <= lr <= 1e-1 for lr in lrs)


class TestParameterDistributions:
    """Test parameter distribution sampling."""

    def test_uniform_distribution(self):
        """Test uniform distribution sampling."""
        np.random.seed(42)  # For reproducibility

        grid = ParameterGrid("test")
        grid.add_parameter_distribution(
            "dropout", "uniform", n_samples=5, low=0.1, high=0.5
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 5

        # Check all values are in range
        dropouts = [c["dropout"] for c in combinations]
        assert all(0.1 <= d <= 0.5 for d in dropouts)

    def test_loguniform_distribution(self):
        """Test log-uniform distribution sampling."""
        np.random.seed(42)

        grid = ParameterGrid("test")
        grid.add_parameter_distribution(
            "lr", "loguniform", n_samples=10, low=1e-5, high=1e-1
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 10

        # Check all values are in range
        lrs = [c["lr"] for c in combinations]
        assert all(1e-5 <= lr <= 1e-1 for lr in lrs)

    def test_normal_distribution(self):
        """Test normal distribution sampling."""
        np.random.seed(42)

        grid = ParameterGrid("test")
        grid.add_parameter_distribution(
            "weight_decay", "normal", n_samples=5, mean=0.01, std=0.005
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 5

        # Check values are reasonable (within 3 std)
        wds = [c["weight_decay"] for c in combinations]
        assert all(abs(wd - 0.01) <= 0.015 for wd in wds)

    def test_choice_distribution(self):
        """Test choice distribution with weights."""
        np.random.seed(42)

        grid = ParameterGrid("test")
        grid.add_parameter_distribution(
            "activation",
            "choice",
            n_samples=100,
            choices=["relu", "gelu", "silu"],
            weights=[0.5, 0.3, 0.2],
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 100

        # Check distribution roughly matches weights
        activations = [c["activation"] for c in combinations]
        relu_count = activations.count("relu")
        gelu_count = activations.count("gelu")
        silu_count = activations.count("silu")

        # With 100 samples, expect roughly 50, 30, 20
        assert 35 <= relu_count <= 65  # Allow some variance
        assert 20 <= gelu_count <= 40
        assert 10 <= silu_count <= 30


class TestParameterConstraints:
    """Test parameter constraint functionality."""

    def test_simple_constraint(self):
        """Test simple constraint function."""
        grid = ParameterGrid("test")
        grid.add_parameter("batch_size", [16, 32, 64])
        grid.add_parameter("accumulation_steps", [1, 2, 4])
        grid.add_constraint(lambda p: p["batch_size"] * p["accumulation_steps"] <= 64)

        combinations = list(grid.generate_permutations())

        # Check all combinations satisfy constraint
        for combo in combinations:
            assert combo["batch_size"] * combo["accumulation_steps"] <= 64

        # Check that invalid combinations are excluded
        invalid = {"batch_size": 64, "accumulation_steps": 4}
        assert invalid not in combinations

    def test_multiple_constraints(self):
        """Test multiple constraint functions."""
        grid = ParameterGrid("test")
        grid.add_parameter("lr", [0.001, 0.01, 0.1])
        grid.add_parameter("batch_size", [16, 32, 64])

        # Constraint 1: Large lr needs small batch size
        grid.add_constraint(lambda p: not (p["lr"] > 0.01 and p["batch_size"] > 32))
        # Constraint 2: Small lr needs large batch size
        grid.add_constraint(lambda p: not (p["lr"] < 0.01 and p["batch_size"] < 32))

        combinations = list(grid.generate_permutations())

        # Check constraints are satisfied
        for combo in combinations:
            if combo["lr"] > 0.01:
                assert combo["batch_size"] <= 32
            if combo["lr"] < 0.01:
                assert combo["batch_size"] >= 32


class TestExclusionPatterns:
    """Test parameter exclusion patterns."""

    def test_exclude_specific_combinations(self):
        """Test excluding specific parameter combinations."""
        grid = ParameterGrid("test")
        grid.add_parameter("optimizer", ["adam", "sgd"])
        grid.add_parameter("lr", [0.01, 0.1])
        grid.exclude_combinations(
            [
                {"optimizer": "sgd", "lr": 0.1}  # SGD with high lr is unstable
            ]
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 3

        # Check excluded combination is not present
        assert {"optimizer": "sgd", "lr": 0.1} not in combinations

    def test_exclude_with_function(self):
        """Test excluding combinations with a function."""
        grid = ParameterGrid("test")
        grid.add_parameter("dropout", [0.0, 0.1, 0.5])
        grid.add_parameter("weight_decay", [0.0, 0.01, 0.1])
        grid.exclude_combinations(
            [
                lambda p: p["dropout"] == 0
                and p["weight_decay"] == 0  # Need some regularization
            ]
        )

        combinations = list(grid.generate_permutations())

        # Check that no combination has both dropout and weight_decay as 0
        for combo in combinations:
            assert not (combo["dropout"] == 0 and combo["weight_decay"] == 0)

    def test_exclude_patterns_complex(self):
        """Test complex exclusion patterns."""
        grid = ParameterGrid("test")
        grid.add_parameter("model_size", ["small", "medium", "large"])
        grid.add_parameter("batch_size", [16, 32, 64])

        # Exclude incompatible combinations
        grid.exclude_combinations(
            [
                {"model_size": "large", "batch_size": 64},  # OOM
                lambda p: p["model_size"] == "small"
                and p["batch_size"] < 32,  # Inefficient
            ]
        )

        combinations = list(grid.generate_permutations())

        # Check exclusions
        assert {"model_size": "large", "batch_size": 64} not in combinations
        assert {"model_size": "small", "batch_size": 16} not in combinations


class TestComplexScenarios:
    """Test complex combinations of features."""

    def test_all_features_combined(self):
        """Test using multiple advanced features together."""
        grid = ParameterGrid("test")

        # Traditional parameters
        grid.add_parameter("dataset", ["mnist", "cifar10"])

        # Linked parameters (model configs)
        grid.add_parameter_sets(
            [
                {"model": "small", "hidden_size": 256, "num_layers": 2},
                {"model": "large", "hidden_size": 512, "num_layers": 4},
            ]
        )

        # Conditional parameters
        grid.add_conditional_parameter(
            "augmentation", ["basic", "advanced"], when={"dataset": "cifar10"}
        )

        # Computed parameters
        grid.add_computed_parameter(
            "total_params",
            lambda p: p.get("hidden_size", 0) * p.get("num_layers", 0) * 100,
        )

        # Constraints
        grid.add_constraint(
            lambda p: not (p.get("model") == "large" and p.get("dataset") == "mnist")
        )

        combinations = list(grid.generate_permutations())

        # Verify constraints and computed values
        for combo in combinations:
            # Check constraint
            if combo.get("model") == "large":
                assert combo.get("dataset") != "mnist"

            # Check computed parameter
            assert (
                combo["total_params"]
                == combo["hidden_size"] * combo["num_layers"] * 100
            )

            # Check conditional parameter
            if combo["dataset"] == "cifar10":
                assert "augmentation" in combo
            else:
                assert "augmentation" not in combo

    def test_grid_search_integration(self):
        """Test that new features work with ParameterGridSearch."""
        base_config = {
            "experiment_name": "test",
            "model": {"type": "transformer"},
            "training": {"max_epochs": 10},
            "data": {"dataset": "test"},
            "optimizer": {"type": "adam", "lr": 0.001},
        }

        search = ParameterGridSearch(base_config)
        grid = search.create_grid("advanced_test", "Testing advanced features")

        # Use advanced features
        grid.add_linked_parameters(
            ["optimizer.lr", "optimizer.weight_decay"], [(0.01, 0.1), (0.001, 0.01)]
        )
        grid.add_computed_parameter(
            "training.warmup_steps",
            lambda p: int(p.get("training.max_epochs", 10) * 100),
        )

        # Generate experiments
        experiments = list(search.generate_experiment_dicts())
        assert len(experiments) == 2

        # Check that parameters were applied correctly
        for exp in experiments:
            assert exp["training"]["warmup_steps"] == 1000
            if exp["optimizer"]["lr"] == 0.01:
                assert exp["optimizer"]["weight_decay"] == 0.1
            else:
                assert exp["optimizer"]["weight_decay"] == 0.01


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_grid_count(self):
        """Test that empty grid returns 0 combinations."""
        grid = ParameterGrid("empty")
        assert grid.get_parameter_count() == 0

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 0

    def test_unconditional_conditional_parameter_count(self):
        """Test that unconditional conditional parameters are counted correctly."""
        grid = ParameterGrid("test")

        # Add conditional parameter with no condition (always applies)
        grid.add_conditional_parameter("color", ["red", "blue"])

        # Should count as 2 combinations
        assert grid.get_parameter_count() == 2

        # Verify generation matches count
        combinations = list(grid.generate_permutations())
        assert len(combinations) == 2
        assert {"color": "red"} in combinations
        assert {"color": "blue"} in combinations

    def test_empty_choice_distribution_validation(self):
        """Test that empty choices list raises appropriate error."""
        grid = ParameterGrid("test")

        with pytest.raises(
            ValueError, match="Choice distribution requires non-empty 'choices'"
        ):
            grid.add_parameter_distribution(
                "activation", "choice", n_samples=5, choices=[]
            )

    def test_mismatched_weights_validation(self):
        """Test that mismatched weights length raises appropriate error."""
        grid = ParameterGrid("test")

        with pytest.raises(
            ValueError, match="Length of 'weights'.*must match length of 'choices'"
        ):
            grid.add_parameter_distribution(
                "activation",
                "choice",
                n_samples=5,
                choices=["relu", "gelu"],
                weights=[0.5, 0.3, 0.2],  # 3 weights for 2 choices
            )

    def test_deserialization_with_invalid_keys(self):
        """Test that deserialization handles invalid keys gracefully."""
        # Test data with invalid conditional spec (no keys)
        data = {
            "name": "test",
            "parameters": {},
            "parameter_specs": [
                {
                    "type": "conditional",
                    "keys": [],  # Invalid: empty keys
                    "values": ["value1"],
                    "condition": {},
                }
            ],
        }

        # Should skip invalid spec with warning
        grid = ParameterGrid.from_dict(data)
        assert len(grid.parameter_specs) == 0  # Invalid spec was skipped

    def test_deserialization_with_multiple_keys(self):
        """Test that deserialization rejects specs with multiple keys."""
        # Test data with invalid distribution spec (multiple keys)
        data = {
            "name": "test",
            "parameters": {},
            "parameter_specs": [
                {
                    "type": "distribution",
                    "keys": ["key1", "key2"],  # Invalid: multiple keys
                    "values": [0.1, 0.2],
                    "distribution": "uniform",
                    "n_samples": 2,
                }
            ],
        }

        # Should skip invalid spec with warning
        grid = ParameterGrid.from_dict(data)
        assert len(grid.parameter_specs) == 0  # Invalid spec was skipped


class TestBackwardCompatibility:
    """Ensure new features don't break existing functionality."""

    def test_traditional_api_still_works(self):
        """Test that the traditional API still works."""
        grid = ParameterGrid("test")
        grid.add_parameter("lr", [0.01, 0.001])
        grid.add_parameter("batch_size", [32, 64])

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 4

        # Check get_parameter_count still works
        assert grid.get_parameter_count() == 4

        # Check get_parameter_names still works
        assert grid.get_parameter_names() == {"lr", "batch_size"}

    def test_mixed_old_and_new_api(self):
        """Test mixing traditional and new API methods."""
        grid = ParameterGrid("test")

        # Traditional
        grid.add_parameter("epochs", [10, 20])

        # New
        grid.add_linked_parameters(
            ["lr", "scheduler"], [(0.01, "cosine"), (0.001, "linear")]
        )

        combinations = list(grid.generate_permutations())
        assert len(combinations) == 4

        # All combinations should have all three parameters
        for combo in combinations:
            assert "epochs" in combo
            assert "lr" in combo
            assert "scheduler" in combo
