"""Test suite for improved experiment naming with brackets and better abbreviations."""

import pytest

from model_training_framework.config.naming import ExperimentNaming
from model_training_framework.config.schemas import NamingStrategy


class TestImprovedNaming:
    """Test the improved naming system with brackets and abbreviations."""

    def test_bracket_formatting_simple(self):
        """Test that simple parameters use bracket formatting."""
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
        }
        name = ExperimentNaming.generate_name(
            "exp", params, NamingStrategy.PARAMETER_BASED
        )

        assert "exp_" in name
        assert "[batch_32]" in name
        assert "[lr_0.001]" in name
        # Check brackets are properly paired
        assert name.count("[") == name.count("]")

    def test_bracket_formatting_nested(self):
        """Test nested dictionary formatting with curly braces."""
        params = {
            "model": {
                "hidden_size": 256,
                "num_layers": 4,
            }
        }
        name = ExperimentNaming.generate_name(
            "exp", params, NamingStrategy.PARAMETER_BASED
        )

        assert "[model_{" in name
        assert "hidden.256" in name
        assert "layers.4" in name
        assert "}]" in name
        # Check both bracket types are properly paired
        assert name.count("[") == name.count("]")
        assert name.count("{") == name.count("}")

    def test_unknown_parameter_abbreviation(self):
        """Test 3-character abbreviation for unknown parameters."""
        # Test single unknown parameter
        assert (
            ExperimentNaming._abbreviate_unknown("some_unknown_param") == "som_unk_par"
        )
        assert (
            ExperimentNaming._abbreviate_unknown("disable_positional_encoding")
            == "dis_pos_enc"
        )
        assert (
            ExperimentNaming._abbreviate_unknown("learnable_temperature") == "lea_tem"
        )
        assert ExperimentNaming._abbreviate_unknown("pairwise_rank") == "pai_ran"

        # Test single word
        assert ExperimentNaming._abbreviate_unknown("temperature") == "tem"
        assert ExperimentNaming._abbreviate_unknown("xy") == "xy"

    def test_known_abbreviations_under_10_chars(self):
        """Test that all known abbreviations are 10 characters or less."""
        # Sample of critical abbreviations to test
        test_params = {
            "learning_rate": "lr",
            "batch_size": "batch",
            "gradient_accumulation_steps": "grad_accum",
            "early_stopping_patience": "es_pat",
            "monitor_metric": "mon_metric",
            "log_system_metrics": "log_sys",
            "persistent_workers": "persist_w",
            "validate_schedule_consistency": "val_sched",
        }

        for param, expected_abbrev in test_params.items():
            abbrev = ExperimentNaming._shorten_parameter_name(param)
            assert abbrev == expected_abbrev
            assert len(abbrev) <= 10, f"{param} -> {abbrev} is {len(abbrev)} chars"

    def test_no_abbreviation_collisions(self):
        """Test that common parameters don't have colliding abbreviations."""
        params_to_test = [
            "requeue_job",
            "requeue",
            "global_metrics",
            "log_global_metrics",
            "monitor_metric",
            "monitor_mode",
            "save_every_n_steps",
            "save_every_n_epochs",
        ]

        abbreviations = {}
        for param in params_to_test:
            abbrev = ExperimentNaming._shorten_parameter_name(param)
            # Check for collisions
            if abbrev in abbreviations:
                pytest.fail(
                    f"Collision: {param} and {abbreviations[abbrev]} both map to {abbrev}"
                )
            abbreviations[abbrev] = param

    def test_complex_nested_parameters(self):
        """Test complex nested parameters with multiple levels."""
        params = {
            "model": {
                "disable_positional_encoding": True,
                "learnable_temperature": False,
                "mlp_dim": 512,
                "num_heads": 8,
            },
            "optimizer": "adam",
            "learning_rate": 0.0001,
        }

        name = ExperimentNaming.generate_name(
            "sweep", params, NamingStrategy.PARAMETER_BASED
        )

        # Check structure
        assert "sweep_" in name
        assert "[lr_" in name
        assert "[model_{" in name
        assert "[optim_adam]" in name

        # Check nested model params are abbreviated
        assert "dis_pos_enc.T" in name  # disable_positional_encoding -> dis_pos_enc
        assert "lea_tem.F" in name  # learnable_temperature -> lea_tem
        assert "mlp_dim.512" in name
        assert "num_hea.8" in name  # num_heads -> num_hea

    def test_flattened_parameter_keys_are_grouped(self):
        """Flat keys like 'model.num_layers' should group under one bracket."""
        params = {
            "model.num_layers": 4,
            "model.mlp_dim": 512,
            "optimizer.lr": 0.001,
        }

        name = ExperimentNaming.generate_name(
            "sweep", params, NamingStrategy.PARAMETER_BASED
        )

        # Only one model bracket should appear
        assert name.count("[model_") == 1
        assert "[model_{" in name
        assert "layers.4" in name
        assert "mlp_dim.512" in name
        # Ensure flattened keys are not emitted separately
        assert "[model." not in name

    def test_boolean_value_formatting(self):
        """Test boolean values are formatted as T/F."""
        params = {
            "use_amp": True,
            "deterministic": False,
            "enable_gradient_monitor": True,
        }

        name = ExperimentNaming.generate_name(
            "test", params, NamingStrategy.PARAMETER_BASED
        )

        assert "[amp_T]" in name
        assert "[determ_F]" in name
        assert "[en_grad_mo_T]" in name

    def test_float_formatting(self):
        """Test float values are formatted correctly."""
        params = {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "dropout": 0.5,
            "epsilon": 1e-8,
        }

        name = ExperimentNaming.generate_name(
            "test", params, NamingStrategy.PARAMETER_BASED
        )

        assert "[lr_0.001]" in name
        assert "[wdecay_1.0e-4]" in name  # Scientific notation for small values
        assert "[dropout_0.5]" in name
        assert "[eps_1.0e-8]" in name

    def test_string_value_formatting(self):
        """Test string values are sanitized and truncated."""
        params = {
            "optimizer": "adamw",
            "dataset_name": "very_long_dataset_name_that_should_be_truncated",
            "mode": "train/val",  # Contains invalid char
        }

        name = ExperimentNaming.generate_name(
            "test", params, NamingStrategy.PARAMETER_BASED
        )

        assert "[optim_adamw]" in name
        assert (
            "[dataset_very_long]" in name
        )  # Truncated to 10 chars ("very_long_" -> "very_long")
        assert "[mod_train_val]" in name  # "/" replaced with "_"

    def test_list_value_formatting(self):
        """Test list/tuple values are formatted correctly."""
        params = {
            "milestones": [10, 20, 30],
            "betas": (0.9, 0.999),
        }

        name = ExperimentNaming.generate_name(
            "test", params, NamingStrategy.PARAMETER_BASED
        )

        assert "[betas_0.9_0.999]" in name
        assert "[milestones_10_20_30]" in name

    def test_name_length_constraint(self):
        """Test that long names are properly truncated."""
        # Create params that would generate a very long name
        params = {f"param_{i}": i for i in range(50)}

        name = ExperimentNaming.generate_name(
            "very_long_experiment_base_name", params, NamingStrategy.PARAMETER_BASED
        )

        # Check name doesn't exceed max length
        assert len(name) <= ExperimentNaming.MAX_NAME_LENGTH
        # Brackets should remain balanced after truncation
        assert name.count("[") == name.count("]")
        # Check hash is added for uniqueness when truncated
        if len(name) == ExperimentNaming.MAX_NAME_LENGTH:
            # Should end with underscore and 8-char hex hash
            suffix = name[-9:]
            assert suffix[0] == "_" and all(c in "0123456789abcdef" for c in suffix[1:])

    def test_special_parameter_handling(self):
        """Test special handling of specific parameter types."""
        params = {
            # SLURM parameters should not collide
            "requeue": True,
            "requeue_job": False,
            # Similar names that could be confusing
            "log_loss_every_n_steps": 100,
            "log_scalars_every_n_steps": 50,
            # Multi-dataloader params
            "dataloader_names": ["train", "val"],
            "dataloader_weights": [0.7, 0.3],
        }

        name = ExperimentNaming.generate_name(
            "test", params, NamingStrategy.PARAMETER_BASED
        )

        # Check distinct abbreviations are used
        assert "[slurm_rq_T]" in name  # requeue -> slurm_rq
        assert "[req_job_F]" in name  # requeue_job -> req_job
        assert "[log_loss_n_100]" in name
        assert "[log_scal_n_50]" in name
        assert "[dl_names_train_val]" in name
        assert "[dl_weights_0.7_0.3]" in name

    def test_empty_parameters(self):
        """Test handling of empty parameters."""
        name = ExperimentNaming.generate_name(
            "test", {}, NamingStrategy.PARAMETER_BASED
        )
        assert name == "test"

    def test_sanitization_preserves_readability(self):
        """Test that name sanitization preserves readability."""
        # Test with problematic characters
        base_name = "test:experiment/2024<>pipeline"
        params = {"model/type": "transformer"}

        name = ExperimentNaming.generate_name(
            base_name, params, NamingStrategy.PARAMETER_BASED
        )

        # Check invalid characters are replaced
        assert ":" not in name
        assert "/" not in name
        assert "<" not in name
        assert ">" not in name
        # But result should still be readable
        assert "test_experiment_2024_pipeline" in name

    def test_parameter_ordering_consistency(self):
        """Test that parameter order is consistent (alphabetical)."""
        params1 = {"zebra": 1, "apple": 2, "monkey": 3}
        params2 = {"apple": 2, "monkey": 3, "zebra": 1}

        name1 = ExperimentNaming.generate_name(
            "test", params1, NamingStrategy.PARAMETER_BASED
        )
        name2 = ExperimentNaming.generate_name(
            "test", params2, NamingStrategy.PARAMETER_BASED
        )

        # Names should be identical regardless of input order
        assert name1 == name2
        # Apple should come first alphabetically
        assert name1.index("[app") < name1.index("[mon") < name1.index("[zeb")


class TestNamingEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_values(self):
        """Test handling of None values in parameters."""
        params = {
            "learning_rate": None,
            "batch_size": 32,
        }

        name = ExperimentNaming.generate_name(
            "test", params, NamingStrategy.PARAMETER_BASED
        )

        assert "[batch_32]" in name
        assert "[lr_None]" in name

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        params = {
            "experiment_name": "测试",  # Chinese characters
            "description": "café",  # Accented characters
        }

        # Should not raise an error
        name = ExperimentNaming.generate_name(
            "test", params, NamingStrategy.PARAMETER_BASED
        )
        assert "test_" in name

    def test_very_deep_nesting(self):
        """Test handling of deeply nested structures."""
        params = {"level1": {"level2": {"level3": {"value": 42}}}}

        name = ExperimentNaming.generate_name(
            "test", params, NamingStrategy.PARAMETER_BASED
        )

        # Should handle deep nesting gracefully
        assert "[lev" in name  # level1 abbreviated
        assert "42" in name

    def test_circular_reference_prevention(self):
        """Test that circular references don't cause infinite recursion."""
        # Note: In practice, this would be caught earlier, but test defensive coding
        params = {"a": 1, "b": 2}
        # Can't create actual circular ref in dict, but test large recursion

        for _ in range(100):
            params = {"nested": params}

        # Should not raise RecursionError
        name = ExperimentNaming.generate_name(
            "test", params, NamingStrategy.PARAMETER_BASED
        )
        assert len(name) <= ExperimentNaming.MAX_NAME_LENGTH

    def test_conflicting_keys_data_loss(self):
        """Test that conflicting keys (scalar vs nested) are handled properly.

        This tests the case where a key is used both as a scalar value and as a
        prefix for nested keys, which can lead to silent data loss.
        """
        # Case 1: Direct conflict - 'a' is both a scalar and a parent key
        params = {
            "a": 1,
            "a.b": 2,
        }

        # This should either preserve both values or raise an error
        # Currently it silently loses the scalar value "a": 1
        with pytest.raises(ValueError, match="Conflicting key"):
            ExperimentNaming.generate_name(
                "test", params, NamingStrategy.PARAMETER_BASED
            )

    def test_conflicting_keys_complex(self):
        """Test more complex conflicting key scenarios."""
        # Case 2: Multiple levels of conflict
        params = {
            "model": "transformer",  # Scalar value
            "model.layers": 4,  # Nested value with same prefix
            "model.hidden": 256,  # Another nested value
        }

        # Should detect the conflict between scalar "model" and nested "model.*"
        with pytest.raises(ValueError, match="Conflicting key"):
            ExperimentNaming.generate_name(
                "test", params, NamingStrategy.PARAMETER_BASED
            )

    def test_long_base_name_uniqueness(self):
        """Test that experiments with long base names but different parameters get unique names.

        When the base name is too long (≥112 chars), the naming system should still
        ensure uniqueness by including parameter information in the hash.
        """
        # Create a base name that is exactly 112 characters (will exceed limit with underscore)
        long_base = "a" * 112

        # Two different parameter sets
        params1 = {"learning_rate": 0.001, "batch_size": 32}
        params2 = {"learning_rate": 0.01, "batch_size": 64}

        name1 = ExperimentNaming.generate_name(
            long_base, params1, NamingStrategy.PARAMETER_BASED
        )
        name2 = ExperimentNaming.generate_name(
            long_base, params2, NamingStrategy.PARAMETER_BASED
        )

        # Both should respect the max length
        assert len(name1) <= ExperimentNaming.MAX_NAME_LENGTH
        assert len(name2) <= ExperimentNaming.MAX_NAME_LENGTH

        # Extract the base and hash portions
        # When base is too long, format should be: truncated_base_hash
        base1 = name1[:-9]  # Everything except underscore and 8-char hash
        hash1 = name1[-8:]  # Last 8 chars (the hash)
        base2 = name2[:-9]
        hash2 = name2[-8:]

        # The base portions should be IDENTICAL (same truncation of the long base name)
        assert base1 == base2, (
            f"Base portions should be identical for same long base name.\n"
            f"Got base1: {base1}\nGot base2: {base2}"
        )

        # The hash portions should be DIFFERENT (because parameters differ)
        assert hash1 != hash2, (
            f"Hash portions should differ for different parameters.\n"
            f"Got hash1: {hash1}\nGot hash2: {hash2}"
        )

        # Verify both are valid hex strings
        assert all(c in "0123456789abcdef" for c in hash1), f"Invalid hash1: {hash1}"
        assert all(c in "0123456789abcdef" for c in hash2), f"Invalid hash2: {hash2}"
