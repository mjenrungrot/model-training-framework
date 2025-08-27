"""
Tests for multi-dataloader training functionality.

This module tests the GenericTrainer with multi-dataloader support:
- Multi-loader training and validation
- Per-loader optimizer routing
- Metrics aggregation strategies
- Gradient accumulation
- AMP support
- Validation frequency options
"""

from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:  # Imported for type hints only
    from torch.optim import Optimizer

from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    MultiDataLoaderConfig,
    PerformanceConfig,
    SamplingStrategy,
    ValAggregation,
    ValidationConfig,
    ValidationFrequency,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=4, hidden_size=8, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def create_dummy_dataloaders(
    num_loaders=2,
    samples_per_loader=20,
    batch_size=4,
    input_size=4,
    output_size=2,
):
    """Create dummy dataloaders for testing."""
    loaders = []
    for i in range(num_loaders):
        # Create slightly different data for each loader
        X = torch.randn(samples_per_loader, input_size) + i * 0.1
        y = torch.randint(0, output_size, (samples_per_loader,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loaders.append(loader)
    return loaders


class TestGenericTrainerMulti:
    """Test GenericTrainer with multi-dataloader support."""

    def test_trainer_initialization(self):
        """Test trainer initialization with multiple optimizers."""
        config = GenericTrainerConfig()
        model = SimpleModel()
        optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
        )

        assert trainer.config == config
        assert trainer.model == model
        assert len(trainer.optimizers) == 1
        assert trainer.dataloader_manager is None  # Not initialized until fit()

    def test_trainer_requires_training_step(self):
        """Test that trainer requires training step to be set."""
        config = GenericTrainerConfig()
        model = SimpleModel()
        optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
        )

        train_loaders = create_dummy_dataloaders(num_loaders=2)

        with pytest.raises(ValueError, match="Training step function not set"):
            trainer.fit(train_loaders)

    def test_trainer_requires_at_least_one_loader(self):
        """Test that trainer requires at least one training loader."""
        config = GenericTrainerConfig()
        model = SimpleModel()
        optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
        )

        # Set dummy training step
        trainer.set_training_step(lambda *args, **kwargs: {"loss": torch.tensor(1.0)})

        with pytest.raises(
            ValueError, match="At least one training dataloader required"
        ):
            trainer.fit([])

    def test_basic_training_loop(self):
        """Test basic training loop with multiple dataloaders."""
        config = GenericTrainerConfig()
        config.log_loss_every_n_steps = None  # Disable logging
        config.validate_every_n_epochs = 10  # No validation

        model = SimpleModel()
        optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
        )

        # Training step function
        def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            x, y = batch
            output = trainer.model(x)
            loss = nn.functional.cross_entropy(output, y)
            return {"loss": loss, "accuracy": 0.5}  # Dummy accuracy

        trainer.set_training_step(training_step)

        # Create dataloaders
        train_loaders = create_dummy_dataloaders(num_loaders=2, samples_per_loader=8)

        # Train for 1 epoch
        trainer.fit(train_loaders, max_epochs=1)

        assert trainer.current_epoch == 1
        assert trainer.global_step > 0

    def test_per_loader_optimizer_routing(self):
        """Test routing different loaders to different optimizers."""
        config = GenericTrainerConfig()
        config.per_loader_optimizer_id = [0, 1, 0]  # 3 loaders, 2 optimizers
        config.log_loss_every_n_steps = None

        model = SimpleModel()
        # Create two optimizers with different learning rates
        optimizers: list[Optimizer] = [
            optim.SGD(model.parameters(), lr=0.01),
            optim.Adam(model.parameters(), lr=0.001),
        ]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
        )

        # Mock optimizer step to track calls
        step_counts = [0, 0]
        original_steps = [opt.step for opt in optimizers]

        def make_mock_step(idx):
            def mock_step(closure=None):
                step_counts[idx] += 1
                return original_steps[idx](closure)

            return mock_step

        optimizers[0].step = make_mock_step(0)
        optimizers[1].step = make_mock_step(1)

        # Training step
        def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            x, y = batch
            output = trainer.model(x)
            loss = nn.functional.cross_entropy(output, y)
            return {"loss": loss}

        trainer.set_training_step(training_step)

        # Create 3 dataloaders
        train_loaders = create_dummy_dataloaders(num_loaders=3, samples_per_loader=8)

        # Train
        trainer.fit(train_loaders, max_epochs=1)

        # Both optimizers should have been used
        assert step_counts[0] > 0  # Used by loaders 0 and 2
        assert step_counts[1] > 0  # Used by loader 1

    def test_validation_with_aggregation(self):
        """Test validation with different aggregation strategies."""
        config = GenericTrainerConfig()
        config.validation = ValidationConfig(
            frequency=ValidationFrequency.PER_EPOCH,
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
            per_loader_metrics=True,
            global_metrics=True,
        )
        config.validate_every_n_epochs = 1
        config.log_loss_every_n_steps = None

        model = SimpleModel()
        optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
        )

        # Step functions
        def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            x, y = batch
            output = trainer.model(x)
            loss = nn.functional.cross_entropy(output, y)
            return {"loss": loss}

        def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            x, y = batch
            output = trainer.model(x)
            loss = nn.functional.cross_entropy(output, y)
            # Different loss per loader for testing aggregation
            loss = loss + dataloader_idx * 0.1
            return {"loss": loss}

        trainer.set_training_step(training_step)
        trainer.set_validation_step(validation_step)

        # Create dataloaders
        train_loaders = create_dummy_dataloaders(num_loaders=2, samples_per_loader=8)
        val_loaders = create_dummy_dataloaders(num_loaders=2, samples_per_loader=8)

        # Train and validate
        trainer.fit(train_loaders, val_loaders, max_epochs=1)

        # Should have completed validation
        assert trainer.current_epoch == 1

    def test_gradient_accumulation(self):
        """Test gradient accumulation across multiple steps."""
        config = GenericTrainerConfig()
        config.performance = PerformanceConfig(
            gradient_accumulation_steps=4,
            use_amp=False,
        )
        config.log_loss_every_n_steps = None

        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=cast("list[Optimizer]", [optimizer]),
        )

        # Track optimizer steps
        step_count = 0
        original_step = optimizer.step

        def mock_step(closure=None):
            nonlocal step_count
            step_count += 1
            return original_step(closure)

        optimizer.step = mock_step

        # Training step
        def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            x, y = batch
            output = trainer.model(x)
            loss = nn.functional.cross_entropy(output, y)
            return {"loss": loss}

        trainer.set_training_step(training_step)

        # Create dataloaders with 16 samples (4 batches of 4)
        train_loaders = create_dummy_dataloaders(
            num_loaders=1,
            samples_per_loader=16,
            batch_size=4,
        )

        # Train
        trainer.fit(train_loaders, max_epochs=1)

        # With 4 batches and accumulation_steps=4, should have 1 optimizer step
        assert step_count == 1

    def test_amp_training(self):
        """Test training with automatic mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for AMP test")

        config = GenericTrainerConfig()
        config.performance = PerformanceConfig(use_amp=True)
        config.log_loss_every_n_steps = None

        model = SimpleModel().cuda()
        optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
        )

        # Training step
        def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            x, y = batch
            x, y = x.cuda(), y.cuda()
            output = trainer.model(x)
            loss = nn.functional.cross_entropy(output, y)
            return {"loss": loss}

        trainer.set_training_step(training_step)

        # Create dataloaders
        train_loaders = create_dummy_dataloaders(num_loaders=1, samples_per_loader=8)

        # Train with AMP
        trainer.fit(train_loaders, max_epochs=1)

        assert trainer.global_step > 0

    def test_step_based_validation(self):
        """Test validation triggered every N steps."""
        config = GenericTrainerConfig()
        config.validation = ValidationConfig(
            frequency=ValidationFrequency.EVERY_N_STEPS,
            every_n_steps=2,
        )
        config.log_loss_every_n_steps = None

        model = SimpleModel()
        optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
        )

        # Track validation calls
        val_call_count = 0

        def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            x, y = batch
            output = trainer.model(x)
            loss = nn.functional.cross_entropy(output, y)
            return {"loss": loss}

        def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            nonlocal val_call_count
            val_call_count += 1
            x, y = batch
            output = trainer.model(x)
            loss = nn.functional.cross_entropy(output, y)
            return {"loss": loss}

        trainer.set_training_step(training_step)
        trainer.set_validation_step(validation_step)

        # Create small dataloaders
        train_loaders = create_dummy_dataloaders(
            num_loaders=1,
            samples_per_loader=8,
            batch_size=2,  # 4 steps per epoch
        )
        val_loaders = create_dummy_dataloaders(
            num_loaders=1,
            samples_per_loader=4,
            batch_size=2,
        )

        # Train
        trainer.fit(train_loaders, val_loaders, max_epochs=1)

        # Should have validated every 2 steps (4 steps total -> 2 validations)
        assert val_call_count > 0

    def test_loss_weights_per_loader(self):
        """Test applying different loss weights to different loaders."""
        config = GenericTrainerConfig()
        config.loss_weights_per_loader = [1.0, 2.0, 0.5]  # Different weights
        config.log_loss_every_n_steps = None

        model = SimpleModel()
        optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
        )

        # Training step that returns fixed loss
        def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            return {"loss": torch.tensor(1.0, requires_grad=True)}

        trainer.set_training_step(training_step)

        # Create 3 dataloaders
        train_loaders = create_dummy_dataloaders(num_loaders=3, samples_per_loader=4)

        # Train
        trainer.fit(train_loaders, max_epochs=1)

        # Training should complete without error
        assert trainer.global_step > 0

    def test_metrics_aggregation_strategies(self):
        """Test different validation aggregation strategies."""
        # Test MICRO_AVG_WEIGHTED_BY_SAMPLES
        config1 = GenericTrainerConfig()
        config1.validation = ValidationConfig(
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
            global_metrics=True,
        )

        # Test MACRO_AVG_EQUAL_LOADERS
        config2 = GenericTrainerConfig()
        config2.validation = ValidationConfig(
            aggregation=ValAggregation.MACRO_AVG_EQUAL_LOADERS,
            global_metrics=True,
        )

        # Test PRIMARY_METRIC_PER_LOADER
        config3 = GenericTrainerConfig()
        config3.validation = ValidationConfig(
            aggregation=ValAggregation.PRIMARY_METRIC_PER_LOADER,
            per_loader_metrics=True,
            global_metrics=False,
        )

        for config in [config1, config2, config3]:
            config.log_loss_every_n_steps = None

            model = SimpleModel()
            optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

            trainer = GenericTrainer(
                config=config,
                model=model,
                optimizers=optimizers,
            )

            # Step functions
            def training_step(
                trainer, batch, batch_idx, dataloader_idx, dataloader_name
            ):
                x, y = batch
                output = trainer.model(x)
                loss = nn.functional.cross_entropy(output, y)
                return {"loss": loss}

            def validation_step(
                trainer, batch, batch_idx, dataloader_idx, dataloader_name
            ):
                # Return different metrics per loader
                return {
                    "loss": torch.tensor(1.0 + dataloader_idx * 0.1),
                    "accuracy": 0.8 + dataloader_idx * 0.05,
                }

            trainer.set_training_step(training_step)
            trainer.set_validation_step(validation_step)

            # Create dataloaders
            train_loaders = create_dummy_dataloaders(
                num_loaders=2, samples_per_loader=4
            )
            val_loaders = create_dummy_dataloaders(num_loaders=2, samples_per_loader=4)

            # Train and validate
            trainer.fit(train_loaders, val_loaders, max_epochs=1)

            assert trainer.current_epoch == 1

    def test_multi_dataloader_scheduling(self):
        """Test different dataloader scheduling strategies."""
        strategies = [
            SamplingStrategy.ROUND_ROBIN,
            SamplingStrategy.SEQUENTIAL,
        ]

        for strategy in strategies:
            config = GenericTrainerConfig()
            config.train_loader_config = MultiDataLoaderConfig(
                sampling_strategy=strategy,
                dataloader_names=["loader0", "loader1"],
            )
            config.val_loader_config = MultiDataLoaderConfig()
            config.log_loss_every_n_steps = None

            model = SimpleModel()
            optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

            trainer = GenericTrainer(
                config=config,
                model=model,
                optimizers=optimizers,
            )

            # Track which loaders are used
            loader_usage = []

            def training_step(
                trainer,
                batch,
                batch_idx,
                dataloader_idx,
                dataloader_name,
                loader_usage=loader_usage,
            ):
                loader_usage.append(dataloader_idx)
                x, y = batch
                output = trainer.model(x)
                loss = nn.functional.cross_entropy(output, y)
                return {"loss": loss}

            trainer.set_training_step(training_step)

            # Create dataloaders
            train_loaders = create_dummy_dataloaders(
                num_loaders=2, samples_per_loader=8
            )

            # Train
            trainer.fit(train_loaders, max_epochs=1)

            # Check that both loaders were used
            assert 0 in loader_usage
            assert 1 in loader_usage

            # For SEQUENTIAL, loader 0 should be used before loader 1
            if strategy == SamplingStrategy.SEQUENTIAL:
                first_0 = loader_usage.index(0)
                first_1 = loader_usage.index(1)
                # Ensure loader 0 begins no later than loader 1
                assert first_0 <= first_1

    def test_early_stopping(self):
        """Test early stopping with multi-dataloader training."""
        config = GenericTrainerConfig()
        config.early_stopping_patience = 2
        config.early_stopping_metric = "val/loss"
        config.early_stopping_mode = "min"
        config.log_loss_every_n_steps = None

        model = SimpleModel()
        optimizers: list[Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=optimizers,
        )

        # Step functions that return increasing validation loss
        epoch_counter = [0]

        def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            x, y = batch
            output = trainer.model(x)
            loss = nn.functional.cross_entropy(output, y)
            return {"loss": loss}

        def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
            # Increasing loss to trigger early stopping
            loss = 1.0 + epoch_counter[0] * 0.5
            return {"loss": torch.tensor(loss)}

        trainer.set_training_step(training_step)
        trainer.set_validation_step(validation_step)

        # Override epoch end to increment counter
        original_train_epoch = trainer._train_epoch

        def mock_train_epoch(epoch):
            result = original_train_epoch(epoch)
            epoch_counter[0] += 1
            return result

        trainer._train_epoch = mock_train_epoch

        # Create dataloaders
        train_loaders = create_dummy_dataloaders(num_loaders=1, samples_per_loader=4)
        val_loaders = create_dummy_dataloaders(num_loaders=1, samples_per_loader=4)

        # Train - should stop early
        trainer.fit(train_loaders, val_loaders, max_epochs=10)

        # Should stop before max_epochs due to early stopping
        assert trainer.current_epoch < 10


class TestBatchSizeDetection:
    """Test batch size detection from various formats."""

    def test_tensor_batch(self):
        """Test batch size detection from tensor."""
        config = GenericTrainerConfig()
        model = SimpleModel()
        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=cast(
                "list[Optimizer]", [optim.SGD(model.parameters(), lr=0.01)]
            ),
        )

        batch = torch.randn(32, 10)
        assert trainer._get_batch_size(batch) == 32

    def test_tuple_batch(self):
        """Test batch size detection from tuple."""
        config = GenericTrainerConfig()
        model = SimpleModel()
        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=cast(
                "list[Optimizer]", [optim.SGD(model.parameters(), lr=0.01)]
            ),
        )

        batch = (torch.randn(16, 10), torch.randn(16, 5))
        assert trainer._get_batch_size(batch) == 16

    def test_dict_batch(self):
        """Test batch size detection from dict."""
        config = GenericTrainerConfig()
        model = SimpleModel()
        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=cast(
                "list[Optimizer]", [optim.SGD(model.parameters(), lr=0.01)]
            ),
        )

        # Test with 'input' key
        batch = {"input": torch.randn(8, 10), "target": torch.randn(8)}
        assert trainer._get_batch_size(batch) == 8

        # Test with 'x' key
        batch = {"x": torch.randn(4, 10), "y": torch.randn(4)}
        assert trainer._get_batch_size(batch) == 4

        # Test with first value fallback
        batch = {"features": torch.randn(12, 10), "labels": torch.randn(12)}
        assert trainer._get_batch_size(batch) == 12

    def test_nested_batch(self):
        """Test batch size detection from nested structures."""
        config = GenericTrainerConfig()
        model = SimpleModel()
        trainer = GenericTrainer(
            config=config,
            model=model,
            optimizers=cast(
                "list[Optimizer]", [optim.SGD(model.parameters(), lr=0.01)]
            ),
        )

        # Nested list
        batch = [[torch.randn(5, 10)]]
        assert trainer._get_batch_size(batch) == 5

        # Dict with tuple value
        batch = {"data": (torch.randn(7, 10), torch.randn(7, 5))}
        assert trainer._get_batch_size(batch) == 7
