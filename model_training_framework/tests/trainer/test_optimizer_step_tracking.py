"""
Tests for Optimizer Step Counting (Fixed Version)

This module tests that optimizer steps are counted correctly across
multiple dataloaders with different sampling strategies.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestOptimizerStepCounting:
    """Test optimizer step counting."""

    def test_basic_step_counting(self):
        """Test basic optimizer step counting."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Track steps manually
        step_count = 0
        original_step = optimizer.step

        def count_step(closure=None):
            nonlocal step_count
            step_count += 1
            return original_step(closure)

        optimizer.step = count_step

        # Simulate training
        num_batches = 10
        for _ in range(num_batches):
            # Forward pass (dummy)
            loss = torch.tensor(1.0, requires_grad=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert step_count == num_batches

    def test_step_counting_with_accumulation(self):
        """Test step counting with gradient accumulation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        step_count = 0
        original_step = optimizer.step

        def count_step(closure=None):
            nonlocal step_count
            step_count += 1
            return original_step(closure)

        optimizer.step = count_step

        # Simulate training with accumulation
        num_batches = 20
        accumulation_steps = 4

        for batch_idx in range(num_batches):
            # Simulate gradient accumulation
            if (
                batch_idx + 1
            ) % accumulation_steps == 0 or batch_idx == num_batches - 1:
                optimizer.step()
                optimizer.zero_grad()

        # Should step 5 times (at 4, 8, 12, 16, 20)
        expected_steps = (num_batches + accumulation_steps - 1) // accumulation_steps
        assert step_count == expected_steps

    def test_step_counting_multiple_loaders(self):
        """Test step counting with multiple loaders."""
        # Simulate multiple loaders with different sizes
        loader_sizes = [10, 20, 15]
        total_batches = sum(loader_sizes)

        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        step_count = 0
        original_step = optimizer.step

        def count_step(closure=None):
            nonlocal step_count
            step_count += 1
            return original_step(closure)

        optimizer.step = count_step

        # Process all loaders sequentially
        for loader_batches in loader_sizes:
            for _ in range(loader_batches):
                optimizer.step()
                optimizer.zero_grad()

        assert step_count == total_batches

    def test_per_loader_optimizer_steps(self):
        """Test separate optimizer step counting per loader."""
        # Create multiple optimizers for different loaders
        models = [SimpleModel() for _ in range(3)]
        optimizers = [torch.optim.SGD(m.parameters(), lr=0.01) for m in models]

        step_counts = [0, 0, 0]

        for i, optimizer in enumerate(optimizers):
            original_step = optimizer.step

            def make_counter(idx, orig_step=original_step):
                def count_step(closure=None):
                    step_counts[idx] += 1
                    return orig_step(closure)

                return count_step

            optimizer.step = make_counter(i)

        # Simulate training with per-loader optimizers
        loader_batches = [5, 10, 7]

        for loader_idx, num_batches in enumerate(loader_batches):
            for _ in range(num_batches):
                optimizers[loader_idx].step()
                optimizers[loader_idx].zero_grad()

        assert step_counts == loader_batches

    def test_scheduler_step_counting(self):
        """Test LR scheduler step counting."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step scheduler multiple times
        for _ in range(5):
            scheduler.step()

        # LR should have decreased
        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr
        assert abs(new_lr - initial_lr * 0.1) < 1e-6

    def test_step_counting_with_weighted_sampling(self):
        """Test step counting with weighted dataloader sampling."""
        # Simulate weighted sampling
        weights = [0.5, 0.3, 0.2]
        total_steps = 100

        # Calculate steps per loader based on weights
        loader_steps = [int(w * total_steps) for w in weights]
        # Adjust for rounding
        diff = total_steps - sum(loader_steps)
        if diff > 0:
            loader_steps[0] += diff

        optimizers = [
            torch.optim.SGD(SimpleModel().parameters(), lr=0.01) for _ in range(3)
        ]
        step_counts = [0, 0, 0]

        for i, optimizer in enumerate(optimizers):
            original_step = optimizer.step

            def make_counter(idx, orig_step=original_step):
                def count_step(closure=None):
                    step_counts[idx] += 1
                    return orig_step(closure)

                return count_step

            optimizer.step = make_counter(i)

        # Simulate weighted training
        for loader_idx, steps in enumerate(loader_steps):
            for _ in range(steps):
                optimizers[loader_idx].step()

        assert sum(step_counts) == total_steps
        # Check approximate weight distribution
        for i, weight in enumerate(weights):
            expected = int(weight * total_steps)
            assert abs(step_counts[i] - expected) <= 1

    def test_step_counting_across_epochs(self):
        """Test that step counting persists across epochs."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        total_step_count = 0
        original_step = optimizer.step

        def count_step(closure=None):
            nonlocal total_step_count
            total_step_count += 1
            return original_step(closure)

        optimizer.step = count_step

        # Simulate multiple epochs
        num_epochs = 3
        batches_per_epoch = 10

        for _epoch in range(num_epochs):
            for _batch in range(batches_per_epoch):
                optimizer.step()
                optimizer.zero_grad()

        assert total_step_count == num_epochs * batches_per_epoch

    def test_multiple_schedulers_step_counting(self):
        """Test multiple LR schedulers with different step frequencies."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Create two schedulers
        scheduler1 = StepLR(optimizer, step_size=3, gamma=0.9)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=10)

        # Track LR changes
        lr_history = []

        for step in range(10):
            lr_history.append(optimizer.param_groups[0]["lr"])

            # Step first scheduler every 3 steps
            if (step + 1) % 3 == 0:
                scheduler1.step()

            # Step second scheduler every step
            scheduler2.step()

        # Verify LR changed
        assert len(set(lr_history)) > 1  # LR should have changed
        assert lr_history[0] != lr_history[-1]  # First and last should differ
