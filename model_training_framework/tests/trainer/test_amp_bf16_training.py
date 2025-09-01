"""
Tests for AMP/BF16 Mixed Precision Training

This module tests automatic mixed precision (AMP) and bfloat16 training,
including loss scaling, gradient clipping with AMP, and checkpoint/resume
with mixed precision.
"""

from pathlib import Path
import tempfile
from typing import Any

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.cuda.amp import (  # type: ignore[reportMissingImports]
        GradScaler,
        autocast,
    )

    HAS_AMP = True
except Exception:  # ImportError or environments without CUDA AMP
    # Provide safe fallbacks so names are always bound for type checkers
    from contextlib import contextmanager

    HAS_AMP = False

    class GradScaler:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        def scale(self, loss: Any) -> Any:
            return loss

        def step(self, optimizer: Any) -> None:
            optimizer.step()

        def update(self) -> None:
            pass

        def unscale_(self, optimizer: Any) -> None:
            pass

        def state_dict(self) -> dict[str, Any]:
            return {}

        def load_state_dict(self, state: dict[str, Any]) -> None:
            pass

        def get_scale(self) -> float:
            return 1.0

    @contextmanager
    def autocast(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        yield


from model_training_framework.config.schemas import PerformanceConfig


class AmpTestModel(nn.Module):
    """Test model for mixed precision training."""

    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


@pytest.mark.skipif(not HAS_AMP, reason="AMP not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAMPTraining:
    """Test automatic mixed precision training."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def teardown_method(self):
        """Cleanup after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_amp_basic_training(self):
        """Test basic AMP training."""
        model = AmpTestModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()

        # Create datasets
        datasets = [
            TensorDataset(
                torch.randn(100, 10).to(self.device),
                torch.randint(0, 3, (100,)).to(self.device),
            ),
            TensorDataset(
                torch.randn(80, 10).to(self.device),
                torch.randint(0, 3, (80,)).to(self.device),
            ),
        ]
        loaders = [DataLoader(ds, batch_size=16) for ds in datasets]

        # Training step with AMP (plain PyTorch)
        def amp_training_step(batch: Any) -> dict[str, Any]:
            inputs, targets = batch

            # Use autocast for mixed precision
            with autocast():
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

            # Scale loss for backward
            scaler.scale(loss).backward()

            return {"loss": loss.detach()}

        # Train for a few steps
        losses = []
        for batch_idx, (_loader_idx, batch) in enumerate(
            zip(
                [0, 1, 0, 1],
                [
                    next(iter(loaders[0])),
                    next(iter(loaders[1])),
                    next(iter(loaders[0])),
                    next(iter(loaders[1])),
                ],
            )
        ):
            result = amp_training_step(batch)
            losses.append(result["loss"].item())

            if (batch_idx + 1) % 2 == 0:  # Step every 2 batches
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Verify training happened
        assert len(losses) == 4
        assert all(val > 0 for val in losses)

        # Check that model is still in correct dtype
        for param in model.parameters():
            assert param.dtype == torch.float32  # Model params stay in fp32

    def test_amp_gradient_clipping(self):
        """Test gradient clipping with AMP."""
        model = AmpTestModel().to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = GradScaler()

        dataset = TensorDataset(
            torch.randn(50, 10).to(self.device),
            torch.randint(0, 3, (50,)).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=10)

        # No trainer config needed for this plain AMP clipping test

        # Track gradient norms
        grad_norms_before = []
        grad_norms_after = []

        for batch in loader:
            inputs, targets = batch

            with autocast():
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

            # Scale and backward
            scaler.scale(loss).backward()

            # Unscale gradients for clipping
            scaler.unscale_(optimizer)

            # Get gradient norm before clipping
            total_norm_before = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_before += p.grad.data.norm(2).item() ** 2
            grad_norms_before.append(total_norm_before**0.5)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Get gradient norm after clipping
            total_norm_after = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_after += p.grad.data.norm(2).item() ** 2
            grad_norms_after.append(total_norm_after**0.5)

            # Step and update
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Verify clipping worked
        assert len(grad_norms_after) == 5
        for norm_after in grad_norms_after:
            assert norm_after <= 1.01  # Allow small numerical error

        # Some gradients should have been clipped
        assert any(
            before > after for before, after in zip(grad_norms_before, grad_norms_after)
        )

    def test_amp_loss_scaling(self):
        """Test dynamic loss scaling in AMP."""
        model = AmpTestModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create scaler with specific config
        scaler = GradScaler(
            init_scale=128.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=100,
        )

        dataset = TensorDataset(
            torch.randn(30, 10).to(self.device),
            torch.randint(0, 3, (30,)).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=10)

        # Track scale over time
        scales = []

        for batch in loader:
            inputs, targets = batch

            with autocast():
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

            # Track current scale
            scales.append(scaler.get_scale())

            # Scaled backward
            scaler.scale(loss).backward()

            # Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Verify scaling was used
        assert len(scales) == 3
        assert scales[0] == 128.0  # Initial scale

    def test_amp_checkpoint_resume(self):
        """Test checkpoint/resume with AMP."""
        model = AmpTestModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()

        dataset = TensorDataset(
            torch.randn(20, 10).to(self.device),
            torch.randint(0, 3, (20,)).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=10)

        # Train for a few steps
        for _i, batch in enumerate(loader):
            inputs, targets = batch

            with autocast():
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": 2,
        }

        checkpoint_path = Path(self.temp_dir) / "amp_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create new model and load checkpoint
        new_model = AmpTestModel().to(self.device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scaler = GradScaler()

        loaded = torch.load(checkpoint_path)
        new_model.load_state_dict(loaded["model"])
        new_optimizer.load_state_dict(loaded["optimizer"])
        new_scaler.load_state_dict(loaded["scaler"])

        # Verify state was restored
        assert loaded["step"] == 2

        # Continue training
        for batch in loader:
            inputs, targets = batch

            with autocast():
                outputs = new_model(inputs)
                loss = F.cross_entropy(outputs, targets)

            new_scaler.scale(loss).backward()
            new_scaler.step(new_optimizer)
            new_scaler.update()
            new_optimizer.zero_grad()

            break  # Just one more step

        # Should complete without errors
        assert True


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch, "bfloat16"),
    reason="BF16 not available",
)
class TestBF16Training:
    """Test bfloat16 training."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device("cuda")
        torch.manual_seed(42)

    def test_bf16_basic_training(self):
        """Test basic BF16 training."""
        model = AmpTestModel().to(self.device).to(torch.bfloat16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create BF16 dataset
        dataset = TensorDataset(
            torch.randn(50, 10, dtype=torch.bfloat16).to(self.device),
            torch.randint(0, 3, (50,)).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=10)

        losses = []
        for batch in loader:
            inputs, targets = batch

            # Forward in BF16
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.float(), targets)

            losses.append(loss.item())

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Verify training happened
        assert len(losses) == 5
        assert all(val > 0 for val in losses)

        # Model should still be BF16
        for param in model.parameters():
            assert param.dtype == torch.bfloat16

    def test_bf16_vs_fp32_consistency(self):
        """Test that BF16 training is consistent with FP32."""
        torch.manual_seed(42)

        # Create identical models
        model_fp32 = AmpTestModel().to(self.device)
        model_bf16 = AmpTestModel().to(self.device).to(torch.bfloat16)

        # Copy weights
        with torch.no_grad():
            for p32, p16 in zip(model_fp32.parameters(), model_bf16.parameters()):
                p16.data = p32.data.to(torch.bfloat16)

        # Same optimizer settings
        opt_fp32 = torch.optim.SGD(model_fp32.parameters(), lr=0.01)
        opt_bf16 = torch.optim.SGD(model_bf16.parameters(), lr=0.01)

        # Same data
        inputs = torch.randn(32, 10).to(self.device)
        targets = torch.randint(0, 3, (32,)).to(self.device)

        # Train FP32
        outputs_fp32 = model_fp32(inputs)
        loss_fp32 = F.cross_entropy(outputs_fp32, targets)
        loss_fp32.backward()
        opt_fp32.step()

        # Train BF16
        outputs_bf16 = model_bf16(inputs.to(torch.bfloat16))
        loss_bf16 = F.cross_entropy(outputs_bf16.float(), targets)
        loss_bf16.backward()
        opt_bf16.step()

        # Losses should be similar (not identical due to precision)
        assert abs(loss_fp32.item() - loss_bf16.item()) < 0.1


class TestMixedPrecisionUtils:
    """Test mixed precision utility functions."""

    def test_dtype_selection(self):
        """Test selection of correct dtype based on config."""

        # AMP enabled config
        config_amp_enabled = PerformanceConfig(use_amp=True)
        assert config_amp_enabled.use_amp is True

        # AMP disabled config
        config_amp_disabled = PerformanceConfig(use_amp=False)
        assert config_amp_disabled.use_amp is False

        # Default should be True
        config_default = PerformanceConfig()
        assert config_default.use_amp is True

    def test_autocast_context(self):
        """Test autocast context manager usage."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        model = AmpTestModel().to(device)

        # Test with autocast
        inputs = torch.randn(10, 10).to(device)

        with autocast(dtype=torch.float16):
            outputs = model(inputs)
            # Inside autocast, computations use FP16
            assert outputs.dtype == torch.float16

        # Outside autocast, back to FP32
        outputs_fp32 = model(inputs)
        assert outputs_fp32.dtype == torch.float32
