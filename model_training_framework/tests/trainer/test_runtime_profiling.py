"""
Tests for lightweight runtime profiling feature.

Covers:
- Forward/backward timing logs when enabled
- Optimizer timing via RuntimeProfilingHook
- Data fetch timing from multi-dataloader iterator
- Validation forward timing
- Respect for scalar log frequency
- DDP primary-only logging behavior (unit-level for hook)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
    LoggingConfig,
    MultiDataLoaderConfig,
)
from model_training_framework.trainer.hooks import RuntimeProfilingHook
from model_training_framework.trainer.loggers import LoggerProtocol


class DummyLogger(LoggerProtocol):
    def __init__(self):
        self.logged: list[tuple[int, dict[str, Any]]] = []

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:  # type: ignore[override]
        self.logged.append((step, metrics))

    def log_epoch_summary(
        self, epoch: int, summary: dict[str, Any]
    ) -> None:  # pragma: no cover - not used
        pass

    def log_loader_proportions(
        self, epoch: int, proportions: dict[str, float], counts: dict[str, int]
    ) -> None:  # pragma: no cover - not used
        pass

    def log_text(
        self, key: str, text: str, step: int | None = None
    ) -> None:  # pragma: no cover - not used
        pass

    def close(self) -> None:  # pragma: no cover - not used
        pass


def make_trainer(
    num_loaders: int = 1, samples: int = 8, batch_size: int = 4, profile: bool = True
) -> tuple[GenericTrainer, list[DataLoader]]:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizers: list[optim.Optimizer] = [optim.SGD(model.parameters(), lr=0.01)]

    cfg = GenericTrainerConfig()
    cfg.profile_training = profile
    cfg.log_loss_every_n_steps = None
    cfg.logging = LoggingConfig(logger_type="console", log_scalars_every_n_steps=None)

    # Provide names for loaders
    cfg.train_loader_config = MultiDataLoaderConfig(
        dataloader_names=[f"loader{i}" for i in range(num_loaders)]
    )
    cfg.val_loader_config = MultiDataLoaderConfig(
        dataloader_names=[f"val_loader{i}" for i in range(num_loaders)]
    )

    trainer = GenericTrainer(cfg, model, optimizers)
    trainer.logger = DummyLogger()  # Override with dummy logger

    # Create loaders
    loaders: list[DataLoader] = []
    for i in range(num_loaders):
        X = torch.randn(samples, 4) + i * 0.01
        y = torch.randint(0, 2, (samples,))
        ds = TensorDataset(X, y)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=False))

    # Define simple steps
    def train_step(tr: GenericTrainer, batch, batch_idx, loader_idx, loader_name):
        x, y = batch
        out = tr.model(x)
        loss = nn.functional.cross_entropy(out, y)
        return {"loss": loss}

    def val_step(tr: GenericTrainer, batch, batch_idx, loader_idx, loader_name):
        x, y = batch
        out = tr.model(x)
        loss = nn.functional.cross_entropy(out, y)
        return {"loss": loss}

    trainer.set_training_step(train_step)
    trainer.set_validation_step(val_step)
    return trainer, loaders


def collect_profile_keys(logger: DummyLogger) -> set[str]:
    keys: set[str] = set()
    for _step, metrics in logger.logged:
        for k in metrics:
            if isinstance(k, str) and k.startswith("profile/"):
                keys.add(k)
    return keys


def test_profiling_enabled_single_dataloader():
    trainer, loaders = make_trainer(
        num_loaders=1, samples=8, batch_size=4, profile=True
    )
    # Train one epoch
    trainer.fit(loaders, max_epochs=1)

    keys = collect_profile_keys(trainer.logger)  # type: ignore[arg-type]
    assert any("profile/train/dl_loader0/time_forward_ms" in k for k in keys)
    assert any("profile/train/dl_loader0/time_backward_ms" in k for k in keys)
    assert any("profile/train/dl_loader0/time_optimizer_ms" in k for k in keys)
    assert any("profile/train/dl_loader0/time_data_ms" in k for k in keys)


def test_profiling_enabled_multi_dataloader():
    trainer, loaders = make_trainer(
        num_loaders=2, samples=4, batch_size=2, profile=True
    )
    trainer.fit(loaders, max_epochs=1)

    keys = collect_profile_keys(trainer.logger)  # type: ignore[arg-type]
    assert any("dl_loader0/time_forward_ms" in k for k in keys)
    assert any("dl_loader1/time_forward_ms" in k for k in keys)


def test_weighted_sampling_profiling():
    # Ensure profiling works under weighted scheduling
    trainer, loaders = make_trainer(
        num_loaders=2, samples=6, batch_size=3, profile=True
    )
    # Configure weighted sampling
    trainer.config.train_loader_config.sampling_strategy = (
        trainer.config.train_loader_config.sampling_strategy.WEIGHTED
    )
    trainer.config.train_loader_config.dataloader_weights = [0.7, 0.3]

    trainer.fit(loaders, max_epochs=1)

    keys = collect_profile_keys(trainer.logger)  # type: ignore[arg-type]
    assert any("dl_loader0/time_forward_ms" in k for k in keys)
    assert any("dl_loader1/time_forward_ms" in k for k in keys)


def test_profiling_disabled_zero_metrics():
    trainer, loaders = make_trainer(
        num_loaders=1, samples=4, batch_size=2, profile=False
    )
    trainer.fit(loaders, max_epochs=1)
    keys = collect_profile_keys(trainer.logger)  # type: ignore[arg-type]
    assert not keys  # No profile/* metrics expected


def test_validation_profiling_metrics():
    trainer, loaders = make_trainer(
        num_loaders=1, samples=4, batch_size=2, profile=True
    )
    # Also supply val loader
    trainer.fit(loaders, val_loaders=loaders, max_epochs=1)

    keys = collect_profile_keys(trainer.logger)  # type: ignore[arg-type]
    assert any("profile/val/dl_val_loader0/time_forward_ms" in k for k in keys)
    assert any("profile/val/dl_val_loader0/time_data_ms" in k for k in keys)


def test_profiling_respects_log_frequency():
    # Two optimizer steps, log every 2 -> only second step should log optimizer time
    trainer, loaders = make_trainer(
        num_loaders=1, samples=8, batch_size=4, profile=True
    )
    # Adjust frequency to 2
    trainer.config.logging.log_scalars_every_n_steps = 2
    trainer.fit(loaders, max_epochs=1)

    # Extract optimizer timing steps
    opt_steps = [
        step
        for step, metrics in trainer.logger.logged  # type: ignore[attr-defined]
        if any(k.endswith("time_optimizer_ms") for k in metrics)
    ]
    # Expect steps to include only multiples of 2 (e.g., step==2)
    assert all(step % 2 == 0 for step in opt_steps)
    assert len(opt_steps) >= 1


def test_dataloader_exhaustion_handling():
    # One short loader and one longer; no cycling to force exhaustion path
    trainer, _ = make_trainer(num_loaders=2, samples=6, batch_size=3, profile=True)
    # Build loaders with differing lengths
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X0 = torch.randn(3, 4)  # one batch
    y0 = torch.randint(0, 2, (3,))
    X1 = torch.randn(9, 4)  # three batches
    y1 = torch.randint(0, 2, (9,))
    dl0 = DataLoader(TensorDataset(X0, y0), batch_size=3, shuffle=False)
    dl1 = DataLoader(TensorDataset(X1, y1), batch_size=3, shuffle=False)

    # Disable cycling so exhaustion path triggers
    trainer.config.train_loader_config.cycle_short_loaders = False

    trainer.fit([dl0, dl1], max_epochs=1)

    keys = collect_profile_keys(trainer.logger)  # type: ignore[arg-type]
    # Should have logged some metrics for the longer loader at least
    assert any("profile/train/dl_loader1/time_forward_ms" in k for k in keys)


def test_ddp_primary_only_in_hook():
    # Unit-test RuntimeProfilingHook's primary-only gate
    hook = RuntimeProfilingHook()
    trainer = MagicMock()

    # Fake non-primary fabric
    class _Fabric:
        is_global_zero = False

    trainer.fabric = _Fabric()
    trainer.global_step = 1
    trainer.current_dataloader_name = "loader0"
    # Logger that would raise if called
    bad_logger = MagicMock()
    trainer.logger = bad_logger

    hook.on_before_optimizer_step(trainer, 0)
    hook.on_after_optimizer_step(trainer, 0)

    # Ensure no logging when not primary
    assert not bad_logger.log_metrics.called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_profiling_no_error():
    # Sanity check that profiling with CUDA model does not error
    trainer, loaders = make_trainer(
        num_loaders=1, samples=4, batch_size=2, profile=True
    )
    trainer.model = trainer.model.cuda()
    trainer.fit(loaders, max_epochs=1)
    # At least one profile metric should be logged
    keys = collect_profile_keys(trainer.logger)  # type: ignore[arg-type]
    assert keys
