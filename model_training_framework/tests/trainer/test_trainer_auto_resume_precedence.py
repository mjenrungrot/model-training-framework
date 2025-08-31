"""
Tests for auto-resume precedence and explicit resume behavior in GenericTrainer.fit().

Covers three behaviors:
1) No resume when no latest and no resume_from_checkpoint provided
2) Auto-resume from latest checkpoint when available
3) Explicit resume from user-provided checkpoint when auto-resume is disabled
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from model_training_framework.trainer import GenericTrainer, GenericTrainerConfig


class _SimpleModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=8, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def _create_loaders(
    samples_per_loader: int = 4, batch_size: int = 2, num_loaders: int = 2
):
    loaders = []
    input_size = 4
    output_size = 2
    for i in range(num_loaders):
        X = torch.randn(samples_per_loader, input_size) + i * 0.1
        y = torch.randint(0, output_size, (samples_per_loader,))
        ds = TensorDataset(X, y)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=False))
    return loaders


def _make_trainer(tmp_path):
    config = GenericTrainerConfig()
    # Minimize noise, focus on training epochs
    config.log_loss_every_n_steps = None
    config.validate_every_n_epochs = 10  # Do not run validation in these tests
    config.checkpoint.root_dir = tmp_path / "experiment"
    model = _SimpleModel()
    optimizers = [optim.SGD(model.parameters(), lr=0.01)]
    return GenericTrainer(config=config, model=model, optimizers=optimizers)


def _set_training_step(trainer: GenericTrainer):
    def training_step(
        tr: GenericTrainer, batch, batch_idx, dataloader_idx, dataloader_name
    ):
        x, y = batch
        out = tr.model(x)
        loss = nn.functional.cross_entropy(out, y)
        return {"loss": loss}

    trainer.set_training_step(training_step)


def test_no_resume_starts_from_scratch(tmp_path):
    trainer = _make_trainer(tmp_path)
    _set_training_step(trainer)
    loaders = _create_loaders(samples_per_loader=4, batch_size=2, num_loaders=2)

    # 2 loaders * (4 samples / 2 batch) = 4 steps per epoch
    trainer.fit(loaders, max_epochs=1)
    assert trainer.current_epoch == 1
    assert trainer.global_step == 4


def test_auto_resume_from_latest(tmp_path):
    # First run: produce a checkpoint at end of epoch 1
    trainer1 = _make_trainer(tmp_path)
    _set_training_step(trainer1)
    loaders = _create_loaders(samples_per_loader=4, batch_size=2, num_loaders=2)
    trainer1.fit(loaders, max_epochs=1)

    latest = trainer1.checkpoint_manager.get_latest_checkpoint()
    assert latest is not None

    # Second run: same root_dir; auto-resume from latest and continue to epoch 2
    trainer2 = _make_trainer(tmp_path)
    _set_training_step(trainer2)
    trainer2.fit(loaders, max_epochs=2)

    # After resume, current_epoch should advance to 2;
    # resume metrics indicate full resume path executed
    assert trainer2.current_epoch == 2
    assert trainer2.resume_time_sec is not None
    assert (trainer2.resume_checkpoint_size_mb or 0) > 0


def test_explicit_resume_from_user_checkpoint(tmp_path):
    # First run: produce a checkpoint at end of epoch 1
    trainer1 = _make_trainer(tmp_path)
    _set_training_step(trainer1)
    loaders = _create_loaders(samples_per_loader=4, batch_size=2, num_loaders=2)
    trainer1.fit(loaders, max_epochs=1)
    latest = trainer1.checkpoint_manager.get_latest_checkpoint()
    assert latest is not None

    # Disable auto-resume and explicitly provide checkpoint path
    trainer2 = _make_trainer(tmp_path)
    trainer2.config.preemption.resume_from_latest_symlink = False
    _set_training_step(trainer2)

    trainer2.fit(loaders, max_epochs=2, resume_from_checkpoint=str(latest))

    assert trainer2.current_epoch == 2
    assert trainer2.resume_time_sec is not None
    assert (trainer2.resume_checkpoint_size_mb or 0) > 0


def test_skip_auto_resume_when_preloaded(tmp_path):
    # Seed manager latest by running one full epoch
    trainer1 = _make_trainer(tmp_path)
    _set_training_step(trainer1)
    loaders = _create_loaders(samples_per_loader=4, batch_size=2, num_loaders=2)
    trainer1.fit(loaders, max_epochs=1)
    latest = trainer1.checkpoint_manager.get_latest_checkpoint()
    assert latest is not None

    # New trainer: manually preload the checkpoint before fit
    trainer2 = _make_trainer(tmp_path)
    _set_training_step(trainer2)

    from model_training_framework.trainer.checkpoints import (
        load_checkpoint as free_load,
    )

    free_load(path=latest, trainer=trainer2)

    # With preloaded logic, auto-resume should be skipped and resume_time_sec stays None
    trainer2.fit(loaders, max_epochs=2)

    assert trainer2.current_epoch == 2
    assert trainer2.resume_time_sec is None
