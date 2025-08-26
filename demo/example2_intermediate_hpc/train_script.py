"""
Training script: model + trainer logic for a single experiment.

Acts as both:
- Library: exposes run_training_from_experiment(exp) for local orchestration
- CLI worker: python train_script.py <EXPERIMENT_NAME> loads experiments/<name>/config.yaml and runs training
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import cast

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from model_training_framework.config import ConfigurationManager, ExperimentConfig
from model_training_framework.trainer import (
    CheckpointConfig,
    DDPConfig,
    FaultToleranceConfig,
    GenericTrainer,
    GenericTrainerConfig,
    LoggingConfig,
    MultiDataLoaderConfig,
    PerformanceConfig,
    ValidationConfig,
)
from model_training_framework.trainer.config import (
    EpochLengthPolicy,
    SamplingStrategy,
    ValAggregation,
    ValidationFrequency,
)

# Treat this demo folder as the project root
PROJECT_ROOT = Path(__file__).resolve().parent


class DistributedTransformer(nn.Module):
    """Sane-sized transformer model for demo and distributed training."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        return cast(torch.Tensor, self.output(x))


def distributed_training_step(
    trainer: GenericTrainer,
    batch: tuple[torch.Tensor, torch.Tensor],
    batch_idx: int,
    dataloader_idx: int,
    dataloader_name: str,
) -> dict[str, float | torch.Tensor]:
    """Training step for multi-loader training."""
    inputs, targets = batch
    print(
        f"[TRAIN] dataloader={dataloader_name} (idx={dataloader_idx}) batch_idx={batch_idx}"
    )

    outputs = trainer.model(inputs)
    loss = F.cross_entropy(
        outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-100
    )

    with torch.no_grad():
        _, predicted = outputs.max(-1)
        mask = targets != -100
        accuracy = ((predicted == targets) & mask).float().sum() / mask.sum()

    return {
        "loss": loss,
        f"{dataloader_name}/loss": loss.item(),
        f"{dataloader_name}/accuracy": accuracy.item(),
        f"{dataloader_name}/perplexity": torch.exp(loss).item(),
    }


def distributed_validation_step(
    trainer: GenericTrainer,
    batch: tuple[torch.Tensor, torch.Tensor],
    batch_idx: int,
    dataloader_idx: int,
    dataloader_name: str,
) -> dict[str, float | torch.Tensor]:
    """Validation step for multi-loader training."""
    inputs, targets = batch
    with torch.no_grad():
        outputs = trainer.model(inputs)
        loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-100
        )
        _, predicted = outputs.max(-1)
        mask = targets != -100
        accuracy = ((predicted == targets) & mask).float().sum() / mask.sum()

    print(
        f"[VALID] dataloader={dataloader_name} (idx={dataloader_idx}) batch_idx={batch_idx}"
    )

    return {
        "val_loss": loss,
        f"val_{dataloader_name}/loss": loss.item(),
        f"val_{dataloader_name}/accuracy": accuracy.item(),
    }


def create_distributed_dataloaders(
    batch_size: int, world_size: int, rank: int, num_loaders: int = 1
) -> tuple[list[DataLoader], list[DataLoader]]:
    """Create synthetic distributed dataloaders for demo purposes."""
    train_loaders: list[DataLoader] = []
    val_loaders: list[DataLoader] = []

    # Keep the demo fast: small datasets and shorter sequence length
    seq_len = 64
    for i in range(num_loaders):
        torch.manual_seed(42 + i)
        train_size = 64 * (i + 1)
        val_size = 32 * (i + 1)

        train_data = TensorDataset(
            torch.randint(0, 32000, (train_size, seq_len)),
            torch.randint(0, 32000, (train_size, seq_len)),
        )
        val_data = TensorDataset(
            torch.randint(0, 32000, (val_size, seq_len)),
            torch.randint(0, 32000, (val_size, seq_len)),
        )

        train_sampler: DistributedSampler = DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=42
        )
        val_sampler: DistributedSampler = DistributedSampler(
            val_data, num_replicas=world_size, rank=rank, shuffle=False
        )

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size * 2,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=False,
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders


def build_trainer_config_from_experiment(exp: ExperimentConfig) -> GenericTrainerConfig:
    """Map ExperimentConfig (from config components) to trainer config."""
    multi = exp.custom_params.get("multi_loader", {}) if exp.custom_params else {}
    num_loaders = int(multi.get("num_loaders", 1))
    strategy_str = str(multi.get("sampling_strategy", "sequential")).lower()
    sampling_strategy = (
        SamplingStrategy.SEQUENTIAL
        if strategy_str == "sequential"
        else SamplingStrategy.ROUND_ROBIN
    )
    names = [f"loader_{i+1}" for i in range(num_loaders)]
    val_names = [f"val_{n}" for n in names]

    ckpt_root = PROJECT_ROOT / "experiments" / exp.experiment_name / "checkpoints"

    return GenericTrainerConfig(
        train_loader_config=MultiDataLoaderConfig(
            sampling_strategy=sampling_strategy,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            dataloader_names=names,
            cycle_short_loaders=True,
        ),
        val_loader_config=MultiDataLoaderConfig(
            sampling_strategy=sampling_strategy,
            dataloader_names=val_names,
        ),
        ddp=DDPConfig(
            backend="nccl",
            sync_schedules_across_ranks=True,
            validate_schedule_consistency=True,
        ),
        performance=PerformanceConfig(
            gradient_accumulation_steps=exp.training.gradient_accumulation_steps,
            use_amp=True,
            clip_grad_norm=1.0,
        ),
        checkpoint=CheckpointConfig(
            root_dir=str(ckpt_root),
            save_every_n_epochs=None,  # Disable periodic checkpointing
            max_checkpoints=2,
            save_optimizer=True,
            save_rng=True,
        ),
        fault_tolerance=FaultToleranceConfig(
            save_sampler_state=True,
            save_dataset_state=False,
        ),
        logging=LoggingConfig(
            logger_type="console",
            all_reduce_metrics=True,
            log_per_loader_metrics=True,
        ),
        validation=ValidationConfig(
            frequency=ValidationFrequency.PER_EPOCH,
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
        ),
    )


def run_training_from_experiment(exp: ExperimentConfig) -> None:
    """Train a single ExperimentConfig locally using GenericTrainer."""
    trainer_config = build_trainer_config_from_experiment(exp)

    if exp.seed is not None:
        torch.manual_seed(exp.seed)

    model = DistributedTransformer(
        vocab_size=32000,
        hidden_size=exp.model.hidden_size,
        num_layers=exp.model.num_layers,
        num_heads=max(4, exp.model.hidden_size // 64),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=exp.optimizer.lr, weight_decay=exp.optimizer.weight_decay
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    multi = exp.custom_params.get("multi_loader", {}) if exp.custom_params else {}
    num_loaders = int(multi.get("num_loaders", 1))
    train_loaders, val_loaders = create_distributed_dataloaders(
        batch_size=exp.data.batch_size,
        world_size=world_size,
        rank=rank,
        num_loaders=num_loaders,
    )

    exp_dir = PROJECT_ROOT / "experiments" / exp.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ConfigurationManager(project_root=Path.cwd()).save_config(
        exp, exp_dir / "config.yaml", format="yaml"
    )

    trainer = GenericTrainer(config=trainer_config, model=model, optimizers=[optimizer])
    trainer.set_training_step(distributed_training_step)
    trainer.set_validation_step(distributed_validation_step)

    print(
        f"\n▶️ Local run: {exp.experiment_name} | lr={exp.optimizer.lr}, batch_size={exp.data.batch_size}, grad_accum={exp.training.gradient_accumulation_steps}, loaders={num_loaders}"
    )
    trainer.fit(
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        max_epochs=exp.training.max_epochs,
    )


def run_training_from_config_identifier(identifier: str) -> None:
    """Worker entrypoint: load config by experiment name and run training."""
    config_path = PROJECT_ROOT / "experiments" / identifier / "config.yaml"
    config_manager = ConfigurationManager(project_root=PROJECT_ROOT)
    exp_dict = config_manager.load_config(config_path, validate=False)
    exp = config_manager.create_experiment_config(exp_dict)
    run_training_from_experiment(exp)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_script.py <EXPERIMENT_NAME>")
        sys.exit(1)
    run_training_from_config_identifier(sys.argv[1])
