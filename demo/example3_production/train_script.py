"""
Example 3 training script: single-experiment run with auto-resume.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import threading

import torch

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
    PreemptionConfig,
    ValidationConfig,
)
from model_training_framework.trainer.config import (
    EpochLengthPolicy,
    SamplingStrategy,
    ValAggregation,
    ValidationFrequency,
)

from .data import create_loaders
from .model import SmallMLP

PROJECT_ROOT = Path(__file__).resolve().parent


def _build_trainer_config(exp: ExperimentConfig) -> GenericTrainerConfig:
    multi = exp.custom_params.get("multi_loader", {}) if exp.custom_params else {}
    nload = int(multi.get("num_loaders", 1))
    strategy = str(multi.get("sampling_strategy", "sequential")).lower()
    sampling = (
        SamplingStrategy.SEQUENTIAL
        if strategy == "sequential"
        else SamplingStrategy.ROUND_ROBIN
    )
    names = [f"loader_{i+1}" for i in range(nload)]
    val_names = [f"val_{n}" for n in names]

    ckpt_root = PROJECT_ROOT / "experiments" / exp.experiment_name / "checkpoints"

    return GenericTrainerConfig(
        train_loader_config=MultiDataLoaderConfig(
            sampling_strategy=sampling,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            dataloader_names=names,
            cycle_short_loaders=True,
        ),
        val_loader_config=MultiDataLoaderConfig(
            sampling_strategy=sampling,
            dataloader_names=val_names,
        ),
        ddp=DDPConfig(
            backend="nccl",
            sync_schedules_across_ranks=True,
            validate_schedule_consistency=True,
        ),
        performance=PerformanceConfig(
            gradient_accumulation_steps=exp.training.gradient_accumulation_steps,
            use_amp=False,
        ),
        checkpoint=CheckpointConfig(
            root_dir=str(ckpt_root),
            save_every_n_steps=10,
            save_every_n_epochs=None,
            max_checkpoints=3,
            save_optimizer=True,
            save_rng=True,
        ),
        fault_tolerance=FaultToleranceConfig(
            save_sampler_state=True, save_dataset_state=False
        ),
        preemption=PreemptionConfig(resume_from_latest_symlink=True),
        logging=LoggingConfig(
            logger_type="console", all_reduce_metrics=True, log_per_loader_metrics=True
        ),
        validation=ValidationConfig(
            frequency=ValidationFrequency.PER_EPOCH,
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
        ),
    )


def _train_step(
    trainer: GenericTrainer,
    batch,
    batch_idx: int,
    dataloader_idx: int,
    dataloader_name: str,
):
    x, y = batch
    logits = trainer.model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
    print(f"[TRAIN] {dataloader_name} batch={batch_idx}")
    return {"loss": loss, f"{dataloader_name}/acc": acc}


def _val_step(
    trainer: GenericTrainer,
    batch,
    batch_idx: int,
    dataloader_idx: int,
    dataloader_name: str,
):
    x, y = batch
    with torch.no_grad():
        logits = trainer.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
    print(f"[VALID] {dataloader_name} batch={batch_idx}")
    return {"val_loss": loss, f"val_{dataloader_name}/acc": acc}


def run_training_from_config(identifier: str) -> None:
    cfg_path = PROJECT_ROOT / "experiments" / identifier / "config.yaml"
    cm = ConfigurationManager(project_root=PROJECT_ROOT)
    exp_dict = cm.load_config(cfg_path, validate=False)
    exp = cm.create_experiment_config(exp_dict)

    # Build trainer config and artifacts
    trainer_cfg = _build_trainer_config(exp)
    model = SmallMLP(
        input_size=64,
        hidden_size=exp.model.hidden_size,
        num_layers=exp.model.num_layers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=exp.optimizer.lr, weight_decay=exp.optimizer.weight_decay
    )

    # Prepare loaders
    world_size = 1  # No DDP in this minimal demo; value is carried for signatures
    rank = 0
    multi = exp.custom_params.get("multi_loader", {}) if exp.custom_params else {}
    num_loaders = int(multi.get("num_loaders", 1))
    train_loaders, val_loaders = create_loaders(
        batch_size=exp.data.batch_size,
        world_size=world_size,
        rank=rank,
        num_loaders=num_loaders,
    )

    # Save config (idempotent)
    exp_dir = PROJECT_ROOT / "experiments" / exp.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    cm.save_config(exp, exp_dir / "config.yaml", format="yaml")

    # Trainer
    trainer = GenericTrainer(config=trainer_cfg, model=model, optimizers=[optimizer])
    trainer.set_training_step(_train_step)
    trainer.set_validation_step(_val_step)

    # Resume if latest checkpoint exists
    latest = (
        trainer.checkpoint_manager.latest_path
        if hasattr(trainer, "checkpoint_manager")
        else None
    )
    resume_path = str(latest) if latest and latest.exists() else None

    # Optional preemption timeout (sec) via EXAMPLE3_TIMEOUT_SEC
    timeout_env = os.environ.get("EXAMPLE3_TIMEOUT_SEC")
    timer: threading.Timer | None = None
    if timeout_env:
        try:
            timeout = float(timeout_env)
            if timeout > 0:

                def _preempt() -> None:
                    print(f"[demo3] Simulating preemption after {timeout}s via SIGUSR1")
                    os.kill(os.getpid(), signal.SIGUSR1)

                timer = threading.Timer(timeout, _preempt)
                timer.daemon = True
                timer.start()
        except ValueError:
            pass

    print(
        f"▶️  Run: {exp.experiment_name} lr={exp.optimizer.lr} bs={exp.data.batch_size} loaders={num_loaders}"
    )
    try:
        trainer.fit(
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            max_epochs=exp.training.max_epochs,
            resume_from_checkpoint=resume_path,
        )
    finally:
        if timer is not None:
            timer.cancel()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_script.py <EXPERIMENT_NAME>")
        sys.exit(1)
    run_training_from_config(sys.argv[1])
