"""
Example 3 training script: single-experiment run with auto-resume.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import TypedDict

from data import create_loaders
from model import SmallMLP
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

PROJECT_ROOT = Path(__file__).resolve().parent


def _setup_basic_logging() -> None:
    """Ensure INFO logs are emitted to stdout in Slurm runs.

    Local runs configure logging in orchestrate.py. For Slurm, configure
    a simple root logger here so both framework and demo logs appear.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Prevent duplicate handlers on re-entry
    for h in list(root.handlers):
        root.removeHandler(h)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)


LOGGER = logging.getLogger("demo3")


def _build_trainer_config(exp: ExperimentConfig) -> GenericTrainerConfig:
    multi = exp.custom_params.get("multi_loader", {}) if exp.custom_params else {}
    nload = int(multi.get("num_loaders", 1))
    strategy = str(multi.get("sampling_strategy", "sequential")).lower()
    sampling = (
        SamplingStrategy.SEQUENTIAL
        if strategy == "sequential"
        else SamplingStrategy.ROUND_ROBIN
    )
    names = [f"loader_{i + 1}" for i in range(nload)]
    val_names = [f"val_{n}" for n in names]

    ckpt_root = PROJECT_ROOT / "experiments" / exp.experiment_name / "checkpoints"

    return GenericTrainerConfig(
        train_loader_config=MultiDataLoaderConfig(
            sampling_strategy=(
                SamplingStrategy.WEIGHTED if strategy == "weighted" else sampling
            ),
            dataloader_weights=(
                multi.get("dataloader_weights") if strategy == "weighted" else None
            ),
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
            frequency=ValidationFrequency.EVERY_N_STEPS,
            every_n_steps=50,
            aggregation=ValAggregation.MACRO_AVG_EQUAL_LOADERS,
        ),
    )


# Environment-tunable pacing and preemption for the demo
SLEEP_PER_BATCH_SEC = float(os.environ.get("EX3_SLEEP_SEC", "0.5"))
# Preemption control: allow demo to preempt once, then complete on requeue
_disable = os.environ.get("EX3_DISABLE_PREEMPT", "0") == "1"
_timeout = float(os.environ.get("EX3_PREEMPT_SEC", "30.0"))
_restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
if _disable:
    PREEMPT_TIMEOUT_SEC = None
elif _restart_count >= 1:
    # After the first requeue, finish without further simulated preemption
    PREEMPT_TIMEOUT_SEC = None
else:
    PREEMPT_TIMEOUT_SEC = _timeout


class _TrainState(TypedDict):
    last_time: float
    durations: list[float]
    counts: dict[str, int]
    counter: int


def make_train_step(start_time: float, preempt_timeout: float | None):
    # Use a mutable holder to track last observed global_step and recent timings
    last_global_step_holder = [-1]
    state: _TrainState = {
        "last_time": start_time,
        "durations": [],  # rolling batch times
        "counts": {},  # per-loader batch counts
        "counter": 0,  # micro-step counter
    }

    def _train_step(
        trainer: GenericTrainer,
        batch,
        batch_idx: int,
        dataloader_idx: int,
        dataloader_name: str,
    ):
        # Log optimizer step events (global_step increments only on optimizer step)
        last_gs = last_global_step_holder[0]
        if last_gs >= 0 and trainer.global_step > last_gs:
            LOGGER.info(
                "[OPTIM] optimizer step committed: global_step=%s", trainer.global_step
            )
        last_global_step_holder[0] = trainer.global_step

        x, y = batch
        logits = trainer.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()

        # Demo timing: show current elapsed and remaining time to preemption
        now = time.time()
        elapsed = int(now - start_time)
        if preempt_timeout is not None:
            remaining = max(0, int(preempt_timeout - (now - start_time)))
            timing = f"t={elapsed}s, preempt in ~{remaining}s"
        else:
            timing = f"t={elapsed}s"

        # Track per-batch duration and rolling throughput
        dt = now - state["last_time"]
        state["last_time"] = now
        if dt > 0:
            state["durations"].append(dt)
            if len(state["durations"]) > 10:
                state["durations"].pop(0)
        avg_ms = (
            (sum(state["durations"]) / len(state["durations"]) * 1000.0)
            if state["durations"]
            else 0.0
        )
        try:
            bs = int(x.shape[0])
            sps = (bs / (avg_ms / 1000.0)) if avg_ms > 0 else 0.0
        except Exception:
            sps = 0.0

        # Accumulation index [i/Y]
        gas = max(1, trainer.config.performance.gradient_accumulation_steps)
        state["counter"] += 1
        micro = ((state["counter"] - 1) % gas) + 1

        # Per-loader batch count
        state["counts"][dataloader_name] = state["counts"].get(dataloader_name, 0) + 1

        # ETA to next checkpoint (optimizer-step based)
        try:
            sev = trainer.config.checkpoint.save_every_n_steps
            if sev:
                next_ckpt = ((trainer.global_step // sev) + 1) * sev
                steps_to_ckpt = max(0, next_ckpt - trainer.global_step)
                ckpt_hint = f", next_ckpt≈{steps_to_ckpt} steps"
            else:
                ckpt_hint = ""
        except Exception:
            ckpt_hint = ""

        # Small sleep to pace the demo without extra complexity
        if SLEEP_PER_BATCH_SEC > 0:
            time.sleep(SLEEP_PER_BATCH_SEC)

        LOGGER.info(
            "[TRAIN][demo3] loader=%s#%d batch=%d [acc %d/%d] | %s | %.0fms/batch, %.0f samp/s%s | freq checkpoints on; safe to Ctrl+C - resume will continue",
            dataloader_name,
            state["counts"][dataloader_name],
            batch_idx,
            micro,
            gas,
            timing,
            avg_ms,
            sps,
            ckpt_hint,
        )
        return {"loss": loss, f"{dataloader_name}/acc": acc}

    return _train_step


def _log_run_banner(
    *,
    exp: ExperimentConfig,
    trainer_cfg: GenericTrainerConfig,
    num_loaders: int,
    loader_names: list[str],
    world_size: int,
    rank: int,
    resumed: bool,
    resume_ckpt: str | None,
    current_epoch: int,
    global_step: int,
) -> None:
    line = "=" * 72
    sub = "-" * 72
    LOGGER.info(line)
    LOGGER.info("Run Summary")
    LOGGER.info(sub)
    LOGGER.info("Experiment: %s", exp.experiment_name)
    LOGGER.info(
        "Resume: %s%s",
        "yes" if resumed else "no",
        f" (ckpt={resume_ckpt})" if resume_ckpt else "",
    )
    LOGGER.info("World: size=%s rank=%s", world_size, rank)
    LOGGER.info(sub)
    LOGGER.info(
        "Model: SmallMLP(input_size=64, hidden_size=%s, num_layers=%s)",
        exp.model.hidden_size,
        exp.model.num_layers,
    )
    LOGGER.info(
        "Data: batch_size=%s num_workers=%s loaders=%s",
        exp.data.batch_size,
        exp.data.num_workers,
        num_loaders,
    )
    LOGGER.info("Loaders: %s", ", ".join(loader_names))
    LOGGER.info(
        "Training: epochs=%s grad_accum=%s",
        exp.training.max_epochs,
        exp.training.gradient_accumulation_steps,
    )
    LOGGER.info(
        "Optimizer: %s lr=%s weight_decay=%s",
        exp.optimizer.type,
        exp.optimizer.lr,
        exp.optimizer.weight_decay,
    )
    LOGGER.info(
        "Checkpoint: dir=%s save_every_n_steps=%s save_every_n_epochs=%s",
        trainer_cfg.checkpoint.root_dir,
        trainer_cfg.checkpoint.save_every_n_steps,
        trainer_cfg.checkpoint.save_every_n_epochs,
    )
    LOGGER.info(
        "Multi-Loader: strategy=%s",
        trainer_cfg.train_loader_config.sampling_strategy.value,
    )
    if trainer_cfg.validation is not None:
        try:
            mode = trainer_cfg.validation.frequency.value
        except Exception:
            mode = str(trainer_cfg.validation.frequency)
        LOGGER.info(
            "Validation: mode=%s, every_n_steps=%s, aggregator=%s",
            mode,
            trainer_cfg.validation.every_n_steps,
            trainer_cfg.validation.aggregation.value,
        )
    LOGGER.info(
        "Current State: epoch=%s step=%s",
        current_epoch,
        global_step,
    )
    LOGGER.info(line)


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
    LOGGER.info("[VALID] %s batch=%s", dataloader_name, batch_idx)
    return {"val_loss": loss, f"val_{dataloader_name}/acc": acc}


def run_training_from_config(identifier: str) -> None:
    # Configure logging for Slurm execution so INFO logs show in .out
    _setup_basic_logging()
    cfg_path = PROJECT_ROOT / "experiments" / identifier / "config.json"
    cm = ConfigurationManager(project_root=PROJECT_ROOT)
    exp_dict = cm.load_config(cfg_path, validate=False)
    # Remove optional sections explicitly set to null in JSON
    for opt in ("scheduler", "slurm"):
        if opt in exp_dict and exp_dict[opt] is None:
            del exp_dict[opt]
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
    cm.save_config(exp, exp_dir / "config.json", format="json")

    # Trainer
    trainer = GenericTrainer(config=trainer_cfg, model=model, optimizers=[optimizer])
    trainer.set_validation_step(_val_step)

    # Resume if latest checkpoint exists (restore states directly)
    latest = (
        trainer.checkpoint_manager.latest_path
        if hasattr(trainer, "checkpoint_manager")
        else None
    )
    resumed = False
    resume_name: str | None = None
    if latest and latest.exists():
        try:
            ep, gs, rs = trainer.checkpoint_manager.restore_from_checkpoint(
                model=trainer.model,
                optimizers=trainer.optimizers,
                schedulers=None,
                checkpoint_path=latest,
            )
            trainer.current_epoch = ep
            trainer.global_step = gs
            LOGGER.info("Resumed from %s (epoch=%s, step=%s)", latest.name, ep, gs)
            resumed = True
            resume_name = latest.name
        except Exception:
            LOGGER.exception("Failed to restore from %s; starting fresh", latest)
            latest = None

    # Optional preemption timeout (sec) via EXAMPLE3_TIMEOUT_SEC
    # Simple, always-on pre-emption timer for the demo
    timer: threading.Timer | None = None
    preempt_timeout: float | None = PREEMPT_TIMEOUT_SEC

    def _preempt() -> None:
        LOGGER.warning(
            "[demo3] Simulating preemption after %ss via SIGUSR1", PREEMPT_TIMEOUT_SEC
        )
        os.kill(os.getpid(), signal.SIGUSR1)

    if preempt_timeout is not None:
        timer = threading.Timer(preempt_timeout, _preempt)
        timer.daemon = True
        timer.start()

    # Log a nice run banner with expectations

    loader_sizes = [len(getattr(ld, "dataset", [])) for ld in train_loaders]
    batches_per_epoch = sum(
        math.ceil(n / max(1, exp.data.batch_size)) for n in loader_sizes
    )
    updates_per_epoch = math.ceil(
        batches_per_epoch / max(1, exp.training.gradient_accumulation_steps)
    )
    _log_run_banner(
        exp=exp,
        trainer_cfg=trainer_cfg,
        num_loaders=num_loaders,
        loader_names=[f"loader_{i + 1}" for i in range(num_loaders)],
        world_size=world_size,
        rank=rank,
        resumed=resumed,
        resume_ckpt=resume_name,
        current_epoch=trainer.current_epoch,
        global_step=trainer.global_step,
    )
    LOGGER.info(
        "Expected per-epoch: batches=%s, optimizer_updates≈%s",
        batches_per_epoch,
        updates_per_epoch,
    )
    try:
        start_time = time.time()
        trainer.set_training_step(make_train_step(start_time, preempt_timeout))
        trainer.fit(
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            max_epochs=exp.training.max_epochs,
        )
    finally:
        if timer is not None:
            timer.cancel()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_script.py <EXPERIMENT_NAME>")
        sys.exit(1)
    run_training_from_config(sys.argv[1])
