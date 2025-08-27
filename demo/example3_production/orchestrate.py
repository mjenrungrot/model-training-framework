"""
Example 3 orchestrator: build configs, run locally or submit to SLURM.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

# Local imports when run as a script
from config import build_base_config, build_parameter_grid_search
from train_script import run_training_from_config

from model_training_framework.config import ConfigurationManager
from model_training_framework.slurm import SLURMLauncher

PROJECT_ROOT = Path(__file__).resolve().parent


def _write_min_template(path: Path) -> None:
    content = """#!/bin/bash

#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodes={{NODES}}
#SBATCH --ntasks-per-node={{NTASKS_PER_NODE}}
#SBATCH --gpus-per-node={{GPUS_PER_NODE}}
#SBATCH --cpus-per-task={{CPUS_PER_TASK}}
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}
# Send SIGUSR1 N seconds before termination (for graceful checkpointing)
# Adjust N as needed (e.g., 60 for 1 minute). For demo we use 30s.
#SBATCH --signal=USR1@30
# Ensure Slurm appends to existing output on requeue
#SBATCH --open-mode=append
#SBATCH --constraint={{CONSTRAINT}}
#SBATCH --output={{OUTPUT_FILE}}
#SBATCH --error={{ERROR_FILE}}
#SBATCH --requeue

echo "Job ID: $${SLURM_JOB_ID}"
echo "Job Name: {{JOB_NAME}}"

mkdir -p experiments/{{EXPERIMENT_NAME}}
export PYTHONPATH="$${PYTHONPATH}:$$(pwd)"

# Allow caller to tune demo pacing and preemption via env vars
export EX3_SLEEP_SEC="$${EX3_SLEEP_SEC:-0.5}"
export EX3_PREEMPT_SEC="$${EX3_PREEMPT_SEC:-30}"
export EX3_DISABLE_PREEMPT="$${EX3_DISABLE_PREEMPT:-0}"

# Use the project-local environment's Python
./venv/bin/python {{SCRIPT_PATH}} {{CONFIG_NAME}}
"""
    path.write_text(content)


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("demo3")
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates across runs
    for h in list(logger.handlers):
        logger.removeHandler(h)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # Also configure root logger so framework logs (e.g., checkpoint saves) are captured
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Clear root handlers to prevent duplicates
    for h in list(root.handlers):
        root.removeHandler(h)
    root_fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    root_fh.setFormatter(formatter)
    root_ch = logging.StreamHandler(sys.stdout)
    root_ch.setFormatter(formatter)
    root.addHandler(root_fh)
    root.addHandler(root_ch)
    # Ensure demo logger doesn't double-propagate
    logger.propagate = False
    return logger


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"local", "slurm"}:
        print("Usage: python orchestrate.py <local|slurm> [submit]")
        return
    mode = sys.argv[1]
    submit = len(sys.argv) > 2 and sys.argv[2] == "submit"

    experiments_dir = PROJECT_ROOT / "experiments"
    script_path = PROJECT_ROOT / "train_script.py"

    # Build configs
    base = build_base_config()
    grid_search = build_parameter_grid_search(base)
    experiments = list(grid_search.generate_experiments())
    cm = ConfigurationManager(project_root=PROJECT_ROOT)
    print(f"Generated {len(experiments)} experiment configs")

    for exp in experiments:
        exp_dir = experiments_dir / exp.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        cm.save_config(exp, exp_dir / "config.json", format="json")

    if mode == "local":
        for i, exp in enumerate(experiments, 1):
            log_path = experiments_dir / exp.experiment_name / "local_run.log"
            logger = _setup_logger(log_path)
            logger.info("[%d/%d] Starting %s", i, len(experiments), exp.experiment_name)
            run_training_from_config(exp.experiment_name)
            logger.info(
                "[%d/%d] Finished %s\n", i, len(experiments), exp.experiment_name
            )
        logging.getLogger("demo3").info("All local runs completed.")
        return

    # SLURM
    template_path = PROJECT_ROOT / "_slurm_template.sbatch"
    _write_min_template(template_path)

    repo_root = _find_repo_root(PROJECT_ROOT)
    launcher = SLURMLauncher(
        template_path=template_path,
        project_root=repo_root,
        experiments_dir=experiments_dir,
    )

    result = launcher.submit_experiment_batch(
        experiments=experiments,
        script_path=script_path,
        use_git_branch=False,
        dry_run=not submit,
    )

    print(
        f"SLURM {'submission' if submit else 'dry-run'}: {result.success_count}/{result.total_experiments} handled"
    )
    for res in result.job_results:
        print(
            f" - {res.experiment_name}: {'OK' if res.success else 'FAIL'} | script={res.sbatch_script_path}"
        )


if __name__ == "__main__":
    main()
