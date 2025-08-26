"""
Example 3 orchestrator: build configs, run locally or submit to SLURM.
"""

from __future__ import annotations

from pathlib import Path
import sys

from model_training_framework.config import ConfigurationManager
from model_training_framework.slurm import SLURMLauncher

from .config import build_base_config, build_parameter_grid_search
from .train_script import run_training_from_config

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
#SBATCH --output={{OUTPUT_FILE}}
#SBATCH --error={{ERROR_FILE}}
#SBATCH --requeue={{REQUEUE}}

echo "Job ID: $${SLURM_JOB_ID}"
echo "Job Name: {{JOB_NAME}}"

mkdir -p experiments/{{EXPERIMENT_NAME}}
export PYTHONPATH="$${PYTHONPATH}:$$(pwd)"

{{PYTHON_EXECUTABLE}} {{SCRIPT_PATH}} {{CONFIG_NAME}}
"""
    path.write_text(content)


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start


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
    grid = build_parameter_grid_search(base)
    experiments = list(grid.generate_experiments())
    print(f"Generated {len(experiments)} experiment configs")

    cm = ConfigurationManager(project_root=PROJECT_ROOT)
    for exp in experiments:
        exp_dir = experiments_dir / exp.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        cm.save_config(exp, exp_dir / "config.yaml", format="yaml")

    if mode == "local":
        for i, exp in enumerate(experiments, 1):
            print(f"[{i}/{len(experiments)}] Starting {exp.experiment_name}")
            run_training_from_config(exp.experiment_name)
            print(f"[{i}/{len(experiments)}] Finished {exp.experiment_name}\n")
        print("All local runs completed.")
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
