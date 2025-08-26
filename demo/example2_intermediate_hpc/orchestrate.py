"""
Orchestrator: build configs and run locally or submit SLURM jobs.

Usage:
- python orchestrate.py local  # run all configs in sequence locally
- python orchestrate.py slurm  # render and (dry-run) submit SBATCH jobs
"""

from __future__ import annotations

from pathlib import Path
import sys

# Import local modules (script dir is on sys.path when run directly)
from config import build_base_config, build_parameter_grid_search
from train_script import run_training_from_experiment

from model_training_framework.config import ConfigurationManager
from model_training_framework.slurm import SLURMLauncher


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"local", "slurm"}:
        print("Usage: python orchestrate.py <local|slurm>")
        return

    mode = sys.argv[1]

    # Anchor project_root to this demo folder
    project_root = Path(__file__).resolve().parent
    experiments_dir = project_root / "experiments"
    script_path = project_root / "train_script.py"

    # Build base config and grid
    base_config = build_base_config()
    grid_search = build_parameter_grid_search(base_config)
    experiments = list(grid_search.generate_experiments())

    print(f"Generated {len(experiments)} experiment configs")

    # Save each experiment config
    config_manager = ConfigurationManager(project_root=project_root)
    for exp in experiments:
        exp_dir = experiments_dir / exp.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        config_manager.save_config(exp, exp_dir / "config.yaml", format="yaml")

    if mode == "local":
        print("Running experiments locally in sequence...\n")
        for i, exp in enumerate(experiments, 1):
            print(f"[{i}/{len(experiments)}] Starting {exp.experiment_name}")
            run_training_from_experiment(exp)
            print(f"[{i}/{len(experiments)}] Finished {exp.experiment_name}\n")
        print("All local runs completed.")
        return

    # SLURM path: generate default template if missing and dry-run submit
    template_path = project_root / "_slurm_template.sbatch"

    # Write a minimal, safe template (idempotent) with properly escaped shell vars
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

# Environment setup
module load python/3.9

# Job information
echo "Job ID: $${SLURM_JOB_ID}"
echo "Job Name: {{JOB_NAME}}"
echo "Node: $${SLURM_NODEID}"
echo "Start Time: $$(date)"
echo "Working Directory: $$(pwd)"

# Create experiment directory
mkdir -p experiments/{{EXPERIMENT_NAME}}

# Setup environment
export PYTHONPATH="$${PYTHONPATH}:$$(pwd)"
export CUDA_VISIBLE_DEVICES=$${SLURM_LOCALID}

# Run training script
echo "Starting training..."
{{PYTHON_EXECUTABLE}} {{SCRIPT_PATH}} {{CONFIG_NAME}}

echo "Job completed at $$(date)"
"""
        path.write_text(content)

    _write_min_template(template_path)

    # Use the repository root for SLURM launcher to satisfy GitManager
    def _find_repo_root(start: Path) -> Path:
        for p in [start, *start.parents]:
            if (p / ".git").exists():
                return p
        return start

    repo_root = _find_repo_root(project_root)

    launcher = SLURMLauncher(
        template_path=template_path,
        project_root=repo_root,
        experiments_dir=experiments_dir,
    )

    # Verify sbatch availability; dry-run always
    job_result = launcher.submit_experiment_batch(
        experiments=experiments,
        script_path=script_path,
        use_git_branch=False,
        dry_run=True,
    )

    print(
        f"SLURM dry-run: {job_result.success_count}/{job_result.total_experiments} sbatch scripts rendered"
    )
    for res in job_result.job_results:
        print(
            f" - {res.experiment_name}: {'OK' if res.success else 'FAIL'} | script={res.sbatch_script_path}"
        )


if __name__ == "__main__":
    main()
