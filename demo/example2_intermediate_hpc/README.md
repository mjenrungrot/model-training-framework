# Example 2 — Config + SLURM + Trainer

Minimal, practical demo that connects all three parts of the framework:

- Config: Define experiments and generate parameter sweeps
- SLURM: Render sbatch scripts and (dry-run) submit jobs
- Trainer: Run multi‑dataloader training with clear per‑batch logging

The demo is intentionally small and easy to read. It uses only three files and
saves everything under `experiments/<experiment_name>/` for reproducibility.

## Files

- `config.py`: base ExperimentConfig and parameter grid search
- `train_script.py`: model + trainer logic for a single experiment
- `orchestrate.py`: local/SLURM orchestrator (builds configs and runs/creates jobs)

## Quick Start

- Preview the composed configs (after grid overrides):
  - `python demo/example2_intermediate_hpc/config.py`

- Run locally (sequentially):
  - `python demo/example2_intermediate_hpc/orchestrate.py local`

- Generate SLURM job scripts (dry‑run):
  - `python demo/example2_intermediate_hpc/orchestrate.py slurm`

The orchestrator auto‑creates a default sbatch template (`_slurm_template.sbatch`)
on first use, and writes each experiment config to `experiments/<name>/config.yaml`.

## How It Works

- `config.py`
  - `build_base_config()`: small, self‑contained config dict
  - `build_parameter_grid_search()`: tiny grid over LR and batch size
  - Running this file prints all composed `ExperimentConfig`s as JSON

- `orchestrate.py`
  - Builds configs, saves them, then either:
    - Runs locally (calls `run_training_from_experiment()` for each config), or
    - Renders sbatch scripts via `SLURMLauncher` (dry‑run) for cluster usage

- `train_script.py`
  - Implements a small Transformer, dataloaders, and trainer step functions
  - Maps `ExperimentConfig` → `GenericTrainerConfig` (multi‑dataloader aware)
  - CLI worker: `python train_script.py <EXPERIMENT_NAME>` loads the saved config

## Outputs

- `experiments/<name>/config.yaml`: exact config used
- `experiments/<name>/checkpoints/`: checkpoints (periodic saving disabled by default)
- `experiments/<name>/<name>.sbatch`: rendered job script (SLURM dry‑run)

## Multi‑Dataloader Behavior

The trainer runs with the multi‑dataloader API and prints the dataloader name per
batch (both train and validation). To change the number of loaders or sampling
strategy, edit `custom_params.multi_loader` in `config.py`.

## SLURM Notes

- This demo’s SLURM mode is dry‑run by default to keep it safe. To actually submit,
  set `dry_run=False` where the launcher is called in `orchestrate.py`.
- The default template uses a single task. For multi‑GPU DDP, customize the template
  to use `srun` with multiple tasks per node and add any cluster‑specific modules.

## Why This Structure?

- Separation of concerns keeps each file focused and easy to maintain:
  - `config.py` — what to run
  - `orchestrate.py` — where/how to run
  - `train_script.py` — the actual training logic
