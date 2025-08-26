# Example 3 — Production SLURM + Auto-Resume (Minimal)

Small, production-minded demo that launches training locally or on a SLURM cluster,
supports multiple configurations, and automatically resumes from the last checkpoint
after interruptions (Ctrl+C, SIGTERM, SIGUSR1).

## Files

- `config.py`: base ExperimentConfig dict and a tiny ParameterGridSearch
- `data.py`: small synthetic datasets and DataLoaders
- `model.py`: compact MLP model
- `train_script.py`: single-experiment trainer (auto-resume from latest.ckpt)
- `orchestrate.py`: builds configs and runs them locally or submits SLURM jobs
- `_slurm_template.sbatch`: minimal, safe SBATCH template (auto-created)

## Quick Start

- Preview configs:
  - `python demo/example3_production/config.py`
- Local run (sequential):
  - `python demo/example3_production/orchestrate.py local`
- SLURM: dry-run submit scripts:
  - `python demo/example3_production/orchestrate.py slurm`
- SLURM: actually submit (if you have SLURM):
  - `python demo/example3_production/orchestrate.py slurm submit`

## Auto-Resume Behavior

- Checkpoints are saved frequently (every N steps) to `experiments/<name>/checkpoints/`.
- On start, the trainer automatically resumes from `latest.ckpt` if present.
- Ctrl+C, SIGTERM, or SIGUSR1 causes a checkpoint to be written before exit; the next run continues seamlessly.

### Simulate Pre-emption Locally

- Use an environment variable to set a timeout in seconds. The training script will send SIGUSR1 to itself after the timeout, forcing a checkpoint and clean exit.
- Example:
  - `EXAMPLE3_TIMEOUT_SEC=5 python demo/example3_production/orchestrate.py local`
  - Re-run the same command again; it will resume from the latest checkpoint and continue.

### Simulate Pre-emption on SLURM

- You can also embed the same env var in your submission environment. For actual SLURM submission, use:
  - `EXAMPLE3_TIMEOUT_SEC=60 python demo/example3_production/orchestrate.py slurm submit`
  - Jobs will receive SIGUSR1 after 60s, save a checkpoint, and exit; re-submitting resumes.

## Notes

- The demo anchors all paths under `demo/example3_production` so local and SLURM runs share the same layout.
- The SLURM template escapes shell variables so scripts render correctly on non-SLURM systems.
