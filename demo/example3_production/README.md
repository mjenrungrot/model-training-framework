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
- Uses 2 dataloaders and gradient accumulation by default for realism; adjust in `config.py` if desired.

### Simulate Pre-emption Locally

- No flags needed. The demo pre-empts automatically every ~30 seconds and saves a checkpoint.
- Just run:
  - `python demo/example3_production/orchestrate.py local`
- Re-run the same command; the run resumes from the latest checkpoint and continues.

### Simulate Pre-emption on SLURM

- For actual SLURM submission:
  - `python demo/example3_production/orchestrate.py slurm submit`
  - Jobs will pre-empt every ~30s, save a checkpoint, and exit; re-submitting resumes.

## Notes

- The demo anchors all paths under `demo/example3_production` so local and SLURM runs share the same layout.
- The SLURM template escapes shell variables so scripts render correctly on non-SLURM systems.
