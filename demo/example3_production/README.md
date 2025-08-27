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

## SLURM Requeue & Signals — Recommended Pattern

Long‑running SLURM jobs should handle preemption/time‑limit gracefully by:

- Using `#SBATCH --requeue` so SLURM may requeue when it preempts jobs (QoS/node events).
- Requesting an early warning signal before termination with `#SBATCH --signal=USR1@N`.
  - Example: `#SBATCH --signal=USR1@60` sends SIGUSR1 60s before job termination.
- Handling `SIGUSR1` in your code to quickly save a checkpoint and exit cleanly.
- Resuming from the latest checkpoint on the next run.

This example implements that pattern:

- `train_script.py` installs a SIGUSR1 handler and saves checkpoints upon preemption.
- The minimal SBATCH template now includes `#SBATCH --signal=USR1@30` for an early warning.
- The trainer optionally calls `scontrol requeue "$SLURM_JOB_ID"` after checkpointing so the job requeues immediately (demo‑friendly). You can disable that if you rely on SLURM to requeue.

### Logging and Output Files

- The template uses `#SBATCH --open-mode=append` so requeues append to the same
  `experiments/<exp>/<jobid>.out` and `.err` files rather than overwriting them.
- INFO logs are emitted to stdout from within `train_script.py`, so run banners,
  progress, and checkpoint messages appear in the `.out` file.

### Code Snippet (standard convention)

In Python, set a SIGUSR1 handler and checkpoint on receipt:

```python
import signal
import logging

preempt_requested = False

def _on_sigusr1(signum, frame):
    logging.warning("Received SIGUSR1; will checkpoint and exit")
    global preempt_requested
    preempt_requested = True

signal.signal(signal.SIGUSR1, _on_sigusr1)

# In your training loop
for batch in loader:
    if preempt_requested:
        save_checkpoint()
        # Optionally request requeue (demo does this)
        # subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=True)
        break  # exit cleanly; next run resumes
    train_step(batch)
```

### Demo: Two Ways to Preempt

1) Simulated preemption (default):

- The demo triggers SIGUSR1 inside Python every ~30s.
- Run as usual: `python demo/example3_production/orchestrate.py slurm submit`
- The job saves a checkpoint and requeues; the next submission resumes.

2) SLURM signal‑driven preemption (recommended for production):

- The SBATCH template includes `#SBATCH --signal=USR1@30`.
- To test quickly, lower the time limit in `config.py` (e.g., `time: "00:02:00"`) and disable the internal simulator:

```bash
export EX3_DISABLE_PREEMPT=1
python demo/example3_production/orchestrate.py slurm submit
```

- SLURM sends SIGUSR1 ~30s before termination; the trainer checkpoints, requests requeue (demo), and exits. The job requeues and resumes from the latest checkpoint.

In the demo, simulated preemption is disabled automatically after the first
requeue (via `SLURM_RESTART_COUNT`) so the next run completes, showcasing a
full preempt→checkpoint→requeue→resume→complete lifecycle.

### When Does Requeuing Stop?

- The trainer only requeues after a preemption event (SIGUSR1). A normal completion does not requeue; the job exits `COMPLETED`.
- Your cluster QoS may also cap the maximum requeues. For stricter control, add a small counter (e.g., env var or file) and stop requeuing after N cycles.
