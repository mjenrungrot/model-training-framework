# Example 3: Complete Production Workflow

This example demonstrates all three main components of the Model Training Framework working together in a production-ready setup for HPC clusters.

## 🎯 Three Components Integration

### 1. Config Search Sweep (`config.py`)

Generates multiple experiment configurations using parameter grid search:

```python
# config.py demonstrates:
def build_parameter_grid_search(base_config):
    gs = ParameterGridSearch(base_config)
    gs.set_naming_strategy(NamingStrategy.PARAMETER_BASED)

    grid = ParameterGrid("prod_demo")
    grid.add_parameter("optimizer.lr", [1e-4, 3e-4])
    grid.add_parameter("data.batch_size", [16])
    grid.add_parameter("training.gradient_accumulation_steps", [1, 2])
    grid.add_parameter("custom_params.multi_loader.sampling_strategy",
                      ["round_robin", "weighted"])

    gs.add_grid(grid)
    return gs
```

Generates 8 experiment configurations with different hyperparameters.

### 2. SLURM Launcher (`orchestrate.py`)

Submits experiments to SLURM cluster with automatic job management:

```python
# orchestrate.py demonstrates:
launcher = SLURMLauncher(
    template_path=template_path,
    project_root=repo_root,
    experiments_dir=experiments_dir,
)

result = launcher.submit_experiment_batch(
    experiments=experiments,
    script_path=script_path,
    use_git_branch=False,
    dry_run=not submit,  # Preview mode available
)
```

Creates SBATCH scripts with preemption handling and requeue support.

### 3. Fault-Tolerant Trainer (`train_script.py`)

Trains models with automatic checkpointing and exact resume:

```python
# train_script.py demonstrates:
trainer = GenericTrainer(
    config=trainer_config,
    model=model,
    optimizers=[optimizer],
    fabric=fabric,
)

# Automatic resume from checkpoint
latest = trainer.checkpoint_manager.get_latest_checkpoint()
if latest:
    trainer.load_checkpoint(latest)
    logger.info(f"Resumed from {latest}")

# Train with preemption handling
trainer.fit(
    train_loaders=train_loaders,
    val_loaders=val_loaders,
    max_epochs=config["training"]["max_epochs"],
)
```

## 📁 Files Overview

- **`config.py`**: Defines base configuration and parameter grid search
- **`orchestrate.py`**: Main entry point - coordinates all three components
- **`train_script.py`**: Training script with fault tolerance and multi-loader support
- **`data.py`**: Creates synthetic datasets for demonstration
- **`model.py`**: Simple MLP model for quick training
- **`_slurm_template.sbatch`**: SLURM template (auto-generated)

## 🚀 Quick Start

### Preview Configuration

See what experiments will be generated:

```bash
python demo/example3_production/config.py
```

### Local Execution

Run experiments locally (sequential):

```bash
python demo/example3_production/orchestrate.py local
```

### SLURM Submission

**Dry run** (preview SBATCH scripts without submitting):

```bash
python demo/example3_production/orchestrate.py slurm
```

**Actual submission** to SLURM:

```bash
python demo/example3_production/orchestrate.py slurm submit
```

## 🔄 Workflow Demonstration

### Step 1: Configuration Generation

The `config.py` module creates:

- Base configuration with model, optimizer, and training settings
- Parameter grid with 4 hyperparameters to search
- 8 total experiment configurations

### Step 2: Job Orchestration

The `orchestrate.py` script:

- Loads configurations from Step 1
- Creates experiment directories
- Generates SLURM scripts from template
- Submits jobs with proper dependencies

### Step 3: Fault-Tolerant Training

The `train_script.py`:

- Loads assigned configuration
- Creates multi-dataloader setup (2 loaders)
- Implements automatic checkpointing
- Handles SIGUSR1 for graceful preemption
- Resumes from exact batch on restart

## 🔧 Key Features Demonstrated

### Multi-DataLoader Training

```python
# Two dataloaders with configurable sampling
train_loaders = [
    create_synthetic_loader(64, 10, batch_size, num_workers),
    create_synthetic_loader(32, 10, batch_size, num_workers),
]

# Sampling strategy from config
sampling_strategy = config["custom_params"]["multi_loader"]["sampling_strategy"]
if sampling_strategy == "weighted":
    weights = config["custom_params"]["multi_loader"]["dataloader_weights"]
```

### Preemption Handling

```python
# Signal handler for SLURM preemption
def _on_sigusr1(signum, frame):
    logger.warning("Received SIGUSR1; will checkpoint and exit")
    global preempt_requested
    preempt_requested = True

signal.signal(signal.SIGUSR1, _on_sigusr1)
```

### Automatic Resume

```python
# Check for existing checkpoints
latest = trainer.checkpoint_manager.get_latest_checkpoint()
if latest:
    trainer.load_checkpoint(latest)
    # Training continues from exact batch/epoch
```

## 📊 Output Structure

After running, you'll find:

```text
experiments/
├── prod_demo_lr1e-4_bs16_ga1_ssround_robin/
│   ├── config.json
│   ├── checkpoints/
│   │   ├── epoch_0_step_10.ckpt
│   │   └── latest.ckpt -> epoch_0_step_10.ckpt
│   ├── local_run.log
│   └── slurm_12345.out  # If submitted to SLURM
├── prod_demo_lr1e-4_bs16_ga1_ssweighted/
│   └── ...
└── ... (6 more experiments)
```

## 🔍 Monitoring Progress

### Local Runs

```bash
# Watch log file
tail -f experiments/*/local_run.log

# Check training progress
grep "Epoch" experiments/*/local_run.log
```

### SLURM Jobs

```bash
# Check job status
squeue -u $USER

# Monitor output
tail -f experiments/*/slurm_*.out

# Check for checkpoints
ls -la experiments/*/checkpoints/
```

## ⚙️ Configuration Options

### Adjust Training Duration

```python
# config.py
"training": {"max_epochs": 2}  # Increase for longer training
```

### Change Parameter Search

```python
# config.py
grid.add_parameter("optimizer.lr", [1e-5, 1e-4, 1e-3])  # More learning rates
grid.add_parameter("model.hidden_size", [64, 128, 256])  # Model sizes
```

### Modify SLURM Settings

```python
# config.py
"slurm": {
    "partition": "gpu",        # Your partition
    "account": "my-account",   # Your account
    "time": "04:00:00",       # Time limit
    "gpus_per_node": 1,       # GPU count
}
```

## 🐛 Troubleshooting

### Issue: SLURM submission fails

- Check account/partition names in `config.py`
- Verify SLURM is available: `sinfo`
- Review generated scripts: `cat experiments/*/job_*.sbatch`

### Issue: Training doesn't resume

- Check checkpoint exists: `ls experiments/*/checkpoints/`
- Verify checkpoint loading in logs: `grep "Resumed" experiments/*/local_run.log`

### Issue: Out of memory

- Reduce batch size in `config.py`
- Increase gradient accumulation steps
- Request more memory in SLURM config

## 📚 Learn More

- [Main README](../../README.md) - Framework overview
- [API Documentation](../../docs/API.md) - Complete API reference
- [Configuration Guide](../../docs/CONFIGURATION.md) - All configuration options
- [SLURM Guide](../../docs/ADVANCED_FEATURES.md#preemption-handling) - Preemption details

## 💡 Next Steps

1. **Customize the model**: Edit `model.py` to use your architecture
2. **Use real data**: Replace `data.py` with your dataset loaders
3. **Add metrics**: Extend `train_script.py` with your metrics
4. **Scale up**: Increase parameter grid and submit more jobs
5. **Enable logging**: Add WandB or TensorBoard in config
