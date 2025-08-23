"""
SLURM Batch Job Submission Example

This example demonstrates how to submit multiple training jobs
to SLURM with automatic git branch management and job monitoring.
"""

from pathlib import Path

from model_training_framework import ModelTrainingFramework
from model_training_framework.config import (
    ConfigurationManager,
    ExecutionMode,
    ExperimentConfig,
)
from model_training_framework.slurm import SLURMLauncher
from model_training_framework.slurm.git_ops import GitManager


def create_experiment_configs() -> list[ExperimentConfig]:
    """Create multiple experiment configurations for batch submission."""

    base_config = {
        "description": "SLURM batch submission example",
        "model": {"name": "transformer", "num_layers": 6, "num_heads": 8},
        "training": {"epochs": 100, "batch_size": 32, "gradient_accumulation_steps": 1},
        "data": {
            "dataset_name": "custom_dataset",
            "train_split": "train",
            "val_split": "validation",
        },
        "optimizer": {"name": "adamw", "weight_decay": 0.01},
        "scheduler": {"name": "cosine", "warmup_steps": 1000},
        "slurm": {
            "job_name": "batch_training",
            "time_limit": "24:00:00",
            "nodes": 1,
            "ntasks_per_node": 1,
            "cpus_per_task": 4,
            "mem": "32G",
            "gres": "gpu:1",
            "partition": "gpu",
        },
        "logging": {
            "log_level": "INFO",
            "use_wandb": True,
            "wandb_project": "slurm_batch_example",
        },
    }

    # Create multiple configurations with different parameters
    experiments = []

    # Experiment 1: Small model
    config1 = base_config.copy()
    config1.update(
        {
            "experiment_name": "small_model_lr1e-3",
            "model": {**base_config["model"], "hidden_size": 256},
            "training": {**base_config["training"], "learning_rate": 1e-3},
        }
    )

    # Experiment 2: Medium model
    config2 = base_config.copy()
    config2.update(
        {
            "experiment_name": "medium_model_lr5e-4",
            "model": {**base_config["model"], "hidden_size": 512},
            "training": {**base_config["training"], "learning_rate": 5e-4},
        }
    )

    # Experiment 3: Large model
    config3 = base_config.copy()
    config3.update(
        {
            "experiment_name": "large_model_lr1e-4",
            "model": {**base_config["model"], "hidden_size": 1024},
            "training": {**base_config["training"], "learning_rate": 1e-4},
            "slurm": {**base_config["slurm"], "mem": "64G", "time_limit": "36:00:00"},
        }
    )

    # Convert to ExperimentConfig objects
    config_manager = ConfigurationManager(Path.cwd())

    for config_dict in [config1, config2, config3]:
        experiment_config = config_manager.create_experiment_config(config_dict)
        experiments.append(experiment_config)

    return experiments


def main():
    """Main SLURM batch submission example."""

    print("🚀 SLURM Batch Job Submission Example")

    # Setup paths
    project_root = Path.cwd()
    config_dir = project_root / "configs"
    output_dir = project_root / "slurm_experiments"

    # Initialize framework
    ModelTrainingFramework(project_root=project_root, config_dir=config_dir)

    # Create experiment configurations
    print("📋 Creating experiment configurations...")
    experiments = create_experiment_configs()

    for i, exp in enumerate(experiments, 1):
        print(f"   {i}. {exp.experiment_name}")
        print(f"      Model size: {exp.model.hidden_size}")
        print(f"      Learning rate: {exp.training.learning_rate}")
        print(f"      Memory: {exp.slurm.mem if exp.slurm else 'N/A'}")
        print()

    # Setup SLURM launcher
    print("⚙️  Setting up SLURM launcher...")

    # Create SLURM template if it doesn't exist
    slurm_template_path = project_root / "scripts" / "slurm_template.txt"
    slurm_template_path.parent.mkdir(exist_ok=True)

    if not slurm_template_path.exists():
        create_slurm_template(slurm_template_path)
        print(f"   Created SLURM template: {slurm_template_path}")

    # Initialize SLURM launcher
    launcher = SLURMLauncher(
        slurm_template_path=str(slurm_template_path), output_dir=str(output_dir)
    )

    # Option 1: Dry run to validate configurations
    print("🔍 Running dry run validation...")

    try:
        dry_run_result = launcher.submit_experiment_batch(
            experiments=experiments, execution_mode=ExecutionMode.DRY_RUN
        )

        print("✅ Dry run validation passed")
        print(f"   Would submit {len(dry_run_result.successful_jobs)} jobs")

        if dry_run_result.failed_jobs:
            print("⚠️  Some configurations had issues:")
            for exp_name, error in dry_run_result.failed_jobs.items():
                print(f"      {exp_name}: {error}")

    except Exception as e:
        print(f"❌ Dry run failed: {e}")
        return

    # Option 2: Actual SLURM submission (commented out for safety)
    """
    print("🚀 Submitting jobs to SLURM...")

    try:
        submission_result = launcher.submit_experiment_batch(
            experiments=experiments,
            execution_mode=ExecutionMode.SLURM,
            max_concurrent_jobs=3  # Limit concurrent jobs
        )

        print(f"✅ Successfully submitted {len(submission_result.successful_jobs)} jobs")

        # Display submitted job IDs
        for job_id in submission_result.successful_jobs:
            print(f"   Job ID: {job_id}")

        # Display any failures
        if submission_result.failed_jobs:
            print("❌ Failed to submit some jobs:")
            for exp_name, error in submission_result.failed_jobs.items():
                print(f"   {exp_name}: {error}")

        # Monitor job status
        print("📊 Monitoring job status...")
        status_summary = launcher.get_job_status_summary(submission_result.successful_jobs)

        for status, count in status_summary.items():
            print(f"   {status}: {count} jobs")

    except Exception as e:
        print(f"❌ SLURM submission failed: {e}")
        return
    """

    # Option 3: Git integration demo
    print("🌿 Demonstrating git integration...")

    try:
        git_manager = GitManager(str(project_root))

        # Create experiment branch
        branch_name = "experiment_batch_demo"

        with git_manager.branch_context(branch_name):
            print(f"   Working on branch: {branch_name}")

            # Here you would make any code changes needed for the experiments
            # For demo, we'll just show the current branch
            current_branch = git_manager.get_current_branch()
            print(f"   Current branch: {current_branch}")

        print("   Returned to original branch")

    except Exception as e:
        print(f"⚠️  Git integration demo failed: {e}")
        print("   (This is expected if not in a git repository)")

    print("\n✅ SLURM batch submission example completed!")

    # Show how to check job status later
    print("\n📋 To check job status later, use:")
    print("   squeue -u $USER")
    print("   sacct -j <job_id>")
    print("   scancel <job_id>  # to cancel a job")


def create_slurm_template(template_path: Path):
    """Create a basic SLURM template file."""

    template_content = """#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --time={{time_limit}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --cpus-per-task={{cpus_per_task}}
#SBATCH --mem={{mem}}
#SBATCH --gres={{gres}}
#SBATCH --partition={{partition}}
#SBATCH --output={{output_dir}}/{{experiment_name}}.out
#SBATCH --error={{output_dir}}/{{experiment_name}}.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo ""

# Environment setup
# source activate myenv  # Uncomment and modify for your environment
cd {{project_root}}

# Print environment info
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "GPU info:"
nvidia-smi || echo "No GPUs available"
echo ""

# Run the training script
echo "Starting training..."
python -m model_training_framework.scripts.train \\
    --config {{config_path}} \\
    --experiment-name {{experiment_name}} \\
    --output-dir {{output_dir}}

echo "Training completed at: $(date)"
"""

    with template_path.open("w") as f:
        f.write(template_content)


def demonstrate_advanced_slurm_features():
    """Demonstrate advanced SLURM features."""

    print("\n🔬 Advanced SLURM Features Demo:")

    # 1. Job dependencies
    print("   1. Job Dependencies:")
    print("      - Use --dependency=afterok:<job_id> for sequential jobs")
    print("      - Use --dependency=afterany:<job_id> for jobs that run regardless")

    # 2. Array jobs
    print("   2. Array Jobs:")
    print("      - Use #SBATCH --array=1-10 for parameter sweeps")
    print("      - Access array index with $SLURM_ARRAY_TASK_ID")

    # 3. Resource allocation
    print("   3. Advanced Resource Allocation:")
    print("      - GPU types: --gres=gpu:v100:2, --gres=gpu:a100:1")
    print("      - Memory per CPU: --mem-per-cpu=4G")
    print("      - Exclusive nodes: --exclusive")

    # 4. Job monitoring
    print("   4. Job Monitoring:")
    print(
        "      - Real-time: squeue -u $USER --format='%.10i %.20j %.8t %.10M %.6D %R'"
    )
    print(
        "      - Historical: sacct -j <job_id> --format=JobID,JobName,State,ExitCode,DerivedExitCode"
    )
    print("      - Efficiency: seff <job_id>")

    print("✅ Advanced features demo completed")


if __name__ == "__main__":
    main()
    demonstrate_advanced_slurm_features()
