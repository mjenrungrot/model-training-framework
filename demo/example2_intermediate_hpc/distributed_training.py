"""
Distributed Training - Intermediate HPC Usage

This example demonstrates how to set up and manage distributed training
across multiple compute nodes using SLURM job scheduling. Perfect for:

- Multi-GPU and multi-node training
- Large-scale model training that requires distributed computing
- Batch job submission with automatic resource management
- Production-grade experiment tracking and monitoring

Target Audience: Researchers scaling to multi-node training, HPC teams
"""

from datetime import datetime
import json
from pathlib import Path

from model_training_framework import ModelTrainingFramework
from model_training_framework.config import (
    ConfigurationManager,
    ExecutionMode,
    ExperimentConfig,
)
from model_training_framework.slurm import SLURMLauncher
from model_training_framework.slurm.git_ops import GitManager


def create_distributed_experiment_configs() -> list[ExperimentConfig]:
    """
    Create experiment configurations for distributed training scenarios.

    This function demonstrates how to set up experiments that scale across
    multiple nodes and GPUs, with different distributed training strategies.

    Returns:
        List of experiment configurations for different distributed setups
    """

    # Base configuration for distributed training
    base_config = {
        "description": "Distributed training on HPC cluster",
        # Model configuration - larger models that benefit from distribution
        "model": {
            "name": "large_transformer",
            "num_layers": 12,
            "num_heads": 16,
            "attention_dropout": 0.1,
            "layer_dropout": 0.1,
        },
        # Training configuration for distributed setup
        "training": {
            "epochs": 100,
            "gradient_accumulation_steps": 1,
            "gradient_clip_norm": 1.0,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "logging_steps": 100,
        },
        # Data configuration
        "data": {
            "dataset_name": "large_research_dataset",
            "train_split": "train",
            "val_split": "validation",
            "test_split": "test",
            "num_workers": 8,  # Per GPU
            "pin_memory": True,
        },
        # Optimizer for large-scale training
        "optimizer": {
            "name": "adamw",
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        # Scheduler for long training
        "scheduler": {
            "name": "cosine",
            "warmup_ratio": 0.1,
            "min_lr": 1e-6,
        },
        # Distributed training configuration
        "distributed": {
            "backend": "nccl",  # NCCL for GPU communication
            "init_method": "env://",  # Use environment variables
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
        },
        # Logging and monitoring for distributed training
        "logging": {
            "log_level": "INFO",
            "use_wandb": True,
            "wandb_project": "distributed_training_hpc",
            "log_interval": 50,
            "save_logs": True,
        },
        # Checkpointing for fault tolerance
        "checkpoint": {
            "save_every_n_epochs": 5,
            "save_best_only": False,  # Save all for distributed training
            "checkpoint_dir": "./distributed_checkpoints",
            "async_save": True,  # Non-blocking checkpoint saves
        },
        # Preemption handling for long jobs
        "preemption": {
            "timeout_minutes": 60,  # 1 hour timeout
            "grace_period_seconds": 300,  # 5 minutes grace period
            "save_on_signal": True,
            "resume_from_checkpoint": True,
        },
    }

    experiments = []
    config_manager = ConfigurationManager(Path.cwd())

    # Experiment 1: Single-node multi-GPU (4 GPUs)
    config_single_node = base_config.copy()
    config_single_node.update(
        {
            "experiment_name": "single_node_4gpu_medium",
            "model": {
                **base_config["model"],
                "hidden_size": 768,
            },
            "training": {
                **base_config["training"],
                "batch_size": 32,  # Per GPU
                "learning_rate": 2e-4,
            },
            "slurm": {
                "job_name": "dist_1node_4gpu",
                "time_limit": "24:00:00",
                "nodes": 1,
                "ntasks_per_node": 4,  # 4 GPUs
                "cpus_per_task": 8,  # 8 CPUs per GPU
                "mem": "128G",  # High memory for large model
                "gres": "gpu:4",  # 4 GPUs per node
                "partition": "gpu",
                "exclusive": True,  # Exclusive node access
            },
        }
    )

    # Experiment 2: Multi-node training (2 nodes, 8 GPUs total)
    config_multi_node = base_config.copy()
    config_multi_node.update(
        {
            "experiment_name": "multi_node_2x4gpu_large",
            "model": {
                **base_config["model"],
                "hidden_size": 1024,
            },
            "training": {
                **base_config["training"],
                "batch_size": 24,  # Smaller per-GPU batch for large model
                "learning_rate": 3e-4,
            },
            "slurm": {
                "job_name": "dist_2node_8gpu",
                "time_limit": "36:00:00",
                "nodes": 2,
                "ntasks_per_node": 4,  # 4 GPUs per node
                "cpus_per_task": 10,  # More CPUs for data processing
                "mem": "256G",  # Very high memory
                "gres": "gpu:4",
                "partition": "gpu",
                "exclusive": True,
            },
        }
    )

    # Experiment 3: Large-scale training (4 nodes, 16 GPUs)
    config_large_scale = base_config.copy()
    config_large_scale.update(
        {
            "experiment_name": "large_scale_4x4gpu_xl",
            "model": {
                **base_config["model"],
                "hidden_size": 1536,
                "num_layers": 24,
            },
            "training": {
                **base_config["training"],
                "batch_size": 16,  # Small per-GPU batch for very large model
                "learning_rate": 1e-4,  # Conservative LR for large scale
                "gradient_accumulation_steps": 2,  # Increase effective batch size
            },
            "slurm": {
                "job_name": "dist_4node_16gpu",
                "time_limit": "72:00:00",  # Longer time limit
                "nodes": 4,
                "ntasks_per_node": 4,
                "cpus_per_task": 12,
                "mem": "512G",  # Maximum memory
                "gres": "gpu:4",
                "partition": "gpu",
                "exclusive": True,
                "account": "research",  # May need specific account for large jobs
            },
        }
    )

    # Convert to ExperimentConfig objects
    for config_dict in [config_single_node, config_multi_node, config_large_scale]:
        experiment_config = config_manager.create_experiment_config(config_dict)
        experiments.append(experiment_config)

    return experiments


def create_slurm_distributed_template(template_path: Path) -> None:
    """
    Create a SLURM template optimized for distributed training.

    This template includes all necessary setup for multi-node distributed
    training with proper environment configuration and error handling.
    """

    template_content = """#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --time={{time_limit}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --cpus-per-task={{cpus_per_task}}
#SBATCH --mem={{mem}}
#SBATCH --gres={{gres}}
#SBATCH --partition={{partition}}
{{#exclusive}}#SBATCH --exclusive{{/exclusive}}
{{#account}}#SBATCH --account={{account}}{{/account}}
#SBATCH --output={{output_dir}}/{{experiment_name}}_%j.out
#SBATCH --error={{output_dir}}/{{experiment_name}}_%j.err

# Print job information
echo "=========================================="
echo "SLURM Job Information:"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Environment setup
source ~/.bashrc
# source activate distributed_training  # Uncomment for conda

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# NCCL configuration for optimal performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# CUDA configuration
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "Distributed Training Environment:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Change to project directory
cd {{project_root}}

# Print system information
echo "System Information:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "NCCL version: $(python -c 'import torch; print(torch.cuda.nccl.version())')"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"
echo "GPU information:"
nvidia-smi
echo "=========================================="

# Create output directory
mkdir -p {{output_dir}}/{{experiment_name}}

# Function to handle cleanup on job termination
cleanup() {
    echo "Job termination signal received. Performing cleanup..."
    # Kill all background processes
    pkill -P $$
    echo "Cleanup completed at: $(date)"
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start distributed training
echo "Starting distributed training..."
echo "Command: python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT -m model_training_framework.scripts.distributed_train --config {{config_path}} --experiment-name {{experiment_name}} --output-dir {{output_dir}}"

python -m torch.distributed.launch \\
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \\
    --nnodes=$SLURM_JOB_NUM_NODES \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    -m model_training_framework.scripts.distributed_train \\
    --config {{config_path}} \\
    --experiment-name {{experiment_name}} \\
    --output-dir {{output_dir}} \\
    --distributed

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully at: $(date)"
    echo "Results saved to: {{output_dir}}/{{experiment_name}}"
else
    echo "Training failed at: $(date)"
    echo "Check error logs for details"
    exit 1
fi

echo "=========================================="
echo "Job completed at: $(date)"
"""

    with template_path.open("w") as f:
        f.write(template_content)


def main():
    """
    Main distributed training function.

    Demonstrates the complete workflow for setting up and managing
    distributed training jobs on HPC clusters.
    """

    print("🚀 Distributed Training - HPC Setup")
    print("=" * 50)

    # Setup paths
    project_root = Path.cwd()
    config_dir = project_root / "demo" / "example2_intermediate_hpc" / "configs"
    output_dir = project_root / "distributed_experiments"
    scripts_dir = project_root / "scripts"

    # Create directories
    config_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    scripts_dir.mkdir(exist_ok=True)

    # Initialize framework
    print("🔧 Initializing framework...")
    ModelTrainingFramework(project_root=project_root, config_dir=config_dir)

    # Create experiment configurations
    print("📋 Creating distributed training configurations...")
    experiments = create_distributed_experiment_configs()

    print(f"Created {len(experiments)} distributed training experiments:")
    for i, exp in enumerate(experiments, 1):
        slurm_config = exp.slurm
        total_gpus = (
            slurm_config.nodes * slurm_config.ntasks_per_node if slurm_config else 0
        )

        print(f"   {i}. {exp.experiment_name}")
        print(f"      Model: {exp.model.hidden_size}h x {exp.model.num_layers}l")
        print(
            f"      Scale: {slurm_config.nodes if slurm_config else 0} nodes, "
            f"{total_gpus} GPUs total"
        )
        print(
            f"      Batch: {exp.training.batch_size} per GPU "
            f"(effective: {exp.training.batch_size * total_gpus})"
        )
        print(
            f"      Resources: {slurm_config.mem if slurm_config else 'N/A'}, "
            f"{slurm_config.time_limit if slurm_config else 'N/A'}"
        )
        print()

    # Create SLURM template for distributed training
    slurm_template_path = scripts_dir / "distributed_slurm_template.sh"
    if not slurm_template_path.exists():
        print("🔧 Creating distributed SLURM template...")
        create_slurm_distributed_template(slurm_template_path)
        print(f"   Template created: {slurm_template_path}")

    # Setup SLURM launcher
    print("⚙️  Setting up SLURM launcher...")
    launcher = SLURMLauncher(
        slurm_template_path=str(slurm_template_path), output_dir=str(output_dir)
    )

    # Save experiment configurations for reference
    configs_summary = []
    for exp in experiments:
        config_summary = {
            "experiment_name": exp.experiment_name,
            "model_size": exp.model.hidden_size,
            "num_layers": exp.model.num_layers,
            "batch_size": exp.training.batch_size,
            "learning_rate": exp.training.learning_rate,
            "nodes": exp.slurm.nodes if exp.slurm else 0,
            "gpus_per_node": exp.slurm.ntasks_per_node if exp.slurm else 0,
            "total_gpus": (exp.slurm.nodes * exp.slurm.ntasks_per_node)
            if exp.slurm
            else 0,
        }
        configs_summary.append(config_summary)

    summary_path = output_dir / "experiment_summary.json"
    with summary_path.open("w") as f:
        json.dump(configs_summary, f, indent=2)
    print(f"💾 Saved experiment summary: {summary_path}")

    # Dry run validation
    print("\n🔍 Running dry run validation...")

    try:
        dry_run_result = launcher.submit_experiment_batch(
            experiments=experiments, execution_mode=ExecutionMode.DRY_RUN
        )

        print("✅ Dry run validation completed")
        print(
            f"   Success rate: {len(dry_run_result.successful_jobs)}/{len(experiments)}"
        )

        if dry_run_result.failed_jobs:
            print("⚠️  Some configurations had issues:")
            for exp_name, error in dry_run_result.failed_jobs.items():
                print(f"      {exp_name}: {error}")
        else:
            print("🎯 All configurations validated successfully")

    except Exception as e:
        print(f"❌ Dry run validation failed: {e}")
        return

    # Git integration demonstration
    print("\n🌿 Git integration for experiment tracking...")

    try:
        git_manager = GitManager(str(project_root))

        # Create experiment branch
        branch_name = (
            f"distributed_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        with git_manager.branch_context(branch_name):
            print(f"   Created experiment branch: {branch_name}")
            print(f"   Commit hash: {git_manager.get_current_commit()}")

            # Save git info with experiments
            git_info = {
                "branch": branch_name,
                "commit": git_manager.get_current_commit(),
                "timestamp": datetime.now().isoformat(),
                "experiments": [exp.experiment_name for exp in experiments],
            }

            git_info_path = output_dir / "git_info.json"
            with git_info_path.open("w") as f:
                json.dump(git_info, f, indent=2)

        print("   Returned to original branch")

    except Exception as e:
        print(f"⚠️  Git integration failed: {e}")
        print("   (This is normal if not in a git repository)")

    # Ready for submission
    print("\n🚀 Ready for SLURM submission!")
    print("=" * 30)

    print("To submit the jobs, uncomment the submission section below.")
    print("Consider starting with the smallest experiment first.")

    # Commented out for safety - uncomment to actually submit
    """
    print("Submitting distributed training jobs...")

    try:
        submission_result = launcher.submit_experiment_batch(
            experiments=experiments,
            execution_mode=ExecutionMode.SLURM,
            max_concurrent_jobs=2  # Limit concurrent large jobs
        )

        print(f"✅ Successfully submitted {len(submission_result.successful_jobs)} jobs")

        for job_id in submission_result.successful_jobs:
            print(f"   Job ID: {job_id}")

        if submission_result.failed_jobs:
            print("❌ Failed submissions:")
            for exp_name, error in submission_result.failed_jobs.items():
                print(f"   {exp_name}: {error}")

        # Monitor jobs
        print("\\n📊 Monitoring distributed training jobs...")
        print("Use the job monitoring utilities to track progress")

    except Exception as e:
        print(f"❌ Submission failed: {e}")
    """

    # Provide next steps and monitoring guidance
    print("\n📋 Next Steps:")
    print("1. Review experiment configurations and resource requirements")
    print("2. Test with the smallest experiment first")
    print("3. Monitor GPU utilization and communication efficiency")
    print("4. Check for proper load balancing across nodes")
    print("5. Use distributed training best practices")

    print("\n💡 Monitoring Commands:")
    print("   squeue -u $USER                    # Check job queue")
    print("   scontrol show job <job_id>         # Detailed job info")
    print("   ssh <node> nvidia-smi              # Check GPU utilization")
    print("   tail -f <output_file>.out          # Monitor training progress")

    print("\n🔧 Troubleshooting Tips:")
    print("   - Check NCCL communication between nodes")
    print("   - Verify network configuration for multi-node jobs")
    print("   - Monitor memory usage on all nodes")
    print("   - Check for load balancing issues")

    print("\n✅ Distributed training setup complete!")


def demonstrate_distributed_best_practices():
    """
    Demonstrate best practices for distributed training on HPC systems.
    """

    print("\n🎯 Distributed Training Best Practices")
    print("-" * 40)

    practices = [
        "1. Use gradient accumulation to maintain effective batch size",
        "2. Scale learning rate with number of GPUs (linear scaling rule)",
        "3. Use warm-up periods for large-scale training",
        "4. Monitor communication overhead vs computation",
        "5. Use appropriate data loading strategies (multiple workers)",
        "6. Implement proper checkpointing for fault tolerance",
        "7. Use mixed precision training to reduce memory usage",
        "8. Profile your training to identify bottlenecks",
    ]

    for practice in practices:
        print(f"   {practice}")

    print("\n🚨 Common Issues and Solutions:")
    issues = [
        "NCCL timeouts → Check network configuration and firewall",
        "Uneven GPU utilization → Check data distribution and loading",
        "Memory errors → Reduce batch size or use gradient checkpointing",
        "Slow training → Profile communication vs computation ratio",
    ]

    for issue in issues:
        print(f"   {issue}")

    print("\n✅ Best practices review complete")


if __name__ == "__main__":
    main()
    demonstrate_distributed_best_practices()
