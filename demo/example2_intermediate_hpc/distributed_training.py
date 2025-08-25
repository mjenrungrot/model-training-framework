"""
Distributed Training with Multi-DataLoader Architecture - Intermediate HPC Usage

This example demonstrates distributed training using the multi-dataloader-only
architecture across multiple compute nodes with SLURM job scheduling. Perfect for:

- Multi-GPU and multi-node training with multiple datasets
- Large-scale model training with the multi-loader API
- Batch job submission with automatic resource management
- Production-grade experiment tracking with multiple data sources

Target Audience: Researchers scaling to multi-node training, HPC teams
"""

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# Multi-loader architecture imports
from model_training_framework.trainer import (
    CheckpointConfig,
    DDPConfig,
    FaultToleranceConfig,
    GenericTrainerConfig,
    LoggingConfig,
    MultiDataLoaderConfig,
    PerformanceConfig,
    PreemptionConfig,
    ValidationConfig,
)
from model_training_framework.trainer.config import (
    EpochLengthPolicy,
    SamplingStrategy,
    ValAggregation,
    ValidationFrequency,
)

# Note: In production, you'd import SLURM helpers from your utilities, e.g.
# Example import: model_training_framework.slurm.SLURMLauncher
# Example import: model_training_framework.slurm.git_ops.GitManager


class DistributedTransformer(nn.Module):
    """Large transformer model for distributed training."""

    def __init__(self, vocab_size=50000, hidden_size=768, num_layers=12, num_heads=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        return self.output(x)


def create_distributed_experiment_configs() -> list[GenericTrainerConfig]:
    """
    Create experiment configurations for distributed training with multi-loader architecture.

    This function demonstrates various distributed setups, including:
    - Single-node multi-GPU with single dataloader
    - Multi-node training with single dataloader
    - Multi-node training with multiple dataloaders

    Returns:
        List of GenericTrainerConfig for different distributed setups
    """

    configs = []

    # ===========================================================================
    # Config 1: Single-node multi-GPU (4 GPUs) with single dataloader
    # ===========================================================================
    single_node_config = GenericTrainerConfig(
        # Multi-loader config (required even for single loader)
        multi=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            dataloader_names=["main"],  # Single loader in list
        ),
        # Distributed configuration
        ddp=DDPConfig(
            backend="nccl",
            sync_schedules_across_ranks=True,
            validate_schedule_consistency=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=True,
            bucket_cap_mb=25,
        ),
        # Performance settings for distributed training
        performance=PerformanceConfig(
            gradient_accumulation_steps=1,
            use_amp=True,  # Mixed precision for efficiency
            clip_grad_norm=1.0,
            dataloader_num_workers=8,  # Per GPU
            pin_memory=True,
            persistent_workers=True,
        ),
        # Checkpointing for fault tolerance
        checkpoint=CheckpointConfig(
            root_dir="./distributed_checkpoints/single_node",
            save_every_n_epochs=5,
            save_every_n_steps=1000,
            max_checkpoints=10,
            save_rng=True,
            save_optimizer=True,
        ),
        # Fault tolerance settings
        fault_tolerance=FaultToleranceConfig(
            save_sampler_state=True,
            save_dataset_state=True,
            verify_deterministic_resume=True,
            checkpoint_timeout_sec=300.0,
        ),
        # Preemption handling for HPC
        preemption=PreemptionConfig(
            max_checkpoint_sec=300.0,
            requeue_job=True,
            resume_from_latest_symlink=True,
        ),
        # Logging configuration
        logging=LoggingConfig(
            logger_type="composite",
            composite_loggers=["console", "tensorboard"],
            log_per_loader_metrics=True,
            log_loader_proportions=False,  # Not needed for single loader
            all_reduce_metrics=True,  # Aggregate across GPUs
        ),
        # Validation configuration
        validation=ValidationConfig(
            frequency=ValidationFrequency.PER_EPOCH,
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
        ),
    )
    configs.append(("single_node_4gpu", single_node_config))

    # ===========================================================================
    # Config 2: Multi-node training (2 nodes, 8 GPUs) with single dataloader
    # ===========================================================================
    multi_node_config = GenericTrainerConfig(
        multi=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.SEQUENTIAL,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            dataloader_names=["main"],
        ),
        ddp=DDPConfig(
            backend="nccl",
            sync_schedules_across_ranks=True,
            validate_schedule_consistency=True,
            # Larger bucket for multi-node communication
            bucket_cap_mb=50,
        ),
        performance=PerformanceConfig(
            gradient_accumulation_steps=2,  # Larger effective batch
            use_amp=True,
            clip_grad_norm=1.0,
            dataloader_num_workers=10,
        ),
        checkpoint=CheckpointConfig(
            root_dir="./distributed_checkpoints/multi_node",
            save_every_n_epochs=2,
            save_every_n_steps=500,
        ),
        fault_tolerance=FaultToleranceConfig(
            save_sampler_state=True,
            save_dataset_state=True,
        ),
        logging=LoggingConfig(
            logger_type="wandb",
            wandb_project="distributed_multi_node",
            all_reduce_metrics=True,
        ),
        validation=ValidationConfig(
            frequency=ValidationFrequency.EVERY_N_STEPS,
            every_n_steps=500,
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
        ),
    )
    configs.append(("multi_node_2x4gpu", multi_node_config))

    # ===========================================================================
    # Config 3: Multi-node with multiple dataloaders (Advanced)
    # ===========================================================================
    multi_loader_distributed_config = GenericTrainerConfig(
        # Multiple dataloaders with round-robin sampling
        multi=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.ROUND_ROBIN,
            epoch_length_policy=EpochLengthPolicy.SUM_OF_LENGTHS,
            dataloader_names=["primary", "auxiliary", "augmented"],
            cycle_short_loaders=True,  # Important for uneven datasets
        ),
        ddp=DDPConfig(
            backend="nccl",
            sync_schedules_across_ranks=True,
            validate_schedule_consistency=True,  # Critical for multi-loader
        ),
        performance=PerformanceConfig(
            gradient_accumulation_steps=4,
            use_amp=True,
            clip_grad_norm=1.0,
        ),
        checkpoint=CheckpointConfig(
            root_dir="./distributed_checkpoints/multi_loader",
            save_every_n_epochs=1,
        ),
        logging=LoggingConfig(
            logger_type="composite",
            composite_loggers=["console", "tensorboard", "wandb"],
            log_per_loader_metrics=True,  # Track each dataset
            log_loader_proportions=True,  # Monitor sampling balance
            all_reduce_metrics=True,
        ),
        validation=ValidationConfig(
            frequency=ValidationFrequency.PER_EPOCH,
            aggregation=ValAggregation.MACRO_AVG_EQUAL_LOADERS,  # Equal weight per dataset
            per_loader_metrics=True,
        ),
    )
    configs.append(("multi_node_multi_loader", multi_loader_distributed_config))

    # ===========================================================================
    # Config 4: Large-scale with weighted multi-loader sampling
    # ===========================================================================
    weighted_distributed_config = GenericTrainerConfig(
        multi=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            dataloader_weights=[0.6, 0.3, 0.1],  # Primary, secondary, rare
            epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
            steps_per_epoch=10000,  # Fixed steps for consistent epochs
            dataloader_names=["primary", "secondary", "rare"],
            choice_rng_seed=42,  # Deterministic weighted sampling
        ),
        ddp=DDPConfig(
            backend="nccl",
            sync_schedules_across_ranks=True,
            validate_schedule_consistency=True,
        ),
        performance=PerformanceConfig(
            gradient_accumulation_steps=8,  # Very large effective batch
            use_amp=True,
        ),
        checkpoint=CheckpointConfig(
            root_dir="./distributed_checkpoints/weighted",
            save_every_n_steps=2000,
        ),
        logging=LoggingConfig(
            logger_type="wandb",
            log_per_loader_metrics=True,
            log_loader_proportions=True,  # Monitor actual vs expected weights
            all_reduce_metrics=True,
        ),
        validation=ValidationConfig(
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
        ),
    )
    configs.append(("large_scale_weighted", weighted_distributed_config))

    return configs


def distributed_training_step(trainer, batch, dataloader_idx, dataloader_name):
    """
    Training step for distributed multi-loader training.

    This function handles batches from any dataloader in the distributed setting.

    Args:
        trainer: GenericTrainer instance
        batch: Current batch (inputs, targets)
        dataloader_idx: Index of the current dataloader
        dataloader_name: Name of the current dataloader

    Returns:
        Dictionary with loss and metrics
    """
    inputs, targets = batch

    # Forward pass
    outputs = trainer.model(inputs)

    # Compute loss (assuming language modeling)
    loss = F.cross_entropy(
        outputs.view(-1, outputs.size(-1)),
        targets.view(-1),
        ignore_index=-100,  # Padding token
    )

    # Compute metrics
    with torch.no_grad():
        _, predicted = outputs.max(-1)
        mask = targets != -100
        accuracy = ((predicted == targets) & mask).float().sum() / mask.sum()

    # Return metrics with dataloader prefix
    return {
        "loss": loss,
        f"{dataloader_name}/loss": loss.item(),
        f"{dataloader_name}/accuracy": accuracy.item(),
        f"{dataloader_name}/perplexity": torch.exp(loss).item(),
    }


def distributed_validation_step(trainer, batch, dataloader_idx, dataloader_name):
    """Validation step for distributed multi-loader training."""
    inputs, targets = batch

    with torch.no_grad():
        outputs = trainer.model(inputs)
        loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-100
        )

        _, predicted = outputs.max(-1)
        mask = targets != -100
        accuracy = ((predicted == targets) & mask).float().sum() / mask.sum()

    return {
        "val_loss": loss,
        f"val_{dataloader_name}/loss": loss.item(),
        f"val_{dataloader_name}/accuracy": accuracy.item(),
    }


def create_distributed_dataloaders(
    batch_size: int, world_size: int, rank: int, num_loaders: int = 1
) -> tuple[list, list]:
    """
    Create distributed dataloaders with proper sampling.

    Args:
        batch_size: Batch size per GPU
        world_size: Total number of processes
        rank: Current process rank
        num_loaders: Number of dataloaders to create

    Returns:
        Tuple of (train_loaders, val_loaders)
    """
    train_loaders = []
    val_loaders = []

    for i in range(num_loaders):
        # Create synthetic datasets (in production, load real data)
        torch.manual_seed(42 + i)  # Different seed per dataset

        # Different sizes for different datasets
        train_size = 10000 * (i + 1)
        val_size = 1000 * (i + 1)

        # Create tensors (simulating tokenized text)
        train_data = TensorDataset(
            torch.randint(0, 50000, (train_size, 128)),  # Input tokens
            torch.randint(0, 50000, (train_size, 128)),  # Target tokens
        )
        val_data = TensorDataset(
            torch.randint(0, 50000, (val_size, 128)),
            torch.randint(0, 50000, (val_size, 128)),
        )

        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        )
        val_sampler = DistributedSampler(
            val_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size * 2,  # Larger batch for validation
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True,
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders


def create_slurm_distributed_template(template_path: Path) -> None:
    """
    Create a SLURM template for distributed multi-loader training.

    This template handles multi-node setup and calls the training script
    with proper multi-loader configuration.
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
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Node list: $SLURM_JOB_NODELIST"
echo "=========================================="

# Environment setup
source ~/.bashrc
# module load cuda/11.8  # Adjust for your system
# source activate distributed_env

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# NCCL configuration
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# CUDA configuration
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "Distributed Environment:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "=========================================="

# Change to project directory
cd {{project_root}}

# Run distributed training with multi-loader architecture
srun python demo/example2_intermediate_hpc/distributed_train_script.py \\
    --config-name {{config_name}} \\
    --num-loaders {{num_loaders}} \\
    --batch-size {{batch_size}} \\
    --learning-rate {{learning_rate}} \\
    --max-epochs {{max_epochs}} \\
    --output-dir {{output_dir}}/{{experiment_name}}

echo "=========================================="
echo "Job completed at: $(date)"
"""

    with template_path.open("w") as f:
        f.write(template_content)


def create_distributed_training_script(script_path: Path) -> None:
    """
    Create the actual distributed training script that will be called by SLURM.

    This script demonstrates proper multi-loader usage in distributed setting.
    """

    script_content = '''#!/usr/bin/env python
"""
Distributed training script for multi-loader architecture.
This script is called by SLURM with proper environment variables set.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from lightning.fabric import Fabric

from demo.example2_intermediate_hpc.distributed_training import (
    DistributedTransformer,
    create_distributed_dataloaders,
    distributed_training_step,
    distributed_validation_step,
    create_distributed_experiment_configs,
)
from model_training_framework.trainer import GenericTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--num-loaders", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()

    # Get distributed info from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize Fabric for distributed training
    fabric = Fabric(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
    )
    fabric.launch()

    # Get configuration
    configs = create_distributed_experiment_configs()
    config = None
    for name, cfg in configs:
        if name == args.config_name:
            config = cfg
            break

    if config is None:
        raise ValueError(f"Config {args.config_name} not found")

    # Create model
    model = DistributedTransformer(
        vocab_size=50000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
    )

    # Create optimizer (always a list for multi-loader architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Setup with Fabric
    model, optimizer = fabric.setup(model, optimizer)

    # Create dataloaders
    train_loaders, val_loaders = create_distributed_dataloaders(
        batch_size=args.batch_size,
        world_size=world_size,
        rank=rank,
        num_loaders=args.num_loaders,
    )

    # Setup dataloaders with Fabric
    train_loaders = [fabric.setup_dataloaders(loader) for loader in train_loaders]
    val_loaders = [fabric.setup_dataloaders(loader) for loader in val_loaders]

    # Create trainer with multi-loader API
    trainer = GenericTrainer(
        config=config,
        model=model,
        optimizers=[optimizer],  # Always a list
        fabric=fabric,
    )

    # Set training and validation functions
    trainer.set_training_step(distributed_training_step)
    trainer.set_validation_step(distributed_validation_step)

    # Train with multi-loader API
    trainer.fit(
        train_loaders=train_loaders,  # List of loaders
        val_loaders=val_loaders,      # List of loaders
        max_epochs=args.max_epochs,
    )

    print(f"Training completed on rank {rank}")


if __name__ == "__main__":
    main()
'''

    with script_path.open("w") as f:
        f.write(script_content)

    # Make script executable (demo-only)  # nosec B103
    script_path.chmod(0o755)


def main():
    """
    Main function demonstrating distributed training with multi-loader architecture.
    """

    print("🚀 Distributed Training with Multi-DataLoader Architecture")
    print("=" * 60)

    # Setup paths
    project_root = Path.cwd()
    config_dir = project_root / "demo" / "example2_intermediate_hpc" / "configs"
    output_dir = project_root / "distributed_experiments"
    scripts_dir = project_root / "demo" / "example2_intermediate_hpc"

    # Create directories
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment configurations
    print("\n📋 Creating distributed training configurations...")
    configs = create_distributed_experiment_configs()

    print(f"Created {len(configs)} distributed configurations:\n")
    for i, (name, config) in enumerate(configs, 1):
        print(f"{i}. {name}")
        print(f"   Sampling Strategy: {config.multi.sampling_strategy.value}")
        print(f"   DataLoaders: {config.multi.dataloader_names}")
        print(f"   DDP Backend: {config.ddp.backend if config.ddp else 'None'}")
        print(
            f"   Gradient Accumulation: {config.performance.gradient_accumulation_steps}"
        )
        print()

    # Create SLURM template
    slurm_template_path = scripts_dir / "slurm_template.sh"
    print("📝 Creating SLURM template for distributed training...")
    create_slurm_distributed_template(slurm_template_path)
    print(f"   Template saved: {slurm_template_path}")

    # Create training script
    train_script_path = scripts_dir / "distributed_train_script.py"
    print("\n📜 Creating distributed training script...")
    create_distributed_training_script(train_script_path)
    print(f"   Script saved: {train_script_path}")

    # Save configurations for reference
    print("\n💾 Saving configuration details...")
    for name, config in configs:
        config_path = config_dir / f"{name}_config.json"
        config_dict = {
            "name": name,
            "sampling_strategy": config.multi.sampling_strategy.value,
            "dataloader_names": config.multi.dataloader_names,
            "ddp_backend": config.ddp.backend if config.ddp else None,
            "checkpoint_dir": str(config.checkpoint.root_dir),
        }
        with config_path.open("w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"   Saved: {config_path.name}")

    # Provide usage instructions
    print("\n" + "=" * 60)
    print("📚 Usage Instructions")
    print("=" * 60)

    print("\n1. Single-Node Multi-GPU (4 GPUs):")
    print("   sbatch --job-name=single_node \\")
    print("          --nodes=1 --ntasks-per-node=4 --gres=gpu:4 \\")
    print("          slurm_template.sh")

    print("\n2. Multi-Node (2 nodes, 8 GPUs total):")
    print("   sbatch --job-name=multi_node \\")
    print("          --nodes=2 --ntasks-per-node=4 --gres=gpu:4 \\")
    print("          slurm_template.sh")

    print("\n3. Multi-Node with Multiple DataLoaders:")
    print("   sbatch --job-name=multi_loader \\")
    print("          --nodes=2 --ntasks-per-node=4 --gres=gpu:4 \\")
    print("          --export=num_loaders=3 \\")
    print("          slurm_template.sh")

    print("\n" + "=" * 60)
    print("🔑 Key Features Demonstrated")
    print("=" * 60)

    features = [
        "✓ Multi-loader architecture in distributed setting",
        "✓ Single loader wrapped in list (multi-loader API)",
        "✓ Multiple dataloaders with different sampling strategies",
        "✓ DDP configuration for multi-node training",
        "✓ Fault-tolerant checkpointing with exact resume",
        "✓ Per-loader metrics tracking in distributed mode",
        "✓ SLURM integration with proper environment setup",
        "✓ Lightning Fabric for simplified distributed training",
    ]

    for feature in features:
        print(f"   {feature}")

    print("\n" + "=" * 60)
    print("⚠️  Important Notes")
    print("=" * 60)

    notes = [
        "• Always use lists for dataloaders: train_loaders=[loader]",
        "• Always use lists for optimizers: optimizers=[optimizer]",
        "• Training step signature: (trainer, batch, dataloader_idx, dataloader_name)",
        "• DDP synchronization is automatic with sync_schedules_across_ranks=True",
        "• Checkpoints include sampler state for exact resume",
    ]

    for note in notes:
        print(f"   {note}")

    print("\n✅ Distributed training setup complete!")
    print("   Ready for SLURM submission with multi-loader architecture")


if __name__ == "__main__":
    main()
