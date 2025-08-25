#!/usr/bin/env python
"""
Distributed training script for multi-loader architecture.
This script is called by SLURM with proper environment variables set.
"""

import argparse
import os

from lightning.fabric import Fabric
import torch

from demo.example2_intermediate_hpc.distributed_training import (
    DistributedTransformer,
    create_distributed_dataloaders,
    create_distributed_experiment_configs,
    distributed_training_step,
    distributed_validation_step,
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
        val_loaders=val_loaders,  # List of loaders
        max_epochs=args.max_epochs,
    )

    print(f"Training completed on rank {rank}")


if __name__ == "__main__":
    main()
