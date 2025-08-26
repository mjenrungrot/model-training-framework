"""
Example 3 data: small synthetic datasets + loaders.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def create_loaders(
    batch_size: int, world_size: int = 1, rank: int = 0, num_loaders: int = 1
) -> tuple[list[DataLoader], list[DataLoader]]:
    train_loaders: list[DataLoader] = []
    val_loaders: list[DataLoader] = []

    seq_len = 64
    for i in range(num_loaders):
        torch.manual_seed(100 + i)
        # Larger sizes to target ~2-3 minutes on CPU with periodic pre-emption
        train_size = 512 * (i + 1)
        val_size = 128 * (i + 1)

        x_train = torch.randn(train_size, seq_len)
        y_train = (x_train.mean(dim=1) > 0).long()
        x_val = torch.randn(val_size, seq_len)
        y_val = (x_val.mean(dim=1) > 0).long()

        dtrain = TensorDataset(x_train, y_train)
        dval = TensorDataset(x_val, y_val)

        train_sampler: DistributedSampler = DistributedSampler(
            dtrain, num_replicas=world_size, rank=rank, shuffle=True, seed=42
        )
        val_sampler: DistributedSampler = DistributedSampler(
            dval, num_replicas=world_size, rank=rank, shuffle=False
        )

        train_loader = DataLoader(
            dtrain,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            dval,
            batch_size=max(1, batch_size * 2),
            sampler=val_sampler,
            num_workers=0,
            pin_memory=False,
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders
