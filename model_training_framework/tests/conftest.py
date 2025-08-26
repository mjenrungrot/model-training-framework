"""
Test configuration and fixtures for the Model Training Framework test suite.

This module provides common fixtures and test utilities used across all test modules.
"""

from collections.abc import Generator
from pathlib import Path
import subprocess
import tempfile
from typing import Any, cast

import pytest


@pytest.fixture(scope="session")
def test_project_root() -> Generator[Path, None, None]:
    """Create a temporary project root directory for testing."""
    with tempfile.TemporaryDirectory(prefix="mtf_test_") as temp_dir:
        project_root = Path(temp_dir)

        # Create basic project structure
        (project_root / "configs").mkdir()
        (project_root / "experiments").mkdir()
        (project_root / "scripts").mkdir()
        (project_root / ".git").mkdir()

        # Create a basic SLURM template
        slurm_template = project_root / "slurm_template.txt"
        slurm_template.write_text("""#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodes={{NODES}}
#SBATCH --gpus-per-node={{GPUS_PER_NODE}}
#SBATCH --time={{TIME}}
#SBATCH --output={{OUTPUT_FILE}}

echo "Running {{SCRIPT_PATH}} {{CONFIG_NAME}}"
""")

        yield project_root


@pytest.fixture
def sample_config_dict():
    """Sample experiment configuration dictionary for testing."""
    return {
        "experiment_name": "test_experiment",
        "model": {
            "type": "resnet18",
            "hidden_size": 512,
            "num_layers": 6,
            "dropout": 0.1,
        },
        "training": {
            "max_epochs": 10,
            "gradient_accumulation_steps": 1,
        },
        "data": {"dataset_name": "test_dataset", "batch_size": 32, "num_workers": 2},
        "optimizer": {"type": "adamw", "lr": 1e-4, "weight_decay": 0.01},
        "logging": {"use_wandb": False, "log_scalars_every_n_steps": 10},
    }


@pytest.fixture
def sample_slurm_config():
    """Sample SLURM configuration for testing."""
    return {
        "account": "test_account",
        "partition": "test_partition",
        "nodes": 1,
        "gpus_per_node": 1,
        "time": "01:00:00",
        "mem": "16G",
    }


@pytest.fixture
def mock_git_repo(test_project_root: Path):
    """Create a mock git repository structure."""
    git_dir = test_project_root / ".git"

    # Create basic git structure
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "refs" / "heads" / "main").write_text("abc123def456\n")
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

    return git_dir


class MockFabric:
    """Mock Lightning Fabric for testing with DDP support."""

    def __init__(self, rank: int = 0, world_size: int = 1):
        """Initialize mock fabric with rank and world size."""
        self.global_rank = rank
        self.rank = rank
        self.world_size = world_size
        self.is_global_zero = rank == 0

    def setup(self, *args):
        """Mock setup that returns args as-is or as tuple."""
        if len(args) == 1:
            return args[0]
        return args

    def setup_dataloaders(self, dataloader):
        """Mock dataloader setup."""
        return dataloader

    def backward(self, loss):
        """Mock backward pass."""
        if hasattr(loss, "backward"):
            loss.backward()

    def barrier(self):
        """Mock barrier for synchronization."""

    def broadcast(self, obj, src: int = 0):
        """Mock broadcast that returns object unchanged."""
        return obj

    def all_gather(self, tensor):
        """Mock all_gather that returns list with single tensor."""
        return [tensor] * self.world_size

    def all_reduce(self, tensor, op: str = "mean"):
        """Mock all_reduce that returns tensor unchanged."""
        return tensor


class MockModel:
    """Mock PyTorch model for testing."""

    def __init__(self):
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def parameters(self):
        return []

    def state_dict(self):
        return {"layer1.weight": "mock_weight"}

    def load_state_dict(self, state_dict, strict=True):
        pass


class MockOptimizer:
    """Mock optimizer for testing."""

    def step(self):
        pass

    def zero_grad(self):
        pass

    def state_dict(self):
        return {"param_groups": []}

    def load_state_dict(self, state_dict):
        pass


@pytest.fixture
def mock_fabric():
    """Provide mock Fabric instance."""
    return MockFabric()


@pytest.fixture
def mock_model():
    """Provide mock model instance."""
    return MockModel()


@pytest.fixture
def mock_optimizer():
    """Provide mock optimizer instance."""
    return MockOptimizer()


# Test markers are declared in pytest.ini; no reassignment needed here.


# Skip tests if external dependencies are not available
def skip_if_no_slurm():
    """Skip test if SLURM is not available."""
    try:
        subprocess.run(["sbatch", "--version"], capture_output=True, check=True)
        return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


# Provide a custom marker alias with a typing-safe cast for static checkers
_mark = cast(Any, pytest.mark)
_mark.skipif_no_slurm = pytest.mark.skipif(
    skip_if_no_slurm(), reason="SLURM not available"
)
