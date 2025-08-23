"""
SLURM Integration Component

This module provides SLURM job management and integration:
- Job launcher and batch submission
- SBATCH template management and rendering
- Git integration and branch management
- Job monitoring and status tracking
"""

from .git_ops import (
    BranchManager,
    GitManager,
    GitOperationLock,
)
from .launcher import (
    BatchSubmissionResult,
    SLURMJobResult,
    SLURMLauncher,
)
from .monitor import (
    JobInfo,
    JobStatus,
    SLURMJobMonitor,
)
from .templates import (
    SBATCHTemplateEngine,
    TemplateContext,
)

__all__ = [
    "BatchSubmissionResult",
    "BranchManager",
    "GitManager",
    # Git operations
    "GitOperationLock",
    "JobInfo",
    "JobStatus",
    # Templates
    "SBATCHTemplateEngine",
    # Monitoring
    "SLURMJobMonitor",
    "SLURMJobResult",
    # Launcher
    "SLURMLauncher",
    "TemplateContext",
]
