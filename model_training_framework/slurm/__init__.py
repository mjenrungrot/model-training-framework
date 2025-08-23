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
from .utils import (
    get_slurm_job_info,
    parse_slurm_output,
    validate_slurm_params,
)

__all__ = [
    # Launcher
    "SLURMLauncher",
    "BatchSubmissionResult",
    "SLURMJobResult",
    # Git operations
    "GitOperationLock",
    "GitManager",
    "BranchManager",
    # Templates
    "SBATCHTemplateEngine",
    "TemplateContext",
    # Monitoring
    "SLURMJobMonitor",
    "JobStatus",
    "JobInfo",
    # Utilities
    "validate_slurm_params",
    "get_slurm_job_info",
    "parse_slurm_output",
]
