"""
SLURM Job Launcher

This module provides the main SLURM integration functionality:
- Job submission and batch management
- Experiment configuration handling
- Git integration for reproducibility
- Job monitoring and status tracking
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Any

from ..config.manager import ConfigurationManager
from .git_ops import BranchManager, GitManager
from .templates import (
    SBATCHTemplateEngine,
    create_template_context_from_config,
)

if TYPE_CHECKING:
    from ..config.schemas import ExperimentConfig

logger = logging.getLogger(__name__)


class SLURMError(Exception):
    """Exception raised for SLURM-related errors."""


@dataclass
class SLURMJobResult:
    """Result of a single SLURM job submission."""

    experiment_name: str
    job_id: str | None = None
    success: bool = False
    error_message: str | None = None
    sbatch_script_path: Path | None = None
    submission_time: float = field(default_factory=time.time)


@dataclass
class BatchSubmissionResult:
    """Result of batch experiment submission."""

    total_experiments: int
    successful_jobs: list[str] = field(default_factory=list)
    failed_jobs: dict[str, str] = field(
        default_factory=dict
    )  # experiment_name -> error_message
    job_results: list[SLURMJobResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        """Number of successfully submitted jobs."""
        return len(self.successful_jobs)

    @property
    def failure_count(self) -> int:
        """Number of failed job submissions."""
        return len(self.failed_jobs)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_experiments == 0:
            return 100.0
        return (self.success_count / self.total_experiments) * 100.0


class SLURMLauncher:
    """Manages SLURM job submission and lifecycle."""

    def __init__(
        self,
        template_path: Path,
        project_root: Path,
        experiments_dir: Path | None = None,
    ):
        """
        Initialize SLURM launcher.

        Args:
            template_path: Path to SBATCH template file
            project_root: Path to project root directory
            experiments_dir: Directory for experiment outputs (default: project_root/experiments)
        """
        self.template_path = Path(template_path)
        self.project_root = Path(project_root)
        self.experiments_dir = experiments_dir or self.project_root / "experiments"

        # Initialize components
        self.template_engine = SBATCHTemplateEngine(self.template_path)
        self.git_manager = GitManager(self.project_root)
        self.branch_manager = BranchManager(self.git_manager)

        # Ensure directories exist
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized SLURMLauncher with template: {self.template_path}")

    def submit_single_experiment(
        self,
        config: ExperimentConfig,
        script_path: str | Path,
        use_git_branch: bool = False,
        dry_run: bool = False,
    ) -> SLURMJobResult:
        """
        Submit a single experiment as SLURM job.

        Args:
            config: Experiment configuration
            script_path: Path to training script
            use_git_branch: Whether to create temporary git branch
            dry_run: If True, generate scripts but don't submit

        Returns:
            SLURMJobResult with submission details
        """
        result = SLURMJobResult(experiment_name=config.experiment_name)

        try:
            # Setup experiment directory
            exp_dir = self.experiments_dir / config.experiment_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Get git information
            git_info = None
            if use_git_branch:
                git_info = self._setup_git_branch(config.experiment_name)
            else:
                git_info = {
                    "commit_hash": self.git_manager.get_current_commit(),
                    "branch_name": self.git_manager.get_current_branch(),
                }

            # Generate SBATCH script
            sbatch_path = self._generate_sbatch_script(
                config, script_path, exp_dir, git_info
            )
            result.sbatch_script_path = sbatch_path

            # Save configuration for reproducibility (both YAML and JSON)
            try:
                cm = ConfigurationManager(project_root=self.project_root)
                cm.save_config(config, exp_dir / "config.yaml", format="yaml")
                cm.save_config(config, exp_dir / "config.json", format="json")
            except Exception:
                logger.exception(
                    "Failed to serialize experiment config for %s",
                    config.experiment_name,
                )

            if dry_run:
                logger.info(f"Dry run: Would submit job for {config.experiment_name}")
                result.success = True
                result.job_id = "DRY_RUN"
                return result

            # Submit job
            job_id = self._submit_sbatch_job(sbatch_path)
            result.job_id = job_id
            result.success = True

            logger.info(
                f"Submitted job {job_id} for experiment: {config.experiment_name}"
            )

        except Exception as e:
            result.error_message = str(e)
            logger.exception(f"Failed to submit experiment {config.experiment_name}: ")

        return result

    def submit_experiment_batch(
        self,
        experiments: list[ExperimentConfig],
        script_path: str | Path,
        use_git_branch: bool = False,
        dry_run: bool = False,
        max_concurrent: int | None = None,
    ) -> BatchSubmissionResult:
        """
        Submit multiple experiments as SLURM jobs.

        Args:
            experiments: List of experiment configurations
            script_path: Path to training script
            use_git_branch: Whether to create temporary git branches
            dry_run: If True, generate scripts but don't submit
            max_concurrent: Maximum number of concurrent submissions

        Returns:
            BatchSubmissionResult with overall submission status
        """
        result = BatchSubmissionResult(total_experiments=len(experiments))

        logger.info(f"Submitting batch of {len(experiments)} experiments")

        for i, config in enumerate(experiments):
            # Respect max concurrent limit (simple implementation)
            if max_concurrent and i >= max_concurrent:
                logger.warning(
                    f"Reached max concurrent limit ({max_concurrent}), skipping remaining experiments"
                )
                break

            job_result = self.submit_single_experiment(
                config, script_path, use_git_branch, dry_run
            )

            result.job_results.append(job_result)

            if job_result.success:
                result.successful_jobs.append(job_result.job_id or "unknown")
            else:
                result.failed_jobs[config.experiment_name] = (
                    job_result.error_message or "Unknown error"
                )

        logger.info(
            f"Batch submission complete: {result.success_count}/{result.total_experiments} successful"
        )

        return result

    def _setup_git_branch(self, experiment_name: str) -> dict[str, str]:
        """
        Setup git branch for experiment isolation.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with git information
        """
        try:
            current_commit = self.git_manager.get_current_commit()
            current_branch = self.git_manager.get_current_branch()

            # Create temporary branch
            job_branch = self.branch_manager.create_job_branch(
                experiment_name, current_commit, current_branch
            )

            return {
                "commit_hash": current_commit,
                "branch_name": job_branch,
                "original_branch": current_branch,
            }

        except Exception:
            logger.exception("Failed to setup git branch: ")
            # Fall back to current state
            return {
                "commit_hash": self.git_manager.get_current_commit(),
                "branch_name": self.git_manager.get_current_branch(),
            }

    def _generate_sbatch_script(
        self,
        config: ExperimentConfig,
        script_path: str | Path,
        exp_dir: Path,
        git_info: dict[str, str] | None,
    ) -> Path:
        """
        Generate SBATCH script for experiment.

        Args:
            config: Experiment configuration
            script_path: Path to training script
            exp_dir: Experiment directory
            git_info: Git information dictionary

        Returns:
            Path to generated SBATCH script
        """
        # Create template context
        slurm_config = self._extract_slurm_config(config)

        template_context = create_template_context_from_config(
            experiment_name=config.experiment_name,
            script_path=str(script_path),
            config_name=config.experiment_name,  # Use experiment name as config identifier
            slurm_config=slurm_config,
            git_info=git_info,
        )

        # Generate script
        sbatch_path = exp_dir / f"{config.experiment_name}.sbatch"

        self.template_engine.generate_sbatch_script(
            context=template_context, output_path=sbatch_path
        )

        return sbatch_path

    def _extract_slurm_config(self, config: ExperimentConfig) -> dict[str, Any]:
        """
        Extract SLURM configuration from experiment config.

        Args:
            config: Experiment configuration

        Returns:
            Dictionary with SLURM parameters
        """
        if config.slurm is None:
            # Use default SLURM configuration
            return {
                "account": "realitylab",
                "partition": "ckpt-all",
                "nodes": 1,
                "ntasks_per_node": 1,
                "gpus_per_node": 1,
                "cpus_per_task": 8,
                "mem": "256G",
                "time": "1-00:00:00",
                "constraint": "a40|a100",
                "requeue": True,
            }

        # Convert SLURM config to dictionary
        return asdict(config.slurm)

    def _submit_sbatch_job(self, sbatch_path: Path) -> str:
        """
        Submit SBATCH job and return job ID.

        Args:
            sbatch_path: Path to SBATCH script

        Returns:
            SLURM job ID

        Raises:
            SLURMError: If job submission fails
        """
        try:
            cmd = ["sbatch", str(sbatch_path)]

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=self.project_root
            )

            # Parse job ID from output (format: "Submitted batch job 12345")
            output = result.stdout.strip()
            if "Submitted batch job" in output:
                return output.split()[-1]
            raise SLURMError(f"Unexpected sbatch output: {output}")

        except subprocess.CalledProcessError as e:
            error_msg = f"sbatch failed: {e.stderr or e.stdout}"
            raise SLURMError(error_msg) from e
        except Exception as e:
            raise SLURMError(f"Failed to submit job: {e}") from e

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Get status of a SLURM job.

        Args:
            job_id: SLURM job ID

        Returns:
            Dictionary with job status information
        """
        try:
            cmd = ["squeue", "-j", job_id, "--format=%i,%T,%N,%S,%P", "--noheader"]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stdout.strip():
                fields = result.stdout.strip().split(",")
                if len(fields) >= 5:
                    return {
                        "job_id": fields[0],
                        "state": fields[1],
                        "nodes": fields[2],
                        "start_time": fields[3],
                        "partition": fields[4],
                        "status": "running"
                        if fields[1] in ["RUNNING", "PENDING"]
                        else "completed",
                    }

            # Job not in queue, check if it's completed
            return {"job_id": job_id, "status": "completed", "state": "COMPLETED"}

        except subprocess.CalledProcessError:
            return {
                "job_id": job_id,
                "status": "unknown",
                "error": "Failed to query job status",
            }
        except Exception as e:
            return {"job_id": job_id, "status": "error", "error": str(e)}

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a SLURM job.

        Args:
            job_id: SLURM job ID to cancel

        Returns:
            True if cancellation was successful
        """
        try:
            cmd = ["scancel", job_id]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            logger.info(f"Cancelled job: {job_id}")
            return True

        except subprocess.CalledProcessError:
            logger.exception(f"Failed to cancel job {job_id}: ")
            return False
        except Exception:
            logger.exception(f"Error cancelling job {job_id}: ")
            return False

    def list_user_jobs(self, user: str | None = None) -> list[dict[str, Any]]:
        """
        List SLURM jobs for user.

        Args:
            user: Username (uses current user if None)

        Returns:
            List of job information dictionaries
        """
        try:
            cmd = ["squeue", "--format=%i,%j,%T,%M,%N,%P", "--noheader"]
            if user:
                cmd.extend(["-u", user])

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            jobs = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                fields = line.split(",")
                if len(fields) >= 6:
                    jobs.append(
                        {
                            "job_id": fields[0],
                            "job_name": fields[1],
                            "state": fields[2],
                            "time": fields[3],
                            "nodes": fields[4],
                            "partition": fields[5],
                        }
                    )

            return jobs

        except subprocess.CalledProcessError:
            logger.exception("Failed to list jobs: ")
            return []
        except Exception:
            logger.exception("Error listing jobs: ")
            return []

    def cleanup_old_experiments(
        self, days: int = 30, dry_run: bool = True
    ) -> list[str]:
        """
        Clean up old experiment directories and job branches.

        Args:
            days: Remove experiments older than this many days
            dry_run: If True, only return what would be cleaned up

        Returns:
            List of cleaned up items
        """
        cleaned_items = []

        try:
            # Clean up old job branches
            old_branches = self.branch_manager.cleanup_job_branches(
                older_than_days=days, dry_run=dry_run
            )
            cleaned_items.extend([f"branch:{branch}" for branch in old_branches])

            # Remove experiment directories older than the cutoff
            cutoff = time.time() - (days * 24 * 3600)
            try:
                for child in self.experiments_dir.iterdir():
                    if not child.is_dir():
                        continue

                    # Get newest mtime within the directory tree; fall back to dir mtime
                    newest_mtime = child.stat().st_mtime
                    for p in child.rglob("*"):
                        st_mtime: float | None = None
                        try:
                            st = p.stat()
                            st_mtime = st.st_mtime
                        except FileNotFoundError:
                            logger.debug("Path disappeared during cleanup scan: %s", p)
                            st_mtime = None
                        except (PermissionError, OSError) as e:
                            logger.debug(
                                "Skipping path due to access/OS error: %s (%s)", p, e
                            )
                            st_mtime = None
                        if st_mtime is not None:
                            newest_mtime = max(newest_mtime, st_mtime)

                    if newest_mtime < cutoff:
                        cleaned_items.append(f"dir:{child}")
                        if not dry_run:
                            try:
                                shutil.rmtree(child)
                                logger.info(
                                    "Deleted old experiment directory: %s", child
                                )
                            except Exception as e:
                                logger.warning("Failed to delete %s: %s", child, e)
            except Exception:
                logger.exception(
                    "Failed while scanning experiment directories for cleanup"
                )

            if dry_run:
                logger.info(f"Dry run: Would clean up {len(cleaned_items)} items")
            else:
                logger.info(f"Cleaned up {len(cleaned_items)} old items")

        except Exception:
            logger.exception("Error during cleanup: ")

        return cleaned_items

    def validate_slurm_environment(self) -> dict[str, Any]:
        """
        Validate SLURM environment and return status.

        Returns:
            Dictionary with validation results
        """
        status: dict[str, Any] = {
            "slurm_available": False,
            "commands_available": [],
            "missing_commands": [],
            "template_valid": False,
            "template_issues": [],
        }

        # Check SLURM commands
        slurm_commands = ["sbatch", "squeue", "scancel", "sinfo"]

        commands_available: list[str] = []
        missing_commands: list[str] = []
        for cmd in slurm_commands:
            try:
                subprocess.run([cmd, "--version"], capture_output=True, check=True)
                commands_available.append(cmd)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_commands.append(cmd)

        status["commands_available"] = commands_available
        status["missing_commands"] = missing_commands

        status["slurm_available"] = len(missing_commands) == 0

        # Validate template
        if self.template_path.exists():
            try:
                template_content = self.template_path.read_text()
                issues = self.template_engine.validate_template(template_content)
                status["template_issues"] = issues
                status["template_valid"] = len(issues) == 0
            except Exception as e:
                status["template_issues"] = [f"Failed to read template: {e}"]
        else:
            status["template_issues"] = ["Template file not found"]

        return status
