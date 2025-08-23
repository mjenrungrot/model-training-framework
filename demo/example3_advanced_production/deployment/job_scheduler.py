"""
Production Job Scheduler for Enterprise ML Training

This module provides enterprise-grade job scheduling capabilities for production
ML training workflows, including dependency management, resource optimization,
and comprehensive monitoring.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import subprocess  # nosec B404
import time
from typing import Any


class JobPriority(Enum):
    """Job priority levels for scheduling."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class JobStatus(Enum):
    """Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ResourceRequirements:
    """Resource requirements for a job."""

    cpus: int = 8
    memory_gb: int = 32
    gpus: int = 1
    gpu_type: str | None = None  # "v100", "a100", etc.
    time_limit: str = "24:00:00"
    partition: str = "gpu"
    exclusive: bool = False


@dataclass
class JobConfig:
    """Configuration for a training job."""

    name: str
    config_path: Path
    experiment_name: str
    priority: JobPriority = JobPriority.NORMAL
    resources: ResourceRequirements = None
    dependencies: list[str] = None  # Job IDs this job depends on
    environment: dict[str, str] = None
    notifications: dict[str, Any] = None
    retry_config: dict[str, Any] = None

    def __post_init__(self):
        if self.resources is None:
            self.resources = ResourceRequirements()
        if self.dependencies is None:
            self.dependencies = []
        if self.environment is None:
            self.environment = {}
        if self.notifications is None:
            self.notifications = {}
        if self.retry_config is None:
            self.retry_config = {"max_retries": 3, "retry_delay": 300}


class ProductionJobScheduler:
    """
    Enterprise-grade job scheduler for ML training workflows.

    This scheduler provides advanced features for production environments
    including dependency management, resource optimization, and monitoring.
    """

    def __init__(
        self,
        slurm_template_path: Path,
        output_dir: Path,
        max_concurrent_jobs: int = 10,
        resource_limits: dict | None = None,
    ):
        """
        Initialize production job scheduler.

        Args:
            slurm_template_path: Path to SLURM job template
            output_dir: Directory for job outputs
            max_concurrent_jobs: Maximum concurrent jobs
            resource_limits: Global resource limits
        """
        self.slurm_template_path = Path(slurm_template_path)
        self.output_dir = Path(output_dir)
        self.max_concurrent_jobs = max_concurrent_jobs
        self.resource_limits = resource_limits or {
            "max_total_cpus": 1000,
            "max_total_memory_gb": 8000,
            "max_total_gpus": 100,
        }

        # Job tracking
        self.submitted_jobs: dict[str, JobConfig] = {}
        self.job_status_cache: dict[str, dict] = {}
        self.job_dependencies: dict[str, list[str]] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Job queue management
        self.pending_queue: list[JobConfig] = []
        self.running_jobs: dict[str, JobConfig] = {}

    def setup_logging(self) -> None:
        """Setup comprehensive logging."""
        log_dir = self.output_dir / "scheduler_logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger("ProductionJobScheduler")
        self.logger.info("Production job scheduler initialized")

    def submit_job(self, job_config: JobConfig) -> str | None:
        """
        Submit a job to the scheduler.

        Args:
            job_config: Job configuration

        Returns:
            Job ID if submission successful, None otherwise
        """
        # Validate job configuration
        if not self._validate_job_config(job_config):
            self.logger.error(f"Invalid job configuration: {job_config.name}")
            return None

        # Check resource availability
        if not self._check_resource_availability(job_config.resources):
            self.logger.warning(f"Insufficient resources for job: {job_config.name}")
            # Add to pending queue
            self.pending_queue.append(job_config)
            self.logger.info(f"Job {job_config.name} added to pending queue")
            return None

        # Check dependencies
        if not self._check_dependencies(job_config):
            self.logger.info(f"Dependencies not met for job: {job_config.name}")
            self.pending_queue.append(job_config)
            return None

        # Submit to SLURM
        job_id = self._submit_to_slurm(job_config)

        if job_id:
            self.submitted_jobs[job_id] = job_config
            self.running_jobs[job_id] = job_config
            self.logger.info(
                f"Successfully submitted job {job_config.name} with ID {job_id}"
            )

            # Save job metadata
            self._save_job_metadata(job_id, job_config)

        return job_id

    def submit_job_batch(self, job_configs: list[JobConfig]) -> dict[str, str | None]:
        """
        Submit multiple jobs with dependency management.

        Args:
            job_configs: List of job configurations

        Returns:
            Dictionary mapping job names to job IDs
        """
        results = {}

        # Sort jobs by dependencies (topological sort)
        sorted_jobs = self._topological_sort(job_configs)

        for job_config in sorted_jobs:
            job_id = self.submit_job(job_config)
            results[job_config.name] = job_id

            # Update dependencies with actual job IDs
            self._update_job_dependencies(job_config, job_id, results)

        return results

    def _validate_job_config(self, job_config: JobConfig) -> bool:
        """Validate job configuration."""
        # Check required fields
        if not job_config.name or not job_config.config_path:
            return False

        # Check config file exists
        if not job_config.config_path.exists():
            self.logger.error(f"Config file not found: {job_config.config_path}")
            return False

        # Validate resources
        resources = job_config.resources
        return not (resources.cpus <= 0 or resources.memory_gb <= 0)

    def _check_resource_availability(self, resources: ResourceRequirements) -> bool:
        """Check if resources are available."""
        # Get current resource usage
        current_usage = self._get_current_resource_usage()

        # Check against limits
        if (
            current_usage["cpus"] + resources.cpus
            > self.resource_limits["max_total_cpus"]
            or current_usage["memory_gb"] + resources.memory_gb
            > self.resource_limits["max_total_memory_gb"]
            or current_usage["gpus"] + resources.gpus
            > self.resource_limits["max_total_gpus"]
        ):
            return False

        # Check concurrent job limit
        return not len(self.running_jobs) >= self.max_concurrent_jobs

    def _check_dependencies(self, job_config: JobConfig) -> bool:
        """Check if job dependencies are satisfied."""
        if not job_config.dependencies:
            return True

        for dep_job_id in job_config.dependencies:
            if dep_job_id in self.submitted_jobs:
                status = self.get_job_status(dep_job_id)
                if status.get("state") != "COMPLETED":
                    return False
            else:
                # Dependency job not found
                return False

        return True

    def _submit_to_slurm(self, job_config: JobConfig) -> str | None:
        """Submit job to SLURM."""
        try:
            # Generate SLURM script
            script_content = self._generate_slurm_script(job_config)

            # Write script to file
            script_file = self.output_dir / f"{job_config.name}_slurm.sh"
            with script_file.open("w") as f:
                f.write(script_content)

            # Make script executable
            script_file.chmod(0o755)

            # Submit to SLURM
            cmd = ["sbatch", str(script_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603

            # Extract job ID from output
            output = result.stdout.strip()
            return output.split()[-1]  # Last word is typically the job ID

        except subprocess.CalledProcessError:
            self.logger.exception("SLURM submission failed")
            return None
        except Exception:
            self.logger.exception("Error submitting job")
            return None

    def _generate_slurm_script(self, job_config: JobConfig) -> str:
        """Generate SLURM script from template."""
        # Read template
        with self.slurm_template_path.open() as f:
            template = f.read()

        # Prepare template variables
        variables = {
            "job_name": job_config.name,
            "experiment_name": job_config.experiment_name,
            "config_path": str(job_config.config_path.absolute()),
            "output_dir": str(self.output_dir.absolute()),
            "project_root": str(Path.cwd().absolute()),
            # Resource variables
            "time_limit": job_config.resources.time_limit,
            "nodes": 1,  # Single node for now
            "ntasks_per_node": job_config.resources.gpus,
            "cpus_per_task": job_config.resources.cpus // job_config.resources.gpus,
            "mem": f"{job_config.resources.memory_gb}G",
            "gres": f"gpu:{job_config.resources.gpu_type or ''}:{job_config.resources.gpus}".rstrip(
                ":"
            ),
            "partition": job_config.resources.partition,
        }

        # Add environment variables
        env_vars = "\n".join(
            [f"export {k}={v}" for k, v in job_config.environment.items()]
        )
        variables["environment_vars"] = env_vars

        # Add exclusive flag if needed
        if job_config.resources.exclusive:
            variables["exclusive"] = "#SBATCH --exclusive"
        else:
            variables["exclusive"] = ""

        # Simple template substitution (in production, use proper templating)
        script = template
        for key, value in variables.items():
            script = script.replace(f"{{{{{key}}}}}", str(value))

        return script

    def _get_current_resource_usage(self) -> dict[str, int]:
        """Get current resource usage from running jobs."""
        usage = {"cpus": 0, "memory_gb": 0, "gpus": 0}

        for job_config in self.running_jobs.values():
            usage["cpus"] += job_config.resources.cpus
            usage["memory_gb"] += job_config.resources.memory_gb
            usage["gpus"] += job_config.resources.gpus

        return usage

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a specific job."""
        try:
            # Use cached status if recent
            if (
                job_id in self.job_status_cache
                and time.time() - self.job_status_cache[job_id].get("timestamp", 0) < 30
            ):
                return self.job_status_cache[job_id]

            # Query SLURM
            cmd = ["scontrol", "show", "job", job_id]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603

            # Parse output (simplified)
            status = {"job_id": job_id, "state": "UNKNOWN", "timestamp": time.time()}

            for line in result.stdout.split("\n"):
                if "JobState=" in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith("JobState="):
                            status["state"] = part.split("=")[1]
                            break

            # Cache status
            self.job_status_cache[job_id] = status

            return status

        except subprocess.CalledProcessError:
            # Job not found, might be completed
            return {"job_id": job_id, "state": "COMPLETED", "timestamp": time.time()}
        except Exception:
            self.logger.exception("Error getting job status")
            return {"job_id": job_id, "state": "UNKNOWN", "timestamp": time.time()}

    def monitor_jobs(self) -> dict[str, dict]:
        """Monitor all submitted jobs."""
        status_summary = {}

        for job_id in list(self.running_jobs.keys()):
            status = self.get_job_status(job_id)
            status_summary[job_id] = status

            # Update running jobs list
            if status["state"] in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                if job_id in self.running_jobs:
                    del self.running_jobs[job_id]
                self.logger.info(f"Job {job_id} finished with state: {status['state']}")

        return status_summary

    def process_pending_queue(self) -> list[str]:
        """Process pending jobs queue."""
        submitted_jobs = []
        remaining_queue = []

        for job_config in self.pending_queue:
            # Check if resources are now available and dependencies met
            if self._check_resource_availability(
                job_config.resources
            ) and self._check_dependencies(job_config):
                job_id = self._submit_to_slurm(job_config)
                if job_id:
                    self.submitted_jobs[job_id] = job_config
                    self.running_jobs[job_id] = job_config
                    submitted_jobs.append(job_id)
                    self.logger.info(
                        f"Submitted pending job {job_config.name} with ID {job_id}"
                    )
                else:
                    remaining_queue.append(job_config)
            else:
                remaining_queue.append(job_config)

        self.pending_queue = remaining_queue
        return submitted_jobs

    def _save_job_metadata(self, job_id: str, job_config: JobConfig) -> None:
        """Save job metadata for tracking."""
        metadata_dir = self.output_dir / "job_metadata"
        metadata_dir.mkdir(exist_ok=True)

        metadata = {
            "job_id": job_id,
            "job_config": asdict(job_config),
            "submission_time": datetime.now().isoformat(),
            "status": "submitted",
        }

        metadata_file = metadata_dir / f"{job_id}.json"
        try:
            with metadata_file.open("w") as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception:
            self.logger.exception("Failed to save job metadata")

    def generate_status_report(self) -> str:
        """Generate comprehensive status report."""
        # Monitor current jobs
        status_summary = self.monitor_jobs()

        # Count jobs by status
        status_counts = {}
        for status_info in status_summary.values():
            state = status_info.get("state", "UNKNOWN")
            status_counts[state] = status_counts.get(state, 0) + 1

        # Calculate resource usage
        current_usage = self._get_current_resource_usage()

        return f"""
Production Job Scheduler Status Report
=====================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Job Status Summary:
{json.dumps(status_counts, indent=2)}

Resource Usage:
- CPUs: {current_usage["cpus"]}/{self.resource_limits["max_total_cpus"]}
- Memory: {current_usage["memory_gb"]}GB/{self.resource_limits["max_total_memory_gb"]}GB
- GPUs: {current_usage["gpus"]}/{self.resource_limits["max_total_gpus"]}

Queue Status:
- Running jobs: {len(self.running_jobs)}
- Pending jobs: {len(self.pending_queue)}
- Total submitted: {len(self.submitted_jobs)}

Resource Utilization:
- CPU: {current_usage["cpus"] / self.resource_limits["max_total_cpus"] * 100:.1f}%
- Memory: {current_usage["memory_gb"] / self.resource_limits["max_total_memory_gb"] * 100:.1f}%
- GPU: {current_usage["gpus"] / self.resource_limits["max_total_gpus"] * 100:.1f}%
"""


def main():
    """Main function demonstrating production job scheduler."""
    print("🏭 Production Job Scheduler")
    print("=" * 40)

    # Initialize scheduler
    scheduler = ProductionJobScheduler(
        slurm_template_path=Path("./configs/production_slurm_template.sh"),
        output_dir=Path("./production_jobs"),
        max_concurrent_jobs=5,
    )

    # Example job configurations
    job_configs = [
        JobConfig(
            name="production_training_small",
            config_path=Path("./configs/production_config.yaml"),
            experiment_name="prod_small_model",
            priority=JobPriority.HIGH,
            resources=ResourceRequirements(
                cpus=16, memory_gb=64, gpus=2, time_limit="12:00:00"
            ),
        ),
        JobConfig(
            name="production_training_large",
            config_path=Path("./configs/production_config.yaml"),
            experiment_name="prod_large_model",
            priority=JobPriority.NORMAL,
            resources=ResourceRequirements(
                cpus=32, memory_gb=128, gpus=4, time_limit="24:00:00"
            ),
            dependencies=["production_training_small"],  # Depends on small model
        ),
    ]

    # Submit jobs
    print("📋 Submitting production jobs...")
    results = scheduler.submit_job_batch(job_configs)

    for job_name, job_id in results.items():
        if job_id:
            print(f"✅ {job_name}: {job_id}")
        else:
            print(f"⏳ {job_name}: queued")

    # Generate status report
    print("\n📊 Status Report:")
    report = scheduler.generate_status_report()
    print(report)


if __name__ == "__main__":
    main()
