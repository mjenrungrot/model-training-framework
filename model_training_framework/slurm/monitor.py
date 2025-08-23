"""
SLURM Job Monitoring

This module provides job monitoring and status tracking capabilities:
- Real-time job status monitoring
- Job history and logging
- Performance metrics collection
- Job completion notifications
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """SLURM job status enumeration."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUSPENDED = "SUSPENDED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    PREEMPTED = "PREEMPTED"
    NODE_FAIL = "NODE_FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass
class JobInfo:
    """Information about a SLURM job."""

    job_id: str
    name: str
    status: JobStatus
    partition: str
    nodes: str = ""
    start_time: str | None = None
    end_time: str | None = None
    elapsed_time: str | None = None
    exit_code: str | None = None
    user: str = ""
    account: str = ""

    # Resource usage
    cpu_time: str | None = None
    memory_used: str | None = None
    memory_requested: str | None = None

    # Additional fields
    work_dir: str | None = None
    std_out: str | None = None
    std_err: str | None = None

    last_updated: float = field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        """Check if job is currently active (pending or running)."""
        return self.status in [
            JobStatus.PENDING,
            JobStatus.RUNNING,
            JobStatus.SUSPENDED,
        ]

    @property
    def is_finished(self) -> bool:
        """Check if job has finished (completed, failed, or cancelled)."""
        return self.status in [
            JobStatus.COMPLETED,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
            JobStatus.TIMEOUT,
            JobStatus.PREEMPTED,
            JobStatus.NODE_FAIL,
        ]

    @property
    def was_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED and (
            self.exit_code is None or self.exit_code == "0"
        )


class SLURMJobMonitor:
    """Monitor and track SLURM jobs."""

    def __init__(self, update_interval: float = 60.0):
        """
        Initialize job monitor.

        Args:
            update_interval: How often to update job status (seconds)
        """
        self.update_interval = update_interval
        self.tracked_jobs: dict[str, JobInfo] = {}
        self.job_history: list[JobInfo] = []
        self.last_update = 0.0

        logger.info(
            f"Initialized SLURMJobMonitor with {update_interval}s update interval"
        )

    def track_job(self, job_id: str) -> None:
        """
        Start tracking a job.

        Args:
            job_id: SLURM job ID to track
        """
        if job_id not in self.tracked_jobs:
            job_info = self.get_job_info(job_id)
            if job_info:
                self.tracked_jobs[job_id] = job_info
                logger.info(f"Started tracking job: {job_id}")

    def stop_tracking_job(self, job_id: str) -> None:
        """
        Stop tracking a job.

        Args:
            job_id: SLURM job ID to stop tracking
        """
        if job_id in self.tracked_jobs:
            job_info = self.tracked_jobs.pop(job_id)
            self.job_history.append(job_info)
            logger.info(f"Stopped tracking job: {job_id}")

    def update_job_status(self, job_id: str | None = None) -> None:
        """
        Update status for tracked jobs.

        Args:
            job_id: Specific job ID to update (updates all if None)
        """
        current_time = time.time()

        # Rate limiting
        if current_time - self.last_update < self.update_interval:
            return

        if job_id:
            # Update specific job
            if job_id in self.tracked_jobs:
                updated_info = self.get_job_info(job_id)
                if updated_info:
                    self.tracked_jobs[job_id] = updated_info
        else:
            # Update all tracked jobs
            for tracked_job_id in list(self.tracked_jobs.keys()):
                updated_info = self.get_job_info(tracked_job_id)
                if updated_info:
                    self.tracked_jobs[tracked_job_id] = updated_info

                    # Move finished jobs to history
                    if updated_info.is_finished:
                        self.stop_tracking_job(tracked_job_id)

        self.last_update = current_time

    def get_job_info(self, job_id: str) -> JobInfo | None:
        """
        Get detailed information about a job.

        Args:
            job_id: SLURM job ID

        Returns:
            JobInfo object or None if job not found
        """
        try:
            # First try squeue for active jobs
            job_info = self._get_job_info_from_squeue(job_id)
            if job_info:
                return job_info

            # Fall back to sacct for completed jobs
            return self._get_job_info_from_sacct(job_id)

        except Exception:
            logger.exception(f"Failed to get job info for {job_id}: ")
            return None

    def _get_job_info_from_squeue(self, job_id: str) -> JobInfo | None:
        """Get job info from squeue (for active jobs)."""
        try:
            cmd = [
                "squeue",
                "-j",
                job_id,
                "--format=%i,%j,%T,%P,%N,%S,%u,%a,%M",
                "--noheader",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if not result.stdout.strip():
                return None

            fields = result.stdout.strip().split(",")
            if len(fields) >= 9:
                return JobInfo(
                    job_id=fields[0],
                    name=fields[1],
                    status=self._parse_job_state(fields[2]),
                    partition=fields[3],
                    nodes=fields[4],
                    start_time=fields[5] if fields[5] != "N/A" else None,
                    user=fields[6],
                    account=fields[7],
                    elapsed_time=fields[8],
                )

        except subprocess.CalledProcessError:
            return None

        return None

    def _get_job_info_from_sacct(self, job_id: str) -> JobInfo | None:
        """Get job info from sacct (for completed jobs)."""
        try:
            cmd = [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,JobName,State,Partition,NodeList,Start,End,Elapsed,ExitCode,User,Account,ReqMem,MaxRSS,CPUTime,WorkDir",
                "--parsable2",
                "--noheader",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if not result.stdout.strip():
                return None

            # Parse first line (main job, not job steps)
            for line in result.stdout.strip().split("\n"):
                fields = line.split("|")
                if len(fields) >= 15 and not fields[0].endswith(".batch"):
                    return JobInfo(
                        job_id=fields[0],
                        name=fields[1],
                        status=self._parse_job_state(fields[2]),
                        partition=fields[3],
                        nodes=fields[4],
                        start_time=fields[5] if fields[5] else None,
                        end_time=fields[6] if fields[6] else None,
                        elapsed_time=fields[7],
                        exit_code=fields[8],
                        user=fields[9],
                        account=fields[10],
                        memory_requested=fields[11],
                        memory_used=fields[12],
                        cpu_time=fields[13],
                        work_dir=fields[14] if fields[14] else None,
                    )

        except subprocess.CalledProcessError:
            return None

        return None

    def _parse_job_state(self, state_str: str) -> JobStatus:
        """Parse SLURM job state string to JobStatus enum."""
        state_mapping = {
            "PENDING": JobStatus.PENDING,
            "PD": JobStatus.PENDING,
            "RUNNING": JobStatus.RUNNING,
            "R": JobStatus.RUNNING,
            "SUSPENDED": JobStatus.SUSPENDED,
            "S": JobStatus.SUSPENDED,
            "COMPLETED": JobStatus.COMPLETED,
            "CD": JobStatus.COMPLETED,
            "CANCELLED": JobStatus.CANCELLED,
            "CA": JobStatus.CANCELLED,
            "FAILED": JobStatus.FAILED,
            "F": JobStatus.FAILED,
            "TIMEOUT": JobStatus.TIMEOUT,
            "TO": JobStatus.TIMEOUT,
            "PREEMPTED": JobStatus.PREEMPTED,
            "PR": JobStatus.PREEMPTED,
            "NODE_FAIL": JobStatus.NODE_FAIL,
            "NF": JobStatus.NODE_FAIL,
        }

        return state_mapping.get(state_str.upper(), JobStatus.UNKNOWN)

    def get_tracked_jobs(self) -> dict[str, JobInfo]:
        """Get all currently tracked jobs."""
        return self.tracked_jobs.copy()

    def get_active_jobs(self) -> dict[str, JobInfo]:
        """Get all active (running/pending) tracked jobs."""
        return {
            job_id: job_info
            for job_id, job_info in self.tracked_jobs.items()
            if job_info.is_active
        }

    def get_finished_jobs(self) -> list[JobInfo]:
        """Get all finished jobs from history."""
        return self.job_history.copy()

    def get_job_summary(self) -> dict[str, Any]:
        """Get summary of job monitoring status."""
        active_jobs = self.get_active_jobs()

        # Count jobs by status
        status_counts = {}
        for job_info in active_jobs.values():
            status = job_info.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Count finished jobs by status
        finished_counts = {}
        for job_info in self.job_history:
            status = job_info.status.value
            finished_counts[status] = finished_counts.get(status, 0) + 1

        return {
            "active_jobs": len(active_jobs),
            "finished_jobs": len(self.job_history),
            "total_tracked": len(active_jobs) + len(self.job_history),
            "active_by_status": status_counts,
            "finished_by_status": finished_counts,
            "last_update": self.last_update,
        }

    def wait_for_jobs(
        self,
        job_ids: list[str],
        timeout: float | None = None,
        check_interval: float = 30.0,
    ) -> dict[str, JobInfo]:
        """
        Wait for jobs to complete.

        Args:
            job_ids: List of job IDs to wait for
            timeout: Maximum time to wait (None for no timeout)
            check_interval: How often to check job status

        Returns:
            Dictionary of job_id -> final JobInfo
        """
        start_time = time.time()

        # Start tracking all jobs
        for job_id in job_ids:
            self.track_job(job_id)

        remaining_jobs = set(job_ids)
        completed_jobs = {}

        while remaining_jobs:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for jobs: {remaining_jobs}")
                break

            # Update job status
            for job_id in list(remaining_jobs):
                job_info = self.get_job_info(job_id)
                if job_info and job_info.is_finished:
                    completed_jobs[job_id] = job_info
                    remaining_jobs.remove(job_id)
                    logger.info(
                        f"Job {job_id} finished with status: {job_info.status.value}"
                    )

            if remaining_jobs:
                time.sleep(check_interval)

        return completed_jobs

    def cancel_tracked_jobs(self) -> dict[str, bool]:
        """
        Cancel all currently tracked jobs.

        Returns:
            Dictionary of job_id -> success status
        """
        results = {}

        for job_id in list(self.tracked_jobs.keys()):
            try:
                cmd = ["scancel", job_id]
                subprocess.run(cmd, check=True, capture_output=True)
                results[job_id] = True
                logger.info(f"Cancelled job: {job_id}")
            except subprocess.CalledProcessError:
                results[job_id] = False
                logger.exception(f"Failed to cancel job {job_id}: ")

        return results

    def export_job_history(self, output_path: Path) -> None:
        """
        Export job history to file.

        Args:
            output_path: Path to save job history
        """
        try:
            # Convert job history to serializable format
            history_data = []
            for job_info in self.job_history:
                job_dict = asdict(job_info)
                job_dict["status"] = job_info.status.value  # Convert enum to string
                history_data.append(job_dict)

            # Add current tracked jobs
            for job_info in self.tracked_jobs.values():
                job_dict = asdict(job_info)
                job_dict["status"] = job_info.status.value
                history_data.append(job_dict)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(history_data, f, indent=2, default=str)

            logger.info(f"Exported job history to {output_path}")

        except Exception:
            logger.exception("Failed to export job history: ")


def get_slurm_queue_info() -> dict[str, Any]:
    """
    Get general SLURM queue information.

    Returns:
        Dictionary with queue status
    """
    try:
        # Get partition info
        result = subprocess.run(
            ["sinfo", "--format=%P,%N,%T,%C", "--noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

        partitions = []
        for line in result.stdout.strip().split("\n"):
            if line:
                fields = line.split(",")
                if len(fields) >= 4:
                    partitions.append(
                        {
                            "partition": fields[0],
                            "nodes": fields[1],
                            "state": fields[2],
                            "cpus": fields[3],
                        }
                    )

        return {
            "partitions": partitions,
            "timestamp": time.time(),
        }

    except subprocess.CalledProcessError as e:
        logger.exception("Failed to get queue info: ")
        return {"error": str(e), "timestamp": time.time()}
