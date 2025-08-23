"""
Job Monitoring Utilities for HPC Environments

This module provides utilities for monitoring and managing SLURM jobs
in HPC environments, including job status tracking, resource utilization
monitoring, and result analysis.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess  # nosec B404
import time

import yaml


@dataclass
class JobStatus:
    """Data class for job status information."""

    job_id: str
    name: str
    user: str
    state: str
    nodes: str
    runtime: str
    time_limit: str
    submit_time: str
    start_time: str | None = None
    end_time: str | None = None


@dataclass
class JobResources:
    """Data class for job resource utilization."""

    job_id: str
    cpu_efficiency: float
    memory_efficiency: float
    gpu_utilization: float
    max_memory_used: str
    allocated_memory: str


class SLURMJobMonitor:
    """
    Monitor SLURM jobs and provide detailed status and resource information.

    This class provides comprehensive job monitoring capabilities for HPC
    environments, including real-time status updates and resource utilization.
    """

    def __init__(self, user: str | None = None):
        """
        Initialize job monitor.

        Args:
            user: Username to monitor (defaults to current user)
        """
        self.user = user or self._get_current_user()

    def _get_current_user(self) -> str:
        """Get current username."""
        result = subprocess.run(["whoami"], check=False, capture_output=True, text=True)  # nosec B603 B607
        return result.stdout.strip()

    def get_job_status(self, job_id: str | None = None) -> list[JobStatus]:
        """
        Get status of jobs.

        Args:
            job_id: Specific job ID to check (optional)

        Returns:
            List of JobStatus objects
        """
        cmd = ["squeue", "-u", self.user, "--format=%i,%j,%u,%T,%N,%M,%l,%V,%S"]

        if job_id:
            cmd.extend(["-j", job_id])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603
            lines = result.stdout.strip().split("\n")[1:]  # Skip header

            jobs = []
            for line in lines:
                if line.strip():
                    parts = line.split(",")
                    if len(parts) >= 8:
                        job = JobStatus(
                            job_id=parts[0],
                            name=parts[1],
                            user=parts[2],
                            state=parts[3],
                            nodes=parts[4],
                            runtime=parts[5],
                            time_limit=parts[6],
                            submit_time=parts[7],
                            start_time=parts[8]
                            if len(parts) > 8 and parts[8] != "N/A"
                            else None,
                        )
                        jobs.append(job)

            return jobs

        except subprocess.CalledProcessError as e:
            print(f"Error getting job status: {e}")
            return []

    def get_job_efficiency(self, job_id: str) -> JobResources | None:
        """
        Get job resource efficiency using seff.

        Args:
            job_id: Job ID to analyze

        Returns:
            JobResources object or None if not available
        """
        try:
            result = subprocess.run(
                ["seff", job_id], capture_output=True, text=True, check=True
            )  # nosec B603 B607
            output = result.stdout

            # Parse seff output
            cpu_eff = self._extract_efficiency(output, r"CPU Efficiency:\s+(\d+\.\d+)%")
            mem_eff = self._extract_efficiency(
                output, r"Memory Efficiency:\s+(\d+\.\d+)%"
            )
            max_mem = self._extract_value(output, r"Memory Utilized:\s+(.+)")
            alloc_mem = self._extract_value(output, r"Memory Efficiency:.+of\s+(.+)\)")

            return JobResources(
                job_id=job_id,
                cpu_efficiency=cpu_eff,
                memory_efficiency=mem_eff,
                gpu_utilization=0.0,  # Not available from seff
                max_memory_used=max_mem or "N/A",
                allocated_memory=alloc_mem or "N/A",
            )

        except subprocess.CalledProcessError:
            return None

    def _extract_efficiency(self, text: str, pattern: str) -> float:
        """Extract efficiency percentage from text."""
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 0.0

    def _extract_value(self, text: str, pattern: str) -> str | None:
        """Extract value from text using regex."""
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None

    def get_job_details(self, job_id: str) -> dict:
        """
        Get detailed job information using scontrol.

        Args:
            job_id: Job ID to query

        Returns:
            Dictionary with detailed job information
        """
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True,
                check=True,
            )  # nosec B603 B607

            # Parse scontrol output
            details = {}
            for line in result.stdout.split("\n"):
                if "=" in line:
                    parts = line.split("=")
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        value = "=".join(parts[1:]).strip()
                        details[key] = value

            return details

        except subprocess.CalledProcessError:
            return {}

    def monitor_jobs_realtime(self, refresh_interval: int = 30) -> None:
        """
        Monitor jobs in real-time with periodic updates.

        Args:
            refresh_interval: Seconds between updates
        """
        try:
            while True:
                print(f"\n{'=' * 80}")
                print(
                    f"Job Status Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                print(f"{'=' * 80}")

                jobs = self.get_job_status()

                if jobs:
                    self._print_job_table(jobs)
                else:
                    print("No active jobs found.")

                print(f"\nNext update in {refresh_interval} seconds...")
                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

    def _print_job_table(self, jobs: list[JobStatus]) -> None:
        """Print jobs in a formatted table."""
        header = (
            f"{'Job ID':<10} {'Name':<25} {'State':<12} {'Runtime':<10} {'Nodes':<8}"
        )
        print(header)
        print("-" * len(header))

        for job in jobs:
            print(
                f"{job.job_id:<10} {job.name[:24]:<25} {job.state:<12} "
                f"{job.runtime:<10} {job.nodes:<8}"
            )


class ExperimentTracker:
    """
    Track and analyze experiment results from multiple jobs.

    This class helps manage and analyze results from hyperparameter
    optimization and distributed training experiments.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize experiment tracker.

        Args:
            output_dir: Directory containing experiment outputs
        """
        self.output_dir = Path(output_dir)
        self.experiments = {}

    def scan_experiments(self) -> dict:
        """
        Scan output directory for completed experiments.

        Returns:
            Dictionary mapping experiment names to their results
        """
        experiments = {}

        for exp_dir in self.output_dir.iterdir():
            if exp_dir.is_dir():
                result = self._analyze_experiment(exp_dir)
                if result:
                    experiments[exp_dir.name] = result

        self.experiments = experiments
        return experiments

    def _analyze_experiment(self, exp_dir: Path) -> dict | None:
        """
        Analyze a single experiment directory.

        Args:
            exp_dir: Path to experiment directory

        Returns:
            Dictionary with experiment analysis or None
        """
        # Look for common result files
        metrics_file = exp_dir / "metrics.json"
        config_file = exp_dir / "config.yaml"
        log_file = exp_dir / "training.log"

        result = {
            "experiment_name": exp_dir.name,
            "status": "unknown",
            "metrics": {},
            "config": {},
            "duration": None,
        }

        # Check if experiment completed successfully
        if (exp_dir / "training_complete.flag").exists():
            result["status"] = "completed"
        elif (exp_dir / "training_failed.flag").exists():
            result["status"] = "failed"
        elif log_file.exists():
            # Check log file for completion indicators
            with log_file.open() as f:
                log_content = f.read()
                if "Training completed successfully" in log_content:
                    result["status"] = "completed"
                elif "Training failed" in log_content:
                    result["status"] = "failed"
                else:
                    result["status"] = "running"

        # Load metrics if available
        if metrics_file.exists():
            try:
                with metrics_file.open() as f:
                    result["metrics"] = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        # Load configuration if available
        if config_file.exists():
            try:
                with config_file.open() as f:
                    result["config"] = yaml.safe_load(f)
            except (OSError, yaml.YAMLError):
                pass

        return result

    def get_best_experiments(self, metric: str = "val_loss", n: int = 5) -> list[dict]:
        """
        Get best performing experiments based on a metric.

        Args:
            metric: Metric to rank by
            n: Number of top experiments to return

        Returns:
            List of best experiment results
        """
        completed_experiments = [
            exp
            for exp in self.experiments.values()
            if exp["status"] == "completed" and metric in exp.get("metrics", {})
        ]

        # Sort by metric (assuming lower is better for most metrics)
        if metric.startswith("val_") or metric in ["loss", "error"]:
            # Lower is better
            sorted_experiments = sorted(
                completed_experiments, key=lambda x: x["metrics"][metric]
            )
        else:
            # Higher is better
            sorted_experiments = sorted(
                completed_experiments, key=lambda x: x["metrics"][metric], reverse=True
            )

        return sorted_experiments[:n]

    def generate_summary_report(self, output_file: Path | None = None) -> str:
        """
        Generate a summary report of all experiments.

        Args:
            output_file: Optional file to save the report

        Returns:
            Report content as string
        """
        if not self.experiments:
            self.scan_experiments()

        total_experiments = len(self.experiments)
        completed = sum(
            1 for exp in self.experiments.values() if exp["status"] == "completed"
        )
        failed = sum(
            1 for exp in self.experiments.values() if exp["status"] == "failed"
        )
        running = sum(
            1 for exp in self.experiments.values() if exp["status"] == "running"
        )

        report = f"""
Experiment Summary Report
========================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Overview:
- Total experiments: {total_experiments}
- Completed: {completed}
- Failed: {failed}
- Running: {running}
- Success rate: {completed / total_experiments * 100:.1f}%

"""

        if completed > 0:
            # Add best experiments section
            best_experiments = self.get_best_experiments()
            if best_experiments:
                report += "Best Performing Experiments:\n"
                report += "-" * 30 + "\n"
                for i, exp in enumerate(best_experiments, 1):
                    metrics = exp.get("metrics", {})
                    val_loss = metrics.get("val_loss", "N/A")
                    report += f"{i}. {exp['experiment_name']}\n"
                    report += f"   Validation Loss: {val_loss}\n\n"

        if output_file:
            with output_file.open("w") as f:
                f.write(report)

        return report


def main():
    """
    Main function demonstrating job monitoring utilities.
    """
    print("🔍 SLURM Job Monitoring Utilities")
    print("=" * 40)

    # Initialize job monitor
    monitor = SLURMJobMonitor()

    # Get current job status
    print("Current job status:")
    jobs = monitor.get_job_status()

    if jobs:
        monitor._print_job_table(jobs)

        # Show efficiency for first job
        if jobs:
            first_job = jobs[0]
            print(f"\nDetailed analysis for job {first_job.job_id}:")

            efficiency = monitor.get_job_efficiency(first_job.job_id)
            if efficiency:
                print(f"CPU Efficiency: {efficiency.cpu_efficiency:.1f}%")
                print(f"Memory Efficiency: {efficiency.memory_efficiency:.1f}%")
                print(f"Memory Used: {efficiency.max_memory_used}")

            details = monitor.get_job_details(first_job.job_id)
            if details:
                print(f"Job State: {details.get('JobState', 'N/A')}")
                print(f"Reason: {details.get('Reason', 'N/A')}")
    else:
        print("No active jobs found.")

    # Demonstrate experiment tracking
    print("\n📊 Experiment Tracking Example:")
    output_dir = Path("./hpc_experiments")

    if output_dir.exists():
        tracker = ExperimentTracker(output_dir)
        experiments = tracker.scan_experiments()

        print(f"Found {len(experiments)} experiments")

        # Generate summary report
        report = tracker.generate_summary_report()
        print(report)
    else:
        print("No experiment directory found. Run some experiments first!")

    print("\n💡 Usage Tips:")
    print("- Use 'python job_monitoring.py' to check job status")
    print("- Use 'squeue -u $USER' for quick status check")
    print("- Use 'scancel <job_id>' to cancel jobs")
    print("- Monitor GPU usage with 'nvidia-smi' on compute nodes")


if __name__ == "__main__":
    main()
