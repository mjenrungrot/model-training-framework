"""
Git Operations

This module provides Git integration for SLURM job management:
- Git operation locking for concurrent job submissions
- Branch management for experiment isolation
- Commit tracking and verification
- Safe git operations with error handling
"""

from __future__ import annotations

import fcntl
import logging
from pathlib import Path
import subprocess
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)


class GitOperationError(Exception):
    """Exception raised for Git operation failures."""


class GitOperationLock:
    """
    Context manager for file-based locking to serialize git operations.

    This lock ensures that only one launcher process performs git operations at a time,
    preventing conflicts when multiple jobs are submitted concurrently with
    different target commits. The lock uses fcntl file locking which is
    process-safe and automatically released if the process terminates.

    Example:
        with GitOperationLock(Path(".git/slurm_checkout.lock")):
            # Perform git operations here
            subprocess.run(["git", "checkout", commit_hash])
    """

    def __init__(self, lock_file_path: Path, timeout: float = 60.0) -> None:
        """
        Initialize the lock.

        Args:
            lock_file_path: Path to the lock file
            timeout: Timeout in seconds for acquiring the lock
        """
        self.lock_file_path = lock_file_path
        self.timeout = timeout
        self.lock_file: Any | None = None
        self.acquired = False

    def __enter__(self) -> GitOperationLock:
        """Acquire the lock with timeout."""
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.lock_file = self.lock_file_path.open("w")

            # Try to acquire lock with timeout
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                try:
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.acquired = True
                    logger.debug(f"Acquired git operation lock: {self.lock_file_path}")
                    return self
                except BlockingIOError:
                    time.sleep(0.1)

            raise TimeoutError(f"Failed to acquire git lock within {self.timeout}s")

        except Exception as e:
            if self.lock_file:
                self.lock_file.close()
                self.lock_file = None
            raise GitOperationError(f"Failed to acquire git lock: {e}") from e

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the lock and cleanup."""
        if self.acquired and self.lock_file:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.acquired = False
                logger.debug(f"Released git operation lock: {self.lock_file_path}")
            except Exception as e:
                logger.warning(f"Error releasing git lock: {e}")

        if self.lock_file:
            self.lock_file.close()
            self.lock_file = None


class GitManager:
    """Manages Git operations for the training framework."""

    def __init__(self, project_root: Path):
        """
        Initialize Git manager.

        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)
        self.git_dir = self.project_root / ".git"

        if not self.git_dir.exists():
            raise GitOperationError(f"Not a git repository: {project_root}")

        self.lock_file = self.git_dir / "slurm_checkout.lock"

    def run_git_command(
        self,
        args: list[str],
        check: bool = True,
        capture_output: bool = True,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a git command safely.

        Args:
            args: Git command arguments (without 'git')
            check: Whether to raise exception on non-zero exit
            capture_output: Whether to capture stdout/stderr
            cwd: Working directory for command

        Returns:
            CompletedProcess result

        Raises:
            GitOperationError: If command fails and check=True
        """
        cmd = ["git", *args]
        working_dir = cwd or self.project_root

        try:
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                check=check,
                capture_output=capture_output,
                text=True,
            )

            logger.debug(f"Git command succeeded: {' '.join(cmd)}")
            return result

        except subprocess.CalledProcessError as e:
            error_msg = f"Git command failed: {' '.join(cmd)}"
            if e.stderr:
                error_msg += f"\nError: {e.stderr.strip()}"

            logger.exception(error_msg)
            raise GitOperationError(error_msg) from e

    def get_current_commit(self) -> str:
        """
        Get the current commit hash.

        Returns:
            Current commit hash (full SHA)
        """
        result = self.run_git_command(["rev-parse", "HEAD"])
        return result.stdout.strip()

    def get_current_branch(self) -> str:
        """
        Get the current branch name.

        Returns:
            Current branch name, or "HEAD" if in detached state
        """
        try:
            result = self.run_git_command(["branch", "--show-current"])
            branch = result.stdout.strip()
            return branch if branch else "HEAD"
        except GitOperationError:
            return "HEAD"

    def is_working_tree_clean(self) -> bool:
        """
        Check if working tree is clean (no uncommitted changes).

        Returns:
            True if working tree is clean
        """
        try:
            result = self.run_git_command(["status", "--porcelain"])
            return len(result.stdout.strip()) == 0
        except GitOperationError:
            return False

    def checkout_commit(self, commit_hash: str, with_lock: bool = True) -> None:
        """
        Checkout a specific commit.

        Args:
            commit_hash: Commit hash to checkout
            with_lock: Whether to use git operation lock

        Raises:
            GitOperationError: If checkout fails
        """

        def _do_checkout():
            # Verify commit exists
            try:
                self.run_git_command(["cat-file", "-e", commit_hash])
            except GitOperationError as e:
                raise GitOperationError(f"Commit does not exist: {commit_hash}") from e

            # Perform checkout
            self.run_git_command(["checkout", commit_hash])
            logger.info(f"Checked out commit: {commit_hash}")

        if with_lock:
            with GitOperationLock(self.lock_file):
                _do_checkout()
        else:
            _do_checkout()

    def get_commit_info(self, commit_hash: str) -> dict[str, str]:
        """
        Get information about a commit.

        Args:
            commit_hash: Commit hash to query

        Returns:
            Dictionary with commit information
        """
        try:
            # Get commit details
            result = self.run_git_command(
                ["show", "--format=%H|%h|%an|%ae|%ad|%s", "--no-patch", commit_hash]
            )

            parts = result.stdout.strip().split("|")
            if len(parts) >= 6:
                return {
                    "full_hash": parts[0],
                    "short_hash": parts[1],
                    "author_name": parts[2],
                    "author_email": parts[3],
                    "date": parts[4],
                    "subject": parts[5],
                }

        except GitOperationError:
            pass

        return {"full_hash": commit_hash, "short_hash": commit_hash[:8]}

    def list_recent_commits(
        self, max_count: int = 10, branch: str = "HEAD"
    ) -> list[dict[str, str]]:
        """
        List recent commits.

        Args:
            max_count: Maximum number of commits to return
            branch: Branch or commit to start from

        Returns:
            List of commit information dictionaries
        """
        try:
            result = self.run_git_command(
                ["log", f"--max-count={max_count}", "--format=%H|%h|%an|%ad|%s", branch]
            )

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) >= 5:
                    commits.append(
                        {
                            "full_hash": parts[0],
                            "short_hash": parts[1],
                            "author_name": parts[2],
                            "date": parts[3],
                            "subject": parts[4],
                        }
                    )

            return commits

        except GitOperationError:
            return []


class BranchManager:
    """Manages temporary branches for SLURM jobs."""

    def __init__(self, git_manager: GitManager):
        """
        Initialize branch manager.

        Args:
            git_manager: GitManager instance
        """
        self.git = git_manager

    def create_job_branch(
        self, job_name: str, commit_hash: str, base_branch: str = "main"
    ) -> str:
        """
        Create a temporary branch for a SLURM job.

        Args:
            job_name: Name of the job/experiment
            commit_hash: Commit hash to branch from
            base_branch: Base branch name for branch naming

        Returns:
            Name of created branch

        Raises:
            GitOperationError: If branch creation fails
        """
        timestamp = int(time.time())
        short_hash = commit_hash[:8]
        branch_name = f"slurm-job/{job_name}/{timestamp}/{short_hash}"

        with GitOperationLock(self.git.lock_file):
            try:
                # Create and checkout new branch
                self.git.run_git_command(["checkout", "-b", branch_name, commit_hash])

                logger.info(f"Created job branch: {branch_name}")
                return branch_name

            except GitOperationError:
                logger.exception("Failed to create job branch: ")
                raise

    def cleanup_job_branches(
        self, older_than_days: int = 7, dry_run: bool = False
    ) -> list[str]:
        """
        Clean up old job branches.

        Args:
            older_than_days: Delete branches older than this many days
            dry_run: If True, only return branches that would be deleted

        Returns:
            List of branch names that were (or would be) deleted
        """
        try:
            # List all job branches
            result = self.git.run_git_command(
                [
                    "branch",
                    "--format=%(refname:short) %(committerdate:unix)",
                    "--list",
                    "slurm-job/*",
                ]
            )

            current_time = time.time()
            cutoff_time = current_time - (older_than_days * 24 * 3600)

            branches_to_delete = []

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    branch_name = parts[0]
                    commit_time = float(parts[1])

                    if commit_time < cutoff_time:
                        branches_to_delete.append(branch_name)

            if not dry_run and branches_to_delete:
                with GitOperationLock(self.git.lock_file):
                    for branch_name in branches_to_delete:
                        try:
                            self.git.run_git_command(["branch", "-D", branch_name])
                            logger.info(f"Deleted old job branch: {branch_name}")
                        except GitOperationError as e:
                            logger.warning(
                                f"Failed to delete branch {branch_name}: {e}"
                            )

            return branches_to_delete

        except GitOperationError:
            logger.exception("Failed to cleanup job branches: ")
            return []

    def list_job_branches(self) -> list[dict[str, str]]:
        """
        List all job branches with their information.

        Returns:
            List of branch information dictionaries
        """
        try:
            result = self.git.run_git_command(
                [
                    "branch",
                    "--format=%(refname:short)|%(committerdate:iso)|%(subject)",
                    "--list",
                    "slurm-job/*",
                ]
            )

            branches = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) >= 3:
                    branch_name = parts[0]

                    # Parse job info from branch name
                    branch_parts = branch_name.split("/")
                    job_info = {
                        "branch_name": branch_name,
                        "commit_date": parts[1],
                        "commit_subject": parts[2],
                    }

                    if len(branch_parts) >= 4:
                        job_info.update(
                            {
                                "job_name": branch_parts[1],
                                "timestamp": branch_parts[2],
                                "commit_hash": branch_parts[3],
                            }
                        )

                    branches.append(job_info)

            return branches

        except GitOperationError:
            return []

    def verify_job_commit(self, expected_commit: str) -> bool:
        """
        Verify that current HEAD matches expected commit.

        Args:
            expected_commit: Expected commit hash

        Returns:
            True if current commit matches expected
        """
        try:
            current_commit = self.git.get_current_commit()
            return current_commit.startswith(
                expected_commit
            ) or expected_commit.startswith(current_commit)
        except GitOperationError:
            return False

    def ensure_correct_commit(self, expected_commit: str) -> bool:
        """
        Ensure we're on the correct commit, checking out if necessary.

        Args:
            expected_commit: Expected commit hash

        Returns:
            True if successfully on correct commit
        """
        if self.verify_job_commit(expected_commit):
            return True

        try:
            logger.info(
                f"Current commit doesn't match expected {expected_commit}, checking out..."
            )
            self.git.checkout_commit(expected_commit, with_lock=True)
            return self.verify_job_commit(expected_commit)
        except GitOperationError:
            logger.exception("Failed to checkout expected commit: ")
            return False
