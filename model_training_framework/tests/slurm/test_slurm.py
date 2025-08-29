"""
Tests for SLURM integration component.

This module tests all aspects of the SLURM system including:
- Job launcher functionality
- Git operations and branch management
- Template rendering
- Job monitoring
"""

from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pytest

from model_training_framework.config.schemas import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from model_training_framework.slurm import (
    BatchSubmissionResult,
    GitOperationLock,
    JobInfo,
    JobStatus,
    SLURMJobMonitor,
    SLURMLauncher,
)
from model_training_framework.slurm.git_ops import (
    BranchManager,
    GitManager,
    GitOperationError,
)
from model_training_framework.slurm.templates import (
    SBATCHTemplateEngine,
    TemplateContext,
    TemplateError,
    create_template_context_from_config,
)


class TestGitOperationLock:
    """Test Git operation locking functionality."""

    def test_lock_creation(self):
        """Test Git operation lock creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "test.lock"
            lock = GitOperationLock(lock_path, timeout=1.0)

            assert lock.lock_file_path == lock_path
            assert lock.timeout == 1.0
            assert lock.acquired is False

    def test_lock_acquisition(self):
        """Test lock acquisition and release."""
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "test.lock"

            with GitOperationLock(lock_path, timeout=1.0) as lock:
                assert lock.acquired is True
                assert lock_path.exists()

    def test_lock_timeout(self):
        """Test lock timeout behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "test.lock"

            # Hold lock in first context
            lock1 = GitOperationLock(lock_path, timeout=1.0)
            lock1.acquire()
            try:
                # Second lock should timeout (wrapped in GitOperationError)
                with (
                    pytest.raises(GitOperationError),
                    GitOperationLock(lock_path, timeout=0.1),
                ):
                    pass
            finally:
                lock1.release()


class TestGitManager:
    """Test Git manager functionality."""

    def test_git_manager_creation(self, mock_git_repo):
        """Test Git manager creation."""
        project_root = mock_git_repo.parent
        manager = GitManager(project_root)

        assert manager.project_root == project_root
        assert manager.git_dir == mock_git_repo

    def test_git_manager_invalid_repo(self):
        """Test Git manager with invalid repository."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(GitOperationError),
        ):
            GitManager(Path(temp_dir))

    @patch("subprocess.run")
    def test_run_git_command(self, mock_run):
        """Test running git commands."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            git_dir = project_root / ".git"
            git_dir.mkdir()

            manager = GitManager(project_root)

            # Mock successful command
            mock_result = Mock()
            mock_result.stdout = "abc123def456\n"  # pragma: allowlist secret
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = manager.run_git_command(["rev-parse", "HEAD"])

            assert result == mock_result
            mock_run.assert_called_once()

            # Verify command construction
            call_args = mock_run.call_args
            assert call_args[0][0] == ["git", "rev-parse", "HEAD"]

    @patch("subprocess.run")
    def test_get_current_commit(self, mock_run):
        """Test getting current commit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            git_dir = project_root / ".git"
            git_dir.mkdir()

            manager = GitManager(project_root)

            mock_result = Mock()
            mock_result.stdout = "abc123def456789\n"  # pragma: allowlist secret
            mock_run.return_value = mock_result

            commit = manager.get_current_commit()

            assert commit == "abc123def456789"  # pragma: allowlist secret

    @patch("subprocess.run")
    def test_get_current_branch(self, mock_run):
        """Test getting current branch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            git_dir = project_root / ".git"
            git_dir.mkdir()

            manager = GitManager(project_root)

            mock_result = Mock()
            mock_result.stdout = "main\n"
            mock_run.return_value = mock_result

            branch = manager.get_current_branch()

            assert branch == "main"

    @patch("subprocess.run")
    def test_is_working_tree_clean(self, mock_run):
        """Test checking if working tree is clean."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            git_dir = project_root / ".git"
            git_dir.mkdir()

            manager = GitManager(project_root)

            # Clean working tree
            mock_result = Mock()
            mock_result.stdout = ""
            mock_run.return_value = mock_result

            assert manager.is_working_tree_clean()

            # Dirty working tree
            mock_result.stdout = "M modified_file.py\n"
            mock_run.return_value = mock_result

            assert not manager.is_working_tree_clean()


class TestBranchManager:
    """Test branch management functionality."""

    @patch("subprocess.run")
    def test_create_job_branch(self, mock_run):
        """Test creating job branch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            git_dir = project_root / ".git"
            git_dir.mkdir()

            git_manager = GitManager(project_root)
            branch_manager = BranchManager(git_manager)

            mock_result = Mock()
            mock_result.stdout = ""
            mock_run.return_value = mock_result

            branch_name = branch_manager.create_job_branch(
                "test_experiment",
                "abc123def456",  # pragma: allowlist secret
            )

            assert "slurm-job/test_experiment/" in branch_name
            assert "abc123de" in branch_name  # Short hash
            mock_run.assert_called()

    @patch("subprocess.run")
    def test_verify_job_commit(self, mock_run):
        """Test verifying job commit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            git_dir = project_root / ".git"
            git_dir.mkdir()

            git_manager = GitManager(project_root)
            branch_manager = BranchManager(git_manager)

            mock_result = Mock()
            mock_result.stdout = "abc123def456789\n"  # pragma: allowlist secret
            mock_run.return_value = mock_result

            # Exact match
            assert branch_manager.verify_job_commit(
                "abc123def456789"  # pragma: allowlist secret
            )

            # Prefix match
            assert branch_manager.verify_job_commit(
                "abc123def456"  # pragma: allowlist secret
            )

            # No match
            assert not branch_manager.verify_job_commit("different123")


class TestSBATCHTemplateEngine:
    """Test SBATCH template engine."""

    def test_template_engine_creation(self):
        """Test template engine creation."""
        engine = SBATCHTemplateEngine()

        assert len(engine.template_cache) == 0

    def test_load_template(self):
        """Test loading template from file."""
        engine = SBATCHTemplateEngine()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            template_content = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
echo "Hello {{EXPERIMENT_NAME}}"
"""
            f.write(template_content)
            f.flush()

            loaded_content = engine.load_template(Path(f.name))

            assert "{{JOB_NAME}}" in loaded_content
            assert "{{ACCOUNT}}" in loaded_content
            assert "{{EXPERIMENT_NAME}}" in loaded_content

    def test_validate_template(self):
        """Test template validation."""
        engine = SBATCHTemplateEngine()

        # Valid template
        valid_template = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "Running job"
"""

        issues = engine.validate_template(valid_template)
        assert len(issues) == 0

        # Invalid template (missing required directives)
        invalid_template = """#!/bin/bash
echo "Missing SBATCH directives"
"""

        issues = engine.validate_template(invalid_template)
        assert len(issues) > 0

    def test_render_template(self):
        """Test template rendering."""
        engine = SBATCHTemplateEngine()

        template = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
echo "Running {{EXPERIMENT_NAME}}"
"""

        context = {
            "JOB_NAME": "test_job",
            "ACCOUNT": "test_account",
            "EXPERIMENT_NAME": "my_experiment",
        }

        rendered = engine.render_template(template, context)

        assert "#SBATCH --job-name=test_job" in rendered
        assert "#SBATCH --account=test_account" in rendered
        assert 'echo "Running my_experiment"' in rendered

    def test_create_default_template(self):
        """Test creating default template."""
        engine = SBATCHTemplateEngine()

        template = engine.create_default_template()

        assert "#!/bin/bash" in template
        assert "#SBATCH --job-name={{JOB_NAME}}" in template
        assert "#SBATCH --account={{ACCOUNT}}" in template
        assert "{{SCRIPT_PATH}}" in template

    def test_init_with_template_string(self):
        """Test initializing engine with template string."""
        template_string = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "Test template"
"""
        engine = SBATCHTemplateEngine(template_string=template_string)

        assert engine.template_string == template_string
        # Check that a content-hashed key exists in cache
        cache_keys = list(engine.template_cache.keys())
        assert any("string_template_" in key for key in cache_keys)
        # Verify the template is cached
        assert template_string in engine.template_cache.values()

    def test_load_template_from_string(self):
        """Test loading template from string."""
        engine = SBATCHTemplateEngine()

        template_string = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "String template: {{EXPERIMENT_NAME}}"
"""

        loaded = engine.load_template_from_string(template_string)

        assert loaded == template_string
        assert engine.template_string == template_string
        # Check that a content-hashed key exists in cache
        cache_keys = list(engine.template_cache.keys())
        assert any("string_template_" in key for key in cache_keys)

    def test_set_template_string(self):
        """Test setting template string after initialization."""
        engine = SBATCHTemplateEngine()

        template_string = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "Updated template"
"""

        engine.set_template_string(template_string)

        assert engine.template_string == template_string
        # Check that a content-hashed key exists in cache
        cache_keys = list(engine.template_cache.keys())
        assert any("string_template_" in key for key in cache_keys)

    def test_generate_sbatch_with_string_template(self):
        """Test generating SBATCH script with string template."""
        template_string = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "Job: {{JOB_NAME}}"
echo "Experiment: {{EXPERIMENT_NAME}}"
"""

        engine = SBATCHTemplateEngine()

        context = TemplateContext(
            job_name="string_test_job",
            experiment_name="string_test_exp",
            script_path="train.py",
            config_name="config.yaml",
            account="test_account",
            partition="test_partition",
        )

        # Generate with string template passed directly
        rendered = engine.generate_sbatch_script(
            context, template_string=template_string
        )

        assert "#SBATCH --job-name=string_test_job" in rendered
        assert "#SBATCH --account=test_account" in rendered
        assert "#SBATCH --partition=test_partition" in rendered
        assert 'echo "Job: string_test_job"' in rendered
        assert 'echo "Experiment: string_test_exp"' in rendered

    def test_generate_sbatch_with_default_string_template(self):
        """Test generating SBATCH script with default string template."""
        template_string = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "Default string template"
"""

        # Initialize with default string template
        engine = SBATCHTemplateEngine(template_string=template_string)

        context = TemplateContext(
            job_name="default_string_job",
            experiment_name="default_string_exp",
            script_path="train.py",
            config_name="config.yaml",
        )

        # Generate using default string template
        rendered = engine.generate_sbatch_script(context)

        assert "#SBATCH --job-name=default_string_job" in rendered
        assert 'echo "Default string template"' in rendered

    def test_preview_with_string_template(self):
        """Test preview with string template."""
        engine = SBATCHTemplateEngine()

        template_string = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "Preview: {{JOB_NAME}}"
"""

        context = TemplateContext(
            job_name="preview_job",
            experiment_name="preview_exp",
            script_path="train.py",
            config_name="config.yaml",
        )

        preview = engine.preview_rendered_script(
            context, template_string=template_string
        )

        assert "#SBATCH --job-name=preview_job" in preview
        assert 'echo "Preview: preview_job"' in preview

    def test_string_template_precedence(self):
        """Test that string template takes precedence over file template."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            file_template = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "File template"
"""
            f.write(file_template)
            f.flush()

            string_template = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "String template"
"""

            # Initialize with both file and string
            engine = SBATCHTemplateEngine(
                template_path=Path(f.name), template_string=string_template
            )

            context = TemplateContext(
                job_name="precedence_job",
                experiment_name="precedence_exp",
                script_path="train.py",
                config_name="config.yaml",
            )

            # Should use string template
            rendered = engine.generate_sbatch_script(context)

            assert 'echo "String template"' in rendered
            assert 'echo "File template"' not in rendered

    def test_jinja_block_detection(self):
        """Test that Jinja control blocks are detected and rejected."""
        engine = SBATCHTemplateEngine()

        # Template with Jinja if block
        jinja_template = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
{% if CONSTRAINT %}
#SBATCH --constraint={{CONSTRAINT}}
{% endif %}
echo "This has Jinja blocks"
"""

        # Validation should detect Jinja blocks
        issues = engine.validate_template(jinja_template)
        assert any("Unsupported Jinja" in issue for issue in issues)

        # Generation should fail with Jinja blocks
        context = TemplateContext(
            job_name="jinja_test",
            experiment_name="jinja_exp",
            script_path="train.py",
            config_name="config.yaml",
        )

        with pytest.raises(TemplateError) as exc_info:
            engine.generate_sbatch_script(context, template_string=jinja_template)

        assert "unsupported Jinja control blocks" in str(exc_info.value)

    def test_content_hashed_caching(self):
        """Test that string templates are cached with content-based keys."""
        engine = SBATCHTemplateEngine()

        template1 = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "Template 1"
"""

        template2 = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
echo "Template 2"
"""

        # Load both templates
        engine.load_template_from_string(template1)
        engine.load_template_from_string(template2)

        # Both should be in cache with different keys
        assert len(engine.template_cache) >= 2

        # Cache keys should be content-based (contain hash)
        cache_keys = list(engine.template_cache.keys())
        assert any("string_template_" in key for key in cache_keys)

    def test_default_template_has_scalable_launch(self):
        """Test that default template includes scalable launch logic."""
        engine = SBATCHTemplateEngine()
        template = engine.create_default_template()

        # Check for multi-task launch logic
        assert 'if [ "$NTASKS_PER_NODE" -gt 1 ]' in template
        assert "srun --ntasks-per-node=$NTASKS_PER_NODE" in template

        # Check for thread exports
        assert "export OMP_NUM_THREADS={{CPUS_PER_TASK}}" in template
        assert "export MKL_NUM_THREADS={{CPUS_PER_TASK}}" in template

        # Check for NCCL settings
        assert "export NCCL_ASYNC_ERROR_HANDLING=1" in template

        # Check that Jinja blocks are NOT present
        assert "{% if" not in template
        assert "{% endif" not in template


class TestTemplateContext:
    """Test template context functionality."""

    def test_template_context_creation(self):
        """Test template context creation."""
        context = TemplateContext(
            job_name="test_job",
            experiment_name="test_experiment",
            script_path="scripts/train.py",
            config_name="test_config",
        )

        assert context.job_name == "test_job"
        assert context.experiment_name == "test_experiment"
        assert context.script_path == "scripts/train.py"
        assert context.config_name == "test_config"

    def test_template_context_to_dict(self):
        """Test converting context to dictionary."""
        context = TemplateContext(
            job_name="test_job",
            experiment_name="test_experiment",
            script_path="scripts/train.py",
            config_name="test_config",
            account="test_account",
            partition="gpu",
        )

        context_dict = context.to_dict()

        assert context_dict["JOB_NAME"] == "test_job"
        assert context_dict["EXPERIMENT_NAME"] == "test_experiment"
        assert context_dict["SCRIPT_PATH"] == "scripts/train.py"
        assert context_dict["CONFIG_NAME"] == "test_config"
        assert context_dict["ACCOUNT"] == "test_account"
        assert context_dict["PARTITION"] == "gpu"

    def test_create_template_context_from_config(self):
        """Test creating context from configuration."""
        slurm_config = {
            "account": "realitylab",
            "partition": "gpu",
            "nodes": 2,
            "gpus_per_node": 4,
            "time": "12:00:00",
        }

        git_info = {
            "branch_name": "main",
            "commit_hash": "abc123def456",  # pragma: allowlist secret
        }

        context = create_template_context_from_config(
            experiment_name="test_exp",
            script_path="train.py",
            config_name="config.yaml",
            slurm_config=slurm_config,
            git_info=git_info,
        )

        assert context.experiment_name == "test_exp"
        assert context.script_path == "train.py"
        assert context.config_name == "config.yaml"
        assert context.account == "realitylab"
        assert context.partition == "gpu"
        assert context.nodes == 2
        assert context.gpus_per_node == 4
        assert context.branch_name == "main"
        assert context.commit_hash == "abc123def456"  # pragma: allowlist secret


class TestJobStatus:
    """Test job status enumeration."""

    def test_job_status_values(self):
        """Test job status enumeration values."""
        assert JobStatus.PENDING.value == "PENDING"
        assert JobStatus.RUNNING.value == "RUNNING"
        assert JobStatus.COMPLETED.value == "COMPLETED"
        assert JobStatus.CANCELLED.value == "CANCELLED"
        assert JobStatus.FAILED.value == "FAILED"


class TestJobInfo:
    """Test job information class."""

    def test_job_info_creation(self):
        """Test job info creation."""
        job_info = JobInfo(
            job_id="12345", name="test_job", status=JobStatus.RUNNING, partition="gpu"
        )

        assert job_info.job_id == "12345"
        assert job_info.name == "test_job"
        assert job_info.status == JobStatus.RUNNING
        assert job_info.partition == "gpu"

    def test_job_info_properties(self):
        """Test job info properties."""
        # Active job
        active_job = JobInfo(
            job_id="12345", name="test_job", status=JobStatus.RUNNING, partition="gpu"
        )

        assert active_job.is_active
        assert not active_job.is_finished
        assert not active_job.was_successful

        # Completed job
        completed_job = JobInfo(
            job_id="12346",
            name="test_job2",
            status=JobStatus.COMPLETED,
            partition="gpu",
            exit_code="0",
        )

        assert not completed_job.is_active
        assert completed_job.is_finished
        assert completed_job.was_successful

        # Failed job
        failed_job = JobInfo(
            job_id="12347",
            name="test_job3",
            status=JobStatus.FAILED,
            partition="gpu",
            exit_code="1",
        )

        assert not failed_job.is_active
        assert failed_job.is_finished
        assert not failed_job.was_successful


class TestSLURMJobMonitor:
    """Test SLURM job monitoring."""

    def test_job_monitor_creation(self):
        """Test job monitor creation."""
        monitor = SLURMJobMonitor(update_interval=30.0)

        assert monitor.update_interval == 30.0
        assert len(monitor.tracked_jobs) == 0
        assert len(monitor.job_history) == 0

    def test_track_job(self):
        """Test tracking a job."""
        monitor = SLURMJobMonitor()

        with patch.object(monitor, "get_job_info") as mock_get_info:
            mock_job_info = JobInfo(
                job_id="12345",
                name="test_job",
                status=JobStatus.RUNNING,
                partition="gpu",
            )
            mock_get_info.return_value = mock_job_info

            monitor.track_job("12345")

            assert "12345" in monitor.tracked_jobs
            assert monitor.tracked_jobs["12345"] == mock_job_info

    def test_stop_tracking_job(self):
        """Test stopping job tracking."""
        monitor = SLURMJobMonitor()

        # Add a job to tracking
        job_info = JobInfo(
            job_id="12345", name="test_job", status=JobStatus.COMPLETED, partition="gpu"
        )
        monitor.tracked_jobs["12345"] = job_info

        monitor.stop_tracking_job("12345")

        assert "12345" not in monitor.tracked_jobs
        assert len(monitor.job_history) == 1
        assert monitor.job_history[0] == job_info

    @patch("subprocess.run")
    def test_parse_job_state(self, mock_run):
        """Test parsing job state from SLURM output."""
        monitor = SLURMJobMonitor()

        # Test various state mappings
        assert monitor._parse_job_state("PENDING") == JobStatus.PENDING
        assert monitor._parse_job_state("PD") == JobStatus.PENDING
        assert monitor._parse_job_state("RUNNING") == JobStatus.RUNNING
        assert monitor._parse_job_state("R") == JobStatus.RUNNING
        assert monitor._parse_job_state("COMPLETED") == JobStatus.COMPLETED
        assert monitor._parse_job_state("CD") == JobStatus.COMPLETED
        assert monitor._parse_job_state("UNKNOWN_STATE") == JobStatus.UNKNOWN

    def test_get_job_summary(self):
        """Test getting job summary."""
        monitor = SLURMJobMonitor()

        # Add some tracked jobs
        monitor.tracked_jobs["12345"] = JobInfo(
            job_id="12345", name="job1", status=JobStatus.RUNNING, partition="gpu"
        )
        monitor.tracked_jobs["12346"] = JobInfo(
            job_id="12346", name="job2", status=JobStatus.PENDING, partition="gpu"
        )

        # Add some finished jobs
        monitor.job_history.append(
            JobInfo(
                job_id="12347", name="job3", status=JobStatus.COMPLETED, partition="gpu"
            )
        )

        summary = monitor.get_job_summary()

        assert summary["active_jobs"] == 2
        assert summary["finished_jobs"] == 1
        assert summary["total_tracked"] == 3
        assert "RUNNING" in summary["active_by_status"]
        assert "PENDING" in summary["active_by_status"]
        assert "COMPLETED" in summary["finished_by_status"]


@pytest.mark.skipif_no_slurm
class TestSLURMLauncher:
    """Test SLURM launcher functionality."""

    def test_slurm_launcher_creation(self, test_project_root):
        """Test SLURM launcher creation."""
        template_path = test_project_root / "slurm_template.txt"

        launcher = SLURMLauncher(
            template_path=template_path, project_root=test_project_root
        )

        assert launcher.template_path == template_path
        assert launcher.project_root == test_project_root
        assert launcher.experiments_dir.exists()

    def test_extract_slurm_config(self, test_project_root):
        """Test extracting SLURM configuration."""
        template_path = test_project_root / "slurm_template.txt"
        launcher = SLURMLauncher(template_path, test_project_root)

        # Test with no SLURM config
        config = ExperimentConfig(
            experiment_name="test",
            model=ModelConfig(type="test"),
            training=TrainingConfig(),
            data=DataConfig(dataset_name="test"),
            optimizer=OptimizerConfig(),
        )

        slurm_config = launcher._extract_slurm_config(config)

        assert "account" in slurm_config
        assert "partition" in slurm_config
        assert slurm_config["account"] == "realitylab"

    @patch("subprocess.run")
    def test_get_job_status(self, mock_run, test_project_root):
        """Test getting job status."""
        template_path = test_project_root / "slurm_template.txt"
        launcher = SLURMLauncher(template_path, test_project_root)

        # Mock squeue output
        mock_result = Mock()
        mock_result.stdout = "12345,RUNNING,node01,2023-01-01T10:00:00,gpu\n"
        mock_run.return_value = mock_result

        status = launcher.get_job_status("12345")

        assert status["job_id"] == "12345"
        assert status["state"] == "RUNNING"
        assert status["nodes"] == "node01"

    @patch("subprocess.run")
    def test_cancel_job(self, mock_run, test_project_root):
        """Test cancelling a job."""
        template_path = test_project_root / "slurm_template.txt"
        launcher = SLURMLauncher(template_path, test_project_root)

        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        success = launcher.cancel_job("12345")

        assert success
        mock_run.assert_called_with(
            ["scancel", "12345"], capture_output=True, text=True, check=True
        )

    def test_validate_slurm_environment(self, test_project_root):
        """Test validating SLURM environment."""
        template_path = test_project_root / "slurm_template.txt"
        launcher = SLURMLauncher(template_path, test_project_root)

        status = launcher.validate_slurm_environment()

        assert "slurm_available" in status
        assert "commands_available" in status
        assert "missing_commands" in status
        assert "template_valid" in status
        assert "template_issues" in status


class TestBatchSubmissionResult:
    """Test batch submission result."""

    def test_batch_result_creation(self):
        """Test batch submission result creation."""
        result = BatchSubmissionResult(total_experiments=10)

        assert result.total_experiments == 10
        assert len(result.successful_jobs) == 0
        assert len(result.failed_jobs) == 0
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.success_rate == 0.0  # No jobs submitted yet = 0% success

    def test_batch_result_with_jobs(self):
        """Test batch result with actual jobs."""
        result = BatchSubmissionResult(total_experiments=5)
        result.successful_jobs = ["12345", "12346", "12347"]
        result.failed_jobs = {"exp4": "SLURM error", "exp5": "Config error"}

        assert result.success_count == 3
        assert result.failure_count == 2
        assert result.success_rate == 60.0  # 3/5 = 60%
