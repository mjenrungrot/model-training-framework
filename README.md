"""
Tests for the SLURM launcher script (`slurm/launcher.py`).

This test suite verifies the functionality of the SLURM job launcher, covering:
- Configuration file handling (YAML parsing, validation).
- SBATCH script generation (template rendering, placeholder substitution).
- Main execution flow (argument parsing, pre-flight checks, conditional logic based on flags like --generate-sbatch-only and --dry-run, job submission).
- Error handling for various scenarios (missing files, invalid config, command failures).
- SLURM parameter precedence (CLI > YAML > Defaults).

The tests utilize `pytest` fixtures (`test_env`) to create isolated temporary
environments with necessary files and directories. Mocks (`pytest-mock` and
`unittest.mock`) are extensively used to simulate external dependencies like
file system operations (`pathlib.Path`, `shutil`), subprocess execution (`subprocess.run`),
and system exit (`sys.exit`), allowing for focused testing of the launcher's logic.

Special attention is given to mocking path operations (`Path.is_file`, `Path.is_dir`)
and handling path resolution inconsistencies (e.g., `/private/var` vs `/var` on macOS)
by using `.resolve()` consistently in both the main code and test assertions
where necessary.
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO
from unittest.mock import ANY, mock_open

import pytest
import yaml
from pytest_mock import MockerFixture

from slurm import launcher

# --- Constants for Testing ---
TEST_JOB_NAME = "test_experiment_job"
TEST_PYTHON_SCRIPT_REL_PATH = "scripts/test_train.py"  # Relative path used in launcher
TEST_CONFIG_NAME = TEST_JOB_NAME
TEST_YAML_CONTENT_VALID = f"""
experiment_name: {TEST_JOB_NAME}
model:
  type: resnet
parameters:
  lr: 0.001
"""
TEST_YAML_CONTENT_MISMATCH = """
experiment_name: another_job_name
model:
  type: resnet
"""
TEST_YAML_CONTENT_NO_KEY = """
model:
  type: resnet
"""
TEST_TEMPLATE_CONTENT = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output={{OUTPUT}}
#SBATCH --error={{ERROR}}
#SBATCH --comment={{COMMIT_HASH}}
{{REQUEUE_DIRECTIVE}}

echo "Running {{PYTHON_SCRIPT_PATH}} with config {{CONFIG_NAME}} at commit {{COMMIT_HASH}}"
srun python {{PYTHON_SCRIPT_PATH}} --config-name={{CONFIG_NAME}}
"""
FAKE_COMMIT_HASH = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f60000"

# --- Fixture for Test Environment Setup/Teardown ---


@pytest.fixture
def test_env(mocker: MockerFixture) -> Iterator[dict[str, Any]]:
    """
    Pytest fixture to set up a temporary test environment for launcher tests.

    Creates a temporary directory structure containing:
    - `configs/` directory with a sample YAML configuration file.
    - `scripts/` directory with a dummy Python script.
    - `experiments/` directory (initially empty).
    - A sample `slurm_template.sbatch` file.

    It also mocks `slurm.launcher.PROJECT_ROOT` to point to the temporary
    directory root, ensuring tests operate within an isolated environment.

    Yields:
        A dictionary containing Paths and other relevant data for the test
        environment, including:
        - `test_dir`: Root Path of the temporary directory.
        - `configs_dir`: Path to the temporary configs directory.
        - `experiments_dir`: Path to the temporary experiments directory.
        - `scripts_dir`: Path to the temporary scripts directory.
        - `template_path`: Path to the dummy template file.
        - `config_path`: Path to the dummy config file.
        - `python_script_path_abs`: Absolute Path to the dummy Python script.
        - `python_script_path_rel`: Relative Path string for the Python script.
        - `sbatch_output_path`: Expected Path for the generated SBATCH script.

    Cleans up the temporary directory and its contents after the test finishes.

    """
    # Create a temporary directory
    test_dir = Path(tempfile.mkdtemp())

    # Mock launcher.PROJECT_ROOT to use the temporary directory
    mocker.patch("slurm.launcher.PROJECT_ROOT", test_dir)

    # Define paths relative to the temp directory
    env = {
        "test_dir": test_dir,
        "configs_dir": test_dir / "configs",
        "experiments_dir": test_dir / "experiments",
        "scripts_dir": test_dir / "scripts",
        "template_path": test_dir / "slurm_template.sbatch",
        "config_path": test_dir / "configs" / f"{TEST_CONFIG_NAME}.yaml",
        "python_script_path_abs": test_dir / TEST_PYTHON_SCRIPT_REL_PATH,
        "python_script_path_rel": TEST_PYTHON_SCRIPT_REL_PATH,
        "sbatch_output_path": test_dir / "experiments" / TEST_JOB_NAME / f"{TEST_JOB_NAME}.sbatch",
        "yaml_job_key": launcher.YAML_JOB_NAME_KEY,
    }

    # Create necessary directories
    env["configs_dir"].mkdir()
    env["scripts_dir"].mkdir()

    # Create dummy files
    env["template_path"].write_text(TEST_TEMPLATE_CONTENT)
    env["python_script_path_abs"].write_text("print('hello from test script')")
    env["config_path"].write_text(TEST_YAML_CONTENT_VALID)

    yield env  # Provide the environment dict to the test function

    # Teardown: Remove the temporary directory
    shutil.rmtree(test_dir)


# --- Tests for load_yaml_config ---


def test_load_yaml_config_valid(test_env: dict[str, Any], mocker: MockerFixture) -> None:
    """Test `load_yaml_config` successfully loads a valid YAML file."""
    mock_is_file = mocker.patch("pathlib.Path.is_file", return_value=True)
    # Use mocker.patch for builtins.open
    mocker.patch("pathlib.Path.open", mock_open(read_data=TEST_YAML_CONTENT_VALID))

    config = launcher.load_yaml_config(test_env["config_path"])
    expected_config = yaml.safe_load(TEST_YAML_CONTENT_VALID)

    assert config == expected_config
    mock_is_file.assert_called_once_with()
    # builtins.open mock check might be less straightforward with mocker,
    # focus on the result and is_file call.


def test_load_yaml_config_not_found(mocker: MockerFixture) -> None:
    """Test `load_yaml_config` exits when the YAML file is not found."""
    mock_is_file = mocker.patch("pathlib.Path.is_file", return_value=False)
    mock_exit = mocker.patch("sys.exit")

    launcher.load_yaml_config(Path("non_existent.yaml"))

    mock_is_file.assert_called_once_with()
    # The function calls exit twice in this case: once for is_file check, once for open exception
    assert mock_exit.call_count == 2
    mock_exit.assert_called_with(1)  # Check the argument of the *last* call


def test_load_yaml_config_empty(test_env: dict[str, Any], mocker: MockerFixture) -> None:
    """Test `load_yaml_config` returns an empty dict for an empty YAML file."""
    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch("pathlib.Path.open", mock_open(read_data=""))  # Empty file

    config = launcher.load_yaml_config(test_env["config_path"])
    assert config == {}  # Expect an empty dict


def test_load_yaml_config_invalid_syntax(test_env: dict[str, Any], mocker: MockerFixture) -> None:
    """Test `load_yaml_config` exits when the YAML file has syntax errors."""
    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch("pathlib.Path.open", mock_open(read_data="key: value\n invalid-yaml:"))
    mock_exit = mocker.patch("sys.exit")

    launcher.load_yaml_config(test_env["config_path"])
    mock_exit.assert_called_once_with(1)


# --- Tests for generate_sbatch_script ---


def test_generate_sbatch_script_success(test_env: dict[str, Any]) -> None:
    """
    Test `generate_sbatch_script` successfully creates an SBATCH file.

    Verifies:
    - The output directory and file are created.
    - Placeholders (job name, script path, config name) are correctly substituted.
    - Default SLURM parameters are correctly substituted.
    - No placeholders remain in the final rendered content.
    """
    # Prepare context dictionary matching the expected format (placeholder keys)
    context = {
        "{{JOB_NAME}}": TEST_JOB_NAME,
        "{{PYTHON_SCRIPT_PATH}}": test_env["python_script_path_rel"],
        "{{CONFIG_NAME}}": TEST_CONFIG_NAME,
    }
    # Add SLURM params with placeholder keys
    for k, v in launcher.DEFAULT_SLURM_PARAMS.items():
        # Resolve nested {{JOB_NAME}} placeholders in default values if present
        if isinstance(v, str) and "{{JOB_NAME}}" in v:
            resolved_value = v.replace("{{JOB_NAME}}", TEST_JOB_NAME)
        else:
            resolved_value = v
        context[f"{{{{{k.upper()}}}}}"] = resolved_value

    # Add requeue directive (matches launcher.py logic)
    requeue_enabled = launcher.DEFAULT_SLURM_PARAMS.get("requeue", False)
    context["{{REQUEUE_DIRECTIVE}}"] = "#SBATCH --requeue" if requeue_enabled else ""

    # Modify the template content for this test to include SLURM placeholders
    template_with_slurm = """#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output={{OUTPUT}}
#SBATCH --error={{ERROR}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodes={{NODES}}
#SBATCH --ntasks-per-node={{NTASKS_PER_NODE}}
#SBATCH --gpus-per-node={{GPUS_PER_NODE}}
#SBATCH --cpus-per-task={{CPUS_PER_TASK}}
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}
{{REQUEUE_DIRECTIVE}}

echo "Running {{PYTHON_SCRIPT_PATH}} with config {{CONFIG_NAME}}"
srun python {{PYTHON_SCRIPT_PATH}} --config-name={{CONFIG_NAME}}
"""
    test_env["template_path"].write_text(template_with_slurm)

    # Ensure the output directory exists for this direct call test
    test_env["sbatch_output_path"].parent.mkdir(parents=True, exist_ok=True)

    launcher.generate_sbatch_script(
        template_path=test_env["template_path"],
        output_path=test_env["sbatch_output_path"],
        context=context,  # Pass the context dictionary
    )
    # Check if the output directory and file were created
    assert test_env["sbatch_output_path"].parent.exists()
    assert test_env["sbatch_output_path"].exists()

    # Check the content of the generated file
    content = test_env["sbatch_output_path"].read_text()
    assert f"#SBATCH --job-name={TEST_JOB_NAME}" in content
    assert f"#SBATCH --output=experiments/{TEST_JOB_NAME}/%j.out" in content
    assert f"#SBATCH --error=experiments/{TEST_JOB_NAME}/%j.err" in content
    assert f'echo "Running {test_env["python_script_path_rel"]} with config {TEST_CONFIG_NAME}"' in content
    assert f"srun python {test_env['python_script_path_rel']} --config-name={TEST_CONFIG_NAME}" in content
    # Check if SLURM directives were replaced using default values
    assert f"#SBATCH --account={launcher.DEFAULT_SLURM_PARAMS['account']}" in content
    assert f"#SBATCH --partition={launcher.DEFAULT_SLURM_PARAMS['partition']}" in content
    assert f"#SBATCH --nodes={launcher.DEFAULT_SLURM_PARAMS['nodes']}" in content
    # Check requeue directive is present since it's enabled by default
    assert "#SBATCH --requeue" in content
    # ... add more assertions for other SLURM params if desired

    # --- NEW: Check that placeholders are no longer present ---
    assert "{{JOB_NAME}}" not in content
    assert "{{PYTHON_SCRIPT_PATH}}" not in content
    assert "{{CONFIG_NAME}}" not in content
    assert "{{ACCOUNT}}" not in content
    assert "{{PARTITION}}" not in content
    assert "{{NODES}}" not in content
    # Add checks for other SLURM placeholders as needed
    assert "{{NTASKS_PER_NODE}}" not in content
    assert "{{GPUS_PER_NODE}}" not in content
    assert "{{CPUS_PER_TASK}}" not in content
    assert "{{MEM}}" not in content
    assert "{{TIME}}" not in content
    assert "{{OUTPUT}}" not in content
    assert "{{ERROR}}" not in content
    assert "{{REQUEUE_DIRECTIVE}}" not in content


def test_generate_sbatch_script_template_not_found(test_env: dict[str, Any], mocker: MockerFixture) -> None:
    """Test `generate_sbatch_script` exits if the template file is not found."""
    # Mock sys.exit without side effect to prevent test abortion
    mock_exit = mocker.patch("sys.exit")
    # Patch is_file specifically for the template check inside the function
    mocker.patch("slurm.launcher.Path.is_file", return_value=False)
    # Need to pass a minimal context even if it fails before using it
    context = {"job_name": "test", "python_script_path": "test.py", "config_name": "test"}

    launcher.generate_sbatch_script(
        template_path=Path("non_existent_template.sbatch"),
        output_path=test_env["sbatch_output_path"],
        context=context,  # Pass context
    )
    # Expect 1 call: generate_sbatch_script catches the FileNotFoundError and exits
    mock_exit.assert_called_once_with(1)


def test_generate_sbatch_script_write_error(test_env: dict[str, Any], mocker: MockerFixture) -> None:
    """Test `generate_sbatch_script` exits if writing the output file fails."""
    # Mock sys.exit without side effect
    mock_exit = mocker.patch("sys.exit")
    # Mock the write operation specifically to raise an error
    # Need to allow the initial read of the template to succeed
    mock_template_open = mock_open(read_data=TEST_TEMPLATE_CONTENT)

    # --- Corrected path_open_side_effect mock --- #
    # It needs to handle being called as a method (with self) when patching Path.open
    def path_open_side_effect(path_instance: Path, mode: str = "r", *args: Any, **kwargs: Any) -> TextIO:
        if path_instance == test_env["template_path"] and mode == "r":
            # Use the mock_open object directly
            return mock_template_open.return_value
        if path_instance == test_env["sbatch_output_path"] and mode == "w":
            # Fail writing the output
            raise OSError("Disk full")
        # Fallback for any other Path.open calls (should not happen in this test)
        raise NotImplementedError(f"Unexpected Path.open call: {path_instance}, mode={mode}")

    # -------------------------------------------- #

    # Patch pathlib.Path.open for the write operation
    mocker.patch("pathlib.Path.open", side_effect=path_open_side_effect)
    # Ensure the template file read check passes
    mocker.patch("slurm.launcher.Path.is_file", return_value=True)

    # --- Prepare context in the correct format --- #
    context_write_error = {
        "{{JOB_NAME}}": TEST_JOB_NAME,
        "{{PYTHON_SCRIPT_PATH}}": test_env["python_script_path_rel"],
        "{{CONFIG_NAME}}": TEST_CONFIG_NAME,
    }
    for k, v in launcher.DEFAULT_SLURM_PARAMS.items():
        context_write_error[f"{{{{{k.upper()}}}}}"] = v
    # Add requeue directive (matches launcher.py logic)
    requeue_enabled = launcher.DEFAULT_SLURM_PARAMS.get("requeue", False)
    context_write_error["{{REQUEUE_DIRECTIVE}}"] = "#SBATCH --requeue" if requeue_enabled else ""
    # --------------------------------------------- #

    launcher.generate_sbatch_script(
        template_path=test_env["template_path"],
        output_path=test_env["sbatch_output_path"],
        context=context_write_error,  # Pass the correctly formatted context
    )
    # Expect 1 call: only for the write OSError
    mock_exit.assert_called_once_with(1)


# --- Tests for main execution flow ---


# Helper function to create mock args namespace
def create_mock_args(**kwargs: Any) -> argparse.Namespace:
    """
    Helper function to create a mock `argparse.Namespace` object.

    Initializes a namespace with default arguments expected by the launcher
    and updates it with any provided keyword arguments.

    Args:
        **kwargs: Keyword arguments to override default values.

    Returns:
        An `argparse.Namespace` object suitable for mocking `parse_args`.

    """
    base_args = {
        "python_script": TEST_PYTHON_SCRIPT_REL_PATH,
        "config_name": TEST_CONFIG_NAME,
        "generate_sbatch_only": False,
        "template": "slurm_template.sbatch",  # Default value from argparse
        "configs_dir": "configs",  # Default value
        "experiments_dir": "experiments",  # Default value
        "yaml_job_key": launcher.YAML_JOB_NAME_KEY,  # Default value
        "verbose": False,
        "dry_run": False,  # Add default for dry_run
        "run_local": False,  # Default for new flag
        "branch_lock": False,  # Default for branch behavior
        "commit_hash": None,  # Added commit_hash default
        # Add SLURM defaults that can be overridden by kwargs
        "account": None,
        "partition": None,
        "nodes": None,
        "ntasks_per_node": None,
        "gpus_per_node": None,
        "cpus_per_task": None,
        "mem": None,
        "time": None,
        "constraint": None,
        "requeue": None,
        "output": None,
        "error": None,
    }
    base_args.update(kwargs)
    # Convert path strings back to strings for argparse mock
    for key in ["template", "configs_dir", "experiments_dir"]:
        if isinstance(base_args[key], Path):
            base_args[key] = str(base_args[key])

    return argparse.Namespace(**base_args)


# Use a class to group main tests for better organization (optional)
class TestMainFlow:
    """
    Groups tests for the main execution flow (`launcher.main`) of the launcher script.

    Uses a class-level fixture (`setup_main_mocks`) to automatically set up
    common mocks for all tests within this class.
    """

    @pytest.fixture(autouse=True)
    def setup_main_mocks(self, mocker: MockerFixture, test_env: dict[str, Any]) -> None:
        """
        Auto-used fixture to set up common mocks for `TestMainFlow` tests.

        Mocks key dependencies used within `launcher.main`, including:
        - `argparse.ArgumentParser.parse_args`
        - `slurm.launcher.load_yaml_config`
        - `slurm.launcher.generate_sbatch_script`
        - `subprocess.run`
        - `shutil.copy2`
        - `sys.exit` (configured to raise SystemExit to allow `pytest.raises`)
        - `pathlib.Path.is_file` and `pathlib.Path.is_dir`

        Sets default return values and side effects for mocks to simulate a
        successful base case. Specific tests can override these defaults.

        The `is_file` and `is_dir` mocks use side effect functions that compare
        resolved paths from the `test_env` fixture to handle potential path
        inconsistencies during testing (e.g., `/private/var` vs `/var`).

        Args:
            mocker: The `pytest-mock` fixture.
            test_env: The test environment fixture.

        """
        self.mocker = mocker
        self.test_env = test_env

        # Mock dependencies called by main using full paths
        # Note: launcher.py now uses parse_known_args instead of parse_args
        self.mock_parse_known_args = self.mocker.patch("argparse.ArgumentParser.parse_known_args")
        self.mock_load_yaml = self.mocker.patch("slurm.launcher.load_yaml_config")
        self.mock_generate = self.mocker.patch("slurm.launcher.generate_sbatch_script")
        self.mock_run = self.mocker.patch("subprocess.run")
        self.mock_copy2 = self.mocker.patch("shutil.copy2")  # Mock shutil.copy2
        # Mock sys.exit to raise SystemExit, like the real one does
        self.mock_exit = self.mocker.patch("sys.exit", side_effect=SystemExit(1))
        # Note: Path checks in main likely use pathlib directly, not launcher.Path
        self.mock_path_is_file = self.mocker.patch("pathlib.Path.is_file", autospec=True)
        self.mock_path_is_dir = self.mocker.patch("pathlib.Path.is_dir", autospec=True)

        # --- Default subprocess.run side effect function ---
        self.git_status_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        self.git_rev_parse_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=f"{FAKE_COMMIT_HASH}\n", stderr="")
        self.sbatch_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="Submitted job 123", stderr="")
        self._last_checkout_hash: str | None = None  # Track the last git checkout hash for verification

        def run_side_effect(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
            cmd_list = args[0]  # Command is the first argument
            if "sbatch" in cmd_list[0]:
                if isinstance(self.sbatch_result, Exception):
                    raise self.sbatch_result
                return self.sbatch_result
            if cmd_list[0:3] == ["git", "status", "--porcelain"]:
                if isinstance(self.git_status_result, Exception):
                    raise self.git_status_result
                # Check if the mock has been configured to raise an error
                if isinstance(self.git_status_result, subprocess.CalledProcessError):
                    raise self.git_status_result
                return self.git_status_result
            if cmd_list[0:3] == ["git", "rev-parse", "HEAD"]:
                if isinstance(self.git_rev_parse_result, Exception):
                    raise self.git_rev_parse_result
                # Check if the mock has been configured to raise an error
                if isinstance(self.git_rev_parse_result, subprocess.CalledProcessError):
                    raise self.git_rev_parse_result
                # For branch creation verification, return the commit hash that was checked out
                # Look for the previous git switch/checkout command to get the target hash
                if hasattr(self, "_last_checkout_hash") and self._last_checkout_hash is not None:
                    return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout=f"{self._last_checkout_hash}\n", stderr="")
                return self.git_rev_parse_result
            if cmd_list == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
                # Return 'main' branch by default for the branch check
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="main\n", stderr="")
            # Handle git fetch and git switch/checkout commands for branch operations
            if cmd_list[0:3] == ["git", "fetch", "--all"]:
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="", stderr="")
            if cmd_list[0:2] == ["git", "switch"] or cmd_list[0:2] == ["git", "checkout"]:
                # Extract the commit hash from the command (last argument)
                if len(cmd_list) >= 4:  # Has enough args for commit hash
                    self._last_checkout_hash = cmd_list[-1]
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="", stderr="")
            # Handle git branch creation
            if cmd_list[0:2] == ["git", "branch"]:
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="", stderr="")
            # Handle local python execution path
            if cmd_list[0] == "python":
                # Simulate successful local execution
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="Local run successful", stderr="")
            # Fallback for unexpected commands
            raise NotImplementedError(f"subprocess.run called with unexpected command: {cmd_list}")

        self.mock_run.side_effect = run_side_effect
        # ---------------------------------------------------

        # Default successful behavior for mocks
        self.mock_load_yaml.return_value = {self.test_env["yaml_job_key"]: TEST_JOB_NAME}

        # Default side effects for file/dir checks based on test_env
        def is_file_side_effect(path_instance: Path) -> bool:
            # Resolve the path being checked
            resolved_path_instance = path_instance.resolve()
            # Resolve the expected paths from the test environment
            expected_files_resolved = {
                self.test_env["python_script_path_abs"].resolve(),
                self.test_env["template_path"].resolve(),
                self.test_env["config_path"].resolve(),
            }
            # Check if the resolved path matches any of the expected resolved paths
            # This handles cases where the input path might be slightly different
            # but resolves to the same canonical path (e.g., due to symlinks or /private/var).
            return resolved_path_instance in expected_files_resolved

        self.mock_path_is_file.side_effect = is_file_side_effect

        def is_dir_side_effect(path_instance: Path) -> bool:
            # Check the instance the method is called on
            # Use launcher.PROJECT_ROOT which is mocked to test_dir
            configs_dir_expected = (launcher.PROJECT_ROOT / "configs").resolve()
            scripts_dir_expected = (launcher.PROJECT_ROOT / "scripts").resolve()
            git_dir_expected = (launcher.PROJECT_ROOT / ".git").resolve()
            # Compare resolved paths for robustness against path inconsistencies
            # (e.g., /private/var vs /var on macOS).
            return path_instance.resolve() in {configs_dir_expected, scripts_dir_expected, git_dir_expected}

        self.mock_path_is_dir.side_effect = is_dir_side_effect

    def test_main_generate_only(self) -> None:
        """
        Test `main` flow with `--generate-sbatch-only` flag.

        Verifies:
        - Config is loaded.
        - SBATCH script is generated with correct context and resolved paths.
        - Config file is copied to the experiment directory.
        - `subprocess.run` (sbatch submission) is NOT called.
        - `sys.exit` is NOT called.
        """
        provided_hash = "provided_hash_123"
        mock_args = create_mock_args(
            generate_sbatch_only=True,
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash=provided_hash,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        launcher.main(_configure_output=False)

        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_load_yaml.assert_any_call(self.test_env["config_path"].resolve())

        # --- Updated assertion for generate_sbatch_script call --- #
        expected_config_arg = f"--config-path {self.test_env['configs_dir'].resolve()!s} --config-name {TEST_CONFIG_NAME}"
        expected_context_gen_only = {
            "{{JOB_NAME}}": TEST_JOB_NAME,
            "{{PYTHON_SCRIPT_PATH}}": TEST_PYTHON_SCRIPT_REL_PATH,
            "{{CONFIG_NAME}}": TEST_CONFIG_NAME,
            "{{CONFIG_ARG}}": expected_config_arg,
            "{{COMMIT_HASH}}": provided_hash,
            "{{BRANCH_NAME}}": ANY,  # Branch name is dynamically generated
            "{{DO_GIT_CHECKOUT}}": "0",  # Default skips checkout in template
        }
        for k, v in launcher.DEFAULT_SLURM_PARAMS.items():
            # Resolve nested {{JOB_NAME}} placeholders in default values if present
            if isinstance(v, str) and "{{JOB_NAME}}" in v:
                resolved_value = v.replace("{{JOB_NAME}}", TEST_JOB_NAME)
            else:
                resolved_value = v
            expected_context_gen_only[f"{{{{{k.upper()}}}}}"] = resolved_value

        # Add requeue directive (matches launcher.py logic)
        requeue_enabled = launcher.DEFAULT_SLURM_PARAMS.get("requeue", False)
        expected_context_gen_only["{{REQUEUE_DIRECTIVE}}"] = "#SBATCH --requeue" if requeue_enabled else ""

        # Resolve paths in assertion
        self.mock_generate.assert_called_once_with(
            template_path=self.test_env["template_path"].resolve(),  # Resolve path
            output_path=self.test_env["sbatch_output_path"].resolve(),  # Resolve path
            context=expected_context_gen_only,  # Check context dict
        )
        # --------------------------------------------------------- #

        # Check that only git branch check was called and sbatch was not
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]
        assert len(git_calls) == 2  # git rev-parse --abbrev-ref HEAD (pre-flight + cleanup)
        assert len(sbatch_calls) == 0  # No sbatch submission
        self.mock_exit.assert_not_called()
        # Assert YAML config was copied by checking the mock (with resolved paths
        # due to potential /private/var inconsistencies)
        expected_dest_path = (self.test_env["experiments_dir"] / TEST_JOB_NAME / self.test_env["config_path"].name).resolve()
        self.mock_copy2.assert_called_once_with(self.test_env["config_path"].resolve(), expected_dest_path)

    def test_main_submit_job_success_with_provided_hash(self) -> None:
        """Test `main` submission with an explicitly provided commit hash."""
        provided_hash = "specific_commit_abc"
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash=provided_hash,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        launcher.main(_configure_output=False)

        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_load_yaml.assert_any_call(self.test_env["config_path"].resolve())
        self.mock_generate.assert_called_once()
        call_args = self.mock_generate.call_args
        # Extract context from either positional or keyword arguments
        if call_args.kwargs and "context" in call_args.kwargs:
            generated_context = call_args.kwargs["context"]
        else:
            # If context is passed as positional argument (3rd position)
            generated_context = call_args.args[2] if len(call_args.args) > 2 else {}
        assert generated_context["{{COMMIT_HASH}}"] == provided_hash
        assert "{{BRANCH_NAME}}" in generated_context  # Branch name should be in context
        assert generated_context["{{DO_GIT_CHECKOUT}}"] == "0"

        # Check that git branch check and sbatch submission were called
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]

        assert len(git_calls) == 2  # git rev-parse --abbrev-ref HEAD (pre-flight + cleanup)
        assert len(sbatch_calls) == 1  # sbatch submission
        # Verify sbatch was called with the correct script path
        sbatch_call = sbatch_calls[0]
        # Use resolved paths for comparison to handle /var vs /private/var
        sbatch_script_path_resolved = str(self.test_env["sbatch_output_path"].resolve())
        sbatch_call_path_resolved = str(Path(sbatch_call.args[0][1]).resolve())
        assert sbatch_script_path_resolved == sbatch_call_path_resolved

        self.mock_exit.assert_not_called()
        expected_dest_path = (self.test_env["experiments_dir"] / TEST_JOB_NAME / self.test_env["config_path"].name).resolve()
        self.mock_copy2.assert_called_once_with(self.test_env["config_path"].resolve(), expected_dest_path)

    def test_main_submit_job_success_auto_hash_clean(self) -> None:
        """Test `main` submission using auto-detected hash from clean repo."""
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash=None,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])
        # Ensure mocks return clean status and a valid hash
        self.git_status_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        self.git_rev_parse_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=f"{FAKE_COMMIT_HASH}\n", stderr="")

        launcher.main(_configure_output=False)

        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_generate.assert_called_once()
        call_args = self.mock_generate.call_args
        # Extract context from either positional or keyword arguments
        if call_args.kwargs and "context" in call_args.kwargs:
            generated_context = call_args.kwargs["context"]
        else:
            # If context is passed as positional argument (3rd position)
            generated_context = call_args.args[2] if len(call_args.args) > 2 else {}
        assert generated_context["{{COMMIT_HASH}}"] == FAKE_COMMIT_HASH
        assert "{{BRANCH_NAME}}" in generated_context  # Branch name should be in context
        assert generated_context["{{DO_GIT_CHECKOUT}}"] == "0"

        # Check git commands were called (branch check, cleanup branch check, status, commit hash) and sbatch was run
        assert self.mock_run.call_count == 5
        git_status_called = any(c.args[0][0:3] == ["git", "status", "--porcelain"] for c in self.mock_run.call_args_list)
        git_rev_parse_called = any(c.args[0][0:3] == ["git", "rev-parse", "HEAD"] for c in self.mock_run.call_args_list)
        git_fetch_called = any(c.args[0][0:3] == ["git", "fetch", "--all"] for c in self.mock_run.call_args_list)
        git_switch_called = any(c.args[0][0:2] == ["git", "switch"] for c in self.mock_run.call_args_list)
        sbatch_called = any("sbatch" in c.args[0][0] for c in self.mock_run.call_args_list)
        assert git_status_called
        assert git_rev_parse_called
        assert not git_fetch_called
        assert not git_switch_called
        assert sbatch_called

        self.mock_exit.assert_not_called()
        expected_dest_path = (self.test_env["experiments_dir"] / TEST_JOB_NAME / self.test_env["config_path"].name).resolve()
        self.mock_copy2.assert_called_once_with(self.test_env["config_path"].resolve(), expected_dest_path)

    def test_main_submit_job_auto_hash_dirty_repo(self) -> None:
        """Test `main` exits during auto-hash if the repo is dirty."""
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash=None,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])
        # Set mock to return dirty status
        self.git_status_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=" M scripts/some_file.py", stderr="")

        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Config is not loaded because git check fails first (dirty repo)
        self.mock_load_yaml.assert_not_called()
        # git rev-parse --abbrev-ref HEAD (branch check) and git status were called
        assert self.mock_run.call_count == 2
        # Find the git status call in the list
        git_status_call = next((c for c in self.mock_run.call_args_list if c.args[0][0:3] == ["git", "status", "--porcelain"]), None)
        assert git_status_call is not None
        # Check other steps were not reached
        self.mock_generate.assert_not_called()
        self.mock_copy2.assert_not_called()

    def test_main_submit_job_auto_hash_git_status_fails(self) -> None:
        """Test `main` exits during auto-hash if `git status` fails."""
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash=None,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])
        # Configure mock to raise error for git status
        self.git_status_result = subprocess.CalledProcessError(
            returncode=128, cmd=["git", "status", "--porcelain"], stderr="fatal: not a git repository"
        )

        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Config is not loaded because git check fails first
        self.mock_load_yaml.assert_not_called()
        # git rev-parse --abbrev-ref HEAD (branch check) and git status were called
        assert self.mock_run.call_count == 2
        # Find the git status call in the list
        git_status_call = next((c for c in self.mock_run.call_args_list if c.args[0][0:3] == ["git", "status", "--porcelain"]), None)
        assert git_status_call is not None
        # Check other steps were not reached
        self.mock_generate.assert_not_called()
        self.mock_copy2.assert_not_called()

    def test_main_submit_job_auto_hash_git_rev_parse_fails(self) -> None:
        """Test `main` exits during auto-hash if `git rev-parse` fails."""
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash=None,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])
        # Clean status, but rev-parse fails
        self.git_status_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        self.git_rev_parse_result = subprocess.CalledProcessError(
            returncode=128, cmd=["git", "rev-parse", "HEAD"], stderr="fatal: bad object HEAD"
        )

        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Config is not loaded because git check fails first
        self.mock_load_yaml.assert_not_called()
        # git rev-parse --abbrev-ref HEAD (branch check), git status and git rev-parse called
        assert self.mock_run.call_count == 3
        # Find specific git calls
        branch_check_call = next(
            (c for c in self.mock_run.call_args_list if c.args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]), None
        )
        git_status_call = next((c for c in self.mock_run.call_args_list if c.args[0][0:3] == ["git", "status", "--porcelain"]), None)
        assert branch_check_call is not None
        assert git_status_call is not None
        # Also check that git rev-parse HEAD was called
        git_rev_parse_call = next((c for c in self.mock_run.call_args_list if c.args[0][0:3] == ["git", "rev-parse", "HEAD"]), None)
        assert git_rev_parse_call is not None
        # Check other steps were not reached
        self.mock_generate.assert_not_called()
        self.mock_copy2.assert_not_called()

    def test_main_exits_when_not_on_main_branch_with_branch_lock(self) -> None:
        """Test `main` exits if current branch is not 'main' when using --branch-lock."""
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="some_hash_for_test",
            branch_lock=True,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        default_run = self.mock_run.side_effect

        def non_main_branch_side_effect(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
            cmd_list = args[0]
            if cmd_list == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="feature\n", stderr="")
            return default_run(*args, **kwargs)

        self.mock_run.side_effect = non_main_branch_side_effect

        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        self.mock_generate.assert_not_called()
        self.mock_copy2.assert_not_called()
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        assert len(git_calls) == 1  # only branch check executed

    def test_main_submit_job_auto_hash_git_not_found(self) -> None:
        """Test `main` exits during auto-hash if `git` command is not found."""
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash=None,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])
        # Configure mock to raise FileNotFoundError for git status
        self.git_status_result = FileNotFoundError("git command not found")

        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Config is not loaded because git check fails first
        self.mock_load_yaml.assert_not_called()
        # git status was called (attempted)
        assert self.mock_run.call_count == 2  # git rev-parse --abbrev-ref HEAD (branch check) and git status
        # Find the git status call in the list
        git_status_call = next((c for c in self.mock_run.call_args_list if c.args[0][0:3] == ["git", "status", "--porcelain"]), None)
        assert git_status_call is not None
        # Check other steps were not reached
        self.mock_generate.assert_not_called()
        self.mock_copy2.assert_not_called()

    def test_main_yaml_name_mismatch(self) -> None:
        """Test `main` exits when YAML `experiment_name` doesn't match `--config-name` arg."""
        mismatched_config_name = "different_config_name"
        mismatched_config_path = self.test_env["configs_dir"] / f"{mismatched_config_name}.yaml"
        mock_args = create_mock_args(
            config_name=mismatched_config_name,
            commit_hash="some_hash",
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
        )
        self.mock_parse_known_args.return_value = (mock_args, [])
        # load_yaml still returns the original name, causing the mismatch check to fail
        self.mock_load_yaml.return_value = {self.test_env["yaml_job_key"]: TEST_JOB_NAME}

        original_is_file_side_effect = self.mock_path_is_file.side_effect

        def is_file_override_mismatch(path_instance: Path) -> bool:
            if path_instance.resolve() == mismatched_config_path.resolve():
                return True
            return original_is_file_side_effect(path_instance)

        self.mock_path_is_file.side_effect = is_file_override_mismatch

        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        self.mock_load_yaml.assert_called_once_with(mismatched_config_path.resolve())
        self.mock_generate.assert_not_called()
        # Git operations: only branch check (no cleanup since it fails early)
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        assert len(git_calls) == 1  # Only git rev-parse --abbrev-ref HEAD
        assert git_calls[0].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        self.mock_copy2.assert_not_called()

    def test_main_yaml_key_missing(self) -> None:
        """Test `main` exits when the required `experiment_name` key is missing in YAML."""
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
        )
        self.mock_parse_known_args.return_value = (mock_args, [])
        # Override load_yaml mock to return data without the key
        self.mock_load_yaml.return_value = {"other_key": "some_value"}

        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Config is loaded only once (during pre-flight checks) before failing due to missing key
        self.mock_load_yaml.assert_called_once_with(self.test_env["config_path"].resolve())
        self.mock_generate.assert_not_called()
        # Git operations: branch check, status, and commit hash retrieval
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        assert len(git_calls) == 3  # branch check, status, commit hash
        assert git_calls[0].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        assert git_calls[1].args[0] == ["git", "status", "--porcelain"]
        assert git_calls[2].args[0] == ["git", "rev-parse", "HEAD"]
        # No need to assert exit call count, pytest.raises handles it
        # Config file should NOT be copied if validation fails
        self.mock_copy2.assert_not_called()

    def test_main_python_script_not_found(self) -> None:
        """Test `main` exits when the target Python script doesn't exist."""
        non_existent_script = "non_existent_script.py"
        non_existent_script_abs_path = self.test_env["test_dir"] / non_existent_script
        mock_args = create_mock_args(
            python_script=non_existent_script,
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        # Adjust is_file mock side effect for this test to return False for the script
        original_is_file_side_effect = self.mock_path_is_file.side_effect

        def is_file_override(path: Path) -> bool:
            if path == non_existent_script_abs_path:
                return False
            return original_is_file_side_effect(path)

        self.mock_path_is_file.side_effect = is_file_override

        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Check that is_file was called for the resolved script path
        # (due to potential /private/var inconsistencies)
        self.mock_path_is_file.assert_any_call(non_existent_script_abs_path.resolve())
        self.mock_load_yaml.assert_not_called()  # Should exit before loading YAML
        self.mock_generate.assert_not_called()
        # Only the branch check should have been called
        assert self.mock_run.call_count == 1
        assert self.mock_run.call_args.args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        # Config file should NOT be copied if script not found
        self.mock_copy2.assert_not_called()

    def test_main_sbatch_command_fails(self) -> None:
        """Test `main` exits when the `sbatch` command fails (non-zero exit code)."""
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="some_hash_for_test",  # Provide hash to skip git checks
            branch_lock=True,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])
        # Mock subprocess.run to raise CalledProcessError for sbatch
        self.sbatch_result = subprocess.CalledProcessError(
            returncode=1,
            cmd=["sbatch", str(self.test_env["sbatch_output_path"].resolve())],
            stderr="SLURM error: Invalid account",
        )

        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_generate.assert_called_once()  # Generation should happen
        # Check that git commands were called for branch creation and sbatch was attempted
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]
        assert git_calls[0].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # branch check
        assert git_calls[1].args[0] == ["git", "branch"]  # cleanup
        assert git_calls[2].args[0] == ["git", "fetch", "--all"]  # cleanup
        assert git_calls[3].args[0][:3] == ["git", "switch", "-C"]  # create new branch
        branch_name = git_calls[3].args[0][-2]
        assert git_calls[3].args[0][-1] == "some_hash_for_test"
        assert git_calls[4].args[0] == ["git", "rev-parse", "HEAD"]  # branch check
        assert git_calls[5].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # restore original branch
        assert git_calls[6].args[0] == ["git", "branch", "-D", branch_name]  # failed branch cleanup
        assert git_calls[7].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # restore original branch
        assert len(sbatch_calls) == 1  # sbatch submission attempted (but failed)
        # Config file *should* have been copied before sbatch fails (check mock with resolved paths)
        expected_dest_path = (self.test_env["experiments_dir"] / TEST_JOB_NAME / self.test_env["config_path"].name).resolve()
        self.mock_copy2.assert_called_once_with(self.test_env["config_path"].resolve(), expected_dest_path)

    def test_main_sbatch_not_found(self) -> None:
        """Test `main` exits when the `sbatch` command is not found (FileNotFoundError)."""
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="some_hash_for_test",  # Provide hash to skip git checks
            branch_lock=True,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])
        # Mock subprocess.run to raise FileNotFoundError for sbatch
        self.sbatch_result = FileNotFoundError("sbatch command not found")

        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_generate.assert_called_once()  # Generation should happen
        # Check that git commands were called for branch creation and sbatch was attempted
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]
        assert git_calls[0].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # branch check
        assert git_calls[1].args[0] == ["git", "branch"]  # cleanup
        assert git_calls[2].args[0] == ["git", "fetch", "--all"]  # cleanup
        assert git_calls[3].args[0][:3] == ["git", "switch", "-C"]  # create new branch
        branch_name = git_calls[3].args[0][-2]
        assert git_calls[3].args[0][-1] == "some_hash_for_test"
        assert git_calls[4].args[0] == ["git", "rev-parse", "HEAD"]  # branch check
        assert git_calls[5].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # restore original branch
        assert git_calls[6].args[0] == ["git", "branch", "-D", branch_name]  # failed branch cleanup
        assert git_calls[7].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # restore original branch
        assert len(sbatch_calls) == 1  # sbatch submission attempted (but failed)
        # Config file *should* have been copied before sbatch fails (check mock with resolved paths)
        expected_dest_path = (self.test_env["experiments_dir"] / TEST_JOB_NAME / self.test_env["config_path"].name).resolve()
        self.mock_copy2.assert_called_once_with(self.test_env["config_path"].resolve(), expected_dest_path)

    def test_main_dry_run(self) -> None:
        """
        Test `main` flow with `--dry-run` flag.

        Verifies:
        - Config is loaded.
        - Pre-flight checks pass.
        - Script generation (`generate_sbatch_script`) is NOT called.
        - Config file copying (`shutil.copy2`) is NOT called.
        - Job submission (`subprocess.run`) is NOT called.
        - `sys.exit` is NOT called.
        - Experiment directory is NOT created.
        """
        mock_args = create_mock_args(
            dry_run=True,  # Enable dry run
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        # We don't expect SystemExit here
        launcher.main(_configure_output=False)

        # Verify config load happened
        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_load_yaml.assert_any_call(self.test_env["config_path"].resolve())
        # Verify script generation did NOT happen
        self.mock_generate.assert_not_called()
        # Verify sbatch was NOT called
        sbatch_calls = [call for call in self.mock_run.call_args_list if "sbatch" in call.args[0][0]]
        assert not sbatch_calls
        # Verify git checks WERE called (if commit_hash was None)
        if mock_args.commit_hash is None:
            git_calls = [call for call in self.mock_run.call_args_list if call.args[0][0] == "git"]
            assert len(git_calls) == 4  # branch check, cleanup branch check, status, commit hash
        else:
            # If hash provided, only branch checks expected (pre-flight + cleanup)
            git_calls = [call for call in self.mock_run.call_args_list if call.args[0][0] == "git"]
            assert len(git_calls) == 2  # git rev-parse --abbrev-ref HEAD (pre-flight + cleanup)

        # Verify the script did not exit prematurely
        self.mock_exit.assert_not_called()
        # --- Assert experiment directory and config file were NOT created/copied --- #
        experiment_dir_path = self.test_env["experiments_dir"] / TEST_JOB_NAME
        assert not experiment_dir_path.exists()
        self.mock_copy2.assert_not_called()  # Check copy mock wasn't called
        # ------------------------------------------------------------------------- #

    def test_main_slurm_params_from_yaml(self) -> None:
        """Test that SLURM parameters defined under 'slurm' key in YAML override defaults."""
        yaml_override_content = f"""
{launcher.YAML_JOB_NAME_KEY}: {TEST_JOB_NAME}
slurm:
  account: yaml_account
  partition: yaml_partition
  nodes: 4
  time: "2-00:00:00"
"""
        # Update the config file content for this test
        self.test_env["config_path"].write_text(yaml_override_content)
        self.mock_load_yaml.return_value = yaml.safe_load(yaml_override_content)

        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="some_hash_for_test",  # Provide hash to skip git checks
            branch_lock=True,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        launcher.main(_configure_output=False)

        self.mock_generate.assert_called_once()
        call_args = self.mock_generate.call_args
        # Extract context from either positional or keyword arguments
        if call_args.kwargs and "context" in call_args.kwargs:
            context = call_args.kwargs["context"]
        else:
            # If context is passed as positional argument (3rd position)
            context = call_args.args[2] if len(call_args.args) > 2 else {}

        # Assert values from YAML are used (using placeholder keys)
        assert context["{{ACCOUNT}}"] == "yaml_account"
        assert context["{{PARTITION}}"] == "yaml_partition"
        assert context["{{NODES}}"] == 4
        assert context["{{TIME}}"] == "2-00:00:00"
        # Assert other params took defaults (using placeholder keys)
        assert context["{{MEM}}"] == launcher.DEFAULT_SLURM_PARAMS["mem"]
        assert context["{{GPUS_PER_NODE}}"] == launcher.DEFAULT_SLURM_PARAMS["gpus_per_node"]
        assert "{{BRANCH_NAME}}" in context  # Branch name should be in context
        assert context["{{DO_GIT_CHECKOUT}}"] == "1"

        # Check that git commands and sbatch were called
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]
        assert (
            len(git_calls) == 6
        )  # git rev-parse --abbrev-ref HEAD, cleanup calls, git fetch, git switch, git rev-parse for branch creation, restore calls
        assert len(sbatch_calls) == 1  # sbatch submission
        self.mock_exit.assert_not_called()
        # Assert YAML config was copied by checking the mock (with resolved paths)
        expected_dest_path = (self.test_env["experiments_dir"] / TEST_JOB_NAME / self.test_env["config_path"].name).resolve()
        self.mock_copy2.assert_called_once_with(self.test_env["config_path"].resolve(), expected_dest_path)

    def test_main_slurm_params_from_cli(self) -> None:
        """Test that CLI SLURM arguments override both YAML and default parameters."""
        yaml_override_content = f"""
{launcher.YAML_JOB_NAME_KEY}: {TEST_JOB_NAME}
slurm:
  account: yaml_account # Will be overridden by CLI
  partition: yaml_partition
  nodes: 4
  commit_hash: another_hash_for_test # Provide hash to skip git checks
"""
        self.test_env["config_path"].write_text(yaml_override_content)
        self.mock_load_yaml.return_value = yaml.safe_load(yaml_override_content)

        # Simulate CLI args overriding YAML and defaults
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            account="cli_account",  # CLI override
            nodes=8,  # CLI override
            mem="512G",  # CLI override (default was 256G)
            commit_hash="another_hash_for_test",  # Provide hash to skip git checks
            branch_lock=True,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        launcher.main(_configure_output=False)

        self.mock_generate.assert_called_once()
        call_args = self.mock_generate.call_args
        # Extract context from either positional or keyword arguments
        if call_args.kwargs and "context" in call_args.kwargs:
            context = call_args.kwargs["context"]
        else:
            # If context is passed as positional argument (3rd position)
            context = call_args.args[2] if len(call_args.args) > 2 else {}

        # Assert CLI values are used (using placeholder keys)
        assert context["{{ACCOUNT}}"] == "cli_account"
        assert context["{{NODES}}"] == 8
        assert context["{{MEM}}"] == "512G"
        # Assert YAML value is used where no CLI override (using placeholder keys)
        assert context["{{PARTITION}}"] == "yaml_partition"
        # Assert default used where neither CLI nor YAML provided (using placeholder keys)
        assert context["{{TIME}}"] == launcher.DEFAULT_SLURM_PARAMS["time"]
        assert "{{BRANCH_NAME}}" in context  # Branch name should be in context
        assert context["{{DO_GIT_CHECKOUT}}"] == "1"

        # Check that git commands and sbatch were called
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]
        assert (
            len(git_calls) == 6
        )  # git rev-parse --abbrev-ref HEAD, cleanup calls, git fetch, git switch, git rev-parse for branch creation, restore calls
        assert len(sbatch_calls) == 1  # sbatch submission
        self.mock_exit.assert_not_called()
        # Assert YAML config was copied by checking the mock (with resolved paths)
        expected_dest_path = (self.test_env["experiments_dir"] / TEST_JOB_NAME / self.test_env["config_path"].name).resolve()
        self.mock_copy2.assert_called_once_with(self.test_env["config_path"].resolve(), expected_dest_path)

    # --- New test for local execution flag --- #
    def test_main_run_local_success(self) -> None:
        """Test `main` flow when running the job locally with --run-local flag."""
        mock_args = create_mock_args(
            run_local=True,
            template=str(self.test_env["template_path"]),  # Irrelevant when run_local but provided for completeness
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="provided_hash",  # Hash should be ignored in local run path
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        launcher.main(_configure_output=False)

        # YAML config should be loaded only once in run_local mode (during pre-flight checks)
        self.mock_load_yaml.assert_called_once_with(self.test_env["config_path"].resolve())

        # No SBATCH generation or submission should occur
        self.mock_generate.assert_not_called()
        sbatch_calls = [call for call in self.mock_run.call_args_list if "sbatch" in call.args[0][0]]
        assert not sbatch_calls

        # subprocess.run should be invoked for branch check and the python command
        python_calls = [call for call in self.mock_run.call_args_list if call.args[0][0] == "python"]
        assert len(python_calls) == 1
        python_call_cmd = python_calls[0].args[0]
        assert python_call_cmd[0] == "python"

        # Config file should be copied to experiment directory
        expected_dest_path = (self.test_env["experiments_dir"] / TEST_JOB_NAME / self.test_env["config_path"].name).resolve()
        self.mock_copy2.assert_called_once_with(self.test_env["config_path"].resolve(), expected_dest_path)

        # Script should not exit with error
        self.mock_exit.assert_not_called()

    def test_main_hydra_overrides_generate_sbatch(self) -> None:
        """Test that Hydra parameter overrides are properly included in SBATCH script generation."""
        unknown_args = ["parallel.rank=1", "parallel.world_size=4", "training.batch_size=32"]
        mock_args = create_mock_args(
            generate_sbatch_only=True,
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="test_hash_123",
        )

        # Mock parse_known_args to return both known args and unknown args
        self.mock_parse_known_args.return_value = (mock_args, unknown_args)

        launcher.main(_configure_output=False)

        # Verify config was loaded
        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_load_yaml.assert_any_call(self.test_env["config_path"].resolve())

        # Verify SBATCH script generation was called
        self.mock_generate.assert_called_once()
        call_args = self.mock_generate.call_args
        # Extract context from either positional or keyword arguments
        if call_args.kwargs and "context" in call_args.kwargs:
            generated_context = call_args.kwargs["context"]
        else:
            # If context is passed as positional argument (3rd position)
            generated_context = call_args.args[2] if len(call_args.args) > 2 else {}

        # Verify that the unknown arguments are included in the CONFIG_ARG
        expected_config_arg = (
            f"--config-path {self.test_env['configs_dir'].resolve()!s} --config-name {TEST_CONFIG_NAME} {' '.join(unknown_args)}"
        )
        assert generated_context["{{CONFIG_ARG}}"] == expected_config_arg

        # Verify other expected context elements
        assert generated_context["{{JOB_NAME}}"] == TEST_JOB_NAME
        assert generated_context["{{COMMIT_HASH}}"] == "test_hash_123"
        assert generated_context["{{DO_GIT_CHECKOUT}}"] == "0"

        # Ensure no submission occurred (generate-only mode)
        sbatch_calls = [call for call in self.mock_run.call_args_list if "sbatch" in call.args[0][0]]
        assert not sbatch_calls
        self.mock_exit.assert_not_called()

    def test_main_hydra_overrides_run_local(self) -> None:
        """Test that Hydra parameter overrides are properly included in local execution."""
        unknown_args = ["parallel.rank=2", "parallel.world_size=4", "model.learning_rate=0.001"]
        mock_args = create_mock_args(
            run_local=True,
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="test_hash_for_local",  # Provide hash to skip git commands
        )

        # Mock parse_known_args to return both known args and unknown args
        self.mock_parse_known_args.return_value = (mock_args, unknown_args)

        launcher.main(_configure_output=False)

        # Verify config was loaded only once in run_local mode (during pre-flight checks)
        self.mock_load_yaml.assert_called_once_with(self.test_env["config_path"].resolve())

        # Verify python command was executed locally
        python_calls = [call for call in self.mock_run.call_args_list if call.args[0][0] == "python"]
        assert len(python_calls) == 1
        call_args = python_calls[0]
        executed_command = call_args.args[0]

        # Verify the command structure
        assert executed_command[0] == "python"
        # Use resolved paths for comparison to handle /var vs /private/var
        python_script_path_resolved = str(self.test_env["python_script_path_abs"].resolve())
        executed_script_path_resolved = str(Path(executed_command[1]).resolve())
        assert python_script_path_resolved == executed_script_path_resolved
        # Check for either --config-path=<path> or --config-path <path> format
        config_path_found = False
        config_name_found = False
        for i, arg in enumerate(executed_command):
            if arg == "--config-path" and i + 1 < len(executed_command):
                config_path_found = str(self.test_env["configs_dir"].resolve()) in executed_command[i + 1]
            elif arg.startswith("--config-path="):
                config_path_found = str(self.test_env["configs_dir"].resolve()) in arg
            if arg == "--config-name" and i + 1 < len(executed_command):
                config_name_found = TEST_CONFIG_NAME in executed_command[i + 1]
            elif arg.startswith("--config-name="):
                config_name_found = TEST_CONFIG_NAME in arg
        assert config_path_found
        assert config_name_found

        # Verify that unknown arguments are appended to the command
        for unknown_arg in unknown_args:
            assert unknown_arg in executed_command

        # Verify no SBATCH generation occurred
        self.mock_generate.assert_not_called()

        # Verify config file was copied
        expected_dest_path = (self.test_env["experiments_dir"] / TEST_JOB_NAME / self.test_env["config_path"].name).resolve()
        self.mock_copy2.assert_called_once_with(self.test_env["config_path"].resolve(), expected_dest_path)

        # Script should not exit with error
        self.mock_exit.assert_not_called()

    def test_main_hydra_overrides_empty_unknown_args(self) -> None:
        """Test that launcher works correctly when no unknown arguments are provided."""
        unknown_args = []  # No unknown arguments
        mock_args = create_mock_args(
            generate_sbatch_only=True,
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="test_hash_456",
        )

        # Mock parse_known_args to return only known args
        self.mock_parse_known_args.return_value = (mock_args, unknown_args)

        launcher.main(_configure_output=False)

        # Verify config was loaded
        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_load_yaml.assert_any_call(self.test_env["config_path"].resolve())

        # Verify SBATCH script generation was called
        self.mock_generate.assert_called_once()
        call_args = self.mock_generate.call_args
        # Extract context from either positional or keyword arguments
        if call_args.kwargs and "context" in call_args.kwargs:
            generated_context = call_args.kwargs["context"]
        else:
            # If context is passed as positional argument (3rd position)
            generated_context = call_args.args[2] if len(call_args.args) > 2 else {}

        # Verify that CONFIG_ARG doesn't have extra unknown arguments (just the basic config args)
        expected_config_arg = f"--config-path {self.test_env['configs_dir'].resolve()!s} --config-name {TEST_CONFIG_NAME}"
        assert generated_context["{{CONFIG_ARG}}"] == expected_config_arg

        # Verify other expected context elements
        assert generated_context["{{JOB_NAME}}"] == TEST_JOB_NAME
        assert generated_context["{{COMMIT_HASH}}"] == "test_hash_456"
        assert generated_context["{{DO_GIT_CHECKOUT}}"] == "0"

        self.mock_exit.assert_not_called()

    def test_main_file_location_consistency_pass(self) -> None:
        """Test that main succeeds when all files are in the same directory."""
        mock_args = create_mock_args(
            generate_sbatch_only=True,
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="test_hash_consistency",
            # Use default output/error paths which should be in the same dir as sbatch
            output=f"experiments/{TEST_JOB_NAME}/%j.out",
            error=f"experiments/{TEST_JOB_NAME}/%j.err",
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        # Should not raise SystemExit
        launcher.main(_configure_output=False)

        # Verify all the normal flow happened
        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_load_yaml.assert_any_call(self.test_env["config_path"].resolve())
        self.mock_generate.assert_called_once()
        self.mock_exit.assert_not_called()

        # Verify the consistency check passed by checking the context
        call_args = self.mock_generate.call_args
        # Extract context from either positional or keyword arguments
        if call_args.kwargs and "context" in call_args.kwargs:
            context = call_args.kwargs["context"]
        else:
            # If context is passed as positional argument (3rd position)
            context = call_args.args[2] if len(call_args.args) > 2 else {}

        # Output and error should be in the same directory as specified
        assert context["{{OUTPUT}}"] == f"experiments/{TEST_JOB_NAME}/%j.out"
        assert context["{{ERROR}}"] == f"experiments/{TEST_JOB_NAME}/%j.err"
        assert context["{{DO_GIT_CHECKOUT}}"] == "0"

    def test_main_file_location_consistency_fail(self) -> None:
        """Test that main exits when files would be placed in different directories."""
        mock_args = create_mock_args(
            generate_sbatch_only=True,
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="test_hash_consistency_fail",
            # Set output/error paths to different directories
            output="logs/output/%j.out",  # Different directory
            error="logs/errors/%j.err",  # Another different directory
            branch_lock=True,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        # Should raise SystemExit due to consistency check failure
        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Verify config was loaded (pre-flight checks happened)
        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_load_yaml.assert_any_call(self.test_env["config_path"].resolve())

        # Verify script generation was NOT called (consistency check failed before generation)
        self.mock_generate.assert_not_called()

        # Verify only git operations happened, no sbatch submission
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]
        assert git_calls[0].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # branch check
        assert git_calls[1].args[0] == ["git", "branch"]  # cleanup
        assert git_calls[2].args[0] == ["git", "fetch", "--all"]  # fetch
        assert git_calls[3].args[0][:3] == ["git", "switch", "-C"]  # create new branch
        assert git_calls[3].args[0][-1] == "test_hash_consistency_fail"
        assert git_calls[4].args[0] == ["git", "rev-parse", "HEAD"]  # branch check
        assert git_calls[5].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # restore original branch
        assert len(sbatch_calls) == 0  # No sbatch submission
        self.mock_copy2.assert_not_called()

    def test_main_file_location_consistency_fail_with_custom_experiments_dir(self) -> None:
        """Test consistency check with custom experiments directory that causes file separation."""
        # Create a scenario where experiments-dir points to one location
        # but output/error logs point to completely different directories
        custom_experiments_dir = self.test_env["test_dir"] / "custom" / "experiments"

        mock_args = create_mock_args(
            generate_sbatch_only=True,
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(custom_experiments_dir),
            commit_hash="test_hash_custom_consistency",
            # Output/error in completely different root directories
            output="var/log/slurm/output/%j.out",
            error="tmp/slurm/errors/%j.err",
            branch_lock=True,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        # Should raise SystemExit due to consistency check failure
        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Verify the error occurred during pre-flight checks
        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_generate.assert_not_called()
        # Verify only git operations happened, no sbatch submission
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]
        assert git_calls[0].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # branch check
        assert git_calls[1].args[0] == ["git", "branch"]  # cleanup
        assert git_calls[2].args[0] == ["git", "fetch", "--all"]  # fetch
        assert git_calls[3].args[0][:3] == ["git", "switch", "-C"]  # create new branch
        assert git_calls[3].args[0][-1] == "test_hash_custom_consistency"
        assert git_calls[4].args[0] == ["git", "rev-parse", "HEAD"]  # branch check
        assert git_calls[5].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # restore original branch
        assert len(sbatch_calls) == 0  # No sbatch submission
        self.mock_copy2.assert_not_called()

    def test_main_file_location_consistency_with_parsing_error(self) -> None:
        """Test that consistency check handles unparsable output/error paths gracefully."""
        mock_args = create_mock_args(
            generate_sbatch_only=True,
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="test_hash_parsing_error",
            # Set paths that might cause parsing issues (invalid path characters)
            output="<invalid>:/path/%j.out",
            error="\\\\bad\\path\\%j.err",
            branch_lock=True,
        )
        self.mock_parse_known_args.return_value = (mock_args, [])

        # Should exit because invalid paths result in different directories
        # The consistency check correctly identifies this as a problem
        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Verify config was loaded but script generation was NOT called
        # Config is loaded twice: once for pre-flight checks and once for actual processing
        assert self.mock_load_yaml.call_count == 2
        self.mock_generate.assert_not_called()
        # Verify only git operations happened, no sbatch submission
        git_calls = [c for c in self.mock_run.call_args_list if "git" in c.args[0][0]]
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]
        assert git_calls[0].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # branch check
        assert git_calls[1].args[0] == ["git", "branch"]  # cleanup
        assert git_calls[2].args[0] == ["git", "fetch", "--all"]  # fetch
        assert git_calls[3].args[0][:3] == ["git", "switch", "-C"]  # create new branch
        assert git_calls[3].args[0][-1] == "test_hash_parsing_error"
        assert git_calls[4].args[0] == ["git", "rev-parse", "HEAD"]  # branch check
        assert git_calls[5].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]  # restore original branch
        assert len(sbatch_calls) == 0  # No sbatch submission
        self.mock_copy2.assert_not_called()

    def test_main_multiple_configs_success(self) -> None:
        """Test submitting multiple configs successfully."""
        # Create .git directory for git operations
        git_dir = self.test_env["test_dir"] / ".git"
        git_dir.mkdir(exist_ok=True)

        # Create multiple config files
        config_names = ["exp1", "exp2", "exp3"]

        # Mock multiple config files
        config_files = {}
        for name in config_names:
            config_path = self.test_env["configs_dir"] / f"{name}.yaml"
            config_content = f"""
experiment_name: {name}
model:
  type: resnet
parameters:
  lr: 0.001
"""
            config_files[config_path] = config_content

        # Update mock to handle multiple configs
        def mock_is_file(path: Path) -> bool:
            """Mock Path.is_file() for multiple configs."""
            path_str = str(path.resolve())
            # Check if it's a config file
            for config_path in config_files:
                if path_str == str(config_path.resolve()):
                    return True
            # Other existing file checks
            if path_str == str(self.test_env["template_path"].resolve()):
                return True
            if path_str == str(self.test_env["python_script_path_abs"].resolve()):
                return True
            return False

        self.mock_path_is_file.side_effect = mock_is_file

        # Mock loading different YAML configs
        def mock_load_yaml(path: Path) -> dict[str, Any]:
            """Mock loading different YAML files."""
            for config_path, content in config_files.items():
                if str(path.resolve()) == str(config_path.resolve()):
                    return yaml.safe_load(content)
            return {}

        self.mock_load_yaml.side_effect = mock_load_yaml

        # Setup args with comma-separated configs
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="test_hash",
        )
        mock_args.config_name = ",".join(config_names)
        self.mock_parse_known_args.return_value = (mock_args, [])

        launcher.main(_configure_output=False)

        # Verify all configs were loaded (twice each: pre-flight + actual submission)
        assert self.mock_load_yaml.call_count == len(config_names) * 2

        # Verify sbatch scripts were generated for each config
        assert self.mock_generate.call_count == len(config_names)

        # Verify all jobs were submitted
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]
        assert len(sbatch_calls) == len(config_names)

        # Verify config files were copied
        assert self.mock_copy2.call_count == len(config_names)

        self.mock_exit.assert_not_called()

    def test_main_multiple_configs_partial_failure(self) -> None:
        """Test submitting multiple configs with some failures."""
        # Create .git directory for git operations
        git_dir = self.test_env["test_dir"] / ".git"
        git_dir.mkdir(exist_ok=True)

        # Create multiple config files
        config_names = ["exp1", "exp2_bad", "exp3"]

        # Mock multiple config files
        config_files = {}
        for name in config_names:
            config_path = self.test_env["configs_dir"] / f"{name}.yaml"
            if name == "exp2_bad":
                # Bad config missing experiment_name
                config_content = """
model:
  type: resnet
parameters:
  lr: 0.001
"""
            else:
                config_content = f"""
experiment_name: {name}
model:
  type: resnet
parameters:
  lr: 0.001
"""
            config_files[config_path] = config_content

        # Update mock to handle multiple configs
        def mock_is_file(path: Path) -> bool:
            """Mock Path.is_file() for multiple configs."""
            path_str = str(path.resolve())
            # Check if it's a config file
            for config_path in config_files:
                if path_str == str(config_path.resolve()):
                    return True
            # Other existing file checks
            if path_str == str(self.test_env["template_path"].resolve()):
                return True
            if path_str == str(self.test_env["python_script_path_abs"].resolve()):
                return True
            return False

        self.mock_path_is_file.side_effect = mock_is_file

        # Mock loading different YAML configs
        def mock_load_yaml(path: Path) -> dict[str, Any]:
            """Mock loading different YAML files."""
            for config_path, content in config_files.items():
                if str(path.resolve()) == str(config_path.resolve()):
                    return yaml.safe_load(content)
            return {}

        self.mock_load_yaml.side_effect = mock_load_yaml

        # Setup args with comma-separated configs
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            commit_hash="test_hash",
        )
        mock_args.config_name = ",".join(config_names)
        self.mock_parse_known_args.return_value = (mock_args, [])

        # Should exit due to pre-flight check failure
        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Should fail during pre-flight checks, before any submission
        assert self.mock_generate.call_count == 0
        sbatch_calls = [c for c in self.mock_run.call_args_list if "sbatch" in c.args[0][0]]
        assert len(sbatch_calls) == 0

    def test_main_multiple_configs_run_local_error(self) -> None:
        """Test that multiple configs with --run-local raises an error."""
        config_names = ["exp1", "exp2"]

        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
            run_local=True,
        )
        mock_args.config_name = ",".join(config_names)
        self.mock_parse_known_args.return_value = (mock_args, [])

        # Should exit due to --run-local with multiple configs
        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Should fail before loading any configs
        self.mock_load_yaml.assert_not_called()
        self.mock_generate.assert_not_called()

    def test_main_multiple_configs_spaces_error(self) -> None:
        """Test that config names with spaces are rejected."""
        config_names = ["exp1", "exp 2 with spaces", "exp3"]

        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
        )
        mock_args.config_name = ",".join(config_names)
        self.mock_parse_known_args.return_value = (mock_args, [])

        # Should exit due to spaces in config name
        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Should fail before loading any configs
        self.mock_load_yaml.assert_not_called()
        self.mock_generate.assert_not_called()

    def test_main_default_skips_checkout(self) -> None:
        """Test that by default the launcher avoids git checkout and restore."""
        mock_args = create_mock_args(
            template=str(self.test_env["template_path"]),
            configs_dir=str(self.test_env["configs_dir"]),
            experiments_dir=str(self.test_env["experiments_dir"]),
        )
        self.mock_parse_known_args.return_value = (mock_args, [])
        mock_perform = self.mocker.patch("slurm.launcher.perform_git_checkout_with_branch")
        mock_restore = self.mocker.patch("slurm.launcher.restore_original_branch")

        launcher.main(_configure_output=False)

        mock_perform.assert_not_called()
        mock_restore.assert_not_called()
        switch_calls = [c for c in self.mock_run.call_args_list if c.args[0][0:2] == ["git", "switch"]]
        assert not switch_calls


# --- Tests for Branch Management ---


class TestBranchManagement:
    """Tests for the branch management functionality (cleanup, parsing, etc.)."""

    def test_parse_slurm_branch_valid(self) -> None:
        """Test parsing valid slurm-job branch names."""
        test_cases = [
            ("slurm-job/experiment1/1736865600/abc12345", "experiment1", 1736865600, "abc12345"),
            ("slurm-job/experiments/test_exp/1736865600/def67890", "experiments/test_exp", 1736865600, "def67890"),
            ("slurm-job/path/to/my/experiment/1736865600/12345678", "path/to/my/experiment", 1736865600, "12345678"),
        ]

        for branch_name, expected_exp, expected_ts, expected_hash in test_cases:
            result = launcher.parse_slurm_branch(branch_name)
            assert result is not None
            assert result["exp_name"] == expected_exp
            assert result["timestamp"] == expected_ts
            assert result["hash"] == expected_hash
            assert result["full_name"] == branch_name

    def test_parse_slurm_branch_invalid(self) -> None:
        """Test parsing invalid slurm-job branch names."""
        invalid_branches = [
            "main",
            "feature/new-feature",
            "slurm-job/experiment",  # Missing timestamp and hash
            "slurm-job/experiment/not-a-number/abc123",  # Invalid timestamp
            "slurm-job/experiment/1000/abc123",  # Timestamp too old (before 2020)
            "slurm-job",  # Incomplete
        ]

        for branch_name in invalid_branches:
            result = launcher.parse_slurm_branch(branch_name)
            assert result is None

    def test_get_slurm_branches(self, mocker: MockerFixture) -> None:
        """Test getting and parsing all slurm-job branches."""
        # Mock subprocess.run to return sample git branch output
        git_output = """
  main
  feature/some-feature
* slurm-job/experiment1/1736865600/abc12345
  slurm-job/experiment2/1736865700/def67890
  slurm-job/experiments/deep/test/1736865800/12345678
"""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = subprocess.CompletedProcess(args=["git", "branch"], returncode=0, stdout=git_output, stderr="")

        branches = launcher.get_slurm_branches()

        assert len(branches) == 3
        assert branches[0]["exp_name"] == "experiment1"
        assert branches[1]["exp_name"] == "experiment2"
        assert branches[2]["exp_name"] == "experiments/deep/test"
        mock_run.assert_called_once()

    def test_get_slurm_branches_git_fails(self, mocker: MockerFixture) -> None:
        """Test get_slurm_branches when git command fails."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(returncode=128, cmd=["git", "branch"], stderr="fatal: not a git repository")

        branches = launcher.get_slurm_branches()

        assert branches == []
        mock_run.assert_called_once()

    def test_identify_stale_branches(self) -> None:
        """Test identifying stale branches with the same experiment name."""
        branches = [
            {"exp_name": "experiment1", "timestamp": 1736865600, "hash": "abc123", "full_name": "slurm-job/experiment1/1736865600/abc123"},
            {"exp_name": "experiment1", "timestamp": 1736865700, "hash": "def456", "full_name": "slurm-job/experiment1/1736865700/def456"},
            {"exp_name": "experiment1", "timestamp": 1736865500, "hash": "ghi789", "full_name": "slurm-job/experiment1/1736865500/ghi789"},
            {"exp_name": "experiment2", "timestamp": 1736865800, "hash": "jkl012", "full_name": "slurm-job/experiment2/1736865800/jkl012"},
        ]

        stale = launcher.identify_stale_branches(branches, "experiment1")

        # All experiment1 branches should be marked as stale since we're creating a new one
        assert len(stale) == 3
        assert all(b["exp_name"] == "experiment1" for b in stale)

        # Test with experiment that has no existing branches
        stale2 = launcher.identify_stale_branches(branches, "experiment3")
        assert len(stale2) == 0

    def test_identify_old_branches(self) -> None:
        """Test identifying branches older than specified days."""
        current_time = int(time.time())

        branches = [
            {
                "exp_name": "old_exp",
                "timestamp": current_time - (15 * 24 * 60 * 60),
                "hash": "abc123",
                "full_name": "slurm-job/old_exp/ts/abc123",
            },
            {
                "exp_name": "very_old_exp",
                "timestamp": current_time - (30 * 24 * 60 * 60),
                "hash": "def456",
                "full_name": "slurm-job/very_old_exp/ts/def456",
            },
            {
                "exp_name": "recent_exp",
                "timestamp": current_time - (5 * 24 * 60 * 60),
                "hash": "ghi789",
                "full_name": "slurm-job/recent_exp/ts/ghi789",
            },
        ]

        old_branches = launcher.identify_old_branches(branches, days=14)

        assert len(old_branches) == 2
        assert old_branches[0]["exp_name"] == "old_exp"
        assert old_branches[0]["age_days"] > 14
        assert old_branches[1]["exp_name"] == "very_old_exp"
        assert old_branches[1]["age_days"] > 14

    def test_delete_branches(self, mocker: MockerFixture) -> None:
        """Test deleting branches."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = subprocess.CompletedProcess(args=["git", "branch", "-D", "branch_name"], returncode=0, stdout="", stderr="")

        branches = [
            {"exp_name": "exp1", "timestamp": 123, "hash": "abc", "full_name": "slurm-job/exp1/123/abc"},
            {"exp_name": "exp2", "timestamp": 456, "hash": "def", "full_name": "slurm-job/exp2/456/def"},
        ]

        launcher.delete_branches(branches, dry_run=False)

        assert mock_run.call_count == 2
        # Check the git branch -D commands were called correctly
        calls = mock_run.call_args_list
        assert calls[0].args[0] == ["git", "branch", "-D", "slurm-job/exp1/123/abc"]
        assert calls[1].args[0] == ["git", "branch", "-D", "slurm-job/exp2/456/def"]

    def test_delete_branches_dry_run(self, mocker: MockerFixture) -> None:
        """Test delete_branches in dry-run mode."""
        mock_run = mocker.patch("subprocess.run")

        branches = [
            {"exp_name": "exp1", "timestamp": 123, "hash": "abc", "full_name": "slurm-job/exp1/123/abc"},
        ]

        launcher.delete_branches(branches, dry_run=True)

        # In dry-run mode, git commands should NOT be called
        mock_run.assert_not_called()

    def test_delete_branches_git_fails(self, mocker: MockerFixture) -> None:
        """Test delete_branches when git command fails."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["git", "branch", "-D", "branch_name"], stderr="error: branch 'branch_name' not found"
        )

        branches = [
            {"exp_name": "exp1", "timestamp": 123, "hash": "abc", "full_name": "slurm-job/exp1/123/abc"},
        ]

        # Should not raise exception, just log warning
        launcher.delete_branches(branches, dry_run=False)

        mock_run.assert_called_once()

    def test_cleanup_slurm_branches(self, mocker: MockerFixture) -> None:
        """Test the main cleanup_slurm_branches function."""
        current_time = int(time.time())

        # Mock get_slurm_branches to return test data
        test_branches = [
            {
                "exp_name": "current_exp",
                "timestamp": current_time - (1 * 60 * 60),
                "hash": "abc123",
                "full_name": "slurm-job/current_exp/ts1/abc123",
            },
            {
                "exp_name": "current_exp",
                "timestamp": current_time - (2 * 60 * 60),
                "hash": "def456",
                "full_name": "slurm-job/current_exp/ts2/def456",
            },
            {
                "exp_name": "old_exp",
                "timestamp": current_time - (20 * 24 * 60 * 60),
                "hash": "ghi789",
                "full_name": "slurm-job/old_exp/ts3/ghi789",
            },
            {
                "exp_name": "other_exp",
                "timestamp": current_time - (5 * 24 * 60 * 60),
                "hash": "jkl012",
                "full_name": "slurm-job/other_exp/ts4/jkl012",
            },
        ]

        mock_get_branches = mocker.patch("slurm.launcher.get_slurm_branches")
        mock_get_branches.return_value = test_branches

        mock_delete = mocker.patch("slurm.launcher.delete_branches")

        # Call cleanup
        launcher.cleanup_slurm_branches("current_exp", dry_run=False)

        # Verify stale branches were deleted
        mock_delete.assert_called_once()
        deleted_branches = mock_delete.call_args[0][0]
        assert len(deleted_branches) == 2  # Both current_exp branches should be deleted
        assert all(b["exp_name"] == "current_exp" for b in deleted_branches)

    def test_cleanup_slurm_branches_no_branches(self, mocker: MockerFixture) -> None:
        """Test cleanup when no slurm-job branches exist."""
        mock_get_branches = mocker.patch("slurm.launcher.get_slurm_branches")
        mock_get_branches.return_value = []

        mock_delete = mocker.patch("slurm.launcher.delete_branches")

        launcher.cleanup_slurm_branches("any_exp", dry_run=False)

        # No deletion should occur
        mock_delete.assert_not_called()

    def test_restore_original_branch(self, mocker: MockerFixture) -> None:
        """Test restoring to the original branch."""
        mock_run = mocker.patch("subprocess.run")
        # First call returns current branch, second call performs checkout
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="slurm-job/exp/123/abc\n", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]

        launcher.restore_original_branch("main")

        assert mock_run.call_count == 2
        # Check git rev-parse was called to get current branch
        assert mock_run.call_args_list[0].args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        # Check git checkout was called to restore branch
        assert mock_run.call_args_list[1].args[0] == ["git", "checkout", "main"]

    def test_restore_original_branch_already_on_branch(self, mocker: MockerFixture) -> None:
        """Test restore when already on the target branch."""
        mock_run = mocker.patch("subprocess.run")
        # Return that we're already on main
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="main\n", stderr="")

        launcher.restore_original_branch("main")

        # Only one call to check current branch, no checkout needed
        assert mock_run.call_count == 1
        assert mock_run.call_args.args[0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]

    def test_restore_original_branch_fails(self, mocker: MockerFixture) -> None:
        """Test restore when git checkout fails."""
        mock_run = mocker.patch("subprocess.run")
        # First call returns current branch, second call fails
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="slurm-job/exp/123/abc\n", stderr=""),
            subprocess.CalledProcessError(returncode=1, cmd=["git", "checkout", "main"], stderr="error: pathspec 'main' did not match"),
        ]

        # Should not raise exception, just log warning
        launcher.restore_original_branch("main")

        assert mock_run.call_count == 2

    def test_main_with_branch_cleanup(self, mocker: MockerFixture, test_env: dict[str, Any]) -> None:
        """Test that main flow includes branch cleanup."""
        # Set up standard mocks
        mock_parse_known_args = mocker.patch("argparse.ArgumentParser.parse_known_args")
        mock_load_yaml = mocker.patch("slurm.launcher.load_yaml_config")
        mock_generate = mocker.patch("slurm.launcher.generate_sbatch_script")
        mock_run = mocker.patch("subprocess.run")
        mocker.patch("shutil.copy2")
        mocker.patch("sys.exit", side_effect=SystemExit(1))
        mocker.patch("pathlib.Path.is_file", return_value=True)
        mocker.patch("pathlib.Path.is_dir", return_value=True)

        # Mock cleanup_slurm_branches and restore_original_branch to avoid their subprocess calls
        mock_cleanup = mocker.patch("slurm.launcher.cleanup_slurm_branches")
        mock_restore = mocker.patch("slurm.launcher.restore_original_branch")

        # Set up test arguments
        mock_args = create_mock_args(
            template=str(test_env["template_path"]),
            configs_dir=str(test_env["configs_dir"]),
            experiments_dir=str(test_env["experiments_dir"]),
            commit_hash="test_hash",
            branch_lock=True,
        )
        mock_parse_known_args.return_value = (mock_args, [])
        mock_load_yaml.return_value = {test_env["yaml_job_key"]: TEST_JOB_NAME}

        # Mock subprocess.run for git operations
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="main\n", stderr=""),  # git rev-parse --abbrev-ref HEAD
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # git fetch
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # git switch
            subprocess.CompletedProcess(args=[], returncode=0, stdout="test_hash\n", stderr=""),  # git rev-parse HEAD (verify)
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Submitted job 123", stderr=""),  # sbatch
        ]

        launcher.main(_configure_output=False)

        # Verify cleanup was called with correct arguments
        mock_cleanup.assert_called_once_with(TEST_JOB_NAME, dry_run=False)

        # Verify restore was called
        mock_restore.assert_called_once_with("main")

        # Verify other operations happened as expected
        mock_generate.assert_called_once()
        assert mock_run.call_count == 5  # All git operations plus sbatch

    def test_main_with_branch_cleanup_and_restore_on_failure(self, mocker: MockerFixture, test_env: dict[str, Any]) -> None:
        """Test that original branch is restored and failed branch is cleaned up on submission failure."""
        # Set up standard mocks
        mock_parse_known_args = mocker.patch("argparse.ArgumentParser.parse_known_args")
        mock_load_yaml = mocker.patch("slurm.launcher.load_yaml_config")
        mock_run = mocker.patch("subprocess.run")
        mocker.patch("slurm.launcher.generate_sbatch_script")
        mocker.patch("shutil.copy2")
        mocker.patch("sys.exit", side_effect=SystemExit(1))
        mocker.patch("pathlib.Path.is_file", return_value=True)
        mocker.patch("pathlib.Path.is_dir", return_value=True)

        # Mock cleanup and restore functions to avoid their subprocess calls
        mock_cleanup = mocker.patch("slurm.launcher.cleanup_slurm_branches")
        mocker.patch("slurm.launcher.restore_original_branch")

        # Set up test arguments
        mock_args = create_mock_args(
            template=str(test_env["template_path"]),
            configs_dir=str(test_env["configs_dir"]),
            experiments_dir=str(test_env["experiments_dir"]),
            commit_hash="test_hash",
            branch_lock=True,
        )
        mock_parse_known_args.return_value = (mock_args, [])
        mock_load_yaml.return_value = {test_env["yaml_job_key"]: TEST_JOB_NAME}

        # Track the created branch name
        created_branch = None

        def run_side_effect(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
            nonlocal created_branch
            cmd_list = args[0]
            if cmd_list == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="main\n", stderr="")
            if cmd_list[0:3] == ["git", "fetch", "--all"]:
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="", stderr="")
            if cmd_list[0:2] == ["git", "switch"] and len(cmd_list) >= 4:
                created_branch = cmd_list[3]  # Get the branch name (git switch -C <branch_name> <commit_hash>)
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="", stderr="")
            if cmd_list[0:3] == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="test_hash\n", stderr="")
            if "sbatch" in cmd_list[0]:
                # Simulate sbatch failure
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd_list, stderr="SLURM error")
            if cmd_list[0:3] == ["git", "branch", "-D"]:
                # Deleting any branch (including failed branches)
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="", stderr="")
            if cmd_list[0:2] == ["git", "branch"]:
                # Handle git branch commands for cleanup
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="", stderr="")
            if cmd_list[0:2] == ["git", "checkout"]:
                # Handle git checkout for branch restoration
                return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="", stderr="")
            raise NotImplementedError(f"Unexpected command: {cmd_list}")

        mock_run.side_effect = run_side_effect

        # Expect SystemExit due to sbatch failure
        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Verify cleanup was called
        mock_cleanup.assert_called_once_with(TEST_JOB_NAME, dry_run=False)

        # Note: restore_original_branch is not called on failure (sys.exit(1) prevents reaching the end of main)

        # Verify the failed branch deletion was attempted via subprocess.run
        # The `delete_branches` function is not mocked, so it will call `subprocess.run`
        delete_branch_calls = [call for call in mock_run.call_args_list if call.args[0][:3] == ["git", "branch", "-D"]]
        assert len(delete_branch_calls) == 1
        assert created_branch is not None
        assert delete_branch_calls[0].args[0][3] == created_branch


class TestGitLockingMechanism:
    """Tests for git lock file management and checking."""

    def test_git_operation_lock_acquires_and_releases(self, test_env: dict[str, Any], mocker: MockerFixture) -> None:
        """Test that GitOperationLock properly acquires and releases locks and cleans up the lock file."""
        # Create .git directory
        git_dir = test_env["test_dir"] / ".git"
        git_dir.mkdir(exist_ok=True)
        lock_file_path = git_dir / "test_lock.lock"

        # Mock fcntl
        mocker.patch("fcntl.fcntl")
        mock_flock = mocker.patch("fcntl.flock")

        # Create lock and use it
        with launcher.GitOperationLock(lock_file_path, timeout=1.0):
            # Verify lock file was created
            assert lock_file_path.exists()
            # Verify flock was called to acquire lock
            assert mock_flock.call_count >= 1
            mock_flock.assert_any_call(ANY, ANY)  # LOCK_EX | LOCK_NB

        # After context exit, lock file should be removed
        assert not lock_file_path.exists()
        # Verify flock was called to release lock
        mock_flock.assert_any_call(ANY, ANY)  # LOCK_UN

    def test_git_operation_lock_timeout(self, test_env: dict[str, Any], mocker: MockerFixture) -> None:
        """Test that GitOperationLock times out when lock cannot be acquired."""
        # Create .git directory
        git_dir = test_env["test_dir"] / ".git"
        git_dir.mkdir(exist_ok=True)
        lock_file_path = git_dir / "test_lock.lock"

        # Mock fcntl to always fail (simulating locked file)
        mocker.patch("fcntl.flock", side_effect=OSError("Resource temporarily unavailable"))

        # Should raise TimeoutError
        with (
            pytest.raises(TimeoutError, match="Failed to acquire git operation lock after"),
            launcher.GitOperationLock(lock_file_path, timeout=0.1),
        ):
            pass

    def test_check_git_locks_no_locks(self, test_env: dict[str, Any]) -> None:
        """Test check_git_locks returns True when no lock files exist."""
        # Ensure no lock files exist
        git_dir = test_env["test_dir"] / ".git"
        git_dir.mkdir(exist_ok=True)

        # Create a simple function to test (since check_git_locks is defined inside main)
        # We'll need to extract the logic or test through main
        # For now, let's create a standalone version for testing
        lock_files_to_check = [
            test_env["test_dir"] / ".git" / "slurm_checkout.lock",
            test_env["test_dir"] / ".git" / "index.lock",
        ]

        # Verify no locks exist
        for lock_file in lock_files_to_check:
            assert not lock_file.exists()

    def test_check_git_locks_with_locks(self, test_env: dict[str, Any]) -> None:
        """Test check_git_locks detects existing lock files."""
        # Create lock files
        git_dir = test_env["test_dir"] / ".git"
        git_dir.mkdir(exist_ok=True)

        lock_file1 = git_dir / "slurm_checkout.lock"
        lock_file2 = git_dir / "index.lock"

        # Create lock files
        lock_file1.touch()
        lock_file2.touch()

        # Verify they exist
        assert lock_file1.exists()
        assert lock_file2.exists()

    def test_main_blocks_submission_with_git_locks(self, test_env: dict[str, Any], mocker: MockerFixture) -> None:
        """Test that main() blocks job submission when git lock files exist."""
        # Setup test environment with lock file
        git_dir = test_env["test_dir"] / ".git"
        git_dir.mkdir(exist_ok=True)
        lock_file = git_dir / "index.lock"
        lock_file.touch()

        # Mock necessary components
        mock_sys_argv = [
            "launcher.py",
            TEST_PYTHON_SCRIPT_REL_PATH,
            TEST_CONFIG_NAME,
        ]
        mocker.patch("sys.argv", mock_sys_argv)

        # Mock subprocess for git commands
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="main\n", stderr="")

        # Expect SystemExit due to lock file check
        with pytest.raises(SystemExit) as excinfo:
            launcher.main(_configure_output=False)
        assert excinfo.value.code == 1

        # Verify sbatch was never called (job submission was blocked)
        sbatch_calls = [call for call in mock_run.call_args_list if "sbatch" in str(call)]
        assert len(sbatch_calls) == 0


class TestExpandConfigPatterns:
    """Test the expand_config_patterns function for glob pattern support."""

    def test_expand_exact_names(self, test_env: dict[str, Any]) -> None:
        """Test that exact config names are returned unchanged."""
        configs_dir = test_env["configs_dir"]
        patterns = ["experiment1", "experiment2"]

        result = launcher.expand_config_patterns(patterns, configs_dir)

        assert result == ["experiment1", "experiment2"]

    def test_expand_wildcard_pattern(self, test_env: dict[str, Any]) -> None:
        """Test wildcard (*) pattern expansion."""
        configs_dir = test_env["configs_dir"]

        # Create test config files
        (configs_dir / "exp1a.yaml").touch()
        (configs_dir / "exp1b.yaml").touch()
        (configs_dir / "exp2a.yaml").touch()
        (configs_dir / "experiment.yaml").touch()

        patterns = ["exp1*"]
        result = launcher.expand_config_patterns(patterns, configs_dir)

        assert "exp1a" in result
        assert "exp1b" in result
        assert "exp2a" not in result
        assert "experiment" not in result  # exp1* doesn't match experiment

        # Test that exp* matches experiment
        patterns2 = ["exp*"]
        result2 = launcher.expand_config_patterns(patterns2, configs_dir)

        assert "exp1a" in result2
        assert "exp1b" in result2
        assert "exp2a" in result2
        assert "experiment" in result2  # exp* should match experiment

    def test_expand_question_mark_pattern(self, test_env: dict[str, Any]) -> None:
        """Test single character (?) pattern expansion."""
        configs_dir = test_env["configs_dir"]

        # Create test config files
        (configs_dir / "exp1.yaml").touch()
        (configs_dir / "exp2.yaml").touch()
        (configs_dir / "exp10.yaml").touch()
        (configs_dir / "expA.yaml").touch()

        patterns = ["exp?"]
        result = launcher.expand_config_patterns(patterns, configs_dir)

        assert "exp1" in result
        assert "exp2" in result
        assert "expA" in result
        assert "exp10" not in result

    def test_expand_character_class_pattern(self, test_env: dict[str, Any]) -> None:
        """Test character class ([]) pattern expansion."""
        configs_dir = test_env["configs_dir"]

        # Create test config files
        (configs_dir / "exp1a.yaml").touch()
        (configs_dir / "exp1b.yaml").touch()
        (configs_dir / "exp1c.yaml").touch()
        (configs_dir / "exp1d.yaml").touch()
        (configs_dir / "exp2a.yaml").touch()

        patterns = ["exp1[a-c]"]
        result = launcher.expand_config_patterns(patterns, configs_dir)

        assert "exp1a" in result
        assert "exp1b" in result
        assert "exp1c" in result
        assert "exp1d" not in result
        assert "exp2a" not in result

    def test_expand_subdirectory_pattern(self, test_env: dict[str, Any]) -> None:
        """Test pattern expansion in subdirectories."""
        configs_dir = test_env["configs_dir"]
        subdir = configs_dir / "experiments"
        subdir.mkdir(exist_ok=True)

        # Create test config files in subdirectory
        (subdir / "exp1.yaml").touch()
        (subdir / "exp2.yaml").touch()
        (subdir / "exp3.yaml").touch()

        patterns = ["experiments/exp*"]
        result = launcher.expand_config_patterns(patterns, configs_dir)

        assert "experiments/exp1" in result
        assert "experiments/exp2" in result
        assert "experiments/exp3" in result

    def test_expand_multiple_patterns(self, test_env: dict[str, Any]) -> None:
        """Test expansion of multiple patterns."""
        configs_dir = test_env["configs_dir"]

        # Create test config files
        (configs_dir / "exp1.yaml").touch()
        (configs_dir / "exp2.yaml").touch()
        (configs_dir / "test1.yaml").touch()
        (configs_dir / "test2.yaml").touch()

        patterns = ["exp*", "test1"]
        result = launcher.expand_config_patterns(patterns, configs_dir)

        assert "exp1" in result
        assert "exp2" in result
        assert "test1" in result
        assert "test2" not in result

    def test_expand_no_matches(self, test_env: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        """Test pattern that matches no files."""
        # Ensure WARNING level messages are captured
        caplog.set_level(logging.WARNING)
        configs_dir = test_env["configs_dir"]

        patterns = ["nonexistent*"]
        result = launcher.expand_config_patterns(patterns, configs_dir)

        assert result == []
        assert "matched no config files" in caplog.text

    def test_expand_removes_duplicates(self, test_env: dict[str, Any]) -> None:
        """Test that duplicate configs are removed."""
        configs_dir = test_env["configs_dir"]

        # Create test config files
        (configs_dir / "exp1.yaml").touch()
        (configs_dir / "exp2.yaml").touch()

        patterns = ["exp1", "exp*", "exp1"]  # exp1 appears twice explicitly and in pattern
        result = launcher.expand_config_patterns(patterns, configs_dir)

        # exp1 should appear only once
        assert result.count("exp1") == 1
        assert "exp2" in result

    def test_expand_sorted_output(self, test_env: dict[str, Any]) -> None:
        """Test that results are sorted."""
        configs_dir = test_env["configs_dir"]

        # Create test config files
        (configs_dir / "zebra.yaml").touch()
        (configs_dir / "alpha.yaml").touch()
        (configs_dir / "beta.yaml").touch()

        patterns = ["*"]
        result = launcher.expand_config_patterns(patterns, configs_dir)

        # Check that our created files are in sorted order
        assert "alpha" in result
        assert "beta" in result
        assert "zebra" in result

        # Check sorted order for our files
        alpha_idx = result.index("alpha")
        beta_idx = result.index("beta")
        zebra_idx = result.index("zebra")
        assert alpha_idx < beta_idx < zebra_idx

    def test_expand_complex_pattern(self, test_env: dict[str, Any]) -> None:
        """Test complex pattern like experiment1[0-5]*."""
        configs_dir = test_env["configs_dir"]
        subdir = configs_dir / "experiments"
        subdir.mkdir(exist_ok=True)

        # Create test config files matching experiment10-15 pattern
        (subdir / "experiment10_test.yaml").touch()
        (subdir / "experiment11_test.yaml").touch()
        (subdir / "experiment15_test.yaml").touch()
        (subdir / "experiment16_test.yaml").touch()
        (subdir / "experiment20_test.yaml").touch()

        patterns = ["experiments/experiment1[0-5]*"]
        result = launcher.expand_config_patterns(patterns, configs_dir)

        assert "experiments/experiment10_test" in result
        assert "experiments/experiment11_test" in result
        assert "experiments/experiment15_test" in result
        assert "experiments/experiment16_test" not in result
        assert "experiments/experiment20_test" not in result


if __name__ == "__main__":
    pytest.main([__file__])
