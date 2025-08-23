"""
SBATCH Template Engine

This module provides template rendering for SLURM batch scripts:
- Template loading and validation
- Context variable substitution
- SBATCH script generation
- Template management and caching
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import string
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class TemplateError(Exception):
    """Exception raised for template-related errors."""


@dataclass
class TemplateContext:
    """Context for template rendering with SLURM job parameters."""

    # Job identification
    job_name: str
    experiment_name: str

    # Script and execution
    script_path: str
    config_name: str
    python_executable: str = "python"

    # Git information
    branch_name: str | None = None
    commit_hash: str | None = None

    # SLURM parameters
    account: str = "realitylab"
    partition: str = "ckpt-all"
    nodes: int = 1
    ntasks_per_node: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    mem: str = "256G"
    time: str = "1-00:00:00"
    constraint: str | None = "a40|a100"
    requeue: bool = True

    # Output and logging
    output_file: str | None = None
    error_file: str | None = None

    # Email notifications
    mail_type: str | None = None
    mail_user: str | None = None

    # Additional parameters
    extra_params: dict[str, Any] = None

    def __post_init__(self):
        """Initialize extra parameters if not provided."""
        if self.extra_params is None:
            self.extra_params = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for template rendering."""
        result = {
            # Convert all attributes to uppercase for SLURM convention
            "JOB_NAME": self.job_name,
            "EXPERIMENT_NAME": self.experiment_name,
            "SCRIPT_PATH": self.script_path,
            "CONFIG_NAME": self.config_name,
            "PYTHON_EXECUTABLE": self.python_executable,
            "BRANCH_NAME": self.branch_name or "",
            "COMMIT_HASH": self.commit_hash or "",
            "ACCOUNT": self.account,
            "PARTITION": self.partition,
            "NODES": self.nodes,
            "NTASKS_PER_NODE": self.ntasks_per_node,
            "GPUS_PER_NODE": self.gpus_per_node,
            "CPUS_PER_TASK": self.cpus_per_task,
            "MEM": self.mem,
            "TIME": self.time,
            "CONSTRAINT": self.constraint or "",
            "REQUEUE": "true" if self.requeue else "false",
            "OUTPUT_FILE": self.output_file
            or f"experiments/{self.experiment_name}/%j.out",
            "ERROR_FILE": self.error_file
            or f"experiments/{self.experiment_name}/%j.err",
            "MAIL_TYPE": self.mail_type or "",
            "MAIL_USER": self.mail_user or "",
        }

        # Add extra parameters
        for key, value in self.extra_params.items():
            result[key.upper()] = value

        return result


class SBATCHTemplateEngine:
    """Template engine for generating SBATCH scripts."""

    def __init__(self, template_path: Path | None = None):
        """
        Initialize template engine.

        Args:
            template_path: Path to default template file
        """
        self.template_path = template_path
        self.template_cache: dict[str, str] = {}

        if template_path and template_path.exists():
            self.load_template(template_path)

    def load_template(self, template_path: Path) -> str:
        """
        Load template from file.

        Args:
            template_path: Path to template file

        Returns:
            Template content as string

        Raises:
            TemplateError: If template cannot be loaded
        """
        try:
            template_content = template_path.read_text()

            # Cache template
            cache_key = str(template_path)
            self.template_cache[cache_key] = template_content

            logger.debug(f"Loaded template from {template_path}")
            return template_content

        except Exception as e:
            raise TemplateError(f"Failed to load template {template_path}: {e}") from e

    def validate_template(self, template_content: str) -> list[str]:
        """
        Validate template content and return any issues.

        Args:
            template_content: Template content to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check for required SBATCH directives
        required_directives = [
            "#SBATCH --job-name",
            "#SBATCH --account",
            "#SBATCH --partition",
        ]

        for directive in required_directives:
            if directive not in template_content:
                issues.append(f"Missing required SBATCH directive: {directive}")

        # Check for shebang
        lines = template_content.split("\n")
        if not lines or not lines[0].startswith("#!"):
            issues.append("Template should start with shebang (#!/bin/bash)")

        # Find all template variables
        variables = self.extract_template_variables(template_content)

        # Check for common typos in variable names
        common_vars = {
            "JOB_NAME",
            "ACCOUNT",
            "PARTITION",
            "NODES",
            "GPUS_PER_NODE",
            "SCRIPT_PATH",
            "CONFIG_NAME",
            "OUTPUT_FILE",
            "ERROR_FILE",
        }

        suspicious_vars = []
        for var in variables:
            # Check for lowercase variants of common variables
            if (
                var.lower() in {v.lower() for v in common_vars}
                and var not in common_vars
            ):
                suspicious_vars.append(var)

        if suspicious_vars:
            issues.append(f"Suspicious variable names (check case): {suspicious_vars}")

        return issues

    def extract_template_variables(self, template_content: str) -> set[str]:
        """
        Extract all template variables from template content.

        Args:
            template_content: Template content

        Returns:
            Set of variable names found in template
        """
        # Find variables in {{VAR}} or {VAR} format
        pattern = r"\{\{?(\w+)\}?\}"
        matches = re.findall(pattern, template_content)
        return set(matches)

    def render_template(
        self, template_content: str, context: TemplateContext | dict[str, Any]
    ) -> str:
        """
        Render template with given context.

        Args:
            template_content: Template content string
            context: Template context or dictionary

        Returns:
            Rendered template content

        Raises:
            TemplateError: If rendering fails
        """
        try:
            # Convert context to dictionary if needed
            if isinstance(context, TemplateContext):
                context_dict = context.to_dict()
            else:
                context_dict = context

            # Use string.Template for safe substitution
            template = string.Template(template_content)

            # First pass: substitute variables in {{VAR}} format
            double_brace_pattern = re.compile(r"\{\{(\w+)\}\}")
            content = double_brace_pattern.sub(r"${\1}", template_content)

            # Second pass: substitute variables in {VAR} format
            single_brace_pattern = re.compile(r"\{(\w+)\}")
            content = single_brace_pattern.sub(r"${\1}", content)

            # Create template and substitute
            template = string.Template(content)
            rendered = template.safe_substitute(context_dict)

            # Check for unsubstituted variables
            unsubstituted = re.findall(r"\$\{(\w+)\}", rendered)
            if unsubstituted:
                logger.warning(f"Unsubstituted template variables: {unsubstituted}")

            return rendered

        except Exception as e:
            raise TemplateError(f"Failed to render template: {e}") from e

    def generate_sbatch_script(
        self,
        context: TemplateContext | dict[str, Any],
        template_path: Path | None = None,
        output_path: Path | None = None,
    ) -> str:
        """
        Generate SBATCH script from template and context.

        Args:
            context: Template context
            template_path: Optional template path (uses default if None)
            output_path: Optional output path to save script

        Returns:
            Generated SBATCH script content

        Raises:
            TemplateError: If script generation fails
        """
        # Load template
        if template_path:
            template_content = self.load_template(template_path)
        elif self.template_path:
            template_content = self.load_template(self.template_path)
        else:
            raise TemplateError("No template path provided")

        # Validate template
        issues = self.validate_template(template_content)
        if issues:
            logger.warning(f"Template validation issues: {issues}")

        # Render template
        rendered_script = self.render_template(template_content, context)

        # Save to file if requested
        if output_path:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(rendered_script)
                logger.info(f"Generated SBATCH script: {output_path}")
            except Exception as e:
                raise TemplateError(
                    f"Failed to save SBATCH script to {output_path}: {e}"
                ) from e

        return rendered_script

    def create_default_template(self) -> str:
        """
        Create a default SBATCH template.

        Returns:
            Default template content
        """
        return """#!/bin/bash

#SBATCH --job-name={{JOB_NAME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodes={{NODES}}
#SBATCH --ntasks-per-node={{NTASKS_PER_NODE}}
#SBATCH --gpus-per-node={{GPUS_PER_NODE}}
#SBATCH --cpus-per-task={{CPUS_PER_TASK}}
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}
#SBATCH --output={{OUTPUT_FILE}}
#SBATCH --error={{ERROR_FILE}}
#SBATCH --requeue={{REQUEUE}}

# Optional parameters
{% if CONSTRAINT %}
#SBATCH --constraint={{CONSTRAINT}}
{% endif %}
{% if MAIL_TYPE %}
#SBATCH --mail-type={{MAIL_TYPE}}
{% endif %}
{% if MAIL_USER %}
#SBATCH --mail-user={{MAIL_USER}}
{% endif %}

# Environment setup
module load python/3.9
module load cuda/11.8

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: {{JOB_NAME}}"
echo "Node: $SLURM_NODEID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Git information (if available)
{% if COMMIT_HASH %}
echo "Commit Hash: {{COMMIT_HASH}}"
{% endif %}
{% if BRANCH_NAME %}
echo "Branch: {{BRANCH_NAME}}"
{% endif %}

# Create experiment directory
mkdir -p experiments/{{EXPERIMENT_NAME}}

# Setup environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Run training script
echo "Starting training..."
{{PYTHON_EXECUTABLE}} {{SCRIPT_PATH}} {{CONFIG_NAME}}

echo "Job completed at $(date)"
"""

    def save_default_template(self, output_path: Path) -> None:
        """
        Save default template to file.

        Args:
            output_path: Path to save template
        """
        template_content = self.create_default_template()

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(template_content)
            logger.info(f"Saved default template to {output_path}")
        except Exception as e:
            raise TemplateError(f"Failed to save default template: {e}") from e

    def preview_rendered_script(
        self,
        context: TemplateContext | dict[str, Any],
        template_path: Path | None = None,
    ) -> str:
        """
        Preview rendered script without saving.

        Args:
            context: Template context
            template_path: Optional template path

        Returns:
            Rendered script preview
        """
        return self.generate_sbatch_script(context, template_path, output_path=None)


def create_template_context_from_config(
    experiment_name: str,
    script_path: str,
    config_name: str,
    slurm_config: dict[str, Any],
    git_info: dict[str, str] | None = None,
) -> TemplateContext:
    """
    Create template context from experiment and SLURM configuration.

    Args:
        experiment_name: Name of the experiment
        script_path: Path to training script
        config_name: Name of configuration file
        slurm_config: SLURM configuration dictionary
        git_info: Optional git information dictionary

    Returns:
        Configured TemplateContext
    """
    return TemplateContext(
        job_name=slurm_config.get("job_name", experiment_name),
        experiment_name=experiment_name,
        script_path=script_path,
        config_name=config_name,
        python_executable=slurm_config.get("python_executable", "python"),
        branch_name=git_info.get("branch_name") if git_info else None,
        commit_hash=git_info.get("commit_hash") if git_info else None,
        account=slurm_config.get("account", "realitylab"),
        partition=slurm_config.get("partition", "ckpt-all"),
        nodes=slurm_config.get("nodes", 1),
        ntasks_per_node=slurm_config.get("ntasks_per_node", 1),
        gpus_per_node=slurm_config.get("gpus_per_node", 1),
        cpus_per_task=slurm_config.get("cpus_per_task", 8),
        mem=slurm_config.get("mem", "256G"),
        time=slurm_config.get("time", "1-00:00:00"),
        constraint=slurm_config.get("constraint"),
        requeue=slurm_config.get("requeue", True),
        output_file=slurm_config.get("output"),
        error_file=slurm_config.get("error"),
        mail_type=slurm_config.get("mail_type"),
        mail_user=slurm_config.get("mail_user"),
        extra_params=slurm_config.get("extra_args", {}),
    )
