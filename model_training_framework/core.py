"""
Model Training Framework - Main API

This module provides the high-level API for the model training framework:
- ModelTrainingFramework: Main orchestration class
- Integration of all framework components
- Simplified interface for common use cases
- Experiment lifecycle management
"""

from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any

from .config import (
    ConfigurationManager,
    ConfigValidator,
    ExecutionMode,
    ExperimentConfig,
    GridSearchConfig,
    GridSearchResult,
    NamingStrategy,
    ParameterGrid,
    ParameterGridSearch,
    ValidationResult,
)
from .slurm import SLURMJobMonitor, SLURMLauncher
from .utils import get_project_root, setup_logging, validate_project_structure

logger = logging.getLogger(__name__)


class FrameworkError(Exception):
    """Exception raised for framework-level errors."""


class ModelTrainingFramework:
    """
    High-level interface for the training framework.

    This class orchestrates all framework components and provides a simplified
    interface for experiment management, configuration handling, and job execution.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        config_dir: Path | None = None,
        experiments_dir: Path | None = None,
        slurm_template_path: Path | None = None,
        setup_logging_config: bool = True,
    ):
        """
        Initialize the training framework.

        Args:
            project_root: Path to project root (auto-detected if None)
            config_dir: Configuration directory (default: project_root/configs)
            experiments_dir: Experiments output directory (default: project_root/experiments)
            slurm_template_path: SLURM template file path (default: project_root/slurm_template.txt)
            setup_logging_config: Whether to setup logging configuration
        """
        # Determine project root
        self.project_root = Path(project_root) if project_root else get_project_root()

        # Setup directory paths
        self.config_dir = config_dir or self.project_root / "configs"
        self.experiments_dir = experiments_dir or self.project_root / "experiments"

        # Setup SLURM template path
        self.slurm_template_path: Path | None = None
        if slurm_template_path:
            self.slurm_template_path = Path(slurm_template_path)
        else:
            # Try common locations
            template_candidates = [
                self.project_root / "slurm_template.txt",
                self.project_root / "slurm" / "slurm_template.txt",
                self.project_root / "slurm" / "slurm_template.sbatch",
            ]
            for candidate in template_candidates:
                if candidate.exists():
                    self.slurm_template_path = candidate
                    break

        # Setup logging
        if setup_logging_config:
            setup_logging(
                level=logging.INFO,
                log_dir=self.experiments_dir / "logs",
                logger_name="model_training_framework",
            )

        # Initialize components
        self.config_manager = ConfigurationManager(
            project_root=self.project_root, config_dir=self.config_dir
        )

        self.slurm_launcher: SLURMLauncher | None = None
        if self.slurm_template_path and self.slurm_template_path.exists():
            self.slurm_launcher = SLURMLauncher(
                template_path=self.slurm_template_path,
                project_root=self.project_root,
                experiments_dir=self.experiments_dir,
            )

        self.job_monitor = SLURMJobMonitor()

        logger.info(f"Initialized ModelTrainingFramework at {self.project_root}")

        # Validate setup
        self._validate_framework_setup()

    def _validate_framework_setup(self) -> None:
        """Validate framework setup and log any issues."""
        # Validate project structure
        issues = validate_project_structure(self.project_root)
        if issues:
            logger.warning(f"Project structure issues: {issues}")

        # Validate configuration manager
        config_validation = self.config_manager.validate_project_structure()
        if config_validation.has_warnings:
            warnings = [issue.message for issue in config_validation.get_warnings()]
            logger.warning(f"Configuration setup warnings: {warnings}")

        # Validate SLURM setup
        if self.slurm_launcher:
            slurm_status = self.slurm_launcher.validate_slurm_environment()
            if not slurm_status["slurm_available"]:
                logger.warning(
                    f"SLURM not available: {slurm_status['missing_commands']}"
                )
            if not slurm_status["template_valid"]:
                logger.warning(
                    f"SLURM template issues: {slurm_status['template_issues']}"
                )
        else:
            logger.warning("SLURM launcher not available (template not found)")

    def create_experiment(
        self, config_dict: dict[str, Any], validate: bool = True
    ) -> ExperimentConfig:
        """
        Create and validate experiment configuration.

        Args:
            config_dict: Dictionary with experiment configuration
            validate: Whether to validate the configuration

        Returns:
            Validated ExperimentConfig object

        Raises:
            FrameworkError: If configuration is invalid
        """
        try:
            experiment_config = self.config_manager.create_experiment_config(
                config_dict, validate=validate
            )

            logger.info(
                f"Created experiment configuration: {experiment_config.experiment_name}"
            )
            return experiment_config

        except Exception as e:
            raise FrameworkError(
                f"Failed to create experiment configuration: {e}"
            ) from e

    def load_experiment_config(
        self, config_path: str | Path, validate: bool = True
    ) -> ExperimentConfig:
        """
        Load experiment configuration from file.

        Args:
            config_path: Path to configuration file
            validate: Whether to validate the configuration

        Returns:
            Loaded ExperimentConfig object

        Raises:
            FrameworkError: If configuration cannot be loaded
        """
        try:
            config_dict = self.config_manager.load_config(config_path, validate=False)
            return self.create_experiment(config_dict, validate=validate)

        except Exception as e:
            raise FrameworkError(f"Failed to load experiment configuration: {e}") from e

    def validate_experiment_config(self, config: ExperimentConfig) -> ValidationResult:
        """
        Validate experiment configuration.

        Args:
            config: Experiment configuration to validate

        Returns:
            ValidationResult with validation details
        """
        return ConfigValidator.validate_config(config)

    def run_single_experiment(
        self,
        config: ExperimentConfig,
        script_path: str | Path,
        execution_mode: ExecutionMode = ExecutionMode.SLURM,
        dry_run: bool = False,
    ) -> Any:  # ExperimentResult type would be defined elsewhere
        """
        Execute a single experiment.

        Args:
            config: Experiment configuration
            script_path: Path to training script
            execution_mode: How to execute the experiment
            dry_run: If True, don't actually execute

        Returns:
            Experiment execution result

        Raises:
            FrameworkError: If execution fails
        """
        logger.info(
            f"Running experiment: {config.experiment_name} (mode: {execution_mode.value})"
        )

        if execution_mode == ExecutionMode.SLURM:
            if not self.slurm_launcher:
                raise FrameworkError("SLURM launcher not available")

            result = self.slurm_launcher.submit_single_experiment(
                config=config,
                script_path=script_path,
                use_git_branch=True,
                dry_run=dry_run,
            )

            if result.success and result.job_id:
                self.job_monitor.track_job(result.job_id)

            return result

        if execution_mode == ExecutionMode.LOCAL:
            # TODO: Implement local execution
            raise FrameworkError("Local execution not yet implemented")

        if execution_mode == ExecutionMode.DRY_RUN:
            logger.info(f"Dry run: Would execute {config.experiment_name}")
            return {"status": "dry_run", "experiment_name": config.experiment_name}

        raise FrameworkError(f"Unsupported execution mode: {execution_mode}")

    def run_grid_search(
        self,
        base_config: ExperimentConfig | dict[str, Any] | str | Path,
        parameter_grids: list[ParameterGrid],
        execution_mode: ExecutionMode = ExecutionMode.SLURM,
        output_dir: Path | None = None,
        max_concurrent_jobs: int | None = None,
        script_path: str | Path | None = None,
    ) -> Any:  # GridSearchResult type
        """
        Execute parameter grid search.

        Args:
            base_config: Base configuration (config object, dict, or file path)
            parameter_grids: List of parameter grids to search
            execution_mode: How to execute experiments
            output_dir: Output directory for grid search results
            max_concurrent_jobs: Maximum concurrent jobs
            script_path: Path to training script

        Returns:
            Grid search execution result

        Raises:
            FrameworkError: If grid search fails
        """
        logger.info(f"Starting grid search with {len(parameter_grids)} parameter grids")

        # Load base configuration if needed
        if isinstance(base_config, str | Path):
            base_config_dict = self.config_manager.load_config(
                base_config, validate=False
            )
        elif isinstance(base_config, ExperimentConfig):
            base_config_dict = asdict(base_config)
        else:
            base_config_dict = base_config

        # Create parameter grid search
        grid_search = ParameterGridSearch(base_config_dict)
        for grid in parameter_grids:
            grid_search.add_grid(grid)

        # Execute grid search
        try:
            # TODO: Implement grid search execution without GridSearchExecutor
            # For now, just generate the experiments
            experiments = list(grid_search.generate_experiments())
            logger.info(f"Generated {len(experiments)} experiments from grid search")

            # Return a placeholder result
            # Create a minimal grid config for the result
            grid_config = GridSearchConfig(
                name="grid_search",
                description="Grid search placeholder",
                base_config=base_config_dict,
                parameter_grids=[],
                naming_strategy=NamingStrategy.PARAMETER_BASED,
                max_concurrent_jobs=max_concurrent_jobs,
                execution_mode=execution_mode,
                output_dir=str(output_dir) if output_dir else None,
            )

            result = GridSearchResult(
                grid_config=grid_config,
                total_experiments=len(experiments),
                generated_experiments=experiments,
                submitted_jobs=[],
                failed_submissions=[],
                execution_time=0.0,
                output_directory=output_dir,
            )

            logger.warning(
                "Grid search execution not yet implemented without GridSearchExecutor"
            )
            return result

        except Exception as e:
            raise FrameworkError(f"Grid search failed: {e}") from e

    def create_parameter_grid(
        self, name: str, parameters: dict[str, list[Any]], description: str = ""
    ) -> ParameterGrid:
        """
        Create a parameter grid for grid search.

        Args:
            name: Name of the parameter grid
            parameters: Dictionary of parameter_name -> list_of_values
            description: Optional description

        Returns:
            ParameterGrid object
        """
        grid = ParameterGrid(name=name, description=description)

        for param_name, values in parameters.items():
            grid.add_parameter(param_name, values)

        logger.info(
            f"Created parameter grid '{name}' with {grid.get_parameter_count()} combinations"
        )
        return grid

    def monitor_jobs(self, job_ids: list[str] | None = None) -> dict[str, Any]:
        """
        Get status of monitored jobs.

        Args:
            job_ids: Specific job IDs to check (all tracked jobs if None)

        Returns:
            Dictionary with job status information
        """
        self.job_monitor.update_job_status()

        if job_ids:
            result = {}
            for job_id in job_ids:
                job_info = self.job_monitor.get_job_info(job_id)
                if job_info:
                    result[job_id] = job_info
            return result
        return {
            "summary": self.job_monitor.get_job_summary(),
            "active_jobs": self.job_monitor.get_active_jobs(),
            "finished_jobs": self.job_monitor.get_finished_jobs(),
        }

    def wait_for_experiments(
        self, job_ids: list[str], timeout: float | None = None
    ) -> dict[str, Any]:
        """
        Wait for experiments to complete.

        Args:
            job_ids: List of job IDs to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Dictionary with completion results
        """
        logger.info(f"Waiting for {len(job_ids)} experiments to complete")

        completed_jobs = self.job_monitor.wait_for_jobs(
            job_ids=job_ids, timeout=timeout
        )

        # Summarize results
        successful = sum(1 for job in completed_jobs.values() if job.was_successful)
        failed = len(completed_jobs) - successful

        logger.info(f"Experiments completed: {successful} successful, {failed} failed")

        return {
            "completed_jobs": completed_jobs,
            "summary": {
                "total": len(completed_jobs),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(completed_jobs)
                if completed_jobs
                else 0.0,
            },
        }

    def cancel_experiments(self, job_ids: list[str] | None = None) -> dict[str, bool]:
        """
        Cancel experiments.

        Args:
            job_ids: Specific job IDs to cancel (all tracked jobs if None)

        Returns:
            Dictionary of job_id -> success status
        """
        if job_ids:
            results = {}
            for job_id in job_ids:
                if self.slurm_launcher:
                    results[job_id] = self.slurm_launcher.cancel_job(job_id)
                else:
                    results[job_id] = False
            return results
        return self.job_monitor.cancel_tracked_jobs()

    def list_configurations(self, pattern: str = "*.yaml") -> list[Path]:
        """
        List available configuration files.

        Args:
            pattern: Glob pattern for configuration files

        Returns:
            List of configuration file paths
        """
        return self.config_manager.list_configs(pattern)

    def create_training_script_template(self, output_path: Path) -> None:
        """
        Create a template training script.

        Args:
            output_path: Path to save the template script
        """
        template_content = '''#!/usr/bin/env python3
"""
Training Script Template

This script demonstrates how to use the Model Training Framework
for training machine learning models with SLURM integration.
"""

import argparse
from pathlib import Path

# Import framework components
from model_training_framework import ModelTrainingFramework
from model_training_framework.trainer import GenericTrainer, GenericTrainerConfig

# Import ML libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lightning.fabric import Fabric


class SimpleModel(nn.Module):
    """Simple example model."""

    def __init__(self, input_size: int = 10, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


def create_dummy_data(num_samples: int = 1000, input_size: int = 10):
    """Create dummy dataset for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, 1)
    return TensorDataset(X, y)


def training_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    """Training step function."""
    x, y = batch
    pred = trainer.model(x)
    loss = nn.functional.mse_loss(pred, y)
    return {"loss": loss}


def validation_step(trainer, batch, batch_idx, dataloader_idx, dataloader_name):
    """Validation step function."""
    x, y = batch
    pred = trainer.model(x)
    loss = nn.functional.mse_loss(pred, y)
    return {"loss": loss}


def main():
    parser = argparse.ArgumentParser(description="Example training script")
    parser.add_argument("config", help="Configuration name or path")
    parser.add_argument("--local", action="store_true", help="Run locally instead of SLURM")
    args = parser.parse_args()

    # Initialize framework
    framework = ModelTrainingFramework()

    # Load experiment configuration
    config = framework.load_experiment_config(args.config)

    # Setup Fabric for distributed training
    fabric = Fabric(devices=1, accelerator="auto")

    # Create model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)

    # Setup model and optimizer with Fabric
    model, optimizer = fabric.setup(model, optimizer)

    # Create data loaders
    train_dataset = create_dummy_data(1000)
    val_dataset = create_dummy_data(200)

    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size)

    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)

    # Create trainer configuration
    trainer_config = GenericTrainerConfig()

    # Initialize trainer
    trainer = GenericTrainer(
        config=trainer_config,
        fabric=fabric,
        model=model,
        optimizer=optimizer
    )

    # Set training and validation step functions
    trainer.set_training_step(training_step)
    trainer.set_validation_step(validation_step)

    # Train model
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=config.training.max_epochs
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
'''

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(template_content)
            logger.info(f"Created training script template: {output_path}")
        except Exception as e:
            raise FrameworkError(
                f"Failed to create training script template: {e}"
            ) from e

    def get_framework_status(self) -> dict[str, Any]:
        """
        Get comprehensive framework status.

        Returns:
            Dictionary with framework status information
        """
        status = {
            "project_root": str(self.project_root),
            "config_dir": str(self.config_dir),
            "experiments_dir": str(self.experiments_dir),
            "slurm_template": str(self.slurm_template_path)
            if self.slurm_template_path
            else None,
            "components": {
                "config_manager": True,
                "slurm_launcher": self.slurm_launcher is not None,
                "job_monitor": True,
                "grid_search_executor": True,
            },
            "configurations": len(self.list_configurations()),
            "job_summary": self.job_monitor.get_job_summary(),
        }

        # Add SLURM environment status
        if self.slurm_launcher:
            status["slurm_environment"] = (
                self.slurm_launcher.validate_slurm_environment()
            )

        return status


# Convenience functions for common operations


def quick_experiment(
    config_path: str | Path,
    script_path: str | Path,
    project_root: Path | None = None,
) -> Any:
    """
    Quick way to run a single experiment.

    Args:
        config_path: Path to configuration file
        script_path: Path to training script
        project_root: Project root directory

    Returns:
        Experiment result
    """
    framework = ModelTrainingFramework(project_root=project_root)
    config = framework.load_experiment_config(config_path)
    return framework.run_single_experiment(config, script_path)


def quick_grid_search(
    base_config_path: str | Path,
    parameter_grids: list[ParameterGrid],
    script_path: str | Path,
    project_root: Path | None = None,
) -> Any:
    """
    Quick way to run parameter grid search.

    Args:
        base_config_path: Path to base configuration file
        parameter_grids: List of parameter grids
        script_path: Path to training script
        project_root: Project root directory

    Returns:
        Grid search result
    """
    framework = ModelTrainingFramework(project_root=project_root)
    return framework.run_grid_search(
        base_config=base_config_path,
        parameter_grids=parameter_grids,
        script_path=script_path,
    )
