"""
Hyperparameter Optimization - Intermediate HPC Usage

This example demonstrates how to run comprehensive hyperparameter optimization
experiments on HPC clusters using SLURM job scheduling. Perfect for research teams who:

- Need to scale experiments across multiple compute nodes
- Want to optimize model hyperparameters systematically
- Require robust job management and monitoring
- Use SLURM-managed HPC environments

Target Audience: Researchers with HPC access, teams scaling experiments
"""

from pathlib import Path

from model_training_framework import ModelTrainingFramework
from model_training_framework.config import (
    ExecutionMode,
    ExperimentConfig,
    NamingStrategy,
    ParameterGridSearch,
)


def create_comprehensive_grid_search() -> ParameterGridSearch:
    """
    Create a comprehensive hyperparameter grid search for HPC experiments.

    This function demonstrates how to set up systematic parameter exploration
    for research-grade experiments that require extensive compute resources.

    Returns:
        Configured ParameterGridSearch instance
    """

    # Base configuration optimized for HPC environments
    base_config = {
        "experiment_name": "hpc_hyperparameter_optimization",
        "description": "Systematic hyperparameter optimization on HPC cluster",
        # Model configuration - transformer architecture
        "model": {
            "name": "transformer",
            "hidden_size": 512,  # Will be varied
            "num_layers": 6,  # Will be varied
            "num_heads": 8,  # Will be varied
            "dropout_rate": 0.1,
            "attention_dropout": 0.1,
        },
        # Training configuration for HPC
        "training": {
            "epochs": 100,  # Longer training for thorough evaluation
            "batch_size": 64,  # Will be varied
            "learning_rate": 1e-3,  # Will be varied
            "gradient_accumulation_steps": 1,  # Will be varied
            "gradient_clip_norm": 1.0,
            "warmup_ratio": 0.1,
        },
        # Data configuration
        "data": {
            "dataset_name": "research_dataset",
            "train_split": "train",
            "val_split": "validation",
            "test_split": "test",
            "preprocessing": "standard",
        },
        # Optimizer configuration
        "optimizer": {
            "name": "adamw",
            "weight_decay": 0.01,  # Will be varied
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        # Scheduler configuration
        "scheduler": {
            "name": "cosine",
            "warmup_steps": 1000,  # Will be varied
            "min_lr": 1e-6,
        },
        # SLURM configuration for HPC
        "slurm": {
            "job_name": "hparam_opt",
            "time_limit": "24:00:00",
            "nodes": 1,
            "ntasks_per_node": 1,
            "cpus_per_task": 8,  # More CPUs for data processing
            "mem": "32G",  # Will be varied for larger models
            "gres": "gpu:1",
            "partition": "gpu",
            "account": None,  # Set this for your cluster
        },
        # Logging and tracking
        "logging": {
            "log_level": "INFO",
            "use_wandb": True,
            "wandb_project": "hpc_hyperparameter_optimization",
            "wandb_entity": None,  # Set this for your team
            "log_interval": 50,
        },
        # Checkpointing for long runs
        "checkpoint": {
            "save_every_n_epochs": 10,
            "save_best_only": True,
            "monitor": "val_loss",
            "mode": "min",
        },
        # Early stopping for efficiency
        "early_stopping": {
            "enabled": True,
            "patience": 15,
            "min_delta": 0.001,
            "monitor": "val_loss",
        },
    }

    # Initialize grid search
    grid_search = ParameterGridSearch(base_config)
    grid_search.set_naming_strategy(NamingStrategy.DESCRIPTIVE)

    return grid_search


def setup_learning_rate_grid(grid_search: ParameterGridSearch) -> None:
    """
    Set up learning rate and optimization parameter grid.

    This grid focuses on finding the optimal learning rate schedule
    and optimization parameters for the given architecture.
    """

    lr_grid = grid_search.create_grid(
        name="learning_rate_optimization",
        description="Systematic exploration of learning rates and optimization",
    )

    # Learning rate range (log scale)
    lr_grid.add_parameter("training.learning_rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])

    # Weight decay for regularization
    lr_grid.add_parameter("optimizer.weight_decay", [0.001, 0.01, 0.1])

    # Gradient accumulation for effective batch size
    lr_grid.add_parameter("training.gradient_accumulation_steps", [1, 2, 4])

    # Warmup steps for stable training
    lr_grid.add_parameter("scheduler.warmup_steps", [500, 1000, 2000])

    print(f"📊 Learning Rate Grid: {lr_grid.get_parameter_count()} combinations")


def setup_architecture_grid(grid_search: ParameterGridSearch) -> None:
    """
    Set up model architecture parameter grid.

    This grid explores different model sizes to find the optimal
    architecture for the given computational budget.
    """

    arch_grid = grid_search.create_grid(
        name="model_architecture",
        description="Model architecture exploration for optimal capacity",
    )

    # Model size variations
    arch_grid.add_parameter("model.hidden_size", [256, 512, 768, 1024])
    arch_grid.add_parameter("model.num_layers", [4, 6, 8, 12])
    arch_grid.add_parameter("model.num_heads", [8, 12, 16])

    # Dropout rates for regularization
    arch_grid.add_parameter("model.dropout_rate", [0.0, 0.1, 0.2])

    print(f"🏗️  Architecture Grid: {arch_grid.get_parameter_count()} combinations")


def setup_training_grid(grid_search: ParameterGridSearch) -> None:
    """
    Set up training configuration parameter grid.

    This grid optimizes training hyperparameters like batch size
    and training schedule for the HPC environment.
    """

    training_grid = grid_search.create_grid(
        name="training_optimization", description="Training hyperparameter optimization"
    )

    # Batch size (considering GPU memory)
    training_grid.add_parameter("training.batch_size", [32, 64, 128])

    # Training epochs (with early stopping)
    training_grid.add_parameter("training.epochs", [50, 100, 150])

    # Gradient clipping for stability
    training_grid.add_parameter("training.gradient_clip_norm", [0.5, 1.0, 2.0])

    print(f"⚙️  Training Grid: {training_grid.get_parameter_count()} combinations")


def setup_resource_allocation(experiment_config: ExperimentConfig) -> ExperimentConfig:
    """
    Dynamically adjust SLURM resource allocation based on model size.

    This function demonstrates how to scale resources based on the
    specific configuration of each experiment.

    Args:
        experiment_config: Base experiment configuration

    Returns:
        Modified configuration with appropriate resource allocation
    """

    hidden_size = experiment_config.model.hidden_size
    num_layers = experiment_config.model.num_layers
    batch_size = experiment_config.training.batch_size

    # Estimate memory requirements based on model size
    model_params = hidden_size * num_layers * 12 * 4  # Rough estimate in bytes
    memory_gb = max(
        32, int(model_params / (1024**3) * batch_size * 2)
    )  # 2x for gradients

    # Adjust time limit based on model complexity
    if hidden_size >= 1024 or num_layers >= 12:
        time_limit = "48:00:00"  # Larger models need more time
    elif hidden_size >= 512 or num_layers >= 8:
        time_limit = "36:00:00"
    else:
        time_limit = "24:00:00"

    # Update SLURM configuration
    if experiment_config.slurm:
        experiment_config.slurm.mem = f"{memory_gb}G"
        experiment_config.slurm.time_limit = time_limit

        # Use more CPUs for larger models (data preprocessing)
        if hidden_size >= 768:
            experiment_config.slurm.cpus_per_task = 12

        # Request high-memory nodes for very large models
        if memory_gb > 64:
            experiment_config.slurm.partition = "highmem"

    return experiment_config


def main():
    """
    Main hyperparameter optimization function.

    Demonstrates the complete workflow for running systematic hyperparameter
    optimization on HPC clusters with proper resource management.
    """

    print("🔍 HPC Hyperparameter Optimization")
    print("=" * 50)

    # Setup paths for HPC environment
    project_root = Path.cwd()
    config_dir = project_root / "demo" / "example2_intermediate_hpc" / "configs"
    output_dir = project_root / "hpc_experiments"

    # Initialize framework
    print("🚀 Initializing framework for HPC environment...")
    framework = ModelTrainingFramework(project_root=project_root, config_dir=config_dir)

    # Create comprehensive grid search
    print("📋 Setting up comprehensive parameter grid...")
    grid_search = create_comprehensive_grid_search()

    # Add parameter grids
    setup_learning_rate_grid(grid_search)
    setup_architecture_grid(grid_search)
    setup_training_grid(grid_search)

    # Calculate total experiments
    total_experiments = grid_search.get_total_experiments()
    print(f"🎯 Total experiments: {total_experiments}")

    if total_experiments > 1000:
        print("⚠️  Large number of experiments detected!")
        print("   Consider reducing parameter space or using random search")

        # Option to sample a subset
        response = input("   Continue with full grid? (y/N): ").lower()
        if response != "y":
            print("   💡 Consider using random sampling or Bayesian optimization")
            return

    # Validate parameter grids
    print("🔍 Validating parameter grids...")
    validation_issues = grid_search.validate_grids()

    if validation_issues:
        print("❌ Validation issues found:")
        for issue in validation_issues:
            print(f"   - {issue}")
        return

    print("✅ Parameter grids validated successfully")

    # Generate experiment preview
    print("\n📋 Experiment Preview (first 5):")
    experiments = list(grid_search.generate_experiments())

    for i, experiment in enumerate(experiments[:5]):
        # Apply resource allocation
        configured_experiment = setup_resource_allocation(experiment)

        print(f"   {i + 1}. {configured_experiment.experiment_name}")
        print(
            f"      Model: {configured_experiment.model.hidden_size}h x {configured_experiment.model.num_layers}l"
        )
        print(
            f"      LR: {configured_experiment.training.learning_rate}, BS: {configured_experiment.training.batch_size}"
        )
        print(
            f"      Resources: {configured_experiment.slurm.mem if configured_experiment.slurm else 'N/A'}, "
            f"{configured_experiment.slurm.time_limit if configured_experiment.slurm else 'N/A'}"
        )
        print()

    if len(experiments) > 5:
        print(f"   ... and {len(experiments) - 5} more experiments")

    # Save grid configuration for reproducibility
    output_dir.mkdir(exist_ok=True)
    grid_config_path = output_dir / "hyperparameter_grid_config.json"
    grid_search.save_grid_config(grid_config_path)
    print(f"💾 Saved grid configuration: {grid_config_path}")

    # Dry run validation
    print("\n🔍 Running dry run validation...")

    try:
        # Apply resource allocation to all experiments
        for experiment_config in experiments:
            setup_resource_allocation(experiment_config)

        dry_run_result = framework.run_grid_search(
            parameter_grids=grid_search.get_all_grids(),
            base_config=grid_search.base_config,
            execution_mode=ExecutionMode.DRY_RUN,
            output_dir=output_dir,
        )

        print("✅ Dry run validation completed")
        print(f"   Success rate: {dry_run_result.success_rate:.1%}")
        print(
            f"   Estimated runtime: {dry_run_result.execution_time:.2f}s per experiment"
        )

        if dry_run_result.success_rate < 1.0:
            print("⚠️  Some experiments failed validation")
            print("   Check configuration and resource requirements")

    except Exception as e:
        print(f"❌ Dry run failed: {e}")
        return

    # Option for actual SLURM submission
    print("\n🚀 Ready for SLURM submission")
    print("Note: Actual submission is commented out for safety")
    print("To submit jobs, uncomment the SLURM submission section")

    # Commented out for safety - uncomment to actually submit
    """
    print("Submitting to SLURM...")

    try:
        slurm_result = framework.run_grid_search(
            parameter_grids=grid_search.get_all_grids(),
            base_config=grid_search.base_config,
            execution_mode=ExecutionMode.SLURM,
            output_dir=output_dir,
            max_concurrent_jobs=20,  # Limit concurrent jobs
        )

        print(f"✅ Submitted {len(slurm_result.submitted_jobs)} jobs")

        # Monitor job status
        print("📊 Monitoring job status...")
        print("Use 'squeue -u $USER' to check job status")
        print("Use the job monitoring utilities to track progress")

    except Exception as e:
        print(f"❌ SLURM submission failed: {e}")
    """

    # Provide guidance for next steps
    print("\n🎯 Next Steps:")
    print("1. Review the generated experiment configurations")
    print("2. Adjust resource allocation if needed")
    print("3. Submit a small batch first to test the setup")
    print("4. Use job monitoring tools to track progress")
    print("5. Analyze results using the result analysis utilities")

    print("\n💡 Monitoring Commands:")
    print("   squeue -u $USER  # Check job queue")
    print("   sacct -S today   # Check today's jobs")
    print("   scancel <job_id> # Cancel a job")

    print("\n✅ Hyperparameter optimization setup complete!")


def demonstrate_resource_scaling():
    """
    Demonstrate how to scale resources based on experiment requirements.

    This function shows advanced resource allocation strategies for
    different types of experiments.
    """

    print("\n🔬 Resource Scaling Demonstration")
    print("-" * 30)

    # Example configurations
    configs = [
        {
            "model": {"hidden_size": 256, "num_layers": 4},
            "training": {"batch_size": 32},
        },
        {
            "model": {"hidden_size": 512, "num_layers": 8},
            "training": {"batch_size": 64},
        },
        {
            "model": {"hidden_size": 1024, "num_layers": 12},
            "training": {"batch_size": 128},
        },
    ]

    for i, config in enumerate(configs, 1):
        hidden_size = config["model"]["hidden_size"]
        num_layers = config["model"]["num_layers"]
        batch_size = config["training"]["batch_size"]

        # Estimate resources
        memory_gb = max(32, int(hidden_size * num_layers * batch_size / 1000))
        cpus = 8 if hidden_size < 512 else 12 if hidden_size < 1024 else 16
        time_hours = 24 if hidden_size < 512 else 36 if hidden_size < 1024 else 48

        print(f"Configuration {i}:")
        print(f"   Model: {hidden_size}h x {num_layers}l, Batch: {batch_size}")
        print(f"   Resources: {memory_gb}G RAM, {cpus} CPUs, {time_hours}h")
        print()

    print("✅ Resource scaling demonstration complete")


if __name__ == "__main__":
    main()
    demonstrate_resource_scaling()
