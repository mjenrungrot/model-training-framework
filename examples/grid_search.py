"""
Parameter Grid Search Example

This example demonstrates how to use the parameter grid search
functionality to explore hyperparameter combinations automatically.
"""

from pathlib import Path

from model_training_framework import ModelTrainingFramework
from model_training_framework.config import (
    ExecutionMode,
    NamingStrategy,
    ParameterGridSearch,
)


def main():
    """Main grid search example."""

    # Setup paths
    project_root = Path.cwd()
    config_dir = project_root / "configs"

    # Base configuration for all experiments
    base_config = {
        "experiment_name": "grid_search_base",
        "description": "Parameter grid search example",
        "model": {
            "name": "transformer",
            "hidden_size": 256,  # Will be overridden by grid search
            "num_layers": 6,
            "num_heads": 8,
        },
        "training": {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,  # Will be overridden by grid search
            "gradient_accumulation_steps": 1,
        },
        "data": {
            "dataset_name": "custom_dataset",
            "train_split": "train",
            "val_split": "validation",
        },
        "optimizer": {
            "name": "adamw",
            "weight_decay": 0.01,  # Will be overridden by grid search
        },
        "scheduler": {"name": "cosine", "warmup_steps": 1000},
        "logging": {
            "log_level": "INFO",
            "use_wandb": True,
            "wandb_project": "grid_search_example",
        },
    }

    print("🔍 Setting up Parameter Grid Search...")

    # Initialize framework
    framework = ModelTrainingFramework(project_root=project_root, config_dir=config_dir)

    # Create parameter grid search
    grid_search = ParameterGridSearch(base_config)
    grid_search.set_naming_strategy(NamingStrategy.DESCRIPTIVE)

    # 1. Learning rate and optimization grid
    lr_grid = grid_search.create_grid(
        name="learning_rate_optimization",
        description="Explore learning rates and optimization parameters",
    )
    lr_grid.add_parameter("training.learning_rate", [1e-4, 5e-4, 1e-3, 5e-3])
    lr_grid.add_parameter("optimizer.weight_decay", [0.01, 0.05, 0.1])
    lr_grid.add_parameter("training.gradient_accumulation_steps", [1, 2, 4])

    print(f"📊 Learning rate grid: {lr_grid.get_parameter_count()} combinations")

    # 2. Model architecture grid
    model_grid = grid_search.create_grid(
        name="model_architecture", description="Explore model size variations"
    )
    model_grid.add_parameter("model.hidden_size", [128, 256, 512])
    model_grid.add_parameter("model.num_layers", [4, 6, 8])
    model_grid.add_parameter("model.num_heads", [8, 12, 16])

    print(
        f"🏗️  Model architecture grid: {model_grid.get_parameter_count()} combinations"
    )

    # 3. Training configuration grid
    training_grid = grid_search.create_grid(
        name="training_config", description="Explore training configurations"
    )
    training_grid.add_parameter("training.batch_size", [16, 32, 64])
    training_grid.add_parameter("scheduler.warmup_steps", [500, 1000, 2000])

    print(
        f"⚙️  Training config grid: {training_grid.get_parameter_count()} combinations"
    )

    # Display total experiments
    total_experiments = grid_search.get_total_experiments()
    print(f"🎯 Total experiments to run: {total_experiments}")

    # Validate grids
    issues = grid_search.validate_grids()
    if issues:
        print("⚠️  Grid validation issues:")
        for issue in issues:
            print(f"   - {issue}")
        return

    print("✅ Grid validation passed")

    # Generate experiments preview
    print("\n📋 Experiment Preview:")
    experiments = list(grid_search.generate_experiments())

    for i, experiment in enumerate(experiments[:5]):  # Show first 5
        print(f"   {i + 1}. {experiment.experiment_name}")
        print(f"      LR: {experiment.training.learning_rate}")
        print(f"      Hidden Size: {experiment.model.hidden_size}")
        print(f"      Batch Size: {experiment.training.batch_size}")
        print()

    if len(experiments) > 5:
        print(f"   ... and {len(experiments) - 5} more experiments")

    # Save grid configuration
    output_dir = project_root / "grid_search_output"
    output_dir.mkdir(exist_ok=True)

    grid_search.save_grid_config(output_dir / "grid_config.json")
    print(f"💾 Saved grid configuration to {output_dir / 'grid_config.json'}")

    # Execute grid search (dry run mode for this example)
    print("\n🚀 Executing Grid Search (Dry Run Mode)...")

    try:
        result = framework.run_grid_search(
            parameter_grids=[lr_grid, model_grid, training_grid],
            base_config=base_config,
            execution_mode=ExecutionMode.DRY_RUN,
            output_dir=output_dir,
        )

        print("\n📈 Grid Search Results:")
        print(f"   Total experiments: {result.total_experiments}")
        print(f"   Generated experiments: {len(result.generated_experiments)}")
        print(f"   Success rate: {result.success_rate:.2%}")
        print(f"   Execution time: {result.execution_time:.2f}s")

        # Display summary
        summary = result.get_summary()
        print(f"\n📊 Summary: {summary}")

    except Exception as e:
        print(f"❌ Grid search failed: {e}")
        raise

    # Example of running with SLURM (commented out)
    """
    # To run with SLURM, you would use:
    result_slurm = framework.run_grid_search(
        parameter_grids=[lr_grid, model_grid, training_grid],
        base_config=base_config,
        execution_mode=ExecutionMode.SLURM,
        output_dir=output_dir,
        slurm_config={
            "time_limit": "24:00:00",
            "nodes": 1,
            "cpus_per_task": 4,
            "mem": "32G",
            "gres": "gpu:1"
        }
    )
    
    print(f"Submitted {len(result_slurm.submitted_jobs)} SLURM jobs")
    """

    print("\n✅ Grid search example completed successfully!")


def demonstrate_advanced_features():
    """Demonstrate advanced grid search features."""

    print("\n🔬 Advanced Grid Search Features Demo:")

    # Create a complex nested parameter grid
    base_config = {"experiment_name": "advanced_example"}
    grid_search = ParameterGridSearch(base_config)

    # 1. Nested parameter paths
    advanced_grid = grid_search.create_grid("advanced_features")

    # Deep nested parameters
    advanced_grid.add_parameter("model.encoder.hidden_size", [256, 512])
    advanced_grid.add_parameter("model.decoder.num_layers", [2, 4])
    advanced_grid.add_parameter(
        "training.optimizer.lr_scheduler.warmup_ratio", [0.1, 0.2]
    )

    # 2. Different naming strategies
    print("   Testing naming strategies:")

    for strategy in NamingStrategy:
        grid_search.set_naming_strategy(strategy)
        experiments = list(grid_search.generate_experiments())

        if experiments:
            example_name = experiments[0].experiment_name
            print(f"      {strategy.value}: {example_name}")

    # 3. Loading and saving configurations
    output_path = Path("advanced_grid_config.json")
    grid_search.save_grid_config(output_path)

    # Load it back
    loaded_grid_search = ParameterGridSearch.load_grid_config(output_path)
    print(
        f"   Successfully loaded grid with {loaded_grid_search.get_total_experiments()} experiments"
    )

    # Cleanup
    output_path.unlink(missing_ok=True)

    print("✅ Advanced features demo completed")


if __name__ == "__main__":
    main()
    demonstrate_advanced_features()
