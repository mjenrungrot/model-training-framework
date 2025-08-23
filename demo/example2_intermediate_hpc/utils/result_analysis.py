"""
Result Analysis Utilities for HPC Experiments

This module provides utilities for analyzing results from hyperparameter
optimization and distributed training experiments, including statistical
analysis, visualization helpers, and performance comparisons.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


@dataclass
class ExperimentResult:
    """Data class for individual experiment results."""

    name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    status: str
    duration: float | None = None
    job_id: str | None = None


class HyperparameterAnalyzer:
    """
    Analyze hyperparameter optimization results.

    This class provides comprehensive analysis of hyperparameter experiments,
    including parameter importance, correlation analysis, and visualization.
    """

    def __init__(self, results_dir: Path):
        """
        Initialize analyzer with results directory.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.experiments: list[ExperimentResult] = []
        self.df: pd.DataFrame | None = None

    def load_experiments(self) -> None:
        """Load all experiment results from the directory."""
        experiments = []

        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                result = self._load_experiment(exp_dir)
                if result:
                    experiments.append(result)

        self.experiments = experiments
        print(f"Loaded {len(experiments)} experiments")

    def _load_experiment(self, exp_dir: Path) -> ExperimentResult | None:
        """
        Load a single experiment result.

        Args:
            exp_dir: Path to experiment directory

        Returns:
            ExperimentResult or None if loading failed
        """
        try:
            # Load configuration
            config_file = exp_dir / "config.yaml"
            if not config_file.exists():
                return None

            with config_file.open() as f:
                config = yaml.safe_load(f)

            # Load metrics
            metrics_file = exp_dir / "metrics.json"
            metrics = {}
            if metrics_file.exists():
                with metrics_file.open() as f:
                    metrics = json.load(f)

            # Determine status
            status = "unknown"
            if (exp_dir / "training_complete.flag").exists():
                status = "completed"
            elif (exp_dir / "training_failed.flag").exists():
                status = "failed"

            # Load job info if available
            job_info_file = exp_dir / "job_info.json"
            job_id = None
            duration = None

            if job_info_file.exists():
                with job_info_file.open() as f:
                    job_info = json.load(f)
                    job_id = job_info.get("job_id")
                    duration = job_info.get("duration_seconds")

            return ExperimentResult(
                name=exp_dir.name,
                config=config,
                metrics=metrics,
                status=status,
                duration=duration,
                job_id=job_id,
            )

        except Exception as e:
            print(f"Error loading experiment {exp_dir.name}: {e}")
            return None

    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame from experiment results.

        Returns:
            DataFrame with flattened experiment data
        """
        if not self.experiments:
            self.load_experiments()

        data = []

        for exp in self.experiments:
            if exp.status != "completed":
                continue

            row = {"experiment_name": exp.name, "status": exp.status}

            # Flatten configuration
            row.update(self._flatten_dict(exp.config, prefix="config"))

            # Add metrics
            row.update(exp.metrics)

            # Add metadata
            if exp.duration:
                row["duration_hours"] = exp.duration / 3600

            data.append(row)

        self.df = pd.DataFrame(data)
        return self.df

    def _flatten_dict(self, d: dict, prefix: str = "", sep: str = ".") -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def analyze_parameter_importance(
        self, target_metric: str = "val_loss"
    ) -> pd.DataFrame:
        """
        Analyze parameter importance using correlation analysis.

        Args:
            target_metric: Target metric to analyze against

        Returns:
            DataFrame with parameter importance scores
        """
        if self.df is None:
            self.create_dataframe()

        # Filter to completed experiments with the target metric
        valid_df = self.df[
            (self.df["status"] == "completed") & (self.df[target_metric].notna())
        ].copy()

        if valid_df.empty:
            print(f"No valid experiments found with metric: {target_metric}")
            return pd.DataFrame()

        # Find numeric parameters
        numeric_cols = []
        for col in valid_df.columns:
            if col.startswith("config.") and pd.api.types.is_numeric_dtype(
                valid_df[col]
            ):
                numeric_cols.append(col)

        if not numeric_cols:
            print("No numeric configuration parameters found")
            return pd.DataFrame()

        # Calculate correlations
        correlations = []
        for param in numeric_cols:
            correlation = valid_df[param].corr(valid_df[target_metric])
            if not np.isnan(correlation):
                correlations.append(
                    {
                        "parameter": param.replace("config.", ""),
                        "correlation": correlation,
                        "abs_correlation": abs(correlation),
                        "importance": abs(correlation),
                    }
                )

        importance_df = pd.DataFrame(correlations)
        return importance_df.sort_values("abs_correlation", ascending=False)

    def get_best_experiments(
        self, metric: str = "val_loss", n: int = 10
    ) -> pd.DataFrame:
        """
        Get top performing experiments.

        Args:
            metric: Metric to rank by
            n: Number of top experiments to return

        Returns:
            DataFrame with best experiments
        """
        if self.df is None:
            self.create_dataframe()

        valid_df = self.df[
            (self.df["status"] == "completed") & (self.df[metric].notna())
        ].copy()

        # Determine if lower or higher is better
        if any(
            keyword in metric.lower() for keyword in ["loss", "error", "mse", "mae"]
        ):
            # Lower is better
            best_df = valid_df.nsmallest(n, metric)
        else:
            # Higher is better (accuracy, f1, etc.)
            best_df = valid_df.nlargest(n, metric)

        return best_df

    def create_parameter_analysis_plots(
        self, target_metric: str = "val_loss", output_dir: Path | None = None
    ):
        """
        Create visualization plots for parameter analysis.

        Args:
            target_metric: Target metric for analysis
            output_dir: Directory to save plots (optional)
        """
        if self.df is None:
            self.create_dataframe()

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        # Parameter importance plot
        importance_df = self.analyze_parameter_importance(target_metric)

        if not importance_df.empty:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df.head(10), x="abs_correlation", y="parameter")
            plt.title(f"Parameter Importance for {target_metric}")
            plt.xlabel("Absolute Correlation")

            if output_dir:
                plt.savefig(
                    output_dir / "parameter_importance.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            else:
                plt.show()

        # Performance distribution plot
        valid_df = self.df[self.df[target_metric].notna()]

        if not valid_df.empty:
            plt.figure(figsize=(10, 6))
            plt.hist(valid_df[target_metric], bins=30, alpha=0.7, edgecolor="black")
            plt.title(f"Distribution of {target_metric}")
            plt.xlabel(target_metric)
            plt.ylabel("Frequency")

            if output_dir:
                plt.savefig(
                    output_dir / "performance_distribution.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            else:
                plt.show()

    def create_convergence_analysis(
        self, experiment_name: str, output_dir: Path | None = None
    ):
        """
        Analyze training convergence for a specific experiment.

        Args:
            experiment_name: Name of experiment to analyze
            output_dir: Directory to save plots (optional)
        """
        exp_dir = self.results_dir / experiment_name
        training_log = exp_dir / "training_metrics.json"

        if not training_log.exists():
            print(f"Training log not found for {experiment_name}")
            return

        try:
            with training_log.open() as f:
                training_data = json.load(f)

            # Convert to DataFrame
            df = pd.DataFrame(training_data)

            # Create convergence plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Training Convergence: {experiment_name}")

            # Training loss
            if "train_loss" in df.columns:
                axes[0, 0].plot(df["epoch"], df["train_loss"])
                axes[0, 0].set_title("Training Loss")
                axes[0, 0].set_xlabel("Epoch")
                axes[0, 0].set_ylabel("Loss")

            # Validation loss
            if "val_loss" in df.columns:
                axes[0, 1].plot(df["epoch"], df["val_loss"], color="orange")
                axes[0, 1].set_title("Validation Loss")
                axes[0, 1].set_xlabel("Epoch")
                axes[0, 1].set_ylabel("Loss")

            # Learning rate
            if "learning_rate" in df.columns:
                axes[1, 0].plot(df["epoch"], df["learning_rate"], color="green")
                axes[1, 0].set_title("Learning Rate")
                axes[1, 0].set_xlabel("Epoch")
                axes[1, 0].set_ylabel("Learning Rate")
                axes[1, 0].set_yscale("log")

            # Training and validation together
            if "train_loss" in df.columns and "val_loss" in df.columns:
                axes[1, 1].plot(df["epoch"], df["train_loss"], label="Train", alpha=0.7)
                axes[1, 1].plot(
                    df["epoch"], df["val_loss"], label="Validation", alpha=0.7
                )
                axes[1, 1].set_title("Train vs Validation Loss")
                axes[1, 1].set_xlabel("Epoch")
                axes[1, 1].set_ylabel("Loss")
                axes[1, 1].legend()

            plt.tight_layout()

            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True)
                plt.savefig(
                    output_dir / f"convergence_{experiment_name}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"Error creating convergence analysis: {e}")

    def generate_analysis_report(self, output_file: Path | None = None) -> str:
        """
        Generate comprehensive analysis report.

        Args:
            output_file: File to save report (optional)

        Returns:
            Report content as string
        """
        if not self.experiments:
            self.load_experiments()

        if self.df is None:
            self.create_dataframe()

        # Basic statistics
        total_experiments = len(self.experiments)
        completed = len([exp for exp in self.experiments if exp.status == "completed"])
        failed = len([exp for exp in self.experiments if exp.status == "failed"])

        # Performance statistics
        performance_stats = ""
        if "val_loss" in self.df.columns:
            val_loss_stats = self.df["val_loss"].describe()
            best_experiments = self.get_best_experiments("val_loss", 5)

            performance_stats = f"""
Performance Analysis (Validation Loss):
- Mean: {val_loss_stats["mean"]:.4f}
- Std: {val_loss_stats["std"]:.4f}
- Min: {val_loss_stats["min"]:.4f}
- Max: {val_loss_stats["max"]:.4f}

Top 5 Experiments:
{self._format_experiment_table(best_experiments)}
"""

        # Parameter importance
        importance_analysis = ""
        if "val_loss" in self.df.columns:
            importance_df = self.analyze_parameter_importance("val_loss")
            if not importance_df.empty:
                importance_analysis = f"""
Parameter Importance (by correlation with val_loss):
{importance_df.head(10).to_string(index=False)}
"""

        report = f"""
Hyperparameter Optimization Analysis Report
==========================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Results Directory: {self.results_dir}

Experiment Overview:
- Total experiments: {total_experiments}
- Completed: {completed}
- Failed: {failed}
- Success rate: {completed / total_experiments * 100:.1f}%

{performance_stats}

{importance_analysis}

Recommendations:
1. Focus on top-performing parameter combinations
2. Investigate failed experiments for common issues
3. Consider narrowing parameter ranges around best performers
4. Monitor resource utilization for efficiency improvements
"""

        if output_file:
            with output_file.open("w") as f:
                f.write(report)

        return report

    def _format_experiment_table(self, df: pd.DataFrame) -> str:
        """Format experiment DataFrame for display."""
        if df.empty:
            return "No experiments found"

        # Select key columns for display
        display_cols = ["experiment_name", "val_loss"]

        # Add some key hyperparameters if available
        for col in [
            "config.training.learning_rate",
            "config.model.hidden_size",
            "config.training.batch_size",
        ]:
            if col in df.columns:
                display_cols.append(col)

        display_df = df[display_cols].copy()
        return display_df.to_string(index=False)


def main():
    """
    Main function demonstrating result analysis utilities.
    """
    print("📊 Result Analysis Utilities for HPC Experiments")
    print("=" * 50)

    # Example usage
    results_dir = Path("./hpc_experiments")

    if not results_dir.exists():
        print("No results directory found. Create some experiments first!")
        print("Expected directory: ./hpc_experiments")
        return

    # Initialize analyzer
    analyzer = HyperparameterAnalyzer(results_dir)
    analyzer.load_experiments()

    if not analyzer.experiments:
        print("No experiments found in results directory.")
        return

    # Create analysis DataFrame
    df = analyzer.create_dataframe()
    print(f"Created analysis DataFrame with {len(df)} completed experiments")

    # Generate comprehensive report
    report = analyzer.generate_analysis_report()
    print(report)

    # Save analysis plots
    plot_dir = results_dir / "analysis_plots"
    plot_dir.mkdir(exist_ok=True)

    try:
        analyzer.create_parameter_analysis_plots(output_dir=plot_dir)
        print(f"Analysis plots saved to: {plot_dir}")
    except Exception as e:
        print(f"Could not create plots: {e}")
        print("Make sure matplotlib and seaborn are installed")

    # Example convergence analysis for first experiment
    if analyzer.experiments:
        first_exp = analyzer.experiments[0]
        print(f"\nCreating convergence analysis for: {first_exp.name}")
        try:
            analyzer.create_convergence_analysis(first_exp.name, output_dir=plot_dir)
        except Exception as e:
            print(f"Could not create convergence analysis: {e}")

    print(f"\n✅ Analysis complete! Check {plot_dir} for visualizations.")


if __name__ == "__main__":
    main()
