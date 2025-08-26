"""
Fault-Tolerant Training with Multi-DataLoader Architecture - Advanced Production Pipeline

This example demonstrates production-grade fault-tolerant training using the
multi-dataloader-only architecture with comprehensive error handling, monitoring,
and recovery mechanisms. Perfect for:

- Production ML pipelines with multiple data sources
- Long-running training jobs that must handle various failure modes
- Systems requiring comprehensive logging and monitoring
- Enterprise deployments with strict uptime requirements
- Multi-dataset training with fault tolerance

Target Audience: ML engineers, production teams, advanced researchers
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
import traceback
from typing import Any

import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_training_framework.trainer import (
    CheckpointConfig,
    FaultToleranceConfig,
    GenericTrainer,
    GenericTrainerConfig,
    LoggingConfig,
    MultiDataLoaderConfig,
    PerformanceConfig,
    PreemptionConfig,
    ValidationConfig,
)
from model_training_framework.trainer.config import (
    EpochLengthPolicy,
    SamplingStrategy,
    ValAggregation,
    ValidationFrequency,
)


class FailureType(Enum):
    """Types of failures that can occur during training."""

    PREEMPTION = "preemption"
    OOM = "out_of_memory"
    NETWORK = "network_failure"
    FILESYSTEM = "filesystem_error"
    MODEL_DIVERGENCE = "model_divergence"
    DATA_CORRUPTION = "data_corruption"
    HARDWARE = "hardware_failure"
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    """Record of a training failure."""

    timestamp: datetime
    failure_type: FailureType
    message: str
    epoch: int
    global_step: int
    phase: str
    dataloader_name: str  # Added for multi-loader tracking
    recovery_action: str
    stack_trace: str | None = None
    system_stats: dict | None = None


class SystemMonitor:
    """
    Monitor system resources and detect potential issues.

    This class provides real-time monitoring of CPU, memory, GPU,
    and other system resources to predict and prevent failures.
    """

    def __init__(self, alert_thresholds: dict | None = None):
        """
        Initialize system monitor.

        Args:
            alert_thresholds: Dictionary of resource thresholds for alerts
        """
        self.alert_thresholds = alert_thresholds or {
            "memory_percent": 90.0,
            "cpu_percent": 95.0,
            "gpu_memory_percent": 95.0,
            "disk_usage_percent": 95.0,
            "temperature_celsius": 85.0,
        }

        self.monitoring = False
        self.monitor_thread: threading.Thread | None = None
        self.alerts: list[dict] = []
        self.stats_history: list[dict] = []

        # Set up logging
        self.logger = logging.getLogger("SystemMonitor")

    def start_monitoring(self, interval: float = 30.0) -> None:
        """
        Start continuous system monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("System monitoring started")

    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")

    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                stats = self.get_system_stats()
                self.stats_history.append(stats)

                # Keep only last 1000 entries
                if len(self.stats_history) > 1000:
                    self.stats_history = self.stats_history[-1000:]

                # Check for alerts
                alerts = self._check_alerts(stats)
                self.alerts.extend(alerts)

                # Log alerts
                for alert in alerts:
                    self.logger.warning(f"System alert: {alert}")

            except Exception:
                self.logger.exception("Error in monitoring loop")

            time.sleep(interval)

    def get_system_stats(self) -> dict:
        """Get current system statistics."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": dict(psutil.virtual_memory()._asdict()),
            "disk_usage": dict(psutil.disk_usage("/")._asdict()),
            "process_count": len(psutil.pids()),
        }

        # Add GPU stats if available
        try:
            if torch.cuda.is_available():
                gpu_stats = []
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_stats(i)
                    gpu_stats.append(
                        {
                            "device": i,
                            "allocated_gb": gpu_memory.get(
                                "allocated_bytes.all.current", 0
                            )
                            / 1e9,
                            "reserved_gb": gpu_memory.get(
                                "reserved_bytes.all.current", 0
                            )
                            / 1e9,
                            "utilization": torch.cuda.utilization(i)
                            if hasattr(torch.cuda, "utilization")
                            else 0,
                        }
                    )
                stats["gpu"] = gpu_stats
        except Exception as e:
            self.logger.debug(f"Could not get GPU stats: {e}")

        return stats

    def _check_alerts(self, stats: dict) -> list[dict]:
        """Check for system alerts based on thresholds."""
        alerts = []

        # Memory check
        memory_percent = stats["memory"]["percent"]
        if memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(
                {
                    "type": "memory_high",
                    "value": memory_percent,
                    "threshold": self.alert_thresholds["memory_percent"],
                    "message": f"Memory usage high: {memory_percent:.1f}%",
                }
            )

        # CPU check
        cpu_percent = stats["cpu_percent"]
        if cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(
                {
                    "type": "cpu_high",
                    "value": cpu_percent,
                    "threshold": self.alert_thresholds["cpu_percent"],
                    "message": f"CPU usage high: {cpu_percent:.1f}%",
                }
            )

        # Disk check
        disk_percent = (
            stats["disk_usage"]["used"] / stats["disk_usage"]["total"]
        ) * 100
        if disk_percent > self.alert_thresholds["disk_usage_percent"]:
            alerts.append(
                {
                    "type": "disk_full",
                    "value": disk_percent,
                    "threshold": self.alert_thresholds["disk_usage_percent"],
                    "message": f"Disk usage high: {disk_percent:.1f}%",
                }
            )

        return alerts


class ProductionTrainer(GenericTrainer):
    """
    Production-grade trainer with comprehensive fault tolerance for multi-loader training.

    This trainer extends the GenericTrainer with enterprise-level
    fault tolerance, monitoring, and recovery capabilities for
    multi-dataloader scenarios.
    """

    def __init__(
        self,
        config: GenericTrainerConfig,
        model: nn.Module,
        optimizers: list[torch.optim.Optimizer],  # Changed to list
        loss_fn: nn.Module | None = None,
        alert_callbacks: list[Callable] | None = None,
        fabric: Any = None,  # Added fabric support
    ):
        """
        Initialize production trainer with multi-loader support.

        Args:
            config: Trainer configuration with MultiDataLoaderConfig
            model: Model to train
            optimizers: List of optimizers (always a list)
            loss_fn: Loss function
            alert_callbacks: List of functions to call on alerts
            fabric: Optional Lightning Fabric instance
        """
        super().__init__(config, model, optimizers, fabric=fabric)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.alert_callbacks = alert_callbacks or []

        # Failure tracking (enhanced for multi-loader)
        self.failure_history: list[FailureRecord] = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.dataloader_failure_counts: dict[str, int] = {}  # Track per-loader failures

        # System monitoring
        self.system_monitor = SystemMonitor()

        # Performance tracking (per dataloader)
        self.performance_history: dict[str, list[dict]] = {}
        self.last_loss_values: dict[str, list[float]] = {}
        self.divergence_threshold = 10.0  # Loss increase threshold

        # Setup enhanced logging
        self.setup_production_logging()

        # Health checks
        self.last_successful_step = time.time()
        self.max_step_duration = 300.0  # 5 minutes

        # Graceful shutdown handling
        self.shutdown_requested = False
        self.setup_signal_handlers()

    def setup_production_logging(self) -> None:
        """Set up comprehensive logging for production environment."""
        log_dir = Path(self.config.checkpoint.root_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure main logger
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger("ProductionTrainer")
        self.logger.info("Production trainer with multi-loader support initialized")
        self.logger.info(
            f"DataLoaders: {self.config.train_loader_config.dataloader_names}"
        )
        self.logger.info(
            f"Sampling Strategy: {self.config.train_loader_config.sampling_strategy.value}"
        )

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.warning(
                f"Received signal {signum}, initiating graceful shutdown..."
            )
            self.shutdown_requested = True
            self.handle_failure(
                FailureType.PREEMPTION,
                f"Signal {signum} received",
                dataloader_name="all",
            )

        # Handle common signals
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)

    def training_step(
        self,
        trainer,
        batch,
        batch_idx: int,
        dataloader_idx: int,
        dataloader_name: str,
    ) -> dict[str, Any]:
        """
        Enhanced training step with comprehensive error handling for multi-loader.

        Args:
            trainer: The trainer instance (self)
            batch: Training batch
            dataloader_idx: Index of the current dataloader
            dataloader_name: Name of the current dataloader

        Returns:
            Dictionary with training metrics
        """
        step_start_time = time.time()

        try:
            # Check for shutdown request
            if self.shutdown_requested:
                self.logger.info("Shutdown requested, stopping training...")
                raise KeyboardInterrupt("Graceful shutdown requested")

            # Health check
            if step_start_time - self.last_successful_step > self.max_step_duration:
                self.handle_failure(
                    FailureType.UNKNOWN,
                    f"Training step took too long: {step_start_time - self.last_successful_step:.1f}s",
                    dataloader_name=dataloader_name,
                )

            # Memory check before processing
            if torch.cuda.is_available():
                memory_reserved = torch.cuda.memory_reserved() / 1e9

                if memory_reserved > 0.95 * torch.cuda.max_memory_allocated() / 1e9:
                    self.logger.warning(
                        f"GPU memory usage high for {dataloader_name}: {memory_reserved:.1f}GB"
                    )
                    torch.cuda.empty_cache()

            # Forward pass with error handling
            x, y = batch

            try:
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.handle_failure(
                        FailureType.OOM, str(e), dataloader_name=dataloader_name
                    )
                    raise
                self.handle_failure(
                    FailureType.UNKNOWN,
                    f"Forward pass error: {e}",
                    dataloader_name=dataloader_name,
                )
                raise

            # Check for loss divergence
            if torch.isnan(loss) or torch.isinf(loss):
                self.handle_failure(
                    FailureType.MODEL_DIVERGENCE,
                    f"Loss is NaN or Inf: {loss.item()}",
                    dataloader_name=dataloader_name,
                )
                raise RuntimeError("Model divergence detected")

            # Track loss history per dataloader
            loss_value = loss.item()
            if dataloader_name not in self.last_loss_values:
                self.last_loss_values[dataloader_name] = []

            self.last_loss_values[dataloader_name].append(loss_value)
            if len(self.last_loss_values[dataloader_name]) > 10:
                self.last_loss_values[dataloader_name] = self.last_loss_values[
                    dataloader_name
                ][-10:]

            # Check for sudden loss increase
            if len(self.last_loss_values[dataloader_name]) >= 5:
                recent_avg = sum(self.last_loss_values[dataloader_name][-3:]) / 3
                older_avg = sum(self.last_loss_values[dataloader_name][-6:-3]) / 3

                if recent_avg > older_avg * self.divergence_threshold:
                    self.logger.warning(
                        f"Potential divergence in {dataloader_name}: recent={recent_avg:.4f}, "
                        f"older={older_avg:.4f}"
                    )

            # Calculate metrics
            with torch.no_grad():
                _, predicted = torch.max(y_pred, 1)
                accuracy = (predicted == y).float().mean().item()

            # Record successful step
            self.last_successful_step = time.time()
            step_duration = self.last_successful_step - step_start_time

            metrics = {
                "loss": loss,
                f"{dataloader_name}/loss": loss.item(),
                f"{dataloader_name}/accuracy": accuracy,
                f"{dataloader_name}/lr": self.optimizers[0].param_groups[0]["lr"],
                f"{dataloader_name}/step_duration": step_duration,
                "dataloader_idx": dataloader_idx,
                "gpu_memory_gb": torch.cuda.memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0,
            }

            # Track performance per dataloader
            if dataloader_name not in self.performance_history:
                self.performance_history[dataloader_name] = []

            self.performance_history[dataloader_name].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "loss": loss_value,
                    "accuracy": accuracy,
                    "step_duration": step_duration,
                }
            )

            return metrics

        except Exception:
            self.logger.exception(f"Error in training step for {dataloader_name}")
            self.logger.exception(traceback.format_exc())
            raise

    def validation_step(
        self, trainer, batch, dataloader_idx: int, dataloader_name: str
    ) -> dict[str, Any]:
        """
        Enhanced validation step with error handling for multi-loader.

        Args:
            trainer: The trainer instance (self)
            batch: Validation batch
            dataloader_idx: Index of the current dataloader
            dataloader_name: Name of the current dataloader

        Returns:
            Dictionary with validation metrics
        """
        try:
            with torch.no_grad():
                x, y = batch
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)

                _, predicted = torch.max(y_pred, 1)
                accuracy = (predicted == y).float().mean().item()

                return {
                    "val_loss": loss,
                    f"val_{dataloader_name}/loss": loss.item(),
                    f"val_{dataloader_name}/accuracy": accuracy,
                }

        except Exception:
            self.logger.exception(f"Error in validation step for {dataloader_name}")
            raise

    def handle_failure(
        self,
        failure_type: FailureType,
        message: str,
        dataloader_name: str = "unknown",
        recovery_action: str = "checkpoint_and_continue",
    ) -> None:
        """
        Handle training failures with comprehensive logging and recovery.

        Args:
            failure_type: Type of failure
            message: Failure message
            dataloader_name: Name of the dataloader that caused the failure
            recovery_action: Action taken for recovery
        """
        failure_record = FailureRecord(
            timestamp=datetime.now(),
            failure_type=failure_type,
            message=message,
            epoch=getattr(self, "current_epoch", -1),
            global_step=getattr(self, "global_step", -1),
            phase=getattr(self, "current_phase", "unknown"),
            dataloader_name=dataloader_name,
            recovery_action=recovery_action,
            stack_trace=traceback.format_exc(),
            system_stats=self.system_monitor.get_system_stats(),
        )

        self.failure_history.append(failure_record)
        self.recovery_attempts += 1

        # Track per-dataloader failures
        if dataloader_name not in self.dataloader_failure_counts:
            self.dataloader_failure_counts[dataloader_name] = 0
        self.dataloader_failure_counts[dataloader_name] += 1

        # Log failure
        self.logger.error("Training failure detected:")
        self.logger.error(f"  Type: {failure_type.value}")
        self.logger.error(f"  DataLoader: {dataloader_name}")
        self.logger.error(f"  Message: {message}")
        self.logger.error(f"  Recovery attempts: {self.recovery_attempts}")

        # Save failure record
        failure_log = Path(self.config.checkpoint.root_dir) / "failures.json"
        failure_records = []

        if failure_log.exists():
            try:
                with failure_log.open() as f:
                    failure_records = json.load(f)
            except Exception:  # nosec B110
                pass

        failure_records.append(asdict(failure_record))

        try:
            with failure_log.open("w") as f:
                json.dump(failure_records, f, indent=2, default=str)
        except Exception:
            self.logger.exception("Could not save failure record")

        # Trigger alerts
        for callback in self.alert_callbacks:
            try:
                callback(failure_record)
            except Exception:
                self.logger.exception("Alert callback failed")

        # Save emergency checkpoint
        try:
            self.save_checkpoint(emergency=True)
            self.logger.info("Emergency checkpoint saved")
        except Exception:
            self.logger.exception("Emergency checkpoint save failed")

        # Check if maximum recovery attempts reached
        if self.recovery_attempts >= self.max_recovery_attempts:
            self.logger.critical(
                f"Maximum recovery attempts ({self.max_recovery_attempts}) reached. "
                "Training will terminate."
            )
            raise RuntimeError("Maximum recovery attempts exceeded")

    def on_training_start(self) -> None:
        """Enhanced training start with monitoring."""
        super().on_training_start()

        # Start system monitoring
        self.system_monitor.start_monitoring(interval=30.0)

        self.logger.info("Production training started with multi-loader architecture")
        self.logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        self.logger.info(
            f"Number of dataloaders: {len(self.config.train_loader_config.dataloader_names)}"
        )
        self.logger.info(
            f"Sampling strategy: {self.config.train_loader_config.sampling_strategy.value}"
        )

        if torch.cuda.is_available():
            self.logger.info(
                f"GPU memory available: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            )

    def on_training_end(self, logs: dict[str, Any]) -> None:
        """Enhanced training end with comprehensive reporting."""
        super().on_training_end(logs)

        # Stop monitoring
        self.system_monitor.stop_monitoring()

        # Generate training report
        self.generate_training_report()

        self.logger.info("Production training completed")
        self.logger.info(f"Total failures handled: {len(self.failure_history)}")
        self.logger.info(f"Total recovery attempts: {self.recovery_attempts}")

        # Log per-dataloader failure statistics
        if self.dataloader_failure_counts:
            self.logger.info("Failures per dataloader:")
            for loader_name, count in self.dataloader_failure_counts.items():
                self.logger.info(f"  {loader_name}: {count}")

    def generate_training_report(self) -> None:
        """Generate comprehensive training report."""
        report_dir = Path(self.config.checkpoint.root_dir) / "reports"
        report_dir.mkdir(exist_ok=True)

        # Calculate per-dataloader statistics
        dataloader_stats = {}
        for loader_name in self.config.train_loader_config.dataloader_names:
            if loader_name in self.performance_history:
                perf_data = self.performance_history[loader_name]
                dataloader_stats[loader_name] = {
                    "total_steps": len(perf_data),
                    "avg_step_duration": sum(p["step_duration"] for p in perf_data)
                    / len(perf_data)
                    if perf_data
                    else 0,
                    "failures": self.dataloader_failure_counts.get(loader_name, 0),
                }

        report = {
            "training_summary": {
                "start_time": getattr(self, "training_start_time", "unknown"),
                "end_time": datetime.now().isoformat(),
                "total_epochs": getattr(self, "current_epoch", 0) + 1,
                "total_steps": getattr(self, "global_step", 0),
                "failures": len(self.failure_history),
                "recovery_attempts": self.recovery_attempts,
                "dataloaders": self.config.train_loader_config.dataloader_names,
                "sampling_strategy": self.config.train_loader_config.sampling_strategy.value,
            },
            "dataloader_statistics": dataloader_stats,
            "failure_summary": [asdict(f) for f in self.failure_history],
            "system_alerts": self.system_monitor.alerts,
        }

        report_file = (
            report_dir
            / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        try:
            with report_file.open("w") as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Training report saved: {report_file}")
        except Exception:
            self.logger.exception("Could not save training report")


def create_production_model(hidden_size: int = 512) -> nn.Module:
    """Create a production-grade model with proper initialization."""
    model = nn.Sequential(
        nn.Linear(784, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.LayerNorm(hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, 10),
    )

    # Initialize weights properly
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return model


def create_alert_callback() -> Callable:
    """Create an alert callback function for production monitoring."""

    def alert_callback(failure_record: FailureRecord):
        """Handle training alerts (integrate with monitoring systems)."""
        print(f"🚨 PRODUCTION ALERT: {failure_record.failure_type.value}")
        print(f"   DataLoader: {failure_record.dataloader_name}")
        print(f"   Message: {failure_record.message}")
        print(f"   Epoch: {failure_record.epoch}, Step: {failure_record.global_step}")

        # In production, this would integrate with:
        # - Slack/Teams notifications
        # - PagerDuty alerts
        # - Email notifications
        # - Monitoring dashboards (Grafana, etc.)

    return alert_callback


def create_production_dataloaders(
    batch_size: int = 64, num_samples: int = 1000, num_loaders: int = 1
) -> tuple[list[DataLoader], list[DataLoader]]:
    """
    Create production dataloaders for multi-loader training.

    Args:
        batch_size: Batch size per dataloader
        num_samples: Number of samples per dataset
        num_loaders: Number of dataloaders to create

    Returns:
        Tuple of (train_loaders, val_loaders) - both are lists
    """
    train_loaders = []
    val_loaders = []

    for i in range(num_loaders):
        # Create different datasets (simulating different sources)
        torch.manual_seed(42 + i)

        # Training data
        train_x = torch.randn(num_samples, 28, 28)
        train_y = torch.randint(0, 10, (num_samples,))
        train_dataset = TensorDataset(train_x.view(-1, 784), train_y)

        # Validation data
        val_x = torch.randn(num_samples // 5, 28, 28)
        val_y = torch.randint(0, 10, (num_samples // 5,))
        val_dataset = TensorDataset(val_x.view(-1, 784), val_y)

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders


def main():
    """
    Main production training function with multi-loader architecture.

    Demonstrates comprehensive fault-tolerant training suitable for
    production environments with multiple data sources.
    """
    print("🏭 Fault-Tolerant Production Training with Multi-DataLoader Architecture")
    print("=" * 70)

    # Setup paths
    project_root = Path.cwd()
    checkpoint_dir = project_root / "production_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Create production model
    print("\n🔧 Setting up production model...")
    model = create_production_model(hidden_size=512)

    # Production optimizers (always a list for multi-loader)
    optimizers = [
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,  # Conservative learning rate
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    ]

    # Determine number of dataloaders (simulate multiple data sources)
    num_dataloaders = 3  # Primary, auxiliary, and augmented data

    # Production trainer configuration with multi-loader
    print("\n📋 Creating multi-loader configuration...")
    trainer_config = GenericTrainerConfig(
        # Multi-loader configuration (required)
        train_loader_config=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            dataloader_weights=[0.5, 0.3, 0.2],  # Primary gets 50%, aux 30%, aug 20%
            epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
            steps_per_epoch=500,  # Fixed steps for consistent epochs
            dataloader_names=["primary", "auxiliary", "augmented"],
        ),
        val_loader_config=MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.ROUND_ROBIN,
            dataloader_names=["val_primary", "val_auxiliary", "val_augmented"],
        ),
        # Checkpoint configuration
        checkpoint=CheckpointConfig(
            root_dir=checkpoint_dir,
            save_every_n_epochs=1,  # Frequent checkpoints for fault tolerance
            save_every_n_steps=100,
            max_checkpoints=10,
            save_rng=True,
            save_optimizer=True,
        ),
        # Fault tolerance configuration
        fault_tolerance=FaultToleranceConfig(
            save_sampler_state=True,
            save_dataset_state=True,
            verify_deterministic_resume=True,
        ),
        # Preemption configuration
        preemption=PreemptionConfig(
            max_checkpoint_sec=300.0,
            requeue_job=False,
            resume_from_latest_symlink=True,
        ),
        # Performance configuration
        performance=PerformanceConfig(
            gradient_accumulation_steps=2,
            use_amp=torch.cuda.is_available(),
            clip_grad_norm=1.0,
        ),
        # Logging configuration
        logging=LoggingConfig(
            logger_type="console",
            log_per_loader_metrics=True,  # Track each data source
            log_loader_proportions=True,  # Monitor sampling balance
            log_scalars_every_n_steps=10,
        ),
        # Validation configuration
        validation=ValidationConfig(
            frequency=ValidationFrequency.EVERY_N_STEPS,
            every_n_steps=50,
            aggregation=ValAggregation.MICRO_AVG_WEIGHTED_BY_SAMPLES,
        ),
    )

    print(
        f"   Sampling Strategy: {trainer_config.train_loader_config.sampling_strategy.value}"
    )
    print(
        f"   Train DataLoaders: {trainer_config.train_loader_config.dataloader_names}"
    )
    print(f"   Weights: {trainer_config.train_loader_config.dataloader_weights}")

    # Create alert callback
    alert_callback = create_alert_callback()

    # Initialize production trainer with multi-loader support
    trainer = ProductionTrainer(
        config=trainer_config,
        model=model,
        optimizers=optimizers,  # List of optimizers
        alert_callbacks=[alert_callback],
    )

    # Create production datasets (multiple sources)
    print("\n📊 Creating production datasets...")
    train_loaders, val_loaders = create_production_dataloaders(
        batch_size=64, num_samples=1000, num_loaders=num_dataloaders
    )

    print(f"   Created {len(train_loaders)} training dataloaders")
    print(f"   Created {len(val_loaders)} validation dataloaders")
    for i, name in enumerate(trainer_config.train_loader_config.dataloader_names):
        print(
            f"     - {name}: {len(train_loaders[i])} train batches, "
            f"{len(val_loaders[i])} val batches"
        )

    # Setup failure simulation for demonstration
    def simulate_random_failure():
        """Simulate random failures for demonstration."""

        def failure_thread():
            time.sleep(30)  # Wait 30 seconds
            print("\n💥 Simulating production failure...")
            os.kill(os.getpid(), signal.SIGUSR1)

        thread = threading.Thread(target=failure_thread, daemon=True)
        thread.start()
        return thread

    # Start failure simulation (for demo; not used in production)
    # Example: call simulate_random_failure() to trigger SIGUSR1

    print("\n🚀 Starting fault-tolerant production training...")
    print("   Features enabled:")
    print("   • Multi-dataloader training with weighted sampling")
    print("   • Comprehensive error handling and recovery")
    print("   • System resource monitoring")
    print("   • Per-dataloader failure tracking")
    print("   • Emergency checkpointing")
    print("   • Deterministic resume capability")

    try:
        # Note: In production, you would call trainer.fit()
        # For demo purposes, we'll simulate a few steps

        print("\n📈 Training simulation...")

        # Simulate training steps for each dataloader
        for epoch in range(2):
            print(f"\nEpoch {epoch + 1}")

            for batch_idx in range(5):  # Simulate 5 batches
                # Simulate sampling from different dataloaders
                dataloader_idx = batch_idx % num_dataloaders
                dataloader_name = trainer_config.train_loader_config.dataloader_names[
                    dataloader_idx
                ]

                # Get a batch from the selected dataloader
                batch = next(iter(train_loaders[dataloader_idx]))

                # Simulate training step
                metrics = trainer.training_step(
                    trainer, batch, batch_idx, dataloader_idx, dataloader_name
                )

                print(
                    f"   Step {batch_idx + 1}: {dataloader_name} - "
                    f"Loss: {metrics[f'{dataloader_name}/loss']:.4f}, "
                    f"Acc: {metrics[f'{dataloader_name}/accuracy']:.4f}"
                )

        print("\n✅ Production training simulation completed successfully!")

        # In real production, call trainer.fit(train_loaders, val_loaders, max_epochs)

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted - demonstrating graceful shutdown")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("💾 Check checkpoint directory for recovery information")

    finally:
        # Always generate final report
        trainer.generate_training_report()

        # Display failure statistics
        print("\n📊 Training Statistics:")
        print(f"   Failures handled: {len(trainer.failure_history)}")
        print(f"   Recovery attempts: {trainer.recovery_attempts}")
        print(f"   System alerts: {len(trainer.system_monitor.alerts)}")

        # Display per-dataloader statistics
        if trainer.dataloader_failure_counts:
            print("   Failures per dataloader:")
            for loader_name, count in trainer.dataloader_failure_counts.items():
                print(f"     {loader_name}: {count}")

        if trainer.failure_history:
            print("   Failure types:")
            failure_types = {}
            for failure in trainer.failure_history:
                failure_type = failure.failure_type.value
                failure_types[failure_type] = failure_types.get(failure_type, 0) + 1

            for failure_type, count in failure_types.items():
                print(f"     {failure_type}: {count}")

    print(f"\n💾 Check {checkpoint_dir} for:")
    print("   - Training checkpoints")
    print("   - Failure logs")
    print("   - Training reports")
    print("   - System monitoring data")

    print("\n🎯 Key Multi-Loader Features Demonstrated:")
    print("   • Always use lists: optimizers=[optimizer]")
    print("   • Always use lists: train_loaders=[loader1, loader2, ...]")
    print("   • Training step: (trainer, batch, dataloader_idx, dataloader_name)")
    print("   • Per-dataloader failure tracking and recovery")
    print("   • Weighted sampling from multiple data sources")


if __name__ == "__main__":
    main()
