"""
Fault-Tolerant Training - Advanced Production Pipeline

This example demonstrates production-grade fault-tolerant training with
comprehensive error handling, monitoring, and recovery mechanisms. Perfect for:

- Production ML pipelines requiring high reliability
- Long-running training jobs that must handle various failure modes
- Systems requiring comprehensive logging and monitoring
- Enterprise deployments with strict uptime requirements

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

from model_training_framework.trainer import (
    GenericTrainer,
    GenericTrainerConfig,
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
    Production-grade trainer with comprehensive fault tolerance.

    This trainer extends the GenericTrainer with enterprise-level
    fault tolerance, monitoring, and recovery capabilities.
    """

    def __init__(
        self,
        config: GenericTrainerConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module | None = None,
        alert_callbacks: list[Callable] | None = None,
    ):
        """
        Initialize production trainer.

        Args:
            config: Trainer configuration
            model: Model to train
            optimizer: Optimizer
            loss_fn: Loss function
            alert_callbacks: List of functions to call on alerts
        """
        super().__init__(config)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.alert_callbacks = alert_callbacks or []

        # Failure tracking
        self.failure_history: list[FailureRecord] = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

        # System monitoring
        self.system_monitor = SystemMonitor()

        # Performance tracking
        self.performance_history: list[dict] = []
        self.last_loss_values: list[float] = []
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
        log_dir = Path(self.config.checkpoint_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure main logger
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger("ProductionTrainer")
        self.logger.info("Production trainer logging initialized")

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.warning(
                f"Received signal {signum}, initiating graceful shutdown..."
            )
            self.shutdown_requested = True
            self.handle_failure(FailureType.PREEMPTION, f"Signal {signum} received")

        # Handle common signals
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)

    def training_step(self, batch, batch_idx: int) -> dict[str, Any]:
        """
        Enhanced training step with comprehensive error handling.

        Args:
            batch: Training batch
            batch_idx: Batch index

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
                )

            # Memory check before processing
            if torch.cuda.is_available():
                torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9

                if memory_reserved > 0.95 * torch.cuda.max_memory_allocated() / 1e9:
                    self.logger.warning(
                        f"GPU memory usage high: {memory_reserved:.1f}GB"
                    )
                    torch.cuda.empty_cache()

            # Forward pass with error handling
            x, y = batch

            try:
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.handle_failure(FailureType.OOM, str(e))
                    raise
                self.handle_failure(FailureType.UNKNOWN, f"Forward pass error: {e}")
                raise

            # Check for loss divergence
            if torch.isnan(loss) or torch.isinf(loss):
                self.handle_failure(
                    FailureType.MODEL_DIVERGENCE, f"Loss is NaN or Inf: {loss.item()}"
                )
                raise RuntimeError("Model divergence detected")

            # Track loss history for divergence detection
            loss_value = loss.item()
            self.last_loss_values.append(loss_value)
            if len(self.last_loss_values) > 10:
                self.last_loss_values = self.last_loss_values[-10:]

            # Check for sudden loss increase
            if len(self.last_loss_values) >= 5:
                recent_avg = sum(self.last_loss_values[-3:]) / 3
                older_avg = sum(self.last_loss_values[-6:-3]) / 3

                if recent_avg > older_avg * self.divergence_threshold:
                    self.logger.warning(
                        f"Potential divergence detected: recent={recent_avg:.4f}, "
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
                "accuracy": accuracy,
                "lr": self.optimizer.param_groups[0]["lr"],
                "batch_idx": batch_idx,
                "step_duration": step_duration,
                "gpu_memory_gb": torch.cuda.memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0,
            }

            # Track performance
            self.performance_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "loss": loss_value,
                    "accuracy": accuracy,
                    "step_duration": step_duration,
                }
            )

            return metrics

        except Exception:
            self.logger.exception("Error in training step")
            self.logger.exception(traceback.format_exc())
            raise

    def validation_step(self, batch, batch_idx: int) -> dict[str, Any]:
        """Enhanced validation step with error handling."""
        try:
            with torch.no_grad():
                x, y = batch
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)

                _, predicted = torch.max(y_pred, 1)
                accuracy = (predicted == y).float().mean().item()

                return {
                    "val_loss": loss,
                    "val_accuracy": accuracy,
                }

        except Exception:
            self.logger.exception("Error in validation step")
            raise

    def handle_failure(
        self,
        failure_type: FailureType,
        message: str,
        recovery_action: str = "checkpoint_and_continue",
    ) -> None:
        """
        Handle training failures with comprehensive logging and recovery.

        Args:
            failure_type: Type of failure
            message: Failure message
            recovery_action: Action taken for recovery
        """
        failure_record = FailureRecord(
            timestamp=datetime.now(),
            failure_type=failure_type,
            message=message,
            epoch=getattr(self, "current_epoch", -1),
            global_step=getattr(self, "global_step", -1),
            phase=getattr(self, "current_phase", "unknown"),
            recovery_action=recovery_action,
            stack_trace=traceback.format_exc(),
            system_stats=self.system_monitor.get_system_stats(),
        )

        self.failure_history.append(failure_record)
        self.recovery_attempts += 1

        # Log failure
        self.logger.error("Training failure detected:")
        self.logger.error(f"  Type: {failure_type.value}")
        self.logger.error(f"  Message: {message}")
        self.logger.error(f"  Recovery attempts: {self.recovery_attempts}")

        # Save failure record
        failure_log = Path(self.config.checkpoint_dir) / "failures.json"
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

        self.logger.info("Production training started")
        self.logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
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

    def generate_training_report(self) -> None:
        """Generate comprehensive training report."""
        report_dir = Path(self.config.checkpoint_dir) / "reports"
        report_dir.mkdir(exist_ok=True)

        report = {
            "training_summary": {
                "start_time": getattr(self, "training_start_time", "unknown"),
                "end_time": datetime.now().isoformat(),
                "total_epochs": getattr(self, "current_epoch", 0) + 1,
                "total_steps": getattr(self, "global_step", 0),
                "failures": len(self.failure_history),
                "recovery_attempts": self.recovery_attempts,
            },
            "failure_summary": [asdict(f) for f in self.failure_history],
            "performance_summary": {
                "avg_step_duration": np.mean(
                    [p["step_duration"] for p in self.performance_history]
                )
                if self.performance_history
                else 0,
                "total_performance_records": len(self.performance_history),
            },
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
        print(f"   Message: {failure_record.message}")
        print(f"   Epoch: {failure_record.epoch}, Step: {failure_record.global_step}")

        # In production, this would integrate with:
        # - Slack/Teams notifications
        # - PagerDuty alerts
        # - Email notifications
        # - Monitoring dashboards (Grafana, etc.)

        # Example integration points would be implemented here

    return alert_callback


def main():
    """
    Main production training function.

    Demonstrates comprehensive fault-tolerant training suitable for
    production environments.
    """
    print("🏭 Fault-Tolerant Production Training")
    print("=" * 50)

    # Setup paths
    project_root = Path.cwd()
    checkpoint_dir = project_root / "production_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Create production model
    print("🔧 Setting up production model...")
    model = create_production_model(hidden_size=512)

    # Production optimizer with proper settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Conservative learning rate
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Production trainer configuration
    trainer_config = GenericTrainerConfig(
        max_epochs=100,
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_epochs=1,  # Frequent checkpoints for fault tolerance
        preemption_timeout=300,  # 5 minutes
        save_rng_state=True,
        enable_progress_bar=True,
    )

    # Create alert callback
    alert_callback = create_alert_callback()

    # Initialize production trainer
    trainer = ProductionTrainer(
        config=trainer_config,
        model=model,
        optimizer=optimizer,
        alert_callbacks=[alert_callback],
    )

    # Create production dataset (larger for extended training)
    print("📊 Creating production dataset...")

    def create_production_data(batch_size=64, num_batches=500):
        """Create production-scale dummy data."""
        data = []
        for _i in range(num_batches):
            x = torch.randn(batch_size, 28, 28)
            y = torch.randint(0, 10, (batch_size,))
            data.append((x, y))
        return data

    train_data = create_production_data(batch_size=64, num_batches=500)
    val_data = create_production_data(batch_size=64, num_batches=100)

    print(f"   Training batches: {len(train_data)}")
    print(f"   Validation batches: {len(val_data)}")

    # Setup failure simulation for demonstration
    def simulate_random_failure():
        """Simulate random failures for demonstration."""

        def failure_thread():
            time.sleep(60)  # Wait 1 minute
            print("\n💥 Simulating production failure...")
            os.kill(os.getpid(), signal.SIGUSR1)

        thread = threading.Thread(target=failure_thread, daemon=True)
        thread.start()
        return thread

    # Start failure simulation
    simulate_random_failure()

    print("\n🚀 Starting fault-tolerant production training...")
    print("   Training will simulate various failure scenarios")
    print("   Monitor logs for failure handling and recovery")

    try:
        trainer.fit(
            model=model,
            optimizer=optimizer,
            train_loader=train_data,
            val_loader=val_data,
        )

        print("✅ Production training completed successfully!")

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


if __name__ == "__main__":
    # Add numpy import for compatibility
    try:
        import numpy as np
    except ImportError:
        print("NumPy not available, some features may be limited")

        class MockNumpy:
            @staticmethod
            def mean(data):
                return sum(data) / len(data) if data else 0

        np = MockNumpy()

    main()
