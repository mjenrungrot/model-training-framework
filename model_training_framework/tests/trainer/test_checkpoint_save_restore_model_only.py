"""
Unit tests for checkpoint save/restore with full state preservation.

Tests the new multi-dataloader-only checkpoint format version 1 with comprehensive
RNG state persistence, dataloader manager state, and multi-optimizer support.
"""

from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_training_framework.config.schemas import (
    CheckpointConfig,
    EpochLengthPolicy,
    MultiDataLoaderConfig,
    SamplingStrategy,
)
from model_training_framework.trainer.checkpoints import CheckpointManager
from model_training_framework.trainer.multi_dataloader import DataLoaderManager
from model_training_framework.trainer.states import create_initial_resume_state


class TestCheckpointSaveRestore:
    """Test checkpoint save/restore functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_config = CheckpointConfig(
            root_dir=self.temp_dir,
            save_optimizer=True,
            save_scheduler=True,
            max_checkpoints=5,
            save_every_n_epochs=1,
        )

        # Create model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 2),
        )

        # Create optimizers and schedulers
        self.optimizers = [
            torch.optim.SGD(self.model[0].parameters(), lr=0.01),
            torch.optim.Adam(self.model[2].parameters(), lr=0.001),
        ]

        self.schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizers[0], step_size=10),
            torch.optim.lr_scheduler.ExponentialLR(self.optimizers[1], gamma=0.9),
        ]

        # Create dataloaders
        dataset1 = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
        dataset2 = TensorDataset(torch.randn(80, 10), torch.randint(0, 2, (80,)))

        self.train_loaders = [
            DataLoader(dataset1, batch_size=10),
            DataLoader(dataset2, batch_size=10),
        ]

        # Create dataloader manager
        self.dataloader_config = MultiDataLoaderConfig(
            sampling_strategy=SamplingStrategy.WEIGHTED,
            epoch_length_policy=EpochLengthPolicy.FIXED_NUM_STEPS,
            steps_per_epoch=50,
            dataloader_names=["loader1", "loader2"],
            dataloader_weights=[0.7, 0.3],
            choice_rng_seed=42,
        )

        self.dataloader_manager = DataLoaderManager(
            train_loaders=self.train_loaders,
            val_loaders=None,
            train_config=self.dataloader_config,
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.checkpoint_config, "test_experiment"
        )

    def teardown_method(self):
        """Cleanup after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_checkpoint_format_v1(self):
        """Test saving checkpoint in format version 1."""
        # Setup some training state
        epoch = 5
        global_step = 123

        # Create resume state with dataloader manager state
        resume_state = create_initial_resume_state(save_rng=True)
        resume_state.dataloader_manager_state = self.dataloader_manager.get_state()

        # Set some RNG states to non-default values
        random.seed(12345)
        np.random.seed(54321)
        torch.manual_seed(98765)

        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            resume_state=resume_state,
            epoch=epoch,
            global_step=global_step,
            metrics={"train_loss": 0.5, "val_accuracy": 0.85},
        )

        # Verify checkpoint file exists
        assert checkpoint_path.exists()

        # Load and verify checkpoint data
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)

        # Verify format version
        assert checkpoint_data["format_version"] == 1
        # is_multi_dataloader_only is not present in the new format

        # Verify basic training state
        assert checkpoint_data["epoch"] == epoch
        assert checkpoint_data["global_step"] == global_step
        assert "model_state_dict" in checkpoint_data

        # Verify multi-optimizer states
        assert "optimizer_state_dicts" in checkpoint_data
        assert len(checkpoint_data["optimizer_state_dicts"]) == len(self.optimizers)

        # Verify multi-scheduler states
        assert "scheduler_state_dicts" in checkpoint_data
        assert len(checkpoint_data["scheduler_state_dicts"]) == len(self.schedulers)

        # Verify RNG states
        assert "rng_states" in checkpoint_data
        rng_states = checkpoint_data["rng_states"]
        assert "python_random" in rng_states
        assert "numpy_random" in rng_states
        assert "torch_cpu" in rng_states
        assert "torch_cuda" in rng_states  # May be None if no CUDA

        # Verify resume state
        assert "resume_state" in checkpoint_data
        assert checkpoint_data["resume_state"] is not None

    def test_restore_checkpoint_full_state(self):
        """Test restoring checkpoint with full state preservation."""
        # Save initial state
        epoch = 3
        global_step = 75

        # Set specific RNG states
        random.seed(11111)
        np.random.seed(22222)
        torch.manual_seed(33333)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(44444)

        # Build dataloader schedule to create some state
        _ = self.dataloader_manager.create_epoch_iterator("train", epoch)

        # Create resume state
        resume_state = create_initial_resume_state(save_rng=True)
        resume_state.dataloader_manager_state = self.dataloader_manager.get_state()

        # Get initial states for comparison
        initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        initial_optimizer_states = [opt.state_dict() for opt in self.optimizers]
        initial_scheduler_states = [sched.state_dict() for sched in self.schedulers]
        initial_python_state = random.getstate()
        initial_numpy_state = cast(
            "tuple[Any, Any, int, int, float]", np.random.get_state()
        )
        initial_torch_state = torch.get_rng_state()
        initial_dataloader_state = self.dataloader_manager.get_state()

        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            resume_state=resume_state,
            epoch=epoch,
            global_step=global_step,
        )

        # Modify states to simulate training progress
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        for optimizer in self.optimizers:
            # Simulate some optimizer steps
            dummy_loss = torch.tensor(1.0, requires_grad=True)
            optimizer.zero_grad()
            dummy_loss.backward()
            optimizer.step()

        for scheduler in self.schedulers:
            scheduler.step()

        # Change RNG states
        random.seed(99999)
        np.random.seed(88888)
        torch.manual_seed(77777)

        # Modify dataloader manager state
        _ = self.dataloader_manager.create_epoch_iterator("train", epoch + 1)

        # Restore from checkpoint
        restored_epoch, restored_global_step, restored_resume_state = (
            self.checkpoint_manager.restore_from_checkpoint(
                model=self.model,
                optimizers=self.optimizers,
                schedulers=self.schedulers,
                checkpoint_path=checkpoint_path,
            )
        )

        # Verify basic state restoration
        assert restored_epoch == epoch
        assert restored_global_step == global_step
        assert restored_resume_state is not None

        # Verify model state restoration
        restored_model_state = self.model.state_dict()
        for key, init_tensor in initial_model_state.items():
            assert torch.allclose(init_tensor, restored_model_state[key]), (
                f"Model parameter {key} not restored correctly"
            )

        # Verify optimizer state restoration
        for i, optimizer in enumerate(self.optimizers):
            restored_opt_state = optimizer.state_dict()
            initial_opt_state = initial_optimizer_states[i]

            # Compare parameter groups
            assert len(restored_opt_state["param_groups"]) == len(
                initial_opt_state["param_groups"]
            )
            for j, (restored_group, initial_group) in enumerate(
                zip(
                    restored_opt_state["param_groups"],
                    initial_opt_state["param_groups"],
                )
            ):
                for key in initial_group:
                    if key != "params":  # Skip param IDs as they might differ
                        assert restored_group[key] == initial_group[key], (
                            f"Optimizer {i} group {j} key {key} not restored correctly"
                        )

        # Verify scheduler state restoration
        for i, scheduler in enumerate(self.schedulers):
            restored_sched_state = scheduler.state_dict()
            initial_sched_state = initial_scheduler_states[i]

            for key in initial_sched_state:
                assert restored_sched_state[key] == initial_sched_state[key], (
                    f"Scheduler {i} key {key} not restored correctly"
                )

        # Verify RNG state restoration
        restored_python_state = random.getstate()
        restored_numpy_state = cast(
            "tuple[Any, Any, int, int, float]", np.random.get_state()
        )
        restored_torch_state = torch.get_rng_state()

        assert restored_python_state == initial_python_state, (
            "Python RNG state not restored"
        )
        assert restored_numpy_state[0] == initial_numpy_state[0], (
            "NumPy RNG state not restored"
        )
        assert np.array_equal(restored_numpy_state[1], initial_numpy_state[1]), (
            "NumPy RNG array not restored"
        )
        assert torch.equal(restored_torch_state, initial_torch_state), (
            "PyTorch RNG state not restored"
        )

        # Verify dataloader manager state restoration
        from model_training_framework.trainer.states import ResumeState

        if isinstance(restored_resume_state, dict):
            restored_resume_state = ResumeState.from_dict(restored_resume_state)
        if restored_resume_state.dataloader_manager_state:
            # Create new manager and load the state
            new_manager = DataLoaderManager(
                train_loaders=self.train_loaders,
                val_loaders=None,
                train_config=self.dataloader_config,
            )
            new_manager.load_state(restored_resume_state.dataloader_manager_state)

            restored_dataloader_state = new_manager.get_state()

            # Compare key state elements
            assert "train_iterator_state" in restored_dataloader_state
            tis = restored_dataloader_state["train_iterator_state"]
            assert isinstance(tis, dict)
            assert "loader_states" in tis
            assert "schedule_position" in tis

            # Verify iterator states if they exist

            if "train_iterator_state" in initial_dataloader_state:
                assert "train_iterator_state" in restored_dataloader_state
                # Could compare specific fields but iterator state structure may vary

    def test_checkpoint_backward_compatibility(self):
        """Test loading legacy checkpoint format (version 0)."""
        # Create legacy checkpoint format (simulate old format)
        legacy_checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizers[0].state_dict(),  # Single optimizer
            "scheduler_state_dict": self.schedulers[0].state_dict(),  # Single scheduler
            "epoch": 2,
            "global_step": 50,
            "experiment_name": "legacy_test",
            "save_timestamp": 1234567890.0,
            # No format_version (defaults to 0)
            # No is_multi_dataloader_only flag
            # No comprehensive RNG states
        }

        # Save legacy checkpoint
        legacy_path = Path(self.temp_dir) / "legacy.ckpt"
        torch.save(legacy_checkpoint, legacy_path)

        # Try to restore (should work with warnings)
        restored_epoch, restored_global_step, restored_resume_state = (
            self.checkpoint_manager.restore_from_checkpoint(
                model=self.model,
                optimizers=self.optimizers,
                schedulers=self.schedulers,
                checkpoint_path=legacy_path,
            )
        )

        # Verify basic restoration works
        assert restored_epoch == 2
        assert restored_global_step == 50

        # Legacy format may not have all new features
        # but should not crash

    def test_checkpoint_with_cuda_rng(self):
        """Test checkpoint save/restore with CUDA RNG states."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Move model to CUDA
        cuda_model = self.model.cuda()
        cuda_optimizers = [
            torch.optim.SGD(cuda_model[0].parameters(), lr=0.01),
            torch.optim.Adam(cuda_model[2].parameters(), lr=0.001),
        ]

        # Set CUDA RNG state
        torch.cuda.manual_seed_all(12345)
        initial_cuda_state = torch.cuda.get_rng_state_all()

        # Save checkpoint
        resume_state = create_initial_resume_state(save_rng=True)
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=cuda_model,
            optimizers=cuda_optimizers,
            resume_state=resume_state,
            epoch=1,
            global_step=25,
        )

        # Modify CUDA RNG state
        torch.cuda.manual_seed_all(54321)

        # Restore checkpoint
        self.checkpoint_manager.restore_from_checkpoint(
            model=cuda_model,
            optimizers=cuda_optimizers,
            checkpoint_path=checkpoint_path,
        )

        # Verify CUDA RNG state restoration
        restored_cuda_state = torch.cuda.get_rng_state_all()
        for original, restored in zip(initial_cuda_state, restored_cuda_state):
            assert torch.equal(original, restored), (
                "CUDA RNG state not restored correctly"
            )

    def test_checkpoint_without_schedulers(self):
        """Test checkpoint save/restore without schedulers."""
        # Save checkpoint without schedulers
        resume_state = create_initial_resume_state(save_rng=True)
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            schedulers=None,  # No schedulers
            resume_state=resume_state,
            epoch=1,
            global_step=25,
        )

        # Restore without schedulers
        restored_epoch, restored_global_step, restored_resume_state = (
            self.checkpoint_manager.restore_from_checkpoint(
                model=self.model,
                optimizers=self.optimizers,
                schedulers=None,
                checkpoint_path=checkpoint_path,
            )
        )

        assert restored_epoch == 1
        assert restored_global_step == 25

    def test_partial_optimizer_restoration(self):
        """Test restoration when checkpoint has different number of optimizers."""
        # Save checkpoint with 2 optimizers
        resume_state = create_initial_resume_state(save_rng=True)
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            resume_state=resume_state,
            epoch=1,
            global_step=25,
        )

        # Try to restore with different number of optimizers
        single_optimizer = [self.optimizers[0]]

        # Should restore successfully (may only restore first optimizer)
        restored_epoch, restored_global_step, restored_resume_state = (
            self.checkpoint_manager.restore_from_checkpoint(
                model=self.model,
                optimizers=single_optimizer,
                checkpoint_path=checkpoint_path,
            )
        )

        assert restored_epoch == 1
        assert restored_global_step == 25

    def test_deterministic_schedule_restoration(self):
        """Test that schedule determinism is preserved across checkpoint/restore."""
        # Build initial schedule
        iterator = self.dataloader_manager.create_epoch_iterator("train", 0)
        initial_schedule = iterator.schedule.copy()

        # Save state
        resume_state = create_initial_resume_state(save_rng=True)
        resume_state.dataloader_manager_state = self.dataloader_manager.get_state()

        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            resume_state=resume_state,
            epoch=0,
            global_step=0,
        )

        # Create new manager and restore state
        new_manager = DataLoaderManager(
            train_loaders=self.train_loaders,
            val_loaders=None,
            train_config=self.dataloader_config,
        )

        _, _, restored_resume_state = self.checkpoint_manager.restore_from_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            checkpoint_path=checkpoint_path,
        )

        from model_training_framework.trainer.states import ResumeState

        if isinstance(restored_resume_state, dict):
            restored_resume_state = ResumeState.from_dict(restored_resume_state)
        if restored_resume_state and restored_resume_state.dataloader_manager_state:
            new_manager.load_state(restored_resume_state.dataloader_manager_state)

        # Build schedule with restored state
        new_iterator = new_manager.create_epoch_iterator("train", 0)
        restored_schedule = new_iterator.schedule

        # Schedules should be identical
        assert initial_schedule == restored_schedule, (
            "Schedule determinism not preserved across checkpoint/restore"
        )

    def test_rng_independence_after_restore(self):
        """Test that RNG state restoration maintains independence."""
        # Set initial RNG states
        random.seed(1111)
        np.random.seed(2222)
        torch.manual_seed(3333)

        # Generate some random numbers (discard values; only mutate states)
        _ = [random.random() for _ in range(5)]
        _ = np.random.rand(5).tolist()
        _ = torch.rand(5).tolist()

        # Save checkpoint
        resume_state = create_initial_resume_state(save_rng=True)
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            resume_state=resume_state,
            epoch=0,
            global_step=0,
        )

        # Generate more random numbers (state should advance)
        advanced_random = [random.random() for _ in range(5)]
        advanced_numpy = np.random.rand(5).tolist()
        advanced_torch = torch.rand(5).tolist()

        # Restore checkpoint (should reset RNG states)
        self.checkpoint_manager.restore_from_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            checkpoint_path=checkpoint_path,
        )

        # Generate random numbers again (should match post-save values)
        restored_random = [random.random() for _ in range(5)]
        restored_numpy = np.random.rand(5).tolist()
        restored_torch = torch.rand(5).tolist()

        # Restored values should match advanced values (continuation from save point)
        assert restored_random == advanced_random, (
            "Python RNG state not properly restored"
        )
        assert np.allclose(restored_numpy, advanced_numpy), (
            "NumPy RNG state not properly restored"
        )
        assert np.allclose(restored_torch, advanced_torch), (
            "PyTorch RNG state not properly restored"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for AMP test"
    )
    def test_amp_scaler_checkpoint(self):
        """Test AMP scaler state save and restore."""
        # Create scaler and perform some updates
        scaler = torch.cuda.amp.GradScaler()

        # Simulate training steps to modify scaler state
        for _ in range(3):
            # Create dummy loss and scale it
            loss = torch.tensor(1.0, device="cuda", requires_grad=True)
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()

            # Update scaler state
            scaler.step(self.optimizers[0])
            scaler.update()
            self.optimizers[0].zero_grad()

        # Get scaler state before save
        initial_scaler_state = scaler.state_dict()
        initial_scale = initial_scaler_state["scale"]
        initial_growth_tracker = initial_scaler_state["_growth_tracker"]

        # Save checkpoint with scaler
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            scaler=scaler,
            epoch=2,
            global_step=50,
        )

        # Create new scaler and verify it has different state
        new_scaler = torch.cuda.amp.GradScaler()
        new_state = new_scaler.state_dict()
        assert new_state["scale"] != initial_scale
        assert new_state["_growth_tracker"] != initial_growth_tracker

        # Restore checkpoint with scaler
        self.checkpoint_manager.restore_from_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            scaler=new_scaler,
            checkpoint_path=checkpoint_path,
        )

        # Verify scaler state was restored
        restored_state = new_scaler.state_dict()
        assert restored_state["scale"] == initial_scale
        assert restored_state["_growth_tracker"] == initial_growth_tracker
        assert (
            restored_state["_growth_interval"]
            == initial_scaler_state["_growth_interval"]
        )
