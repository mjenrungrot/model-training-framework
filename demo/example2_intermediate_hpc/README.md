# Example 2: Intermediate HPC Usage
<!-- markdownlint-disable MD029 -->

Welcome to the intermediate HPC scenario! This example demonstrates how to scale your training to High-Performance Computing (HPC) environments with SLURM job scheduling, distributed training, and comprehensive hyperparameter optimization.

## 🎯 Target Audience

- **Researchers** with HPC cluster access
- **Teams** scaling experiments beyond single machines
- **Data scientists** running systematic hyperparameter optimization
- **ML engineers** implementing distributed training workflows

## 📚 What You'll Learn

- SLURM job submission and management
- Multi-node distributed training setup
- Systematic hyperparameter optimization
- Resource allocation and optimization
- Job monitoring and result analysis
- Production-grade experiment tracking

## 🗂️ Directory Structure

```text
example2_intermediate_hpc/
├── README.md                           # This guide
├── hyperparameter_optimization.py     # Grid search for HPC
├── distributed_training.py            # Multi-node training
├── configs/                           # Configuration templates
│   ├── slurm_template.sh              # Basic SLURM template
│   ├── grid_search_config.yaml        # Grid search configuration
│   └── distributed_config.yaml        # Distributed training config
└── utils/                             # Monitoring and analysis tools
    ├── job_monitoring.py              # SLURM job monitoring
    └── result_analysis.py             # Result analysis utilities
```

## 🚀 Quick Start

### Prerequisites

1. **HPC Environment Requirements:**

   ```bash
   # Must have access to SLURM cluster
   which sbatch squeue scancel

   # GPU nodes available
   sinfo -p gpu
   ```

2. **Framework Installation on Cluster:**

   ```bash
   # On compute nodes
   module load python/3.8+
   pip install --user -e /path/to/model_training_framework
   pip install --user torch torchvision
   ```

### Run Hyperparameter Optimization

1. **Navigate to HPC demo:**

   ```bash
   cd demo/example2_intermediate_hpc
   ```

2. **Configure your search space:**

Edit [`configs/grid_search_config.yaml`](configs/grid_search_config.yaml) to match your requirements.

3. **Run optimization (dry-run first):**

   ```bash
   python hyperparameter_optimization.py
   ```

4. **Submit to SLURM:**
   Uncomment the SLURM submission section in the script.

## 📖 Detailed Scenarios

### Scenario A: Systematic Hyperparameter Optimization

[`hyperparameter_optimization.py`](hyperparameter_optimization.py) demonstrates research-grade hyperparameter exploration:

#### Key Features — Hyperparameter Optimization

- **Comprehensive Parameter Grids**: Learning rates, model architectures, training configurations
- **Resource-Aware Scheduling**: Automatic memory and time allocation based on model size
- **Validation and Monitoring**: Pre-flight checks and progress tracking
- **Result Analysis**: Statistical analysis and visualization

#### Example Grid Search

```python
# Learning rate and optimization grid
lr_grid = grid_search.create_grid("learning_rate_optimization")
lr_grid.add_parameter("training.learning_rate", [1e-4, 3e-4, 1e-3, 3e-3])
lr_grid.add_parameter("optimizer.weight_decay", [0.001, 0.01, 0.1])
lr_grid.add_parameter("training.gradient_accumulation_steps", [1, 2, 4])

# Model architecture grid
model_grid = grid_search.create_grid("model_architecture")
model_grid.add_parameter("model.hidden_size", [256, 512, 768, 1024])
model_grid.add_parameter("model.num_layers", [4, 6, 8, 12])
```

#### Resource Scaling

```python
def setup_resource_allocation(experiment_config):
    hidden_size = experiment_config.model.hidden_size

    # Dynamic memory allocation
    memory_gb = max(32, int(hidden_size / 64) * 8)

    # Dynamic time limits
    time_limit = "48:00:00" if hidden_size >= 1024 else "24:00:00"

    return memory_gb, time_limit
```

### Scenario B: Multi-Node Distributed Training

[`distributed_training.py`](distributed_training.py) shows how to scale training across multiple compute nodes:

#### Key Features — Distributed Training

- **Multi-Node Setup**: 1-4 nodes with 4-16 GPUs total
- **Communication Optimization**: NCCL backend with InfiniBand support
- **Fault Tolerance**: Comprehensive error handling and recovery
- **Performance Monitoring**: GPU utilization and communication efficiency

#### Example Configurations

**Single-Node Multi-GPU (4 GPUs):**

```yaml
slurm:
  nodes: 1
  ntasks_per_node: 4
  cpus_per_task: 8
  mem: "128G"
  gres: "gpu:4"
```

**Multi-Node Scaling (2 nodes, 8 GPUs):**

```yaml
slurm:
  nodes: 2
  ntasks_per_node: 4
  cpus_per_task: 10
  mem: "256G"
  gres: "gpu:4"
  exclusive: true
```

**Large-Scale Training (4 nodes, 16 GPUs):**

```yaml
slurm:
  nodes: 4
  ntasks_per_node: 4
  cpus_per_task: 12
  mem: "512G"
  gres: "gpu:4"
```

#### Distributed Training Environment

```bash
# Automatic environment setup in SLURM template
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# NCCL optimization
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_DISABLE=0
```

## 🔧 Configuration Deep Dive

### Grid Search Configuration

[`configs/grid_search_config.yaml`](configs/grid_search_config.yaml) provides a template for systematic hyperparameter exploration:

```yaml
# Base model configuration
model:
  name: "transformer"
  hidden_size: 512          # Will be varied
  num_layers: 6             # Will be varied
  num_heads: 8              # Will be varied

# Grid search parameters
grid_search:
  learning_rates: [1e-4, 3e-4, 1e-3, 3e-3]
  weight_decays: [0.001, 0.01, 0.1]
  hidden_sizes: [256, 512, 768, 1024]
  num_layers: [4, 6, 8, 12]

# Resource scaling based on model size
resource_scaling:
  memory_scaling:
    256: 16    # 16GB for small models
    512: 32    # 32GB for medium models
    1024: 64   # 64GB for large models
```

### Distributed Training Configuration

[`configs/distributed_config.yaml`](configs/distributed_config.yaml) optimizes for multi-node training:

```yaml
# Large model for distributed training
model:
  hidden_size: 1024
  num_layers: 24
  gradient_checkpointing: true

# Distributed-specific settings
distributed:
  backend: "nccl"
  init_method: "env://"
  timeout: 1800
  find_unused_parameters: false

# Performance optimization
performance:
  mixed_precision: true
  compile_model: false        # Disable for distributed
  dataloader_num_workers: 8
```

### SLURM Template Customization

[`configs/slurm_template.sh`](configs/slurm_template.sh) provides a robust template for HPC job submission:

```bash
#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --time={{time_limit}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --cpus-per-task={{cpus_per_task}}
#SBATCH --mem={{mem}}
#SBATCH --gres={{gres}}
#SBATCH --partition={{partition}}

# Environment setup with error handling
source ~/.bashrc
{{#conda_env}}source activate {{conda_env}}{{/conda_env}}

# Comprehensive system information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
nvidia-smi

# Graceful shutdown handling
cleanup() {
    echo "Received termination signal. Cleaning up..."
    kill -USR1 $PID 2>/dev/null || true
    wait $PID
}
trap cleanup SIGTERM SIGINT

# Run training with monitoring
python -m model_training_framework.scripts.train \
    --config {{config_path}} \
    --experiment-name {{experiment_name}} \
    --output-dir {{output_dir}} &
PID=$!
wait $PID
```

## 📊 Monitoring and Analysis

### Job Monitoring

[`utils/job_monitoring.py`](utils/job_monitoring.py) provides comprehensive job tracking:

```python
# Monitor all jobs
monitor = SLURMJobMonitor()
jobs = monitor.get_job_status()

# Get detailed job information
efficiency = monitor.get_job_efficiency(job_id)
print(f"CPU Efficiency: {efficiency.cpu_efficiency:.1f}%")
print(f"Memory Efficiency: {efficiency.memory_efficiency:.1f}%")

# Real-time monitoring
monitor.monitor_jobs_realtime(refresh_interval=30)
```

### Result Analysis

[`utils/result_analysis.py`](utils/result_analysis.py) enables systematic analysis of hyperparameter experiments:

```python
# Analyze experiment results
analyzer = HyperparameterAnalyzer("./hpc_experiments")
analyzer.load_experiments()

# Parameter importance analysis
importance_df = analyzer.analyze_parameter_importance("val_loss")
print(importance_df.head(10))

# Get best performing experiments
best_experiments = analyzer.get_best_experiments("val_loss", n=5)

# Generate comprehensive report
report = analyzer.generate_analysis_report()
print(report)
```

## 🔍 Best Practices for HPC

### Resource Management

1. **Start Small, Scale Gradually:**

   ```bash
   # Test with single node first
   sbatch --nodes=1 --time=2:00:00 test_job.sh

   # Then scale to multiple nodes
   sbatch --nodes=4 --time=24:00:00 production_job.sh
   ```

2. **Use Appropriate Partitions:**

      ```bash
      # For development and testing
      #SBATCH --partition=debug
      #SBATCH --time=1:00:00

      # For production runs
      #SBATCH --partition=gpu
      #SBATCH --time=24:00:00
      ```

3. **Request Specific GPU Types:**

      ```bash
      #SBATCH --gres=gpu:v100:4    # 4x V100 GPUs
      #SBATCH --gres=gpu:a100:2    # 2x A100 GPUs
      ```

### Distributed Training Optimization

1. **Gradient Accumulation for Large Effective Batch Sizes:**

   ```yaml
   training:
     batch_size: 32             # Per GPU
     gradient_accumulation_steps: 4  # Effective batch size: 32 * 4 * num_gpus
   ```

2. **Learning Rate Scaling:**

   ```python
   # Scale learning rate with number of GPUs
   base_lr = 1e-4
   scaled_lr = base_lr * world_size
   ```

3. **Data Loading Optimization:**

   ```yaml
   data:
     num_workers: 8             # Per GPU
     pin_memory: true
     persistent_workers: true
   ```

### Experiment Organization

1. **Systematic Naming:**

   ```python
   experiment_name = f"hparam_opt_{model_size}_{datetime.now().strftime('%Y%m%d_%H%M')}"
   ```

2. **Git Integration:**

   ```python
   # Track experiment with git commit
   git_manager = GitManager(project_root)
   with git_manager.branch_context(f"experiment_{experiment_name}"):
       # Run experiments
       pass
   ```

3. **Result Aggregation:**

   ```bash
   # Collect results from all experiments
   mkdir -p results_summary
   find ./hpc_experiments -name "metrics.json" -exec cp {} results_summary/ \;
   ```

## 🚨 Troubleshooting

### Common SLURM Issues

1. **Job Stuck in Queue:**

   ```bash
   # Check job priority and resource availability
   squeue -u $USER --start
   scontrol show job $JOB_ID
   ```

2. **Out of Memory Errors:**

   ```bash
   # Check memory usage
   sacct -j $JOB_ID --format=JobID,MaxRSS,MaxVMSize

   # Increase memory allocation
   #SBATCH --mem=128G  # Instead of 64G
   ```

3. **GPU Utilization Issues:**

   ```bash
   # Monitor GPU usage during training
   ssh $NODE nvidia-smi -l 1

   # Check for GPU memory leaks
   nvidia-smi --query-gpu=memory.used --format=csv -l 1
   ```

### Distributed Training Issues

1. **NCCL Communication Errors:**

   ```bash
   # Enable detailed NCCL logging
   export NCCL_DEBUG=DETAIL

   # Check network connectivity
   export NCCL_DEBUG_SUBSYS=NET
   ```

2. **Uneven GPU Utilization:**

   ```python
   # Ensure proper data distribution
   sampler = DistributedSampler(dataset, shuffle=True)
   dataloader = DataLoader(dataset, sampler=sampler)
   ```

3. **Training Divergence:**

   ```yaml
   # Use gradient clipping
   training:
     gradient_clip_norm: 1.0

   # Reduce learning rate for large batch sizes
   optimizer:
     learning_rate: 1e-4  # Instead of 1e-3
   ```

## 📈 Performance Monitoring

### Key Metrics to Track

1. **GPU Utilization:**

   ```bash
   # Target: >90% GPU utilization
   nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1
   ```

2. **Communication Efficiency:**

   ```python
   # Monitor all-reduce times
   # Target: <10% of step time for communication
   ```

3. **Memory Usage:**

   ```bash
   # Monitor memory efficiency
   seff $JOB_ID
   ```

4. **Training Speed:**

   ```python
   # Monitor samples/second
   # Target: Linear scaling with number of GPUs (up to communication bound)
   ```

## 🎓 Next Steps

After mastering HPC workflows, consider:

### 1. **Advanced HPC Techniques**

- Array jobs for parameter sweeps
- Job dependencies and workflows
- Dynamic resource allocation
- Multi-cluster deployments

### 1. **Production Deployment** → [`../example3_advanced_production/`](../example3_advanced_production/)

- Enterprise-grade fault tolerance
- Comprehensive monitoring systems
- Production deployment pipelines
- Advanced multi-task learning

### 1. **Research Applications**

- Large language model training
- Computer vision at scale
- Scientific computing workflows
- Multi-modal model development

## 📚 Additional Resources

### HPC and SLURM

- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [HPC Best Practices](https://hpc-tutorials.llnl.gov/)
- [Distributed Training Guide](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### Performance Optimization

- [PyTorch Performance Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NCCL Optimization](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [GPU Computing Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

**Ready to Scale!** 🚀 You now have the tools to run systematic experiments and scale training across HPC clusters.
