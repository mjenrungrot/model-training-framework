#!/bin/bash
# shellcheck disable=SC1009,SC1054,SC1073,SC1072,SC1083,SC1056
#SBATCH --job-name={{job_name}}
#SBATCH --time={{time_limit}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --cpus-per-task={{cpus_per_task}}
#SBATCH --mem={{mem}}
#SBATCH --gres={{gres}}
#SBATCH --partition={{partition}}
{{#exclusive}}#SBATCH --exclusive{{/exclusive}}
{{#account}}#SBATCH --account={{account}}{{/account}}
#SBATCH --output={{output_dir}}/{{experiment_name}}_%j.out
#SBATCH --error={{output_dir}}/{{experiment_name}}_%j.err

# Print job information
echo "=========================================="
echo "SLURM Job Information:"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Node list: $SLURM_JOB_NODELIST"
echo "=========================================="

# Environment setup
source ~/.bashrc
# module load cuda/11.8  # Adjust for your system
# source activate distributed_env

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# NCCL configuration
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# CUDA configuration
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "Distributed Environment:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "=========================================="

# Change to project directory
cd {{project_root}}

# Run distributed training with multi-loader architecture
srun python demo/example2_intermediate_hpc/distributed_train_script.py \
    --config-name {{config_name}} \
    --num-loaders {{num_loaders}} \
    --batch-size {{batch_size}} \
    --learning-rate {{learning_rate}} \
    --max-epochs {{max_epochs}} \
    --output-dir {{output_dir}}/{{experiment_name}}

echo "=========================================="
echo "Job completed at: $(date)"
