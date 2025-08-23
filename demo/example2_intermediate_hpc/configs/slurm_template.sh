#!/bin/bash
# shellcheck disable=SC1083,SC1056,SC1072,SC1054,SC1073,SC1009
#SBATCH --job-name={{job_name}}
#SBATCH --time={{time_limit}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --cpus-per-task={{cpus_per_task}}
#SBATCH --mem={{mem}}
#SBATCH --gres={{gres}}
#SBATCH --partition={{partition}}
{{#account}}#SBATCH --account={{account}}{{/account}}
{{#constraint}}#SBATCH --constraint={{constraint}}{{/constraint}}
#SBATCH --output={{output_dir}}/{{experiment_name}}_%j.out
#SBATCH --error={{output_dir}}/{{experiment_name}}_%j.err

# =============================================================================
# SLURM Template for Intermediate HPC Training
# This template provides a robust setup for HPC training with proper
# environment configuration and error handling.
# =============================================================================

echo "=========================================="
echo "SLURM Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="

# Environment setup
source ~/.bashrc

# Activate conda environment if specified
{{#conda_env}}
echo "Activating conda environment: {{conda_env}}"
source activate {{conda_env}}
{{/conda_env}}

# Set up Python path
export PYTHONPATH={{project_root}}:$PYTHONPATH

# Change to project directory
cd {{project_root}}

# Print environment information
echo "Environment Information:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo "PyTorch CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
else
    echo "No GPU support detected"
fi

echo "=========================================="

# Create output directory if it doesn't exist
mkdir -p {{output_dir}}/{{experiment_name}}

# Set up signal handling for graceful shutdown
cleanup() {
    echo "Received termination signal. Cleaning up..."
    # Save any intermediate results
    if [ -f "{{output_dir}}/{{experiment_name}}/training.pid" ]; then
        PID=$(cat {{output_dir}}/{{experiment_name}}/training.pid)
        echo "Sending SIGUSR1 to training process $PID for graceful shutdown..."
        kill -USR1 $PID 2>/dev/null || true
        sleep 30  # Give time for graceful shutdown
        kill -TERM $PID 2>/dev/null || true
    fi
    echo "Cleanup completed at: $(date)"
}

# Set signal handlers
trap cleanup SIGTERM SIGINT

# Start training
echo "Starting training with configuration: {{config_path}}"
echo "Experiment name: {{experiment_name}}"
echo "Output directory: {{output_dir}}"

# Save process ID for signal handling
echo $$ > {{output_dir}}/{{experiment_name}}/training.pid

# Run the training command
python -m model_training_framework.scripts.train \
    --config {{config_path}} \
    --experiment-name {{experiment_name}} \
    --output-dir {{output_dir}} \
    {{#additional_args}}{{additional_args}}{{/additional_args}}

# Capture exit code
EXIT_CODE=$?

# Remove PID file
rm -f {{output_dir}}/{{experiment_name}}/training.pid

# Print completion information
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: {{output_dir}}/{{experiment_name}}"
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo "Check error logs for details"
fi

echo "End time: $(date)"
echo "Job duration: $SECONDS seconds"
echo "=========================================="

# Copy important logs to a summary file
{
    echo "Job Summary for {{experiment_name}}"
    echo "======================================"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exit Code: $EXIT_CODE"
    echo "Duration: $SECONDS seconds"
    echo "Node: $SLURMD_NODENAME"
    echo "Start: $(date -d @$SLURM_JOB_START_TIME)"
    echo "End: $(date)"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Status: SUCCESS"
    else
        echo "Status: FAILED"
    fi
} > {{output_dir}}/{{experiment_name}}/job_summary.txt

exit $EXIT_CODE
