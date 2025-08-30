# Installation Guide

## Prerequisites

- Python 3.12 or higher
- PyTorch 2.0.0 or higher
- CUDA (optional, for GPU support)

## Installation Methods

### Install from Source

```bash
# Clone the repository
git clone https://github.com/example/model-training-framework.git
cd model-training-framework

# Install in development mode
pip install -e .
```

### Install with Optional Dependencies

```bash
# Install with all optional dependencies
pip install -e ".[all]"

# Or install specific extras
pip install -e ".[dev]"           # Development tools
pip install -e ".[tensorboard]"   # TensorBoard logging
pip install -e ".[wandb]"         # Weights & Biases integration
pip install -e ".[docs]"          # Documentation tools
```

## Core Dependencies

- `torch>=2.0.0` - PyTorch framework
- `lightning>=2.0.0` - PyTorch Lightning
- `lightning-fabric>=2.0.0` - Lightning Fabric for distributed training
- `tensorboard>=2.10.0` - TensorBoard logging
- `numpy>=1.21.0` - Numerical operations
- `pyyaml>=6.0` - Configuration files
- `gitpython>=3.1.0` - Git integration

## Local Development Setup

### Quick Start

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install -U pip

# Install with development dependencies
pip install -e ".[dev]"

# Run checks
ruff check . && ruff format --check .
mypy model_training_framework/
pytest -vv --maxfail=1
```

### CI & Testing

Primary CI runs on CircleCI and includes:

- Linting (ruff)
- Type checking (mypy)
- Unit tests (pytest)
- Security checks
- Documentation build

GitHub Actions workflows are available for manual runs and additional tooling.

### Coverage Configuration

For code coverage with Codecov:

1. CI produces `coverage.xml` automatically
2. Add `CODECOV_TOKEN` to CircleCI project settings (Project → Environment Variables)
3. Coverage reports upload automatically if token is present

### CircleCI Setup

Maintainers should:

1. Enable the repository in CircleCI
2. Configure environment variables as needed
3. Ensure PR/branch builds are triggered automatically

## Verifying Installation

After installation, verify everything is working:

```python
# Test import
python -c "from model_training_framework import ModelTrainingFramework; print('Installation successful!')"

# Run tests
pytest model_training_framework/tests/

# Check CLI tools (if installed)
python -m model_training_framework.scripts.train --help
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed correctly

   ```bash
   pip install -e ".[all]"
   ```

2. **CUDA issues**: Verify PyTorch is installed with CUDA support

   ```python
   import torch
   print(torch.cuda.is_available())
   ```

3. **Permission errors on Windows**: Run terminal as administrator or use developer mode for symlinks

4. **Missing dependencies**: Install the appropriate extra

   ```bash
   # For Weights & Biases
   pip install -e ".[wandb]"

   # For TensorBoard
   pip install -e ".[tensorboard]"
   ```

## Next Steps

After installation:

1. Read the [Quick Start Guide](../README.md#quick-start)
2. Review [Configuration Guide](CONFIGURATION.md)
3. Try the [examples](../demo/example3_production/)
