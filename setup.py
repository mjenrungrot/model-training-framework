"""
Setup script for Model Training Framework.

This package provides comprehensive model training, launching, and configuration
management with advanced parameter grid search capabilities and SLURM integration.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


# Read requirements
def read_requirements(filename):
    """Read requirements from a file."""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with requirements_path.open(encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


# Core requirements
install_requires = [
    "torch>=1.12.0",
    "lightning-fabric>=1.8.0",
    "pyyaml>=6.0",
    "gitpython>=3.1.0",
    "numpy>=1.21.0",
    "dataclasses-json>=0.5.0",
    "typing-extensions>=4.0.0",
]

# Development requirements
dev_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.8.0",
    "ruff>=0.1.0",
    "mypy>=0.991",
    "pre-commit>=2.20.0",
]

# Documentation requirements
docs_requires = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=0.18.0",
]

# Optional requirements
extras_require = {
    "dev": dev_requires,
    "docs": docs_requires,
    "wandb": ["wandb>=0.13.0"],
    "tensorboard": ["tensorboard>=2.10.0"],
    "all": dev_requires + docs_requires + ["wandb>=0.13.0", "tensorboard>=2.10.0"],
}

setup(
    name="model-training-framework",
    version="1.0.0",
    author="Model Training Framework Team",
    author_email="contact@modeltrainingframework.com",
    description="Comprehensive model training, launching, and configuration management framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjenrungrot/model-training-framework",
    project_urls={
        "Bug Tracker": "https://github.com/mjenrungrot/model-training-framework/issues",
        "Documentation": "https://model-training-framework.readthedocs.io/",
        "Source Code": "https://github.com/mjenrungrot/model-training-framework",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    package_data={
        "model_training_framework": [
            "py.typed",  # PEP 561 marker file for type information
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "mtf-train=model_training_framework.scripts.train:main",
            "mtf-grid-search=model_training_framework.scripts.grid_search:main",
            "mtf-submit=model_training_framework.scripts.submit:main",
            "mtf-status=model_training_framework.scripts.status:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine learning",
        "deep learning",
        "model training",
        "hyperparameter optimization",
        "slurm",
        "distributed training",
        "configuration management",
        "experiment management",
    ],
)
