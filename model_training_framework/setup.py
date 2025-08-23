"""Setup script for Model Training Framework."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with requirements_path.open() as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "lightning>=2.0.0",
        "numpy>=1.21.0",
        "PyYAML>=6.0",
        "hydra-core>=1.3.0",
        "colorlog>=6.7.0",
        "pandas>=1.5.0",
        "typing-extensions>=4.0.0",
    ]

setup(
    name="model-training-framework",
    version="1.0.0",
    author="Model Training Framework Team",
    author_email="team@example.com",
    description="A comprehensive Python package for ML model training, launching, and configuration management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/model-training-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mtf-launcher=model_training_framework.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "model_training_framework": [
            "templates/*.txt",
            "templates/*.sbatch",
        ],
    },
    keywords="machine learning, training, slurm, distributed computing, experiment management",
    project_urls={
        "Bug Reports": "https://github.com/example/model-training-framework/issues",
        "Source": "https://github.com/example/model-training-framework",
        "Documentation": "https://model-training-framework.readthedocs.io/",
    },
)
