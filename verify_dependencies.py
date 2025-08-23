#!/usr/bin/env python3
"""Verify all dependencies are properly installed."""

import sys
from importlib import import_module


def verify_import(module_name: str, package_name: str | None = None) -> bool:
    """Try to import a module and return success status."""
    try:
        import_module(module_name)
        print(f"✓ {package_name or module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {package_name or module_name}: {e}")
        return False


def main():
    """Verify all dependencies."""
    print("Verifying Model Training Framework dependencies...\n")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("lightning_fabric", "Lightning Fabric"),
        ("yaml", "PyYAML"),
        ("git", "GitPython"),
        ("numpy", "NumPy"),
        ("dataclasses_json", "dataclasses-json"),
        ("typing_extensions", "typing-extensions"),
        ("colorlog", "colorlog"),
    ]
    
    optional_dependencies = [
        ("wandb", "Weights & Biases"),
        ("pytest", "pytest"),
        ("ruff", "ruff"),
    ]
    
    print("Core dependencies:")
    all_core_ok = all(verify_import(mod, name) for mod, name in dependencies)
    
    print("\nOptional dependencies:")
    for mod, name in optional_dependencies:
        verify_import(mod, name)
    
    print("\nVerifying package import:")
    package_ok = verify_import("model_training_framework", "Model Training Framework")
    
    if all_core_ok and package_ok:
        print("\n✅ All core dependencies verified successfully!")
        return 0
    else:
        print("\n❌ Some dependencies are missing. Please run: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())