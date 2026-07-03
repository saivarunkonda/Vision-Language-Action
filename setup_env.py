"""
Environment setup script.
Creates necessary directories and validates installation.
"""

import os
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "models/baselines",
        "logs",
        "tmp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def check_dependencies():
    """Check if key dependencies are installed."""
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
            print(f"  Number of GPUs: {gpu_count}")
            return True
        else:
            print("✗ No GPU available. Training will be very slow on CPU.")
            return False
    except:
        print("✗ Could not check GPU availability")
        return False


def main():
    print("="*50)
    print("SLM RL Reasoning - Environment Setup")
    print("="*50)
    print()
    
    # Create directories
    print("Creating directories...")
    create_directories()
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    # Check GPU
    print("Checking GPU...")
    check_gpu()
    print()
    
    if deps_ok:
        print("="*50)
        print("Setup complete! You can now:")
        print("  - Train: python scripts/train_rl.py --config configs/base_config.yaml")
        print("  - Evaluate: python scripts/evaluate.py --model <model_path>")
        print("  - Inference: python scripts/inference.py --model <model_path> --prompt <prompt>")
        print("="*50)
    else:
        print("="*50)
        print("Please install missing dependencies first:")
        print("  pip install -r requirements.txt")
        print("="*50)


if __name__ == "__main__":
    main()
