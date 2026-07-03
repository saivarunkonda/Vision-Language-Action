"""
Helper script to configure accelerate for multi-GPU training.
"""

import subprocess
import sys


def configure_accelerate():
    """Run accelerate configuration wizard."""
    print("Starting Accelerate configuration...")
    print("This will guide you through setting up multi-GPU training.")
    print()
    
    # Run accelerate config command
    subprocess.run(["accelerate", "config"], check=True)
    
    print("\nConfiguration saved!")
    print("You can now run multi-GPU training with:")
    print("  python scripts/train_multigpu.py --config configs/base_config.yaml")


if __name__ == "__main__":
    configure_accelerate()
