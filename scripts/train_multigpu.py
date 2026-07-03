"""
Multi-GPU training launch script using Accelerate.
"""

import subprocess
import sys
import argparse


def launch_multigpu(config_path: str, num_gpus: int = None, resume: str = None, accelerate_config: str = None):
    """
    Launch training on multiple GPUs using accelerate.
    
    Args:
        config_path: Path to config file
        num_gpus: Number of GPUs to use (default: all available)
        resume: Path to checkpoint to resume from
        accelerate_config: Path to accelerate config file
    """
    # Build accelerate launch command
    cmd = [
        "accelerate", "launch"
    ]
    
    # Use custom accelerate config if provided
    if accelerate_config:
        cmd.extend(["--config_file", accelerate_config])
    else:
        cmd.extend(["--config_file", "configs/accelerate_config.yaml"])
    
    if num_gpus:
        cmd.extend(["--num_processes", str(num_gpus)])
    
    cmd.extend([
        "scripts/train_rl.py",
        "--config", config_path
    ])
    
    if resume:
        cmd.extend(["--resume", resume])
    
    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch multi-GPU training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--accelerate_config", type=str, default=None, help="Path to accelerate config file")
    
    args = parser.parse_args()
    
    launch_multigpu(args.config, args.num_gpus, args.resume, args.accelerate_config)
