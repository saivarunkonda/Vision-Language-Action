"""
Simple training script for Kaggle - All parameters at the top for easy modification.
Edit the parameters below and run this script directly.
"""

# ============================================================================
# ===== EDIT THESE PARAMETERS IN KAGGLE NOTEBOOK ============================
# ============================================================================

# Model settings
MODEL_NAME = "Qwen/Qwen2.5-7B"  # Options: "Qwen/Qwen2.5-7B", "microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN = "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"

# GPU settings
USE_MULTI_GPU = False  # Set to True for 2 GPUs, False for 1 GPU
NUM_GPUS = 2  # Number of GPUs if USE_MULTI_GPU = True
BATCH_SIZE = 4  # Per GPU batch size (reduce if OOM: try 2 or 1)
GRADIENT_ACCUMULATION = 4  # Increase to maintain effective batch size

# Memory settings
GRADIENT_CHECKPOINTING = True
MIXED_PRECISION = "bf16"  # Options: "bf16", "fp16", "no"

# Training settings
NUM_EPOCHS = 3
LEARNING_RATE = 1.5e-5
MAX_TRAIN_SAMPLES = 10000
SAVE_STEPS = 500
LOGGING_STEPS = 50

# Output settings
OUTPUT_DIR = "/kaggle/working/models/checkpoints"

# ============================================================================
# ===== DO NOT EDIT BELOW (unless you know what you're doing) ===============
# ============================================================================

import os
import sys
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path (works for both script and notebook)
try:
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Running in notebook/jupyter
    script_dir = os.getcwd()
sys.path.append(script_dir)

from src.training import PPOTrainer, PPOConfig, CurriculumScheduler
from src.rewards import CombinedReward, KLRegularization
from src.utils import ReasoningDataset, format_prompt


def create_config():
    """Create config from the parameters above."""
    config = {
        'seed': 42,
        'huggingface': {'token': HF_TOKEN},
        'model': {
            'name': MODEL_NAME,
            'max_length': 2048,
            'trust_remote_code': True
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION,
            'learning_rate': LEARNING_RATE,
            'warmup_ratio': 0.1,
            'num_epochs': NUM_EPOCHS,
            'save_steps': SAVE_STEPS,
            'eval_steps': SAVE_STEPS,
            'logging_steps': LOGGING_STEPS,
            'fp16': False,
            'bf16': MIXED_PRECISION == "bf16",
            'gradient_checkpointing': GRADIENT_CHECKPOINTING
        },
        'rl': {
            'algorithm': 'ppo',
            'ppo': {
                'clip_range': 0.2,
                'value_loss_coef': 0.1,
                'entropy_coef': 0.02,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'num_minibatches': 2,
                'ppo_epochs': 4
            },
            'kl_penalty': {
                'initial_kl_coef': 0.15,
                'target_kl': 6.0,
                'adaptive_kl': True
            },
            'rewards': {
                'outcome_weight': 0.6,
                'process_weight': 0.4,
                'process_metrics': ['step_count', 'logical_consistency', 'self_correction']
            },
            'curriculum': {
                'enabled': True,
                'stages': 4,
                'start_ratio': 0.4,
                'warmup_steps': 1500
            }
        },
        'data': {
            'train_datasets': ['gsm8k'],
            'eval_datasets': ['gsm8k', 'mmlu', 'strategyqa'],
            'max_train_samples': MAX_TRAIN_SAMPLES,
            'max_eval_samples': 500,
            'preprocessing': {
                'truncate': True,
                'max_length': 1024
            }
        },
        'evaluation': {
            'benchmarks': {
                'gsm8k': {'metric': 'accuracy', 'target': 0.50},
                'mmlu': {'metric': 'accuracy', 'target': 0.45},
                'strategyqa': {'metric': 'accuracy', 'target': 0.65}
            },
            'baseline_model': MODEL_NAME,
            'compare_to_baseline': True
        },
        'logging': {
            'project_name': 'slm-rl-reasoning-kaggle',
            'entity': None,
            'use_wandb': False,
            'log_dir': 'logs'
        },
        'output': {
            'output_dir': OUTPUT_DIR,
            'checkpoint_dir': OUTPUT_DIR,
            'save_total_limit': 2
        },
        'hardware': {
            'num_gpus': NUM_GPUS if USE_MULTI_GPU else 1,
            'num_cpus': 4,
            'mixed_precision': MIXED_PRECISION,
            'distributed_type': 'MULTI_GPU' if USE_MULTI_GPU else 'NO',
            'gradient_checkpointing': GRADIENT_CHECKPOINTING,
            'cpu_offload': False
        }
    }
    return config


if __name__ == "__main__":
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Multi-GPU: {USE_MULTI_GPU}")
    print(f"Number of GPUs: {NUM_GPUS if USE_MULTI_GPU else 1}")
    print(f"Batch size per GPU: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"Effective batch size: {BATCH_SIZE * (NUM_GPUS if USE_MULTI_GPU else 1) * GRADIENT_ACCUMULATION}")
    print(f"Mixed precision: {MIXED_PRECISION}")
    print(f"Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
    print("=" * 80)
    
    config = create_config()
    
    # Import and run the main training function
    from scripts.train_rl import train
    train(config)
