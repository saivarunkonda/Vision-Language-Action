# Kaggle Quick Start

## Option 1: Edit Parameters in Notebook Cell (Recommended)

Copy this entire code block into a Kaggle notebook cell, modify the parameters at the top, and run:

```python
# ============================================================================
# ===== EDIT THESE PARAMETERS ================================================
# ============================================================================

# Model settings
MODEL_NAME = "Qwen/Qwen2.5-7B"  # Options: "Qwen/Qwen2.5-7B", "microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN = "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"

# GPU settings
USE_MULTI_GPU = True  # Set to True for 2 GPUs, False for 1 GPU
NUM_GPUS = 2  # Number of GPUs if USE_MULTI_GPU = True
BATCH_SIZE = 2  # Per GPU batch size (reduce if OOM: try 1)
GRADIENT_ACCUMULATION = 8  # Increase to maintain effective batch size

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
# ===== RUN THIS TO START TRAINING ==========================================
# ============================================================================

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedSampler
from accelerate.utils import set_seed
from dotenv import load_dotenv

# Load environment
load_dotenv()
# Set working directory (for Kaggle)
sys.path.append('/kaggle/working/vision')

# Import project modules
from src.training import PPOTrainer, PPOConfig
from src.rewards import CombinedReward, KLRegularization
from src.utils import ReasoningDataset, format_prompt

# Create config
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
        'max_train_samples': MAX_TRAIN_SAMPLES,
        'preprocessing': {'truncate': True, 'max_length': 1024}
    },
    'logging': {
        'project_name': 'slm-rl-reasoning-kaggle',
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
        'gradient_checkpointing': GRADIENT_CHECKPOINTING
    }
}

# Print config
print("=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Multi-GPU: {USE_MULTI_GPU}")
print(f"Number of GPUs: {NUM_GPUS if USE_MULTI_GPU else 1}")
print(f"Batch size per GPU: {BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
eff_batch = BATCH_SIZE * (NUM_GPUS if USE_MULTI_GPU else 1) * GRADIENT_ACCUMULATION
print(f"Effective batch size: {eff_batch}")
print(f"Mixed precision: {MIXED_PRECISION}")
print("=" * 80)

# Run training
from scripts.train_rl import train
train(config)
```

## Option 2: Use the Simple Script

```python
# Edit and run:
!python scripts/train_simple.py
```

Then edit the parameters at the top of `scripts/train_simple.py` and re-run.

## Common Parameter Adjustments

### If you get CUDA Out of Memory:
1. Reduce `BATCH_SIZE` to 1 or 2
2. Increase `GRADIENT_ACCUMULATION` to maintain effective batch size
3. Set `USE_MULTI_GPU = False` to use single GPU

### For faster training:
1. Set `USE_MULTI_GPU = True` (if you have 2 GPUs)
2. Increase `BATCH_SIZE` to 4 (if memory allows)

### For more stable training:
1. Reduce `LEARNING_RATE` to 1e-5
2. Increase `GRADIENT_ACCUMULATION` to 8 or 16
