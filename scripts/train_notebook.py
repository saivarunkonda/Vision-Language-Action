"""
Complete self-contained training script for Kaggle notebooks.
Copy this entire file into a Kaggle notebook cell.
"""

# ============================================================================
# ===== EDIT THESE PARAMETERS ================================================
# ============================================================================

# Model settings
MODEL_NAME = "Qwen/Qwen2.5-7B"
HF_TOKEN = "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"

# GPU settings
USE_MULTI_GPU = False  # Set to True for 2 GPUs
NUM_GPUS = 2
BATCH_SIZE = 4  # Per GPU batch size
GRADIENT_ACCUMULATION = 4

# Memory settings
GRADIENT_CHECKPOINTING = True
MIXED_PRECISION = "bf16"

# Training settings
NUM_EPOCHS = 3
LEARNING_RATE = 1.5e-5
MAX_TRAIN_SAMPLES = 10000
SAVE_STEPS = 500
LOGGING_STEPS = 50

# Output settings
OUTPUT_DIR = "/kaggle/working/models/checkpoints"

# ============================================================================
# ===== TRAINING CODE (DO NOT EDIT) ==========================================
# ============================================================================

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import numpy as np

# Set path
sys.path.append('/kaggle/working/vision')

# Simple dataset wrapper
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def format_prompt(item, dataset_name):
    """Format a prompt for the model."""
    if dataset_name == "gsm8k":
        return f"Solve this math problem step by step: {item['question']}\n\nAnswer:"
    elif dataset_name == "mmlu":
        return f"Answer this multiple choice question: {item['question']}\nOptions: {item['choices']}\n\nAnswer:"
    else:
        return f"Answer this question: {item['question']}\n\nAnswer:"


def setup_model(config, accelerator):
    """Setup model and tokenizer."""
    print(f"Loading model: {config['model']['name']}")
    
    dtype = torch.bfloat16 if config['hardware']['mixed_precision'] == 'bf16' else torch.float16
    if config['hardware']['mixed_precision'] == 'no':
        dtype = torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code'],
        token=config.get('huggingface', {}).get('token')
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code'],
        torch_dtype=dtype,
        device_map=None,
        token=config.get('huggingface', {}).get('token'),
        low_cpu_mem_usage=True,
        use_cache=False,
        attn_implementation="eager"
    )
    
    if config['training'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model = accelerator.prepare(model)
    
    return model, None, tokenizer


def load_datasets(config, accelerator):
    """Load training datasets."""
    print("Loading datasets...")
    
    # For now, create dummy data (replace with real dataset loading)
    dummy_data = []
    for i in range(config['data']['max_train_samples']):
        dummy_data.append({
            'question': f"What is {i} + {i}?",
            'answer': str(i * 2),
            'dataset': 'gsm8k'
        })
    
    dataset = SimpleDataset(dummy_data)
    
    if accelerator.num_processes > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True
        )
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=False
    )
    
    dataloader = accelerator.prepare(dataloader)
    
    return dataloader, sampler


def train(config):
    """Main training loop."""
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config['hardware']['mixed_precision'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
    )
    
    set_seed(config.get('seed', 42))
    
    if accelerator.is_main_process:
        print(f"Training with {accelerator.num_processes} GPUs")
        print(f"Batch size per GPU: {config['training']['batch_size']}")
        eff_batch = config['training']['batch_size'] * accelerator.num_processes * config['training']['gradient_accumulation_steps']
        print(f"Effective batch size: {eff_batch}")
    
    # Setup model
    model, _, tokenizer = setup_model(config, accelerator)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    optimizer = accelerator.prepare_optimizer(optimizer)
    
    # Load datasets
    dataloader, sampler = load_datasets(config, accelerator)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    global_step = 0
    
    if accelerator.is_main_process:
        print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
            # Handle batch format
            if isinstance(batch, list):
                prompts = [format_prompt(item, item['dataset']) for item in batch]
            else:
                # Batch is a dict with tensors
                prompts = ["Sample prompt"] * len(batch[list(batch.keys())[0]])
            
            # Simple training step (replace with actual PPO logic)
            with accelerator.accumulate(model):
                # Placeholder for actual training
                pass
            
            global_step += 1
            
            if accelerator.is_main_process and global_step % config['training']['logging_steps'] == 0:
                print(f"Step {global_step}")
    
    if accelerator.is_main_process:
        print("Training complete!")
    
    accelerator.wait_for_everyone()


# ============================================================================
# ===== CREATE CONFIG AND RUN ================================================
# ============================================================================

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
    'data': {
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

print("=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Multi-GPU: {USE_MULTI_GPU}")
print(f"Number of GPUs: {NUM_GPUS if USE_MULTI_GPU else 1}")
print(f"Batch size per GPU: {BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"Mixed precision: {MIXED_PRECISION}")
print("=" * 80)

# Run training
train(config)
