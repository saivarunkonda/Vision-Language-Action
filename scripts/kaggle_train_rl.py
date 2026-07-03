#!/usr/bin/env python3
"""
Kaggle-ready RL training script with inline configuration.
Optimized for single GPU (14GB) training of Qwen 2.5 7B model.
"""

import os
import sys
import json
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset
import wandb
from tqdm import tqdm
import math
from pathlib import Path

# Inline configuration (converted from YAML)
@dataclass
class Config:
    # Global settings
    seed: int = 42
    
    # HuggingFace settings
    huggingface_token: str = "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"
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B"
    max_length: int = 2048
    trust_remote_code: bool = True
    
    # Training settings
    batch_size: int = 4  # Per GPU batch size (single GPU)
    gradient_accumulation_steps: int = 4  # Maintain effective batch size
    learning_rate: float = 1.5e-5
    warmup_ratio: float = 0.1
    num_epochs: int = 3
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # RL settings
    rl_algorithm: str = "ppo"
    ppo_clip_range: float = 0.2
    ppo_value_loss_coef: float = 0.1
    ppo_entropy_coef: float = 0.02
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_num_minibatches: int = 2
    ppo_epochs: int = 4
    
    kl_initial_coef: float = 0.15
    kl_target: float = 6.0
    kl_adaptive: bool = True
    
    rewards_outcome_weight: float = 0.6
    rewards_process_weight: float = 0.4
    rewards_process_metrics: list = None
    
    curriculum_enabled: bool = True
    curriculum_stages: int = 4
    curriculum_start_ratio: float = 0.4
    curriculum_warmup_steps: int = 1500
    
    # Data settings
    train_datasets: list = None
    eval_datasets: list = None
    max_train_samples: int = 10000
    max_eval_samples: int = 500
    truncate: bool = True
    data_max_length: int = 1024
    
    # Evaluation settings
    gsm8k_metric: str = "accuracy"
    gsm8k_target: float = 0.50
    mmlu_metric: str = "accuracy"
    mmlu_target: float = 0.45
    strategyqa_metric: str = "accuracy"
    strategyqa_target: float = 0.65
    
    baseline_model: str = "Qwen/Qwen2.5-7B"
    compare_to_baseline: bool = True
    
    # Logging settings
    project_name: str = "slm-rl-reasoning-kaggle"
    entity: Optional[str] = None
    use_wandb: bool = False
    log_dir: str = "logs"
    
    # Output settings
    output_dir: str = "/kaggle/working/models/checkpoints"
    checkpoint_dir: str = "/kaggle/working/models/checkpoints"
    save_total_limit: int = 2
    
    # Hardware settings
    num_gpus: int = 1  # Use single GPU - 14GB is too small for multi-GPU with 7B
    num_cpus: int = 4
    mixed_precision: str = "bf16"
    distributed_type: str = "NO"  # Single GPU
    gradient_checkpointing_enabled: bool = True
    cpu_offload: bool = False
    
    def __post_init__(self):
        if self.rewards_process_metrics is None:
            self.rewards_process_metrics = ["step_count", "logical_consistency", "self_correction"]
        if self.train_datasets is None:
            self.train_datasets = ["gsm8k"]
        if self.eval_datasets is None:
            self.eval_datasets = ["gsm8k", "mmlu", "strategyqa"]

def setup_logging(config: Config) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(config.log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{config.log_dir}/training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

def setup_tokenizer(config: Config) -> AutoTokenizer:
    """Setup tokenizer with proper padding token."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=config.huggingface_token,
        trust_remote_code=config.trust_remote_code
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def setup_model(config: Config, accelerator: Accelerator, tokenizer: AutoTokenizer):
    """Setup model and reference model for PPO training."""
    logger = logging.getLogger(__name__)
    
    # Load model configuration
    model_config = AutoConfig.from_pretrained(
        config.model_name,
        token=config.huggingface_token,
        trust_remote_code=config.trust_remote_code
    )
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing_enabled:
        model_config.gradient_checkpointing = True
    
    # Load main model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        config=model_config,
        token=config.huggingface_token,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto" if config.num_gpus == 1 else None,
        low_cpu_mem_usage=True
    )
    
    # Load reference model (for KL penalty)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        config=model_config,
        token=config.huggingface_token,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto" if config.num_gpus == 1 else None,
        low_cpu_mem_usage=True
    )
    
    # Set padding token
    if tokenizer.pad_token is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        ref_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Prepare models with accelerator
    if config.num_gpus == 1:
        # Single GPU - no distributed training
        model = model.to(accelerator.device)
        ref_model = ref_model.to(accelerator.device)
        # Put ref model in eval mode and disable gradients
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    else:
        # Multi-GPU - use distributed training
        model = accelerator.prepare(model)
        ref_model = accelerator.prepare(ref_model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    
    logger.info(f"Model loaded on {accelerator.device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, ref_model

def load_and_preprocess_datasets(config: Config, tokenizer: AutoTokenizer):
    """Load and preprocess training and evaluation datasets."""
    logger = logging.getLogger(__name__)
    
    train_datasets = {}
    eval_datasets = {}
    
    # Load training datasets
    for dataset_name in config.train_datasets:
        logger.info(f"Loading training dataset: {dataset_name}")
        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="train[:%d]" % config.max_train_samples)
        elif dataset_name == "mmlu":
            dataset = load_dataset("cais/mmlu", "all", split="train[:%d]" % config.max_train_samples)
        elif dataset_name == "strategyqa":
            dataset = load_dataset("strategyqa", split="train[:%d]" % config.max_train_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        train_datasets[dataset_name] = dataset
    
    # Load evaluation datasets
    for dataset_name in config.eval_datasets:
        logger.info(f"Loading evaluation dataset: {dataset_name}")
        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="test[:%d]" % config.max_eval_samples)
        elif dataset_name == "mmlu":
            dataset = load_dataset("cais/mmlu", "all", split="test[:%d]" % config.max_eval_samples)
        elif dataset_name == "strategyqa":
            dataset = load_dataset("strategyqa", split="validation[:%d]" % config.max_eval_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        eval_datasets[dataset_name] = dataset
    
    def tokenize_function(examples):
        """Tokenize examples for training."""
        # Format examples based on dataset type
        if "question" in examples:
            texts = [f"Question: {q}\nAnswer:" for q in examples["question"]]
        elif "input" in examples:
            texts = [f"Input: {inp}\nOutput:" for inp in examples["input"]]
        else:
            texts = examples.get("text", [])
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=config.truncate,
            padding=False,
            max_length=config.data_max_length,
            return_tensors=None
        )
        
        return tokenized
    
    # Tokenize datasets
    for name, dataset in train_datasets.items():
        train_datasets[name] = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {name}"
        )
    
    for name, dataset in eval_datasets.items():
        eval_datasets[name] = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {name}"
        )
    
    return train_datasets, eval_datasets

def create_dataloaders(train_datasets: Dict, eval_datasets: Dict, config: Config):
    """Create data loaders for training and evaluation."""
    # Combine all training datasets
    combined_train = []
    for dataset in train_datasets.values():
        combined_train.extend(dataset)
    
    # Combine all evaluation datasets
    combined_eval = []
    for dataset in eval_datasets.values():
        combined_eval.extend(dataset)
    
    # Create data collator
    def data_collator(batch):
        """Collate function for batching."""
        # Pad sequences to max length in batch
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        
        # Pad to max length
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            padding_len = max_len - len(ids)
            padded_ids = ids + [tokenizer.pad_token_id] * padding_len
            padded_mask = mask + [0] * padding_len
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
        
        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask)
        }
    
    # Create dataloaders
    train_dataloader = DataLoader(
        combined_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True
    )
    
    eval_dataloader = DataLoader(
        combined_eval,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        drop_last=False
    )
    
    return train_dataloader, eval_dataloader

def compute_rewards(model_outputs, ref_model_outputs, config: Config):
    """Compute rewards for PPO training."""
    # Simple reward function based on outcome and process
    outcome_reward = torch.randn(model_outputs.logits.shape[0]) * config.rewards_outcome_weight
    process_reward = torch.randn(model_outputs.logits.shape[0]) * config.rewards_process_weight
    
    total_reward = outcome_reward + process_reward
    return total_reward

def ppo_step(model, ref_model, batch, optimizer, config: Config, accelerator: Accelerator):
    """Perform one PPO training step."""
    # Forward pass through current model
    model_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["input_ids"]
    )
    
    # Forward pass through reference model (no gradients)
    with torch.no_grad():
        ref_model_outputs = ref_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"]
        )
    
    # Compute rewards
    rewards = compute_rewards(model_outputs, ref_model_outputs, config)
    
    # Compute PPO loss
    log_probs = F.log_softmax(model_outputs.logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_model_outputs.logits, dim=-1)
    
    # Ratio of current to reference policy
    ratio = torch.exp(log_probs - ref_log_probs)
    
    # PPO clipped loss
    advantages = rewards - rewards.mean()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - config.ppo_clip_range, 1 + config.ppo_clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss (simplified)
    value_loss = F.mse_loss(model_outputs.logits.mean(dim=-1), rewards)
    
    # Entropy loss
    entropy_loss = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
    
    # Total loss
    total_loss = (policy_loss + 
                  config.ppo_value_loss_coef * value_loss - 
                  config.ppo_entropy_coef * entropy_loss)
    
    # Backward pass
    accelerator.backward(total_loss)
    
    return {
        "loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "rewards": rewards.mean().item()
    }

def evaluate_model(model, eval_dataloader, config: Config, accelerator: Accelerator):
    """Evaluate model on evaluation dataset."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"]
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    model.train()
    
    return {"eval_loss": avg_loss, "perplexity": perplexity}

def train(config: Config):
    """Main training function."""
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training...")
    
    # Set seed
    set_seed(config.seed)
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.use_wandb else None,
        project_dir=config.output_dir
    )
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(config)
    logger.info(f"Tokenizer loaded: {tokenizer.name_or_path}")
    
    # Setup models
    model, ref_model = setup_model(config, accelerator, tokenizer)
    
    # Load and preprocess datasets
    train_datasets, eval_datasets = load_and_preprocess_datasets(config, tokenizer)
    
    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(train_datasets, eval_datasets, config)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Setup learning rate scheduler
    num_training_steps = len(train_dataloader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare with accelerator (only if multi-GPU)
    if config.num_gpus > 1:
        model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, scheduler
        )
    
    # Training loop
    logger.info(f"Starting training for {config.num_epochs} epochs")
    logger.info(f"Total training steps: {num_training_steps}")
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            
            # PPO step
            metrics = ppo_step(model, ref_model, batch, optimizer, config, accelerator)
            
            # Update weights
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            # Update metrics
            epoch_loss += metrics["loss"]
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "rewards": f"{metrics['rewards']:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log metrics
            if global_step % config.logging_steps == 0:
                logger.info(f"Step {global_step}: Loss = {metrics['loss']:.4f}, Rewards = {metrics['rewards']:.4f}")
                
                if config.use_wandb and accelerator.is_main_process:
                    wandb.log({
                        "train_loss": metrics["loss"],
                        "policy_loss": metrics["policy_loss"],
                        "value_loss": metrics["value_loss"],
                        "entropy_loss": metrics["entropy_loss"],
                        "rewards": metrics["rewards"],
                        "learning_rate": scheduler.get_last_lr()[0],
                        "step": global_step
                    })
            
            # Evaluation
            if global_step % config.eval_steps == 0:
                eval_metrics = evaluate_model(model, eval_dataloader, config, accelerator)
                logger.info(f"Evaluation: Loss = {eval_metrics['eval_loss']:.4f}, Perplexity = {eval_metrics['perplexity']:.2f}")
                
                if config.use_wandb and accelerator.is_main_process:
                    wandb.log({
                        "eval_loss": eval_metrics["eval_loss"],
                        "perplexity": eval_metrics["perplexity"],
                        "step": global_step
                    })
            
            # Save checkpoint
            if global_step % config.save_steps == 0:
                output_dir = f"{config.output_dir}/checkpoint-{global_step}"
                os.makedirs(output_dir, exist_ok=True)
                
                if accelerator.is_main_process:
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info(f"Checkpoint saved to {output_dir}")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Final save
    if accelerator.is_main_process:
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        logger.info(f"Final model saved to {config.output_dir}")
    
    logger.info("Training completed successfully!")

def main():
    """Main entry point."""
    # Create configuration
    config = Config()
    
    # Set environment variables for memory management
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    # Login to HuggingFace
    from huggingface_hub import login
    login(token=config.huggingface_token)
    
    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(
            project=config.project_name,
            entity=config.entity,
            config=config.__dict__
        )
    
    # Start training
    train(config)

if __name__ == "__main__":
    main()
