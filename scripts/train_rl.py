"""
Main RL training script for SLM reasoning with multi-GPU support.
"""

import os
import sys
import yaml
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import PPOTrainer, PPOConfig, CurriculumScheduler
from src.rewards import CombinedReward, KLRegularization
from src.utils import ReasoningDataset, format_prompt


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_model(config: dict, accelerator: Accelerator):
    """Setup model and tokenizer with distributed support."""
    print(f"Loading model: {config['model']['name']}")
    
    # Get HuggingFace token from config or environment
    hf_token = config.get('huggingface', {}).get('token') or os.environ.get('HF_TOKEN')
    if hf_token:
        print("Using HuggingFace token for authenticated downloads")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code'],
        token=hf_token if hf_token else None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype based on mixed precision setting
    dtype = torch.bfloat16 if config['hardware']['mixed_precision'] == 'bf16' else torch.float16
    if config['hardware']['mixed_precision'] == 'no':
        dtype = torch.float32
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code'],
        torch_dtype=dtype,
        device_map=None,  # Let Accelerate handle device placement
        token=hf_token if hf_token else None,
        low_cpu_mem_usage=True,
        use_cache=False,  # Disable KV cache to save memory
        attn_implementation="eager"  # Use eager attention to avoid flash-attn issues
    )
    
    # Enable gradient checkpointing for memory efficiency
    if config['training'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Load reference model (frozen copy) - only on main process to save memory
    # Also offload to CPU if configured
    ref_model = None
    if accelerator.is_main_process:
        print("Loading reference model...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            trust_remote_code=config['model']['trust_remote_code'],
            torch_dtype=dtype,
            token=hf_token if hf_token else None,
            low_cpu_mem_usage=True,
            device_map="cpu",  # Keep on CPU to save GPU memory
            attn_implementation="eager"  # Use eager attention
        )
        ref_model.eval()
        # Don't prepare with accelerator - keep on CPU
        print("Reference model loaded on CPU")
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    
    return model, ref_model, tokenizer


def setup_training_components(config: dict):
    """Setup reward function, KL regularization, and curriculum."""
    # Reward function
    reward_fn = CombinedReward(
        outcome_weight=config['rl']['rewards']['outcome_weight'],
        process_weight=config['rl']['rewards']['process_weight']
    )
    
    # KL regularization
    kl_reg = KLRegularization(
        initial_kl_coef=config['rl']['kl_penalty']['initial_kl_coef'],
        target_kl=config['rl']['kl_penalty']['target_kl']
    )
    
    # Curriculum scheduler
    curriculum = None
    if config['rl']['curriculum']['enabled']:
        curriculum = CurriculumScheduler(
            num_stages=config['rl']['curriculum']['stages'],
            warmup_steps=config['rl']['curriculum']['warmup_steps'],
            start_ratio=config['rl']['curriculum']['start_ratio']
        )
    
    return reward_fn, kl_reg, curriculum


def load_datasets(config: dict, accelerator: Accelerator):
    """Load training datasets with distributed sampling."""
    datasets = []
    
    for dataset_name in config['data']['train_datasets']:
        if accelerator.is_main_process:
            print(f"Loading dataset: {dataset_name}")
        
        dataset = ReasoningDataset(
            dataset_name=dataset_name,
            split="train",
            max_samples=config['data']['max_train_samples'] // len(config['data']['train_datasets']),
            include_difficulty=config['rl']['curriculum']['enabled']
        )
        
        datasets.append(dataset)
    
    # Combine datasets
    combined_data = []
    for dataset in datasets:
        combined_data.extend([dataset[i] for i in range(len(dataset))])
    
    # Wrap in a simple dataset class for DataLoader
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = SimpleDataset(combined_data)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=config.get('seed', 42)
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=config['hardware'].get('num_cpus', 4) // accelerator.num_processes,
        pin_memory=True
    )
    
    # Prepare dataloader with accelerator
    dataloader = accelerator.prepare(dataloader)
    
    return dataloader, sampler


def train(config: dict):
    """Main training loop with multi-GPU support."""
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config['hardware']['mixed_precision'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        log_with="wandb" if config['logging']['use_wandb'] else None,
        project_dir=config['logging']['log_dir']
    )
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Setup logging only on main process
    if accelerator.is_main_process:
        if config['logging']['use_wandb']:
            accelerator.init_trackers(
                project_name=config['logging']['project_name'],
                config=config
            )
        print(f"Training with {accelerator.num_processes} GPUs")
        print(f"Mixed precision: {config['hardware']['mixed_precision']}")
        print(f"Batch size per GPU: {config['training']['batch_size']}")
        print(f"Effective batch size: {config['training']['batch_size'] * accelerator.num_processes * config['training']['gradient_accumulation_steps']}")
    
    # Setup model
    model, ref_model, tokenizer = setup_model(config, accelerator)
    
    # Setup training components
    reward_fn, kl_reg, curriculum = setup_training_components(config)
    
    # Setup PPO trainer with accelerator
    ppo_config = PPOConfig(
        learning_rate=config['training']['learning_rate'],
        clip_range=config['rl']['ppo']['clip_range'],
        value_loss_coef=config['rl']['ppo']['value_loss_coef'],
        entropy_coef=config['rl']['ppo']['entropy_coef'],
        gamma=config['rl']['ppo']['gamma'],
        gae_lambda=config['rl']['ppo']['gae_lambda'],
        num_minibatches=config['rl']['ppo']['num_minibatches'],
        ppo_epochs=config['rl']['ppo']['ppo_epochs'],
        batch_size=config['training']['batch_size']
    )
    
    trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        kl_reg=kl_reg,
        config=ppo_config,
        curriculum=curriculum,
        accelerator=accelerator
    )
    
    # Load datasets with distributed sampling
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
        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            # Format prompts and answers
            prompts = [format_prompt(item, item['dataset']) for item in batch]
            answers = [item['answer'] for item in batch]
            datasets = [item['dataset'] for item in batch]
            
            # Perform PPO step
            with accelerator.accumulate(trainer.model):
                metrics = trainer.ppo_step(prompts, answers, datasets[0])
            
            # Gather metrics from all processes
            metrics = accelerator.gather_object(metrics)
            if accelerator.is_main_process:
                # Average metrics across processes
                avg_metrics = {}
                for key in metrics[0].keys():
                    if isinstance(metrics[0][key], (int, float)):
                        avg_metrics[key] = sum(m[key] for m in metrics) / len(metrics)
                    else:
                        avg_metrics[key] = metrics[0][key]
                
                # Logging
                if global_step % config['training']['logging_steps'] == 0:
                    print(f"Step {global_step}: {avg_metrics}")
                    if config['logging']['use_wandb']:
                        accelerator.log(avg_metrics, step=global_step)
                
                # Save checkpoint
                if global_step % config['training']['save_steps'] == 0 and global_step > 0:
                    checkpoint_path = os.path.join(
                        config['output']['checkpoint_dir'],
                        f"checkpoint-{global_step}"
                    )
                    trainer.save_checkpoint(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
            
            global_step += 1
    
    # Save final model
    if accelerator.is_main_process:
        final_path = os.path.join(config['output']['output_dir'], "final_model")
        trainer.save_checkpoint(final_path)
        print(f"Saved final model to {final_path}")
    
    # Cleanup
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Train SLM with RL")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
