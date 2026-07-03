"""
PPO trainer for SLM RL training with multi-GPU support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass

from ..rewards import CombinedReward, KLRegularization
from .curriculum import CurriculumScheduler, DifficultyEstimator

if TYPE_CHECKING:
    from accelerate import Accelerator


@dataclass
class PPOConfig:
    """PPO configuration."""
    learning_rate: float = 1.0e-5
    clip_range: float = 0.2
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    ppo_epochs: int = 4
    max_grad_norm: float = 1.0
    batch_size: int = 16


class PPOTrainer:
    """PPO trainer for language models with multi-GPU support."""
    
    def __init__(self,
                 model: nn.Module,
                 ref_model: nn.Module,
                 tokenizer,
                 reward_fn: CombinedReward,
                 kl_reg: KLRegularization,
                 config: PPOConfig,
                 curriculum: Optional[CurriculumScheduler] = None,
                 accelerator: Optional['Accelerator'] = None):
        """
        Initialize PPO trainer.
        
        Args:
            model: Policy model to train
            ref_model: Reference model for KL penalty
            tokenizer: Tokenizer
            reward_fn: Reward function
            kl_reg: KL regularization
            config: PPO configuration
            curriculum: Optional curriculum scheduler
            accelerator: Optional Accelerate for multi-GPU training
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.kl_reg = kl_reg
        self.config = config
        self.curriculum = curriculum
        self.accelerator = accelerator
        
        self.device = next(model.parameters()).device
        if self.ref_model is not None:
            self.ref_model.eval()
        
        # Value head for critic (maps hidden state to scalar value)
        hidden_size = getattr(self.model.config, 'hidden_size', None) or getattr(self.model.config, 'n_embd', None)
        if hidden_size is None:
            raise RuntimeError('Unable to infer model hidden size for value head')
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)
        
        # Optimizer
        # Optimize model + value head parameters
        params = list(model.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate
        )
        
        # Prepare optimizer with accelerator if provided
        if self.accelerator is not None:
            self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        
        # Training state
        self.global_step = 0
    
    def generate_responses(self,
                          prompts: List[str],
                          max_length: int = 512,
                          temperature: float = 1.0,
                          top_p: float = 0.9) -> Tuple[List[str], torch.Tensor]:
        """
        Generate responses from policy model.
        
        Args:
            prompts: List of prompt strings
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Tuple of (responses, log_probs)
        """
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Get generated tokens
            generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
            
            # Calculate log probabilities
            log_probs = self._compute_log_probs(
                inputs.input_ids,
                generated_ids
            )

            # Estimate values for generated responses (critic)
            values = self._get_value_estimates(inputs.input_ids, generated_ids)
            
            # Decode responses
            responses = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        
        return responses, log_probs, values
    
    def _compute_log_probs(self,
                          input_ids: torch.Tensor,
                          generated_ids: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for generated tokens."""
        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        
        with torch.no_grad():
            outputs = self.model(full_ids, labels=full_ids)
            logits = outputs.logits
            
            # Shift logits for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_ids[..., 1:].contiguous()
            
            # Compute log probabilities
            log_probs = nn.functional.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs for actual tokens
            gathered_log_probs = log_probs.gather(
                -1,
                shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask input tokens, keep only generated
            mask = torch.zeros_like(shift_labels, dtype=torch.bool)
            mask[:, input_ids.shape[1]-1:] = True
            
            generated_log_probs = gathered_log_probs * mask
            return generated_log_probs.sum(dim=-1)
    
    def compute_advantages(self,
                          rewards: torch.Tensor,
                          values: torch.Tensor,
                          dones: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            rewards: Reward tensor
            values: Value function estimates
            dones: Done flags
            
        Returns:
            Advantage tensor
        """
        # Simplified advantage computation for single-step episodes / per-sample
        # advantages = reward - value (bootstrap-to-zero at episode end)
        values = values.view(-1)
        advantages = rewards - values
        return advantages
    
    def ppo_step(self,
                 prompts: List[str],
                 ground_truths: List[str],
                 dataset: str = 'default') -> Dict[str, float]:
        """
        Perform one PPO update step.
        
        Args:
            prompts: List of prompts
            ground_truths: List of ground truth answers
            dataset: Dataset name
            
        Returns:
            Dictionary of metrics
        """
        # Generate responses
        responses, log_probs, values = self.generate_responses(prompts)
        
        # Compute rewards
        reward_outputs = self.reward_fn.compute_batch(
            responses,
            ground_truths,
            dataset
        )
        rewards = torch.tensor(
            [r.total_reward for r in reward_outputs],
            device=self.device,
            dtype=torch.float32
        )
        
        # Get reference log probs for KL penalty
        ref_log_probs = self._get_ref_log_probs(prompts, responses)
        
        # Compute KL penalty
        kl_penalty = self.kl_reg.compute_penalty(log_probs, ref_log_probs)
        
        # Adjust rewards with KL penalty
        adjusted_rewards = rewards - kl_penalty
        
        # Compute advantages using critic values
        dones = torch.zeros_like(rewards)
        # values from generate_responses are estimates from the critic
        advantages = self.compute_advantages(adjusted_rewards, values, dones)
        # normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        metrics = self._ppo_update(
            prompts,
            responses,
            log_probs,
            ref_log_probs,
            advantages,
            rewards
        )
        
        # Update KL coefficient
        mean_kl = kl_penalty.mean().item()
        self.kl_reg.update_coef(mean_kl, adaptive=True)
        
        # Update curriculum
        if self.curriculum:
            self.curriculum.step()
        
        self.global_step += 1
        
        metrics.update({
            'mean_reward': rewards.mean().item(),
            'mean_kl': mean_kl,
            'kl_coef': self.kl_reg.current_kl_coef,
            'curriculum_stage': self.curriculum.get_current_stage().stage if self.curriculum else 0
        })
        
        return metrics
    
    def _get_ref_log_probs(self,
                          prompts: List[str],
                          responses: List[str]) -> torch.Tensor:
        """Get log probabilities from reference model."""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        
        inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(self.device)
        
        prompt_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.ref_model(**inputs)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()
            
            log_probs = nn.functional.log_softmax(shift_logits, dim=-1)
            gathered_log_probs = log_probs.gather(
                -1,
                shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask prompt tokens
            mask = torch.zeros_like(shift_labels, dtype=torch.bool)
            mask[:, prompt_inputs.input_ids.shape[1]-1:] = True
            
            generated_log_probs = gathered_log_probs * mask
            return generated_log_probs.sum(dim=-1)

    def _get_value_estimates(self,
                             input_ids: torch.Tensor,
                             generated_ids: torch.Tensor) -> torch.Tensor:
        """Estimate scalar values for each prompt+response pair using value_head."""
        full_ids = torch.cat([input_ids, generated_ids], dim=1)

        # Run model to get hidden states
        with torch.no_grad():
            outputs = self.model(full_ids, output_hidden_states=True)
            # Last hidden state
            hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
            # Use last token hidden state as summary
            last_hidden = hidden_states[:, -1, :]
            values = self.value_head(last_hidden).squeeze(-1)

        return values.to(self.device)
    
    def _ppo_update(self,
                    prompts: List[str],
                    responses: List[str],
                    old_log_probs: torch.Tensor,
                    ref_log_probs: torch.Tensor,
                    advantages: torch.Tensor,
                    rewards: torch.Tensor) -> Dict[str, float]:
        """Perform PPO update."""
        metrics = {}
        
        for _ in range(self.config.ppo_epochs):
            # Get new log probabilities
            new_log_probs = self._compute_log_probs_batch(prompts, responses)
            # Current value estimates
            new_values = self._get_value_estimates(
                self.tokenizer(prompts, padding=True, return_tensors="pt").input_ids.to(self.device),
                self.tokenizer(responses, padding=True, return_tensors="pt").input_ids.to(self.device)
            )
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            entropy = self._compute_entropy(prompts, responses)
            entropy_loss = -self.config.entropy_coef * entropy.mean()
            
            # Value loss (MSE between returns and value estimates)
            returns = rewards.detach()
            value_loss = F.mse_loss(new_values, returns)
            total_value_loss = self.config.value_loss_coef * value_loss

            # Total loss
            loss = policy_loss + entropy_loss + total_value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            
            metrics['policy_loss'] = policy_loss.item()
            metrics['entropy'] = entropy.mean().item()
        
        return metrics
    
    def _compute_log_probs_batch(self,
                                 prompts: List[str],
                                 responses: List[str]) -> torch.Tensor:
        """Compute log probabilities for a batch."""
        # Simplified version
        return self._compute_log_probs(
            self.tokenizer(prompts, padding=True, return_tensors="pt").input_ids.to(self.device),
            self.tokenizer(responses, padding=True, return_tensors="pt").input_ids.to(self.device)
        )
    
    def _compute_entropy(self,
                        prompts: List[str],
                        responses: List[str]) -> torch.Tensor:
        """Compute entropy of policy distribution."""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1)
        
        return entropy
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint with multi-GPU support."""
        # Unwrap model if using accelerator
        model_to_save = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        
        # Save only on main process
        if self.accelerator is None or self.accelerator.is_main_process:
            model_to_save.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            # Save optimizer state
            if self.accelerator is not None:
                self.accelerator.save_state(path)
            else:
                torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.optimizer.load_state_dict(torch.load(f"{path}/optimizer.pt"))
