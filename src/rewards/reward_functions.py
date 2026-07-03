"""
Reward functions for RL training of SLMs.
Combines outcome-based and process-based rewards.
"""

import re
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RewardOutput:
    """Output of reward computation."""
    total_reward: float
    outcome_reward: float
    process_reward: float
    components: Dict[str, float]


class OutcomeReward:
    """Outcome-based reward: correctness of final answer."""
    
    def __init__(self):
        self.answer_patterns = {
            'gsm8k': r'(?:answer|is|equals?)\s*[:=]?\s*([+-]?\d*\.?\d+)',
            'default': r'(?:answer|result|therefore)\s*[:=]?\s*([^\n.]+)',
        }
    
    def extract_answer(self, response: str, dataset: str = 'default') -> Optional[str]:
        """Extract the final answer from model response."""
        pattern = self.answer_patterns.get(dataset, self.answer_patterns['default'])
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def compute(self, 
                response: str, 
                ground_truth: str, 
                dataset: str = 'default') -> float:
        """
        Compute outcome reward based on answer correctness.
        
        Args:
            response: Model's response
            ground_truth: Correct answer
            dataset: Dataset name for parsing
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        pred_answer = self.extract_answer(response, dataset)
        
        if pred_answer is None:
            return 0.0
        
        # Normalize answers for comparison
        pred_normalized = self._normalize_answer(pred_answer)
        gt_normalized = self._normalize_answer(ground_truth)
        
        if pred_normalized == gt_normalized:
            return 1.0
        elif self._partial_match(pred_normalized, gt_normalized):
            return 0.5
        else:
            return 0.0
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        answer = answer.lower().strip()
        # Remove common phrases
        answer = re.sub(r'^(the|a|an)\s+', '', answer)
        # Remove punctuation
        answer = re.sub(r'[^\w\s.-]', '', answer)
        return answer
    
    def _partial_match(self, pred: str, gt: str) -> bool:
        """Check for partial match (e.g., numeric values)."""
        try:
            pred_num = float(pred)
            gt_num = float(gt)
            return abs(pred_num - gt_num) < 0.01 * abs(gt_num)
        except ValueError:
            return gt in pred or pred in gt


class ProcessReward:
    """Process-based reward: quality of reasoning chain."""
    
    def __init__(self):
        self.step_indicators = [
            'first', 'second', 'third', 'then', 'next', 'finally',
            'step', 'because', 'since', 'therefore', 'thus', 'so',
            'let', 'given', 'we have', 'we know'
        ]
    
    def compute(self, response: str) -> float:
        """
        Compute process reward based on reasoning quality.
        
        Args:
            response: Model's full response including reasoning
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        step_count = self._count_reasoning_steps(response)
        consistency = self._check_logical_consistency(response)
        self_correction = self._detect_self_correction(response)
        length_penalty = self._length_penalty(response)
        
        # Weighted combination
        step_score = min(step_count / 5.0, 1.0) * 0.4
        consistency_score = consistency * 0.3
        correction_score = self_correction * 0.2
        length_score = length_penalty * 0.1
        
        return step_score + consistency_score + correction_score + length_score
    
    def _count_reasoning_steps(self, response: str) -> int:
        """Count explicit reasoning steps."""
        count = 0
        response_lower = response.lower()
        for indicator in self.step_indicators:
            count += response_lower.count(indicator)
        return count
    
    def _check_logical_consistency(self, response: str) -> float:
        """Check for logical consistency markers."""
        consistency_markers = [
            'because', 'since', 'therefore', 'thus', 'consequently',
            'as a result', 'this means', 'this implies'
        ]
        response_lower = response.lower()
        score = sum(1 for marker in consistency_markers if marker in response_lower)
        return min(score / 3.0, 1.0)
    
    def _detect_self_correction(self, response: str) -> float:
        """Detect self-correction patterns."""
        correction_patterns = [
            'wait', 'actually', 'correction', 'let me reconsider',
            'on second thought', 'i made a mistake'
        ]
        response_lower = response.lower()
        return 1.0 if any(pattern in response_lower for pattern in correction_patterns) else 0.0
    
    def _length_penalty(self, response: str) -> float:
        """Penalize too short or too long responses."""
        length = len(response.split())
        if length < 10:
            return 0.0
        elif length < 50:
            return 0.5
        elif length < 200:
            return 1.0
        else:
            return max(0.5, 1.0 - (length - 200) / 400.0)


class CombinedReward:
    """Combined outcome + process reward."""
    
    def __init__(self, 
                 outcome_weight: float = 0.7,
                 process_weight: float = 0.3):
        """
        Initialize combined reward.
        
        Args:
            outcome_weight: Weight for outcome reward
            process_weight: Weight for process reward
        """
        assert abs(outcome_weight + process_weight - 1.0) < 1e-6
        self.outcome_weight = outcome_weight
        self.process_weight = process_weight
        
        self.outcome_reward = OutcomeReward()
        self.process_reward = ProcessReward()
    
    def compute(self,
                response: str,
                ground_truth: str,
                dataset: str = 'default') -> RewardOutput:
        """
        Compute combined reward.
        
        Args:
            response: Model's response
            ground_truth: Correct answer
            dataset: Dataset name
            
        Returns:
            RewardOutput with total and component rewards
        """
        outcome = self.outcome_reward.compute(response, ground_truth, dataset)
        process = self.process_reward.compute(response)
        
        total = (self.outcome_weight * outcome + 
                 self.process_weight * process)
        
        components = {
            'outcome': outcome,
            'process': process,
            'step_count': self.process_reward._count_reasoning_steps(response),
            'logical_consistency': self.process_reward._check_logical_consistency(response),
            'self_correction': self.process_reward._detect_self_correction(response)
        }
        
        return RewardOutput(
            total_reward=total,
            outcome_reward=outcome,
            process_reward=process,
            components=components
        )
    
    def compute_batch(self,
                      responses: List[str],
                      ground_truths: List[str],
                      dataset: str = 'default') -> List[RewardOutput]:
        """Compute rewards for a batch of responses."""
        return [
            self.compute(r, gt, dataset)
            for r, gt in zip(responses, ground_truths)
        ]


class KLRegularization:
    """KL divergence regularization for stability."""
    
    def __init__(self, initial_kl_coef: float = 0.1, target_kl: float = 6.0):
        """
        Initialize KL regularization.
        
        Args:
            initial_kl_coef: Initial KL penalty coefficient
            target_kl: Target KL divergence
        """
        self.initial_kl_coef = initial_kl_coef
        self.target_kl = target_kl
        self.current_kl_coef = initial_kl_coef
    
    def compute_penalty(self, 
                        log_probs: torch.Tensor,
                        ref_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence penalty.
        
        Args:
            log_probs: Log probabilities from policy
            ref_log_probs: Log probabilities from reference model
            
        Returns:
            KL penalty tensor
        """
        kl_div = (ref_log_probs - log_probs).sum(dim=-1)
        return self.current_kl_coef * kl_div
    
    def update_coef(self, mean_kl: float, adaptive: bool = True):
        """
        Update KL coefficient based on observed KL.
        
        Args:
            mean_kl: Mean KL divergence from recent batch
            adaptive: Whether to use adaptive KL tuning
        """
        if not adaptive:
            return
        
        if mean_kl < self.target_kl / 2.0:
            # Increase penalty if KL is too low
            self.current_kl_coef *= 1.5
        elif mean_kl > self.target_kl * 1.5:
            # Decrease penalty if KL is too high
            self.current_kl_coef *= 0.5
        
        # Clamp to reasonable range
        self.current_kl_coef = max(0.01, min(self.current_kl_coef, 1.0))
    
    def reset(self):
        """Reset KL coefficient to initial value."""
        self.current_kl_coef = self.initial_kl_coef
