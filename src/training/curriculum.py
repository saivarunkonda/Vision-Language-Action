"""
Curriculum learning scheduler for progressive difficulty.
"""

import torch
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    """A stage in curriculum learning."""
    stage: int
    easy_ratio: float
    medium_ratio: float
    hard_ratio: float
    threshold_step: int


class CurriculumScheduler:
    """Curriculum learning scheduler for progressive difficulty."""
    
    def __init__(self,
                 num_stages: int = 3,
                 warmup_steps: int = 1000,
                 start_ratio: float = 0.3):
        """
        Initialize curriculum scheduler.
        
        Args:
            num_stages: Number of curriculum stages
            warmup_steps: Steps before curriculum starts
            start_ratio: Initial ratio of easy examples
        """
        self.num_stages = num_stages
        self.warmup_steps = warmup_steps
        self.start_ratio = start_ratio
        self.current_step = 0
        
        # Generate curriculum stages
        self.stages = self._generate_stages()
    
    def _generate_stages(self) -> List[CurriculumStage]:
        """Generate curriculum stages."""
        stages = []
        total_steps_per_stage = 1000  # Configurable
        
        for i in range(self.num_stages):
            # Progressively decrease easy examples, increase hard ones
            easy_ratio = max(0.1, self.start_ratio - i * 0.1)
            hard_ratio = min(0.5, 0.1 + i * 0.15)
            medium_ratio = 1.0 - easy_ratio - hard_ratio
            
            stage = CurriculumStage(
                stage=i + 1,
                easy_ratio=easy_ratio,
                medium_ratio=medium_ratio,
                hard_ratio=hard_ratio,
                threshold_step=self.warmup_steps + i * total_steps_per_stage
            )
            stages.append(stage)
        
        return stages
    
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage based on step."""
        if self.current_step < self.warmup_steps:
            # Warmup: use starting ratio
            return CurriculumStage(
                stage=0,
                easy_ratio=self.start_ratio,
                medium_ratio=1.0 - self.start_ratio,
                hard_ratio=0.0,
                threshold_step=0
            )
        
        for stage in reversed(self.stages):
            if self.current_step >= stage.threshold_step:
                return stage
        
        return self.stages[-1]
    
    def step(self):
        """Advance step counter."""
        self.current_step += 1
    
    def get_sampling_weights(self, 
                             difficulties: List[str]) -> List[float]:
        """
        Get sampling weights for batch based on current stage.
        
        Args:
            difficulties: List of difficulty labels ('easy', 'medium', 'hard')
            
        Returns:
            List of sampling weights
        """
        stage = self.get_current_stage()
        
        weights = []
        for diff in difficulties:
            if diff == 'easy':
                weights.append(stage.easy_ratio)
            elif diff == 'medium':
                weights.append(stage.medium_ratio)
            elif diff == 'hard':
                weights.append(stage.hard_ratio)
            else:
                weights.append(stage.medium_ratio)  # Default
        
        return weights
    
    def reset(self):
        """Reset scheduler."""
        self.current_step = 0


class DifficultyEstimator:
    """Estimate difficulty of reasoning problems."""
    
    def __init__(self):
        # Heuristic rules for difficulty estimation
        self.easy_keywords = ['simple', 'basic', 'single', 'one step']
        self.hard_keywords = ['complex', 'multiple', 'challenging', 'prove']
    
    def estimate(self, question: str, dataset: str = 'default') -> str:
        """
        Estimate difficulty of a question.
        
        Args:
            question: Question text
            dataset: Dataset name
            
        Returns:
            Difficulty label: 'easy', 'medium', or 'hard'
        """
        question_lower = question.lower()
        
        # Check for explicit difficulty markers
        if any(kw in question_lower for kw in self.easy_keywords):
            return 'easy'
        if any(kw in question_lower for kw in self.hard_keywords):
            return 'hard'
        
        # Dataset-specific heuristics
        if dataset == 'gsm8k':
            return self._estimate_gsm8k_difficulty(question)
        elif dataset == 'mmlu':
            return self._estimate_mmlu_difficulty(question)
        else:
            return 'medium'
    
    def _estimate_gsm8k_difficulty(self, question: str) -> str:
        """Estimate GSM8K difficulty based on question length and operations."""
        # Count mathematical operations
        operations = len(re.findall(r'[+\-*/]', question))
        word_count = len(question.split())
        
        if operations <= 1 and word_count < 30:
            return 'easy'
        elif operations <= 3 and word_count < 60:
            return 'medium'
        else:
            return 'hard'
    
    def _estimate_mmlu_difficulty(self, question: str) -> str:
        """Estimate MMLU difficulty."""
        word_count = len(question.split())
        
        if word_count < 20:
            return 'easy'
        elif word_count < 40:
            return 'medium'
        else:
            return 'hard'
    
    def estimate_batch(self, 
                       questions: List[str],
                       dataset: str = 'default') -> List[str]:
        """Estimate difficulties for a batch of questions."""
        return [self.estimate(q, dataset) for q in questions]


import re
