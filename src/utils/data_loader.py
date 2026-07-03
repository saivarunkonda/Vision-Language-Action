"""
Data loading utilities for training.
"""

from datasets import load_dataset
from typing import Dict, List, Optional
from torch.utils.data import Dataset
import random


class ReasoningDataset(Dataset):
    """Dataset for reasoning tasks."""
    
    def __init__(self,
                 dataset_name: str,
                 split: str = "train",
                 max_samples: Optional[int] = None,
                 include_difficulty: bool = False):
        """
        Initialize reasoning dataset.
        
        Args:
            dataset_name: Name of dataset (gsm8k, aqua_rat, etc.)
            split: Dataset split
            max_samples: Maximum number of samples
            include_difficulty: Whether to include difficulty labels
        """
        self.dataset_name = dataset_name
        self.split = split
        self.include_difficulty = include_difficulty
        
        data = self._load_dataset(dataset_name, split)
        
        if max_samples:
            data = data[:max_samples]
        
        self.data = data
    
    def _load_dataset(self, dataset_name: str, split: str) -> List[Dict]:
        """Load dataset from HuggingFace."""
        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split=split)
            return [
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "dataset": "gsm8k"
                }
                for item in dataset
            ]
        
        elif dataset_name == "aqua_rat":
            dataset = load_dataset("aqua_rat", split=split)
            return [
                {
                    "question": item["question"],
                    "options": item["options"],
                    "answer": item["correct"],
                    "dataset": "aqua_rat"
                }
                for item in dataset
            ]
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        if self.include_difficulty:
            # Estimate difficulty (simplified)
            difficulty = self._estimate_difficulty(item["question"])
            item["difficulty"] = difficulty
        
        return item
    
    def _estimate_difficulty(self, question: str) -> str:
        """Simple difficulty estimation."""
        word_count = len(question.split())
        
        if word_count < 30:
            return "easy"
        elif word_count < 60:
            return "medium"
        else:
            return "hard"


def format_prompt(item: Dict, dataset: str = "default") -> str:
    """Format prompt for training."""
    question = item["question"]
    
    if dataset == "gsm8k":
        prompt = f"""Solve the following math problem step by step.

Question: {question}

Answer:"""
    elif dataset == "aqua_rat":
        options = item.get("options", [])
        options_str = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        prompt = f"""Solve the following problem step by step.

Question: {question}

Options:
{options_str}

Answer:"""
    else:
        prompt = f"""Question: {question}

Answer:"""
    
    return prompt


def collate_fn(batch: List[Dict], tokenizer, max_length: int = 1024):
    """Collate function for dataloader."""
    prompts = [format_prompt(item, item["dataset"]) for item in batch]
    answers = [item["answer"] for item in batch]
    
    # Tokenize prompts
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        "prompts": prompts,
        "answers": answers,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }
