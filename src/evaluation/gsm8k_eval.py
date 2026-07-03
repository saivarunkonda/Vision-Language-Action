"""
GSM8K evaluation script.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Dict, List
import re
from tqdm import tqdm


class GSM8KEvaluator:
    """Evaluator for GSM8K benchmark."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize GSM8K evaluator.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run evaluation on
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dataset(self, split: str = "test", max_samples: int = None):
        """Load GSM8K dataset."""
        dataset = load_dataset("gsm8k", "main", split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset
    
    def format_prompt(self, question: str) -> str:
        """Format prompt for GSM8K."""
        prompt = f"""Solve the following math problem step by step. Show your work and give the final answer.

Question: {question}

Answer:"""
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """Extract numerical answer from response."""
        # Look for final answer pattern
        patterns = [
            r'(?:answer|is|equals?)\s*[:=]?\s*([+-]?\d*\.?\d+)',
            r'(?:therefore|thus|so)\s*,?\s*([+-]?\d*\.?\d+)',
            r'####\s*([+-]?\d*\.?\d+)',  # GSM8K format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: extract last number
        numbers = re.findall(r'([+-]?\d*\.?\d+)', response)
        if numbers:
            return numbers[-1]
        
        return ""
    
    def normalize_answer(self, answer: str) -> float:
        """Normalize answer to float."""
        try:
            # Remove commas and other formatting
            answer = answer.replace(',', '').strip()
            return float(answer)
        except ValueError:
            return None
    
    def evaluate(self, max_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate model on GSM8K.
        
        Args:
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of metrics
        """
        dataset = self.load_dataset("test", max_samples)
        
        correct = 0
        total = 0
        results = []
        
        for example in tqdm(dataset, desc="Evaluating GSM8K"):
            question = example["question"]
            ground_truth = example["answer"]
            
            # Extract ground truth answer
            gt_answer = self.extract_answer(ground_truth)
            gt_normalized = self.normalize_answer(gt_answer)
            
            # Generate response
            prompt = self.format_prompt(question)
            response = self.generate(prompt)
            
            # Extract predicted answer
            pred_answer = self.extract_answer(response)
            pred_normalized = self.normalize_answer(pred_answer)
            
            # Check correctness
            is_correct = False
            if gt_normalized is not None and pred_normalized is not None:
                is_correct = abs(gt_normalized - pred_normalized) < 0.01
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "question": question,
                "ground_truth": gt_answer,
                "prediction": pred_answer,
                "response": response,
                "correct": is_correct
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default="gsm8k_results.json", help="Output file")
    
    args = parser.parse_args()
    
    evaluator = GSM8KEvaluator(args.model, args.device)
    results = evaluator.evaluate(args.max_samples)
    
    print(f"GSM8K Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    
    # Save results
    import json
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
