"""
StrategyQA evaluation script.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Dict, List
from tqdm import tqdm


class StrategyQAEvaluator:
    """Evaluator for StrategyQA benchmark."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize StrategyQA evaluator.
        
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
        """Load StrategyQA dataset."""
        dataset = load_dataset("EleutherAI/strategyqa", split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset
    
    def format_prompt(self, question: str) -> str:
        """Format prompt for StrategyQA."""
        prompt = f"""Answer the following yes/no question with reasoning. Provide your step-by-step thinking and then give the final answer.

Question: {question}

Answer:"""
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """Extract yes/no answer from response."""
        response = response.strip().lower()
        
        # Look for explicit yes/no
        if "yes" in response and "no" not in response:
            return "yes"
        elif "no" in response and "yes" not in response:
            return "no"
        
        # Check if answer starts with yes/no
        if response.startswith("yes"):
            return "yes"
        elif response.startswith("no"):
            return "no"
        
        # Look for final answer pattern
        patterns = [
            r'(?:answer|therefore|thus|so)\s*(?:is|:)?\s*(yes|no)',
            r'(?:the answer is|final answer)\s*(?::)?\s*(yes|no)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).lower()
        
        return ""
    
    def evaluate(self, max_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate model on StrategyQA.
        
        Args:
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of metrics
        """
        import re
        
        dataset = self.load_dataset("test", max_samples)
        
        correct = 0
        total = 0
        results = []
        
        for example in tqdm(dataset, desc="Evaluating StrategyQA"):
            question = example["question"]
            answer = example["answer"]
            
            # Ground truth answer (boolean to yes/no)
            gt_answer = "yes" if answer else "no"
            
            # Generate response
            prompt = self.format_prompt(question)
            response = self.generate(prompt)
            
            # Extract predicted answer
            pred_answer = self.extract_answer(response)
            
            # Check correctness
            is_correct = pred_answer == gt_answer
            
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
    parser.add_argument("--output", type=str, default="strategyqa_results.json", help="Output file")
    
    args = parser.parse_args()
    
    evaluator = StrategyQAEvaluator(args.model, args.device)
    results = evaluator.evaluate(args.max_samples)
    
    print(f"StrategyQA Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    
    # Save results
    import json
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
