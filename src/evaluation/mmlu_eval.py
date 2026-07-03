"""
MMLU evaluation script.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Dict, List
from tqdm import tqdm
import numpy as np


class MMLUEvaluator:
    """Evaluator for MMLU benchmark."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize MMLU evaluator.
        
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
        """Load MMLU dataset."""
        # MMLU has multiple subjects, load all
        subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios",
            "nutrition", "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions"
        ]
        
        all_data = []
        for subject in subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split=split)
                for item in dataset:
                    item["subject"] = subject
                    all_data.append(item)
            except:
                continue
        
        if max_samples:
            all_data = all_data[:max_samples]
        
        return all_data
    
    def format_prompt(self, question: str, choices: List[str]) -> str:
        """Format prompt for MMLU."""
        prompt = f"""Answer the following multiple-choice question. Choose the best option (A, B, C, or D).

Question: {question}

Options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer:"""
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """Extract answer choice from response."""
        response = response.strip().upper()
        
        # Look for explicit choice
        for choice in ["A", "B", "C", "D"]:
            if response.startswith(choice) or f"{choice})" in response or f"({choice})" in response:
                return choice
        
        # Fallback: look for choice in response
        for choice in ["A", "B", "C", "D"]:
            if choice in response:
                return choice
        
        return ""
    
    def evaluate(self, max_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate model on MMLU.
        
        Args:
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of metrics
        """
        dataset = self.load_dataset("test", max_samples)
        
        correct = 0
        total = 0
        subject_correct = {}
        subject_total = {}
        results = []
        
        for example in tqdm(dataset, desc="Evaluating MMLU"):
            question = example["question"]
            choices = example["choices"]
            answer_idx = example["answer"]
            subject = example["subject"]
            
            # Ground truth choice
            gt_choice = chr(65 + answer_idx)  # 0 -> A, 1 -> B, etc.
            
            # Generate response
            prompt = self.format_prompt(question, choices)
            response = self.generate(prompt)
            
            # Extract predicted choice
            pred_choice = self.extract_answer(response)
            
            # Check correctness
            is_correct = pred_choice == gt_choice
            
            if is_correct:
                correct += 1
            
            total += 1
            
            # Track per-subject accuracy
            if subject not in subject_correct:
                subject_correct[subject] = 0
                subject_total[subject] = 0
            subject_correct[subject] += int(is_correct)
            subject_total[subject] += 1
            
            results.append({
                "question": question,
                "choices": choices,
                "ground_truth": gt_choice,
                "prediction": pred_choice,
                "response": response,
                "correct": is_correct,
                "subject": subject
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Compute per-subject accuracy
        subject_accuracy = {
            subject: subject_correct[subject] / subject_total[subject]
            for subject in subject_correct
        }
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "subject_accuracy": subject_accuracy,
            "results": results
        }
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
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
                temperature=0.5,
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
    parser.add_argument("--output", type=str, default="mmlu_results.json", help="Output file")
    
    args = parser.parse_args()
    
    evaluator = MMLUEvaluator(args.model, args.device)
    results = evaluator.evaluate(args.max_samples)
    
    print(f"MMLU Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    
    # Save results
    import json
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
