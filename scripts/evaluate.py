"""
Evaluation script for trained models.
"""

import os
import sys
import argparse
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import GSM8KEvaluator, MMLUEvaluator, StrategyQAEvaluator


def evaluate_benchmark(model_path: str, benchmark: str, max_samples: int = 1000, device: str = "cuda"):
    """Evaluate model on a specific benchmark."""
    print(f"Evaluating {benchmark}...")
    
    if benchmark == "gsm8k":
        evaluator = GSM8KEvaluator(model_path, device)
        results = evaluator.evaluate(max_samples)
    elif benchmark == "mmlu":
        evaluator = MMLUEvaluator(model_path, device)
        results = evaluator.evaluate(max_samples)
    elif benchmark == "strategyqa":
        evaluator = StrategyQAEvaluator(model_path, device)
        results = evaluator.evaluate(max_samples)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--benchmark", type=str, choices=["gsm8k", "mmlu", "strategyqa", "all"], 
                        default="all", help="Benchmark to evaluate")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples per benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    benchmarks = ["gsm8k", "mmlu", "strategyqa"] if args.benchmark == "all" else [args.benchmark]
    
    all_results = {}
    
    for benchmark in benchmarks:
        try:
            results = evaluate_benchmark(args.model, benchmark, args.max_samples, args.device)
            all_results[benchmark] = results
            print(f"{benchmark.upper()} Accuracy: {results['accuracy']:.2%}")
        except Exception as e:
            print(f"Error evaluating {benchmark}: {e}")
            all_results[benchmark] = {"error": str(e)}
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
