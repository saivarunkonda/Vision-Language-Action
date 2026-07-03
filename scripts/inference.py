"""
Inference script for generating responses with trained model.
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    return model, tokenizer


def generate_response(model,
                     tokenizer,
                     prompt: str,
                     max_new_tokens: int = 512,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     device: str = "cuda") -> str:
    """Generate response from model."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Generate responses with trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, args.device)
    
    # Generate response
    print("Generating response...")
    response = generate_response(
        model,
        tokenizer,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.device
    )
    
    print("\n" + "="*50)
    print("RESPONSE:")
    print("="*50)
    print(response)
    print("="*50)


if __name__ == "__main__":
    main()
