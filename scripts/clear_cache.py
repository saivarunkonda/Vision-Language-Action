"""
Script to clear HuggingFace cache for specific models.
Run this if you get KeyError or compatibility issues with cached models.
"""

import os
import shutil
from huggingface_hub import scan_cache_dir


def clear_model_cache(model_name: str):
    """Clear cache for a specific model."""
    print(f"Scanning cache for {model_name}...")
    
    # Scan cache
    cache_info = scan_cache_dir()
    
    # Find the model in cache
    found = False
    for repo in cache_info.repos:
        if model_name in repo.repo_id:
            print(f"\nFound cached model: {repo.repo_id}")
            print(f"Size: {repo.size_on_disk / (1024**3):.2f} GB")
            print(f"Location: {repo.repo_path}")
            
            # Delete the cache
            print(f"\nDeleting cache...")
            shutil.rmtree(repo.repo_path)
            print(f"✓ Cache cleared for {repo.repo_id}")
            found = True
    
    if not found:
        print(f"No cache found for {model_name}")
    else:
        print("\n✓ Cache clearing complete")
        print("Next time you run training, the model will be re-downloaded")


if __name__ == "__main__":
    # Clear Phi-3 cache
    clear_model_cache("microsoft/Phi-3-mini-4k-instruct")
    
    # Also clear Qwen cache if needed
    # clear_model_cache("Qwen/Qwen2.5-7B")
