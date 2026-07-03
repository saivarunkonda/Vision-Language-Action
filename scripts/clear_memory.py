"""
Clear GPU memory and cache.
Run this in Kaggle before starting training to free up memory.
"""

import torch
import gc

print("Clearing GPU memory and cache...")

# Clear PyTorch cache
torch.cuda.empty_cache()

# Force garbage collection
gc.collect()

# Clear additional cache
if hasattr(torch.cuda, 'memory'):
    torch.cuda.memory.empty_cache()
    torch.cuda.ipc_collect()

print("✓ GPU memory and cache cleared")
print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Optional: Clear HuggingFace cache (uncomment if needed)
# print("\nClearing HuggingFace cache...")
# import shutil
# import os
# hf_cache = os.path.expanduser("~/.cache/huggingface")
# if os.path.exists(hf_cache):
#     shutil.rmtree(hf_cache)
#     print("✓ HuggingFace cache cleared")
