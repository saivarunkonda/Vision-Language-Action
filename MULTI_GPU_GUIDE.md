# Multi-GPU Training Guide

## Overview

This project now supports true multi-GPU data parallel training using HuggingFace Accelerate. This allows you to utilize multiple GPUs simultaneously for faster training.

## What Changed

### Before (Model Parallelism)
- Used `device_map="auto"` to split a single model across GPUs
- Only beneficial for models too large for one GPU
- No speedup from parallel batch processing

### After (Data Parallelism)
- Uses Accelerate for distributed data parallel training
- Each GPU processes different batches simultaneously
- Near-linear speedup with multiple GPUs
- Automatic gradient synchronization across GPUs

## Architecture

```
GPU 0: Model Copy → Batch 1 → Gradients ↘
                                         → Synchronize → Update
GPU 1: Model Copy → Batch 2 → Gradients ↗
```

## Configuration

### Accelerate Config (`configs/accelerate_config.yaml`)
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_processes: 2  # Number of GPUs
```

### Training Config (`configs/base_config.yaml`)
```yaml
training:
  batch_size: 8  # Per GPU batch size
  gradient_accumulation_steps: 4

hardware:
  num_gpus: 2
  mixed_precision: "bf16"
```

**Effective Batch Size Calculation:**
```
effective_batch_size = batch_size × num_gpus × gradient_accumulation_steps
                     = 8 × 2 × 4 = 64
```

## Usage

### Option 1: Use Pre-configured Script
```bash
# Activate venv
venv\Scripts\activate

# Train with 2 GPUs
python scripts/train_multigpu.py --config configs/base_config.yaml --num_gpus 2

# Train with all available GPUs
python scripts/train_multigpu.py --config configs/base_config.yaml
```

### Option 2: Use Accelerate Directly
```bash
# Activate venv
venv\Scripts\activate

# Train with accelerate
accelerate launch --config_file configs/accelerate_config.yaml scripts/train_rl.py --config configs/base_config.yaml
```

### Option 3: Interactive Configuration
```bash
# Run configuration wizard
python scripts/configure_accelerate.py

# Then train
accelerate launch scripts/train_rl.py --config configs/base_config.yaml
```

## Performance

### Expected Speedup
- **2 GPUs**: ~1.8-2.0x speedup (near-linear)
- **4 GPUs**: ~3.5-3.8x speedup
- **8 GPUs**: ~7.0-7.5x speedup

### Memory Usage
- Each GPU stores a full copy of the model
- Reference model only loaded on main process (saves memory)
- Gradient checkpointing reduces memory footprint

## Tuning for Your Hardware

### For 24GB VRAM GPUs (Recommended)
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4
```

### For 16GB VRAM GPUs
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8
```

### For 12GB VRAM GPUs
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 16
```

### For 8GB VRAM GPUs
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 32
  gradient_checkpointing: true  # Must be enabled
```

## Key Implementation Details

### 1. Distributed Sampling
- Uses `DistributedSampler` to ensure each GPU gets different data
- Shuffles data differently on each epoch
- Prevents duplicate processing across GPUs

### 2. Gradient Accumulation
- Accumulates gradients over multiple batches before updating
- Allows larger effective batch sizes with limited memory
- Handled automatically by Accelerate

### 3. Mixed Precision
- Uses BF16 (Bfloat16) for faster computation
- Reduces memory usage by ~50%
- Maintains numerical stability

### 4. Checkpointing
- Only main process saves checkpoints
- Uses `accelerator.save_state()` for distributed state
- Includes optimizer state and scheduler

### 5. Logging
- Only main process logs to console
- Metrics gathered from all processes and averaged
- WandB logging from main process only

## Troubleshooting

### "Out of Memory"
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Ensure `gradient_checkpointing: true`

### "CUDA Out of Memory" on One GPU
- Check if both GPUs have same VRAM
- Reduce per-GPU batch size
- Close other GPU-intensive applications

### Slow Training
- Verify both GPUs are being used (check `nvidia-smi`)
- Ensure mixed precision is enabled
- Check if data loading is bottleneck (increase `num_workers`)

### "NCCL Error"
- Ensure NCCL is properly installed
- Check GPU communication (same PCIe lane preferred)
- Try setting `NCCL_P2P_DISABLE=1` environment variable

### "Distributed Data Not Equal"
- Ensure `DistributedSampler` is used
- Check `set_epoch()` is called each epoch
- Verify seed is set correctly

## Monitoring GPU Usage

### Check GPU Utilization
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

### Expected Output
```
GPU 0: 80-95% utilization, 20-22GB VRAM
GPU 1: 80-95% utilization, 20-22GB VRAM
```

If one GPU is at 0%, check:
- Accelerate configuration
- Number of processes matches GPUs
- No CUDA errors in logs

## Single GPU Fallback

If you only have one GPU, the training script automatically falls back to single GPU mode:

```bash
python scripts/train_rl.py --config configs/base_config.yaml
```

No changes needed - it will detect available GPUs and adapt accordingly.

## Best Practices

1. **Use Mixed Precision**: Always enable BF16 if supported
2. **Gradient Checkpointing**: Enable for large models or limited VRAM
3. **Batch Size Tuning**: Find the largest batch size that fits in VRAM
4. **Gradient Accumulation**: Use to achieve desired effective batch size
5. **Monitoring**: Watch GPU utilization to ensure both GPUs are working
6. **Checkpointing**: Save frequently to avoid losing progress
7. **Data Loading**: Use multiple workers to prevent GPU starvation

## Advanced: Customizing Number of GPUs

To use a specific number of GPUs (e.g., 4 out of 8 available):

```bash
# Using accelerate launch
accelerate launch --num_processes=4 scripts/train_rl.py --config configs/base_config.yaml

# Using the wrapper script
python scripts/train_multigpu.py --config configs/base_config.yaml --num_gpus 4
```

## Summary

Multi-GPU training is now fully integrated with:
- ✓ Distributed data parallelism via Accelerate
- ✓ Automatic gradient synchronization
- ✓ Efficient checkpointing
- ✓ Proper logging across processes
- ✓ Configurable for any number of GPUs
- ✓ Optimized for 2× 24GB VRAM setup

For optimal performance with your 2× 24GB GPU setup, use the default configuration in `configs/base_config.yaml`.
