# Quick Start Guide

This guide will help you get started with training and evaluating SLMs using RL for reasoning.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 2× NVIDIA GPU with ≥24 GB VRAM)
- 64 GB system RAM
- 12+ CPU cores

## Installation

1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Training

### Single GPU Training

```bash
# Activate venv first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Train with Qwen 2.5 7B
python scripts/train_rl.py --config configs/base_config.yaml

# Train with Phi-3-Mini (3.8B)
python scripts/train_rl.py --config configs/phi3_config.yaml
```

### Multi-GPU Training (Recommended)

```bash
# Activate venv first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Train with 2 GPUs (uses Accelerate for distributed training)
python scripts/train_multigpu.py --config configs/base_config.yaml --num_gpus 2

# Or use all available GPUs
python scripts/train_multigpu.py --config configs/base_config.yaml
```

**Multi-GPU Benefits:**
- Near-linear speedup with multiple GPUs
- Larger effective batch size (batch_size × num_gpus × gradient_accumulation)
- Better utilization of GPU resources
- Automatic gradient synchronization across GPUs

**Configuration for Multi-GPU:**
- Per-GPU batch size: 8 (configurable in `configs/base_config.yaml`)
- Gradient accumulation: 4
- With 2 GPUs: Effective batch size = 8 × 2 × 4 = 64

### Resume training from checkpoint

```bash
# Single GPU
python scripts/train_rl.py --config configs/base_config.yaml --resume models/checkpoints/checkpoint-1000

# Multi-GPU
python scripts/train_multigpu.py --config configs/base_config.yaml --resume models/checkpoints/checkpoint-1000
```

## Evaluation

### Evaluate on all benchmarks

```bash
python scripts/evaluate.py --model models/checkpoints/final_model --benchmark all
```

### Evaluate on specific benchmark

```bash
# GSM8K
python scripts/evaluate.py --model models/checkpoints/final_model --benchmark gsm8k

# MMLU
python scripts/evaluate.py --model models/checkpoints/final_model --benchmark mmlu

# StrategyQA
python scripts/evaluate.py --model models/checkpoints/final_model --benchmark strategyqa
```

## Inference

Generate responses with your trained model:

```bash
python scripts/inference.py --model models/checkpoints/final_model --prompt "Solve step by step: What is 15% of 240?"
```

## Configuration

Edit the configuration files in `configs/` to customize:

- **Model selection**: Change `model.name` in config
- **Training parameters**: Adjust learning rate, batch size, etc.
- **RL parameters**: Tune KL penalty, entropy, reward weights
- **Curriculum**: Enable/disable curriculum learning, adjust stages
- **Data**: Change datasets, max samples

## Expected Results

Target improvements over baseline:

| Benchmark | Target | Minimum |
|-----------|--------|---------|
| GSM8K | ≥50% (+5%) | 45% |
| MMLU | ≥45% (+5%) | 40% |
| StrategyQA | ≥65% (+5%) | 60% |

## Monitoring Training

Training logs are saved to:
- **TensorBoard**: `logs/` directory
- **Weights & Biases**: If enabled in config

View TensorBoard:
```bash
tensorboard --logdir logs
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `gradient_accumulation_steps`
- Enable gradient checkpointing (already enabled by default)

### Slow Training
- Use mixed precision (BF16) - enabled by default
- Increase `batch_size` if memory allows
- Use multiple GPUs

### Poor Results
- Adjust reward weights (`outcome_weight`, `process_weight`)
- Tune KL penalty (`initial_kl_coef`, `target_kl`)
- Increase training steps
- Enable curriculum learning

## Project Structure

```
.
├── configs/              # Configuration files
│   ├── base_config.yaml   # Qwen 2.5 7B config
│   └── phi3_config.yaml   # Phi-3-Mini config
├── data/                 # Datasets (auto-downloaded)
├── models/               # Model checkpoints
│   └── checkpoints/       # Training checkpoints
├── scripts/              # Training/evaluation scripts
│   ├── train_rl.py        # Main training script
│   ├── evaluate.py        # Evaluation script
│   └── inference.py       # Inference script
├── src/
│   ├── training/          # RL training logic
│   ├── rewards/           # Reward functions
│   ├── evaluation/        # Benchmark evaluators
│   └── utils/             # Utilities
├── logs/                 # Training logs
└── requirements.txt      # Dependencies
```

## Next Steps

1. **Baseline evaluation**: First evaluate the base model without RL training
2. **Train with RL**: Run training script with your chosen config
3. **Evaluate**: Compare RL-trained model against baseline
4. **Iterate**: Adjust hyperparameters based on results
