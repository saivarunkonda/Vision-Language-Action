# Enhancing Reasoning in Small Language Models with RL

This project implements Reinforcement Learning techniques to improve reasoning capabilities in Small Language Models (≤7B parameters) while maintaining efficiency.

## Target Benchmarks
- **GSM8K**: ≥50% accuracy (≥+5% over baseline)
- **MMLU**: ≥45% accuracy (≥+5% over baseline)  
- **StrategyQA**: ≥65% accuracy (≥+5% over baseline)

## Project Structure
```
.
├── configs/              # Configuration files
├── data/                 # Datasets and preprocessing
├── models/               # Model checkpoints
├── src/
│   ├── training/         # RL training pipeline
│   ├── rewards/          # Reward mechanisms
│   ├── evaluation/       # Benchmark evaluation
│   └── utils/            # Utilities
├── scripts/              # Training and evaluation scripts
├── logs/                 # Training logs
└── requirements.txt      # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Download datasets** (automatically handled by scripts)
2. **Train with RL**:
   ```bash
   # Single GPU
   python scripts/train_rl.py --config configs/base_config.yaml
   
   # Multi-GPU (recommended for 2+ GPUs)
   python scripts/train_multigpu.py --config configs/base_config.yaml --num_gpus 2
   ```
3. **Evaluate**:
   ```bash
   python scripts/evaluate.py --model models/checkpoint --benchmark gsm8k
   ```

## Supported Models
- Qwen 2.5 7B
- Phi-3-Mini (3.8B)
- Gemma 4 E4B

## Key Features
- **Multi-GPU Training**: Distributed training with Accelerate for near-linear speedup
- **Outcome + Process-based Rewards**: Combines answer correctness with reasoning quality
- **KL Divergence Tuning**: Adaptive penalty for stability
- **Entropy Regularization**: Maintains exploration
- **Curriculum Learning**: Progressive difficulty scheduling
- **Efficient Training**: Mixed precision, gradient checkpointing

## Hardware Requirements
- 2× NVIDIA GPU with ≥24 GB VRAM
- 12+ CPU cores
- 64 GB system RAM

## Citation
If you use this code, please cite the original benchmarks and models used.
