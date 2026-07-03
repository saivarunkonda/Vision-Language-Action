# Project Architecture

This document describes the proposed system architecture, data flow, technical plausibility, constraints, and a short implementation checklist for the SLM reasoning pipeline.

## Visual Data Flow (Mermaid)

Copy the block below into a Mermaid renderer (mermaid.live or VS Code Mermaid preview) to view/export the diagram.

```mermaid
flowchart LR
  subgraph Data_Sources[Data Sources]
    A[GSM8K / MMLU / StrategyQA / AquaRAT]
  end

  subgraph Ingest[Data Ingest]
    DL[`Data Loader\n(src/utils/data_loader.py)`]
    Preproc[Prompting & Preprocessing]
  end

  A --> DL --> Preproc

  subgraph Training[Training Pipeline]
    SFT[`SFT Warmstart\n(Supervised CoT fine-tune)`]
    Policy[`Policy SLM\n(Phi-3-mini / Qwen-2.5)`]
    Ref[`Reference Model\n(frozen baseline)`]
    PPO[`PPO Loop\n(src/training/ppo_trainer.py)`]
    Critic[`Value Head (critic)`]
    Reward[`CombinedReward\n(src/rewards/reward_functions.py)`]
    KL[`Adaptive KL Controller`]
    Curriculum[`Curriculum Scheduler\n(src/training/curriculum.py)`]
  end

  Preproc --> SFT --> Policy
  Policy --> PPO
  Ref --> PPO
  Reward --> PPO
  Curriculum --> PPO
  PPO --> Critic
  PPO --> Checkpoints[(Checkpoints & Logs)]

  subgraph Eval_Deploy[Evaluation & Deployment]
    Eval[`Evaluators\n(src/evaluation/*.py)`]
    SelfCons[`Self-Consistency + Deterministic Eval`]
    Distill[`Distillation / Quantization`]
    Deploy[`Deployment (edge/cloud)`]
    Metrics[(KPIs: GSM8K, MMLU, StrategyQA, Latency)]
  end

  Checkpoints --> Eval --> SelfCons --> Metrics
  Checkpoints --> Distill --> Deploy --> Metrics
  Eval --> Metrics
```

## Architecture Overview

- Data: load and format datasets (GSM8K, MMLU, StrategyQA, AquaRAT) using `src/utils/data_loader.py`; generate chain-of-thought (CoT) prompts for SFT and RL.
- SFT Warmstart: supervised fine-tune the SLM on CoT-augmented examples to build a stable initial policy.
- PPO Loop: `src/training/ppo_trainer.py` runs generation → scoring → PPO update. The trainer includes policy updates, an added value head (critic), entropy bonus, and adaptive KL regularization.
- Reward: `src/rewards/reward_functions.py` implements a CombinedReward (outcome + process); plan to add a learned reward model trained on anonymized preference pairs.
- Curriculum: `src/training/curriculum.py` provides easy→hard sampling schedules to stabilize learning.
- Evaluation: `src/evaluation/gsm8k_eval.py` and `src/evaluation/mmlu_eval.py` support deterministic and self-consistency evaluation. Best checkpoints are candidates for distillation and deployment.

## Technical Plausibility

- PPO with a critic and adaptive KL is a tested approach for LM fine-tuning; SFT warmstart is standard practice to stabilize RL on low-capacity models.
- Self-consistency and deterministic evaluation are proven to improve GSM8K accuracy for math reasoning.
- With bf16, gradient checkpointing, and gradient accumulation, training 3–7B parameter models on 2×24GB GPUs is feasible (longer runtimes but practical).

## Constraints & Failure Modes

- Compute: recommended 2×24GB GPUs, 12+ CPU cores, 64GB RAM. Smaller infra requires more gradient accumulation or smaller models.
- Reward noise: heuristic rewards can mislabel CoT outputs. Mitigations: canonicalize answers, programmatic checks, and train a learned reward model.
- RL instability: mitigate using SFT warmstart, critic/value loss, adaptive KL, careful LR/entropy tuning, and gradient clipping. Monitor KL and value loss closely.
- Licensing & privacy: do not redistribute base checkpoints that forbid redistribution. Remove PII from any published data.

## Implementation Checklist (short)

1. Run SFT warmstart on a CoT-augmented subset (validate generation quality).  
2. Run small-scale PPO with critic on a GSM8K subset; log per-sample rewards and KL stats.  
3. Validate deterministic & self-consistency evaluation with `src/evaluation/gsm8k_eval.py`.  
4. Train small learned reward model from anonymized preference pairs (optional).  
5. Distill/quantize best checkpoint and measure latency on target hardware.

## How to render/export the diagram

- Use mermaid.live or the Mermaid extension in VS Code to paste the Mermaid block and export PNG/SVG.  
- Or install the `mmdc` (Mermaid CLI) and run:  

```bash
npx @mermaid-js/mermaid-cli -i docs/ARCHITECTURE.md -o docs/architecture.png
```

Replace input/output flags as needed. For quick inclusion in a PPT, export as PNG or SVG.

---

If you want I can also: (A) add a one-slide PPT-ready text version, (B) export the diagram as `docs/architecture.png`, or (C) create speaker notes per section—tell me which.
