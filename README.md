# GPT-200M: A 200M Parameter Language Model from Scratch

A ~200M parameter decoder-only transformer language model built from scratch in PyTorch. Implements modern architectural techniques (RoPE, SwiGLU, knowledge distillation) and a full training pipeline from pre-training through supervised fine-tuning.

Built as a deep learning research project to understand LLM internals — every component is written from first principles with no high-level training frameworks.

## Architecture

| Component | Detail |
|-----------|--------|
| Parameters | ~200M (with weight tying) |
| Layers | 12 transformer blocks |
| Hidden dim | 1024 |
| Attention heads | 16 (d_head = 64) |
| FFN | SwiGLU (LLaMA/Mistral-style), inner dim 2688 |
| Position encoding | Rotary Position Embeddings (RoPE) |
| Vocabulary | 32,768 tokens (Mistral tokenizer) |
| Context window | 1024 tokens |
| Weight tying | Embedding ↔ LM head shared |

### Key Design Decisions

**RoPE over learned positional embeddings** — Encodes position by rotating Q/K vectors in attention, making the model sensitive to relative token distance by construction. No learned positional parameters; better length generalisation. Used in LLaMA, Mistral, Phi, and every modern open model.

**SwiGLU over GELU** — Gated feed-forward network using `SiLU(gate) * up` with three projections at 2/3 the standard FFN width, matching total parameter count. Empirically converges faster to lower loss (PaLM, LLaMA).

**Mistral tokenizer (32K vocab)** — Smaller vocabulary than GPT-2's 50K frees ~18M parameters from the embedding table for actual transformer capacity. Shared with the distillation teacher for direct logit-level comparison.

**Weight tying** — Embedding and output projection share the same weight matrix, reducing parameters while empirically improving performance.

## Training Pipeline

```
Phase 1: Pre-training (CE loss)
    High-quality structured data → language fundamentals + reasoning patterns
                    ↓
Phase 2: Knowledge Distillation
    Mistral-7B teacher → soft probability targets → richer learning signal
                    ↓
Phase 3: Supervised Fine-Tuning (SFT)
    Instruction-following data → conversational behavior
```

### Phase 1 — Pre-training

Trains on a weighted mix of high-quality educational datasets:

- **Cosmopedia** (openstax, auto_math_text, khanacademy) — synthetic textbook content
- **OpenWebMath** — mathematical proofs, papers, Stack Exchange

Each source is tokenised once and cached as a flat `uint16` binary file using streaming tokenisation (O(1) memory regardless of dataset size). Sources are mixed by weight at load time with no repetition — total tokens constrained by the most limited source relative to its weight fraction.

```bash
py train.py --no_distill
```

### Phase 2 — Knowledge Distillation

Uses Mistral-7B-v0.1 (7B parameters) as a teacher model. Instead of training only against hard one-hot targets ("the next word is X"), the student also learns to match the teacher's full probability distribution over all 32,768 tokens.

The loss function combines standard cross-entropy with a KL divergence term:

```
L = α × CE(student, targets) + (1-α) × T² × KL(teacher_soft ‖ student_soft)
```

Where soft distributions are computed at temperature T to expose more of the teacher's knowledge about token relationships. The T² factor normalises gradient magnitudes (Hinton et al., 2015).

Both models share the same tokenizer, so logit distributions are directly comparable with no vocabulary alignment needed.

```bash
py train.py --phase_start <step> --max_steps <step+N> --lr 1e-4
```

### Phase 3 — Supervised Fine-Tuning

Fine-tunes the pre-trained model on instruction-following data (SlimOrca, Dolly, Alpaca, or any HF dataset with standard schemas). Loss is computed only on assistant response tokens — prompt tokens are masked with `ignore_index=-100` so the model learns to generate responses, not memorise prompts.

```bash
py finetune.py --dataset Open-Orca/SlimOrca
```

## Project Structure

```
├── config.py       Model and training hyperparameters
├── model.py        GPT implementation (RoPE, SwiGLU, causal attention)
├── dataset.py      Multi-source dataset loading, tokenisation, caching, mixing
├── train.py        Pre-training and distillation loop
├── finetune.py     Supervised fine-tuning with instruction masking
├── generate.py     Text generation / interactive REPL
```

## Usage

### Pre-training
```bash
# Default: Cosmopedia + OpenWebMath, CE loss only
py train.py --no_distill

# Custom data sources (name:config:weight)
py train.py --no_distill --sources HuggingFaceTB/cosmopedia:openstax:2 open-web-math/open-web-math::1

# With knowledge distillation (downloads Mistral-7B-v0.1 on first run)
py train.py

# Multi-phase training (continue from checkpoint with fresh LR schedule)
py train.py --phase_start 60000 --max_steps 140000 --lr 1e-4 --warmup_steps 500
```

### Fine-tuning
```bash
py finetune.py --dataset Open-Orca/SlimOrca
py finetune.py --dataset databricks/databricks-dolly-15k --epochs 3 --lr 5e-5
```

### Text Generation
```bash
# Single prompt
py generate.py --checkpoint checkpoints/step_0070000.pt --prompt "The universe"

# Interactive mode
py generate.py --checkpoint checkpoints/step_0070000.pt

# Adjust sampling
py generate.py --checkpoint checkpoints/step_0070000.pt --prompt "def fibonacci(" \
    --temperature 0.8 --top_k 50 --max_new_tokens 300
```

## Technical Details

### Dataset Pipeline
- **Streaming tokenisation**: Documents are tokenised and written directly to binary files one at a time, using O(1) memory. Handles datasets with millions of documents (e.g., FineWeb-Edu's 9.6M docs) without memory issues.
- **Per-source caching**: Each dataset is cached independently as a flat `uint16` numpy array. Re-runs skip download/tokenisation if cache exists.
- **Weighted mixing**: Multiple sources are mixed by specified weights. Total token count is constrained by the most limited source relative to its weight fraction — no source is ever artificially repeated.
- **Automatic validation splits**: Datasets without a validation split are shuffled (seed=42) and 2% is carved off as validation.

### Training Loop
- Mixed-precision training (BF16) with gradient accumulation
- Cosine LR schedule with linear warmup
- AdamW optimiser with decoupled weight decay (applied only to 2D+ parameters)
- Gradient clipping (max norm 1.0)
- Checkpoint rotation (keeps only N most recent)
- `--phase_start` for clean multi-phase training without LR schedule artifacts
- Optional `--auto_batch` to probe maximum batch size for available VRAM

### Knowledge Distillation
- Teacher runs in inference mode (no gradients) alongside the student
- Soft targets at configurable temperature (default T=2)
- Configurable CE/KL loss weighting (default α=0.5)
- Same tokenizer ensures logit-level alignment with zero overhead
- Teacher VRAM: ~14GB (BF16); student training: ~8GB; fits on a single 32GB GPU

### SFT Implementation
- Auto-detects dataset schema (Dolly, Alpaca, prompt/completion formats)
- Labels masked with -100 for prompt tokens — loss computed only on response tokens
- Custom collator pads to longest sequence in batch
- Separate checkpoint directory to preserve pre-training weights
- Low LR (5e-5) to prevent catastrophic forgetting

## Requirements

```
torch >= 2.0
tiktoken
datasets
transformers        # for distillation teacher + Mistral tokenizer
huggingface_hub
numpy
```

Hardware: trained on a single NVIDIA RTX 5090 (32GB VRAM).

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Shazeer, "GLU Variants Improve Transformer" (2020)
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Gunasekar et al., "Textbooks Are All You Need" (2023)
- Mukherjee et al., "Orca: Progressive Learning from Complex Explanation Traces" (2023)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
