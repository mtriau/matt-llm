"""
Evaluation benchmarks for the trained model.

Runs HellaSwag and ARC-Easy in log-likelihood scoring mode:
for each question, score every candidate completion and pick the highest.

Usage:
    py eval.py --checkpoint checkpoints/step_0040000.pt
    py eval.py --checkpoint checkpoints/best.pt --benchmarks hellaswag arc_easy
"""

import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from config import ModelConfig
from model import GPT
from dataset import get_tokenizer


# ---------------------------------------------------------------------------
# Model loading (mirrors generate.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_cfg", ModelConfig())
    model = GPT(cfg).to(device)
    state = ckpt["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Log-likelihood scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_completion(model, enc, context_text: str, completion_text: str,
                     device: torch.device, max_seq_len: int) -> float:
    """
    Compute the average log-probability of completion_text given context_text.
    Returns the mean log-prob per token of the completion.
    """
    ctx_tokens = enc.encode_ordinary(context_text)
    comp_tokens = enc.encode_ordinary(completion_text)

    if not comp_tokens:
        return float("-inf")
    if not ctx_tokens:
        ctx_tokens = [1]  # fallback: use a dummy token

    # Truncate context from the left if too long
    max_ctx = max_seq_len - len(comp_tokens) - 1
    if max_ctx < 1:
        max_ctx = 1
    if len(ctx_tokens) > max_ctx:
        ctx_tokens = ctx_tokens[-max_ctx:]

    all_tokens = ctx_tokens + comp_tokens
    if len(all_tokens) > max_seq_len:
        all_tokens = all_tokens[:max_seq_len]

    idx = torch.tensor([all_tokens], dtype=torch.long, device=device)
    logits, _ = model(idx)          # (1, T, vocab), loss
    logits = logits[0]              # (T, vocab)

    # We only care about the log-probs at positions corresponding to completion tokens
    # Position i predicts token i+1, so completion starts at position len(ctx_tokens)-1
    start = len(ctx_tokens) - 1
    end = len(all_tokens) - 1

    if start >= end:
        return float("-inf")

    log_probs = F.log_softmax(logits[start:end].float(), dim=-1)
    target_tokens = torch.tensor(all_tokens[start + 1:end + 1], dtype=torch.long,
                                 device=device)

    token_log_probs = log_probs[range(len(target_tokens)), target_tokens]
    return token_log_probs.mean().item()


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------

def eval_hellaswag(model, enc, device, max_seq_len, limit=0):
    """
    HellaSwag: pick the most likely sentence completion from 4 choices.
    Uses the validation split.
    """
    print("\n=== HellaSwag ===")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for item in tqdm(ds, desc="HellaSwag"):
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        scores = []
        for ending in endings:
            s = score_completion(model, enc, ctx, " " + ending, device, max_seq_len)
            scores.append(s)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1

    acc = correct / total * 100
    print(f"HellaSwag: {correct}/{total} = {acc:.1f}%")
    return acc


# ---------------------------------------------------------------------------
# ARC-Easy
# ---------------------------------------------------------------------------

def eval_arc_easy(model, enc, device, max_seq_len, limit=0):
    """
    ARC-Easy: elementary science multiple-choice questions.
    Score each "Question: ... Answer: <choice>" and pick the highest.
    """
    print("\n=== ARC-Easy ===")
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for item in tqdm(ds, desc="ARC-Easy"):
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        ctx = f"Question: {question}\nAnswer:"

        scores = []
        for choice in choices:
            s = score_completion(model, enc, ctx, " " + choice, device, max_seq_len)
            scores.append(s)

        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        pred_label = labels[pred_idx]

        if pred_label == answer_key:
            correct += 1
        total += 1

    acc = correct / total * 100
    print(f"ARC-Easy: {correct}/{total} = {acc:.1f}%")
    return acc


# ---------------------------------------------------------------------------
# LAMBADA
# ---------------------------------------------------------------------------

def eval_lambada(model, enc, device, max_seq_len, limit=0):
    """
    LAMBADA: predict the last word of a passage.
    Score = accuracy of the model's top-1 prediction for the final word.
    """
    print("\n=== LAMBADA ===")
    ds = load_dataset("cimec/lambada", split="test")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for item in tqdm(ds, desc="LAMBADA"):
        text = item["text"]
        # Split into context (all but last word) and target (last word)
        words = text.rsplit(" ", 1)
        if len(words) != 2:
            continue
        context, last_word = words

        ctx_tokens = enc.encode_ordinary(context + " ")
        target_tokens = enc.encode_ordinary(last_word)
        if not target_tokens:
            continue

        # Truncate context from left if needed
        if len(ctx_tokens) + len(target_tokens) > max_seq_len:
            ctx_tokens = ctx_tokens[-(max_seq_len - len(target_tokens)):]

        all_tokens = ctx_tokens + target_tokens
        idx = torch.tensor([all_tokens], dtype=torch.long, device=device)
        logits, _ = model(idx)  # (1, T, vocab)
        logits = logits[0]      # (T, vocab)

        # Check if model predicts each target token correctly (greedy)
        start = len(ctx_tokens) - 1
        all_correct = True
        for i, t in enumerate(target_tokens):
            pred = logits[start + i].argmax().item()
            if pred != t:
                all_correct = False
                break

        if all_correct:
            correct += 1
        total += 1

    acc = correct / total * 100
    print(f"LAMBADA: {correct}/{total} = {acc:.1f}%")
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "hellaswag": eval_hellaswag,
    "arc_easy": eval_arc_easy,
    "lambada": eval_lambada,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on standard benchmarks")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["hellaswag", "arc_easy", "lambada"],
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of examples per benchmark (0 = all)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    max_seq_len = model.cfg.max_seq_len
    print(f"Model loaded ({model.count_parameters()/1e6:.1f}M params)")

    enc = get_tokenizer()

    results = {}
    for name in args.benchmarks:
        acc = BENCHMARKS[name](model, enc, device, max_seq_len, limit=args.limit)
        results[name] = acc

    print("\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    for name, acc in results.items():
        print(f"  {name:12s}: {acc:.1f}%")
    print("=" * 40)

    # Reference baselines (random chance)
    print("\nRandom baselines: HellaSwag=25.0%, ARC-Easy=25.0%, LAMBADA=0.0%")


if __name__ == "__main__":
    main()
