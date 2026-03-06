"""
Supervised Fine-Tuning (SFT) script.

Takes a pretrained checkpoint and fine-tunes it on instruction/response pairs
with proper loss masking — loss is computed ONLY on the assistant's response
tokens, not the prompt.

Chat format (using Mistral BOS/EOS tokens):
    <s>User: {instruction}
    Assistant: {response}</s>

Dataset: yahma/alpaca-cleaned (~52K instruction/response pairs).

Usage:
    py sft.py --checkpoint checkpoints/step_0076000.pt
    py sft.py --checkpoint checkpoints/best.pt --epochs 3 --lr 2e-5
    py sft.py --checkpoint checkpoints/best.pt --limit 1000   # quick test
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import ModelConfig
from model import GPT


# ---------------------------------------------------------------------------
# Tokenizer (same as dataset.py — Mistral tokenizer)
# ---------------------------------------------------------------------------

_ENC = None

def get_tokenizer():
    global _ENC
    if _ENC is None:
        from transformers import AutoTokenizer
        _ENC = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1", use_fast=True,
        )
    return _ENC


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100   # PyTorch CE loss ignores this label

def format_example(instruction: str, input_text: str, output: str, tokenizer, max_len: int):
    """
    Format one instruction example and return (input_ids, labels) tensors.

    The prompt portion gets IGNORE_INDEX labels so loss is only on the response.
    """
    if input_text and input_text.strip():
        prompt = f"User: {instruction}\n{input_text}\nAssistant:"
    else:
        prompt = f"User: {instruction}\nAssistant:"

    response = f" {output}"

    # Tokenize separately so we know exact boundary
    prompt_ids   = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    eos_id       = tokenizer.eos_token_id  # 2

    # Full sequence: BOS + prompt + response + EOS
    bos_id    = tokenizer.bos_token_id  # 1
    input_ids = [bos_id] + prompt_ids + response_ids + [eos_id]

    # Labels: IGNORE for BOS + prompt, real ids for response + EOS
    n_prompt = 1 + len(prompt_ids)  # BOS + prompt tokens
    labels   = [IGNORE_INDEX] * n_prompt + response_ids + [eos_id]

    # Truncate to max_len
    input_ids = input_ids[:max_len]
    labels    = labels[:max_len]

    return input_ids, labels


class SFTDataset(Dataset):
    """
    Holds pre-tokenized instruction/response pairs with loss masking labels.
    """

    def __init__(self, examples: list, tokenizer, max_len: int = 512, limit: int = 0):
        self.data = []
        if limit > 0:
            examples = examples[:limit]

        skipped = 0
        for ex in examples:
            input_ids, labels = format_example(
                ex["instruction"], ex.get("input", ""), ex["output"],
                tokenizer, max_len,
            )
            if len(input_ids) < 4:  # skip degenerate examples
                skipped += 1
                continue
            self.data.append((input_ids, labels))

        if skipped:
            print(f"  Skipped {skipped} degenerate examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Pad sequences to the longest in the batch.
    Padding uses 0 for input_ids and IGNORE_INDEX for labels.
    """
    max_len = max(len(ids) for ids, _ in batch)

    input_ids_padded = []
    labels_padded = []

    for ids, labs in batch:
        pad_len = max_len - len(ids)
        input_ids_padded.append(ids + [0] * pad_len)
        labels_padded.append(labs + [IGNORE_INDEX] * pad_len)

    return (
        torch.tensor(input_ids_padded, dtype=torch.long),
        torch.tensor(labels_padded, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_sft_loss(model, input_ids, labels, device, dtype):
    """
    Forward pass with masked loss — only compute CE on non-IGNORE tokens.
    """
    input_ids = input_ids.to(device)
    labels    = labels.to(device)

    with torch.autocast(device_type="cuda", dtype=dtype):
        logits, _ = model(input_ids)  # ignore model's built-in loss
        # Shift: predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
    return loss


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def train_sft(args):
    device = torch.device("cuda")
    dtype  = torch.bfloat16

    # ----- Load checkpoint -----
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    n_params = model.count_parameters()
    print(f"Model loaded ({n_params/1e6:.1f}M params)")

    # ----- Load dataset -----
    print("Loading Alpaca dataset ...")
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    examples = list(ds)
    print(f"  {len(examples):,} examples total")

    # Shuffle deterministically then split 95/5
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(examples))
    cut = int(len(examples) * 0.95)
    train_indices = indices[:cut]
    val_indices   = indices[cut:]

    tokenizer = get_tokenizer()
    max_len = min(args.max_len, model_cfg.max_seq_len)

    train_examples = [examples[i] for i in train_indices]
    val_examples   = [examples[i] for i in val_indices]

    print(f"Tokenizing train set ({len(train_examples):,} examples, max_len={max_len}) ...")
    train_ds = SFTDataset(train_examples, tokenizer, max_len, limit=args.limit)
    print(f"Tokenizing val set ({len(val_examples):,} examples) ...")
    val_ds   = SFTDataset(val_examples, tokenizer, max_len,
                           limit=max(args.limit // 10, 50) if args.limit else 0)

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,} examples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=True, num_workers=0,
    )

    # ----- Optimiser -----
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)

    optimiser = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True,
    )

    # ----- Training -----
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(100, total_steps // 10)
    print(f"\nSFT Training: {args.epochs} epochs, {len(train_loader)} steps/epoch, "
          f"{total_steps} total steps")
    print(f"  LR: {args.lr:.1e} -> {args.lr/10:.1e}, warmup: {warmup_steps} steps")
    print(f"  Batch size: {args.batch_size}, grad_accum: {args.grad_accum}\n")

    model.train()
    step = 0
    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        for batch_idx, (input_ids, labels) in enumerate(train_loader):

            lr = get_lr(step, warmup_steps, total_steps, args.lr, args.lr / 10)
            for group in optimiser.param_groups:
                group["lr"] = lr

            loss = compute_sft_loss(model, input_ids, labels, device, dtype)
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                step += 1

                if step % 10 == 0:
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1
                    print(f"  step {step:5d}/{total_steps} | loss {loss.item() * args.grad_accum:.4f} | "
                          f"lr {lr:.2e} | {dt:.1f}s")

        # ----- End of epoch validation -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vl = compute_sft_loss(model, vx, vy, device, dtype)
                val_losses.append(vl.item())
        val_loss = sum(val_losses) / len(val_losses)
        model.train()
        print(f"\n  Epoch {epoch+1}/{args.epochs} — val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "sft_best.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "model_cfg": model_cfg,
                "val_loss": val_loss,
                "epoch": epoch + 1,
                "step": step,
            }, save_path)
            print(f"  [saved best → {save_path}  val_loss={val_loss:.4f}]\n")
        else:
            print()

    # ----- Save final -----
    final_path = os.path.join(args.output_dir, "sft_final.pt")
    torch.save({
        "model": model.state_dict(),
        "model_cfg": model_cfg,
        "val_loss": val_loss,
        "epoch": args.epochs,
        "step": step,
    }, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")
    print(f"Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SFT fine-tuning for instruction following")
    p.add_argument("--checkpoint", required=True, help="Path to pretrained checkpoint")
    p.add_argument("--output_dir", default="checkpoints_sft")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--grad_accum", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--max_len",    type=int,   default=512,
                   help="Max sequence length for SFT examples")
    p.add_argument("--limit",      type=int,   default=0,
                   help="Limit training examples (0=use all)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sft(args)
