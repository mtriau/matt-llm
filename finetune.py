"""
Supervised Fine-Tuning (SFT) script.

Loads a pre-trained checkpoint and fine-tunes on instruction-following data.
Loss is computed ONLY on assistant response tokens — prompt tokens are masked.

Usage:
    py finetune.py                                          # default: dolly-15k, latest checkpoint
    py finetune.py --checkpoint checkpoints/step_0080000.pt
    py finetune.py --dataset tatsu-lab/alpaca
    py finetune.py --epochs 2 --lr 3e-5

Supported datasets (auto-detected schema):
    databricks/databricks-dolly-15k   (instruction, context, response)
    tatsu-lab/alpaca                   (instruction, input, output)
    Any HF dataset with prompt/completion columns
"""

import os
import math
import time
import random
import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import ModelConfig
from model import GPT
from dataset import get_tokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    # Checkpoint to start from (empty = latest in checkpoint_dir)
    checkpoint_path: str = ""
    checkpoint_dir: str = "checkpoints"
    sft_checkpoint_dir: str = "checkpoints_sft"

    # Dataset
    dataset: str = "databricks/databricks-dolly-15k"
    dataset_config: str = ""
    val_fraction: float = 0.05      # fraction held out for validation

    # Sequence
    max_seq_len: int = 1024

    # Optimiser
    lr: float = 5e-5
    min_lr: float = 5e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Batch
    batch_size: int = 4
    grad_accum_steps: int = 4

    # Schedule
    epochs: int = 3
    warmup_steps: int = 100

    # Logging / checkpointing
    log_every: int = 10
    eval_every: int = 200
    save_every: int = 500
    keep_checkpoints: int = 3

    # Hardware
    dtype: str = "bfloat16"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _format_example(ex: dict) -> tuple[str, str]:
    """
    Return (prompt, response) for one example.
    The prompt ends just before the response text so we know where to apply the mask.
    Handles multiple common dataset schemas.
    """
    # databricks/databricks-dolly-15k: instruction / context / response
    if "instruction" in ex and "response" in ex:
        instruction = ex["instruction"].strip()
        context     = ex.get("context", "").strip()
        body        = f"{instruction}\n{context}" if context else instruction
        return f"User: {body}\n\nAssistant: ", ex["response"].strip()

    # tatsu-lab/alpaca: instruction / input / output
    if "instruction" in ex and "output" in ex:
        instruction = ex["instruction"].strip()
        inp         = ex.get("input", "").strip()
        body        = f"{instruction}\n{inp}" if inp else instruction
        return f"User: {body}\n\nAssistant: ", ex["output"].strip()

    # generic prompt / completion
    if "prompt" in ex and "completion" in ex:
        return ex["prompt"].strip(), ex["completion"].strip()

    raise ValueError(f"Unrecognised dataset schema. Keys: {list(ex.keys())}")


class SFTDataset(Dataset):
    """
    Pre-tokenises all examples.

    Labels for prompt tokens are set to -100 so F.cross_entropy ignores them.
    Loss is computed only on the assistant response tokens.
    Examples where the response was entirely truncated are dropped.
    """

    def __init__(self, examples: list, max_seq_len: int):
        enc = get_tokenizer()
        self.items = []

        for ex in examples:
            try:
                prompt, response = _format_example(ex)
            except ValueError:
                continue

            full_text  = prompt + response
            prompt_ids = enc.encode_ordinary(prompt)
            full_ids   = enc.encode_ordinary(full_text)

            # Truncate so input_ids has at most max_seq_len tokens
            full_ids = full_ids[: max_seq_len + 1]

            input_ids = full_ids[:-1]
            labels    = full_ids[1:]

            # Mask prompt positions — shift by 1 for next-token prediction
            n_prompt_toks = len(prompt_ids) - 1
            labels = [-100] * n_prompt_toks + labels[n_prompt_toks:]

            # Drop examples where truncation removed the entire response
            if all(l == -100 for l in labels):
                continue

            self.items.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def sft_collate(batch):
    """Pad to the longest sequence in the batch; labels padded with -100."""
    max_len = max(len(x) for x, _ in batch)
    xs, ys  = [], []
    for x, y in batch:
        pad = max_len - len(x)
        xs.append(x + [0]    * pad)
        ys.append(y + [-100] * pad)
    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    if not os.path.isdir(checkpoint_dir):
        return ""
    files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".pt"))
    return os.path.join(checkpoint_dir, files[-1]) if files else ""


def get_lr(step: int, max_steps: int, warmup_steps: int,
           lr: float, min_lr: float) -> float:
    """Linear warmup → cosine decay."""
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr - min_lr)


def save_checkpoint(model, optimiser, step: int, val_loss: float, cfg: SFTConfig):
    os.makedirs(cfg.sft_checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.sft_checkpoint_dir, f"sft_step_{step:06d}.pt")
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "val_loss":  val_loss,
    }, path)
    print(f"  [checkpoint → {path}]")

    files = sorted(f for f in os.listdir(cfg.sft_checkpoint_dir) if f.endswith(".pt"))
    for old in files[:-cfg.keep_checkpoints]:
        os.remove(os.path.join(cfg.sft_checkpoint_dir, old))
        print(f"  [deleted: {old}]")


@torch.no_grad()
def estimate_val_loss(model, val_loader, device, dtype) -> float:
    model.eval()
    losses = []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(cfg: SFTConfig):
    device = torch.device(cfg.device)
    dtype  = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }[cfg.dtype]

    # ----- Load pre-trained checkpoint -----
    ckpt_path = cfg.checkpoint_path or find_latest_checkpoint(cfg.checkpoint_dir)
    if not ckpt_path:
        raise FileNotFoundError(
            f"No checkpoint found in '{cfg.checkpoint_dir}'. "
            "Run train.py first, or pass --checkpoint <path>."
        )
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]

    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    pre_loss = ckpt.get("val_loss", float("nan"))
    print(f"  pre-train step={ckpt['step']}  val_loss={pre_loss:.4f}")

    # ----- Dataset -----
    print(f"\nLoading dataset: {cfg.dataset}")
    from datasets import load_dataset
    kwargs = {"name": cfg.dataset_config} if cfg.dataset_config else {}
    ds = load_dataset(cfg.dataset, **kwargs)

    split    = "train" if "train" in ds else list(ds.keys())[0]
    examples = list(ds[split])

    random.seed(42)
    random.shuffle(examples)
    n_val    = max(1, int(len(examples) * cfg.val_fraction))
    val_ex   = examples[:n_val]
    train_ex = examples[n_val:]
    print(f"  {len(train_ex)} train / {len(val_ex)} val examples")

    print("Tokenising ...")
    train_ds = SFTDataset(train_ex, cfg.max_seq_len)
    val_ds   = SFTDataset(val_ex,   cfg.max_seq_len)
    print(f"  {len(train_ds)} train / {len(val_ds)} val sequences")

    common = dict(collate_fn=sft_collate, pin_memory=True, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **common)

    # ----- Optimiser -----
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)

    optimiser = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr    = cfg.lr,
        betas = (cfg.beta1, cfg.beta2),
        fused = True,
    )

    # ----- Schedule -----
    batches_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    opt_steps_per_epoch = math.ceil(batches_per_epoch / cfg.grad_accum_steps)
    total_opt_steps = opt_steps_per_epoch * cfg.epochs
    print(f"\n  {batches_per_epoch} batches/epoch × {cfg.epochs} epochs "
          f"= {total_opt_steps} optimiser steps\n")

    # ----- Training -----
    model.train()
    global_step = 0
    loss_accum  = 0.0
    log_steps   = 0
    t0          = time.time()

    for epoch in range(cfg.epochs):
        print(f"=== Epoch {epoch + 1}/{cfg.epochs} ===")
        micro_step = 0
        optimiser.zero_grad(set_to_none=True)

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward — don't pass y to model; compute loss with ignore_index ourselves
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100,
                )
            loss = loss / cfg.grad_accum_steps
            loss_accum += loss.item()
            loss.backward()
            micro_step += 1

            if micro_step % cfg.grad_accum_steps != 0:
                continue

            # Optimiser step
            lr = get_lr(global_step, total_opt_steps, cfg.warmup_steps,
                        cfg.lr, cfg.min_lr)
            for group in optimiser.param_groups:
                group["lr"] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            log_steps   += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                t1       = time.time()
                avg_loss = loss_accum / log_steps
                print(f"  step {global_step:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {t1 - t0:.1f}s")
                loss_accum = 0.0
                log_steps  = 0
                t0         = t1

            if global_step % cfg.eval_every == 0:
                val_loss = estimate_val_loss(model, val_loader, device, dtype)
                print(f"  [val loss: {val_loss:.4f}]")

            if global_step % cfg.save_every == 0:
                val_loss = estimate_val_loss(model, val_loader, device, dtype)
                save_checkpoint(model, optimiser, global_step, val_loss, cfg)

        # End-of-epoch checkpoint
        val_loss = estimate_val_loss(model, val_loader, device, dtype)
        print(f"\nEpoch {epoch + 1} done — val loss: {val_loss:.4f}")
        save_checkpoint(model, optimiser, global_step, val_loss, cfg)

    print("\nSFT complete.")
    print(f"Checkpoints saved to: {cfg.sft_checkpoint_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SFT fine-tuning")
    p.add_argument("--checkpoint",         default="",
                   help="Path to pre-trained .pt (default: latest in checkpoint_dir)")
    p.add_argument("--checkpoint_dir",     default="checkpoints")
    p.add_argument("--sft_checkpoint_dir", default="checkpoints_sft")
    p.add_argument("--dataset",            default="databricks/databricks-dolly-15k")
    p.add_argument("--dataset_config",     default="")
    p.add_argument("--val_fraction",       type=float, default=0.05)
    p.add_argument("--max_seq_len",        type=int,   default=1024)
    p.add_argument("--lr",                 type=float, default=5e-5)
    p.add_argument("--batch_size",         type=int,   default=4)
    p.add_argument("--grad_accum_steps",   type=int,   default=4)
    p.add_argument("--epochs",             type=int,   default=3)
    p.add_argument("--dtype",              default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = SFTConfig(
        checkpoint_path     = args.checkpoint,
        checkpoint_dir      = args.checkpoint_dir,
        sft_checkpoint_dir  = args.sft_checkpoint_dir,
        dataset             = args.dataset,
        dataset_config      = args.dataset_config,
        val_fraction        = args.val_fraction,
        max_seq_len         = args.max_seq_len,
        lr                  = args.lr,
        batch_size          = args.batch_size,
        grad_accum_steps    = args.grad_accum_steps,
        epochs              = args.epochs,
        dtype               = args.dtype,
    )
    finetune(cfg)
