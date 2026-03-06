"""
Training script — v2.

New vs v1:
  - RoPE positional embeddings (baked into model)
  - Knowledge distillation from GPT-2-XL (1.5B teacher, same GPT-2 tokenizer)
    Loss = alpha * CE(student, hard_targets) + (1-alpha) * KL(teacher_soft || student_soft)
    The soft targets carry richer signal than one-hot labels — the teacher's full
    probability distribution tells the student *how wrong* each wrong answer is.
  - High-quality default data: Cosmopedia + OpenWebMath
  - phase_start for clean multi-phase LR schedules

Usage:
    py train.py                                     # distillation on, Cosmopedia+OWM
    py train.py --no_distill                        # CE loss only (faster, less VRAM)
    py train.py --sources wikitext:wikitext-103-raw-v1:1   # custom data
    py train.py --phase_start 62580 --max_steps 142580     # continue from checkpoint

Teacher: mistralai/Mistral-7B-v0.1 (7B params, same 32768-token vocab as student)
  Downloaded automatically on first run (~14GB BF16). Uses ~14GB VRAM for inference.
  Disable with --no_distill if VRAM is tight or you want faster iteration.
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.nn.functional as F

from config import ModelConfig, TrainConfig
from model import GPT
from dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Knowledge distillation helpers
# ---------------------------------------------------------------------------

def load_teacher(cfg: TrainConfig, device: torch.device, dtype: torch.dtype):
    """Load and freeze the teacher model. Returns None if distillation disabled."""
    if not cfg.teacher_model:
        return None

    print(f"Loading teacher: {cfg.teacher_model}")
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model,
        torch_dtype=dtype,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    n = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher loaded: {n/1e6:.0f}M params  ({n*2/1e9:.1f} GB BF16)")
    return teacher


def distill_loss(student_logits: torch.Tensor,
                 teacher_logits: torch.Tensor,
                 temperature: float) -> torch.Tensor:
    """
    KL divergence between teacher and student distributions at given temperature.

    Returns per-token KL (averaged over batch AND sequence length) scaled by T^2
    so it's on the same scale as CE loss (~2-5 range).
    """
    T            = temperature
    # Truncate student logits to teacher vocab size if they differ
    min_vocab     = min(student_logits.size(-1), teacher_logits.size(-1))
    log_p_student = F.log_softmax(student_logits[..., :min_vocab].float() / T, dim=-1)
    p_teacher     = F.softmax(teacher_logits[..., :min_vocab].float()  / T, dim=-1)
    # "batchmean" divides by batch but not sequence length — divide manually
    B, T_seq      = student_logits.shape[:2]
    kl            = F.kl_div(log_p_student, p_teacher, reduction="batchmean") / T_seq
    return kl * T * T


# ---------------------------------------------------------------------------
# Batch size probe
# ---------------------------------------------------------------------------

def find_max_batch_size(model, optimiser, dtype, seq_len, vocab_size,
                        device, headroom=0.90) -> int:
    print("Probing max batch size (this takes ~30 s) ...")
    total_vram = torch.cuda.get_device_properties(device).total_memory
    best       = 1
    candidates = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 512]

    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            x = torch.randint(0, vocab_size, (bs, seq_len), device=device)
            y = torch.randint(0, vocab_size, (bs, seq_len), device=device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(x, y)
            loss.backward()
            used = torch.cuda.memory_allocated()
            optimiser.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if used / total_vram > headroom:
                print(f"  batch_size={bs:4d} → {used/1e9:.1f}/{total_vram/1e9:.1f} GB (over {headroom:.0%} — stopping)")
                break
            print(f"  batch_size={bs:4d} → {used/1e9:.1f}/{total_vram/1e9:.1f} GB  OK")
            best = bs
        except Exception as e:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            print(f"  batch_size={bs:4d} → OOM ({type(e).__name__})")
            break

    print(f"Using batch_size={best}")
    return best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup → cosine decay, relative to phase_start."""
    s           = step - cfg.phase_start
    phase_steps = cfg.max_steps - cfg.phase_start
    if s < cfg.warmup_steps:
        return cfg.lr * s / max(1, cfg.warmup_steps)
    if s >= phase_steps:
        return cfg.min_lr
    progress = (s - cfg.warmup_steps) / (phase_steps - cfg.warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_val_loss(model, val_loader, device, dtype, n_batches=20):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, optimiser, step, val_loss, cfg: TrainConfig, keep=3):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"step_{step:07d}.pt")
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "val_loss":  val_loss,
        "model_cfg": model.cfg,
        "train_cfg": cfg,
    }, path)
    print(f"  [checkpoint → {path}]")

    # Save best checkpoint separately (never rotated out)
    best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
    prev_best = None
    if os.path.exists(best_path):
        prev_best = torch.load(best_path, map_location="cpu", weights_only=False).get("val_loss")
    if prev_best is None or val_loss < prev_best:
        torch.save({
            "step":      step,
            "model":     model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "val_loss":  val_loss,
            "model_cfg": model.cfg,
            "train_cfg": cfg,
        }, best_path)
        print(f"  [new best checkpoint → val_loss {val_loss:.4f}]")

    # Rotate old checkpoints (exclude best.pt)
    files = sorted(f for f in os.listdir(cfg.checkpoint_dir)
                   if f.endswith(".pt") and f != "best.pt")
    for old in files[:-keep]:
        os.remove(os.path.join(cfg.checkpoint_dir, old))
        print(f"  [deleted: {old}]")


def load_latest_checkpoint(model, optimiser, cfg: TrainConfig) -> int:
    if not os.path.isdir(cfg.checkpoint_dir):
        return 0
    files = sorted(f for f in os.listdir(cfg.checkpoint_dir) if f.endswith(".pt"))
    if not files:
        return 0
    path = os.path.join(cfg.checkpoint_dir, files[-1])
    print(f"Resuming from {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimiser.load_state_dict(ckpt["optimiser"])
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(model_cfg: ModelConfig, train_cfg: TrainConfig):
    device = torch.device(train_cfg.device)
    dtype  = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }[train_cfg.dtype]

    # ----- Student model -----
    model = GPT(model_cfg).to(device)
    print(model_cfg)
    print(f"Actual trainable parameters: {model.count_parameters()/1e6:.2f}M")

    # ----- Optimiser -----
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)

    optimiser = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": train_cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr    = train_cfg.lr,
        betas = (train_cfg.beta1, train_cfg.beta2),
        fused = True,
    )

    # ----- Resume -----
    start_step = load_latest_checkpoint(model, optimiser, train_cfg)

    # ----- Auto batch -----
    if train_cfg.auto_batch:
        train_cfg.batch_size = find_max_batch_size(
            model, optimiser, dtype,
            seq_len    = train_cfg.seq_len,
            vocab_size = model_cfg.vocab_size,
            device     = device,
        )
        train_cfg.grad_accum_steps = 1
        scale = (train_cfg.batch_size / 8) ** 0.5
        train_cfg.lr     *= scale
        train_cfg.min_lr *= scale
        for group in optimiser.param_groups:
            group["lr"] = train_cfg.lr
        print(f"  → LR scaled to {train_cfg.lr:.2e} (×{scale:.2f})")

    # ----- Teacher (distillation) -----
    # Loaded AFTER student is on GPU so we can see total VRAM usage clearly.
    teacher = load_teacher(train_cfg, device, dtype)
    if teacher is not None:
        used_gb = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  VRAM after loading teacher: {used_gb:.1f}/{total_gb:.1f} GB")

    # ----- Compile -----
    if train_cfg.compile:
        print("Compiling model with torch.compile() ...")
        model = torch.compile(model)

    # ----- Data -----
    train_loader, val_loader = build_dataloaders(train_cfg)
    train_iter = iter(train_loader)

    def next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    # ----- Training -----
    distill_enabled = teacher is not None
    if distill_enabled:
        print(f"\nDistillation ON  — alpha={train_cfg.distill_alpha}  temp={train_cfg.distill_temp}")
        print(f"  Loss = {train_cfg.distill_alpha:.1f} × CE  +  "
              f"{1-train_cfg.distill_alpha:.1f} × KL(teacher‖student)")
        print(f"  KL warmup: {train_cfg.distill_warmup} steps (starts pure CE, ramps KL to target)\n")
    else:
        print("\nDistillation OFF — training with CE loss only.\n")

    model.train()
    t0             = time.time()
    loss_accum     = 0.0
    ce_accum       = 0.0
    kl_accum       = 0.0
    log_steps      = 0
    best_train_loss = float("inf")   # tracks smoothed training loss for spike detection
    spike_threshold = 2.0            # halt if loss exceeds best × this factor

    for step in range(start_step, train_cfg.max_steps):

        lr = get_lr(step, train_cfg)
        for group in optimiser.param_groups:
            group["lr"] = lr

        optimiser.zero_grad(set_to_none=True)

        for _ in range(train_cfg.grad_accum_steps):
            x, y = next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Teacher forward — no gradient, same autocast
            teacher_logits = None
            if distill_enabled:
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        teacher_logits = teacher(x).logits   # (B, T, 50257)

            # Student forward
            with torch.autocast(device_type="cuda", dtype=dtype):
                student_logits, ce_loss = model(x, y)

                if distill_enabled:
                    kl_loss  = distill_loss(student_logits, teacher_logits,
                                            train_cfg.distill_temp)
                    # Ramp KL weight from 0 to (1-alpha) over distill_warmup steps
                    ds = step - train_cfg.phase_start
                    if ds < train_cfg.distill_warmup:
                        kl_weight = (1 - train_cfg.distill_alpha) * ds / max(1, train_cfg.distill_warmup)
                        ce_weight = 1 - kl_weight
                    else:
                        ce_weight = train_cfg.distill_alpha
                        kl_weight = 1 - train_cfg.distill_alpha
                    loss     = ce_weight * ce_loss + kl_weight * kl_loss
                else:
                    loss     = ce_loss
                    kl_loss  = torch.tensor(0.0)

            loss = loss / train_cfg.grad_accum_steps
            loss_accum += loss.item()
            ce_accum   += ce_loss.item() / train_cfg.grad_accum_steps
            kl_accum   += kl_loss.item() / train_cfg.grad_accum_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimiser.step()
        log_steps += 1

        # ----- Loss spike / NaN detection -----
        current_loss = loss_accum / log_steps if log_steps > 0 else 0.0
        if step > start_step + 100:  # let loss stabilise first
            if math.isnan(current_loss) or math.isinf(current_loss):
                print(f"\n*** NaN/Inf loss detected at step {step}! ***")
                print(f"    Stopping training. Roll back to best.pt or latest checkpoint.")
                return
            if best_train_loss < float("inf") and current_loss > best_train_loss * spike_threshold:
                print(f"\n*** Loss spike detected at step {step}! ***")
                print(f"    Current loss: {current_loss:.4f}  |  Best: {best_train_loss:.4f}  |  Ratio: {current_loss/best_train_loss:.1f}×")
                print(f"    Stopping training. Resume from best.pt:")
                print(f"    py train.py --checkpoint_dir {train_cfg.checkpoint_dir} ...")
                return

        # ----- Logging -----
        if step % train_cfg.log_every == 0:
            t1  = time.time()
            dt  = t1 - t0
            t0  = t1
            tok_per_sec = (
                log_steps * train_cfg.grad_accum_steps
                * train_cfg.batch_size * train_cfg.seq_len / dt
            )
            avg_loss = loss_accum / log_steps
            if distill_enabled:
                avg_ce = ce_accum / log_steps
                avg_kl = kl_accum / log_steps
                ds = step - train_cfg.phase_start
                cur_kl_w = min((1 - train_cfg.distill_alpha) * ds / max(1, train_cfg.distill_warmup),
                               1 - train_cfg.distill_alpha)
                print(
                    f"step {step:7d} | loss {avg_loss:.4f} "
                    f"(ce {avg_ce:.4f}  kl {avg_kl:.4f}  kl_w {cur_kl_w:.2f}) | "
                    f"lr {lr:.2e} | {tok_per_sec/1e3:.1f}K tok/s"
                )
            else:
                print(
                    f"step {step:7d} | loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | {tok_per_sec/1e3:.1f}K tok/s"
                )
            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
            loss_accum = ce_accum = kl_accum = 0.0
            log_steps  = 0

        # ----- Validation -----
        if step % train_cfg.eval_every == 0 and step > 0:
            val_loss = estimate_val_loss(model, val_loader, device, dtype)
            print(f"  [val loss: {val_loss:.4f}]")

        # ----- Checkpointing -----
        if step % train_cfg.save_every == 0 and step > 0 and step > start_step:
            val_loss = estimate_val_loss(model, val_loader, device, dtype)
            save_checkpoint(model, optimiser, step, val_loss, train_cfg)

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a ~200M GPT (v2: RoPE + distillation)")

    # Model
    p.add_argument("--vocab_size",  type=int,   default=32768)
    p.add_argument("--max_seq_len", type=int,   default=1024)
    p.add_argument("--d_model",     type=int,   default=1024)
    p.add_argument("--n_heads",     type=int,   default=16)
    p.add_argument("--d_ff",        type=int,   default=4096)
    p.add_argument("--n_layers",    type=int,   default=12)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--no_rope",     action="store_true",
                   help="Use learned positional embeddings instead of RoPE")

    # Data
    p.add_argument(
        "--sources", nargs="+", default=None, metavar="NAME:CONFIG:WEIGHT",
        help='Dataset sources. Format: "name:config:weight". '
             'Default: Cosmopedia (openstax+auto_math_text+khanacademy) + OpenWebMath'
    )
    p.add_argument("--data_dir", default="data")
    p.add_argument("--data_seed", type=int, default=0,
                   help="Random seed for data slice offset (0=read from start, >0=random offset)")

    # Training
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--grad_accum_steps", type=int,   default=8)
    p.add_argument("--seq_len",          type=int,   default=1024)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--max_steps",        type=int,   default=100_000)
    p.add_argument("--warmup_steps",     type=int,   default=2000)
    p.add_argument("--phase_start",      type=int,   default=0,
                   help="Treat this step as step 0 for the LR schedule")
    p.add_argument("--dtype",            default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--no_compile",       action="store_true")
    p.add_argument("--auto_batch",       action="store_true")
    p.add_argument("--checkpoint_dir",   default="checkpoints")

    # Distillation
    p.add_argument("--no_distill",     action="store_true",
                   help="Disable knowledge distillation (CE loss only)")
    p.add_argument("--teacher_model",  default="mistralai/Mistral-7B-v0.1",
                   help="HuggingFace model ID for the teacher (default: Mistral-7B-v0.1)")
    p.add_argument("--distill_alpha",  type=float, default=0.5,
                   help="Weight for CE loss; (1-alpha) for distillation KL loss")
    p.add_argument("--distill_temp",   type=float, default=2.0,
                   help="Temperature for softening teacher/student distributions")
    p.add_argument("--distill_warmup", type=int, default=2000,
                   help="Steps to ramp KL weight from 0 to (1-alpha). Prevents KL from destabilizing early training.")

    # Resume
    p.add_argument("--resume", action="store_true",
                   help="Resume training from latest checkpoint using saved config. "
                        "Ignores all other args except --checkpoint_dir.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.resume:
        # Load everything from the latest checkpoint
        ckpt_dir = args.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            print(f"No checkpoint directory found: {ckpt_dir}")
            sys.exit(1)
        files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and f != "best.pt")
        if not files:
            print(f"No checkpoints found in {ckpt_dir}")
            sys.exit(1)
        path = os.path.join(ckpt_dir, files[-1])
        print(f"Resuming from {path} (--resume mode)")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_cfg = ckpt["model_cfg"]
        train_cfg = ckpt["train_cfg"]
        # Update phase_start to the checkpoint step so LR schedule continues cleanly
        train_cfg.phase_start = ckpt["step"]
        print(f"  Restored config: step={ckpt['step']}, lr={train_cfg.lr}, "
              f"data_seed={getattr(train_cfg, 'data_seed', 0)}, "
              f"distill={'ON' if train_cfg.teacher_model else 'OFF'}")
    else:
        model_cfg = ModelConfig(
            vocab_size  = args.vocab_size,
            max_seq_len = args.max_seq_len,
            d_model     = args.d_model,
            n_heads     = args.n_heads,
            d_ff        = args.d_ff,
            n_layers    = args.n_layers,
            dropout     = args.dropout,
            use_rope    = not args.no_rope,
        )

        if args.sources:
            parsed_sources = []
            for s in args.sources:
                parts = s.split(":")
                parsed_sources.append({
                    "name":   parts[0],
                    "config": parts[1] if len(parts) > 1 else "",
                    "weight": float(parts[2]) if len(parts) > 2 and parts[2] else 1.0,
                })
        else:
            parsed_sources = None

        train_cfg = TrainConfig(
            sources          = parsed_sources,
            data_dir         = args.data_dir,
            batch_size       = args.batch_size,
            grad_accum_steps = args.grad_accum_steps,
            seq_len          = args.seq_len,
            lr               = args.lr,
            max_steps        = args.max_steps,
            warmup_steps     = args.warmup_steps,
            phase_start      = args.phase_start,
            dtype            = args.dtype,
            compile          = TrainConfig().compile and not args.no_compile,
            auto_batch       = args.auto_batch,
            checkpoint_dir   = args.checkpoint_dir,
            teacher_model    = "" if args.no_distill else args.teacher_model,
            distill_alpha    = args.distill_alpha,
            distill_temp     = args.distill_temp,
            distill_warmup   = args.distill_warmup,
            data_seed        = args.data_seed,
        )

    train(model_cfg, train_cfg)
