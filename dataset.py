"""
Dataset utilities — supports mixing multiple HuggingFace datasets or .txt files.

Each source is tokenised once and cached separately as a flat numpy uint16 array.
At load time, token arrays are mixed according to per-source weights.

Sources are configured via TrainConfig.sources, a list of dicts:
    [
        {"name": "wikitext", "config": "wikitext-103-raw-v1", "weight": 1.0},
        {"name": "Skylion007/openwebtext", "weight": 2.0},
    ]

Mixing rule: total combined tokens = determined by the most constrained source
(the one with the fewest tokens relative to its weight fraction). This means no
source is ever repeated — each contributes its natural share.

Run with --sources to override:
    py train.py --sources wikitext:wikitext-103-raw-v1:1 Skylion007/openwebtext::2
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import TrainConfig

# ---------------------------------------------------------------------------
# Tokenizer — Mistral-7B tokenizer (32,768 vocab).
# Using the same tokenizer as the teacher means logit distributions are
# directly comparable for knowledge distillation — no vocab alignment needed.
# Wrapped so callers use .encode_ordinary() matching the v1 tiktoken API.
# ---------------------------------------------------------------------------

_ENC = None


class _TokenizerWrapper:
    """Thin wrapper so the HuggingFace tokenizer exposes encode_ordinary() like tiktoken."""
    def __init__(self, hf_tok):
        self._tok = hf_tok

    def encode_ordinary(self, text: str) -> list:
        return self._tok.encode(text, add_special_tokens=False)


def get_tokenizer() -> _TokenizerWrapper:
    global _ENC
    if _ENC is None:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            use_fast=True,
        )
        _ENC = _TokenizerWrapper(hf_tok)
    return _ENC


# ---------------------------------------------------------------------------
# Per-source tokenisation and caching
# ---------------------------------------------------------------------------

def _slug(name: str, config: str = "") -> str:
    """Make a filesystem-safe identifier from dataset name + optional config."""
    key = f"{name}_{config}" if config else name
    return re.sub(r"[^a-zA-Z0-9_-]", "_", key)


def _extract_text(doc) -> str:
    """Pull text from a document dict (or raw string)."""
    if not isinstance(doc, dict):
        return doc
    if "text" in doc:
        return doc["text"]
    if "content" in doc:
        return doc["content"]
    return "\n".join(v for v in doc.values() if isinstance(v, str))


def _tokenize_batch(texts: list) -> bytes:
    """Tokenize a list of texts and return raw uint16 bytes. Runs in worker process."""
    enc = get_tokenizer()
    parts = []
    for text in texts:
        if not text or not text.strip():
            continue
        toks = np.array(enc.encode_ordinary(text), dtype=np.uint16)
        parts.append(toks.tobytes())
    return b"".join(parts)


def _stream_to_file(doc_iter, path: str, n_workers: int = 0) -> int:
    """
    Tokenise documents from an iterator and write directly to a binary file.
    Uses multiprocessing when n_workers > 0 for faster tokenisation.
    Returns total token count written.
    """
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 16)

    # Collect texts in batches, tokenize in parallel, write sequentially
    from multiprocessing import Pool

    BATCH_SIZE = 1000
    n = 0
    batch = []

    with open(path, "wb") as f, Pool(n_workers) as pool:
        def flush(text_batch):
            nonlocal n
            # Split batch across workers
            chunk_size = max(1, len(text_batch) // n_workers)
            chunks = [text_batch[i:i + chunk_size]
                      for i in range(0, len(text_batch), chunk_size)]
            for raw_bytes in pool.map(_tokenize_batch, chunks):
                f.write(raw_bytes)
                n += len(raw_bytes) // 2  # uint16 = 2 bytes

        for doc in doc_iter:
            text = _extract_text(doc)
            batch.append(text)
            if len(batch) >= BATCH_SIZE:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

    return n


def _cache_source(name: str, config: str, data_dir: str) -> dict:
    """
    Ensure a source is tokenised and written to disk.
    Returns {"train": path, "val": path}.
    Skips download/tokenisation if cache files already exist.
    """
    slug = _slug(name, config)
    train_path = os.path.join(data_dir, f"{slug}_train.bin")
    val_path   = os.path.join(data_dir, f"{slug}_val.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        n_train = os.path.getsize(train_path) // 2   # uint16 = 2 bytes each
        n_val   = os.path.getsize(val_path)   // 2
        print(f"  [{slug}] cached — train={n_train:,}  val={n_val:,} tokens")
        return {"train": train_path, "val": val_path}

    os.makedirs(data_dir, exist_ok=True)

    # ---- Plain text file ----
    if name.endswith(".txt") and os.path.isfile(name):
        print(f"  Tokenising {name} ...")
        enc = get_tokenizer()
        with open(name, "r", encoding="utf-8") as f:
            text = f.read()
        all_tok = np.array(enc.encode_ordinary(text), dtype=np.uint16)
        cut = int(len(all_tok) * 0.95)
        all_tok[:cut].tofile(train_path)
        all_tok[cut:].tofile(val_path)

    # ---- HuggingFace dataset ----
    else:
        from datasets import load_dataset
        label = name + (f" ({config})" if config else "")
        print(f"  Downloading {label} ...")
        kwargs = {"name": config} if config else {}
        ds = load_dataset(name, **kwargs)

        if "validation" in ds:
            n_train_docs = len(ds["train"])
            n_val_docs   = len(ds["validation"])
            print(f"    tokenising train ({n_train_docs:,} docs) ...")
            _stream_to_file(ds["train"].shuffle(seed=42), train_path)
            print(f"    tokenising validation ({n_val_docs:,} docs) ...")
            _stream_to_file(ds["validation"], val_path)
        else:
            # No val split — shuffle then carve off last 2% as val
            shuffled    = ds["train"].shuffle(seed=42)
            n_total     = len(shuffled)
            cut         = int(n_total * 0.98)
            print(f"    tokenising train ({cut:,} docs) ...")
            _stream_to_file(shuffled.select(range(cut)),         train_path)
            print(f"    tokenising val   ({n_total - cut:,} docs) ...")
            _stream_to_file(shuffled.select(range(cut, n_total)), val_path)

    n_train = os.path.getsize(train_path) // 2
    n_val   = os.path.getsize(val_path)   // 2
    print(f"  [{slug}] train={n_train:,}  val={n_val:,} tokens")
    return {"train": train_path, "val": val_path}


# ---------------------------------------------------------------------------
# Weighted mixing
# ---------------------------------------------------------------------------

def _read_n(path: str, n: int, offset: int = 0) -> np.ndarray:
    """Read exactly min(n, available) uint16 tokens from a binary cache file,
    starting at the given token offset."""
    file_n = os.path.getsize(path) // 2
    offset = min(offset, max(file_n - n, 0))
    count  = min(n, file_n - offset)
    return np.fromfile(path, dtype=np.uint16, count=count, offset=offset * 2)


def _build_mixed(source_infos: list, split: str, data_seed: int = 0) -> np.ndarray:
    """
    Load and mix token arrays for one split.

    source_infos: list of {"paths": {"train": ..., "val": ...}, "weight": float}
    data_seed:    when > 0, each source reads from a random offset into its cache
                  so different seeds yield different slices of large datasets.

    Mixing rule: the total token count is set by whichever source is most
    constrained (smallest tokens / weight-fraction). No source is ever repeated.
    Other sources contribute proportionally less if they have fewer tokens.
    """
    weights    = [s["weight"] for s in source_infos]
    total_w    = sum(weights)
    fracs      = [w / total_w for w in weights]
    file_sizes = [os.path.getsize(s["paths"][split]) // 2 for s in source_infos]

    # Total tokens = limited by the most-constrained source
    total = int(min(sz / f for sz, f in zip(file_sizes, fracs)))

    # Generate per-source random offsets from the seed
    if data_seed > 0:
        rng = np.random.default_rng(data_seed)
        offsets = [int(rng.integers(0, max(fs - int(total * f), 1)))
                   for fs, f in zip(file_sizes, fracs)]
    else:
        offsets = [0] * len(source_infos)

    parts = []
    for info, frac, file_sz, off in zip(source_infos, fracs, file_sizes, offsets):
        n = int(total * frac)
        if n == 0:
            continue
        arr = _read_n(info["paths"][split], n, offset=off)
        # Tile only if this source has fewer tokens than its allocation
        if len(arr) < n:
            reps = n // len(arr) + 1
            arr = np.tile(arr, reps)[:n]
        parts.append(arr)
        pct = frac * 100
        off_str = f", offset={off:,}" if off > 0 else ""
        print(f"  {info['paths'][split].split(os.sep)[-1].replace('_' + split + '.bin', '')}: "
              f"{n:,} tokens ({pct:.0f}%{off_str})")

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Main entry point for loading data
# ---------------------------------------------------------------------------

def _load_or_create_cache(cfg: TrainConfig) -> dict:
    os.makedirs(cfg.data_dir, exist_ok=True)
    sources = cfg.sources

    print("Loading dataset sources:")
    source_infos = []
    for src in sources:
        name   = src["name"]
        config = src.get("config", "")
        weight = float(src.get("weight", 1.0))
        paths  = _cache_source(name, config, cfg.data_dir)
        source_infos.append({"paths": paths, "weight": weight})

    seed = getattr(cfg, "data_seed", 0)
    if seed > 0:
        print(f"Data seed: {seed} (shuffling read offsets)")

    if len(sources) == 1:
        paths = source_infos[0]["paths"]
        train_tok = np.fromfile(paths["train"], dtype=np.uint16)
        val_tok   = np.fromfile(paths["val"],   dtype=np.uint16)
    else:
        print(f"Mixing {len(sources)} sources ...")
        train_tok = _build_mixed(source_infos, "train", data_seed=seed)
        val_tok   = _build_mixed(source_infos, "val",   data_seed=seed)

    print(f"Total: train={len(train_tok):,}  val={len(val_tok):,} tokens")
    return {"train": train_tok, "val": val_tok}


# ---------------------------------------------------------------------------
# Dataset and DataLoaders
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """
    Samples random windows of (seq_len + 1) tokens.
    Input:  tokens[i : i+seq_len]
    Target: tokens[i+1 : i+seq_len+1]  (next-token prediction)
    """

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens  = tokens
        self.seq_len = seq_len
        self.n = len(tokens) - seq_len - 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def build_dataloaders(cfg: TrainConfig):
    splits = _load_or_create_cache(cfg)

    train_ds = TokenDataset(splits["train"], cfg.seq_len)
    val_ds   = TokenDataset(splits["val"],   cfg.seq_len)

    common = dict(
        batch_size  = cfg.batch_size,
        pin_memory  = True,
        num_workers = 0,   # Windows-safe; increase to 2-4 on Linux
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)

    return train_loader, val_loader
