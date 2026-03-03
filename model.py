"""
GPT-style decoder-only transformer (~200M parameters).

Improvements over v1:
  - Rotary Position Embeddings (RoPE) replace learned positional embeddings.
    RoPE encodes position directly into Q/K vectors via rotation, giving better
    length generalisation and used in every modern small model (LLaMA, Mistral, Phi).

Architecture (pre-norm):
  Token embedding -> n_layers x TransformerBlock -> LayerNorm -> LM head

Each TransformerBlock:
  x = x + Attention(LayerNorm(x))   # RoPE applied inside attention to Q and K
  x = x + SwiGLU(LayerNorm(x))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope(d_head: int, max_seq_len: int, base: float = 10000.0):
    """
    Precompute cos and sin tables for RoPE.

    For each position m and each pair of head dimensions (2i, 2i+1):
        theta_i = base^(-2i / d_head)
        angle   = m * theta_i

    Returns cos, sin each of shape (max_seq_len, d_head // 2).
    Stored as model buffers so they move to the right device automatically.
    """
    theta     = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs     = torch.outer(positions, theta)   # (max_seq_len, d_head//2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to a tensor of shape (B, n_heads, T, d_head).

    Rotates each adjacent pair of dimensions (2i, 2i+1) by the angle for
    that position:
        x'[2i]   = x[2i]   * cos - x[2i+1] * sin
        x'[2i+1] = x[2i+1] * cos + x[2i]   * sin

    cos, sin: (T, d_head//2) — broadcast over B and n_heads automatically.
    """
    x_even = x[..., ::2]    # (B, n_heads, T, d_head//2)
    x_odd  = x[..., 1::2]
    # Stack rotated pairs then flatten last two dims back to d_head
    rotated = torch.stack(
        [x_even * cos - x_odd * sin,
         x_even * sin + x_odd * cos],
        dim=-1,
    )                        # (B, n_heads, T, d_head//2, 2)
    return rotated.flatten(-2)  # (B, n_heads, T, d_head)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.d_head   = cfg.d_head
        self.d_model  = cfg.d_model
        self.use_rope = cfg.use_rope

        self.qkv_proj   = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj    = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop   = nn.Dropout(cfg.dropout)
        self.resid_drop  = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        qkv      = self.qkv_proj(x)
        q, k, v  = qkv.split(self.d_model, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Apply RoPE to Q and K (not V)
        if self.use_rope and rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )                                                    # (B, n_heads, T, d_head)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class MLP(nn.Module):
    """SwiGLU or GELU feed-forward block (same as v1)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.use_swiglu = cfg.use_swiglu
        d = cfg.d_ff_inner
        if cfg.use_swiglu:
            self.gate = nn.Linear(cfg.d_model, d, bias=False)
            self.up   = nn.Linear(cfg.d_model, d, bias=False)
            self.down = nn.Linear(d, cfg.d_model, bias=False)
        else:
            self.fc1 = nn.Linear(cfg.d_model, d)
            self.fc2 = nn.Linear(d, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))
        return self.drop(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp  = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # RoPE: no learned positional embedding table — position is encoded
        # mathematically inside attention via cos/sin buffers.
        # Learned pos embeddings kept as fallback if use_rope=False.
        if cfg.use_rope:
            cos, sin = precompute_rope(cfg.d_head, cfg.max_seq_len)
            self.register_buffer("rope_cos", cos)   # (max_seq_len, d_head//2)
            self.register_buffer("rope_sin", sin)
        else:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.drop   = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        scale = (2 * self.cfg.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "fc2.weight", "mlp.down.weight")):
                p.data.mul_(scale)

    def forward(
        self,
        idx: torch.Tensor,              # (B, T)
        targets: torch.Tensor = None,   # (B, T)
    ):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len

        x = self.tok_emb(idx)           # (B, T, d_model)

        if self.cfg.use_rope:
            rope_cos = self.rope_cos[:T]    # (T, d_head//2)
            rope_sin = self.rope_sin[:T]
        else:
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
            x   = x + self.pos_emb(pos)
            rope_cos = rope_sin = None

        x = self.drop(x)

        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)

        x      = self.ln_f(x)
        logits = self.lm_head(x)        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = sorted_logits.softmax(-1).cumsum(-1)
                sorted_logits[cumprobs - sorted_logits.softmax(-1) > top_p] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs    = logits.softmax(-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)

        return idx
