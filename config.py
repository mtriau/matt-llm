import sys
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Vocabulary and sequence length
    vocab_size:   int   = 32768   # Mistral tokenizer vocab size
    max_seq_len:  int   = 1024

    # Model dimensions (~200M params with weight tying)
    d_model:  int   = 1024
    n_heads:  int   = 16
    d_ff:     int   = 4096
    n_layers: int   = 12

    # Regularization
    dropout: float = 0.1

    # Architecture flags
    weight_tying: bool = True
    use_swiglu:   bool = True   # SwiGLU FFN (LLaMA-style)
    use_rope:     bool = True   # Rotary Position Embeddings instead of learned pos embeddings

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def d_ff_inner(self) -> int:
        if self.use_swiglu:
            d = int(self.d_ff * 2 / 3)
            return (d + 63) // 64 * 64
        return self.d_ff

    def count_params(self) -> int:
        embed     = self.vocab_size * self.d_model
        pos       = 0 if self.use_rope else self.max_seq_len * self.d_model
        ffn       = (3 * self.d_model * self.d_ff_inner if self.use_swiglu
                     else 2 * self.d_model * self.d_ff + self.d_ff + self.d_model)
        per_layer = (4 * self.d_model ** 2          # Q K V O projections
                     + 4 * self.d_model             # biases
                     + ffn
                     + 4 * self.d_model)            # two LayerNorms
        ln_f      = 2 * self.d_model
        lm_head   = 0 if self.weight_tying else self.vocab_size * self.d_model
        return embed + pos + self.n_layers * per_layer + ln_f + lm_head

    def __repr__(self) -> str:
        n         = self.count_params()
        ffn_label = (f"SwiGLU(d_ff_inner={self.d_ff_inner})" if self.use_swiglu
                     else f"GELU(d_ff={self.d_ff})")
        pos_label = "RoPE" if self.use_rope else f"LearnedPos(max={self.max_seq_len})"
        return (
            f"ModelConfig(\n"
            f"  vocab_size={self.vocab_size}, max_seq_len={self.max_seq_len}\n"
            f"  d_model={self.d_model}, n_heads={self.n_heads}, d_head={self.d_head}\n"
            f"  {ffn_label}, n_layers={self.n_layers}\n"
            f"  pos={pos_label}, dropout={self.dropout}, weight_tying={self.weight_tying}\n"
            f"  ~{n/1e6:.1f}M parameters\n"
            f")"
        )


@dataclass
class TrainConfig:
    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------
    # Defaults to a high-quality reasoning-dense mix:
    #   Cosmopedia (openstax + auto_math_text + khanacademy) + OpenWebMath
    sources:   list = None
    data_dir:  str  = "data"
    data_seed: int  = 0      # >0 = read from random offset into each cache file (different slice each seed)

    def __post_init__(self):
        if self.sources is None:
            self.sources = [
                {"name": "HuggingFaceTB/cosmopedia", "config": "openstax",        "weight": 2.0},
                {"name": "HuggingFaceTB/cosmopedia", "config": "auto_math_text",  "weight": 2.0},
                {"name": "HuggingFaceTB/cosmopedia", "config": "khanacademy",     "weight": 1.0},
                {"name": "open-web-math/open-web-math", "config": "",             "weight": 1.0},
            ]

    # ---------------------------------------------------------------------------
    # Batch / sequence
    # ---------------------------------------------------------------------------
    batch_size:       int = 8
    grad_accum_steps: int = 8
    seq_len:          int = 1024

    # ---------------------------------------------------------------------------
    # Optimiser
    # ---------------------------------------------------------------------------
    lr:           float = 3e-4
    min_lr:       float = 3e-5
    weight_decay: float = 0.1
    beta1:        float = 0.9
    beta2:        float = 0.95
    grad_clip:    float = 1.0

    # ---------------------------------------------------------------------------
    # Schedule
    # ---------------------------------------------------------------------------
    warmup_steps: int = 2000
    max_steps:    int = 100_000
    phase_start:  int = 0       # set to checkpoint step when starting a new phase

    # ---------------------------------------------------------------------------
    # Knowledge distillation
    # Uses GPT-2-XL (1.5B) as teacher — same GPT-2 tokenizer (50257 vocab),
    # so logit distributions are directly comparable with no vocab alignment needed.
    # Set teacher_model="" to disable distillation and train with CE loss only.
    # ---------------------------------------------------------------------------
    teacher_model: str   = "mistralai/Mistral-7B-v0.1"
    distill_alpha: float = 0.5    # weight for hard CE loss; (1-alpha) for soft distill loss
    distill_temp:  float = 2.0    # temperature for softening teacher/student distributions

    # ---------------------------------------------------------------------------
    # Logging / checkpointing
    # ---------------------------------------------------------------------------
    log_every:      int = 10
    eval_every:     int = 500
    save_every:     int = 1000
    checkpoint_dir: str = "checkpoints"

    # ---------------------------------------------------------------------------
    # Hardware
    # ---------------------------------------------------------------------------
    dtype:      str  = "bfloat16"
    compile:    bool = sys.platform != "win32"
    device:     str  = "cuda"
    auto_batch: bool = False
