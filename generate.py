"""
Interactive text generation from a trained checkpoint.

Usage:
    py generate.py --checkpoint checkpoints/step_0010000.pt --prompt "The universe"
    py generate.py --checkpoint checkpoints/step_0010000.pt  # interactive REPL
"""

import argparse
import torch

from config import ModelConfig
from model import GPT
from dataset import get_tokenizer


def load_model(checkpoint_path: str, device: torch.device) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config (saved alongside weights)
    cfg = ckpt.get("model_cfg", ModelConfig())
    model = GPT(cfg).to(device)

    # Strip torch.compile prefix if present
    state = ckpt["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def generate(
    model: GPT,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 200,
    top_p: float = None,
    device: torch.device = torch.device("cuda"),
    stop_at_eos: bool = False,
) -> str:
    enc = get_tokenizer()
    tokens = enc.encode_ordinary(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    eos_id = enc._tok.eos_token_id if stop_at_eos else None

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_id,
        )

    new_tokens = out[0, len(tokens):].tolist()
    # Strip EOS token from output if present
    if eos_id is not None and new_tokens and new_tokens[-1] == eos_id:
        new_tokens = new_tokens[:-1]
    return enc._tok.decode(new_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--prompt",          default=None,
                        help="Prompt text. Omit for interactive REPL.")
    parser.add_argument("--max_new_tokens",  type=int,   default=200)
    parser.add_argument("--temperature",     type=float, default=0.8)
    parser.add_argument("--top_k",           type=int,   default=200)
    parser.add_argument("--top_p",           type=float, default=None)
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--sft",             action="store_true",
                        help="Wrap prompts in SFT chat format (User: ... Assistant:)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print(f"Model loaded ({model.count_parameters()/1e6:.1f}M params)")
    if args.sft:
        print("SFT mode: prompts will be wrapped as 'User: {prompt}\\nAssistant:'")
    print()

    def _run(prompt):
        display_prompt = prompt
        if args.sft:
            prompt = f"User: {prompt}\nAssistant:"
        print(f"\n--- Prompt ---\n{display_prompt}\n--- Completion ---")
        completion = generate(
            model, prompt,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
            device         = device,
            stop_at_eos    = args.sft,
        )
        print(completion)
        print("---\n")

    if args.prompt:
        _run(args.prompt)
    else:
        print("Interactive mode (Ctrl+C to quit).")
        while True:
            try:
                prompt = input("Prompt> ").strip()
                if prompt:
                    _run(prompt)
            except KeyboardInterrupt:
                print("\nBye.")
                break


if __name__ == "__main__":
    main()
