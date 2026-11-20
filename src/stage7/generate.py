"""
generate.py — Stage 7 offline text generation (BPE + TinyLM).
============================================================

Loads a Stage 7 checkpoint and its matching BPE tokenizer, then generates
text one token at a time — completely offline, CPU-only.

Supports:
- user-provided prompt
- configurable generation length
- configurable temperature
- optional top-k sampling
- CPU inference
- clear command-line help

Decoding reconstructs valid UTF-8 text. Token IDs are never confused with
Unicode characters: tokens are decoded to bytes via the BPE vocabulary,
then the bytes are decoded as UTF-8.

All reads/writes stay inside this project folder.
"""

import argparse
import json
import os
import sys

import torch

STAGE7_DIR = os.path.dirname(os.path.abspath(__file__))
if STAGE7_DIR not in sys.path:
    sys.path.insert(0, STAGE7_DIR)

from byte_bpe_tokenizer import ByteBPETokenizer
from model import Stage7LM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(STAGE7_DIR)))
CHECKPOINT_FORMAT = "my-LLM stage-7 checkpoint v1"


def fail(message):
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate text offline with the Stage 7 BPE + TinyLM model "
                    "(CPU-only, no training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", default=None,
                        help="Starting text. Omit for interactive mode.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a Stage 7 checkpoint (.pt). "
                             "Defaults to artifacts/stage7/model.pt")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to a BPE tokenizer JSON. "
                             "Defaults to artifacts/stage7/tokenizer.json")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature: <1 safer/repetitive, >1 wilder")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="How many new tokens to generate")
    parser.add_argument("--top-k", type=int, default=None,
                        help="If set, sample only from the top-k most likely tokens")
    parser.add_argument("--device", default="cpu",
                        help="Device (only 'cpu' supported)")
    return parser.parse_args(argv)


def load_checkpoint_and_tokenizer(checkpoint_path, tokenizer_path):
    if not os.path.isfile(checkpoint_path):
        fail(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"       Train first (python src\\stage7\\train.py --steps N) or pass "
            f"--checkpoint with the exact path."
        )

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        fail(f"Could not load checkpoint {checkpoint_path}: {exc}")

    if ckpt.get("format") != CHECKPOINT_FORMAT:
        fail(
            f"'{checkpoint_path}' is not a {CHECKPOINT_FORMAT} file "
            f"(found format={ckpt.get('format')!r}); refusing unknown checkpoints."
        )

    saved_cfg = ckpt.get("model_config")
    if not saved_cfg:
        fail("Checkpoint has no saved model configuration.")

    # Tokenizer: prefer embedded metadata, fall back to standalone file
    tok_meta = ckpt.get("tokenizer")
    if tok_meta and "merges_count" in tok_meta:
        tok = ByteBPETokenizer(
            vocab_size=int(tok_meta.get("vocab_size", 1024)),
            special_tokens=tok_meta.get("special_tokens", {}),
        )
        # Restore merges
        tok.next_id = int(tok_meta.get("next_id", 256))
        # Rebuild id_to_bytes from special tokens + bytes 0-255
        tok.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for tid in tok.special_tokens.values():
            tok.id_to_bytes[tid] = None
        # We need the full byte table and merges — these aren't all in
        # the checkpoint meta, so we also need the tokenizer file.
        # If a standalone tokenizer file is available, load from it.
        tok_path_used = "checkpoint metadata (partial)"

    if tokenizer_path is None:
        tokenizer_path = os.path.join(
            PROJECT_ROOT, "artifacts", "stage7", "tokenizer.json"
        )

    if not os.path.isfile(tokenizer_path):
        fail(
            f"Tokenizer file not found: {tokenizer_path}\n"
            f"       Pass --tokenizer with the path to your BPE tokenizer JSON."
        )

    try:
        tok = ByteBPETokenizer.load(tokenizer_path)
    except Exception as exc:
        fail(f"Could not load tokenizer {tokenizer_path}: {exc}")

    # Vocabulary compatibility check
    saved_vocab = int(saved_cfg.get("vocab_size", -1))
    if tok.vocab_size != saved_vocab:
        fail(
            f"Vocabulary mismatch: checkpoint vocab_size={saved_vocab}, "
            f"tokenizer vocab_size={tok.vocab_size}. "
            "Use the tokenizer that belongs to this checkpoint."
        )

    config = Stage7Config(**saved_cfg)
    model = Stage7LM(config)

    try:
        model.load_state_dict(ckpt["model_state"])
    except (RuntimeError, KeyError) as exc:
        fail(f"Checkpoint weights do not match our architecture: {exc}")

    model.eval()
    return model, tok, config, int(ckpt.get("step", 0)), rel(tokenizer_path)


@torch.no_grad()
def generate_tokens(model, tokenizer, prompt, max_new_tokens, temperature, top_k):
    """Generate max_new_tokens after prompt. Returns (prompt_text, generated_text)."""
    if not prompt or not prompt.strip():
        fail("Prompt is empty. Provide a non-empty --prompt.")

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        fail("Prompt produced no token IDs. Try different prompt text.")

    prompt_ids = prompt_ids[-model.config.block_size:]
    idx = torch.tensor([prompt_ids], dtype=torch.long)

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    out_ids = out[0].tolist()

    prompt_text = tokenizer.decode(out_ids[:len(prompt_ids)])
    generated_text = tokenizer.decode(out_ids[len(prompt_ids):])
    return prompt_text, generated_text


def interactive_loop(model, tokenizer, max_new_tokens, temperature, top_k):
    print("Interactive generation. Type a prompt and press Enter.")
    print("Type 'quit' or 'exit' to leave. Ctrl+C also stops.")
    print(f"Temperature {temperature} | top-k {top_k} | max new tokens {max_new_tokens}")
    print("-" * 60)
    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if prompt.lower() in ("quit", "exit"):
            print("Goodbye.")
            return
        if not prompt:
            print("(empty prompt — type some text, or 'quit' to leave)\n")
            continue
        try:
            _, generated = generate_tokens(
                model, tokenizer, prompt, max_new_tokens, temperature, top_k
            )
            print(f"generated: {generated}\n")
        except SystemExit:
            raise
        except Exception as exc:
            print(f"generation error: {exc}\n")


def main():
    args = parse_args()

    if args.device.lower() != "cpu":
        fail(f"Unsupported device '{args.device}'; Stage 7 runs CPU-only.")

    if args.temperature is None or args.temperature <= 0:
        fail("--temperature must be a positive number")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        fail("--max-new-tokens must be at least 1")

    default_ckpt = os.path.join(PROJECT_ROOT, "artifacts", "stage7", "model.pt")
    default_tok = os.path.join(PROJECT_ROOT, "artifacts", "stage7", "tokenizer.json")

    checkpoint_path = args.checkpoint or default_ckpt
    tokenizer_path = args.tokenizer or default_tok

    model, tok, config, step, tok_source = load_checkpoint_and_tokenizer(
        checkpoint_path, tokenizer_path
    )

    print(f"Checkpoint: {rel(checkpoint_path)} (step {step}, "
          f"vocab {tok.vocab_size}, context {config.block_size})")
    print(f"Tokenizer: {tok_source}")
    print(f"Device: cpu | temperature {args.temperature} | "
          f"top-k {args.top_k} | max new tokens {args.max_new_tokens}\n")

    if args.prompt is not None:
        _, generated = generate_tokens(
            model, tok, args.prompt, args.max_new_tokens,
            args.temperature, args.top_k
        )
        print(f"prompt:    {args.prompt}")
        print(f"generated: {generated}")
    else:
        interactive_loop(model, tok, args.max_new_tokens, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
