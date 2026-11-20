"""
generate.py — Stage 4: offline, local text generation with our trained TinyLM.
==============================================================================

WHAT IT DOES
    Loads a saved checkpoint produced by src/train.py (checkpoints/model.pt by
    default) together with its matching character tokenizer, then generates
    new text one character at a time — completely offline, CPU-only.

HOW GENERATION WORKS (plain English)
    The model knows one thing: "given some characters, guess the next one."
    Generation loops:

        1. take the prompt characters we already have,
        2. keep only the last `block_size` of them (the model's context window),
        3. ask the model for scores over every possible next character,
        4. divide the scores by `temperature` and turn them into probabilities,
        5. sample ONE next character at random from those probabilities,
        6. append it and repeat until we have enough new characters.

    `temperature` controls how bold the guesses are:
      - 1.0   = sample exactly according to the model's probabilities
      - < 1.0 = the model plays it safer (peaked probabilities, more repetition)
      - > 1.0 = the model takes wilder risks (flatter probabilities, more variety)

WHAT THE MODEL IS (and is NOT)
    This is a character-level model trained on ~7 kilobytes of toy text. It has
    no understanding of questions, facts, or instructions. It predicts the next
    character in a sequence. Output will look like half-remembered fragments of
    the toy corpus, or outright gibberish — especially right after training.
    That is expected, not a bug.

WHAT THIS SCRIPT DOES NOT DO
    - No training, no gradient updates, no checkpoint modification.
    - No GPU required (CPU-only).
    - No downloads, no network, no hosted APIs.
    - It never overwrites checkpoints, tokenizer files, training history,
      or the corpus.

CLI EXAMPLES (run from the project root; `python` = Python 3.12.10)
    python src\\generate.py --prompt "The" --max-new-chars 30 --temperature 1.0
    python src\\generate.py              # interactive mode: type prompts at the prompt
    python src\\generate.py --checkpoint checkpoints\\step_000016.pt
    python src\\generate.py --temperature 0.7 --max-new-chars 120
"""

import argparse
import os
import sys

import torch

from model import TinyConfig, TinyLM
from tokenizer import CharacterTokenizer

# --- project-relative paths ----------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "model.pt")
DEFAULT_TOKENIZER = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.json")

CHECKPOINT_FORMAT = "my-LLM checkpoint v1"


def fail(message):
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def resolve_path(path):
    """Interpret user paths relative to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate text offline with our own trained TinyLM "
                    "(character-level, CPU-only, no training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", default=None,
                        help="starting text. Omit this to enter interactive mode.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="path to a training checkpoint (.pt)")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="path to a tokenizer file (.json). Defaults to the one "
                             "saved next to the checkpoint during training.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature: <1 = safer/repetitive, >1 = wilder")
    parser.add_argument("--max-new-chars", type=int, default=100,
                        help="how many new characters to generate")
    parser.add_argument("--device", default="cpu",
                        help="device for inference (only 'cpu' is supported)")

    args = parser.parse_args(argv)

    # --- argument validation --------------------------------------------
    if args.temperature is None or args.temperature <= 0:
        fail("--temperature must be a positive number (e.g. 1.0, 0.7, 1.5)")
    if args.max_new_chars < 1:
        fail("--max-new-chars must be at least 1")
    if args.max_new_chars > 5000:
        fail("--max-new-chars must not exceed 5000 (keep generations short and safe)")
    if args.device.lower() != "cpu":
        fail(f"unsupported device '{args.device}'; this project runs CPU-only "
             f"(no CUDA/XPU/DirectML on this hardware). Use --device cpu.")
    return args


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def load_checkpoint_and_tokenizer(checkpoint_path, tokenizer_path):
    """Load checkpoint + tokenizer and verify they are mutually compatible.
    Returns (model, tokenizer, config, step, tok_source). Stops with a clear
    error on any mismatch — never generates with a silently wrong vocabulary."""
    if not os.path.isfile(checkpoint_path):
        fail(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"       Train first (python src\\train.py --steps 48) or pass "
            f"--checkpoint with the exact path to a saved checkpoint."
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
        fail("Checkpoint has no saved model configuration; cannot rebuild the model.")

    # --- tokenizer ------------------------------------------------------
    tok_meta = ckpt.get("tokenizer")
    tok_path_used = None

    if tok_meta and "characters" in tok_meta:
        # Prefer the tokenizer embedded in the checkpoint (guaranteed to match
        # the vocabulary the weights were trained with).
        tok = CharacterTokenizer(tok_meta["characters"])
        tok_path_used = "checkpoint metadata"
    else:
        # Fall back to a standalone tokenizer file if the checkpoint lacks one.
        if not os.path.isfile(tokenizer_path):
            fail(
                "Checkpoint has no embedded tokenizer and no tokenizer file "
                f"was found at {tokenizer_path}. Cannot establish the vocabulary."
            )
        try:
            tok = CharacterTokenizer.load(tokenizer_path)
            tok_path_used = tokenizer_path
        except Exception as exc:
            fail(f"Could not load tokenizer {tokenizer_path}: {exc}")

    # --- vocabulary compatibility --------------------------------------
    saved_vocab = int(saved_cfg.get("vocab_size", -1))
    if tok.vocab_size != saved_vocab:
        fail(
            "Vocabulary mismatch between checkpoint and tokenizer.\n"
            f"  checkpoint model_config.vocab_size = {saved_vocab}\n"
            f"  tokenizer vocab_size                   = {tok.vocab_size}\n"
            "They must be identical. Use the tokenizer that belongs to this checkpoint."
        )

    # Saved character table (if present) must match the tokenizer exactly.
    saved_chars = saved_cfg.get("characters")
    if saved_chars and sorted(saved_chars) != tok.characters:
        fail(
            "Checkpoint character table disagrees with the tokenizer's "
            "character table. Unsafe to generate; vocabularies must match exactly."
        )

    # --- rebuild model with the SAME configuration ----------------------
    config = TinyConfig(**saved_cfg)
    model = TinyLM(config)  # random weights first...

    try:
        model.load_state_dict(ckpt["model_state"])
    except (RuntimeError, KeyError) as exc:
        fail(f"Checkpoint weights do not match our architecture: {exc}")
    # ...now carrying the trained weights. Architecture and dimensions are
    # exactly as saved; nothing was silently changed.

    model.eval()  # inference mode: dropout off
    return model, tok, config, int(ckpt.get("step", 0)), tok_path_used


# ----------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------

@torch.no_grad()
def generate_chars(model, tokenizer, prompt, max_new_chars, temperature):
    """Generate max_new_chars characters after `prompt`.
    Returns (prompt_text, generated_text) as plain strings."""
    if not prompt or not prompt.strip():
        fail("Prompt is empty. Provide a non-empty --prompt.")

    # Encode the prompt; characters outside our vocabulary become  Ucrania.
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        # e.g. a prompt consisting only of characters the tokenizer maps away
        # (decode() skips <pad>). Treat as empty input.
        fail("Prompt produced no token IDs. Try different prompt text.")

    # Crop to the model's context window, exactly like TinyLM.generate() does.
    prompt_ids = prompt_ids[-model.config.block_size:]
    idx = torch.tensor([prompt_ids], dtype=torch.long)

    # model.generate() keeps model.eval(), applies temperature, samples one
    # token per step, and restores the previous train/eval mode afterwards.
    out = model.generate(idx, max_new_tokens=max_new_chars, temperature=temperature)
    out_ids = out[0].tolist()

    prompt_text = tokenizer.decode(out_ids[: len(prompt_ids)])
    generated_text = tokenizer.decode(out_ids[len(prompt_ids):])
    return prompt_text, generated_text


def interactive_loop(model, tokenizer, max_new_chars, temperature):
    """Repeatedly read prompts from the user until they type 'quit'/'exit'."""
    print("Interactive generation. Type a prompt and press Enter.")
    print("Type 'quit' or 'exit' to leave. Ctrl+C also stops.")
    print(f"Temperature {temperature} | up to {max_new_chars} new chars per prompt")
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
            _, generated = generate_chars(model, tokenizer, prompt,
                                          max_new_chars, temperature)
            print(f"generated: {generated}\n")
        except SystemExit:
            raise
        except Exception as exc:  # generation error should not kill the session
            print(f"generation error: {exc}\n")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    tokenizer_path = resolve_path(args.tokenizer)

    model, tok, config, step, tok_source = load_checkpoint_and_tokenizer(
        checkpoint_path, tokenizer_path
    )

    print(f"Checkpoint: {rel(checkpoint_path)} (step {step}, "
          f"vocab {tok.vocab_size}, context {config.block_size})")
    print(f"Tokenizer source: {tok_source}")
    print(f"Device: cpu | temperature: {args.temperature} | "
          f"max new chars: {args.max_new_chars}\n")

    if args.prompt is not None:
        _, generated = generate_chars(model, tok, args.prompt,
                                      args.max_new_chars, args.temperature)
        print(f"prompt:    {args.prompt}")
        print(f"generated: {generated}")
    else:
        interactive_loop(model, tok, args.max_new_chars, args.temperature)


if __name__ == "__main__":
    main()
