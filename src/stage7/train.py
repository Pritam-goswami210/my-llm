"""
train.py — Stage 7 training pipeline (BPE + TinyLM).
====================================================

Separate training script for Stage 7. Does NOT replace src/train.py.
Does NOT start training when imported; training only runs when invoked
as a script with --steps > 0.

Trains a from-scratch TinyLM on BPE-tokenized Shakespeare text using:
- next-token prediction with shifted targets
- causal attention (no future leakage)
- cross-entropy loss
- PyTorch autograd + AdamW
- gradient clipping
- train/validation loss reporting
- CPU-only execution

All outputs are written under artifacts/stage7/ only.
No existing checkpoints or data files are touched.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import sys
import time

import torch

# Make src/stage7 modules importable when run as a script
STAGE7_DIR = os.path.dirname(os.path.abspath(__file__))
if STAGE7_DIR not in sys.path:
    sys.path.insert(0, STAGE7_DIR)

from byte_bpe_tokenizer import ByteBPETokenizer
from model import Stage7Config, Stage7LM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(STAGE7_DIR)))
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "stage7")

CHECKPOINT_FORMAT = "my-LLM stage-7 checkpoint v1"


def fail(message):
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_text(path):
    if not os.path.isfile(path):
        fail(f"Data file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        fail(f"Could not read {path}: {exc}")


def encode_corpus(tokenizer, text):
    """Encode full text to a flat list of BPE token IDs."""
    return tokenizer.encode(text)


def chronological_split(token_ids, val_frac, block_size):
    """Split token IDs chronologically (first = train, last = validation).

    No shuffling. Every batch is drawn strictly inside one split so no
    input/target pair crosses the boundary.
    """
    total = len(token_ids)
    n_val = int(total * val_frac)
    if val_frac > 0 and n_val < block_size + 2:
        fail(
            f"Not enough tokens for validation: {n_val} val tokens, "
            f"need at least {block_size + 2}. Use --val-frac 0 or more data."
        )
    train_end = total - n_val
    if train_end < block_size + 2:
        fail(
            f"Training part would have only {train_end} tokens, "
            f"need at least {block_size + 2}."
        )
    return token_ids[:train_end], token_ids[train_end:] if n_val > 0 else None


def get_batch(data_part, batch_size, block_size):
    """Random input/target windows: inputs=t[0:T-1], targets=t[1:T]."""
    upper = len(data_part) - block_size - 1
    if upper < 1:
        fail(f"Data part too small for block_size={block_size}: {len(data_part)} tokens")
    ix = torch.randint(0, upper, (batch_size,))
    x = torch.tensor([data_part[i : i + block_size] for i in ix], dtype=torch.long)
    y = torch.tensor([data_part[i + 1 : i + block_size + 1] for i in ix], dtype=torch.long)
    return x, y


@torch.no_grad()
def evaluate(model, data_part, batch_size, block_size, n_batches):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data_part, batch_size, block_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


class HistoryWriter:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(["step", "split", "loss", "elapsed_s", "tokens_seen"])
            self._file.flush()

    def add(self, step, split, loss, elapsed_s, tokens_seen):
        self._writer.writerow([step, split, f"{loss:.4f}", f"{elapsed_s:.1f}", tokens_seen])
        self._file.flush()

    def close(self):
        self._file.close()


def save_checkpoint(path, model, optimizer, config, tokenizer_meta,
                    step, tokens_seen, elapsed_s, seed, lr):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "elapsed_s": float(elapsed_s),
        "seed": int(seed),
        "lr": float(lr),
        "model_config": dataclasses.asdict(config),
        "tokenizer": tokenizer_meta,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 7 training pipeline: BPE + TinyLM (CPU-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="Path to training text (e.g. data/stage7/train.txt)")
    parser.add_argument("--val", default=None,
                        help="Path to validation text (e.g. data/stage7/val.txt)")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to a pre-trained BPE tokenizer JSON. "
                             "If omitted, a new tokenizer is trained on --data.")
    parser.add_argument("--vocab-size", type=int, default=1024,
                        help="BPE target vocabulary size (min 256)")
    parser.add_argument("--steps", type=int, default=0,
                        help="Number of optimizer steps. 0 = tokenizer prep only.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Sequences per step")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Context length in tokens")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="AdamW learning rate")
    parser.add_argument("--clip", type=float, default=1.0,
                        help="Gradient norm clipping limit (0 disables)")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate train/val loss every N steps")
    parser.add_argument("--eval-batches", type=int, default=5,
                        help="Batches averaged per loss evaluation")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Validation fraction (used only if --val not given)")
    parser.add_argument("--save-interval", type=int, default=20,
                        help="Save checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed")
    parser.add_argument("--train-tokenizer", action="store_true",
                        help="Train a new BPE tokenizer on --data and save it")
    parser.add_argument("--device", default="cpu",
                        help="Device (only 'cpu' supported)")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if args.device.lower() != "cpu":
        fail(f"Unsupported device '{args.device}'; Stage 7 runs CPU-only.")

    # ---------------------------------------------------------------
    # 1. Tokenizer
    # ---------------------------------------------------------------
    tokenizer_path = args.tokenizer
    if args.train_tokenizer or tokenizer_path is None:
        if tokenizer_path is None:
            tokenizer_path = os.path.join(ARTIFACT_DIR, "tokenizer.json")

        if args.train_tokenizer:
            print(f"Training BPE tokenizer on {rel(args.data)} ...")
            text = load_text(args.data)
            tok = ByteBPETokenizer(vocab_size=args.vocab_size)
            tok.train([text])  # Train only on training text — never validation
            tok.save(tokenizer_path)
            print(f"Tokenizer saved: {rel(tokenizer_path)} "
                  f"(vocab {tok.vocab_size}, merges {len(tok.merges)})")
        else:
            fail(
                f"No tokenizer found at {tokenizer_path} and --train-tokenizer not set.\n"
                f"       Run with --train-tokenizer first, or pass --tokenizer."
            )
    else:
        if not os.path.isfile(tokenizer_path):
            fail(f"Tokenizer file not found: {tokenizer_path}")
        tok = ByteBPETokenizer.load(tokenizer_path)
        print(f"Tokenizer loaded: {rel(tokenizer_path)} (vocab {tok.vocab_size})")

    tokenizer_meta = {
        "format": "my-LLM byte BPE tokenizer v1",
        "vocab_size": tok.vocab_size,
        "special_tokens": tok.special_tokens,
        "next_id": tok.next_id,
        "merges_count": len(tok.merges),
        "path": rel(tokenizer_path),
    }

    # ---------------------------------------------------------------
    # 2. Data
    # ---------------------------------------------------------------
    train_text = load_text(args.data)
    if args.val:
        val_text = load_text(args.val)
    else:
        # Simple character-level split if no separate val file
        total = len(train_text)
        n_val = int(total * args.val_frac)
        train_text, val_text = train_text[:-n_val], train_text[-n_val:] if n_val > 0 else ""

    train_ids = encode_corpus(tok, train_text)
    val_ids = encode_corpus(tok, val_text) if val_text else None

    print(f"Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):, if val_ids else 0}")
    print(f"Model vocab: {tok.vocab_size} | block_size: {args.block_size}")

    if args.steps < 1:
        print("Steps = 0: tokenizer preparation only. No model training.")
        return

    # ---------------------------------------------------------------
    # 3. Model
    # ---------------------------------------------------------------
    config = Stage7Config(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
    )
    model = Stage7LM(config)
    print(f"Model: {model.num_params():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    tokens_seen = 0
    prior_elapsed = 0.0

    # ---------------------------------------------------------------
    # 4. Training loop
    # ---------------------------------------------------------------
    history_path = os.path.join(ARTIFACT_DIR, "training_history.csv")
    history = HistoryWriter(history_path)

    t_start = time.perf_counter()
    last_step = start_step + args.steps
    report_clock = time.perf_counter()
    report_tokens = tokens_seen
    completed = 0

    print(f"Training {args.steps} steps (eval every {args.eval_interval}, "
          f"checkpoint every {args.save_interval})...")

    try:
        model.train()
        for step in range(start_step + 1, last_step + 1):
            if step == start_step + 1 or (step - start_step) % args.eval_interval == 0 \
                    or step == last_step:
                now = time.perf_counter()
                elapsed_total = prior_elapsed + (now - t_start)
                window_tokens = tokens_seen - report_tokens
                speed = ""
                if window_tokens > 0:
                    speed = f" | ~{window_tokens / max(now - report_clock, 1e-6):,.0f} tok/s"

                train_loss = evaluate(model, train_ids, args.batch_size,
                                      args.block_size, args.eval_batches)
                history.add(step, "train", train_loss, elapsed_total, tokens_seen)
                line = f"  step {step:>6} | train loss {train_loss:.4f}"
                if val_ids is not None:
                    val_loss = evaluate(model, val_ids, args.batch_size,
                                        args.block_size, args.eval_batches)
                    history.add(step, "val", val_loss, elapsed_total, tokens_seen)
                    line += f" | val loss {val_loss:.4f}"
                print(line + speed)
                report_clock = now
                report_tokens = tokens_seen

            x, y = get_batch(train_ids, args.batch_size, args.block_size)
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            tokens_seen += args.batch_size * args.block_size
            completed += 1

            if step % args.save_interval == 0 or step == last_step:
                elapsed_total = prior_elapsed + (time.perf_counter() - t_start)
                ckpt_path = os.path.join(ARTIFACT_DIR, f"model_step_{step:06d}.pt")
                save_checkpoint(
                    ckpt_path, model, optimizer, config, tokenizer_meta,
                    step, tokens_seen, elapsed_total, args.seed, args.lr
                )
                latest_path = os.path.join(ARTIFACT_DIR, "model.pt")
                save_checkpoint(
                    latest_path, model, optimizer, config, tokenizer_meta,
                    step, tokens_seen, elapsed_total, args.seed, args.lr
                )
                print(f"  saved: {rel(ckpt_path)} + {rel(latest_path)}")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint before exit...")

    # ---------------------------------------------------------------
    # 5. Summary
    # ---------------------------------------------------------------
    elapsed_total = prior_elapsed + (time.perf_counter() - t_start)
    final_step = start_step + completed

    if completed == 0:
        print("No optimizer steps completed.")
    else:
        print(f"\nDone. Steps: {completed}/{args.steps}")
        print(f"Time: {elapsed_total:.1f}s | ~{tokens_seen / max(elapsed_total, 1e-6):,.0f} tok/s")
        latest_path = os.path.join(ARTIFACT_DIR, "model.pt")
        save_checkpoint(
            latest_path, model, optimizer, config, tokenizer_meta,
            final_step, tokens_seen, elapsed_total, args.seed, args.lr
        )
        print(f"Final checkpoint: {rel(latest_path)}")

    history.close()


if __name__ == "__main__":
    main()
