"""
finetune.py — Stage 10 Phase F: Supervised Fine-Tuning Pipeline.
================================================================

Trains TinyLM on conversational data using supervised fine-tuning with
assistant-only loss masking. The model learns to generate assistant
responses given user prompts, while user-message tokens are excluded
from the loss computation.

KEY DESIGN:

  * Loads a pretrained checkpoint (Stage 9 model).
  * Loads conversational JSONL data.
  * Tokenizes conversations with role markers (<|user|>, <|assistant|>,
    <|end|>).
  * Applies assistant-only loss masking: only assistant-message tokens
    contribute to the loss; user-message tokens and padding are masked.
  * Handles context-length limits: splits long conversations into
    overlapping windows that preserve conversational context.
  * Saves checkpoints under checkpoints/stage10/ (never overwriting
    original checkpoints).
  * Supports resume from checkpoint.
  * Logs training and validation metrics to runs/stage10/.
  * CPU-only execution.

SAFETY:
  * Original checkpoints are never modified.
  * No network calls, no downloads.
  * All reads/writes stay inside the project folder.
  * Empty or malformed conversations are skipped, not fabricated.

HOW TO RUN
    python src\\stage10\\finetune.py --data <path> --steps <n>

This file is part of the from-scratch my-LLM project. We authored it.
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

# Make src/stage9 importable for the dataset splitter
STAGE9_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "stage9",
)
if STAGE9_DIR not in sys.path:
    sys.path.insert(0, STAGE9_DIR)

from weighted_train import WeightedCorpusSampler

# Make src importable for the model and tokenizer
SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import TinyConfig, TinyLM
from tokenizer import CharacterTokenizer, PAD, UNK, SPECIAL_TOKENS


# --- project-relative paths ----------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

CHECKPOINT_STAGE10_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "stage10")
RUNS_STAGE10_DIR = os.path.join(PROJECT_ROOT, "runs", "stage10")
TOKENIZER_STAGE10_PATH = os.path.join(CHECKPOINT_STAGE10_DIR, "tokenizer.json")

CHECKPOINT_FORMAT = "my-LLM stage-10 checkpoint v1"
HISTORY_HEADER = ["step", "split", "loss", "elapsed_s", "tokens_seen"]

# Special tokens for conversational roles
SPECIAL_CHAT_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]


# --- paths -----------------------------------------------------------------


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


# --- conversation loading --------------------------------------------------

def load_conversations(path):
    """Load conversation records from a JSONL file.

    Each line is a JSON object with:
        {"messages": [{"role": str, "content": str}, ...], ...}

    Returns a list of conversation dicts (each is a list of message dicts).
    """
    if not os.path.isfile(path):
        fail(f"Conversation file not found: {path}")

    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                messages = record.get("messages", [])
                if messages:
                    conversations.append(messages)
            except json.JSONDecodeError as exc:
                print(f"WARNING: Skipping malformed line {line_num}: {exc}")

    print(f"Loaded {len(conversations)} conversations from {rel(path)}")
    return conversations


# --- tokenizer preparation -------------------------------------------------

def prepare_tokenizer(conversations, existing_tokenizer_path=None):
    """Prepare a tokenizer for conversational training.

    If an existing tokenizer file exists, loads it and adds special
    conversation tokens (<|user|>, <|assistant|>, <|end|>) while
    preserving existing token IDs.

    Returns a CharacterTokenizer with special tokens added.
    """
    from tokenizer import CharacterTokenizer

    if existing_tokenizer_path and os.path.isfile(existing_tokenizer_path):
        tok = CharacterTokenizer.load(existing_tokenizer_path)
        print(f"Loaded existing tokenizer: vocab={tok.vocab_size}")
    else:
        # Build a new tokenizer from the conversation text
        all_text = ""
        for conv in conversations:
            for msg in conv:
                all_text += msg.get("content", "") + "\n"
        tok = CharacterTokenizer.from_text(all_text)
        print(f"Built new tokenizer: vocab={tok.vocab_size}")

    # Add special conversation tokens while preserving existing IDs
    new_chars = []
    for token in SPECIAL_CHAT_TOKENS:
        if token not in tok.characters:
            new_chars.append(token)

    if new_chars:
        # Build expanded character list preserving original order
        expanded_characters = list(tok.characters) + new_chars
        all_tokens = list(SPECIAL_TOKENS) + expanded_characters
        stoi = {t: i for i, t in enumerate(all_tokens)}
        itos = {i: t for i, t in enumerate(all_tokens)}

        # Create a new tokenizer with expanded vocabulary
        class ExpandedTokenizer:
            def __init__(self, characters, stoi, itos):
                self.characters = characters
                self.stoi = stoi
                self.itos = itos
                self.pad_id = stoi[SPECIAL_TOKENS[0]]
                self.unk_id = stoi[SPECIAL_TOKENS[1]]

            @property
            def vocab_size(self):
                return len(self.stoi)

            def encode(self, text):
                return [self.stoi.get(ch, self.unk_id) for ch in text]

            def decode(self, ids):
                out = []
                for i in ids:
                    if i == self.pad_id:
                        continue
                    out.append(self.itos.get(int(i), self.unk_char))
                return "".join(out)

        tok = ExpandedTokenizer(expanded_characters, stoi, itos)
        print(f"Expanded tokenizer: vocab={tok.vocab_size} "
              f"(added {len(new_chars)} special tokens: {new_chars})")

    return tok


# --- conversation tokenization ----------------------------------------------

def encode_conversation(tokenizer, stripped, block_size):
    """Encode a conversation into token IDs with role markers.

    Format: <|user|> user text <|end|> <|assistant|> assistant text
    <|end|> ...

    Returns:
        token_ids: List of token IDs.
        loss_mask: List of booleans (True for assistant tokens that
                   should contribute to loss, False otherwise).
    """
    all_ids = []
    all_mask = []

    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Role token
        if role == "assistant":
            role_token = "<|assistant|>"
        else:
            role_token = "<|user|>"

        role_ids = tokenizer.encode(role_token)
        content_ids = tokenizer.encode(content)
        end_token = "<|end|>"
        end_ids = tokenizer.encode(end_token)

        # Add role token (no loss on role tokens)
        all_ids.extend(role_ids)
        all_mask.extend([False] * len(role_ids))

        # Add content (loss only on assistant dialogue)
        all_ids.extend(content_ids)
        if role == "assistant":
            all_mask.extend([True] * len(content_ids))
        else:
            all_mask.extend([False] * len(content_ids))

        # Add end token (no loss on end tokens)
        all_ids.extend(end_ids)
        all_mask.extend([False] * len(end_ids))

    return all_ids, all_mask


def split_conversation_into_windows(token_ids, mask, block_size):
    """Split a tokenized conversation into overlapping windows.

    Each window is at most block_size tokens. Windows overlap to
    preserve conversational context.

    Returns list of (token_ids_window, mask_window) tuples.
    """
    windows = []
    if len(token_ids) <= block_size:
        windows.append((token_ids, mask))
    else:
        stride = block_size // 2
        for start in range(0, len(token_ids), stride):
            end = min(start + block_size, len(token_ids))
            window_ids = token_ids[start:end]
            window_mask = mask[start:end]
            windows.append((window_ids, window_mask))
            if end >= len(token_ids):
                break

    return windows


# --- training --------------------------------------------------------------

class HistoryWriter:
    """Appends per-step loss rows to runs/stage10/train_history.csv."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(HISTORY_HEADER)
            self._file.flush()

    def add(self, step, split, loss, elapsed_s, tokens_seen):
        self._writer.writerow(
            [step, split, f"{loss:.4f}", f"{elapsed_s:.1f}", tokens_seen]
        )
        self._file.flush()

    def close(self):
        self._file.close()


def save_checkpoint(path, model, optimizer, config, tokenizer,
                    step, tokens_seen, elapsed_s, seed, lr,
                    manifest=None):
    """Save a Stage 10 checkpoint with full metadata."""
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "elapsed_s": float(elapsed_s),
        "seed": int(seed),
        "lr": float(lr),
        "model_config": dataclasses.asdict(config),
        "tokenizer": {
            "format": "my-LLM character tokenizer v1",
            "special_tokens": list(SPECIAL_TOKENS),
            "characters": tokenizer.characters,
        },
        "manifest": manifest,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


def open_checkpoint(path):
    if not os.path.isfile(path):
        fail(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        fail(f"Could not load checkpoint {path}: {exc}")


@torch.no_grad()
def compute_loss(model, data_windows, batch_size, block_size, n_batches):
    """Average loss over n_batches from validation windows.

    Uses assistant-only loss masking: only assistant-message token
    positions contribute to the loss.
    """
    import torch.nn.functional as F

    model.eval()
    losses = []
    total = len(data_windows)
    if total == 0:
        return float("nan")
    for _ in range(n_batches):
        ix = torch.randint(0, total, (batch_size,))
        batch_losses = []
        for i in ix:
            i = int(i)
            ids, mask = data_windows[i]
            x = torch.tensor([ids], dtype=torch.long)
            mask_t = torch.tensor([mask], dtype=torch.bool)
            logits, _ = model(x, y=x)  # logits shape: (1, T, V)
            # Compute loss only on assistant positions
            # logits: (1, T, V), targets: (1, T)
            logits_flat = logits.reshape(-1, logits.size(-1))  # (T, V)
            targets_flat = x.reshape(-1)  # (T,)
            mask_flat = mask_t.reshape(-1)  # (T,)
            # Compute cross-entropy only on masked positions
            ce = F.cross_entropy(logits_flat, targets_flat,
                                 reduction="none")  # (T,)
            masked_loss = (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            batch_losses.append(masked_loss.item())
        losses.append(sum(batch_losses) / len(batch_losses))
    if was_training:
        model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def run_training(args):
    """Main training entry point."""
    torch.manual_seed(args.seed)
    t_start = time.perf_counter()

    print("=" * 60)
    print("Stage 10 Phase F — Supervised Fine-Tuning")
    print("=" * 60)

    # Load manifest (for reference)
    manifest_path = os.path.join(PROJECT_ROOT, "data", "stage9",
                                 "corpus_manifest.json")
    manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"Manifest: {rel(manifest_path)}")

    # Load or build tokenizer
    print("\nPreparing tokenizer ...")
    tok = prepare_tokenizer([], existing_tokenizer_path=None)
    print(f"Tokenizer: vocab={tok.vocab_size} characters")

    # Save tokenizer
    tok.save(os.path.join(CHECKPOINT_STAGE10_DIR, "tokenizer.json"))
    print(f"Tokenizer saved to: "
          f"{rel(os.path.join(CHECKPOINT_STAGE10_DIR, 'tokenizer.json'))}")

    # Load conversations
    print("\nLoading conversations ...")
    conversations = load_conversations(args.data)

    if not conversations:
        fail("No conversations loaded. Cannot train.")

    # Encode conversations into windows
    print("\nEncoding conversations ...")
    all_windows = []
    for conv in conversations:
        ids, mask = encode_conversation(tok, conv, args.block_size)
        windows = split_conversation_into_windows(ids, mask,
                                                  args.block_size)
        all_windows.extend(windows)

    print(f"  Created {len(all_windows)} training windows from "
          f"{len(conversations)} conversations")

    if not all_windows:
        fail("No training windows created. Cannot train.")

    # Split into train/val
    n_val = int(len(all_windows) * args.val_frac)
    if args.val_frac > 0 and n_val < 1:
        n_val = 0
    train_end = len(all_windows) - n_val
    train_windows = all_windows[:train_end]
    val_windows = all_windows[train_end:] if n_val > 0 else None

    print(f"  Train windows: {len(train_windows)}")
    print(f"  Validation windows: {len(val_windows) if val_windows else 0}")

    # Build model
    print("\nBuilding model ...")
    config = TinyConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
    )
    model = TinyLM(config)
    print(f"Model: {model.num_params():,} parameters | "
          f"vocab={config.vocab_size} | block={config.block_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    tokens_seen = 0
    prior_elapsed = 0.0

    # Resume support
    if args.resume:
        ckpt = open_checkpoint(args.resume)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            fail(f"'{args.resume}' is not a {CHECKPOINT_FORMAT} file")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = int(ckpt.get("step", 0))
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        prior_elapsed = float(ckpt.get("elapsed_s", 0.0))
        effective_lr = args.lr if args.lr is not None \
            else float(ckpt.get("lr", 1e-3))
        optimizer.param_groups[0]["lr"] = effective_lr
        print(f"Resumed from {rel(args.resume)} at step {start_step:,}")
    else:
        effective_lr = args.lr

    # History logger
    history_path = os.path.join(RUNS_STAGE10_DIR, "train_history.csv")
    history = HistoryWriter(history_path)
    print(f"History: {rel(history_path)}")

    # Training loop
    print(f"\nTraining {args.steps} steps (eval every {args.eval_interval}, "
          f"snapshot every {args.save_interval})...")

    last_step = start_step + args.steps
    report_clock = time.perf_counter()
    report_tokens = tokens_seen
    snapshot_steps = set()
    interrupted = False
    completed = 0

    def do_eval(step, tokens_now):
        nonlocal report_clock, report_tokens
        now = time.perf_counter()
        elapsed_total = prior_elapsed + (now - t_start)
        window_tokens = tokens_now - report_tokens
        speed_part = ""
        if window_tokens > 0:
            speed_part = (f" | ~{window_tokens / max(now - report_clock,"
                          f"1e-6):,.0f} chars/s")

        # Evaluate on validation set
        if val_windows:
            val_loss = compute_loss(model, val_windows, args.batch_size,
                                    args.block_size, args.eval_batches)
            history.add(step, "val", val_loss, elapsed_total, tokens_now)

        # Evaluate on training set
        total_train_loss = 0.0
        n_eval = 0
        step_size = max(1, len(train_windows) // max(1, args.eval_batches))
        for i in range(0, len(train_windows), step_size):
            batch_windows = train_windows[i:i + args.eval_batches]
            batch_losses = []
            for ids, mask in batch_windows:
                x = torch.tensor([ids], dtype=torch.long)
                mask_t = torch.tensor([mask], dtype=torch.bool)
                logits, _ = model(x, y=x)
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = x.reshape(-1)
                mask_flat = mask_t.reshape(-1)
                ce = F.cross_entropy(logits_flat, targets_flat,
                                     reduction="none")
                masked_loss = (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)
                batch_losses.append(masked_loss.item())
            if batch_losses:
                total_train_loss += sum(batch_losses) / len(batch_losses)
                n_eval += 1
        avg_train_loss = (total_train_loss / n_eval
                          if n_eval > 0 else float("nan"))
        history.add(step, "train", avg_train_loss, elapsed_total, tokens_seen)

        line = f"  step {step:>6} | train loss {avg_train_loss:.4f}"
        if val_windows:
            line += f" | val loss {val_loss:.4f}"
        print(line + speed_part)
        report_clock = now
        report_tokens = tokens_now

    try:
        model.train()
        t_start = time.perf_counter()
        for step in range(start_step + 1, last_step + 1):
            if step == start_step + 1 or \
                    (step - start_step) % args.eval_interval == 0 or \
                    step == last_step:
                do_eval(step, tokens_seen)

            # Sample a random window
            ix = torch.randint(0, len(train_windows), (1,))
            ids, mask = all_windows[int(ix.item())]
            x = torch.tensor([ids], dtype=torch.long)
            y = torch.tensor([ids], dtype=torch.long)

            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip)
            optimizer.step()

            tokens_seen += args.batch_size * args.block_size
            completed += 1

            if step % args.save_interval == 0 or step == last_step:
                elapsed_total = prior_elapsed + \
                    (time.perf_counter() - t_start)
                snapshot_path = os.path.join(
                    CHECKPOINT_STAGE10_DIR, f"step_{step:06d}.pt"
                )
                corpora_info = {
                    "conversations": len(conversations),
                    "train_windows": len(train_windows),
                    "val_windows": len(val_windows) if val_windows else 0,
                    "data_file": args.data,
                }
                save_checkpoint(
                    snapshot_path, model, optimizer, config, tok,
                    step, tokens_seen, elapsed_total, args.seed, args.lr,
                    manifest,
                )
                latest_path = os.path.join(CHECKPOINT_STAGE10_DIR,
                                           "model.pt")
                save_checkpoint(
                    latest_path, model, optimizer, config, tokenizer,
                    step, tokens_seen, elapsed_total, args.seed, args.lr,
                    manifest, corpora_info,
                )
                snapshot_steps.add(step)
                print(f"  saved: {rel(snapshot_path)} + {rel(latest_path)}")

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted (Ctrl+C). Saving checkpoint before exit...")

    # Summary
    elapsed_total = prior_elapsed + (time.perf_counter() - t_start)
    final_step = start_step + completed

    if completed == 0:
        print("No optimizer steps completed.")
    else:
        status = " (interrupted early)" if interrupted else ""
        print(f"\nDone. Steps: {completed}/{args.steps}{status}")
        print(f"Time: {elapsed_total:.1f}s | "
              f"~{tokens_seen / max(elapsed_total, 1e-6):,.0f} chars/s")

        latest_path = os.path.join(CHECKPOINT_STAGE10_DIR, "model.pt")
        save_checkpoint(
            latest_path, model, optimizer, config, tokenizer,
            final_step, tokens_seen, elapsed_total, args.seed, args.lr,
            manifest,
            {
                "conversations": len(conversations),
                "train_windows": len(train_windows),
                "val_windows": len(val_windows) if val_windows else 0,
                "data_file": args.data,
            },
        )
        print(f"Checkpoint: {rel(latest_path)} (step {final_step:,})")

        # Generation sample
        model.eval()
        prompt = args.prompt if args.prompt else "The"
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [tokenizer.unk_id]
        ids = ids[-config.block_size:]
        idx = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=args.sample_tokens,
                                 temperature=1.0)
        out_ids = out[0].tolist()
        generated = tokenizer.decode(out_ids[len(ids):])
        print(f"\nSample (prompt: {prompt!r}):")
        print("  +" + "-" * 66 + "+")
        for i in range(0, len(generated), 64):
            print("  | " + generated[i:i + 64].replace("\n", " ").ljust(64) + " |")
        print("  +" + "-" * 66 + "+")

    history.close()
    if interrupted:
        print(f"Stopped at step {final_step:,}. Resume with --resume "
              f"{rel(os.path.join(CHECKPOINT_STAGE10_DIR, 'model.pt'))}")
        sys.exit(130)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 10: Supervised fine-tuning for conversational "
                    "TinyLM (CPU-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="Path to conversation JSONL file.")
    parser.add_argument("--steps", type=int, default=48,
                        help="Number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Sequences per step")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Context length in characters")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="AdamW learning rate")
    parser.add_argument("--clip", type=float, default=1.0,
                        help="Gradient norm clipping limit")
    parser.add_argument("--eval-interval", type=int, default=8,
                        help="Evaluate loss every N steps")
    parser.add_argument("--eval-batches", type=int, default=5,
                        help="Batches per evaluation")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Validation fraction")
    parser.add_argument("--save-interval", type=int, default=16,
                        help="Save snapshot every N steps")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed")
    parser.add_argument("--resume", default=None,
                        help="Path to a Stage 10 checkpoint to resume from")
    parser.add_argument("--prompt", default="The",
                        help="Prompt for generation sample")
    parser.add_argument("--sample-tokens", type=int, default=100,
                        help="Characters to generate for sample")

    args = parser.parse_args(argv)

    if args.steps < 1:
        fail("--steps must be at least 1")
    if args.batch_size < 1:
        fail("--batch-size must be at least 1")
    if args.block_size < 8:
        fail("--block-size must be at least 8")
    if args.lr <= 0:
        fail("--lr must be positive")
    if args.clip < 0:
        fail("--clip must be >= 0")
    if not (0.0 <= args.val_frac <= 0.9):
        fail("--val-frac must be between 0 and 0.9")
    if args.eval_interval < 1:
        fail("--eval-interval must be at least 1")
    if args.save_interval < 1:
        fail("--save-interval must be at least 1")
    return args


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
