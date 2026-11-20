"""
train.py — Stage 3: the first REAL training loop for my-LLM.
=============================================================

WHAT IT DOES, IN ORDER
    1. loads a LOCAL text corpus (default  data/tiny_corpus.txt) and
       refuses politely if it is missing or too short,
    2. builds our character tokenizer from that exact text,
    3. splits the encoded corpus CHRONOLOGICALLY (no shuffling!) into a
       training part and a LATER validation part,
    4. trains TinyLM on next-character prediction: every step draws small
       windows of characters, and the targets are the SAME windows shifted
       by one position — so the model is literally rewarded for guessing
       the true next character,
    5. reports training AND validation loss periodically. Evaluation runs
       in model.eval() inside torch.no_grad(), and the model's previous
       train/eval mode is restored afterwards, so training untouched,
    6. saves RESUMABLE checkpoints (weights + optimizer state + step
       number + model config + tokenizer metadata) into checkpoints/,
       with dated snapshots along the way,
    7. appends a loss-history CSV into runs/ for later plotting,
    8. prints elapsed time, steps, characters processed and an approximate
       characters/second rate, then a short sample of what was learned.

WHY THE SPLIT IS CHRONOLOGICAL
    A model trained on the same text it is tested on can score well by
    pure memorization. Holding out the END of the corpus keeps the
    validation pages unseen during training: if the validation loss stops
    dropping (or rises) while training loss keeps dropping, the model is
    memorizing instead of generalizing. That signal is the whole point of
    having a validation loss.

TYPICAL COMMANDS (run from the project root; `python` = 3.12.10 here)
    python src\\train.py --benchmark
    python src\\train.py --steps 48
    python src\\train.py --resume checkpoints\\model.pt --steps 100

SAFETY RULES THIS SCRIPT OBEYS
    * No downloads, no network, no hosted APIs, no package installs.
    * Writes ONLY inside this project folder (checkpoints/ and runs/).
    * Ctrl+C is caught and the latest usable state is saved before exit.
    * Seeds are set for reproducibility; bit-exact reproducibility across
      machines or thread counts is NOT guaranteed (normal for float math).
    * A checkpoint is never silently applied to a different vocabulary or
      architecture: mismatches stop the run with a clear error instead.
"""

import argparse
import csv
import dataclasses
import math
import os
import sys
import time

import torch

from model import TinyConfig, TinyLM
from tokenizer import CharacterTokenizer, SPECIAL_TOKENS

# --- project-relative paths (everything stays inside the project) ------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer.json")
HISTORY_PATH = os.path.join(RUNS_DIR, "train_history.csv")

CHECKPOINT_FORMAT = "my-LLM checkpoint v1"
HISTORY_HEADER = ["step", "split", "loss", "elapsed_s", "tokens_seen"]


def fail(message):
    """Print a clear error and stop with a non-zero exit code."""
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def resolve_path(path):
    """Interpret user paths relative to the PROJECT ROOT (not the current
    directory), so the documented commands work from anywhere inside the project."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def rel(path):
    """Absolute path -> short, friendly path relative to the project root."""
    return os.path.relpath(path, PROJECT_ROOT)


# ----------------------------------------------------------------------
# command line + validation
# ----------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tiny CPU-only training loop for the from-scratch my-LLM model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data/tiny_corpus.txt",
                        help="path to the training text file (relative to the project root)")
    parser.add_argument("--steps", type=int, default=48,
                        help="number of optimizer steps for this run")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="sequences processed per step")
    parser.add_argument("--block-size", type=int, default=None,
                        help="context length in characters "
                             "(default: 128 fresh, the checkpoint's value when resuming)")
    parser.add_argument("--lr", type=float, default=None,
                        help="AdamW learning rate (default: 1e-3 fresh, "
                             "the checkpoint's stored value when resuming)")
    parser.add_argument("--clip", type=float, default=1.0,
                        help="gradient-norm clipping limit (0 disables clipping)")
    parser.add_argument("--eval-interval", type=int, default=8,
                        help="evaluate train/val loss every N steps")
    parser.add_argument("--eval-batches", type=int, default=5,
                        help="batches averaged per loss evaluation")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="fraction of the corpus held out for validation (0 disables)")
    parser.add_argument("--save-interval", type=int, default=16,
                        help="save a dated snapshot every N steps (and at the last step)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="random seed (deterministic on this machine; not guaranteed "
                             "bit-exact on different hardware)")
    parser.add_argument("--resume", default=None, metavar="PATH",
                        help="path to a checkpoint to resume from, e.g. checkpoints/model.pt")
    parser.add_argument("--prompt", default="The",
                        help="prompt for the short generation sample at the end")
    parser.add_argument("--sample-tokens", type=int, default=200,
                        help="characters to generate for the final sample")

    bench = parser.add_argument_group("benchmark mode")
    bench.add_argument("--benchmark", action="store_true",
                       help="run ONLY a small timed benchmark and exit (writes nothing)")
    bench.add_argument("--bench-steps", type=int, default=5,
                       help="timed optimizer steps in benchmark mode (after 1 warm-up)")

    args = parser.parse_args(argv)

    # Validate BEFORE touching files or models.
    if args.steps < 1:
        fail("--steps must be at least 1")
    if args.batch_size < 1:
        fail("--batch-size must be at least 1")
    if args.batch_size > 32:
        print(f"NOTE: batch-size {args.batch_size} is heavy for a 2-core CPU; "
              f"8 is the recommended setting.")
    if args.block_size is not None and args.block_size < 8:
        fail("--block-size must be at least 8")
    if args.lr is not None and args.lr <= 0:
        fail("--lr must be positive")
    if args.clip < 0:
        fail("--clip must be 0 or a positive norm limit")
    if not (0.0 <= args.val_frac <= 0.9):
        fail("--val-frac must be between 0 and 0.9")
    if args.eval_interval < 1:
        fail("--eval-interval must be at least 1")
    if args.eval_batches < 1:
        fail("--eval-batches must be at least 1")
    if args.save_interval < 1:
        fail("--save-interval must be at least 1")
    if args.bench_steps < 1:
        fail("--bench-steps must be at least 1")
    if args.sample_tokens < 1:
        fail("--sample-tokens must be at least 1")
    return args


# ----------------------------------------------------------------------
# corpus handling
# ----------------------------------------------------------------------

def load_text(path):
    if not os.path.isfile(path):
        fail(f"Training text not found: {path}\n"
             f"       Create a corpus (we keep ours at data/tiny_corpus.txt) or pass "
             f"--data with the path to your own .txt file.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        fail(f"Could not read {path}: {exc}")
    if not text.strip():
        fail(f"Corpus file is empty: {path}")
    return text


def chronological_split(data, block_size, val_frac):
    """Split a 1-D id tensor in ORDER: first part trains, LAST part validates.
    Returns (train_part, validation_part_or_None). We never shuffle before
    the split, and every batch is drawn strictly inside one part, so no
    input/target pair can cross the boundary (see get_batch)."""
    total = len(data)
    n_val = int(total * val_frac)
    if val_frac > 0 and n_val < block_size + 2:
        fail(
            f"The corpus has {total:,} characters; holding out {n_val} for validation "
            f"is too few to build one full window of block_size={block_size} "
            f"(need at least {block_size + 2}). Options: longer corpus, smaller "
            f"--block-size, or --val-frac 0 to train without validation."
        )
    train_end = total - n_val
    if train_end < block_size + 2:
        fail(
            f"The training part would have only {train_end:,} characters, but "
            f"block_size={block_size} needs at least {block_size + 2}. "
            f"Add more text or use a smaller --block-size."
        )
    train_part = data[:train_end]
    val_part = data[train_end:] if n_val > 0 else None
    return train_part, val_part


def get_batch(data_part, batch_size, block_size):
    """Random input/target windows from ONE part of the corpus.
    targets = inputs shifted by one: position i must predict character i+1.
    The farthest index touched is i+block_size <= len-2, so a batch can
    NEVER cross into the other split's territory."""
    upper = len(data_part) - block_size - 1
    ix = torch.randint(0, upper, (batch_size,))
    x = torch.stack([data_part[i : i + block_size] for i in ix])
    y = torch.stack([data_part[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data_part, batch_size, block_size, n_batches):
    """Average loss over n_batches of windows. Runs in eval mode (dropout
    off) and RESTORES the previous train/eval mode afterwards, so callers
    can evaluate at any point without changing loop behavior."""
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


# ----------------------------------------------------------------------
# output helpers: loss history (CSV) and checkpoints
# ----------------------------------------------------------------------

class HistoryWriter:
    """Appends one row per evaluation to runs/train_history.csv.
    Existing files are appended to (resume-friendly); the header is only
    written for brand-new empty files."""

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
        self._writer.writerow([step, split, f"{loss:.4f}", f"{elapsed_s:.1f}", tokens_seen])
        self._file.flush()

    def close(self):
        self._file.close()


def save_checkpoint(path, model, optimizer, config, tokenizer,
                    step, tokens_seen, elapsed_s, seed, lr):
    """One self-contained, resumable snapshot. Includes optimizer.state_dict()
    (AdamW keeps internal per-parameter moments) and the tokenizer metadata,
    so resuming can verify the vocabulary instead of guessing."""
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
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


def open_checkpoint(path):
    """Load a checkpoint, or stop with a clear error."""
    if not os.path.isfile(path):
        fail(f"Checkpoint not found: {path}\n"
             f"       Pass the exact path printed by the earlier training run.")
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:  # corrupt file, older format, foreign content...
        fail(f"Could not load checkpoint {path}: {exc}")


def sample_text(model, tokenizer, prompt, n_tokens):
    """Small end-of-run demonstration: continue a prompt for n characters."""
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = tokenizer.encode(" ")
    if not ids:
        ids = [1]  # <unk> as an absolute last resort
    ids = ids[-model.config.block_size:]  # crop like generate() does
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=n_tokens, temperature=1.0)
    return tokenizer.decode(out[0].tolist())


# ----------------------------------------------------------------------
# main mode: training
# ----------------------------------------------------------------------

def run_training(args):
    torch.manual_seed(args.seed)  # set BEFORE anything random happens
    t_start = time.perf_counter()

    corpus_path = resolve_path(args.data)
    text = load_text(corpus_path)
    print(f"Corpus: {rel(corpus_path)}  ({len(text):,} characters)")
    print(f"Seed {args.seed}: deterministic on this machine in practice; "
          f"bit-exact reproducibility on other hardware/thread counts is NOT guaranteed.")

    # --- 1) build or resume model / optimizer / tokenizer --------------
    if args.resume:
        resume_path = resolve_path(args.resume)
        ckpt = open_checkpoint(resume_path)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            fail(f"'{resume_path}' is not a {CHECKPOINT_FORMAT} file "
                 f"(found format={ckpt.get('format')!r}); refusing unknown checkpoints.")
        meta = ckpt.get("tokenizer")
        if not meta or "characters" not in meta:
            fail("Checkpoint has no tokenizer metadata; the vocabulary would be "
                 "unknown, so it is unsafe to resume.")
        tok = CharacterTokenizer(meta["characters"])

        # Vocabulary lock: a different charset would silently change what
        # every ID means. Refuse instead of corrupting the model.
        corpus_chars = sorted(set(text))
        if corpus_chars != tok.characters:
            fail(
                "The corpus characters do not match the checkpoint vocabulary.\n"
                f"  corpus has    {len(corpus_chars)} distinct characters\n"
                f"  checkpoint has {len(tok.characters)} distinct characters\n"
                "A resume must use the SAME corpus the checkpoint was trained on. "
                "Start a fresh run (no --resume) or pass the original --data file."
            )
        saved_cfg = dict(ckpt.get("model_config") or {})
        if args.block_size is not None and args.block_size != saved_cfg.get("block_size"):
            fail(
                f"This checkpoint was built with block_size={saved_cfg.get('block_size')}, "
                f"but you passed --block-size {args.block_size}. Changing the context "
                "length silently changes the architecture. Omit --block-size to keep "
                "the checkpoint's choice, or start a fresh run."
            )
        config = TinyConfig(**saved_cfg)
        if config.vocab_size != tok.vocab_size:
            fail("Checkpoint model_config.vocab_size disagrees with its own tokenizer "
                 "metadata; this checkpoint is internally inconsistent.")
        model = TinyLM(config)
        try:
            model.load_state_dict(ckpt["model_state"])
        except (RuntimeError, KeyError) as exc:
            fail(f"Checkpoint weights do not match our architecture: {exc}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or 1e-3)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except (KeyError, ValueError, RuntimeError) as exc:
            fail(f"Optimizer state in the checkpoint is unusable: {exc}")
        # Effective learning rate: CLI overrides, otherwise the stored one.
        effective_lr = args.lr if args.lr is not None else float(ckpt.get("lr", 1e-3))
        if effective_lr <= 0:
            fail("Learning rate must be positive (check the checkpoint's stored lr "
                 "or pass --lr).")
        optimizer.param_groups[0]["lr"] = effective_lr
        start_step = int(ckpt.get("step", 0))
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        prior_elapsed = float(ckpt.get("elapsed_s", 0.0))
        print(f"Resumed from {rel(resume_path)} at step {start_step:,} "
              f"({'CLI' if args.lr is not None else 'checkpoint'} lr = {effective_lr:g})")
    else:
        tok = CharacterTokenizer.from_text(text)
        config = TinyConfig(
            vocab_size=tok.vocab_size,
            block_size=args.block_size if args.block_size is not None else 128,
        )
        model = TinyLM(config)
        effective_lr = args.lr if args.lr is not None else 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr)
        start_step = 0
        tokens_seen = 0
        prior_elapsed = 0.0
        tok.save(TOKENIZER_PATH)
        print(f"New run. Tokenizer saved to {rel(TOKENIZER_PATH)} "
              f"(vocab {tok.vocab_size} characters).")

    print(f"Model: {model.num_params():,} parameters | context {config.block_size} | "
          f"one step processes {args.batch_size * config.block_size:,} characters "
          f"| lr {effective_lr:g}")

    # --- 2) encode + chronological split --------------------------------
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_part, val_part = chronological_split(data, config.block_size, args.val_frac)
    if val_part is None:
        print(f"Split: {len(train_part):,} training characters | "
              f"validation DISABLED (--val-frac 0)")
    else:
        print(f"Split: {len(train_part):,} training | {len(val_part):,} validation "
              f"characters (chronological; no shuffling before the split)")

    # --- 3) the loop ------------------------------------------------------
    history = HistoryWriter(HISTORY_PATH)
    print(f"Loss history is appended to {rel(HISTORY_PATH)}")
    print(f"Training {args.steps} steps (eval every {args.eval_interval}, "
          f"snapshots every {args.save_interval})...")

    last_step = start_step + args.steps
    report_clock = time.perf_counter()
    report_tokens = tokens_seen
    snapshot_steps = set()
    interrupted = False
    completed = 0

    def do_eval(step, tokens_now):
        """Both losses, CSV rows, one print line. Speed shown is characters
        processed since the previous evaluation."""
        nonlocal report_clock, report_tokens
        now = time.perf_counter()
        elapsed_total = prior_elapsed + (now - t_start)
        window_tokens = tokens_now - report_tokens
        speed_part = ""
        if window_tokens > 0:
            speed_part = f" | ~{window_tokens / max(now - report_clock, 1e-6):,.0f} chars/s"

        train_loss = evaluate(model, train_part, args.batch_size,
                              config.block_size, args.eval_batches)
        history.add(step, "train", train_loss, elapsed_total, tokens_now)
        line = f"  step {step:>6} | train loss {train_loss:.4f}"
        if val_part is not None:
            val_loss = evaluate(model, val_part, args.batch_size,
                                config.block_size, args.eval_batches)
            history.add(step, "val", val_loss, elapsed_total, tokens_now)
            line += f" | val loss {val_loss:.4f}"
        print(line + speed_part)
        report_clock = now
        report_tokens = tokens_now

    try:
        model.train()
        for step in range(start_step + 1, last_step + 1):
            # Evaluate at the first new step (baseline), at fixed intervals,
            # and at the last step.
            if step == start_step + 1 or (step - start_step) % args.eval_interval == 0 \
                    or step == last_step:
                do_eval(step, tokens_seen)

            x, y = get_batch(train_part, args.batch_size, config.block_size)
            _, loss = model(x, y)          # forward: predict next character

            optimizer.zero_grad(set_to_none=True)
            loss.backward()                # backward: blame assignment
            if args.clip > 0:              # keep gradients from exploding
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()               # small nudge of every parameter

            tokens_seen += args.batch_size * config.block_size  # training chars only
            completed += 1

            if step % args.save_interval == 0 or step == last_step:
                elapsed_total = prior_elapsed + (time.perf_counter() - t_start)
                snapshot_path = os.path.join(CHECKPOINT_DIR, f"step_{step:06d}.pt")
                save_checkpoint(snapshot_path, model, optimizer, config, tok,
                                step, tokens_seen, elapsed_total, args.seed, effective_lr)
                save_checkpoint(LATEST_CHECKPOINT, model, optimizer, config, tok,
                                step, tokens_seen, elapsed_total, args.seed, effective_lr)
                snapshot_steps.add(step)
                print(f"  saved: {rel(snapshot_path)} + rolling {rel(LATEST_CHECKPOINT)}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted (Ctrl+C). Saving the last usable checkpoint before exit...")

    # --- 4) summary, final checkpoint, sample -----------------------------
    elapsed_total = prior_elapsed + (time.perf_counter() - t_start)
    final_step = start_step + completed

    if completed == 0:
        print("No optimizer steps completed, so nothing new was saved.")
    else:
        status = " (interrupted early)" if interrupted else ""
        print(f"\nDone. Steps completed this run: {completed}/{args.steps}{status}")
        print(f"Time: {elapsed_total:.1f} s overall "
              f"({time.perf_counter() - t_start:.1f} s this run) | "
              f"~{tokens_seen / max(elapsed_total, 1e-6):,.0f} chars/s overall")
        # The rolling checkpoint is ALWAYS refreshed to the final state,
        # keeping it the single "resume here" pointer.
        save_checkpoint(LATEST_CHECKPOINT, model, optimizer, config, tok,
                        final_step, tokens_seen, elapsed_total, args.seed, effective_lr)
        print(f"Resumable checkpoint: {rel(LATEST_CHECKPOINT)} "
              f"(step {final_step:,}; weights + optimizer + config + tokenizer)")
        if final_step not in snapshot_steps:
            snap = os.path.join(CHECKPOINT_DIR, f"step_{final_step:06d}.pt")
            save_checkpoint(snap, model, optimizer, config, tok,
                            final_step, tokens_seen, elapsed_total, args.seed, effective_lr)
            print(f"Final snapshot: {rel(snap)}")

        # Short sample. Honest note: a model that saw only a few thousand
        # toy characters mostly parrots them or says nonsense.
        model.eval()
        sample = sample_text(model, tok, args.prompt, args.sample_tokens)
        print(f"\nSample from the trained model (prompt: {args.prompt!r}):")
        print("  +" + "-" * 66 + "+")
        for i in range(0, len(sample), 64):
            print("  | " + sample[i:i + 64].replace("\n", " ").ljust(64) + " |")
        print("  +" + "-" * 66 + "+")
        print("  Expect repeated fragments of our own toy corpus or nonsense;")
        print("  broad language knowledge needs a much larger corpus (Stage 5+).")

    history.close()
    if interrupted:
        print(f"Safely stopped at step {final_step:,}; "
              f"resume with --resume {rel(LATEST_CHECKPOINT)}")
        sys.exit(130)


# ----------------------------------------------------------------------
# benchmark mode
# ----------------------------------------------------------------------

def run_benchmark(args):
    """A bounded speed probe: 1 warm-up + a few timed steps, then report.
    Writes NOTHING (no checkpoints, no history) and never trains long."""
    torch.manual_seed(args.seed)
    corpus_path = resolve_path(args.data)
    text = load_text(corpus_path)
    tok = CharacterTokenizer.from_text(text)
    block_size = args.block_size if args.block_size is not None else 128
    config = TinyConfig(vocab_size=tok.vocab_size, block_size=block_size)
    model = TinyLM(config)
    model.train()
    lr = args.lr if args.lr is not None else 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_part, _ = chronological_split(data, block_size, args.val_frac)

    tokens_per_step = args.batch_size * block_size
    print(f"Benchmark — writes nothing, trains a handful of steps only")
    print(f"corpus {len(text):,} chars | vocab {tok.vocab_size} | "
          f"model {model.num_params():,} params")
    print(f"batch {args.batch_size} x block {block_size} = {tokens_per_step:,} "
          f"chars/step | torch threads: {torch.get_num_threads()} | device: cpu")

    # Warm-up (untimed): first-touch allocations (memory, kernels) would
    # otherwise make step 1 look slower than steady state.
    x, y = get_batch(train_part, args.batch_size, block_size)
    _, warm_loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    warm_loss.backward()
    if args.clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    expected = math.log(tok.vocab_size)
    print(f"random-start loss: {warm_loss.item():.4f} "
          f"(math predicts ~ln(vocab) = {expected:.4f})")

    n = args.bench_steps
    t0 = time.perf_counter()
    for _ in range(n):
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch(train_part, args.batch_size, block_size)
        _, loss = model(x, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    dt = time.perf_counter() - t0

    ms = dt / n * 1000.0
    print(f"\nMeasured over {n} steps (after 1 warm-up step):")
    print(f"  {ms:.0f} ms per forward+backward+update step")
    print(f"  ~{tokens_per_step * n / dt:,.0f} characters/second at these settings")
    print("Honesty note: this is a tiny probe, NOT a reliable ETA for long runs;")
    print("real training adds evaluation, checkpoint I/O and startup overhead,")
    print("and speed varies run to run on a 2-core CPU.")


def main():
    args = parse_args()
    if args.benchmark:
        run_benchmark(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
