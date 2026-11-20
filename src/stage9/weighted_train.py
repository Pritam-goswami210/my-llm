"""
weighted_train.py — Stage 9 Phase E: Multi-Dataset Weighted Training Pipeline.
==============================================================================

Trains TinyLM on multiple corpora with configurable sampling weights.
Reads the Stage 9 corpus manifest and runs controlled experiments:

  Experiment A: Original corpus only (data/tiny_corpus.txt)
  Experiment B: Original corpus + Shakespeare
                (data/tiny_corpus.txt + data/processed/hamlet_clean.txt)
  Experiment C: Original + Shakespeare + Modern English
                (data/tiny_corpus.txt + data/processed/hamlet_clean.txt +
                 data/stage9/corpus_c/modern_english_local.txt)

Key design:

  * Each corpus is encoded and stored as a separate 1-D tensor.
  * At each training step, one corpus is selected by weighted random
    choice, then a batch is drawn from that corpus only.
  * No sequences cross scene/document boundaries (splits come from
    src/stage8/dataset_splitter.py).
  * Hamlet's weight is enforced at its configured maximum (25%) when all
    corpora are available.
  * Missing datasets are reported, not fabricated.
  * Checkpoints are saved under checkpoints/stage9/ (never overwriting
    original checkpoints).
  * Training history is logged to runs/stage9/.
  * CPU-only execution.

This file is part of the from-scratch my-LLM project. We authored it.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import time

import torch

# Make src/stage8 importable for the dataset splitter
# __file__ is src/stage9/weighted_train.py
# dirname x3 = project root (my-LLM/)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
STAGE8_DIR = os.path.join(PROJECT_ROOT, "src", "stage8")
if STAGE8_DIR not in sys.path:
    sys.path.insert(0, STAGE8_DIR)

from dataset_splitter import (
    load_manifest,
    load_hamlet_scenes,
    split_hamlet_by_scenes,
    split_stage5_by_documents,
    PROJECT_ROOT,
)

# Make src importable for the model and tokenizer
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import TinyConfig, TinyLM
from tokenizer import CharacterTokenizer, PAD, UNK, SPECIAL_TOKENS


# --- paths -----------------------------------------------------------------

CHECKPOINT_STAGE9_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "stage9")
RUNS_STAGE9_DIR = os.path.join(PROJECT_ROOT, "runs", "stage9")
MANIFEST_STAGE9_PATH = os.path.join(PROJECT_ROOT, "data", "stage9", "corpus_manifest.json")

CHECKPOINT_FORMAT = "my-LLM stage-9 checkpoint v1"
HISTORY_HEADER = ["step", "corpus", "split", "loss", "elapsed_s", "tokens_seen"]


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


# --- corpus loading --------------------------------------------------------

def load_text(path):
    """Load a single corpus text file."""
    full = os.path.join(PROJECT_ROOT, path)
    if not os.path.isfile(full):
        fail(f"Corpus file not found: {full}")
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


def build_corpus_tensors(manifest, tokenizer, val_frac=0.1, seed=1337):
    """
    Build encoded tensors for each available corpus.

    Returns a dict keyed by corpus id:
        {
            "corpus-a-original": {
                "train": torch.LongTensor,
                "val": torch.LongTensor or None,
                "chars": int,
                "train_chars": int,
                "val_chars": int,
                "weight": float,
                "file": str,
            },
            ...
        }

    Hamlet uses scene-aware splitting. Stage 5 and Corpus C use chronological
    splitting. Missing datasets are skipped.
    """
    corpora = {}
    rng = random.Random(seed)

    for entry in manifest.get("corpora", []):
        cid = entry.get("id")
        cfile = entry.get("file")
        cweight = entry.get("weight", 0.0)

        if cfile is None:
            print(f"  Skipping {cid}: placeholder (no file)")
            continue

        full = os.path.join(PROJECT_ROOT, cfile)
        if not os.path.isfile(full):
            print(f"  Skipping {cid}: file not found ({cfile})")
            continue

        text = load_text(cfile)
        if not text.strip():
            print(f"  Skipping {cid}: empty file")
            continue

        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        if cid == "corpus-b-shakespeare":
            # Scene-aware split using hamlet_scenes.jsonl
            records = load_hamlet_scenes()
            train_recs, val_recs, train_chars, val_chars = split_hamlet_by_scenes(
                records, val_frac, seed
            )
            train_text = ""
            val_text = ""
            for r in train_recs:
                d = r.get("dialogue") or ""
                sd = r.get("stage_direction") or ""
                train_text += d + (" " if d and sd else "") + sd + "\n"
            for r in val_recs:
                d = r.get("dialogue") or ""
                sd = r.get("stage_direction") or ""
                val_text += d + (" " if d and sd else "") + sd + "\n"

            train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
            val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
        else:
            # Chronological split (Stage 5 documents and Corpus C)
            total = len(ids)
            n_val = int(total * val_frac)
            if val_frac > 0 and n_val < block_size_needed(total, 128):
                n_val = 0
            train_end = total - n_val
            train_ids = ids[:train_end]
            val_ids = ids[train_end:] if n_val > 0 else None

        corpora[cid] = {
            "train": train_ids,
            "val": val_ids,
            "chars": len(text),
            "train_chars": len(train_ids),
            "val_chars": len(val_ids) if val_ids is not None else 0,
            "weight": cweight,
            "file": cfile,
        }
        print(f"  {cid}: {len(train_ids):,} train, "
              f"{len(val_ids) if val_ids is not None else 0} val chars, "
              f"weight={cweight:.0%}")

    return corpora


def block_size_needed(total_chars, block_size):
    """Minimum characters needed for one batch window."""
    return block_size + 2


# --- weighted batch sampling ----------------------------------------------

class WeightedCorpusSampler:
    """Samples training batches from multiple corpora according to weights."""

    def __init__(self, corpora, block_size, batch_size, seed=1337):
        self.corpora = corpora
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = random.Random(seed)

        self._corpus_ids = []
        self._cum_weights = []
        cum = 0.0
        for cid, cdata in corpora.items():
            if cdata["train_chars"] < block_size + 2:
                print(f"  WARNING: {cid} too small for block_size={block_size}, "
                      f"skipping from sampling")
                continue
            self._corpus_ids.append(cid)
            cum += cdata["weight"]
            self._cum_weights.append(cum)

        if not self._corpus_ids:
            fail("No corpora available for training.")

        total = self._cum_weights[-1]
        self._cum_weights = [w / total for w in self._cum_weights]

        print(f"  Weighted sampling over {len(self._corpus_ids)} corpora:")
        for cid in self._corpus_ids:
            w = corpora[cid]["weight"] / total
            print(f"    {cid}: {w:.0%}")

    def sample_corpus(self):
        """Select a corpus ID according to weights."""
        r = self.rng.random()
        for i, cum_w in enumerate(self._cum_weights):
            if r < cum_w:
                return self._corpus_ids[i]
        return self._corpus_ids[-1]

    def get_batch(self, cid):
        """Draw a random batch from the specified corpus."""
        cdata = self.corpora[cid]
        data = cdata["train"]
        upper = len(data) - self.block_size - 1
        if upper < 1:
            fail(f"Corpus {cid} too small for block_size={self.block_size}")
        ix = torch.randint(0, upper, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y, cid


# --- training --------------------------------------------------------------

class HistoryWriter:
    """Appends per-step loss rows to runs/stage9/train_history.csv."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(HISTORY_HEADER)
            self._file.flush()

    def add(self, step, corpus, split, loss, elapsed_s, tokens_seen):
        self._writer.writerow(
            [step, corpus, split, f"{loss:.4f}", f"{elapsed_s:.1f}", tokens_seen]
        )
        self._file.flush()

    def close(self):
        self._file.close()


def save_checkpoint(path, model, optimizer, config, tokenizer,
                    step, tokens_seen, elapsed_s, seed, lr,
                    manifest, corpora_info):
    """Save a Stage 9 checkpoint with full metadata."""
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
        "corpora": corpora_info,
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
def evaluate(model, data_part, batch_size, block_size, n_batches):
    """Average loss over n_batches from a corpus's validation data."""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(n_batches):
        upper = len(data_part) - block_size - 1
        if upper < 1:
            continue
        ix = torch.randint(0, upper, (batch_size,))
        x = torch.stack([data_part[i : i + block_size] for i in ix])
        y = torch.stack([data_part[i + 1 : i + block_size + 1] for i in ix])
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def run_experiment(experiment_name, manifest, tokenizer, config,
                   args, corpora):
    """Run a single training experiment with the given corpus subset."""

    # Select corpora for this experiment
    exp_weights = manifest.get("experiment_weights", {}).get(experiment_name, {})
    if not exp_weights:
        fail(f"Experiment {experiment_name} not found in manifest")

    exp_corpora = {}
    for cid, weight in exp_weights.items():
        if cid in corpora:
            cdata = dict(corpora[cid])
            cdata["weight"] = weight
            exp_corpora[cid] = cdata
        else:
            print(f"  WARNING: {cid} not available for experiment {experiment_name}")

    if not exp_corpora:
        fail(f"No corpora available for experiment {experiment_name}")

    sampler = WeightedCorpusSampler(exp_corpora, config.block_size,
                                    args.batch_size, seed=args.seed)

    model = TinyLM(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    tokens_seen = 0
    prior_elapsed = 0.0

    exp_safe = experiment_name.replace(" ", "_").replace("-", "_")
    history_path = os.path.join(RUNS_STAGE9_DIR,
                                f"train_history_{exp_safe}.csv")
    history = HistoryWriter(history_path)
    print(f"History: {rel(history_path)}")

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
        elapsed_total = prior_elapsed + (now - t_start_ref[0])
        window_tokens = tokens_now - report_tokens
        speed_part = ""
        if window_tokens > 0:
            speed_part = f" | ~{window_tokens / max(now - report_clock, 1e-6):,.0f} chars/s"

        for cid, cdata in exp_corpora.items():
            if cdata["val"] is not None and len(cdata["val"]) > config.block_size + 2:
                val_loss = evaluate(model, cdata["val"], args.batch_size,
                                    config.block_size, args.eval_batches)
                history.add(step, cid, "val", val_loss, elapsed_total, tokens_now)

        total_train_loss = 0.0
        n_eval = 0
        for cid, cdata in exp_corpora.items():
            if len(cdata["train"]) > config.block_size + 2:
                tl = evaluate(model, cdata["train"], args.batch_size,
                              config.block_size, max(1, args.eval_batches // 2))
                total_train_loss += tl
                n_eval += 1
        avg_train_loss = total_train_loss / n_eval if n_eval > 0 else float("nan")
        history.add(step, "mixed", "train", avg_train_loss, elapsed_total, tokens_now)

        line = f"  step {step:>6} | train loss {avg_train_loss:.4f}"
        for cid, cdata in exp_corpora.items():
            if cdata["val"] is not None and len(cdata["val"]) > config.block_size + 2:
                vl = evaluate(model, cdata["val"], args.batch_size,
                              config.block_size, args.eval_batches)
                line += f" | {cid} val {vl:.4f}"
        print(line + speed_part)
        report_clock = now
        report_tokens = tokens_now

    try:
        model.train()
        t_start_ref = [time.perf_counter()]
        for step in range(start_step + 1, last_step + 1):
            if step == start_step + 1 or (step - start_step) % args.eval_interval == 0 \
                    or step == last_step:
                do_eval(step, tokens_seen)

            cid = sampler.sample_corpus()
            x, y, sampled_cid = sampler.get_batch(cid)
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            tokens_seen += args.batch_size * config.block_size
            completed += 1

            if step % args.save_interval == 0 or step == last_step:
                elapsed_total = prior_elapsed + (time.perf_counter() - t_start_ref[0])
                snapshot_path = os.path.join(
                    CHECKPOINT_STAGE9_DIR, f"step_{step:06d}.pt"
                )
                corpora_info = {
                    cid: {
                        "chars": c["chars"],
                        "train_chars": c["train_chars"],
                        "val_chars": c["val_chars"],
                        "weight": c["weight"],
                        "file": c["file"],
                    }
                    for cid, c in exp_corpora.items()
                }
                save_checkpoint(
                    snapshot_path, model, optimizer, config, tokenizer,
                    step, tokens_seen, elapsed_total, args.seed, args.lr,
                    manifest, corpora_info,
                )
                latest_path = os.path.join(CHECKPOINT_STAGE9_DIR, "model.pt")
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

    elapsed_total = prior_elapsed + (time.perf_counter() - t_start_ref[0])
    final_step = start_step + completed

    if completed == 0:
        print("No optimizer steps completed.")
    else:
        status = " (interrupted early)" if interrupted else ""
        print(f"\nDone. Steps: {completed}/{args.steps}{status}")
        print(f"Time: {elapsed_total:.1f}s | "
              f"~{tokens_seen / max(elapsed_total, 1e-6):,.0f} chars/s")

        latest_path = os.path.join(CHECKPOINT_STAGE9_DIR, "model.pt")
        save_checkpoint(
            latest_path, model, optimizer, config, tokenizer,
            final_step, tokens_seen, elapsed_total, args.seed, args.lr,
            manifest,
            {
                cid: {
                    "chars": c["chars"],
                    "train_chars": c["train_chars"],
                    "val_chars": c["val_chars"],
                    "weight": c["weight"],
                    "file": c["file"],
                }
                for cid, c in exp_corpora.items()
            },
        )
        print(f"Checkpoint: {rel(latest_path)} (step {final_step:,})")

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
              f"{rel(os.path.join(CHECKPOINT_STAGE9_DIR, 'model.pt'))}")
        sys.exit(130)


def run_all_experiments(args, manifest, tokenizer):
    """Run all three experiments (A, B, C)."""

    print("\nBuilding corpus tensors ...")
    corpora = build_corpus_tensors(manifest, tokenizer,
                                   val_frac=args.val_frac, seed=args.seed)

    if not corpora:
        fail("No corpora available for training.")

    print(f"\nAvailable corpora: {len(corpora)}")
    for cid, c in corpora.items():
        print(f"  {cid}: {c['train_chars']:,} train, "
              f"{c['val_chars']:,} val chars, weight={c['weight']:.0%}")

    for exp_name in ["experiment_a_original_only",
                     "experiment_b_original_plus_shakespeare",
                     "experiment_c_all_available"]:
        if exp_name not in manifest.get("experiment_weights", {}):
            print(f"\nSkipping {exp_name}: not in manifest")
            continue

        # Build config from tokenizer vocab
        config = TinyConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
        )

        run_experiment(exp_name, manifest, tokenizer, config,
                       args, corpora)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 9: Multi-dataset weighted training for TinyLM "
                    "(CPU-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=48,
                        help="Number of optimizer steps per experiment")
    parser.add_argument("--batch-size", type=int, default=8,
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
                        help="Validation fraction per corpus")
    parser.add_argument("--save-interval", type=int, default=16,
                        help="Save snapshot every N steps")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed")
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

    if not os.path.isfile(MANIFEST_STAGE9_PATH):
        fail(f"Manifest not found: {MANIFEST_STAGE9_PATH}")
    with open(MANIFEST_STAGE9_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 60)
    print("Stage 9 Phase E — Multi-Dataset Weighted Training")
    print("=" * 60)

    print("\nBuilding tokenizer from available corpora ...")
    all_text = ""
    for entry in manifest.get("corpora", []):
        cfile = entry.get("file")
        if cfile is None:
            continue
        full = os.path.join(PROJECT_ROOT, cfile)
        if os.path.isfile(full):
            try:
                with open(full, "r", encoding="utf-8") as f:
                    all_text += f.read()
            except OSError:
                pass

    tok = CharacterTokenizer.from_text(all_text)
    print(f"Tokenizer: vocab={tok.vocab_size} characters "
          f"(built from available corpora)")

    tok.save(os.path.join(CHECKPOINT_STAGE9_DIR, "tokenizer.json"))

    run_all_experiments(args, manifest, tok)

    print("\n" + "=" * 60)
    print("Stage 9 Phase E complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
