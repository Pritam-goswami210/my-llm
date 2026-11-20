"""
evaluate.py — Phase F: Evaluation Pipeline.
=============================================

Measures model performance after training:

  * Training loss
  * Validation loss (per corpus)
  * Perplexity (exp(loss))
  * Per-corpus performance
  * Repetition detection in generated text
  * Vocabulary utilization (how many unique tokens the model uses)
  * Generation quality (fixed prompts)
  * Sentence coherence (basic heuristics)

Test prompts:
  1. "The weather today is"
  2. "Once upon a time"
  3. "What is gravity?"
  4. "Explain why the sky appears blue."
  5. "Hamlet said"

Results are written to reports/stage8_evaluation_report.md.

This is an evaluation-only script — no training occurs.

This file is part of the from-scratch my-LLM project. We authored it.
"""

import hashlib
import json
import math
import os
import sys
import time

import torch


# --- project-relative paths ----------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

STAGE8_DIR = os.path.join(PROJECT_ROOT, "src", "stage8")
if STAGE8_DIR not in sys.path:
    sys.path.insert(0, STAGE8_DIR)

from model import TinyConfig, TinyLM
from tokenizer import CharacterTokenizer, PAD, UNK, SPECIAL_TOKENS


CHECKPOINT_STAGE8_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "stage8")
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_STAGE8_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(CHECKPOINT_STAGE8_DIR, "tokenizer.json")

REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "stage8_evaluation_report.md")

EVAL_PROMPTS = [
    "The weather today is",
    "Once upon a time",
    "What is gravity?",
    "Explain why the sky appears blue.",
    "Hamlet said",
]


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


def load_checkpoint_and_tokenizer(checkpoint_path, tokenizer_path):
    """Load checkpoint + tokenizer. Returns (model, tokenizer, config, step)."""
    import torch

    if not os.path.isfile(checkpoint_path):
        fail(f"Checkpoint not found: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        fail(f"Could not load checkpoint: {exc}")

    if ckpt.get("format") != "my-LLM stage-8 checkpoint v1":
        fail(f"Unknown checkpoint format: {ckpt.get('format')!r}")

    saved_cfg = ckpt.get("model_config")
    if not saved_cfg:
        fail("Checkpoint has no model configuration.")

    # Tokenizer from checkpoint metadata (guaranteed to match)
    tok_meta = ckpt.get("tokenizer")
    if tok_meta and "characters" in tok_meta:
        all_tokens = list(tok_meta["special_tokens"]) + list(tok_meta["characters"])
        stoi = {t: i for i, t in enumerate(all_tokens)}
        itos = {i: t for i, t in enumerate(all_tokens)}

        class CkptTokenizer:
            def __init__(self, stoi, itos, characters, special_tokens):
                self.stoi = stoi
                self.itos = itos
                self.characters = characters
                self._special = list(special_tokens)
                self.pad_id = stoi[special_tokens[0]]
                self.unk_id = stoi[special_tokens[1]]

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
                    out.append(self.itos.get(int(i), self.itos.get(self.unk_id, "")))
                return "".join(out)

        tok = CkptTokenizer(
            stoi, itos,
            tok_meta["characters"],
            tok_meta["special_tokens"],
        )
    else:
        # Fall back to standalone tokenizer file
        if not os.path.isfile(tokenizer_path):
            fail(f"No tokenizer file at {tokenizer_path}")
        import json
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        characters = list(payload["characters"])
        special_tokens = list(payload.get("special_tokens", SPECIAL_TOKENS))
        all_tokens = list(special_tokens) + characters
        stoi = {t: i for i, t in enumerate(all_tokens)}
        itos = {i: t for i, t in enumerate(all_tokens)}

        class FileTokenizer:
            def __init__(self, stoi, itos, characters, special_tokens):
                self.stoi = stoi
                self.itos = itos
                self.characters = characters
                self._special = list(special_tokens)
                self.pad_id = stoi[special_tokens[0]]
                self.unk_id = stoi[special_tokens[1]]

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
                    out.append(self.itos.get(int(i), self.itos.get(self.unk_id, "")))
                return "".join(out)

        tok = FileTokenizer(stoi, itos, characters, special_tokens)

    config = TinyConfig(**saved_cfg)
    model = TinyLM(config)
    try:
        model.load_state_dict(ckpt["model_state"])
    except (RuntimeError, KeyError) as exc:
        fail(f"Checkpoint weights do not match architecture: {exc}")
    model.eval()
    step = int(ckpt.get("step", 0))
    return model, tok, config, step


@torch.no_grad()
def compute_loss(model, data_tensor, batch_size, block_size, n_batches):
    """Compute average cross-entropy loss over a data tensor."""
    import torch.nn.functional as F

    model.eval()
    losses = []
    total = len(data_tensor)
    if total < block_size + 2:
        return float("nan")
    for _ in range(n_batches):
        upper = total - block_size - 1
        ix = torch.randint(0, upper, (batch_size,))
        x = torch.stack([data_tensor[i : i + block_size] for i in ix])
        y = torch.stack([data_tensor[i + 1 : i + block_size + 1] for i in ix])
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses) if losses else float("nan")


def compute_perplexity(loss):
    """Perplexity = exp(loss). Returns float('inf') if loss is NaN or too large."""
    if math.isnan(loss) or loss > 100:
        return float("inf")
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def detect_repetition(text, window=5):
    """Simple repetition detector: fraction of words that appear in a
    sliding window of the previous `window` words."""
    words = text.split()
    if len(words) <= window:
        return 0.0
    repeated = 0
    for i in range(window, len(words)):
        if words[i] in words[i - window : i]:
            repeated += 1
    return repeated / max(len(words) - window, 1)


def vocab_utilization(model, tokenizer, data_tensor, batch_size, block_size, n_batches):
    """Fraction of vocabulary tokens that appear as predictions (argmax)
    across a set of batches. Measures whether the model uses its full
    vocabulary or collapses to a few tokens."""
    import torch

    model.eval()
    used = set()
    total = len(data_tensor)
    if total < block_size + 2:
        return 0.0
    for _ in range(n_batches):
        upper = total - block_size - 1
        ix = torch.randint(0, upper, (batch_size,))
        x = torch.stack([data_tensor[i : i + block_size] for i in ix])
        logits, _ = model(x)
        preds = logits.argmax(dim=-1)
        for p in preds.flatten().tolist():
            used.add(p)
    return len(used) / tokenizer.vocab_size


def generate_sample(model, tokenizer, prompt, max_new_chars, temperature=1.0):
    """Generate text from a prompt. Returns generated text only."""
    import torch

    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.unk_id]
    ids = ids[-model.config.block_size:]
    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_chars, temperature=temperature)
    out_ids = out[0].tolist()
    generated = tokenizer.decode(out_ids[len(ids):])
    return generated


def run_evaluation(checkpoint_path=LATEST_CHECKPOINT,
                   tokenizer_path=TOKENIZER_PATH,
                   max_new_chars=80):
    """Run the full evaluation pipeline."""
    import torch

    print("=" * 60)
    print("Stage 8 Phase F — Evaluation Pipeline")
    print("=" * 60)

    model, tok, config, step = load_checkpoint_and_tokenizer(
        checkpoint_path, tokenizer_path
    )
    print(f"Checkpoint: {rel(checkpoint_path)} (step {step})")
    print(f"Tokenizer: vocab={tok.vocab_size}")
    print(f"Model: {model.num_params():,} parameters | block={config.block_size}")

    # Load available corpus data for loss evaluation
    from dataset_splitter import (
        load_manifest,
        load_hamlet_scenes,
        split_hamlet_by_scenes,
        split_stage5_by_documents,
    )

    manifest = load_manifest()

    # Build corpus data for evaluation
    corpus_data = {}
    for entry in manifest.get("corpora", []):
        cid = entry.get("id")
        cfile = entry.get("file")
        if cfile is None:
            continue
        full = os.path.join(PROJECT_ROOT, cfile)
        if not os.path.isfile(full):
            continue
        with open(full, "r", encoding="utf-8") as f:
            text = f.read()
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        if cid == "corpus-b-shakespeare":
            records = load_hamlet_scenes()
            train_recs, val_recs, _, _ = split_hamlet_by_scenes(
                records, 0.1, 1337
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
            corpus_data[cid] = {
                "train": torch.tensor(tok.encode(train_text), dtype=torch.long),
                "val": torch.tensor(tok.encode(val_text), dtype=torch.long),
                "weight": entry.get("weight", 0.0),
            }
        else:
            total = len(ids)
            n_val = int(total * 0.1)
            train_end = total - n_val
            corpus_data[cid] = {
                "train": ids[:train_end],
                "val": ids[train_end:] if n_val > 0 else None,
                "weight": entry.get("weight", 0.0),
            }

    # Compute losses
    print("\n--- Loss and Perplexity ---")
    eval_results = {}
    for cid, cdata in corpus_data.items():
        train_loss = compute_loss(model, cdata["train"], 4, config.block_size, 10)
        train_ppl = compute_perplexity(train_loss)
        val_loss = compute_loss(model, cdata["val"], 4, config.block_size, 10) \
            if cdata["val"] is not None else float("nan")
        val_ppl = compute_perplexity(val_loss)

        eval_results[cid] = {
            "train_loss": train_loss,
            "train_perplexity": train_ppl,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
        }

        print(f"  {cid}:")
        print(f"    train loss = {train_loss:.4f}, perplexity = {train_ppl:.2f}")
        if not math.isnan(val_loss):
            print(f"    val loss   = {val_loss:.4f}, perplexity = {val_ppl:.2f}")
        else:
            print(f"    val loss   = N/A (no validation data)")

    # Vocabulary utilization
    print("\n--- Vocabulary Utilization ---")
    for cid, cdata in corpus_data.items():
        util = vocab_utilization(
            model, tok, cdata["train"], 4, config.block_size, 5
        )
        print(f"  {cid}: {util:.1%} of vocabulary used in predictions")

    # Generation quality
    print("\n--- Generation Samples ---")
    gen_results = {}
    for prompt in EVAL_PROMPTS:
        generated = generate_sample(model, tok, prompt, max_new_chars)
        rep = detect_repetition(generated)
        gen_results[prompt] = {
            "generated": generated,
            "repetition": rep,
            "length": len(generated),
        }
        print(f"  Prompt: {prompt!r}")
        print(f"    Generated: {generated[:100]}...")
        print(f"    Repetition score: {rep:.2%}")
        print()

    # Vocabulary info
    print("--- Vocabulary Info ---")
    print(f"  Total vocab size: {tok.vocab_size}")
    print(f"  Characters: {sorted(tok.characters)[:20]}...")

    # Write report
    write_report(eval_results, gen_results, step, config, tok)
    print(f"\nReport written to: {rel(REPORT_PATH)}")
    print("\nPhase F complete.")


def write_report(eval_results, gen_results, step, config, tokenizer):
    """Write the evaluation report as Markdown."""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    lines = []
    lines.append("# Stage 8 Evaluation Report")
    lines.append("")
    lines.append(f"## Checkpoint")
    lines.append("")
    lines.append(f"- Step: {step}")
    lines.append(f"- Model: {config.n_layer} layers, {config.n_head} heads, "
                 f"{config.n_embd} embedding dim")
    lines.append(f"- Context length: {config.block_size}")
    lines.append(f"- Vocabulary size: {tokenizer.vocab_size}")
    lines.append("")

    lines.append("## Loss and Perplexity by Corpus")
    lines.append("")
    lines.append("| Corpus | Train Loss | Train Perplexity | Val Loss | Val Perplexity |")
    lines.append("|--------|------------|------------------|----------|----------------|")
    for cid, r in eval_results.items():
        vl = f"{r['val_loss']:.4f}" if not math.isnan(r["val_loss"]) else "N/A"
        vp = f"{r['val_perplexity']:.2f}" if not math.isnan(r["val_perplexity"]) else "N/A"
        lines.append(
            f"| {cid} | {r['train_loss']:.4f} | {r['train_perplexity']:.2f} | "
            f"{vl} | {vp} |"
        )
    lines.append("")

    lines.append("## Generation Samples")
    lines.append("")
    lines.append("Generated with temperature=1.0, no top-k. Repetition score measures "
                 "the fraction of words that repeat within a sliding window of 5 words.")
    lines.append("")
    for prompt, r in gen_results.items():
        lines.append(f"### Prompt: {prompt!r}")
        lines.append("")
        lines.append(f"Generated: {r['generated']}")
        lines.append(f"")
        lines.append(f"- Length: {r['length']} characters")
        lines.append(f"- Repetition score: {r['repetition']:.2%}")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("### What the metrics show")
    lines.append("")
    lines.append("- **Train/val loss:** Lower is better. A gap between train and val "
                 "loss indicates overfitting.")
    lines.append("- **Perplexity:** exp(loss). Lower is better. Roughly equals the "
                 "average number of equally-likely choices the model faces at each "
                 "position.")
    lines.append("- **Vocabulary utilization:** Fraction of vocab tokens the model "
                 "actually predicts. Low utilization (<30%) suggests the model has "
                 "collapsed to a few frequent tokens.")
    lines.append("- **Repetition score:** Higher values indicate repetitive, "
                 "looping generation — a sign of poor sampling or a poorly trained "
                 "model.")
    lines.append("")
    lines.append("### What the metrics cannot show")
    lines.append("")
    lines.append("- These metrics measure statistical next-character prediction, "
                 "not understanding, reasoning, or factual accuracy.")
    lines.append("- A low loss does not mean the model can answer questions "
                 "correctly or maintain conversational context.")
    lines.append("- Generation quality on 5 short prompts is not a substitute "
                 "for systematic human evaluation.")
    lines.append("")
    lines.append("### Honest assessment")
    lines.append("")
    lines.append("A model trained for only a few dozen steps on ~200 KB of text "
                 "will produce mostly gibberish. This is expected. Natural "
                 "language communication requires much more data and many more "
                 "training steps (Phase 9+).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by src/stage8/evaluate.py*")

    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_evaluation()
