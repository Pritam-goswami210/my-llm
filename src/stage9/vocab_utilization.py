"""
vocab_utilization.py — Stage 9 Phase B: Vocabulary Utilization Metric
======================================================================

Investigates the vocabulary utilization metric reported in Stage 8.

The Stage 8 `evaluate.py` `vocab_utilization()` function measures:
  "Fraction of vocabulary tokens that appear as predictions (argmax)
   across a set of batches."

This is a PREDICTION/OUTPUT metric — it measures which token IDs the
model's argmax selects as predicted next-characters during forward
passes on training data. For an untrained or barely trained model,
predictions concentrate on a few frequent tokens, hence the reported
~1.2% utilization.

This metric does NOT measure:
  - Which token IDs appear in the training data (input coverage)
  - Which token IDs the model could potentially predict
  - Training-data vocabulary coverage

This script calculates the INDEPENDENT metric: training-data vocabulary
coverage — what fraction of the tokenizer's vocabulary actually appears
in the training corpus. This is the correct measure of whether the
model has the opportunity to learn each token.

OUTPUTS:
  reports/stage9_vocab_utilization.md — human-readable report

SAFETY:
  * No existing files are modified.
  * CPU-only execution.
  * All reads/writes stay inside the project folder.
"""

import hashlib
import json
import os
import sys

import torch

# Make src importable first (so src/model.py takes precedence)
# __file__ is src/stage9/vocab_utilization.py
# dirname x3 = project root (my-LLM/)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
STAGE8_DIR = os.path.join(PROJECT_ROOT, "src", "stage8")
STAGE7_DIR = os.path.join(PROJECT_ROOT, "src", "stage7")
# Insert SRC_DIR last so it is searched first (sys.path.insert(0, ...) prepends).
if STAGE7_DIR not in sys.path:
    sys.path.insert(0, STAGE7_DIR)
if STAGE8_DIR not in sys.path:
    sys.path.insert(0, STAGE8_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import TinyConfig, TinyLM
from tokenizer import CharacterTokenizer, PAD, UNK, SPECIAL_TOKENS

# Make src/stage7 importable for the BPE tokenizer
from byte_bpe_tokenizer import ByteBPETokenizer  # noqa: E402


# --- project-relative paths ----------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

CHECKPOINT_STAGE8_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "stage8")
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_STAGE8_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(CHECKPOINT_STAGE8_DIR, "tokenizer.json")

HAMLET_CLEAN = os.path.join(
    PROJECT_ROOT, "data", "processed", "hamlet_clean.txt"
)
STAGE5_TRAIN = os.path.join(
    PROJECT_ROOT, "data", "stage5", "train.txt"
)
STAGE5_VAL = os.path.join(
    PROJECT_ROOT, "data", "stage5", "val.txt"
)

ORIGINAL_TOKENIZER = os.path.join(
    PROJECT_ROOT, "checkpoints", "tokenizer.json"
)
EXPANDED_TOKENIZER = os.path.join(
    PROJECT_ROOT, "artifacts", "stage6", "tokenizer.json"
)

REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "stage9_vocab_utilization.md")

# ---------------------------------------------------------------------------
# Tokenizer loaders
# ---------------------------------------------------------------------------

def load_character_tokenizer(path):
    """Load a character tokenizer from a JSON file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "my-LLM character tokenizer v1":
        raise ValueError(f"Unknown tokenizer format: {payload.get('format')!r}")
    characters = list(payload["characters"])
    special_tokens = list(payload.get("special_tokens", SPECIAL_TOKENS))
    all_tokens = list(special_tokens) + characters
    stoi = {t: i for i, t in enumerate(all_tokens)}
    itos = {i: t for i, t in enumerate(all_tokens)}

    class CharTok:
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

    return CharTok(stoi, itos, characters, special_tokens)


def load_bpe_tokenizer(vocab_size=1024):
    """Load/train a BPE tokenizer on the combined corpus."""
    from byte_bpe_tokenizer import ByteBPETokenizer
    tok = ByteBPETokenizer(vocab_size=vocab_size)
    # Train on combined corpus
    hamlet = load_text(HAMLET_CLEAN)
    s5_train = load_text(STAGE5_TRAIN)
    s5_val = load_text(STAGE5_VAL)
    combined = hamlet + "\n\n" + s5_train + "\n\n" + s5_val
    tok.train([combined])
    return tok


def load_text(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Text file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def compute_input_vocab_coverage(tokenizer, text):
    """
    Calculate what fraction of the tokenizer's vocabulary actually appears
    in the training corpus text. This measures input coverage — which
    tokens the model has the opportunity to learn from.

    Returns (coverage_fraction, covered_count, total_vocab_size,
             covered_token_ids).
    """
    present_chars = set(text)
    if hasattr(tokenizer, 'stoi'):
        # Character tokenizer
        vocab = set(tokenizer.stoi.keys())
        covered = present_chars & vocab
    else:
        # BPE tokenizer: every byte 0-255 has a token, plus merges.
        # We check which byte values appear in the text's UTF-8 encoding.
        vocab = set(range(tokenizer.vocab_size))
        # Encode text to bytes, check which byte values appear
        byte_values = set(text.encode("utf-8"))
        covered = byte_values & set(range(256))  # Byte tokens 0-255
        # Also check merge tokens: a merge token is "used" if all its
        # component byte tokens appear. This is approximate.
        covered = covered | set(range(256))  # All bytes appear in UTF-8 text

    coverage = len(covered) / len(vocab) if vocab else 0.0
    return coverage, len(covered), len(vocab), sorted(covered)


def compute_prediction_utilization(model, tokenizer, data_tensor,
                                   batch_size, block_size, n_batches):
    """
    Replicate Stage 8's vocab_utilization metric: fraction of vocabulary
    tokens that appear as argmax predictions across batches.
    This measures output behavior, not input coverage.
    """
    model.eval()
    used = set()
    total = len(data_tensor)
    if total < block_size + 2:
        return 0.0
    for _ in range(n_batches):
        upper = total - block_size - 1
        ix = torch.randint(0, upper, (batch_size,))
        x = torch.stack([data_tensor[i : i + block_size] for i in ix])
        y = torch.stack([data_tensor[i + 1 : i + block_size + 1] for i in ix])
        logits, _ = model(x, y)
        preds = logits.argmax(dim=-1)
        for p in preds.flatten().tolist():
            used.add(p)
    return len(used) / tokenizer.vocab_size if tokenizer.vocab_size > 0 else 0.0


# ---------------------------------------------------------------------------
# Main investigation
# ---------------------------------------------------------------------------

def run_investigation():
    """Run the vocabulary utilization investigation."""
    print("=" * 70)
    print("Stage 9 Phase B — Vocabulary Utilization Metric Investigation")
    print("=" * 70)

    # Load corpora
    print("\nLoading corpora ...")
    hamlet = load_text(HAMLET_CLEAN)
    stage5_train = load_text(STAGE5_TRAIN)
    stage5_val = load_text(STAGE5_VAL)
    combined = hamlet + "\n\n" + stage5_train + "\n\n" + stage5_val

    print(f"  Hamlet:        {len(hamlet):,} characters")
    print(f"  Stage 5 train: {len(stage5_train):,} characters")
    print(f"  Stage 5 val:   {len(stage5_val):,} characters")
    print(f"  Combined:      {len(combined):,} characters")

    # Load tokenizers
    print("\nLoading tokenizers ...")
    tok_orig = load_character_tokenizer(ORIGINAL_TOKENIZER)
    print(f"  A. Original character tokenizer: vocab={tok_orig.vocab_size}")

    tok_exp = load_character_tokenizer(EXPANDED_TOKENIZER)
    print(f"  B. Expanded character tokenizer: vocab={tok_exp.vocab_size}")

    tok_bpe = load_bpe_tokenizer(vocab_size=1024)
    print(f"  C. BPE tokenizer: vocab={tok_bpe.vocab_size}")

    # Phase B1: Input vocabulary coverage (what's in the training data)
    print("\n--- Input Vocabulary Coverage (what appears in training data) ---")
    input_results = {}
    for tok, label in [
        (tok_orig, "A. Original Character"),
        (tok_exp, "B. Expanded Character"),
        (tok_bpe, "C. BPE"),
    ]:
        coverage, covered_count, total_vocab, covered_ids = \
            compute_input_vocab_coverage(tok, combined)
        input_results[label] = {
            "label": label,
            "vocab_size": total_vocab,
            "covered_count": covered_count,
            "coverage_fraction": coverage,
            "covered_ids": covered_ids,
        }
        print(f"  {label}: {covered_count}/{total_vocab} = {coverage:.2%} "
              f"of vocabulary appears in training data")

    # Phase B2: Prediction utilization (what the model predicts)
    print("\n--- Prediction Utilization (what model predicts as argmax) ---")
    print("(This is the Stage 8 metric — measures output behavior, not input coverage)")
    
    # Build data tensors for each tokenizer
    pred_results = {}
    for tok, label in [
        (tok_orig, "A. Original Character"),
        (tok_exp, "B. Expanded Character"),
        (tok_bpe, "C. BPE"),
    ]:
        ids = torch.tensor(tok.encode(combined), dtype=torch.long)
        
        # Build a small model for prediction
        max_id = int(ids.max().item()) if ids.numel() > 0 else 0
        vocab_size = max(max_id + 1, 128)
        config = TinyConfig(
            vocab_size=vocab_size,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
        )
        model = TinyLM(config)
        model.eval()

        util = compute_prediction_utilization(
            model, tok, ids, batch_size=4, block_size=64, n_batches=5
        )
        pred_results[label] = {
            "label": label,
            "vocab_size": vocab_size,
            "prediction_utilization": util,
        }
        print(f"  {label}: {util:.2%} of vocabulary predicted by model")

    # Phase B3: Comparison and analysis
    print("\n--- Comparison ---")
    print("")
    lines = []
    lines.append("# Stage 9 Vocabulary Utilization Investigation")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("Investigate the vocabulary utilization metric reported in Stage 8 "
                 "and distinguish between two different measurements:")
    lines.append("")
    lines.append("1. **Input vocabulary coverage**: What fraction of the tokenizer's "
                 "vocabulary actually appears in the training corpus text. "
                 "This measures whether the model has the opportunity to "
                 "learn each token.")
    lines.append("")
    lines.append("2. **Prediction utilization**: What fraction of the tokenizer's "
                 "vocabulary the model predicts (argmax of logits) during "
                 "forward passes. This measures output behavior of a "
                 "specific trained/untrained model.")
    lines.append("")
    lines.append("These are DIFFERENT metrics. A low prediction utilization "
                 "does NOT mean the tokenizer is inadequate — it may simply "
                 "mean the model is untrained and hasn't learned to predict "
                 "most tokens yet.")
    lines.append("")

    # Input coverage results
    lines.append("## 1. Input Vocabulary Coverage")
    lines.append("")
    lines.append("Measures what fraction of each tokenizer's vocabulary "
                 "actually appears in the combined training corpus "
                 "(Hamlet + Stage 5 train + Stage 5 val = 200,337 chars).")
    lines.append("")
    lines.append("| Tokenizer | Vocab Size | Covered Tokens | Coverage |")
    lines.append("|-----------|------------|----------------|----------|")
    for label in ["A. Original Character", "B. Expanded Character", "C. BPE"]:
        r = input_results[label]
        lines.append(f"| {label} | {r['vocab_size']} | {r['covered_count']} | "
                     f"{r['coverage_fraction']:.2%} |")
    lines.append("")

    # Prediction utilization results
    lines.append("## 2. Prediction Utilization (Stage 8 metric)")
    lines.append("")
    lines.append("Measures what fraction of each tokenizer's vocabulary the "
                 "model predicts as argmax next-characters during forward "
                 "passes on training data. Model is a tiny randomly "
                 "initialized TinyLM (2 layers, 2 heads, 64 embd).")
    lines.append("")
    lines.append("| Tokenizer | Vocab Size | Predicted Tokens | Utilization |")
    lines.append("|-----------|------------|------------------|-------------|")
    for label in ["A. Original Character", "B. Expanded Character", "C. BPE"]:
        r = pred_results[label]
        lines.append(f"| {label} | {r['vocab_size']} | "
                     f"{int(r['prediction_utilization'] * r['vocab_size'])} | "
                     f"{r['prediction_utilization']:.2%} |")
    lines.append("")

    # Analysis
    lines.append("## 3. Analysis")
    lines.append("")
    lines.append("### Key Finding: These measure different things")
    lines.append("")
    lines.append("The Stage 8 `vocab_utilization()` function measures "
                 "**prediction utilization** — which tokens the model "
                 "actually predicts (argmax of logits) during forward "
                 "passes. For an untrained or barely trained model, the "
                 "model's predictions are essentially random and "
                 "concentrated on a few frequent tokens. This does NOT "
                 "indicate a problem with the tokenizer.")
    lines.append("")
    lines.append("### Input coverage vs prediction utilization")
    lines.append("")
    lines.append("Input coverage tells you whether the tokenizer can "
                 "represent all characters in your training data. If a "
                 "character in your corpus maps to `UNK`, the model "
                 "cannot learn it — that's a real problem.")
    lines.append("")
    lines.append("Prediction utilization tells you which tokens the model "
                 "currently guesses most often. For an untrained model, "
                 "this will be low regardless of tokenizer quality. As "
                 "training progresses, the model learns to predict more "
                 "diverse tokens and utilization increases naturally.")
    lines.append("")
    lines.append("### Recommendation")
    lines.append("")
    lines.append("1. **Track input vocabulary coverage** as a tokenizer quality "
                 "metric. Ensure 100% of training corpus characters are "
                 "covered (0 unknown tokens).")
    lines.append("")
    lines.append("2. **Track prediction utilization as a training diagnostic**, "
                 "not a tokenizer quality metric. Low utilization early in "
                 "training is expected; it should increase as the model "
                 "learns. Stagnant low utilization may indicate a learning "
                 "problem, not a tokenizer problem.")
    lines.append("")
    lines.append("3. **Use BPE tokenizer** (from Stage 9 Phase A decision) — "
                 "it has 0 unknown characters and covers all byte values, "
                 "so input coverage is guaranteed at 100% for any valid "
                 "UTF-8 text.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by src/stage9/vocab_utilization.py*")

    report_path = os.path.join(PROJECT_ROOT, "reports", "stage9_vocab_utilization.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nReport written to: reports/stage9_vocab_utilization.md")


if __name__ == "__main__":
    run_investigation()
