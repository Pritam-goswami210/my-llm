"""
prepare_corpus.py — Stage 7 corpus preparation.
===============================================

Reads a user-supplied Shakespeare plaintext corpus, validates it, creates
a reproducible train/validation split, and writes a provenance report.

SAFETY
------
- Does NOT download anything.
- Does NOT fabricate Shakespeare text.
- Does NOT use the Stage 5 corpus as a substitute.
- Reads only the file you point it at; writes only under data/stage7/.
- Validation text is never used to learn BPE merges (see train.py).
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE7_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "stage7")
STAGE7_ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "stage7")

DEFAULT_SPLIT = 0.9  # 90% train, 10% validation
SEED = 1337  # Same seed used elsewhere in the project


def fail(message):
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def load_text(path):
    """Read a UTF-8 text file and validate it is non-empty."""
    if not os.path.isfile(path):
        fail(
            f"Corpus file not found: {path}\n"
            f"       Place your Shakespeare plaintext at data/stage7/shakespeare.txt "
            f"or pass --input with the path to your file.\n"
            f"       See data/stage7/README.md for sourcing guidance."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError as exc:
        fail(f"UTF-8 decode failed for {path}: {exc}\n"
             f"       Ensure the file is valid UTF-8 encoded plain text.")
    except OSError as exc:
        fail(f"Could not read {path}: {exc}")
    if not text.strip():
        fail(f"Corpus file is empty: {path}")
    return text


def split_by_paragraphs(text, val_frac, seed):
    """Split text into train/validation by paragraph boundaries.

    Paragraphs are separated by one or more blank lines. Whole paragraphs
    are assigned to one split or the other, so no paragraph is split
    across train/validation. This reduces leakage compared to a raw
    character split.

    Returns (train_text, val_text).
    """
    import random

    # Split into paragraphs on blank lines
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        # Fallback: no paragraph structure; split by characters
        total = len(text)
        n_val = int(total * val_frac)
        return text[:-n_val] if n_val > 0 else text, text[-n_val:] if n_val > 0 else ""

    rng = random.Random(seed)
    indices = list(range(len(paragraphs)))
    rng.shuffle(indices)

    total_chars = sum(len(p) for p in paragraphs)
    target_val = int(total_chars * val_frac)

    val_indices = set()
    val_chars = 0
    for idx in indices:
        if val_chars >= target_val:
            break
        val_indices.add(idx)
        val_chars += len(paragraphs[idx])

    # Ensure at least one paragraph in each split
    if not val_indices and len(paragraphs) > 1:
        val_indices.add(indices[0])
    if len(val_indices) == len(paragraphs) and len(paragraphs) > 1:
        val_indices.discard(indices[-1])

    train_parts = [paragraphs[i] for i in range(len(paragraphs)) if i not in val_indices]
    val_parts = [paragraphs[i] for i in range(len(paragraphs)) if i in val_indices]

    train_text = "\n\n".join(train_parts)
    val_text = "\n\n".join(val_parts)
    return train_text, val_text


def char_report(text):
    freq = Counter(text)
    return {
        "total_characters": len(text),
        "unique_characters": len(freq),
        "frequency": {ch: freq[ch] for ch in sorted(freq)},
        "non_ascii": {ch: freq[ch] for ch in sorted(freq) if ord(ch) > 127},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a Shakespeare corpus for Stage 7: validate, "
                    "split into train/validation, and write a provenance report."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the Shakespeare plaintext file (e.g. data/stage7/shakespeare.txt)"
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of corpus to hold out for validation (0 to 0.9)"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for the split (deterministic)"
    )
    parser.add_argument(
        "--source", default="user-supplied",
        help="Source description for provenance (e.g. Project Gutenberg URL)"
    )
    parser.add_argument(
        "--license", default="unknown",
        help="License or rights status of the source text"
    )
    args = parser.parse_args()

    if not (0.0 <= args.val_frac <= 0.9):
        fail("--val-frac must be between 0 and 0.9")

    corpus_path = os.path.join(PROJECT_ROOT, args.input) if not os.path.isabs(args.input) else args.input
    text = load_text(corpus_path)

    print(f"Corpus: {rel(corpus_path)}  ({len(text):,} characters)")
    print(f"Source: {args.source}")
    print(f"License: {args.license}")

    # Split
    train_text, val_text = split_by_paragraphs(text, args.val_frac, args.seed)

    # Write outputs
    os.makedirs(STAGE7_DATA_DIR, exist_ok=True)
    os.makedirs(STAGE7_ARTIFACT_DIR, exist_ok=True)

    train_path = os.path.join(STAGE7_DATA_DIR, "train.txt")
    val_path = os.path.join(STAGE7_DATA_DIR, "val.txt")

    with open(train_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(train_text)
    with open(val_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(val_text)

    total = len(text)
    train_chars = len(train_text)
    val_chars = len(val_text)

    report = {
        "format": "my-LLM stage-7 corpus report v1",
        "source_file": rel(corpus_path),
        "source_description": args.source,
        "license": args.license,
        "source_sha256": sha256_file(corpus_path),
        "seed": args.seed,
        "val_fraction_target": args.val_frac,
        "total_characters": total,
        "total_bytes": len(text.encode("utf-8")),
        "train_characters": train_chars,
        "validation_characters": val_chars,
        "train_bytes": len(train_text.encode("utf-8")),
        "validation_bytes": len(val_text.encode("utf-8")),
        "train_characters_percent": (train_chars / total * 100) if total else 0.0,
        "validation_characters_percent": (val_chars / total * 100) if total else 0.0,
        "train_path": rel(train_path),
        "validation_path": rel(val_path),
        "train_report": char_report(train_text),
        "validation_report": char_report(val_text),
        "notes": [
            "Validation text is excluded from BPE merge learning (see train.py).",
            "Split is by paragraph boundaries to reduce leakage.",
            "Split does not guarantee absence of all semantic leakage.",
        ],
    }

    report_path = os.path.join(STAGE7_ARTIFACT_DIR, "corpus_manifest.json")
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"Train:  {train_chars:,} chars ({train_chars/total*100:.1f}%) -> {rel(train_path)}")
    print(f"Val:    {val_chars:,} chars ({val_chars/total*100:.1f}%) -> {rel(val_path)}")
    print(f"Report: {rel(report_path)}")
    print("Corpus preparation complete. No model training was performed.")


if __name__ == "__main__":
    main()
