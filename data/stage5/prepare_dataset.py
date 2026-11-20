"""
prepare_dataset.py — Stage 5 dataset preparation and validation.
=================================================================

WHAT IT DOES
    Reads the original source documents under data/stage5/sources/,
    cleans them reproducibly, splits whole documents into training and
    validation sets, and writes the results under data/stage5/.

    Nothing is trained here. No checkpoints or model weights are touched.
    Nothing outside this project folder is written.

WHY THE SPLIT IS DOCUMENT-LEVEL
    Each source file is one document. The train/validation split happens at
    the document level: whole documents go to one side or the other. This
    guarantees that no document appears in both splits, and it keeps
    paragraphs and stories intact instead of cutting them mid-sentence.

CLEANING STEPS
    1. Read each source file as UTF-8.
    2. Normalize line endings: Windows (\\r\\n) and old Mac (\\r) both become \\n.
    3. Strip trailing whitespace on each line; collapse more than two blank
       lines in a row down to exactly two (a single paragraph break).
    4. Strip leading/trailing whitespace from the whole document.
    5. Report empty documents, malformed input, and characters that are not
       printable ASCII or common punctuation/whitespace — but never delete
       unfamiliar characters silently.

DECISIONS RECORDED
    Every cleaning decision and count is written to data/stage5/dataset_report.json.

DETERMINISM
    Same inputs always produce the same outputs. The train/validation split
    uses a documented random seed (1337, matching the project training seed)
    and a fixed ordering, so rerunning is reproducible.

HOW TO RUN (from the project root; python = 3.12.10)
    python data\\stage5\\prepare_dataset.py

OUTPUTS (all inside data/stage5/)
    sources/            — original documents, untouched
    train.txt           — training split text
    val.txt             — validation split text
    manifest.json       — source manifest (created separately)
    dataset_report.json — machine-readable quality report
    README.md           — human-readable quality report
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

# --- project-relative paths ----------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE_DIR = os.path.join(PROJECT_ROOT, "data", "stage5")
SOURCES_DIR = os.path.join(STAGE_DIR, "sources")
MANIFEST_PATH = os.path.join(STAGE_DIR, "manifest.json")
TRAIN_PATH = os.path.join(STAGE_DIR, "train.txt")
VAL_PATH = os.path.join(STAGE_DIR, "val.txt")
REPORT_PATH = os.path.join(STAGE_DIR, "dataset_report.json")

# Deterministic split seed (same value used by src/train.py).
SEED = 1337
TRAIN_FRAC = 0.9

# Characters that are always acceptable: printable ASCII plus common
# whitespace. Anything else is reported but NOT removed.
ALLOWED_EXTRA = set(" \t\n\r\x0b\x0c")  # common whitespace
# Em dash, en dash, curly quotes/apostrophes sometimes appear in original
# prose; we accept them but list them in the report as "non-ASCII" for
# transparency rather than deleting them silently.


def fail(message):
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_newlines(text):
    """Windows CRLF and old Mac CR -> LF."""
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


def clean_document(raw_text, source_name):
    """Apply cleaning steps and return (cleaned_text, decisions_list).

    Never deletes unfamiliar characters silently. Trailing whitespace per
    line is removed, and runs of more than two blank lines are collapsed to
    two (one paragraph break)."""
    decisions = []

    if not isinstance(raw_text, str):
        fail(f"Malformed input in {source_name}: content is not text")

    # Decode check already happened at read time (UTF-8). Here we just note
    # that the file was read as text.
    decisions.append("read as UTF-8 text")

    original_newlines = raw_text.count("\r\n") + raw_text.count("\r")
    if original_newlines:
        decisions.append(f"normalized {original_newlines} non-Unix line ending(s) to \\n")

    text = normalize_newlines(raw_text)

    # Strip trailing whitespace per line, preserve intra-line spacing.
    lines = text.split("\n")
    stripped_lines = [line.rstrip() for line in lines]
    if any(line != stripped for line, stripped in zip(lines, stripped_lines)):
        decisions.append("stripped trailing whitespace from lines")

    # Collapse 3+ consecutive blank lines into 2 blank lines (one break).
    collapsed = re.sub(r"\n{3,}", "\n\n", text)
    if collapsed != text:
        decisions.append("collapsed runs of 3+ blank lines to a single paragraph break")
        text = collapsed

    text = text.strip()
    decisions.append("stripped leading/trailing whitespace from document")

    return text, decisions


def char_report(text):
    """Character frequency and coverage summary."""
    freq = Counter(text)
    total = len(text)
    unique = len(freq)
    return {
        "total_characters": total,
        "unique_characters": unique,
        "frequency": {ch: freq[ch] for ch in sorted(freq)},
        "non_ascii": {ch: freq[ch] for ch in sorted(freq) if ord(ch) > 127},
        "control_chars": {ch: freq[ch] for ch in sorted(freq) if ord(ch) < 32 and ch not in "\n\t"},
    }


def load_documents():
    """Read every source file, preserving document boundaries."""
    if not os.path.isdir(SOURCES_DIR):
        fail(f"Sources directory not found: {SOURCES_DIR}")

    files = sorted(
        f for f in os.listdir(SOURCES_DIR)
        if os.path.isfile(os.path.join(SOURCES_DIR, f)) and f.endswith(".txt")
    )
    if not files:
        fail(f"No .txt source files found in {SOURCES_DIR}")

    documents = []
    for name in files:
        path = os.path.join(SOURCES_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except UnicodeDecodeError as exc:
            fail(f"UTF-8 decode failed for {name}: {exc}")
        except OSError as exc:
            fail(f"Could not read {name}: {exc}")

        cleaned, decisions = clean_document(raw, name)
        documents.append({
            "file": name,
            "path": path,
            "sha256": sha256_file(path),
            "raw_characters": len(raw),
            "cleaned_characters": len(cleaned),
            "cleaning_decisions": decisions,
            "cleaned_text": cleaned,
        })
    return documents


def split_documents(documents):
    """Deterministic document-level split: whole documents to train or val.

    Uses the documented seed and a fixed ordering (sorted by filename), then
    allocates documents so validation gets approximately 10% of characters."""
    import random

    # Stable ordering by filename for full determinism.
    docs = sorted(documents, key=lambda d: d["file"])
    rng = random.Random(SEED)
    # Shuffle with a fixed seed so the same input always gives the same split.
    order = list(range(len(docs)))
    rng.shuffle(order)

    total_chars = sum(d["cleaned_characters"] for d in docs)
    target_val = int(total_chars * (1.0 - TRAIN_FRAC))

    val_indices = set()
    val_chars = 0
    # Greedy from the shuffled order until we reach the validation target.
    for idx in order:
        if val_chars >= target_val:
            break
        val_indices.add(idx)
        val_chars += docs[idx]["cleaned_characters"]

    # Ensure at least one document in each split.
    if not val_indices and docs:
        val_indices.add(order[0])
    if len(val_indices) == len(docs) and len(docs) > 1:
        val_indices.pop()

    train_docs = [d for i, d in enumerate(docs) if i not in val_indices]
    val_docs = [d for i, d in enumerate(docs) if i in val_indices]
    return train_docs, val_docs


def write_outputs(documents, train_docs, val_docs):
    os.makedirs(STAGE_DIR, exist_ok=True)

    # Concatenate whole documents with a clear boundary marker between them.
    def join(docs):
        parts = []
        for d in docs:
            parts.append(d["cleaned_text"])
            if not parts[-1].endswith("\n"):
                parts.append("\n")
            parts.append("\n")  # one blank line between documents
        return "".join(parts)

    train_text = join(train_docs)
    val_text = join(val_docs)

    with open(TRAIN_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(train_text)
    with open(VAL_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(val_text)

    return train_text, val_text


def build_report(documents, train_docs, val_docs, train_text, val_text):
    # Character coverage against the current tokenizer.
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
        from tokenizer import CharacterTokenizer, SPECIAL_TOKENS
    except Exception:
        CharacterTokenizer = None

    tokenizer_chars = None
    coverage = None
    if CharacterTokenizer is not None:
        try:
            tok = CharacterTokenizer.load(os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.json"))
            tokenizer_chars = tok.characters
            corpus_chars = sorted(set(train_text + val_text))
            covered = [c for c in corpus_chars if c in tokenizer_chars]
            missing = [c for c in corpus_chars if c not in tokenizer_chars]
            coverage = {
                "tokenizer_vocab_size": tok.vocab_size,
                "tokenizer_special_tokens": list(SPECIAL_TOKENS),
                "corpus_unique_characters": len(corpus_chars),
                "characters_covered_by_tokenizer": len(covered),
                "characters_missing_from_tokenizer": len(missing),
                "missing_characters": missing,
                "missing_examples": missing[:10],
            }
        except Exception as exc:
            coverage = {"error": f"could not load tokenizer: {exc}"}

    # Duplicate detection across all documents (exact match of cleaned text).
    cleaned_texts = [d["cleaned_text"] for d in documents]
    text_counts = Counter(cleaned_texts)
    exact_duplicates = {t: n for t, n in text_counts.items() if n > 1}
    duplicate_groups = []
    seen = set()
    for d in documents:
        if d["cleaned_text"] in exact_duplicates and d["cleaned_text"] not in seen:
            group = [x["file"] for x in documents if x["cleaned_text"] == d["cleaned_text"]]
            duplicate_groups.append({"text_preview": d["cleaned_text"][:60], "files": group, "count": len(group)})
            seen.add(d["cleaned_text"])

    # Train/val overlap check: no shared cleaned document text.
    train_texts = {d["cleaned_text"] for d in train_docs}
    val_texts = {d["cleaned_text"] for d in val_docs}
    overlap = train_texts & val_texts

    return {
        "format": "my-LLM stage-5 dataset report v1",
        "created": "2026-10-23",
        "seed": SEED,
        "train_fraction_target": TRAIN_FRAC,
        "total_source_documents": len(documents),
        "source_documents": [
            {
                "file": d["file"],
                "sha256": d["sha256"],
                "raw_characters": d["raw_characters"],
                "cleaned_characters": d["cleaned_characters"],
                "cleaning_decisions": d["cleaning_decisions"],
            }
            for d in documents
        ],
        "train_documents": len(train_docs),
        "validation_documents": len(val_docs),
        "train_document_files": [d["file"] for d in train_docs],
        "validation_document_files": [d["file"] for d in val_docs],
        "train_characters": len(train_text),
        "validation_characters": len(val_text),
        "train_bytes": len(train_text.encode("utf-8")),
        "validation_bytes": len(val_text.encode("utf-8")),
        "train_characters_percent": (len(train_text) / max(len(train_text) + len(val_text), 1)) * 100,
        "validation_characters_percent": (len(val_text) / max(len(train_text) + len(val_text), 1)) * 100,
        "combined_report": char_report(train_text + val_text),
        "tokenizer_coverage": coverage,
        "empty_documents": [d["file"] for d in documents if not d["cleaned_text"].strip()],
        "exact_duplicate_groups": duplicate_groups,
        "train_validation_overlap_documents": sorted(overlap),
        "train_validation_overlap_count": len(overlap),
        "cleaning_summary": {
            "line_endings_normalized": sum(1 for d in documents if any("normalized" in dec for dec in d["cleaning_decisions"])),
            "trailing_whitespace_stripped": sum(1 for d in documents if any("trailing whitespace" in dec for dec in d["cleaning_decisions"])),
            "blank_lines_collapsed": sum(1 for d in documents if any("collapsed" in dec for dec in d["cleaning_decisions"])),
        },
        "limitations": [
            "Document-level split does not guarantee absence of semantic or passage-level leakage.",
            "This corpus (~tens of KB) is still small for a character-level model; broad language knowledge needs much more data.",
            "Non-ASCII characters (em dashes, curly quotes) are preserved and reported, not deleted.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and validate the Stage 5 dataset (no training).")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing train.txt/val.txt/report if present")
    args = parser.parse_args()

    if not args.force and (os.path.exists(TRAIN_PATH) or os.path.exists(VAL_PATH) or os.path.exists(REPORT_PATH)):
        fail(
            "Outputs already exist in data/stage5/. Pass --force to overwrite, "
            "or remove the existing files first."
        )

    documents = load_documents()

    # Empty-document detection (after cleaning).
    empty = [d["file"] for d in documents if not d["cleaned_text"].strip()]
    if empty:
        print(f"NOTE: empty document(s) after cleaning: {', '.join(empty)}")

    train_docs, val_docs = split_documents(documents)
    train_text, val_text = write_outputs(documents, train_docs, val_docs)
    report = build_report(documents, train_docs, val_docs, train_text, val_text)

    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Documents read: {len(documents)}")
    print(f"Training documents: {len(train_docs)} | characters: {len(train_text):,}")
    print(f"Validation documents: {len(val_docs)} | characters: {len(val_text):,}")
    print(f"Train share: {report['train_characters_percent']:.1f}% | "
          f"validation share: {report['validation_characters_percent']:.1f}%")
    print(f"Unique characters in corpus: {report['combined_report']['unique_characters']}")
    print(f"Written: {rel(TRAIN_PATH)}")
    print(f"Written: {rel(VAL_PATH)}")
    print(f"Written: {rel(REPORT_PATH)}")

    if report["train_validation_overlap_count"] > 0:
        fail("Train/validation overlap detected — this should not happen.")
    if report["exact_duplicate_groups"]:
        print(f"NOTE: {len(report['exact_duplicate_groups'])} exact duplicate document group(s) found.")


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


if __name__ == "__main__":
    main()
