"""
tokenizer_benchmark.py — Phase A: Tokenizer Benchmark.
=====================================================

Compares three tokenizers available in the project:

  A. Original character tokenizer (checkpoints/tokenizer.json, vocab: 60)
  B. Expanded character tokenizer from Stage 6 (artifacts/stage6/tokenizer.json, vocab: 65)
  C. BPE tokenizer from Stage 7 (src/stage7/byte_bpe_tokenizer.py, vocab: 1024 default)

Measures for each:

  1. Character coverage (lowercase, uppercase, numbers, punctuation,
     smart quotes, apostrophes, en-dashes, em-dashes, Unicode, newlines)
  2. Unknown-token behavior (which corpus chars map to UNK)
  3. Encode/decode round-trip accuracy
  4. Average tokens per word
  5. Average tokens per character
  6. Sequence compression ratio (chars / tokens)
  7. CPU tokenization speed

No existing tokenizer files are modified. Each tokenizer is loaded
read-only. Results are written to reports/stage8_tokenizer_benchmark.md.

This file is part of the from-scratch my-LLM project. We authored it.
"""

import hashlib
import json
import os
import sys
import time

# Make src/stage7 importable for the BPE tokenizer
# __file__ is src/stage8/tokenizer_benchmark.py
# dirname x3 = project root (my-LLM/)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
STAGE7_DIR = os.path.join(PROJECT_ROOT, "src", "stage7")
if STAGE7_DIR not in sys.path:
    sys.path.insert(0, STAGE7_DIR)

from byte_bpe_tokenizer import ByteBPETokenizer  # noqa: E402


# --- project-relative paths ----------------------------------------------

ORIGINAL_TOKENIZER = os.path.join(
    PROJECT_ROOT, "checkpoints", "tokenizer.json"
)
EXPANDED_TOKENIZER = os.path.join(
    PROJECT_ROOT, "artifacts", "stage6", "tokenizer.json"
)
HAMLET_CLEAN = os.path.join(
    PROJECT_ROOT, "data", "processed", "hamlet_clean.txt"
)
STAGE5_TRAIN = os.path.join(
    PROJECT_ROOT, "data", "stage5", "train.txt"
)
STAGE5_VAL = os.path.join(
    PROJECT_ROOT, "data", "stage5", "val.txt"
)
REPORT_PATH = os.path.join(
    PROJECT_ROOT, "reports", "stage8_tokenizer_benchmark.md"
)

# Characters we expect to be representable in natural English text.
EXPECTED_CHARS = {
    "lowercase": set("abcdefghijklmnopqrstuvwxyz"),
    "uppercase": set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "digits": set("0123456789"),
    "punctuation": set(".,;:!?'\"()-–—…"),
    "apostrophe": set("'"),
    "smart_quotes": set("\u2018\u2019\u201c\u201d"),
    "dashes": set("\u2013\u2014"),
    "newline": set("\n"),
    "space": set(" "),
}


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
        fail(f"Text file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Tokenizer wrappers
# ---------------------------------------------------------------------------

class OriginalCharacterTokenizer:
    """Standalone loader for the original character tokenizer.

    Reads checkpoints/tokenizer.json directly. Same JSON format as the
    Stage 6 expanded tokenizer (special_tokens + characters list).
    """

    def __init__(self, path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") != "my-LLM character tokenizer v1":
            fail(f"Unknown tokenizer format: {payload.get('format')!r}")
        self._characters = list(payload["characters"])
        self._special_tokens = list(payload.get("special_tokens", ["<pad>", " Uzbekistan"]))
        # Build stoi/itos preserving the stored character order.
        all_tokens = list(self._special_tokens) + self._characters
        self._stoi = {tok: i for i, tok in enumerate(all_tokens)}
        self._itos = {i: tok for i, tok in enumerate(all_tokens)}
        self._pad_id = self._stoi[self._special_tokens[0]]
        self._unk_id = self._stoi[self._special_tokens[1]]
        self._unk_char = self._special_tokens[1]

    @property
    def vocab_size(self):
        return len(self._stoi)

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def characters(self):
        return self._characters

    @property
    def stoi(self):
        return self._stoi

    def encode(self, text):
        return [self._stoi.get(ch, self._unk_id) for ch in text]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self._pad_id:
                continue
            out.append(self._itos.get(int(i), self._unk_char))
        return "".join(out)

    def is_unknown(self, ch):
        return self._stoi.get(ch, self._unk_id) == self._unk_id


class ExpandedCharacterTokenizer:
    """Standalone loader for the Stage 6 expanded character tokenizer.

    Reads artifacts/stage6/tokenizer.json directly without importing
    prepare_stage6.py (which has conflicting imports with Stage 7).
    """

    def __init__(self, path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") != "my-LLM character tokenizer v1":
            fail(f"Unknown tokenizer format: {payload.get('format')!r}")
        self._characters = list(payload["characters"])
        self._special_tokens = list(payload.get("special_tokens", ["<pad>", " Uzbekistan"]))
        # Build stoi/itos preserving the stored character order (Stage 6
        # appends new chars at the end to preserve original IDs).
        all_tokens = list(self._special_tokens) + self._characters
        self._stoi = {tok: i for i, tok in enumerate(all_tokens)}
        self._itos = {i: tok for i, tok in enumerate(all_tokens)}
        self._pad_id = self._stoi[self._special_tokens[0]]
        self._unk_id = self._stoi[self._special_tokens[1]]
        self._unk_char = self._special_tokens[1]

    @property
    def vocab_size(self):
        return len(self._stoi)

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def characters(self):
        return self._characters

    @property
    def stoi(self):
        return self._stoi

    def encode(self, text):
        return [self._stoi.get(ch, self._unk_id) for ch in text]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self._pad_id:
                continue
            out.append(self._itos.get(int(i), self._unk_char))
        return "".join(out)

    def is_unknown(self, ch):
        return self._stoi.get(ch, self._unk_id) == self._unk_id


class BpeTokenizerWrapper:
    """Wrapper around the Stage 7 ByteBPETokenizer."""

    def __init__(self, vocab_size=1024):
        self._tok = ByteBPETokenizer(vocab_size=vocab_size)
        self._trained = False

    @property
    def vocab_size(self):
        return self._tok.vocab_size

    @property
    def unk_id(self):
        return self._tok.special_tokens.get("<|unk|>", 257)

    @property
    def characters(self):
        return None  # BPE doesn't have a character list

    def train(self, texts):
        self._tok.train(texts)
        self._trained = True

    def encode(self, text):
        return self._tok.encode(text)

    def decode(self, ids):
        return self._tok.decode(ids)

    def is_unknown(self, ch):
        # BPE has no UNK problem: every byte 0-255 has a token.
        return False


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def analyze_coverage(tokenizer, label, text):
    """Check which expected character categories are covered."""
    present = set(text)
    result = {
        "label": label,
        "vocab_size": tokenizer.vocab_size,
        "categories": {},
        "unknown_chars_in_corpus": set(),
        "unknown_count": 0,
    }

    # BPE tokenizer covers all bytes 0-255, so no character is unknown.
    tok_chars = tokenizer.characters
    if tok_chars is None:
        # BPE: every byte has a token, so all corpus chars are representable.
        tok_char_set = set()
        for ch in present:
            try:
                ch.encode("utf-8")
                tok_char_set.add(ch)
            except Exception:
                pass
    else:
        tok_char_set = set(tok_chars)

    total_unknown = 0
    for ch in present:
        if tokenizer.is_unknown(ch):
            result["unknown_chars_in_corpus"].add(ch)
            total_unknown += text.count(ch)
    result["unknown_count"] = total_unknown

    for category_name, char_set in EXPECTED_CHARS.items():
        covered = [c for c in char_set if c in tok_char_set]
        missing = [c for c in char_set if c not in tok_char_set]
        result["categories"][category_name] = {
            "total": len(char_set),
            "covered": len(covered),
            "missing": missing,
            "covered_chars": covered,
        }

    return result


def analyze_roundtrip(tokenizer, label, text):
    """Check encode/decode round-trip accuracy."""
    sample = text[:5000]  # Use first 5000 chars for speed
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    exact = decoded == sample
    return {
        "label": label,
        "sample_chars": len(sample),
        "token_count": len(ids),
        "decoded_chars": len(decoded),
        "exact_roundtrip": exact,
    }


def analyze_compression(tokenizer, label, text):
    """Measure tokens per word, tokens per char, compression ratio."""
    words = text.split()
    if not words:
        return {"label": label, "error": "no words"}

    total_chars = len(text)
    total_words = len(words)
    total_tokens = 0

    for word in words:
        total_tokens += len(tokenizer.encode(word))

    return {
        "label": label,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "tokens_per_word": total_tokens / total_words if total_words else 0,
        "tokens_per_char": total_tokens / total_chars if total_chars else 0,
        "compression_ratio": total_chars / total_tokens if total_tokens else 0,
    }


def analyze_speed(tokenizer, label, text, repetitions=3):
    """Measure CPU tokenization speed."""
    sample = text[:10000]  # 10K chars for speed test
    # Warm-up
    for _ in range(2):
        tokenizer.encode(sample)

    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        tokenizer.encode(sample)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    chars_per_second = len(sample) / avg_time if avg_time > 0 else 0

    return {
        "label": label,
        "sample_chars": len(sample),
        "repetitions": repetitions,
        "avg_seconds": avg_time,
        "chars_per_second": chars_per_second,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    """Run the full tokenizer benchmark and write a report."""
    print("=" * 60)
    print("Stage 8 Phase A — Tokenizer Benchmark")
    print("=" * 60)

    # Load corpora for testing
    print("\nLoading corpora ...")
    hamlet = load_text(HAMLET_CLEAN)
    stage5_train = load_text(STAGE5_TRAIN)
    stage5_val = load_text(STAGE5_VAL)
    combined = hamlet + "\n\n" + stage5_train + "\n\n" + stage5_val

    print(f"  Hamlet:        {len(hamlet):,} characters")
    print(f"  Stage 5 train: {len(stage5_train):,} characters")
    print(f"  Stage 5 val:   {len(stage5_val):,} characters")
    print(f"  Combined:      {len(combined):,} characters")

    # Initialize tokenizers
    print("\nInitializing tokenizers ...")

    tok_a = OriginalCharacterTokenizer(ORIGINAL_TOKENIZER)
    print(f"  A. Original character tokenizer: vocab={tok_a.vocab_size}")

    tok_b = ExpandedCharacterTokenizer(EXPANDED_TOKENIZER)
    print(f"  B. Expanded character tokenizer: vocab={tok_b.vocab_size}")

    tok_c = BpeTokenizerWrapper(vocab_size=1024)
    tok_c.train([combined])  # Train BPE on combined corpus
    print(f"  C. BPE tokenizer: vocab={tok_c.vocab_size}, merges={len(tok_c._tok.merges)}")

    # Run benchmarks
    print("\nRunning benchmarks ...")
    results = []

    for tok, label in [(tok_a, "A. Original Character"), (tok_b, "B. Expanded Character"), (tok_c, "C. BPE")]:
        print(f"\n  Benchmarking {label} ...")

        # Coverage
        coverage = analyze_coverage(tok, label, combined)
        results.append(("coverage", coverage))

        # Round-trip
        roundtrip = analyze_roundtrip(tok, label, combined)
        results.append(("roundtrip", roundtrip))

        # Compression
        compression = analyze_compression(tok, label, combined)
        results.append(("compression", compression))

        # Speed
        speed = analyze_speed(tok, label, combined)
        results.append(("speed", speed))

    # Write report
    write_report(results)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for kind, data in results:
        if kind == "coverage":
            print(f"\n[{data['label']}] vocab={data['vocab_size']}, "
                  f"unknown_chars_in_corpus={data['unknown_count']}")
            for cat, info in data["categories"].items():
                if info["missing"]:
                    print(f"  {cat}: missing {info['missing']}")
        elif kind == "roundtrip":
            print(f"[{data['label']}] exact_roundtrip={data['exact_roundtrip']}, "
                  f"tokens={data['token_count']} for {data['sample_chars']} chars")
        elif kind == "compression":
            print(f"[{data['label']}] tokens/word={data['tokens_per_word']:.2f}, "
                  f"compression={data['compression_ratio']:.2f}x")
        elif kind == "speed":
            print(f"[{data['label']}] {data['chars_per_second']:,.0f} chars/sec")

    print(f"\nFull report written to: {rel(REPORT_PATH)}")


def write_report(results):
    """Write the benchmark report as Markdown."""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    lines = []
    lines.append("# Stage 8 Tokenizer Benchmark Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("Three tokenizers were benchmarked against the combined "
                 "Hamlet + Stage 5 corpus:")
    lines.append("")
    lines.append("| Tokenizer | Source | Vocab Size | Type |")
    lines.append("|-----------|--------|------------|------|")
    lines.append("| A. Original Character | checkpoints/tokenizer.json | 60 | Character |")
    lines.append("| B. Expanded Character | artifacts/stage6/tokenizer.json | 65 | Character |")
    lines.append("| C. BPE | src/stage7/byte_bpe_tokenizer.py | 1,024 | Byte-level BPE |")
    lines.append("")

    # Group results by type
    by_type = {}
    for kind, data in results:
        by_type.setdefault(kind, []).append(data)

    # Coverage section
    lines.append("## 1. Character Coverage")
    lines.append("")
    lines.append("Checks which expected character categories each tokenizer can represent.")
    lines.append("")
    for cov in by_type.get("coverage", []):
        lines.append(f"### {cov['label']} (vocab: {cov['vocab_size']})")
        lines.append("")
        lines.append(f"- Unknown characters in corpus: {cov['unknown_count']:,}")
        if cov["unknown_chars_in_corpus"]:
            chars = ", ".join(repr(c) for c in sorted(cov["unknown_chars_in_corpus"]))
            lines.append(f"  - Unknown chars: {chars}")
        lines.append("")
        lines.append("| Category | Total | Covered | Missing |")
        lines.append("|----------|-------|---------|---------|")
        for cat_name in sorted(cov["categories"].keys()):
            info = cov["categories"][cat_name]
            missing_str = ", ".join(repr(c) for c in info["missing"]) if info["missing"] else "—"
            lines.append(f"| {cat_name} | {info['total']} | {info['covered']} | {missing_str} |")
        lines.append("")

    # Round-trip section
    lines.append("## 2. Encode/Decode Round-Trip")
    lines.append("")
    lines.append("Tests whether encoding then decoding reproduces the original text exactly.")
    lines.append("")
    lines.append("| Tokenizer | Sample Chars | Token Count | Decoded Chars | Exact Round-Trip |")
    lines.append("|-----------|--------------|-------------|---------------|------------------|")
    for rt in by_type.get("roundtrip", []):
        lines.append(f"| {rt['label']} | {rt['sample_chars']:,} | "
                     f"{rt['token_count']:,} | {rt['decoded_chars']:,} | "
                     f"{'Yes' if rt['exact_roundtrip'] else 'No'} |")
    lines.append("")

    # Compression section
    lines.append("## 3. Sequence Compression")
    lines.append("")
    lines.append("Measures tokens per word, tokens per character, and compression ratio "
                 "(chars / tokens).")
    lines.append("")
    lines.append("| Tokenizer | Total Chars | Total Words | Total Tokens | Tokens/Word | Tokens/Char | Compression |")
    lines.append("|-----------|-------------|-------------|--------------|-------------|-------------|-------------|")
    for comp in by_type.get("compression", []):
        lines.append(f"| {comp['label']} | {comp['total_chars']:,} | "
                     f"{comp['total_words']:,} | {comp['total_tokens']:,} | "
                     f"{comp['tokens_per_word']:.2f} | {comp['tokens_per_char']:.4f} | "
                     f"{comp['compression_ratio']:.2f}x |")
    lines.append("")

    # Speed section
    lines.append("## 4. CPU Tokenization Speed")
    lines.append("")
    lines.append("Measures characters per second on a 10,000-character sample "
                 "(Intel Core i5-4570T, CPU-only).")
    lines.append("")
    lines.append("| Tokenizer | Sample Chars | Avg Seconds | Chars/Second |")
    lines.append("|-----------|--------------|-------------|--------------|")
    for sp in by_type.get("speed", []):
        lines.append(f"| {sp['label']} | {sp['sample_chars']:,} | "
                     f"{sp['avg_seconds']:.4f} | {sp['chars_per_second']:,.0f} |")
    lines.append("")

    # Recommendation section
    lines.append("## 5. Recommendation")
    lines.append("")
    lines.append("See `reports/stage8_training_pipeline_report.md` for the "
                 "final tokenizer selection and rationale.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by src/stage8/tokenizer_benchmark.py*")

    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_benchmark()
