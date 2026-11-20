# Stage 5 dataset — better training data for TinyLM (data only)

## What this is

A clean, documented, locally authored training corpus for the character-level
TinyLM, prepared during **Stage 5 — data preparation and validation only**.
No model training happened here, and no model weights, checkpoints, or source
code were modified.

The original tiny corpus (`data/tiny_corpus.txt`, ~7 KB) is **untouched**. The
Stage 5 data lives entirely under this `data/stage5/` directory.

## Contents

| Path | What it is |
|---|---|
| `sources/` | 20 original source documents, one file per document (boundaries preserved) |
| `manifest.json` | Source manifest: each document's title, topic, origin, and rights note |
| `prepare_dataset.py` | Reproducible cleaning + train/validation split script |
| `verify_stage5.py` | Verification tests (run once; can be re-run) |
| `train.txt` | Training split: 18 documents, 11,687 characters |
| `val.txt` | Validation split: 2 documents, 1,536 characters |
| `dataset_report.json` | Machine-readable quality report |

## How the corpus was built

- **20 source documents** were written originally for this project: everyday
  prose, short stories, dialogue, explanations (light bulbs, seeds, the moon,
  the water cycle, shadows, sleep, handwashing), computer fundamentals (binary),
  and how-to pieces (bread, pancakes, shoelaces, bicycle repair).
- No copyrighted books or websites were copied. No external datasets were
  downloaded. No personal data, credentials, or secrets are included.
- No paragraphs were artificially duplicated and no filler was added to inflate
  size.
- Each document stays a whole unit until the split.

## Cleaning performed (`prepare_dataset.py`)

1. Read each source as UTF-8.
2. Normalized Windows/old-Mac line endings to Unix `\n`.
3. Stripped trailing whitespace from each line.
4. Collapsed runs of 3+ blank lines into a single paragraph break.
5. Stripped leading/trailing whitespace from each document.

Unfamiliar characters were **never deleted silently** — they are reported in
`dataset_report.json` instead.

## Train / validation split

- Split happens at the **document level**: whole documents go to one side or
  the other, so no document appears in both splits.
- Deterministic: seeded with `1337` (the same seed used by `src/train.py`),
  documents ordered by filename, then shuffled with a fixed random seed.
- Result: **18 training documents / 2 validation documents**;
  **11,687 train characters (88.4%) / 1,536 validation characters (11.6%)**.
- Zero exact duplicates; zero train/validation document overlap.

> Honest note: a document-level split does not guarantee the absence of all
> semantic or passage-level leakage. It is a practical, reproducible boundary,
> not a proof of independence.

## Tokenizer compatibility

The current tokenizer (`checkpoints/tokenizer.json`, vocab size 60: 58
characters + `<pad>` + ` Uzbekistan`) covers **50 of the 55 unique characters**
in this corpus.

**5 characters are missing from the tokenizer:** `-`, `?`, `J`, `K`, `Y`.

Any of these characters in the new corpus would be encoded as ` Uzbekistan`
(ID 1) by the current tokenizer. This is a real incompatibility if you want
the model to learn those characters properly. Recommended next step: a
**separate tokenizer-expansion experiment** (add the missing characters,
rebuild the model with the larger vocabulary, then train) — **do not change
`src/tokenizer.py` during this stage**, per the Stage 5 rules.

## Limitations

- ~13 KB is still small for a character-level model. Diverse, coherent data
  matters more than raw size, and this corpus is deliberately modest.
- The model cannot learn characters that are not in its tokenizer.
- This is educational data, not a benchmark-grade corpus.

## How to validate

From the project root (`C:\Users\Pritam\Desktop\my-LLM`), run:

```powershell
python data\stage5\verify_stage5.py
```

Expected result: all tests pass (syntax, UTF-8/newline handling, empty-input,
duplicate detection, no train/val overlap, tokenizer coverage, valid JSON,
accurate counts).

## Source manifest

See `manifest.json` for per-document origin, topic, and rights information.
All documents are original writing created for this project; no external
material was copied and no public-domain/freely-licensed claims are made.

---

*Stage 5 complete — data only. No training, no checkpoint changes, no code changes.*
