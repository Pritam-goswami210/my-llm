# my-LLM — a tiny language model built from scratch

> Our own tokenizer · Our own model architecture · Trained on legally usable text ·
> Runs fully offline · Small enough for a 32 GB USB drive

---

## 1. Why we are building this

The most honest way to understand how modern language models work is to build a small
one by hand. This project has firm rules so that everything we produce is genuinely ours:

- We implement the **tokenizer** (how text becomes numbers), the **model architecture**
  (how those numbers flow through the math), and the **training / inference scripts** ourselves.
- We never call a hosted LLM API, and we never download a pretrained model and present it as ours.
- We may use open-source libraries (PyTorch) for tensor math and training plumbing —
  that is like using a hammer to build a house: the tools are borrowed, the house is ours.
- Any text data we train on must be **legally usable** — public domain, permissively
  licensed, or written by us. We download nothing without asking first.
- Everything must run **offline** on our local machine.

## 2. What is a language model, in plain words?

A language model has exactly one job it trains on: **predict the next piece of text,
given the text that came before.**

```
Context:   "the cat sat on the "
Question:  what comes next?
Answer:    probability for every possible next token
           → "mat" 62%  ·  "floor" 21%  ·  "elephant" 0.001%  ·  ...
```

- The pieces of text are called **tokens** (a token might be a character, or a word-pieces
  like "pre" + "train" + "ing"). Our first project uses **characters** as tokens — the
  simplest possible choice.
- The model is a neural network whose *only* skill is turning this prediction question
  into a symbol → probability mapping. Trained long enough on enough text, doing this
  well forces the model to absorb grammar, spelling, style, facts, and arithmetic-looking
  patterns. That *emergence* is why LLMs feel intelligent.
- **Training** means: show the model millions of "guess the next token" examples, measure
  how wrong it was (a single number called the **loss**), and nudge its adjustable numbers
  (the **weights / parameters**) slightly to make the loss smaller. Repeat a very long time.
- **Inference (generation)** means: start with a prompt, predict the next token, append it,
  predict again, and so on until the model produces a full response. Small random choices
  during this loop are what make generations look creative instead of repetitive.

### Small glossary (we will add to this as we go)

| Term | Meaning |
|---|---|
| Token | One chunk of text (character, subword, or word) the model reads/writes |
| Tokenizer | Code that converts text → token IDs, and token IDs → text |
| Embedding | A lookup table turning each token ID into a vector of numbers |
| Attention | The mechanism that lets a token "look back" at earlier tokens and pull in relevant information |
| Transformer block | One layer of (attention + small neural network); models stack many of them |
| Context length | How many tokens the model can see at once |
| Loss | A single number measuring "how wrong the prediction was" — training minimizes it |
| Checkpoint | A saved snapshot of the weights; resume training later or copy to another machine |
| Parameters | All the adjustable numbers; our first model has < 1 million, large ones have billions |

## 3. Training from scratch vs fine-tuning

These two words get confused constantly. The difference is **where the weights start** and
**who did the expensive part**:

| | Training from scratch (our path) | Fine-tuning |
|---|---|---|
| Starting weights | Random initialization — the model knows nothing | Somebody else's fully trained model |
| Data needed | Lots of text (even a small model needs MBs–GBs) | Much less (the hard patterns are already learned) |
| Compute needed | High — this is the expensive part | Low — usually just the final layers / a short run |
| What you truly own | 100% of the result | A modified copy of someone else's training |
| Fails if... | Too little data / too big a model | Your task differs too much from the original model |

**We chose from scratch** so we own every weight and understand every line. As a consequence
our model must be **tiny** — a small model trained on ~1–10 MB of text is the classic way to
learn everything above (/tutorial-style "char-LMs" do exactly this).

As a learning-exercise comparison, fine-tuning would be like practicing editing on top of
someone else's essay; from-scratch is writing the essay from an empty page. We deliberately
picked the empty page.

## 4. Planned stages (roadmap)

Checked stages are complete. Each stage may **not** start without your explicit approval.

- [x] **Stage 1 — Environment audit & project docs** (this step: `hardware_report.md`,
      `requirements.txt`, this README)
- [x] **Stage 2 — Tokenizer + model skeleton**: character-level tokenizer (our own),
      tiny Transformer with randomly initialized weights, and a *sanity check* script that
      merely runs a forward pass to prove shapes/quantities are right. **No training yet.**
- [x] **Stage 3 — First training run**: a small, legally safe, locally provided corpus
      (~Megabytes), a clean training loop, loss curves, first generated text,
      checkpoints saved *inside this folder*.
- [x] **Stage 4 — Offline inference**: a `generate.py` (no server, no API) with sensible
      sampling controls (temperature, top-k).
- [x] **Stage 5 — Improvements**: better sampling, learning-rate schedules, optional
      byte-level or BPE tokenizer, longer context.
- [x] **Stage 6 — Evaluation**: metric-based comparison (perplexity, small benchmark snippets)
      so progress is measurable rather than vibes.
- [x] **Stage 7 — Portability**: checkpoint + code + tiny data, packaged and verified to fit
      on a 32 GB USB drive and run on another plain machine.

### Data decision for Stage 3 — option B: a locally authored toy corpus

We chose to write our own small corpus: `data/tiny_corpus.txt`. It is original
(no copied books, no scraping), contains no personal information, and was
created locally with nothing downloaded.

Be honest with ourselves, in writing:

- This is a **toy** corpus (a few kilobytes). It CAN teach the pipeline:
  how a character model learns, how train vs. validation loss behaves,
  how checkpoints and resuming work. That is exactly our goal.
- It CANNOT give the model broad language knowledge, world facts, or
  reliable reasoning. Expect parroting of corpus fragments and nonsense.
  Real capability would need a much larger, legally usable corpus (Stage 5+).

Public-domain / bigger-corpus options stay on the menu for later stages —
still approval-first, never auto-downloaded.

## 5. Planned folder layout

```
my-LLM/
├── README.md            ← you are here
├── requirements.txt     ← dependencies for the current stage (documented, not auto-installed)
├── hardware_report.md   ← Step 1 output: exact machine specs + training recommendations
├── import os.py         ← a pre-existing experiment script (older attempt; kept untouched as reference — see hardware_report.md for notes)
├── src/                 ← our own code
│   ├── tokenizer.py     (Stage 2 ✓ — character-level, with <pad>/<unk>, save/load)
│   ├── model.py         (Stage 2 ✓ — "TinyLM", pre-norm Transformer, random init)
│   ├── sanity_check.py  (Stage 2 ✓ — exit test; run with:  python src\sanity_check.py)
│   ├── train.py         (Stage 3 ✓ — training loop, benchmark mode, resume, Ctrl+C safety)
│   ├── generate.py      (Stage 4 ✓ — offline CPU-only generation from a checkpoint)
│   └── sanity_check.py  (Stage 2 ✓ — exit test; run with:  python src\sanity_check.py)
├── data/
│   └── tiny_corpus.txt  ← our locally authored toy corpus (a few KB; see the data note above)
├── checkpoints/         ← everything stays INSIDE this folder
│   ├── model.pt         ← rolling resumable checkpoint (always the latest state)
│   ├── step_000016.pt   ← dated snapshots from periodic saves
│   └── tokenizer.json   ← character table (IDs stay stable across runs)
└── runs/
    ├── train_history.csv ← loss history: step, split, loss, seconds, characters seen
    └── tokenizer_demo.json (Stage 2 demo artifact)
```

## 6. Our working rules

1. Step-by-step with **approval gates** — nothing starts before you say yes.
2. Explanations in plain English before/while doing anything new.
3. **No installs, no downloads, no network calls** without explicit approval.
4. All file reads/writes stay **inside this project folder**.
5. Hardware facts are **reported honestly** — if something cannot be detected, we say so.
6. Never overwrite existing files without inspecting them and discussing first.

## 7. Stage 3 guide — running your first training experiment

All commands use the `python` already verified in Step 1 (**Python 3.12.10**) and
assume you run them from the project root (`C:\Users\Pritam\Desktop\my-LLM`).

### 0) Optional: confirm the interpreter

```powershell
python --version        # should print: Python 3.12.10
```

### 1) Quick CPU benchmark (writes nothing, no real training)

```powershell
python src\train.py --benchmark
```

Reports milliseconds per forward+backward+update step and characters per second
for our exact model and batch settings. It is deliberately a **5-step probe with
1 warm-up step** — honest but *not* a full-run ETA.

### 2) The recommended first experiment (about half a minute)

```powershell
python src\train.py --steps 48
```

Defaults: batch 8 × context 128, AdamW lr 1e-3, gradient clipping at 1.0,
train/val loss evaluated every 8 steps (and at the end), snapshots every
16 steps plus a final one, and a short generation sample at the end.

### 3) Changing the run length and other knobs

```powershell
python src\train.py --steps 150        # more work, still short
python src\train.py --steps 48 --lr 0.0005
python src\train.py --steps 48 --batch-size 4     # lighter on the CPU
python src\train.py --steps 48 --sample-tokens 300
```

Other useful switches: `--block-size 64` (fresh runs only), `--val-frac 0`
(skip validation for short corpora), `--seed 7`, `--data path\to\your.txt`.
The full list: `python src\train.py --help`.

### 4) Resuming from a checkpoint

```powershell
python src\train.py --resume checkpoints\model.pt --steps 100
```

The checkpoint carries model weights, optimizer state, step number,
model configuration, learning rate, and tokenizer metadata. Resuming is
**strict**: a corpus with a different character set or a `--block-size`
mismatch is **rejected** with a clear error, never silently applied.
Add `--lr 0.0005` to override the stored learning rate.

Pressing **Ctrl+C** mid-run is safe: the most recent usable checkpoint is
saved before the script exits.

### 5) Where outputs land (all inside this folder)

| Path | What it is |
|---|---|
| `checkpoints\model.pt` | rolling resumable checkpoint — the latest state |
| `checkpoints\step_000016.pt` | dated snapshots from periodic saves |
| `checkpoints\tokenizer.json` | the character ↔ ID table (must stay in sync with any resumed run) |
| `runs\train_history.csv` | one row per evaluation: step, split, loss, seconds, characters seen |

### 6) How to read training vs validation loss

- **Both high, both falling →** healthy early learning. Carry on.
- **Train falling, validation falling more slowly →** still fine; watch them.
- **Train keeps falling while validation plateaus or RISES →** the model is
  **memorizing** instead of generalizing (overfitting). On a few-kilobyte toy
  corpus this is expected eventually. Honest remedies: accept it (we are
  learning the mechanics), use fewer steps, more dropout, or — later — a
  bigger corpus. Not "more training".

### 7) What this experiment can and cannot teach us

**Can:** the full mechanics — encode → batch → forward → loss → backward →
update → checkpoint → resume; how loss curves behave; overfitting vs. healthy
learning; how small a model can be and still produce recognizable text shapes.

**Cannot:** broad language skill, correct facts, useful assistant behavior.
The model only ever sees our own few kilobytes of toy text. That is the
point of Stage 3: understand the machine, not ship a product.

## 8. Stage 4 guide — generating text offline with your trained model

`src\generate.py` loads a saved checkpoint and continues any prompt, one
character at a time, entirely on the CPU. It never trains, never modifies
checkpoints, and needs no GPU.

### 1) One-shot generation (recommended first try)

```powershell
python src\generate.py --prompt "The" --max-new-chars 30 --temperature 1.0
```

Prints the prompt and the newly generated characters separately. Expect
character-shaped fragments or gibberish — the model predicts the next
character from ~7 KB of toy text; it does not understand questions or
instructions.

### 2) Interactive generation (no --prompt)

```powershell
python src\generate.py
```

Type a prompt at `prompt> ` and press Enter; the model continues it. Type
`quit` or `exit` (or press Ctrl+C) to leave.

### 3) Options

```powershell
python src\generate.py --help
```

- `--prompt TEXT`       starting text (omit for interactive mode)
- `--checkpoint PATH`   default `checkpoints\model.pt`
- `--tokenizer PATH`    default `checkpoints\tokenizer.json`
- `--temperature FLOAT` <1 = safer/repetitive, >1 = wilder (default 1.0)
- `--max-new-chars N`   characters to generate (default 100, max 5000)
- `--device DEVICE`     only `cpu` is supported (this project runs CPU-only)

### 4) Safety checks this script performs

- Refuses unknown or mismatched checkpoints (wrong format, missing config,
  vocabulary mismatch vs. the tokenizer) with a clear error — never generates
  with a silently wrong alphabet.
- Rejects non-positive temperature, negative or oversized generation length,
  empty prompts, and any device other than `cpu`.
- Verifies the model state-dict loads into exactly the saved architecture;
  dimensions are never silently changed.

### 5) Honest limitation

The generated text is a sequence-prediction exercise on a toy corpus. It may
look like half-remembered corpus fragments or complete nonsense. That is the
current capability of an 823,552-parameter model trained for 48 steps — not a
bug, and not yet a useful assistant.

---

*Built locally, offline, one small step at a time.*
**warning this repo doesn't contain all the files** 
