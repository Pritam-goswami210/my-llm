# AGENTS.md — my-LLM project instructions

OpenCode loads this file into every session. It combines our project's safety
rules with distilled practices from
`reference/claude-code-best-practice` (kept read-only in this folder). Nothing
here permits installs, downloads, or API calls beyond what our project rules
already allow.

## 1. Project mission (read first)

We build a **small language model from scratch** in `C:\Users\Pritam\Desktop\my-LLM`:

- our own tokenizer, our own Transformer architecture, our own training loop
- trained only on **legally usable, locally authored text**
- runs **fully offline**, no hosted LLM APIs, no pretrained weights, no downloads
- small enough to fit on a **32 GB USB drive**

Existing code: `src/tokenizer.py`, `src/model.py`, `src/sanity_check.py`, `src/train.py`, `data/tiny_corpus.txt` (toy corpus, ~7 KB). Documented stages live in `README.md` ("Planned stages", currently through Stage 3).

## 2. Safety rules (never skip; no exceptions)

1. Step-by-step with **approval gates** — nothing starts before the user says yes.
2. Explain in plain English before/while doing anything new.
3. **No installs, no downloads, no network calls** without explicit approval.
4. All file reads/writes stay **inside this project folder** (`my-LLM/`).
5. Hardware facts reported **honestly** — never guess (CPU-only i5-4570T, no CUDA).
6. Never overwrite existing files without inspecting them first.
7. Do not touch files outside this project folder.

## 3. Approved commands for this project (run from `my-LLM/`)

| Purpose | Command | Notes |
|---|---|---|
| Syntax check | `python -m py_compile src\train.py src\tokenizer.py src\model.py src\sanity_check.py` | Python 3.12.10 (`python --version` must show 3.12.10) |
| Stage-2 exit test | `python src\sanity_check.py` | Writes `runs\tokenizer_demo.json` only |
| Benchmark (bounded, writes nothing) | `python src\train.py --benchmark` | ~5 timed steps + warm-up; not an ETA |
| Short experiment | `python src\train.py --steps 48` | ~20–30 s on our CPU |
| Resume from checkpoint | `python src\train.py --resume checkpoints\model.pt --steps <n>` | strict: same corpus/block-size or it rejects |
| Change knobs | `--steps`, `--batch-size`, `--lr`, `--clip`, `--eval-interval`, `--save-interval`, `--block-size`, `--val-frac`, `--seed` | full list: `python src\train.py --help` |

Outputs land only inside `my-LLM`: `checkpoints\`, `runs\`, `data\`. CPU-only PyTorch 2.14.1+cpu (no GPU).

## 4. Workflow practices (distilled from the best-practice reference)

These habits keep long agentic sessions effective — borrowed from
`reference/claude-code-best-practice/README.md`, `tips/`, and `CLAUDE.md`:

- **Plan before building**: describe the plan, then implement, then verify. Small vertical slices beat one big rewrite.
- **Never babysit**: give a goal + constraints and let the tool do the work; don't micromanage tool choices.
- **Challenge the result**: if a fix looks mediocre, ask to scrap it and implement the elegant solution.
- **Small commits**: one logical change per commit; commit when a task completes (roughly once per hour at most).
- **Context discipline**: rewind (`/compact` or `/clear`) after dead ends instead of letting failed attempts pollute context; keep sessions focused.
- **Subagents for context management**: use them for file-heavy research so only conclusions return to the main context.
- **Start new sessions for genuinely new tasks**; reuse context only for related follow-ups.
- **Reproducibility is best-effort**: seeds are set where practical, but bit-exact reproducibility across machines/thread counts is NOT guaranteed.

## 5. What to change before asking

- Edit only files inside `my-LLM/`. `reference/claude-code-best-practice/` is a read-only library.
- Do not create `.claude/` hooks, settings, or agents here without discussing first — they could trigger permission prompts or run commands outside the approved list.
- Do not install pip/npm packages, download datasets, or call hosted APIs.

## 6. Where things live

```
my-LLM/
├── AGENTS.md            ← this file (loaded by OpenCode every session)
├── README.md            ← goals, stages, LLM explainer, run guide
├── hardware_report.md   ← exact machine specs + model-size recommendations
├── requirements.txt     ← torch, numpy, tqdm (already installed)
├── import os.py         ← older experiment script, kept untouched as reference
├── src\tokenizer.py     ← our character tokenizer (encode/decode/save/load)
├── src\model.py         ← TinyLM pre-norm Transformer, random init
├── src\sanity_check.py  ← Stage-2 exit test
├── src\train.py         ← Stage-3 training loop + benchmark + resume
├── data\tiny_corpus.txt ← toy corpus, ~7 KB, locally authored
├── checkpoints\model.pt ← rolling resumable checkpoint
├── checkpoints\step_*.pt← periodic snapshots
├── checkpoints\tokenizer.json ← character table (IDs stable across runs)
└── runs\train_history.csv ← loss history: step, split, loss, seconds, chars seen
```
