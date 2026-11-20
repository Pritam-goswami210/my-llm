# Stage 7 — BPE Tokenizer + Shakespeare GPT Training Pipeline

## Directory Layout

```
src/stage7/                 # Stage 7 source code (isolated package)
├── __init__.py
├── byte_bpe_tokenizer.py   # Byte-level BPE tokenizer (deterministic)
├── model.py                # TinyLM model for Stage 7 (same architecture, separate from src/model.py)
├── model_config.json       # Stage 7 model configuration
├── prepare_corpus.py       # Corpus preparation: read, validate, split
├── train.py                # Stage 7 training script (NOT run during setup)
└── generate.py             # Stage 7 generation script
data/stage7/                # Shakespeare corpus goes here (user-supplied)
├── shakespeare.txt         # NOT provided — user must supply
└── README.md               # Instructions for obtaining the corpus
artifacts/stage7/           # Stage 7 checkpoints and outputs (created at runtime)
tests/stage7/               # Unit tests
├── __init__.py
└── test_bpe_tokenizer.py   # BPE tokenizer tests
```

## Preservation Rules

- No existing Stage 1–6 files are modified, deleted, or overwritten.
- `checkpoints/model.pt`, all step checkpoints, `checkpoints/tokenizer.json`, `data/tiny_corpus.txt`, and all Stage 5/6 artifacts remain untouched.
- Stage 7 writes only to `src/stage7/`, `data/stage7/`, `artifacts/stage7/`, and `tests/stage7/`.
- No model training is started by this setup. The training script exists but is not executed.
- No dataset is downloaded automatically. The Shakespeare corpus must be supplied by the user.

## How to Use (after setup)

### 1. Supply the Shakespeare corpus
Place a plaintext Shakespeare file at:
```
data/stage7/shakespeare.txt
```
(See `data/stage7/README.md` for sourcing guidance.)

### 2. Prepare the corpus
```
python src\stage7\prepare_corpus.py --input data\stage7\shakespeare.txt
```
This reads, validates, splits, and saves `train.txt` and `val.txt` under `data/stage7/`, plus `corpus_report.json`.

### 3. Train the BPE tokenizer (on training split only)
```
python src\stage7\train.py --train-tokenizer --data data\stage7\train.txt --vocab-size 1024
```
This learns BPE merges and saves the tokenizer to `artifacts/stage7/`. It does NOT train the model.

### 4. Train the model (requires explicit approval — NOT run now)
```
python src\stage7\train.py --data data\stage7\train.txt --val data\stage7\val.txt --tokenizer artifacts\stage7\tokenizer.json --steps 100
```

### 5. Generate text
```
python src\stage7\generate.py --checkpoint artifacts\stage7\model.pt --tokenizer artifacts\stage7\tokenizer.json --prompt "To be" --max-new-chars 100
```

### 6. Run unit tests
```
python -m tests.stage7.test_bpe_tokenizer
```
