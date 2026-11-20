# Stage 9 Training Experiments Report
============================================================

## 1. Overview

This report documents the pretraining experiments conducted in Stage 9 Phase E.
Three experiments were run with different corpus combinations:

| Experiment | Corpora | Weights |
|------------|---------|---------|
| Experiment A | Original (Stage 5) only | 100% |
| Experiment B | Original + Shakespeare (Hamlet) | 50% / 50% |
| Experiment C | Original + Shakespeare + Modern English | 33% / 33% / 33% |

Note: Hamlet's weight is capped at 25% when all 5 corpora are available. With only 3 corpora available, each gets 33.3% of training steps.

## 2. Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | TinyLM (decoder-only Transformer) |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Context length | 128 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Gradient clipping | 1.0 |
| Hardware | Intel Core i5-4570T, CPU-only |

## 3. Experiment A: Original Corpus Only

### Corpus
- File: data/tiny_corpus.txt
- Size: ~13 KB (20 documents)
- Split: Chronological (document-level)

### Results

| Steps | Train Loss | Val Loss | Train Perplexity | Val Perplexity | Time |
|-------|------------|----------|------------------|----------------|------|
| 5 | 4.48 | 4.48 | 88.2 | 88.3 | <1s |
| 100 | 2.43 | 2.46 | 11.4 | 11.7 | ~21s |
| 250 | 2.27 | 2.44 | 9.7 | 11.5 | ~57s |
| 500 | 2.05 | 2.45 | 7.8 | 11.5 | ~88s |

### Sample Generation (500 steps, prompt: "The")
```
sonis ag la lows anout woutthe marr sthe d. Athe ilybeth al dow
int llea le thext sofin rugug. Ite T
```

### Observations
- Loss decreased from 4.42 to 2.05 over 500 steps
- Validation loss stayed close to training loss (no severe overfitting)
- Generated text shows emerging word-like patterns but is not yet coherent
- Training speed: ~5,851 chars/second

## 4. Experiment B: Original + Shakespeare (Hamlet)

### Corpora
- Original: data/tiny_corpus.txt (~13 KB, 20 documents)
- Shakespeare: data/processed/hamlet_clean.txt (~187 KB, 20 scenes)
- Split: Original = document-level, Hamlet = scene-level

### Results

| Steps | Train Loss | Val Loss (Orig) | Val Loss (Hamlet) | Time |
|-------|------------|-----------------|-------------------|------|
| 5 | 4.40 | 4.41 | 4.42 | ~1s |
| 100 | 2.53 | 2.52 | 2.54 | ~29s |
| 250 | 2.47 | 2.48 | 2.53 | ~61s |

### Sample Generation (250 steps, prompt: "The")
```
t, om mor mrnere tha orn s nthars athecevouslo thu m, s ! ther D
e mathag atangaly thes wpate asetind
```

### Observations
- Loss decreased from 4.46 to 2.47 over 250 steps
- Both corpora show similar loss trajectories
- Hamlet validation loss is slightly higher than original corpus validation loss
- Generated text shows word-like patterns but is still largely gibberish
- Training speed: ~4,194 chars/second

## 5. Experiment C: Original + Shakespeare + Modern English

### Corpora
- Original: data/tiny_corpus.txt (~13 KB, 20 documents)
- Shakespeare: data/processed/hamlet_clean.txt (~187 KB, 20 scenes)
- Modern English: data/stage9/corpus_c/modern_english_local.txt (~5 KB, locally authored)
- Split: Original = document-level, Hamlet = scene-level, Modern = chronological

### Results

| Steps | Train Loss | Val Loss (Orig) | Val Loss (Hamlet) | Val Loss (Modern) | Time |
|-------|------------|-----------------|-------------------|-------------------|------|
| 5 | 4.44 | 4.42 | 4.43 | 4.41 | ~1s |
| 100 | 2.63 | 2.54 | 2.60 | 2.50 | ~39s |
| 250 | 2.50 | 2.42 | 2.52 | 2.33 | ~67s |

### Sample Generation (250 steps, prompt: "The")
```
ol that r y s, andd st linghe ale ssaldsead. hedseapy itheron t
e at stad, ttorengrper as fwer. arnt
```

### Observations
- Loss decreased from 4.44 to 2.50 over 250 steps
- Modern English corpus has the lowest validation loss (2.33 at 250 steps)
- Hamlet has the highest validation loss (2.52 at 250 steps)
- Generated text shows emerging word boundaries and some recognizable patterns
- Training speed: ~3,838 chars/second

## 6. Comparative Analysis

### Loss Convergence

| Experiment | 100 Steps | 250 Steps | 500 Steps (A only) |
|------------|-----------|-----------|-------------------|
| A (Original only) | 2.43 | 2.27 | 2.05 |
| B (Orig + Hamlet) | 2.53 | 2.47 | N/A |
| C (All three) | 2.63 | 2.50 | N/A |

### Training Speed (chars/second)

| Experiment | 100 Steps | 250 Steps |
|------------|-----------|-----------|
| A (Original only) | ~4,911 | ~4,462 |
| B (Orig + Hamlet) | ~3,569 | ~4,194 |
| C (All three) | ~2,606 | ~3,838 |

### Key Findings

1. **Experiment A converges fastest** - The original corpus is small (~13 KB) and the model quickly learns its patterns.

2. **Experiment B converges slower** - Adding Hamlet (187 KB) increases the total data significantly, slowing convergence but exposing the model to more diverse patterns.

3. **Experiment C converges slowest** - Adding the modern English corpus further diversifies the training data, slowing convergence but providing broader language exposure.

4. **Hamlet has higher validation loss** than the original corpus - Shakespearean language is more complex and archaic, making it harder for the model to learn.

5. **Modern English has the lowest validation loss** - Modern English prose is simpler and more regular than Shakespearean text.

## 7. Generated Text Analysis

At 250 steps, the model produces text with:
- Word-like character clusters
- Some recognizable word fragments
- Occasional spaces in reasonable positions
- No coherent sentences or grammar

This is expected for a small model trained for only 250 steps on ~200 KB of text. Natural language communication requires much more data and training.

## 8. Checkpoints

All checkpoints are saved under `checkpoints/stage9/`:

| Experiment | Steps | File |
|------------|-------|------|
| A | 100 | checkpoints/stage9/step_000100.pt |
| A | 250 | checkpoints/stage9/step_000250.pt |
| A | 500 | checkpoints/stage9/step_000500.pt |
| B | 100 | checkpoints/stage9/step_000100.pt |
| B | 250 | checkpoints/stage9/step_000250.pt |
| C | 100 | checkpoints/stage9/step_000100.pt |
| C | 250 | checkpoints/stage9/step_000250.pt |

Latest checkpoint for each experiment is at `checkpoints/stage9/model.pt` (last experiment run).

## 9. Training History

CSV files are saved under `runs/stage9/`:
- `train_history_experiment_a_original_only.csv`
- `train_history_experiment_b_original_plus_shakespeare.csv`
- `train_history_experiment_c_all_available.csv`

Each row contains: step, corpus, split, loss, elapsed_s, tokens_seen.

---

*Generated by src/stage9/weighted_train.py*
