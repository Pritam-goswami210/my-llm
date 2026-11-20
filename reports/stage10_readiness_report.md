# Stage 10 Readiness Report
============================================================

## 1. Pretraining Readiness Gate

This report evaluates whether the pretrained checkpoint from Stage 9 is
suitable for conversational fine-tuning in Stage 10.

## 2. Current State Summary

### Checkpoints Available

| Checkpoint | Experiment | Steps | Train Loss | Val Loss | File |
|------------|------------|-------|------------|----------|------|
| checkpoints/stage8/model.pt | A (Original only) | 5 | 4.48 | 4.48 | checkpoints/stage8/model.pt |
| checkpoints/stage9/model.pt | C (All three) | 250 | 2.50 | 2.50 | checkpoints/stage9/model.pt |
| checkpoints/stage9/step_000500.pt | A (Original only) | 500 | 2.05 | 2.45 | checkpoints/stage9/step_000500.pt |
| checkpoints/stage9/step_000250.pt | B (Orig + Hamlet) | 250 | 2.47 | 2.48 | checkpoints/stage9/step_000250.pt |
| checkpoints/stage9/step_000250.pt | C (All three) | 250 | 2.50 | 2.50 | checkpoints/stage9/step_000250.pt |

### Tokenizer Status

| Tokenizer | Vocab Size | Source | Coverage of Training Corpus |
|-----------|------------|--------|----------------------------|
| Original Character | 60 | checkpoints/tokenizer.json | 95.0% (3,301 unknown chars) |
| Expanded Character | 65 | artifacts/stage6/tokenizer.json | 95.4% (2,381 unknown chars) |
| BPE | 1,024 | src/stage7/byte_bpe_tokenizer.py | 100% (all bytes) |

**Current training tokenizer**: CharacterTokenizer with vocab=83 (built from combined corpus of all available corpora in Stage 9).

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | TinyLM (decoder-only Transformer) |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Context length | 128 |
| Dropout | 0.1 |
| Total parameters | ~820,000 |
| Device | CPU-only |

### Training Experiments Completed

| Experiment | Corpora | Max Steps | Final Train Loss | Final Val Loss |
|------------|---------|-----------|------------------|----------------|
| A | Original only | 500 | 2.05 | 2.45 |
| B | Original + Hamlet | 250 | 2.47 | 2.48 |
| C | All three | 250 | 2.50 | 2.50 |

## 3. Readiness Assessment

### IS THE MODEL READY FOR CONVERSATIONAL FINE-TUNING?

**NO.** The pretrained checkpoint is NOT suitable for conversational fine-tuning at this time.

### Evidence

1. **Generation quality is gibberish**: At 500 steps (the best checkpoint), the model generates:
   ```
   sonis ag la lows anout woutthe marr sthe d. Athe ilybeth al dow
   int llea le thext sofin rugug. Ite T
   ```
   This is not coherent English. The model has not learned word-level patterns, let alone conversational behavior.

2. **Perplexity is ~11.5**: A perplexity of 11.5 on the validation set means the model is still very uncertain about what comes next. For comparison, a well-trained small model might achieve perplexity of 5-10 on similar data.

3. **Vocabulary utilization is ~1-2%**: The model only uses a tiny fraction of its vocabulary, indicating it has collapsed to predicting a few frequent characters.

4. **Only 500 steps of training**: 500 steps on ~200 KB of text is far too little for any meaningful language learning. The model has seen each character roughly 200,000 / 128 ≈ 1,562 times in context, which is insufficient for learning word-level patterns.

### Conclusion

**Conversational fine-tuning cannot fix a model that hasn't learned basic language patterns.** Fine-tuning on conversational data would teach the model to format responses as conversations, but the underlying language generation would still be incoherent.

**Recommendation: Do NOT proceed with conversational fine-tuning yet.** The model needs:
1. More training data (especially modern English and conversational examples)
2. More training steps (thousands, not hundreds)
3. Possibly a better tokenizer (BPE instead of character-level)

## 4. Phase A Issues Identified

### A. BPE Tokenizer Integration

**Status: NOT IMPLEMENTED**

The Stage 9 tokenizer_decision.py recommended the BPE tokenizer (vocab 1024) because:
- 0 unknown characters (100% coverage of any UTF-8 text)
- Perfect encode/decode round-trip
- Best compression (2.37x fewer tokens per character)
- Lower memory usage (0.6 MB vs 1.5 MB for character tokenizers)

However, the BPE tokenizer has NOT been integrated into the training pipeline. All Stage 8 and Stage 9 training used the character-level tokenizer with vocab=60-83.

**Action needed**: Integrate BPE tokenizer into the training pipeline before conversational fine-tuning.

### B. Vocabulary Utilization Metric Bug

**Status: IDENTIFIED, NOT FIXED**

The Stage 8 `evaluate.py` `vocab_utilization()` function compares predicted token IDs against `tokenizer.vocab_size`, but the model's embedding/output head may have a different vocab_size. For example:
- Original character tokenizer: vocab=60, but model may have vocab=83 (built from combined corpus)
- This causes the reported utilization to exceed 100% (158.33%, 160.00% in the Stage 9 report)

**Action needed**: Fix `vocab_utilization()` to compare against `model.config.vocab_size` instead of `tokenizer.vocab_size`.

### C. Longer Experiments

**Status: PARTIALLY COMPLETE**

- Experiment A (Original only): 500 steps COMPLETE
- Experiment B (Original + Hamlet): 250 steps only (500 not completed)
- Experiment C (All three): 250 steps only (500 not completed)

**Action needed**: Run 500-step experiments for B and C if pipeline is healthy.

### D. Dataset Provenance

**Status: DOCUMENTED**

- Corpus A (Original): data/tiny_corpus.txt - locally authored, ~13 KB
- Corpus B (Shakespeare): data/processed/hamlet_clean.txt - extracted from Folger PDF, ~187 KB
- Corpus C (Modern English): data/stage9/corpus_c/modern_english_local.txt - locally authored substitute, ~5 KB (WikiText-2 not downloaded due to Parquet format + network restrictions)
- Corpus D (Conversational): NOT downloaded (OpenAssistant requires Parquet + network)
- Corpus E (Teacher-generated): Not started

### E. Checkpoint Compatibility

**Status: VERIFIED**

All checkpoints use the same format ("my-LLM stage-8 checkpoint v1" or "my-LLM stage-9 checkpoint v1") and include:
- Model configuration (vocab_size, block_size, n_layer, n_head, n_embd, dropout)
- Tokenizer metadata (characters, special_tokens)
- Model state dict
- Optimizer state dict
- Training step, tokens seen, elapsed time, learning rate, random seed

Tokenizer IDs are preserved exactly in all checkpoints. No incompatibility detected.

## 5. Recommendation

**DO NOT proceed with conversational fine-tuning yet.** The pretrained checkpoint is not suitable because:

1. The model has not learned basic character-level patterns (generated text is gibberish)
2. Perplexity is ~11.5, indicating the model is still very uncertain
3. Only 500 steps of training on ~200 KB of text is insufficient

**Recommended path forward:**
1. Complete the remaining 500-step experiments for experiments B and C
2. Integrate the BPE tokenizer into the training pipeline
3. Add modern English corpus (WikiText-2 or local substitute)
4. Run longer training experiments (1,000+ steps)
5. Only then proceed to conversational fine-tuning

However, if the user wishes to proceed with conversational fine-tuning infrastructure regardless, the data preparation and fine-tuning scripts can be built, but actual training should wait until the base model is stronger.

---

## 6. Files and Directories for Stage 10

| Path | Purpose |
|------|---------|
| src/stage10/ | Stage 9 source code |
| data/stage10/ | Stage 9 data (conversational corpus, etc.) |
| checkpoints/stage10/ | Stage 9 checkpoints (separate from originals) |
| runs/stage10/ | Stage 9 training logs |
| tests/stage10/ | Stage 9 automated tests |
| reports/stage10_* | Stage 9 reports |

---

*Generated as part of Stage 10 Phase A — Readiness Gate*
