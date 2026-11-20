# Stage 8 Readiness Report
============================================================

## 1. Executive Summary

Stage 7 is complete: the Hamlet corpus has been extracted, cleaned, and
structured into 20 scenes with verified dialogue, stage directions, and
speaker attribution. The pipeline is ready for Stage 8 (training), but
several gaps must be addressed first — most importantly multi-dataset
support and tokenizer expansion.

## 2. Training Pipeline Assessment

### Current State (`src/train.py`)

| Capability | Status | Details |
|------------|--------|---------|
| Single-dataset training | ✅ Implemented | `--data` points to one text file |
| Multiple datasets | ❌ Not implemented | Only one `--data` argument |
| Weighted sampling | ❌ Not implemented | Manifest weights (0.25/0.25/0.20/0.15/0.15) are documented only |
| Chronological train/val split | ✅ Implemented | No shuffling, validation held from end |
| Scene-aware splitting | ❌ Not implemented | Splits by character position, not scene boundary |
| Checkpoint resume | ✅ Implemented | With vocabulary lock |
| Gradient clipping | ✅ Implemented | `--clip` parameter |
| CSV loss history | ✅ Implemented | `runs/train_history.csv` |

### Gap: Manifest Weights Are Documented, Not Implemented

`data/manifests/corpus_manifest.json` documents five corpora with weights
summing to 1.0:

| Corpus | Name | Weight |
|--------|------|--------|
| corpus-a-original | Original language material | 0.25 |
| corpus-b-shakespeare | Hamlet (Shakespearean) | 0.25 |
| corpus-c-modern | Modern English | 0.20 |
| corpus-d-conversation | Conversational examples | 0.15 |
| corpus-e-teacher | Teacher-generated | 0.15 |

**These weights are not implemented.** `src/train.py` reads a single `--data`
file and trains on it exclusively. There is no weighted mixing, no dataset
rotation, and no manifest reading.

### Required Change for Stage 8

Implement multi-dataset training with weighted sampling:

1. Read `data/manifests/corpus_manifest.json` to get corpus list and weights
2. Load each corpus text file
3. Build a combined tokenizer from all corpora (or use per-corpus tokenizers)
4. Implement weighted sampling: each training step picks a dataset
   proportional to its weight, then draws a batch from that dataset
5. Keep train/val split per dataset (scene-aware for Hamlet)
6. Ensure Hamlet (25%) cannot dominate merely because its file is larger

### Gap: Character-Level Splitting, Not Scene-Aware

For Hamlet, `src/train.py` splits by character position after encoding.
This means a validation window can start mid-scene and a training window
can end mid-scene, leaking context across the boundary.

**Required**: Split Hamlet at scene boundaries (using
`data/processed/hamlet_scenes.jsonl`), assigning whole scenes to train or
validation. Other corpora can continue with chronological splitting.

## 3. Tokenizer Assessment

### Current Character Tokenizer

| Aspect | Status |
|--------|--------|
| Implementation | `src/tokenizer.py` — character-level, stdlib only |
| Vocab size | 60 (58 chars + `<pad>` + `UNK`) |
| Hamlet coverage | Incomplete — missing `?`, `-`, `J`, `K`, `Y`, `’`, `‘`, `”`, `–` |
| UNK rate on Hamlet | High — smart quotes (~1,000 occurrences) would become `UNK` |
| Checkpoint compatibility | Vocabulary-locked — resume requires exact charset match |

### Stage 6 Expansion (Separate Lineage)

`artifacts/stage6/` contains an expanded tokenizer (vocab: 65) that adds
`-`, `?`, `J`, `K`, `Y` while preserving all 60 original IDs. However:

- It does NOT cover smart quotes (`’`, `‘`, `”`) or en dash (`–`)
- It is a separate artifact lineage, not integrated into `src/train.py`
- `src/train.py` will not work with it without changes (different format)

### BPE Tokenizer (`src/stage7/byte_bpe_tokenizer.py`)

| Aspect | Status |
|--------|--------|
| Implementation | ✅ Complete, from scratch, stdlib only |
| Tests | ✅ 36/36 pass |
| UTF-8 handling | ✅ Every byte 0-255 has a token — no UNK problem |
| Determinism | ✅ Tie-breaking by token IDs makes training reproducible |
| Save/load | ✅ JSON serialization round-trips correctly |
| Integration with `src/train.py` | ❌ Not integrated — separate stage-7 pipeline |

### Character vs BPE Comparison

| Criterion | Character Tokenizer | BPE Tokenizer |
|-----------|-------------------|---------------|
| Vocab size | ~60-69 | Configurable (e.g. 1,024) |
| Tokens per word | ~5-8 (one per char) | ~1-3 (subword units) |
| UNK problem | Yes (chars outside vocab) | No (all bytes covered) |
| Implementation complexity | Low | Medium |
| Training stability | High (simple) | High (deterministic merges) |
| English modeling | Poor (no subword sharing) | Better (morpheme-like units) |
| Checkpoint compatibility | Strict vocab lock | Strict vocab lock |

### Recommendation for Stage 8

**Expand the character tokenizer** rather than adopting BPE immediately:

1. Add smart quotes (`’`, `‘`, `”`) and en dash (`–`) to reach vocab ~69
2. Preserve all 60 original token IDs exactly
3. Create a versioned tokenizer file (`checkpoints/tokenizer.json` v2)
4. Keep BPE as a Stage 5+ upgrade path (already implemented and tested)

Rationale: The character tokenizer is simpler, well-understood, and sufficient
for the small model (~822K params). BPE adds complexity without clear benefit
at this scale. The BPE implementation is available as a tested fallback.

## 4. Model Architecture Assessment

Current configuration (from `checkpoints/model.pt`):

| Parameter | Value |
|-----------|-------|
| Parameters | ~822,272 |
| Vocab size | 60 (needs expansion for Hamlet) |
| Layers | 4 |
| Heads | 4 |
| Embedding dim | 128 |
| Context length | 128 |
| Dropout | 0.1 |

This is appropriate for Stage 8. No architecture changes needed. The model
can accept an expanded vocabulary by growing `tok_emb` and `lm_head` (as
demonstrated in Stage 6).

## 5. Hardware Assessment

| Resource | Specification | Impact on Stage 8 |
|----------|---------------|-------------------|
| CPU | Intel Core i5-4570T (2 cores, 4 threads) | Training will be slow (~50ms/step) |
| RAM | 16 GB | Sufficient for small model + corpora |
| GPU | Intel HD Graphics 4600 (no CUDA) | CPU-only training confirmed |
| Storage | 32 GB USB drive target | Model + corpora fit easily |

Hamlet corpus is 187,110 characters (~187 KB). Combined with the ~13 KB Stage 5
corpus, total training data is ~200 KB. This is small enough for CPU training
with short experiments.

## 6. Required Changes for Stage 8

### Priority 1: Multi-Dataset Training (Required)

Modify `src/train.py` or create a new training script that:

1. Reads `data/manifests/corpus_manifest.json`
2. Loads each corpus file listed in the manifest
3. Implements weighted random sampling across datasets
4. Preserves the existing CLI interface (`--steps`, `--lr`, `--batch-size`, etc.)
5. Maintains checkpoint compatibility (save/load with manifest metadata)

### Priority 2: Tokenizer Expansion (Required)

1. Expand `checkpoints/tokenizer.json` to add `’`, `‘`, `”`, `–` (and keep `-`, `?`, `J`, `K`, `Y` from Stage 6)
2. Preserve all 60 original token IDs
3. Version the tokenizer file
4. Update `src/train.py` vocabulary lock to work with the expanded charset

### Priority 3: Scene-Aware Splitting for Hamlet (Recommended)

When training on Hamlet, split at scene boundaries using
`data/processed/hamlet_scenes.jsonl` rather than by character position.
This prevents context leakage between train and validation.

### Priority 4: Conversation Dataset Format (Documented, Not Implemented)

The manifest documents the conversation format:

```json
{"messages":[
  {"role":"user","content":"Explain gravity."},
  {"role":"assistant","content":"Gravity is the force that attracts objects toward each other."}
]}
```

This format is ready for use when conversational data is created. No training
pipeline changes are needed yet — the format is just a JSONL file that can be
loaded like any other text corpus (with role markers preserved).

## 7. What Is NOT Needed for Stage 8

| Item | Reason |
|------|--------|
| BPE integration | Character tokenizer is sufficient; BPE available as tested fallback |
| Model architecture change | Current ~822K param model is appropriate |
| GPU support | Project is CPU-only by design |
| Large training runs | Stage 8 should use short experiments (50-200 steps) |
| Pretrained weights | Project builds from scratch only |

## 8. Recommended Stage 8 Implementation Sequence

### Step 1: Expand the Tokenizer (1-2 hours)

Create a script or modify `prepare_stage6.py` to produce a final expanded
tokenizer (vocab ~69) covering all characters in Hamlet + Stage 5 corpus.
Preserve all 60 original IDs. Save as `checkpoints/tokenizer_v2.json`.

### Step 2: Implement Multi-Dataset Training (4-8 hours)

Modify `src/train.py` to:

1. Add `--manifest` argument pointing to `data/manifests/corpus_manifest.json`
2. Read corpus list and weights from manifest
3. Load and combine corpus texts
4. Implement weighted sampling per training step
5. Save manifest metadata in checkpoints
6. Maintain full backwards compatibility (single `--data` still works)

### Step 3: Scene-Aware Splitting for Hamlet (2-4 hours)

When `--data` points to `data/processed/hamlet_clean.txt`, optionally use
`data/processed/hamlet_scenes.jsonl` to split at scene boundaries instead of
character position. Assign whole scenes to train/validation deterministically.

### Step 4: Short Smoke Training Run (30 minutes)

Run a bounded experiment:

```powershell
python src\train.py --manifest data\manifests\corpus_manifest.json --steps 100 --batch-size 4 --block-size 128
```

Verify:
- Loss decreases over steps
- Checkpoint saves and resumes correctly
- No dataset dominates (Hamlet stays at 25% weight)
- Generation produces recognizable text shapes

### Step 5: Document Results (1-2 hours)

Record loss curves, generation samples, and any issues. Decide whether to
proceed with longer training or adjust architecture/hyperparameters.

## 9. Estimated Timeline

| Phase | Task | Est. Time |
|-------|------|-----------|
| 1 | Tokenizer expansion | 1-2 hours |
| 2 | Multi-dataset training | 4-8 hours |
| 3 | Scene-aware splitting | 2-4 hours |
| 4 | Smoke test | 30 minutes |
| 5 | Documentation | 1-2 hours |
| **Total** | | **~8-16 hours** |

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Hamlet dominates due to larger file size | Weighted sampling caps Hamlet at 25% |
| Smart quotes become UNK | Expand tokenizer before training |
| Checkpoint incompatibility | Version tokenizer; never resume with mismatched charset |
| Slow CPU training | Keep experiments short (50-200 steps) |
| Scene leakage in validation | Use scene-boundary splitting for Hamlet |

---
