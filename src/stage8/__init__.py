"""
src/stage8/ — Stage 8: Natural Language Foundation.
====================================================

Stage 8 builds the foundation for natural language learning by:

  A. Tokenizer benchmark (character vs expanded character vs BPE)
  B. Dataset validation and splitting
  C. Modern English corpus report (recommendations only, no download)
  D. Multi-dataset weighted training pipeline
  E. Training system (CPU, checkpoints, resume, evaluation)
  F. Evaluation pipeline

All code lives under src/stage8/. No existing files are modified.
Original checkpoints, tokenizer files, and data remain untouched.

Package layout:
    src/stage8/
        __init__.py          — this file
        tokenizer_benchmark.py  — Phase A: compare 3 tokenizers
        dataset_splitter.py   — Phase B: scene/document-aware splits
        corpus_report.py      — Phase C: modern English corpus recommendations
        weighted_train.py     — Phase D: multi-dataset weighted training
        train.py              — Phase E: training loop (CPU, checkpoints, resume)
        evaluate.py           — Phase F: evaluation pipeline
        tests.py              — automated tests for all phases

All writes go to:
    checkpoints/stage8/     — Stage 8 checkpoints only
    runs/stage8/            — Stage 8 training logs only

Nothing under checkpoints/ (root), data/, or artifacts/ is overwritten.
"""

__version__ = "1.0.0"
__stage__ = "Stage 8 — Natural Language Foundation"
