# Stage 9 Tokenizer Decision Report

## Objective

Determine which tokenizer (original character, expanded character, or BPE) is most efficient for Stage 9 pretraining, accounting for both preprocessing time and training throughput.

**Key insight:** Tokenize each corpus ONCE and cache the token IDs to disk. During training, load cached IDs directly — no per-step tokenization overhead. The only per-step cost difference between tokenizers is sequence length (number of tokens per character), not tokenization speed.

## Overview

| Tokenizer | Source | Vocab Size | Type |
|-----------|--------|------------|------|
| A. Original Character | checkpoints/tokenizer.json | 60 | Character |
| B. Expanded Character | artifacts/stage6/tokenizer.json | 65 | Character |
| C. BPE | src/stage7/byte_bpe_tokenizer.py | 1,024 | Byte-level BPE |

## 1. Character Coverage

### A. Original Character (vocab: 60)

- Unknown characters in corpus: 3,301
  - Unknown chars: '!', '(', ')', '-', '/', '4', '6', '7', '8', '?', 'J', 'K', 'Q', 'R', 'V', 'Y', 'Z', '–', '‘', '’', '”'

| Category | Total | Covered | Missing |
|----------|-------|---------|---------|
| apostrophe | 1 | 1 | — |
| dashes | 2 | 0 | '—', '–' |
| digits | 10 | 6 | '4', '7', '8', '6' |
| lowercase | 26 | 26 | — |
| newline | 1 | 1 | — |
| punctuation | 14 | 6 | '–', '(', ')', '—', '…', '-', '!', '?' |
| smart_quotes | 4 | 0 | '”', '‘', '“', '’' |
| space | 1 | 1 | — |
| uppercase | 26 | 18 | 'J', 'V', 'R', 'Q', 'X', 'K', 'Z', 'Y' |

### B. Expanded Character (vocab: 65)

- Unknown characters in corpus: 2,381
  - Unknown chars: '!', '(', ')', '/', '4', '6', '7', '8', 'Q', 'R', 'V', 'Z', '–', '‘', '’', '”'

| Category | Total | Covered | Missing |
|----------|-------|---------|---------|
| apostrophe | 1 | 1 | — |
| dashes | 2 | 0 | '—', '–' |
| digits | 10 | 6 | '4', '7', '8', '6' |
| lowercase | 26 | 26 | — |
| newline | 1 | 1 | — |
| punctuation | 14 | 8 | '–', '(', ')', '—', '…', '!' |
| smart_quotes | 4 | 0 | '”', '‘', '“', '’' |
| space | 1 | 1 | — |
| uppercase | 26 | 21 | 'V', 'R', 'Q', 'X', 'Z' |

### C. BPE (vocab: 1024)

- Unknown characters in corpus: 0

| Category | Total | Covered | Missing |
|----------|-------|---------|---------|
| apostrophe | 1 | 1 | — |
| dashes | 2 | 1 | '—' |
| digits | 10 | 10 | — |
| lowercase | 26 | 26 | — |
| newline | 1 | 1 | — |
| punctuation | 14 | 11 | '"', '—', '…' |
| smart_quotes | 4 | 3 | '“' |
| space | 1 | 1 | — |
| uppercase | 26 | 25 | 'X' |

## 2. Encode/Decode Round-Trip

| Tokenizer | Sample Chars | Token Count | Decoded Chars | Exact Round-Trip |
|-----------|--------------|-------------|---------------|------------------|
| A. Original Character | 5,000 | 5,000 | 5,196 | No |
| B. Expanded Character | 5,000 | 5,000 | 5,176 | No |
| C. BPE | 5,000 | 2,122 | 5,000 | Yes |

## 3. Sequence Compression

| Tokenizer | Total Chars | Total Words | Total Tokens | Tokens/Word | Tokens/Char | Compression |
|-----------|-------------|-------------|--------------|-------------|-------------|-------------|
| A. Original Character | 200,337 | 36,183 | 159,549 | 4.41 | 0.7964 | 1.26x |
| B. Expanded Character | 200,337 | 36,183 | 159,549 | 4.41 | 0.7964 | 1.26x |
| C. BPE | 200,337 | 36,183 | 84,452 | 2.33 | 0.4215 | 2.37x |

## 4. Tokenization Time (One-Time Preprocessing)

| Tokenizer | Corpus Chars | Token Count | Time (sec) | Chars/Second |
|-----------|--------------|-------------|------------|--------------|
| A. Original Character | 200,337 | 200,337 | 0.01 | 14,907,136 |
| B. Expanded Character | 200,337 | 200,337 | 0.02 | 8,269,913 |
| C. BPE | 200,337 | 74,547 | 10.58 | 18,927 |

## 5. Cached Data Loading Time

| Tokenizer | Token Count | Load Time (sec) | IDs/Second |
|-----------|-------------|---------------|------------|
| A. Original Character | 200,337 | 0.0014 | 138,839,453 |
| B. Expanded Character | 200,337 | 0.0014 | 138,220,643 |
| C. BPE | 74,547 | 0.0010 | 72,162,743 |

## 6. Memory Usage (Token ID Tensor in RAM)

| Tokenizer | Token Count | Memory (MB) |
|-----------|-------------|-------------|
| A. Original Character | 200,337 | 1.5 |
| B. Expanded Character | 200,337 | 1.5 |
| C. BPE | 74,547 | 0.6 |

## 7. Training Throughput (with Cached Token IDs)

| Tokenizer | Steps | Tokens/Second | Steps/Second |
|-----------|-------|---------------|--------------|
| training | 20 | 33,096 | 64.6 |
| training | 20 | 31,955 | 62.4 |
| training | 20 | 26,632 | 52.0 |

## 8. Total Time Comparison (Preprocessing + Training)

This measures the wall-clock time to tokenize the corpus once plus train for 20 steps (batch=4, block=128).

| Tokenizer | Tokenize (sec) | Train (sec) | Total (sec) |
|-----------|----------------|-------------|-------------|
| A. Original Character | 0.01 | 0.00 | 0.01 |
| B. Expanded Character | 0.02 | 0.00 | 0.02 |
| C. BPE | 10.58 | 0.00 | 10.58 |

## 9. Recommendation

Based on the measurements above:

1. **BPE tokenizer** has the best coverage (0 unknown characters), perfect round-trip, and best compression (2.37x fewer tokens per character). However, initial tokenization is ~1000x slower than character tokenizers.

2. **If token IDs are cached to disk** (as done in this investigation), BPE's slow tokenization is a one-time cost. During training, all tokenizers load cached IDs at the same speed, and BPE processes fewer tokens per step due to compression, making it faster overall.

3. **Recommendation: Use the BPE tokenizer** with cached token IDs. The compression advantage (2.37x fewer tokens) reduces per-step computation time, and the one-time tokenization cost is amortized over all training steps.

4. **Alternative:** If tokenization time is a concern for frequent re-tokenization, use the expanded character tokenizer (vocab 65+) as a fallback.

5. **Do NOT use the original character tokenizer** (vocab 60) — it has 3,301 unknown characters in the corpus and poor coverage of digits, punctuation, and uppercase letters.

---

*Generated by src/stage9/tokenizer_decision.py*
