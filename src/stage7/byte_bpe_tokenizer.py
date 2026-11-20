"""
byte_bpe_tokenizer.py — Byte-level BPE tokenizer for Stage 7.
=============================================================

A real, deterministic Byte Pair Encoding (BPE) tokenizer, written from
scratch using only the Python standard library.

DESIGN
------
1. Input text is encoded to UTF-8 bytes.
2. Base vocabulary: byte tokens 0-255 (token ID == byte value).
3. Special tokens: assigned distinct IDs starting at 256.
4. Merged tokens: assigned IDs starting after all special tokens.
5. Merges are learned from training data by repeatedly finding the most
   frequent adjacent pair of token IDs and merging it into a new token.
6. Encoding applies merges in the order they were learned (rank order).
7. Decoding concatenates token bytes and decodes as UTF-8.

NO UNKNOWN-CHARACTER PROBLEM
----------------------------
Every byte value 0-255 has a token ID, so any valid UTF-8 text can be
encoded and perfectly decoded. There are no "unknown" bytes.

SERIALIZATION
-------------
The tokenizer saves to JSON containing:
- vocab_size, special_tokens, next_id
- id_to_bytes (token ID -> byte list, or null for special tokens)
- merges (list of [id1, id2, new_id] in learning order)

Reloading reproduces identical encoding/decoding behavior.

DETERMINISM
-----------
When multiple pairs have the same frequency, ties are broken by the
pair's token IDs (lowest first pair, then lowest second pair). This
makes training fully reproducible on the same input.
"""

import json
import os
from collections import Counter


class ByteBPETokenizer:
    """Byte-level BPE tokenizer.

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size. Must be at least 256 (one token per
        byte value). Default 1024.
    special_tokens : dict[str, int] or None
        Mapping of special token name -> token ID. IDs must be >= 256
        to avoid colliding with byte tokens. Default adds four common
        special tokens: <pad>=256, <|unk|>=257, <|start|>=258,
        <|end|>=259. Pass {} or an empty dict to use no special tokens.
    """

    def __init__(self, vocab_size=1024, special_tokens=None):
        if vocab_size < 256:
            raise ValueError(
                f"vocab_size must be at least 256 (one per byte), got {vocab_size}"
            )
        self._vocab_size = vocab_size

        # Default special tokens (IDs 256-259)
        default_special = {
            "<pad>": 256,
            "<|unk|>": 257,
            "<|start|>": 258,
            "<|end|>": 259,
        }
        self.special_tokens = dict(default_special)
        if special_tokens:
            self.special_tokens.update(special_tokens)

        # Validate special token IDs
        for name, tid in self.special_tokens.items():
            if not isinstance(tid, int) or tid < 256:
                raise ValueError(
                    f"Special token '{name}' has invalid ID {tid!r}; "
                    "special token IDs must be integers >= 256"
                )
            if tid >= vocab_size:
                raise ValueError(
                    f"Special token '{name}' has ID {tid} which exceeds "
                    f"vocab_size {vocab_size}"
                )

        # id_to_bytes: token_id -> bytes (None for special tokens)
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for tid in self.special_tokens.values():
            self.id_to_bytes[tid] = None

        # Learned merges: list of ((id1, id2), new_id) in learning order
        self.merges = []
        # merge_ranks: (id1, id2) -> rank index (lower = higher priority)
        self.merge_ranks = {}

        # Next available token ID
        max_special = max(self.special_tokens.values()) if self.special_tokens else 255
        self.next_id = max(256, max_special + 1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self):
        return self._vocab_size

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pair_stats(token_ids):
        """Count adjacent pairs in a sequence of token IDs."""
        pairs = Counter()
        for i in range(len(token_ids) - 1):
            pairs[(token_ids[i], token_ids[i + 1])] += 1
        return pairs

    @staticmethod
    def _merge_pair(token_ids, pair, new_id):
        """Replace all non-overlapping occurrences of `pair` with `new_id`."""
        new_ids = []
        i = 0
        n = len(token_ids)
        while i < n:
            if i < n - 1 and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(token_ids[i])
                i += 1
        return new_ids

    def train(self, texts):
        """Learn BPE merges from a list of text strings.

        Each text is encoded to UTF-8 bytes, then represented as a
        sequence of byte token IDs (0-255). Merges are learned until
        ``vocab_size`` is reached or no pairs remain.

        The validation text must NOT be passed here — only training
        text, so that validation data never influences the merge table.
        """
        # Encode all texts to sequences of byte token IDs
        sequences = []
        for text in texts:
            byte_data = text.encode("utf-8")
            sequences.append(list(byte_data))

        if not sequences:
            return

        # Learn merges until vocab_size is reached
        while self.next_id < self._vocab_size:
            # Count pairs across all sequences
            pair_stats = Counter()
            for seq in sequences:
                if len(seq) < 2:
                    continue
                for pair, count in self._get_pair_stats(seq).items():
                    pair_stats[pair] += count

            if not pair_stats:
                break  # No pairs left to merge

            # Select most frequent pair with deterministic tie-breaking:
            # lowest (-count, id1, id2) => highest count, then lowest ids
            best_pair = min(
                pair_stats.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )[0]

            new_id = self.next_id
            self.next_id += 1

            # Apply merge to all sequences
            sequences = [
                self._merge_pair(seq, best_pair, new_id) for seq in sequences
            ]

            # Record merge
            rank = len(self.merges)
            self.merges.append((best_pair, new_id))
            self.merge_ranks[best_pair] = rank

            # Compute byte representation of merged token
            b1 = self._token_bytes(best_pair[0])
            b2 = self._token_bytes(best_pair[1])
            if b1 is not None and b2 is not None:
                self.id_to_bytes[new_id] = b1 + b2
            else:
                # Merged token involves a special token -> no byte rep
                self.id_to_bytes[new_id] = None

    def _token_bytes(self, token_id):
        """Return the byte representation of a token ID, or None."""
        return self.id_to_bytes.get(token_id)

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(self, text):
        """Encode text to a list of token IDs using learned BPE merges.

        Applies merges in rank order (earliest merges first). Each merge
        replaces all non-overlapping occurrences of its pair.
        """
        byte_data = text.encode("utf-8")
        token_ids = list(byte_data)  # Start with byte token IDs

        for pair, new_id in self.merges:
            token_ids = self._merge_pair(token_ids, pair, new_id)

        return token_ids

    def decode(self, token_ids):
        """Decode token IDs back to text.

        Special tokens (with no byte representation) are skipped.
        The resulting bytes are decoded as UTF-8 with replacement of
        any invalid sequences.
        """
        byte_parts = []
        for tid in token_ids:
            b = self.id_to_bytes.get(tid)
            if b is not None:
                byte_parts.append(b)

        byte_data = b"".join(byte_parts)
        return byte_data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path):
        """Save tokenizer to a JSON file."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Serialize id_to_bytes: bytes -> list of ints, None for special
        id_to_bytes_serializable = {}
        for tid, b in self.id_to_bytes.items():
            if b is None:
                id_to_bytes_serializable[str(tid)] = None
            else:
                id_to_bytes_serializable[str(tid)] = list(b)

        merges_serializable = [
            [pair[0], pair[1], new_id] for (pair, new_id) in self.merges
        ]

        payload = {
            "format": "my-LLM byte BPE tokenizer v1",
            "vocab_size": self._vocab_size,
            "special_tokens": self.special_tokens,
            "next_id": self.next_id,
            "id_to_bytes": id_to_bytes_serializable,
            "merges": merges_serializable,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load tokenizer from a JSON file saved by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("format") != "my-LLM byte BPE tokenizer v1":
            raise ValueError(
                f"Unknown tokenizer format: {payload.get('format')!r}"
            )

        tok = cls(
            vocab_size=payload.get("vocab_size", 1024),
            special_tokens=payload.get("special_tokens", {}),
        )

        # Restore id_to_bytes
        tok.id_to_bytes = {}
        for tid_str, b in payload.get("id_to_bytes", {}).items():
            tid = int(tid_str)
            if b is None:
                tok.id_to_bytes[tid] = None
            else:
                tok.id_to_bytes[tid] = bytes(b)

        # Ensure base bytes 0-255 are always present
        for i in range(256):
            if i not in tok.id_to_bytes:
                tok.id_to_bytes[i] = bytes([i])

        # Restore merges
        tok.merges = []
        tok.merge_ranks = {}
        for entry in payload.get("merges", []):
            id1, id2, new_id = entry
            pair = (id1, id2)
            tok.merges.append((pair, new_id))
            tok.merge_ranks[pair] = len(tok.merges) - 1

        tok.next_id = payload.get("next_id", 256)
        return tok

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"ByteBPETokenizer(vocab_size={self._vocab_size}, "
            f"merges={len(self.merges)}, special_tokens={len(self.special_tokens)})"
        )
