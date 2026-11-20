"""
tokenizer.py — the CHARACTER-LEVEL tokenizer we wrote ourselves.
================================================================

WHAT A TOKENIZER IS (one paragraph)
    A model can only do math on numbers, so text must be turned into
    numbers first. The tokenizer is that translator:
        encode("cat") -> [3, 8, 7]        (text -> token IDs)
        decode([3, 8, 7]) -> "cat"        (token IDs -> text)
    It is NOT learned from data. We build it by hand: look at all the
    characters that appear in our training text, give every one a
    stable ID number, and build the two-way lookup tables.

WHY CHARACTERS (stage-2 choice)
    The simplest possible choice: every token is exactly ONE character.
    * Pro: tiny vocabulary (only ~50-100 entries), tiny model, easy to
      debug by eye.
    * Con: the model sees less information per token than a modern
      "subword" tokenizer, so it needs more steps to learn the same
      text. Fine for learning: we may add a BPE tokenizer as an
      upgrade in a later stage.

WHY THE IDs MUST BE STABLE
    "c" = 7 today must be 7 tomorrow (and on the USB drive later).
    That is why we sort the characters and pin the special tokens to
    fixed IDs before them. If IDs shuffled between runs, the model
    would effectively "forget its language".

DESIGN NOTES
    * IDs 0 and 1 are reserved: <pad> (0) and <unk> (1).
        - <unk> means "character I have never seen" -> the tokenizer
          still works instead of crashing on foreign text.
        - <pad> is a placeholder reserved for future batch packing.
    * This file uses ONLY the Python standard library. A tokenizer is
      plain bookkeeping; no tensors needed here.

This file is part of the from-scratch my-LLM project. We authored it.
"""

import json
import os

# The two reserved IDs every model save/load cycle will rely on.
PAD = "<pad>"   # id 0: placeholder character (future use for batching)
UNK = "<unk>"   # id 1: "unknown character" fallback

# Order matters: these occupy the first IDs, real characters come after.
SPECIAL_TOKENS = [PAD, UNK]


class CharacterTokenizer:
    """Two-way lookup table text <-> integer IDs, built from a character set."""

    def __init__(self, characters):
        # A set removes duplicates; sorting makes the ID assignment
        # deterministic (same characters -> same IDs on every machine).
        # We also validate that every entry really is one character.
        chars = sorted({c for c in characters})
        bad = [repr(c) for c in chars if not isinstance(c, str) or len(c) != 1]
        if bad:
            raise ValueError(
                f"A tokenizer can only be built from single characters, "
                f"but these are not: {bad}"
            )
        self.characters = chars

        # The lookup tables. Characters start AFTER the special tokens,
        # so deleting/adding characters in the corpus can never move
        # the IDs of <pad>/<unk>.
        self.stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + chars)}
        self.itos = {i: tok for i, tok in enumerate(SPECIAL_TOKENS + chars)}

        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]

    # --- construction helpers -------------------------------------------

    @classmethod
    def from_text(cls, text):
        """Build the vocabulary by scanning a corpus string."""
        return cls(set(text))

    @property
    def vocab_size(self):
        """Number of distinct IDs the model will use."""
        return len(self.stoi)

    # --- the two core operations ----------------------------------------

    def encode(self, text):
        """Text -> list of integer IDs. Unknown characters become <unk>."""
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        """List/tensor of IDs -> text. Padding is skipped (adds no characters)."""
        out = []
        for i in ids:
            if i == self.pad_id:
                continue
            out.append(self.itos.get(int(i), UNK))
        return "".join(out)

    # --- saving / loading (so training and later inference always use
    #     the exact same character->ID mapping) ----------------------------

    def save(self, path):
        """Store the vocabulary as a small, human-readable JSON file."""
        payload = {
            "format": "my-LLM character tokenizer v1",
            "special_tokens": SPECIAL_TOKENS,
            "characters": self.characters,
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        """Rebuild a tokenizer from a file saved by .save()."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("Tokenizer file was made with different special tokens.")
        return cls(payload["characters"])

    # --- convenience ------------------------------------------------------

    def __repr__(self):
        return f"CharacterTokenizer(vocab_size={self.vocab_size})"


# ---------------------------------------------------------------------
# Small self-demo:  python src/tokenizer.py
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample = "the quick brown fox jumps over the lazy dog."
    tok = CharacterTokenizer.from_text(sample)
    print(tok)
    print(f"vocabulary: {tok.characters}")
    print(f"<pad> id = {tok.pad_id}, <unk> id = {tok.unk_id}")

    ids = tok.encode("the fox")
    print(f"encode('the fox') = {ids}")
    print(f"decode back       = {tok.decode(ids)!r}")
    assert tok.decode(ids) == "the fox"

    print(f"unknown char demo: encode('x?y') = {tok.encode('x?')} ...")
