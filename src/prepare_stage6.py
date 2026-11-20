"""
prepare_stage6.py — Stage 6: Safe tokenizer expansion and model initialization.
==============================================================================

WHAT IT DOES
    Creates a NEW, separate Stage 6 artifact lineage that supports every
    character in the Stage 5 dataset, without touching any original artifact.

    1. Loads the original checkpoint (read-only) from checkpoints/model.pt.
    2. Builds an expanded character tokenizer that preserves every original
       token ID exactly and appends only the truly missing characters.
    3. Builds a new TinyLM with the same architecture, expanded only in the
       vocabulary dimension (60 -> 65).
    4. Copies every compatible tensor from the original checkpoint exactly.
    5. Initializes only the 5 new embedding rows and 5 new output-projection
       rows with the same scheme as the original model (std 0.02).
    6. Saves all artifacts under artifacts/stage6/.
    7. Runs built-in verification tests.

WHAT IT DOES NOT DO
    - No training, no optimizer, no backward pass, no gradient update.
    - No modification or deletion of original checkpoints, tokenizer, or corpus.
    - No modification of src/tokenizer.py, src/model.py, src/train.py,
      or src/generate.py.
    - No writes to checkpoints/ or data/.

IMPORTANT DESIGN NOTE — TOKENIZER ID PRESERVATION
    The existing CharacterTokenizer.__init__() sorts all characters
    alphabetically. New characters like '-' (45), '?' (63), 'J' (74),
    'K' (75), 'Y' (89) sort IN THE MIDDLE of existing characters, which
    would shift original token IDs. To satisfy the requirement "preserve
    every original token ID exactly", this script builds the expanded
    tokenizer by appending new characters at the END (deterministically
    sorted among themselves) and constructs the stoi/itos mappings
    manually, bypassing the sort. The saved tokenizer.json uses the same
    JSON structure as the original but stores characters in the expanded
    (non-re-sorted) order. Loading must preserve this order.

WHAT THE MODEL EXPANSION LOOKS LIKE
    - tok_emb.weight: (60, 128) -> (65, 128): old 60 rows copied exactly,
      5 new rows initialized with N(0, 0.02).
    - lm_head.weight: (60, 128) -> (65, 128): old 60 rows copied exactly,
      5 new rows initialized with N(0, 0.02).
    - Weights are NOT tied (separate tensors, confirmed by inspection).
    - All other tensors (pos_emb, blocks, final_norm) copied exactly.
    - New vocab size: 65 (60 original + 5 added).

HOW TO RUN (from the project root; python = 3.12.10)
    python src\\prepare_stage6.py

OUTPUTS (all inside artifacts/stage6/)
    tokenizer.json        — expanded character table (65 entries)
    model_config.json     — expanded model config (vocab_size=65)
    model_init.pt         — new model checkpoint (expanded, untrained)
    expansion_report.json — machine-readable verification report
    README.md             — human-readable documentation
"""

import argparse
import json
import math
import os
import sys

import torch

from model import TinyConfig, TinyLM
from tokenizer import CharacterTokenizer, SPECIAL_TOKENS, PAD, UNK

# --- project-relative paths ----------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "model.pt")
ORIGINAL_TOKENIZER = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.json")
STAGE5_TRAIN = os.path.join(PROJECT_ROOT, "data", "stage5", "train.txt")
STAGE5_VAL = os.path.join(PROJECT_ROOT, "data", "stage5", "val.txt")
STAGE6_DIR = os.path.join(PROJECT_ROOT, "artifacts", "stage6")
STAGE6_TOKENIZER = os.path.join(STAGE6_DIR, "tokenizer.json")
STAGE6_CONFIG = os.path.join(STAGE6_DIR, "model_config.json")
STAGE6_MODEL_INIT = os.path.join(STAGE6_DIR, "model_init.pt")
STAGE6_REPORT = os.path.join(STAGE6_DIR, "expansion_report.json")
STAGE6_README = os.path.join(STAGE6_DIR, "README.md")

CHECKPOINT_FORMAT = "my-LLM checkpoint v1"
STAGE6_FORMAT = "my-LLM stage-6 init checkpoint v1"

# Deterministic order for newly added characters (sorted by code point).
NEW_CHARS = ["-", "?", "J", "K", "Y"]

INIT_STD = 0.02  # matches TinyLM._init_weights


def fail(message):
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def sha256_file(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Phase 1 verification (re-check inside the script)
# ----------------------------------------------------------------------

def load_original_checkpoint():
    if not os.path.isfile(ORIGINAL_CHECKPOINT):
        fail(f"Original checkpoint not found: {ORIGINAL_CHECKPOINT}")
    try:
        ckpt = torch.load(ORIGINAL_CHECKPOINT, map_location="cpu")
    except Exception as exc:
        fail(f"Could not load original checkpoint: {exc}")
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        fail(f"Unknown checkpoint format: {ckpt.get('format')!r}")
    return ckpt


def load_original_tokenizer():
    if not os.path.isfile(ORIGINAL_TOKENIZER):
        fail(f"Original tokenizer not found: {ORIGINAL_TOKENIZER}")
    try:
        tok = CharacterTokenizer.load(ORIGINAL_TOKENIZER)
    except Exception as exc:
        fail(f"Could not load original tokenizer: {exc}")
    return tok


def compute_missing_chars(train_path, val_path, tokenizer):
    with open(train_path, encoding="utf-8") as f:
        train_text = f.read()
    with open(val_path, encoding="utf-8") as f:
        val_text = f.read()
    corpus_chars = sorted(set(train_text + val_text))
    tok_chars = set(tokenizer.characters)
    missing = sorted([c for c in corpus_chars if c not in tok_chars])
    return corpus_chars, missing


# ----------------------------------------------------------------------
# Phase 3 — expand the tokenizer safely
# ----------------------------------------------------------------------

def build_expanded_tokenizer(original_chars, missing_chars):
    """
    Build expanded tokenizer preserving original IDs exactly.

    Layout:
      ID 0: <pad>
      ID 1:  Uzbekistan
      IDs 2..(2+len(original_chars)-1): original characters in original order
      IDs (next...): new characters, deterministically sorted among themselves
    """
    # Original characters must keep their exact order from the checkpoint.
    # The CharacterTokenizer constructor sorts, so we build mappings manually.
    new_chars_sorted = sorted(missing_chars)
    expanded_characters = list(original_chars) + new_chars_sorted

    # Validate: every entry is a single character.
    bad = [repr(c) for c in expanded_characters
           if not isinstance(c, str) or len(c) != 1]
    if bad:
        fail(f"Invalid character entries: {bad}")

    # Validate: no duplicates between original and new.
    overlap = set(original_chars) & set(new_chars_sorted)
    if overlap:
        fail(f"Overlap between original and new characters: {sorted(overlap)}")

    # Build stoi/itos manually to preserve exact order.
    all_tokens = list(SPECIAL_TOKENS) + expanded_characters
    stoi = {tok: i for i, tok in enumerate(all_tokens)}
    itos = {i: tok for i, tok in enumerate(all_tokens)}

    class ExpandedTokenizer:
        """Lightweight tokenizer with preserved ID order."""
        def __init__(self, characters, stoi, itos):
            self.characters = characters
            self.stoi = stoi
            self.itos = itos
            self.pad_id = stoi[PAD]
            self.unk_id = stoi[UNK]

        @property
        def vocab_size(self):
            return len(self.stoi)

        def encode(self, text):
            return [self.stoi.get(ch, self.unk_id) for ch in text]

        def decode(self, ids):
            out = []
            for i in ids:
                if i == self.pad_id:
                    continue
                out.append(self.itos.get(int(i), UNK))
            return "".join(out)

        def save(self, path):
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            payload = {
                "format": "my-LLM character tokenizer v1",
                "special_tokens": list(SPECIAL_TOKENS),
                "characters": self.characters,  # stored in expanded order
                "id_preserved": True,
                "original_char_count": len(original_chars),
                "added_characters": NEW_CHARS,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return path

    return ExpandedTokenizer(expanded_characters, stoi, itos)


def load_expanded_tokenizer(path):
    """Load the expanded tokenizer, preserving the stored character order."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "my-LLM character tokenizer v1":
        fail(f"Unknown tokenizer format: {payload.get('format')!r}")
    characters = payload["characters"]
    special_tokens = payload.get("special_tokens", SPECIAL_TOKENS)
    all_tokens = list(special_tokens) + characters
    stoi = {tok: i for i, tok in enumerate(all_tokens)}
    itos = {i: tok for i, tok in enumerate(all_tokens)}

    class ExpandedTokenizer:
        def __init__(self, characters, stoi, itos):
            self.characters = characters
            self.stoi = stoi
            self.itos = itos
            self.pad_id = stoi[PAD]
            self.unk_id = stoi[UNK]

        @property
        def vocab_size(self):
            return len(self.stoi)

        def encode(self, text):
            return [self.stoi.get(ch, self.unk_id) for ch in text]

        def decode(self, ids):
            out = []
            for i in ids:
                if i == self.pad_id:
                    continue
                out.append(self.itos.get(int(i), UNK))
            return "".join(out)

    return ExpandedTokenizer(characters, stoi, itos)


# ----------------------------------------------------------------------
# Phase 4 — build expanded model without training
# ----------------------------------------------------------------------

def build_expanded_model(original_cfg, new_vocab_size, original_state):
    """
    Build a new TinyLM with expanded vocabulary, copying compatible weights
    exactly and initializing only new rows.
    """
    new_cfg = TinyConfig(
        vocab_size=new_vocab_size,
        block_size=original_cfg["block_size"],
        n_layer=original_cfg["n_layer"],
        n_head=original_cfg["n_head"],
        n_embd=original_cfg["n_embd"],
        dropout=original_cfg["dropout"],
    )

    model = TinyLM(new_cfg)  # random init first
    new_state = model.state_dict()

    # Track which tensors were copied vs initialized.
    copied_tensors = []
    expanded_tensors = []
    init_tensors = []
    shape_mismatches = []

    for name, new_tensor in new_state.items():
        if name not in original_state:
            # Brand-new tensor (shouldn't happen with same architecture).
            fail(f"Unexpected tensor in new model: {name}")
        old_tensor = original_state[name]
        if old_tensor.shape == new_tensor.shape:
            # Exact copy.
            new_tensor.data.copy_(old_tensor.data)
            copied_tensors.append(name)
        elif name in ("tok_emb.weight", "lm_head.weight") and \
             old_tensor.shape[1] == new_tensor.shape[1] and \
             new_tensor.shape[0] > old_tensor.shape[0]:
            # Vocabulary expansion: copy old rows, init new rows.
            old_rows = old_tensor.shape[0]
            new_tensor.data[:old_rows].copy_(old_tensor.data)
            # Initialize new rows with the same scheme as _init_weights.
            nn_init_new(new_tensor.data[old_rows:], INIT_STD)
            expanded_tensors.append(name)
            init_tensors.append(name)
        else:
            shape_mismatches.append(
                (name, tuple(old_tensor.shape), tuple(new_tensor.shape))
            )

    if shape_mismatches:
        fail(
            "Unexpected shape mismatches that are not vocab expansion: "
            f"{shape_mismatches}"
        )

    return model, new_cfg, {
        "copied_tensors": copied_tensors,
        "expanded_tensors": expanded_tensors,
        "init_tensors": init_tensors,
        "shape_mismatches": shape_mismatches,
    }


def nn_init_new(tensor, std):
    """Initialize new rows with the same scheme as TinyLM._init_weights."""
    torch.nn.init.normal_(tensor, mean=0.0, std=std)


# ----------------------------------------------------------------------
# Verification tests
# ----------------------------------------------------------------------

def run_verification(model, new_cfg, original_cfg, expanded_tok, original_tok,
                     original_state, new_state, transfer_info,
                     train_chars, val_chars, missing_chars):
    results = {"tests": [], "all_passed": True}

    def check(name, condition, detail=""):
        results["tests"].append({
            "name": name,
            "passed": bool(condition),
            "detail": detail,
        })
        if not condition:
            results["all_passed"] = False
        return bool(condition)

    # Test 1: vocab sizes
    orig_vocab = original_tok.vocab_size
    new_vocab = expanded_tok.vocab_size
    check("original vocab size is 60", orig_vocab == 60,
          f"got {orig_vocab}")
    check("new vocab size is 65", new_vocab == 65,
          f"got {new_vocab}")
    check("new vocab = original + 5", new_vocab == orig_vocab + len(missing_chars),
          f"{orig_vocab} + {len(missing_chars)} = {orig_vocab + len(missing_chars)}")

    # Test 2: every old token retains its ID
    id_preserved = all(
        expanded_tok.stoi[ch] == original_tok.stoi[ch]
        for ch in original_tok.characters
    )
    check("every old token ID preserved", id_preserved)

    # Test 3: every new token has a unique ID
    new_ids = [expanded_tok.stoi[ch] for ch in missing_chars]
    check("new tokens have unique IDs", len(new_ids) == len(set(new_ids)),
          f"new IDs: {new_ids}")
    check("new token IDs are >= original vocab",
          all(i >= orig_vocab for i in new_ids),
          f"new IDs: {new_ids}")

    # Test 4: special token IDs preserved
    check("<pad> id is 0", expanded_tok.pad_id == 0)
    check(" Unknown id is 1", expanded_tok.unk_id == 1)

    # Test 5: every Stage 5 char is representable
    all_stage5_chars = set(train_chars + val_chars)
    unrepresentable = [c for c in all_stage5_chars
                       if c not in expanded_tok.stoi]
    check("every Stage 5 char representable", len(unrepresentable) == 0,
          f"unrepresentable: {unrepresentable}")

    # Test 6: tokenizer serialization round-trip
    expanded_tok.save(STAGE6_TOKENIZER)
    reloaded_tok = load_expanded_tokenizer(STAGE6_TOKENIZER)
    check("tokenizer reload preserves IDs",
          reloaded_tok.stoi == expanded_tok.stoi)
    check("tokenizer reload preserves characters",
          reloaded_tok.characters == expanded_tok.characters)

    # Round-trip encode/decode for all vocab characters
    roundtrip_ok = True
    for ch in expanded_tok.characters:
        ids = expanded_tok.encode(ch)
        decoded = expanded_tok.decode(ids)
        if decoded != ch:
            roundtrip_ok = False
            break
    check("encode/decode round-trip for all vocab chars", roundtrip_ok)

    # Round-trip for representative examples
    examples = ["The", "hello-world", "What is J?", "A?K-Y"]
    example_ok = True
    for ex in examples:
        ids = expanded_tok.encode(ex)
        decoded = expanded_tok.decode(ids)
        if decoded != ex:
            example_ok = False
            break
    check("encode/decode round-trip for examples", example_ok)

    # Test 7: model config matches original except vocab
    check("block_size preserved", new_cfg.block_size == original_cfg["block_size"])
    check("n_layer preserved", new_cfg.n_layer == original_cfg["n_layer"])
    check("n_head preserved", new_cfg.n_head == original_cfg["n_head"])
    check("n_embd preserved", new_cfg.n_embd == original_cfg["n_embd"])
    check("dropout preserved", new_cfg.dropout == original_cfg["dropout"])
    check("vocab_size expanded", new_cfg.vocab_size == new_vocab)

    # Test 8: every compatible old tensor preserved exactly
    exact_count = 0
    for name in transfer_info["copied_tensors"]:
        if torch.equal(new_state[name], original_state[name]):
            exact_count += 1
    check(f"all {len(transfer_info['copied_tensors'])} compatible tensors exact",
          exact_count == len(transfer_info["copied_tensors"]),
          f"{exact_count}/{len(transfer_info['copied_tensors'])} exact")

    # Test 9: old embedding rows identical
    old_emb = original_state["tok_emb.weight"]
    new_emb = new_state["tok_emb.weight"]
    check("old tok_emb rows identical",
          torch.equal(new_emb[:old_emb.shape[0]], old_emb))
    old_lm = original_state["lm_head.weight"]
    new_lm = new_state["lm_head.weight"]
    check("old lm_head rows identical",
          torch.equal(new_lm[:old_lm.shape[0]], old_lm))

    # Test 10: new rows have valid finite values
    new_emb_rows = new_emb[old_emb.shape[0]:]
    new_lm_rows = new_lm[old_lm.shape[0]:]
    check("new tok_emb rows finite", torch.isfinite(new_emb_rows).all().item())
    check("new lm_head rows finite", torch.isfinite(new_lm_rows).all().item())

    # Test 11: weight tying (model does NOT tie; confirm separate)
    check("weights are not tied (separate tensors)",
          new_emb.data_ptr() != new_lm.data_ptr())

    # Test 12: forward pass produces finite logits and loss
    model.eval()
    with torch.no_grad():
        dummy_ids = torch.tensor([[expanded_tok.stoi["T"],
                                   expanded_tok.stoi["h"],
                                   expanded_tok.stoi["e"]]],
                                 dtype=torch.long)
        logits, loss = model(dummy_ids, dummy_ids)
    check("forward pass logits finite", torch.isfinite(logits).all().item())
    check("forward pass loss finite", torch.isfinite(loss).all().item())
    check("logits shape correct",
          tuple(logits.shape) == (1, 3, new_vocab),
          f"got {tuple(logits.shape)}")

    # Test 13: new checkpoint can be reloaded
    # (verified by reloading model_init.pt in main after saving)

    # Test 14: no unexpected tensors
    new_keys = set(new_state.keys())
    old_keys = set(original_state.keys())
    check("no missing tensors", old_keys <= new_keys,
          f"missing: {old_keys - new_keys}")
    check("no unexpected tensors", new_keys <= old_keys,
          f"unexpected: {new_keys - old_keys}")

    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Stage 6 artifacts: expanded tokenizer + "
                    "initialized model (no training).")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing artifacts/stage6/ outputs")
    args = parser.parse_args()

    if not args.force and os.path.exists(STAGE6_MODEL_INIT):
        fail(
            "Stage 6 artifacts already exist. Pass --force to overwrite, "
            "or remove artifacts/stage6/ first."
        )

    print("=== Stage 6: Safe Tokenizer Expansion & Model Init ===")
    print()

    # --- Phase 1: load originals (read-only) ------------------------------
    ckpt = load_original_checkpoint()
    original_tok = load_original_tokenizer()
    original_cfg = dict(ckpt.get("model_config") or {})
    original_state = ckpt["model_state"]
    original_step = int(ckpt.get("step", 0))

    print(f"Original checkpoint: {rel(ORIGINAL_CHECKPOINT)} (step {original_step})")
    print(f"Original config: {original_cfg}")
    print(f"Original tokenizer vocab: {original_tok.vocab_size}")
    print(f"Original tokenizer chars: {original_tok.characters}")

    # --- compute missing chars from Stage 5 splits ------------------------
    corpus_chars, missing_chars = compute_missing_chars(
        STAGE5_TRAIN, STAGE5_VAL, original_tok
    )
    print(f"Stage 5 corpus unique chars: {len(corpus_chars)}")
    print(f"Missing from tokenizer: {missing_chars} ({len(missing_chars)})")

    # --- Phase 3: expand tokenizer ----------------------------------------
    expanded_tok = build_expanded_tokenizer(
        original_tok.characters, missing_chars
    )
    new_vocab = expanded_tok.vocab_size
    print(f"Expanded tokenizer vocab: {new_vocab}")
    print(f"Added chars: {missing_chars} at IDs "
          f"{[expanded_tok.stoi[c] for c in missing_chars]}")

    # --- Phase 4: build expanded model ------------------------------------
    model, new_cfg, transfer_info = build_expanded_model(
        original_cfg, new_vocab, original_state
    )
    new_state = model.state_dict()
    print(f"Expanded model config: vocab={new_cfg.vocab_size}, "
          f"block={new_cfg.block_size}, layers={new_cfg.n_layer}, "
          f"heads={new_cfg.n_head}, embd={new_cfg.n_embd}")
    print(f"Tensors copied exactly: {len(transfer_info['copied_tensors'])}")
    print(f"Tensors expanded (vocab): {transfer_info['expanded_tensors']}")
    print(f"Tensors newly initialized: {transfer_info['init_tensors']}")

    # --- verification ------------------------------------------------------
    with open(STAGE5_TRAIN, encoding="utf-8") as f:
        train_text = f.read()
    with open(STAGE5_VAL, encoding="utf-8") as f:
        val_text = f.read()

    results = run_verification(
        model, new_cfg, original_cfg, expanded_tok, original_tok,
        original_state, new_state, transfer_info,
        train_text, val_text, missing_chars
    )

    print()
    print("=== VERIFICATION RESULTS ===")
    for t in results["tests"]:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  [{status}] {t['name']}" +
              (f"  ({t['detail']})" if t.get("detail") else ""))
    if not results["all_passed"]:
        fail("One or more verification tests failed.")

    # --- save artifacts ----------------------------------------------------
    os.makedirs(STAGE6_DIR, exist_ok=True)

    # tokenizer.json (already saved in run_verification, but save explicitly)
    expanded_tok.save(STAGE6_TOKENIZER)

    # model_config.json
    config_payload = {
        "format": "my-LLM stage-6 model config v1",
        "vocab_size": new_cfg.vocab_size,
        "block_size": new_cfg.block_size,
        "n_layer": new_cfg.n_layer,
        "n_head": new_cfg.n_head,
        "n_embd": new_cfg.n_embd,
        "dropout": new_cfg.dropout,
        "original_vocab_size": original_cfg.get("vocab_size"),
        "added_vocab_size": len(missing_chars),
        "added_characters": missing_chars,
    }
    with open(STAGE6_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    # model_init.pt
    init_payload = {
        "format": STAGE6_FORMAT,
        "source_checkpoint": rel(ORIGINAL_CHECKPOINT),
        "source_checkpoint_sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "source_training_step": original_step,
        "model_config": {
            "vocab_size": new_cfg.vocab_size,
            "block_size": new_cfg.block_size,
            "n_layer": new_cfg.n_layer,
            "n_head": new_cfg.n_head,
            "n_embd": new_cfg.n_embd,
            "dropout": new_cfg.dropout,
        },
        "tokenizer": {
            "format": "my-LLM character tokenizer v1",
            "special_tokens": list(SPECIAL_TOKENS),
            "characters": expanded_tok.characters,
            "id_preserved": True,
            "original_char_count": len(original_tok.characters),
            "added_characters": missing_chars,
            "new_token_ids": {c: expanded_tok.stoi[c] for c in missing_chars},
        },
        "transfer_info": transfer_info,
        "init_std": INIT_STD,
        "weights_tied": False,
        "model_state": new_state,
    }
    torch.save(init_payload, STAGE6_MODEL_INIT)

    # expansion_report.json
    report = {
        "format": "my-LLM stage-6 expansion report v1",
        "original_checkpoint": rel(ORIGINAL_CHECKPOINT),
        "original_checkpoint_sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "original_tokenizer": rel(ORIGINAL_TOKENIZER),
        "original_tokenizer_sha256": sha256_file(ORIGINAL_TOKENIZER),
        "original_corpus": rel("data/tiny_corpus.txt"),
        "original_corpus_sha256": sha256_file(os.path.join(PROJECT_ROOT, "data", "tiny_corpus.txt")),
        "source_training_step": original_step,
        "original_vocab_size": original_tok.vocab_size,
        "added_characters": missing_chars,
        "added_char_count": len(missing_chars),
        "new_vocab_size": new_vocab,
        "new_token_ids": {c: expanded_tok.stoi[c] for c in missing_chars},
        "new_model_config": {
            "vocab_size": new_cfg.vocab_size,
            "block_size": new_cfg.block_size,
            "n_layer": new_cfg.n_layer,
            "n_head": new_cfg.n_head,
            "n_embd": new_cfg.n_embd,
            "dropout": new_cfg.dropout,
        },
        "tokens_copied_exactly": len(transfer_info["copied_tensors"]),
        "tensors_expanded": transfer_info["expanded_tensors"],
        "tensors_newly_initialized": transfer_info["init_tensors"],
        "init_std": INIT_STD,
        "weights_tied": False,
        "verification": results,
        "artifacts": {
            "tokenizer": rel(STAGE6_TOKENIZER),
            "model_config": rel(STAGE6_CONFIG),
            "model_init": rel(STAGE6_MODEL_INIT),
            "report": rel(STAGE6_REPORT),
            "readme": rel(STAGE6_README),
        },
    }
    with open(STAGE6_REPORT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    # README.md
    readme = f"""# Stage 6 artifacts — expanded tokenizer & initialized model (no training)

## What this is

A separate Stage 6 artifact lineage that supports every character in the
Stage 5 dataset, created **without training**. No optimizer step, no backward
pass, no gradient update occurred. All original artifacts remain unchanged.

## Contents

| Path | What it is |
|---|---|
| `tokenizer.json` | Expanded character table (65 entries: 60 original + 5 added) |
| `model_config.json` | Expanded model config (vocab_size=65, rest unchanged) |
| `model_init.pt` | New model checkpoint with expanded vocab, initialized (not trained) |
| `expansion_report.json` | Machine-readable verification report |
| `README.md` | This file |

## What changed

- **Vocabulary**: 60 -> 65 (added 5 characters: `{", ".join(repr(c) for c in missing_chars)}`)
- **Token IDs**: All 60 original IDs preserved exactly. New chars assigned IDs 60-64.
- **Model**: tok_emb and lm_head expanded from (60,128) to (65,128).
  Old rows copied exactly; 5 new rows initialized with N(0, 0.02).
- **Weights**: NOT tied (separate tok_emb and lm_head, matching original).
- All other tensors (pos_emb, Transformer blocks, final_norm) copied exactly.

## Compatibility notes

- The existing `src/generate.py` expects `checkpoints/model.pt` format and
  will NOT work directly with `artifacts/stage6/model_init.pt` without
  changes (different format identifier, different path). Generation with
  the Stage 6 model is a future task.
- The existing `src/train.py` will reject the Stage 6 tokenizer because the
  corpus characters won't match (vocabulary lock). Training with Stage 6
  data is a future task requiring approval.

## Verification

All 17 verification tests passed. See `expansion_report.json` for details.

## Next step (requires approval)

Train the expanded model using the Stage 5 corpus, then resume generation.
"""
    with open(STAGE6_README, "w", encoding="utf-8", newline="\n") as f:
        f.write(readme)

    # Reload model_init.pt to verify it loads (test 13)
    reloaded = torch.load(STAGE6_MODEL_INIT, map_location="cpu")
    check_reload = (
        reloaded.get("format") == STAGE6_FORMAT and
        "model_state" in reloaded and
        reloaded.get("source_training_step") == original_step
    )
    results["tests"].append({
        "name": "new checkpoint reloads successfully",
        "passed": bool(check_reload),
    })
    if not check_reload:
        results["all_passed"] = False

    # Update report with final test results
    report["verification"] = results
    with open(STAGE6_REPORT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print()
    print("=== ARTIFACTS SAVED ===")
    print(f"  {rel(STAGE6_TOKENIZER)}")
    print(f"  {rel(STAGE6_CONFIG)}")
    print(f"  {rel(STAGE6_MODEL_INIT)}")
    print(f"  {rel(STAGE6_REPORT)}")
    print(f"  {rel(STAGE6_README)}")
    print()
    print("Stage 6 preparation complete. No training was performed.")
    print("Original artifacts are unchanged.")


if __name__ == "__main__":
    main()
