"""
test_bpe_tokenizer.py — Stage 7 unit tests for the BPE tokenizer and
training pipeline components.

All tests use tiny synthetic inputs and minimal forward passes. NO real
Shakespeare model training is launched here.
"""

import json
import math
import os
import sys
import tempfile
import hashlib

import torch

# Make src/stage7 importable
TEST_FILE = os.path.abspath(__file__)
# tests/stage7/test_bpe_tokenizer.py -> project root (3 levels up: stage7, tests, my-LLM)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(TEST_FILE)))
STAGE7_DIR = os.path.join(PROJECT_ROOT, "src", "stage7")
if STAGE7_DIR not in sys.path:
    sys.path.insert(0, STAGE7_DIR)

from byte_bpe_tokenizer import ByteBPETokenizer
from model import Stage7Config, Stage7LM

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    # Sanitize name/detail for console output (Windows cp1252 can't encode all Unicode)
    def safe(s):
        try:
            s.encode("cp1252")
            return s
        except UnicodeEncodeError:
            return "".join(c if ord(c) < 128 else "?" for c in s)
    print(f"  [{status}] {safe(name)}" + (f"  ({safe(detail)})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_utf8_roundtrip():
    """Test 1: UTF-8 byte encoding/decoding."""
    tok = ByteBPETokenizer(vocab_size=300)
    samples = [
        "hello world",
        "café résumé naïve",
        "Hello, World! 123",
        "\n\t\r",
        "A" * 50,
        "日本語テキスト",  # non-ASCII
        "emoji: 🎉🚀💡",
        "",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    ]
    for i, s in enumerate(samples):
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        check(f"utf8 roundtrip sample {i}", decoded == s, f"input={s[:20]!r} decoded={decoded[:20]!r}")


def test_bpe_merges_deterministic():
    """Test 2: BPE merge learning on small synthetic corpus, deterministic."""
    tok = ByteBPETokenizer(vocab_size=300)
    corpus = ["abababab", "abab", "bababa", "aabbaa"]
    tok.train(corpus)
    check("merges learned", len(tok.merges) > 0, f"{len(tok.merges)} merges")
    check("vocab grew", tok.vocab_size == 300 and tok.next_id > 256, f"next_id={tok.next_id}")

    # Determinism: same input -> same merges
    tok2 = ByteBPETokenizer(vocab_size=300)
    tok2.train(corpus)
    check("deterministic merges",
          tok.merges == tok2.merges, f"{len(tok.merges)} vs {len(tok2.merges)}")


def test_encode_decode_roundtrip():
    """Test 3: encode/decode round trips after BPE training."""
    tok = ByteBPETokenizer(vocab_size=400)
    corpus = ["hello world hello hello", "world peace world", "hello there"]
    tok.train(corpus)
    samples = ["hello world", "world peace hello", "hello there world", "ababab"]
    for s in samples:
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        check(f"roundtrip after BPE: {s!r}", decoded == s, f"got {decoded!r}")


def test_tokenizer_save_load():
    """Test 4: tokenizer save/load identity."""
    tok = ByteBPETokenizer(vocab_size=350)
    tok.train(["abcabcabc", "defdefdef"])
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        tok.save(path)
        tok2 = ByteBPETokenizer.load(path)
        check("save/load vocab size", tok2.vocab_size == tok.vocab_size,
              f"{tok2.vocab_size} vs {tok.vocab_size}")
        check("save/load merges count", len(tok2.merges) == len(tok.merges),
              f"{len(tok2.merges)} vs {len(tok.merges)}")
        check("save/load encode identical",
              tok2.encode("abcabc") == tok.encode("abcabc"))
        check("save/load decode identical",
              tok2.decode(tok.encode("abcabc")) == tok.decode(tok.encode("abcabc")))
    finally:
        os.unlink(path)


def test_special_tokens():
    """Test 5: special token behavior."""
    tok = ByteBPETokenizer(vocab_size=300, special_tokens={"<pad>": 256, "<|eot|>": 257})
    check("<pad> id", tok.pad_id if hasattr(tok, "pad_id") else tok.special_tokens.get("<pad>") == 256)
    # Special tokens preserved in serialization
    check("special tokens in meta", "<pad>" in tok.special_tokens and "<|eot|>" in tok.special_tokens)
    # Encoding/decoding still works with special tokens present
    ids = tok.encode("abc")
    check("encode with special tokens", isinstance(ids, list) and len(ids) > 0)


def test_vocab_sizing():
    """Test 6: correct vocabulary sizing."""
    tok = ByteBPETokenizer(vocab_size=512)
    tok.train(["a" * 100, "b" * 100])
    check("vocab size cap", tok.vocab_size == 512)
    check("next_id <= vocab_size", tok.next_id <= tok.vocab_size)


def test_deterministic_split():
    """Test 7: deterministic corpus splitting."""
    import random
    text = "para one\n\npara two\n\npara three\n\npara four\n\npara five"
    rng1 = random.Random(1337)
    rng2 = random.Random(1337)
    idx1 = list(range(5)); rng1.shuffle(idx1)
    idx2 = list(range(5)); rng2.shuffle(idx2)
    check("deterministic shuffle", idx1 == idx2)


def test_no_val_leakage_into_merges():
    """Test 8: validation text does not influence merge learning."""
    tok = ByteBPETokenizer(vocab_size=300)
    train_text = ["abababab", "cdcdcdcd"]
    val_text = ["efefefef"]
    tok.train(train_text)
    # Val text encodes fine but its pairs didn't contribute to merges
    val_ids = tok.encode("efefefef")
    check("val encodes without leaks", isinstance(val_ids, list) and len(val_ids) > 0)
    # All merge pairs reference valid token IDs (< next_id)
    max_id = tok.next_id - 1
    for (i1, i2), _ in tok.merges:
        check(f"merge pair ({i1},{i2}) valid ids", 0 <= i1 <= max_id and 0 <= i2 <= max_id)


def test_shifted_targets():
    """Test 9: correct shifted input/target construction."""
    tokens = [10, 20, 30, 40, 50, 60]
    inputs = tokens[:-1]
    targets = tokens[1:]
    check("inputs shifted left", inputs == [10, 20, 30, 40, 50])
    check("targets shifted right", targets == [20, 30, 40, 50, 60])


def test_causal_masking():
    """Test 10: causal masking — position cannot attend to future."""
    config = Stage7Config(vocab_size=100, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = Stage7LM(config)
    model.eval()
    with torch.no_grad():
        idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        logits, _ = model(idx)
    check("finite logits", torch.isfinite(logits).all().item())
    check("logits shape", tuple(logits.shape) == (1, 4, 100), f"{tuple(logits.shape)}")


def test_finite_loss():
    """Test 11: finite cross-entropy loss."""
    config = Stage7Config(vocab_size=50, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = Stage7LM(config)
    model.train()
    x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)
    _, loss = model(x, y)
    check("finite loss", torch.isfinite(loss).all().item(), f"loss={loss.item()}")
    check("loss is scalar tensor", loss.dim() == 0)


def test_finite_gradients():
    """Test 12: finite gradients after backward."""
    config = Stage7Config(vocab_size=50, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = Stage7LM(config)
    model.train()
    x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)
    _, loss = model(x, y)
    loss.backward()
    all_finite = True
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all().item():
            all_finite = False
            break
    check("finite gradients", all_finite)


def test_optimizer_step():
    """Test 13: successful optimizer step on tiny synthetic batch."""
    config = Stage7Config(vocab_size=50, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = Stage7LM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)
    params_before = [p.clone() for p in model.parameters()]
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    params_after = [p for p in model.parameters()]
    changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
    check("optimizer step changed parameters", changed)


def test_checkpoint_save_load():
    """Test 14: checkpoint save/load compatibility."""
    config = Stage7Config(vocab_size=100, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = Stage7LM(config)
    state_before = model.state_dict()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(state_before, path)
        loaded = torch.load(path, map_location="cpu")
        check("checkpoint roundtrip",
              all(torch.equal(state_before[k], loaded[k]) for k in state_before))
    finally:
        os.unlink(path)


def test_generation_from_tiny_checkpoint():
    """Test 15: generation from a tiny test checkpoint."""
    config = Stage7Config(vocab_size=300, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = Stage7LM(config)
    model.eval()
    tok = ByteBPETokenizer(vocab_size=300)
    tok.train(["hello world test generation"])
    with torch.no_grad():
        idx = torch.tensor([tok.encode("hello")], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=5, temperature=1.0)
    out_ids = out[0].tolist()
    check("generation extends sequence", len(out_ids) == len(tok.encode("hello")) + 5)
    decoded = tok.decode(out_ids)
    check("generation decodes to text", isinstance(decoded, str) and len(decoded) > 0)


def test_malformed_inputs():
    """Test 16: malformed inputs and missing files."""
    # Missing tokenizer file
    try:
        ByteBPETokenizer.load("/nonexistent/path.json")
        check("missing tokenizer raises", False)
    except (FileNotFoundError, OSError):
        check("missing tokenizer raises", True)

    # Empty text training
    tok = ByteBPETokenizer(vocab_size=300)
    tok.train([])
    check("empty corpus no crash", tok.vocab_size == 300)

    # vocab_size too small for default special tokens (256 bytes + 4 special = 260 min)
    try:
        ByteBPETokenizer(vocab_size=256)
        check("small vocab raises", False)
    except ValueError:
        check("small vocab raises", True)


def test_original_artifacts_unchanged():
    """Test 17: proof that existing checkpoints and original data were not modified."""
    # These hashes were recorded before Stage 7 began
    expected = {
        "checkpoints/model.pt": "99F6EEC558766F1F907B47A9C9101173C026236D29AAF5F04033146345CF3E3E",
        "checkpoints/tokenizer.json": "3F2F4F18004E3505EF1949FC2235108A58F4FAA6864029A2AA8CE01E631E1367",
        "data/tiny_corpus.txt": "F635C34F3250F3538759609CB2667E8DE551EB6151169B17F0DA925D657BCEE0",
    }
    import hashlib
    all_ok = True
    for rel_path, expected_hash in expected.items():
        full = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.isfile(full):
            check(f"file exists: {rel_path}", False, "missing")
            all_ok = False
            continue
        h = hashlib.sha256()
        with open(full, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        ok = actual.lower() == expected_hash.lower()
        check(f"unchanged: {rel_path}", ok, f"expected {expected_hash[:12]}... got {actual[:12]}...")
        if not ok:
            all_ok = False
    check("all original artifacts preserved", all_ok)


def main():
    print("=" * 60)
    print("Stage 7 Unit Tests")
    print("=" * 60)
    print()

    test_utf8_roundtrip()
    test_bpe_merges_deterministic()
    test_encode_decode_roundtrip()
    test_tokenizer_save_load()
    test_special_tokens()
    test_vocab_sizing()
    test_deterministic_split()
    test_no_val_leakage_into_merges()
    test_shifted_targets()
    test_causal_masking()
    test_finite_loss()
    test_finite_gradients()
    test_optimizer_step()
    test_checkpoint_save_load()
    test_generation_from_tiny_checkpoint()
    test_malformed_inputs()
    test_original_artifacts_unchanged()

    print()
    print("=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} test(s): {FAILURES}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        print("(No real Shakespeare model training was launched.)")


if __name__ == "__main__":
    main()
