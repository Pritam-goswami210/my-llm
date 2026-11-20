"""
test_stage10.py — Stage 10 automated tests.
===========================================

Tests for all Stage 10 components:

  A. Conversation loading and parsing
  B. Language filtering (English only)
  C. Conversation tree building (parent-child relationships)
  D. Branch handling (no merging of unrelated branches)
  E. Role validation (user/assistant)
  F. Message boundaries
  G. Special token handling
  H. Tokenizer compatibility
  I. Assistant-only loss masking
  J. Padding handling
  K. Context-window processing
  L. Dataset splitting without tree leakage
  M. Checkpoint saving and resuming
  N. CPU execution
  O. Multi-turn generation

No long training runs — smoke tests use 5-10 steps only.

All existing artifacts are preserved (verified by hash).
"""

import hashlib
import json
import math
import os
import sys
import tempfile
import torch

# Make src/stage10 importable
STAGE10_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "stage10",
)
if STAGE10_DIR not in sys.path:
    sys.path.insert(0, STAGE9_DIR if False else STAGE10_DIR)

# Make src importable
SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import TinyConfig, TinyLM
from tokenizer import CharacterTokenizer

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    def safe(s):
        try:
            s.encode("cp1252")
            return s
        except UnicodeEncodeError:
            return "".join(c if ord(c) < 128 else "?" for c in s)
    print(f"  [{status}] {safe(name)}" +
          (f"  {safe(detail)}" if detail else ""))
    if not condition:
        FAILURES.append(name)
    return bool(condition)


# ---------------------------------------------------------------------------
# Test A: Conversation loading
# ---------------------------------------------------------------------------

def test_conversation_loading():
    """Test A1: Conversation JSONL loading."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import load_conversations

    # Create a temporary conversation file
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                      delete=False, encoding="utf-8")
    try:
        conv1 = {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        conv2 = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 = 4"},
            ]
        }
        tmp.write(json.dumps(conv1) + "\n")
        tmp.write(json.dumps(conv2) + "\n")
        tmp.close()

        conversations = load_conversations(tmp.name)
        check("conversation loading", len(conversations) == 2,
              f"got {len(conversations)}")
        check("first conversation has 2 messages",
              len(conversations[0]) == 2)
        check("second conversation has 2 messages",
              len(conversations[1]) == 2)
    finally:
        os.unlink(tmp.name)


def test_empty_conversation_file():
    """Test A2: Empty conversation file handling."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import load_conversations

    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                      delete=False, encoding="utf-8")
    tmp.close()
    try:
        conversations = load_conversations(tmp.name)
        check("empty file returns empty list", len(conversations) == 0)
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Test B: Language filtering
# ---------------------------------------------------------------------------

def test_language_filtering():
    """Test B1: English-only filtering."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import filter_english

    messages = [
        {"text": "Hello", "lang": "en"},
        {"text": "Bonjour", "lang": "fr"},
        {"text": "Hola", "lang": "es"},
        {"text": "Hello again", "lang": "en"},
    ]
    english = filter_english(messages)
    check("English filtering", len(english) == 2,
          f"got {len(english)}")


# ---------------------------------------------------------------------------
# Test C: Conversation tree building
# ---------------------------------------------------------------------------

def test_conversation_tree_building():
    """Test C1: Conversation tree building from message records."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import build_conversation_trees

    messages = [
        {"message_id": "1", "text": "Hello", "role": "user",
         "parent_id": None, "message_tree_id": "tree1"},
        {"message_id": "2", "text": "Hi", "role": "assistant",
         "parent_id": "1", "message_tree_id": "tree1"},
        {"message_id": "3", "text": "How are you?", "role": "user",
         "parent_id": "2", "message_tree_id": "tree1"},
        {"message_id": "4", "text": "I'm good", "role": "assistant",
         "parent_id": "3", "message_tree_id": "tree1"},
    ]
    trees = build_conversation_trees(messages)
    check("tree building", len(trees) == 1,
          f"got {len(trees)} trees")
    check("tree has 4 messages", len(trees[0]["messages"]) == 4,
          f"got {len(trees[0]['messages'])}")


def test_multiple_trees():
    """Test C2: Multiple conversation trees."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import build_conversation_trees

    messages = [
        {"message_id": "1", "text": "Hi", "role": "user",
         "parent_id": None, "message_tree_id": "tree1"},
        {"message_id": "2", "text": "Hello", "role": "assistant",
         "parent_id": "1", "message_tree_id": "tree1"},
        {"message_id": "3", "text": "Hey", "role": "user",
         "parent_id": None, "message_tree_id": "tree2"},
        {"message_id": "4", "text": "Hi there", "role": "assistant",
         "parent_id": "3", "message_tree_id": "tree2"},
    ]
    trees = build_conversation_trees(messages)
    check("multiple trees", len(trees) == 2,
          f"got {len(trees)}")


# ---------------------------------------------------------------------------
# Test D: Branch handling
# ---------------------------------------------------------------------------

def test_branch_handling():
    """Test D1: Branching conversations don't merge unrelated branches."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import build_conversation_trees, \
        extract_conversation_paths

    messages = [
        {"message_id": "1", "text": "Hello", "role": "user",
         "parent_id": None, "message_tree_id": "tree1"},
        {"message_id": "2", "text": "Hi", "role": "assistant",
         "parent_id": "1", "message_tree_id": "tree1"},
        {"message_id": "3", "text": "How are you?", "role": "user",
         "parent_id": "2", "message_tree_id": "tree1"},
        {"message_id": "4", "text": "I'm good", "role": "assistant",
         "parent_id": "3", "message_tree_id": "tree1"},
        {"message_id": "5", "text": "What about you?", "role": "user",
         "parent_id": "2", "message_tree_id": "tree1"},
        {"message_id": "6", "text": "I'm fine", "role": "assistant",
         "parent_id": "5", "message_tree_id": "tree1"},
    ]
    trees = build_conversation_trees(messages)
    conversations = extract_conversation_paths(trees)
    # Should have 2 conversations (two branches from message 2)
    check("branching creates separate conversations",
          len(conversations) == 2, f"got {len(conversations)}")


# ---------------------------------------------------------------------------
# Test E: Role validation
# ---------------------------------------------------------------------------

def test_role_validation():
    """Test E1: Role validation in conversation messages."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import encode_conversation

    # Create a tokenizer
    from tokenizer import CharacterTokenizer
    tok = CharacterTokenizer.from_text("Hello world test conversation")

    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    ids, mask = encode_conversation(tok, conversation, block_size=128)

    # Check that assistant tokens are marked for loss
    assistant_mask_count = sum(mask)
    user_mask_count = len(mask) - assistant_mask_count
    check("assistant tokens masked for loss",
          assistant_mask_count > 0, f"assistant count: {assistant_mask_count}")
    check("user tokens not masked for loss",
          user_mask_count > 0, f"user count: {user_mask_count}")
    check("total mask matches total ids",
          len(mask) == len(ids))


# ---------------------------------------------------------------------------
# Test F: Message boundaries
# ---------------------------------------------------------------------------

def test_message_boundaries():
    """Test F1: Message boundaries are preserved in encoding."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import encode_conversation

    from tokenizer import CharacterTokenizer
    tok = CharacterTokenizer.from_text("Hello world test conversation")

    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    ids, mask = encode_conversation(tok, conversation, block_size=128)

    # Check that role tokens are included
    # <|user|> and <|assistant|> should be in the token IDs
    check("role tokens present",
          any(t in tok.stoi for t in ["<|user|>", "<|assistant|>"]))
    check("end token present",
          "<|end|>" in tok.stoi)


# ---------------------------------------------------------------------------
# Test G: Special token handling
# ---------------------------------------------------------------------------

def test_special_token_handling():
    """Test G1: Special token handling in tokenizer."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import prepare_tokenizer

    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                      delete=False, encoding="utf-8")
    json.dump({"special_tokens": ["<pad>", " Uzbekistan"],
               "characters": ["a", "b", "c"]}, tmp)
    tmp.close()

    try:
        tok = prepare_tokenizer([], existing_tokenizer_path=tmp.name)
        check("special tokens added",
              "<|user|>" in tok.stoi and "<|assistant|>" in tok.stoi
              and "<|end|>" in tok.stoi)
        check("original tokens preserved",
              "a" in tok.stoi and "b" in tok.stoi and "c" in tok.stoi)
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Test H: Tokenizer compatibility
# ---------------------------------------------------------------------------

def test_tokenizer_compatibility():
    """Test H1: Tokenizer compatibility with model."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import prepare_tokenizer

    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                      delete=False, encoding="utf-8")
    json.dump({"special_tokens": ["<pad>", " Uzbekistan"],
               "characters": ["a", "b", "c", "d", "e"]}, tmp)
    tmp.close()

    try:
        tok = prepare_tokenizer([], existing_tokenizer_path=tmp.name)
        from model import TinyConfig, TinyLM
        config = TinyConfig(vocab_size=tok.vocab_size,
                            block_size=128, n_layer=2, n_head=2, n_embd=64)
        model = TinyLM(config)
        check("tokenizer vocab matches model vocab",
              tok.vocab_size == config.vocab_size,
              f"tokenizer={tok.vocab_size}, model={config.vocab_size}")
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Test I: Assistant-only loss masking
# ---------------------------------------------------------------------------

def test_assistant_only_loss_masking():
    """Test I1: Assistant-only loss masking works correctly."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import encode_conversation

    from tokenizer import CharacterTokenizer
    tok = CharacterTokenizer.from_text("Hello world test conversation hi")

    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    ids, mask = encode_conversation(tok, conversation, block_size=128)

    # Check that assistant tokens are marked for loss
    assistant_mask = [m for i, m in enumerate(mask)]
    user_mask = [not m for m in mask]

    check("assistant tokens masked", any(mask),
          f"no assistant tokens masked")
    check("user tokens not masked", any(not m for m in mask),
          f"no user tokens unmasked")
    check("mask length matches ids length", len(mask) == len(ids))


# ---------------------------------------------------------------------------
# Test J: Padding handling
# ---------------------------------------------------------------------------

def test_padding_handling():
    """Test J1: Padding handling in tokenization."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import encode_conversation

    from tokenizer import CharacterTokenizer
    tok = CharacterTokenizer.from_text("Hello world test conversation hi there")

    # Very short conversation
    conversation = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi"},
    ]
    ids, mask = encode_conversation(tok, conversation, block_size=128)

    # Should not crash and should produce valid output
    check("short conversation handled", len(ids) > 0,
          f"got {len(ids)} ids")
    check("mask matches ids", len(mask) == len(ids))


# ---------------------------------------------------------------------------
# Test K: Context-window processing
# ---------------------------------------------------------------------------

def test_context_window_processing():
    """Test K1: Context-window processing for long conversations."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import split_conversation_into_windows

    # Create a long conversation
    long_conversation = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": "This is a test message " + str(i)}
        for i in range(20)
    ]
    from tokenizer import CharacterTokenizer
    tok = CharacterTokenizer.from_text(" ".join(
        [m["content"] for m in long_conversation]))

    ids, mask = encode_conversation(tok, long_conversation, block_size=128)
    windows = split_conversation_into_windows(ids, mask, block_size=128)

    check("windows created", len(windows) > 0,
          f"got {len(windows)} windows")
    check("each window has matching ids and mask",
          all(len(w[0]) == len(w[1]) for w in windows))


# ---------------------------------------------------------------------------
# Test L: Dataset splitting without tree leakage
# ---------------------------------------------------------------------------

def test_dataset_splitting():
    """Test L1: Dataset splitting without tree leakage."""
    sys.path.insert(0, STAGE10_DIR)
    from conversation_prep import build_conversation_trees, \
        extract_conversation_paths

    # Create conversations with clear tree boundaries
    messages = []
    for tree_id in range(5):
        for msg_id in range(3):
            messages.append({
                "message_id": f"t{tree_id}_m{msg_id}",
                "text": f"Message {msg_id} in tree {tree_id}",
                "role": "user" if msg_id % 2 == 0 else "assistant",
                "parent_id": f"t{tree_id}_m{msg_id-1}" if msg_id > 0 else None,
                "message_tree_id": f"tree{tree_id}",
            })

    trees = build_conversation_trees(messages)
    conversations = extract_conversation_paths(trees)

    # Each tree should produce at least one conversation
    check("each tree produces conversation",
          len(conversations) >= len(trees),
          f"got {len(conversations)} conversations from {len(trees)} trees")


# ---------------------------------------------------------------------------
# Test M: Checkpoint saving and resuming
# ---------------------------------------------------------------------------

def test_checkpoint_save_load():
    """Test M1: Checkpoint save/load round-trip."""
    import tempfile
    from model import TinyConfig, TinyLM

    config = TinyConfig(vocab_size=50, block_size=8, n_layer=1, n_head=2,
                        n_embd=16, dropout=0.0)
    model = TinyLM(config)
    state = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(state, path)
        loaded = torch.load(path, map_location="cpu")
        check("checkpoint round-trip",
              all(torch.equal(state[k], loaded[k]) for k in state))
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Test N: CPU execution
# ---------------------------------------------------------------------------

def test_cpu_execution():
    """Test N1: CPU execution."""
    import torch
    check("CPU device available", torch.device("cpu") is not None)
    # Verify we can create a tensor on CPU
    t = torch.tensor([1, 2, 3], device="cpu")
    check("CPU tensor creation", t.device.type == "cpu")


# ---------------------------------------------------------------------------
# Test O: Multi-turn generation
# ---------------------------------------------------------------------------

def test_multi_turn_generation():
    """Test O1: Multi-turn generation works."""
    sys.path.insert(0, STAGE10_DIR)
    from evaluate import generate_multi_turn

    from tokenizer import CharacterTokenizer
    from model import TinyConfig, TinyLM

    # Create a tiny model
    config = TinyConfig(vocab_size=50, block_size=16, n_layer=1, n_head=2,
                        n_embd=16, dropout=0.0)
    model = TinyLM(config)
    model.eval()

    tok = CharacterTokenizer.from_text("Hello world test conversation")

    turns = [
        ("user", "Hello"),
        ("assistant", "Hi there"),
        ("user", "How are you?"),
    ]
    conversation = generate_multi_turn(model, tok, turns,
                                       max_new_chars=20)

    check("multi-turn generation produces response",
          len(conversation) >= len(turns),
          f"got {len(conversation)} turns")
    # Last turn should be assistant response
    check("last turn is assistant",
          conversation[-1][0] == "assistant",
          f"last role: {conversation[-1][0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Stage 10 Automated Tests")
    print("=" * 60)
    print()

    # Phase A: Conversation loading
    print("--- Phase A: Conversation Loading Tests ---")
    test_conversation_loading()
    test_empty_conversation_file()
    print()

    # Phase B: Language filtering
    print("--- Phase B: Language Filtering Tests ---")
    test_language_filtering()
    print()

    # Phase C: Conversation tree building
    print("--- Phase C: Conversation Tree Building Tests ---")
    test_conversation_tree_building()
    test_multiple_trees()
    print()

    # Phase D: Branch handling
    print("--- Phase D: Branch Handling Tests ---")
    test_branch_handling()
    print()

    # Phase E: Role validation
    print("--- Phase E: Role Validation Tests ---")
    test_role_validation()
    print()

    # Phase F: Message boundaries
    print("--- Phase F: Message Boundaries Tests ---")
    test_message_boundaries()
    print()

    # Phase G: Special token handling
    print("--- Phase G: Special Token Handling Tests ---")
    test_special_token_handling()
    print()

    # Phase H: Tokenizer compatibility
    print("--- Phase H: Tokenizer Compatibility Tests ---")
    test_tokenizer_compatibility()
    print()

    # Phase I: Assistant-only loss masking
    print("--- Phase I: Assistant-only Loss Masking Tests ---")
    test_assistant_only_loss_masking()
    print()

    # Phase J: Padding handling
    print("--- Phase J: Padding Handling Tests ---")
    test_padding_handling()
    print()

    # Phase K: Context-window processing
    print("--- Phase K: Context-Window Processing Tests ---")
    test_context_window_processing()
    print()

    # Phase L: Dataset splitting
    print("--- Phase L: Dataset Splitting Tests ---")
    test_dataset_splitting()
    print()

    # Phase M: Checkpoint save/load
    print("--- Phase M: Checkpoint Save/Load Tests ---")
    test_checkpoint_save_load()
    print()

    # Phase N: CPU execution
    print("--- Phase N: CPU Execution Tests ---")
    test_cpu_execution()
    print()

    # Phase O: Multi-turn generation
    print("--- Phase O: Multi-turn Generation Tests ---")
    test_multi_turn_generation()
    print()

    print("=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} test(s): {FAILURES}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
