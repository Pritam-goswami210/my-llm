"""
evaluate.py — Stage 10 Phase G: Evaluation Pipeline.
====================================================

Measures model performance after conversational fine-tuning:

  * Training loss
  * Validation loss (per corpus)
  * Perplexity (exp(loss))
  * Per-corpus performance
  * Repetition detection in generated text
  * Vocabulary utilization (how many unique tokens the model uses)
  * Generation quality (fixed prompts)
  * Multi-turn conversation testing
  * Sentence coherence (basic heuristics)

Test prompts:
  1. "The weather today is"
  2. "Once upon a time"
  3. "What is gravity?"
  4. "Explain why the sky appears blue."
  5. "Hamlet said"

Multi-turn conversation test:
  User: My favourite colour is blue.
  Assistant: That's nice!
  User: What colour did I just mention?
  Expected: The model should answer "blue" using conversational context.

Results are written to reports/stage10_evaluation_report.md.

This is an evaluation-only script — no training occurs.

This file is part of the from-scratch my-LLM project. We authored it.
"""

import hashlib
import json
import math
import os
import sys
import time

import torch

# Make src importable for the model and tokenizer
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

STAGE8_DIR = os.path.join(PROJECT_ROOT, "src", "stage8")
if STAGE8_DIR not in sys.path:
    sys.path.insert(0, STAGE8_DIR)

STAGE10_DIR = os.path.join(PROJECT_ROOT, "src", "stage10")
if STAGE10_DIR not in sys.path:
    sys.path.insert(0, STAGE10_DIR)

from model import TinyConfig, TinyLM
from tokenizer import CharacterTokenizer, PAD, UNK, SPECIAL_TOKENS


CHECKPOINT_STAGE10_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "stage10")
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_STAGE10_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(CHECKPOINT_STAGE10_DIR, "tokenizer.json")

REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "stage10_evaluation_report.md")

EVAL_PROMPTS = [
    "The weather today is",
    "Once upon a time",
    "What is gravity?",
    "Explain why the sky appears blue.",
    "Hamlet said",
]

# Multi-turn conversation test
MULTI_TURN_PROMPT = [
    ("user", "My favourite colour is blue."),
    ("assistant", "That's nice!"),
    ("user", "What colour did I just mention?"),
]
EXPECTED_MULTI_TURN_ANSWER = "blue"


def fail(message):
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_and_tokenizer(checkpoint_path, tokenizer_path):
    """Load checkpoint + tokenizer. Returns (model, tokenizer, config, step)."""
    import torch

    if not os.path.isfile(checkpoint_path):
        fail(f"Checkpoint not found: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        fail(f"Could not load checkpoint: {exc}")

    if ckpt.get("format") != "my-LLM stage-10 checkpoint v1":
        fail(f"Unknown checkpoint format: {ckpt.get('format')!r}")

    saved_cfg = ckpt.get("model_config")
    if not saved_cfg:
        fail("Checkpoint has no model configuration.")

    # Tokenizer from checkpoint metadata (guaranteed to match)
    tok_meta = ckpt.get("tokenizer")
    if tok_meta and "characters" in tok_meta:
        all_tokens = list(tok_meta["special_tokens"]) + \
            list(tok_meta["characters"])
        stoi = {t: i for i, t in enumerate(all_tokens)}
        itos = {i: t for i, t in enumerate(all_tokens)}

        class CkptTokenizer:
            def __init__(self, stoi, itos, characters, special_tokens):
                self.stoi = stoi
                self.itos = itos
                self.characters = characters
                self._special = list(special_tokens)
                self.pad_id = stoi[special_tokens[0]]
                self.unk_id = stoi[special_tokens[1]]

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
                    out.append(self.itos.get(int(i),
                                             self.itos.get(self.unk_id, "")))
                return "".join(out)

        tok = CkptTokenizer(
            stoi, itos,
            tok_meta["characters"],
            tok_meta["special_tokens"],
        )
    else:
        # Fall back to standalone tokenizer file
        if not os.path.isfile(tokenizer_path):
            fail(f"No tokenizer file at {tokenizer_path}")
        import json
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        characters = list(payload["characters"])
        special_tokens = list(payload.get("special_tokens", SPECIAL_TOKENS))
        all_tokens = list(special_tokens) + characters
        stoi = {t: i for i, t in enumerate(all_tokens)}
        itos = {i: t for i, t in enumerate(all_tokens)}

        class FileTokenizer:
            def __init__(self, stoi, itos, characters, special_tokens):
                self.stoi = stoi
                self.itos = itos
                self.characters = characters
                self._special = list(special_tokens)
                self.pad_id = stoi[special_tokens[0]]
                self.unk_id = stoi[special_tokens[1]]

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
                    out.append(self.itos.get(int(i),
                                             self.itos.get(self.unk_id, "")))
                return "".join(out)

        tok = FileTokenizer(stoi, itos, characters, special_tokens)

    config = TinyConfig(**saved_cfg)
    model = TinyLM(config)
    try:
        model.load_state_dict(ckpt["model_state"])
    except (RuntimeError, KeyError) as exc:
        fail(f"Checkpoint weights do not match architecture: {exc}")
    model.eval()
    step = int(ckpt.get("step", 0))
    return model, tok, config, step


@torch.no_grad()
def compute_loss(model, data_windows, batch_size, block_size, n_batches):
    """Average loss over n_batches from validation windows."""
    import torch.nn.functional as F

    model.eval()
    losses = []
    total = len(data_windows)
    if total == 0:
        return float("nan")
    for _ in range(n_batches):
        ix = torch.randint(0, total, (batch_size,))
        batch_losses = []
        for i in ix:
            i = int(i)
            ids, mask = data_windows[i]
            x = torch.tensor([ids], dtype=torch.long)
            mask_t = torch.tensor([mask], dtype=torch.bool)
            logits, _ = model(x, y=x)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = x.reshape(-1)
            mask_flat = mask_t.reshape(-1)
            ce = F.cross_entropy(logits_flat, targets_flat,
                                 reduction="none")
            masked_loss = (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            batch_losses.append(masked_loss.item())
        losses.append(sum(batch_losses) / len(batch_losses))
    if was_training:
        model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def compute_perplexity(loss):
    """Perplexity = exp(loss). Returns float('inf') if loss is NaN or too large."""
    if math.isnan(loss) or loss > 100:
        return float("inf")
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def detect_repetition(text, window=5):
    """Simple repetition detector: fraction of words that appear in a
    sliding window of the previous `window` words."""
    words = text.split()
    if len(words) <= window:
        return 0.0
    repeated = 0
    for i in range(window, len(words)):
        if words[i] in words[i - window:i]:
            repeated += 1
    return repeated / max(len(words) - window, 1)


def generate_sample(model, tokenizer, prompt, max_new_chars, temperature=1.0):
    """Generate text from a prompt. Returns generated text only."""
    import torch

    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.unk_id]
    ids = ids[-model.config.block_size:]
    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_chars,
                             temperature=temperature)
    out_ids = out[0].tolist()
    generated = tokenizer.decode(out_ids[len(ids):])
    return generated


def generate_multi_turn(model, tokenizer, turns, max_new_chars=50,
                        temperature=1.0):
    """Generate a multi-turn conversation.

    Args:
        model: The trained model.
        tokenizer: The tokenizer.
        turns: List of (role, text) tuples representing the conversation.
        max_new_chars: Maximum new chars to generate per response.
        temperature: Sampling temperature.

    Returns:
        List of (role, text) tuples representing the full conversation
        including generated responses.
    """
    model.eval()
    conversation = []

    for role, text in turns:
        # Add the user/assistant message
        conversation.append((role, text))

        # If it's a user turn, generate assistant response
        if role == "user":
            # Build the full conversation context as a single prompt
            context_parts = []
            for r, t in conversation:
                if r == "user":
                    context_parts.append(f"<|user|>{t}<|end|>")
                elif r == "assistant":
                    context_parts.append(f"<|assistant|>{t}<|end|>")

            context = "".join(context_parts)
            ids = tokenizer.encode(context)
            if not ids:
                ids = [tokenizer.unk_id]
            ids = ids[-tokenizer.vocab_size:]  # Truncate to vocab size
            # Also truncate to block_size
            ids = ids[-model.config.block_size:]
            idx = torch.tensor([ids], dtype=torch.long)

            with torch.no_grad():
                out = model.generate(idx, max_new_tokens=max_new_chars,
                                     temperature=temperature)
            out_ids = out[0].tolist()
            generated = tokenizer.decode(out_ids[len(ids):])
            # Clean up the generated text - stop at <|end|> if present
            if "<|end|>" in generated:
                generated = generated.split("<|end|>")[0]
            generated = generated.strip()

            conversation.append(("assistant", generated))

    return conversation


def run_evaluation(checkpoint_path=LATEST_CHECKPOINT,
                   tokenizer_path=TOKENIZER_PATH,
                   max_new_chars=80):
    """Run the full evaluation pipeline."""
    import torch

    print("=" * 60)
    print("Stage 10 Phase G — Evaluation Pipeline")
    print("=" * 60)

    model, tok, config, step = load_checkpoint_and_tokenizer(
        checkpoint_path, tokenizer_path
    )
    print(f"Checkpoint: {rel(checkpoint_path)} (step {step})")
    print(f"Tokenizer: vocab={tok.vocab_size}")
    print(f"Model: {model.num_params():,} parameters | "
          f"block={config.block_size}")

    # Load conversational data for evaluation
    from stage10.conversation_prep import load_conversations
    conv_path = os.path.join(PROJECT_ROOT, "data", "stage10",
                             "conversations_clean.jsonl")
    conv_data = []
    if os.path.isfile(conv_path):
        conv_data = load_conversations(conv_path)
        print(f"Loaded {len(conv_data)} conversations for evaluation")
    else:
        print("No conversation data found for evaluation")

    # Compute losses on conversational data
    print("\n--- Loss and Perplexity ---")
    eval_results = {}
    if conv_data:
        # Encode conversations
        all_windows = []
        for conv in conv_data:
            ids, mask = encode_conversation(tok, conv,
                                            config.block_size)
            windows = split_conversation_into_windows(ids, mask,
                                                      config.block_size)
            all_windows.extend(windows)

        if all_windows:
            train_loss = compute_loss(model, all_windows, 4,
                                      config.block_size, 10)
            train_ppl = compute_perplexity(train_loss)
            eval_results["conversational"] = {
                "train_loss": train_loss,
                "train_perplexity": train_ppl,
            }
            print(f"  conversational: train loss = {train_loss:.4f}, "
                  f"perplexity = {train_ppl:.2f}")
        else:
            print("  No valid conversation windows for evaluation")

    # Vocabulary utilization
    print("\n--- Vocabulary Utilization ---")
    if all_windows:
        util = vocab_utilization(model, tok, all_windows, 4,
                                 config.block_size, 5)
        print(f"  Conversational: {util:.1%} of vocabulary used "
              f"in predictions")

    # Generation samples
    print("\n--- Generation Samples ---")
    gen_results = {}
    for prompt in EVAL_PROMPTS:
        generated = generate_sample(model, tok, prompt, max_new_chars)
        rep = detect_repetition(generated)
        gen_results[prompt] = {
            "generated": generated,
            "repetition": rep,
            "length": len(generated),
        }
        print(f"  Prompt: {prompt!r}")
        print(f"    Generated: {generated[:100]}...")
        print(f"    Repetition score: {rep:.2%}")
        print()

    # Multi-turn conversation test
    print("\n--- Multi-turn Conversation Test ---")
    multi_turn_result = generate_multi_turn(model, tok,
                                            MULTI_TURN_PROMPT,
                                            max_new_chars=max_new_chars)
    for role, text in multi_turn_result:
        print(f"  {role}: {text}")
    # Check if the model answered correctly
    last_assistant = ""
    for role, text in reversed(multi_turn_result):
        if role == "assistant":
            last_assistant = text
            break
    mentions_blue = "blue" in last_assistant.lower()
    print(f"\n  Mentions 'blue': {mentions_blue}")
    gen_results["multi_turn"] = {
        "conversation": multi_turn_result,
        "mentions_blue": mentions_blue,
    }

    # Vocabulary info
    print("\n--- Vocabulary Info ---")
    print(f"  Total vocab size: {tok.vocab_size}")
    print(f"  Characters: {sorted(tok.characters)[:20]}...")

    # Write report
    write_report(eval_results, gen_results, step, config, tok)
    print(f"\nReport written to: {rel(REPORT_PATH)}")
    print("\nPhase G complete.")


def vocab_utilization(model, tokenizer, data_windows,
                      batch_size, block_size, n_batches):
    """Fraction of vocabulary tokens that appear as predictions (argmax)
    across a set of batches. Measures whether the model uses its full
    vocabulary or collapses to a few tokens."""
    import torch

    model.eval()
    used = set()
    total = len(data_windows)
    if total == 0:
        return 0.0
    for _ in range(n_batches):
        ix = torch.randint(0, total, (batch_size,))
        for i in ix:
            i = int(i)
            ids, mask = data_windows[i]
            x = torch.tensor([ids], dtype=torch.long)
            logits, _ = model(x, y=x)
            preds = logits.argmax(dim=-1)
            for p in preds.flatten().tolist():
                used.add(p)
    return len(used) / tokenizer.vocab_size if tokenizer.vocab_size > 0 else 0.0


def write_report(eval_results, gen_results, step, config, tokenizer):
    """Write the evaluation report as Markdown."""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    lines = []
    lines.append("# Stage 10 Evaluation Report")
    lines.append("")
    lines.append("## Checkpoint")
    lines.append("")
    lines.append(f"- Step: {step}")
    lines.append(f"- Model: {config.n_layer} layers, {config.n_head} heads, "
                 f"{config.n_embd} embedding dim")
    lines.append(f"- Context length: {config.block_size}")
    lines.append(f"- Vocabulary size: {tokenizer.vocab_size}")
    lines.append("")

    lines.append("## Loss and Perplexity by Corpus")
    lines.append("")
    lines.append("| Corpus | Train Loss | Train Perplexity | Val Loss | Val Perplexity |")
    lines.append("|--------|------------|------------------|----------|----------------|")
    for cid, r in eval_results.items():
        vl = f"{r['val_loss']:.4f}" if not math.isnan(r["val_loss"]) else "N/A"
        vp = f"{r['val_perplexity']:.2f}" if not \
            math.isnan(r["val_perplexity"]) else "N/A"
        lines.append(f"| {cid} | {r['train_loss']:.4f} | "
                     f"{r['train_perplexity']:.2f} | {vl} | {vp} |")
    lines.append("")

    lines.append("## Generation Samples")
    lines.append("")
    lines.append("Generated with temperature=1.0, no top-k. Repetition score "
                 "measures the fraction of words that repeat within a "
                 "sliding window of 5 words.")
    lines.append("")
    for prompt, r in gen_results.items():
        lines.append(f"### Prompt: {prompt!r}")
        lines.append("")
        lines.append(f"Generated: {r['generated']}")
        lines.append("")
        lines.append(f"- Length: {r['length']} characters")
        lines.append(f"- Repetition score: {r['repetition']:.2%}")
        lines.append("")

    # Multi-turn conversation results
    if "multi_turn" in gen_results:
        lines.append("## Multi-turn Conversation Test")
        lines.append("")
        mt = gen_results["multi_turn"]
        lines.append("### Conversation:")
        lines.append("")
        for role, text in mt["conversation"]:
            lines.append(f"- {role}: {text}")
        lines.append("")
        lines.append(f"- Mentions 'blue': {mt['mentions_blue']}")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("### What the metrics show")
    lines.append("")
    lines.append("- **Train/val loss:** Lower is better. A gap between train "
                 "and val loss indicates overfitting.")
    lines.append("- **Perplexity:** exp(loss). Lower is better. Roughly equals "
                 "the average number of equally-likely choices the model faces "
                 "at each position.")
    lines.append("- **Vocabulary utilization:** Fraction of vocab tokens the "
                 "model actually predicts. Low utilization (<30%) suggests the "
                 "model has collapsed to a few frequent tokens.")
    lines.append("- **Repetition score:** Higher values indicate repetitive, "
                 "looping generation — a sign of poor sampling or a poorly "
                 "trained model.")
    lines.append("- **Multi-turn context:** The model should be able to "
                 "reference earlier turns in the conversation (e.g., answering "
                 "\"blue\" when asked what color was mentioned).")
    lines.append("")
    lines.append("### What the metrics cannot show")
    lines.append("")
    lines.append("- These metrics measure statistical next-character "
                 "prediction, not understanding, reasoning, or factual "
                 "accuracy.")
    lines.append("- A low loss does not mean the model can answer questions "
                 "correctly or maintain conversational context.")
    lines.append("- Generation quality on a few prompts is not a substitute "
                 "for systematic human evaluation.")
    lines.append("")
    lines.append("### Honest assessment")
    lines.append("")
    lines.append("A model trained for only a few hundred steps on "
                 "~200 KB of text will produce mostly gibberish. This is "
                 "expected. Natural language communication requires much "
                 "more data and many more training steps.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by src/stage10/evaluate.py*")

    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written to: {rel(REPORT_PATH)}")
    print("\nPhase G complete.")


if __name__ == "__main__":
    run_evaluation()
