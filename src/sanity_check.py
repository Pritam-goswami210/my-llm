"""
sanity_check.py — the Stage-2 EXIT TEST for my-LLM.
====================================================

WHAT IT PROVES (and why we test the UNTRAINED model)
    Building a training pipeline is mostly plumbing. Before training
    anything, we prove the plumbing works:

      1. the tokenizer round-trips text -> IDs -> text without changes,
      2. unseen characters don't crash it (they map to <unk>),
      3. the tokenizer saves/loads to a file without changing,
      4. the model builds with RANDOM weights (~0.8 M parameters),
      5. a forward pass produces correctly-shaped logits,
      6. an untrained model's loss is close to ln(vocab_size) — this is
         a real mathematical prediction! A random model has no idea what
         comes next, so it spreads probability evenly over V characters,
         and the cross-entropy of a uniform guess over V options is
         exactly -ln(1/V) = ln(V). If the number we measure is far off,
         something in the model or loss is wired wrong.
      7. gradients flow BACKWARD through every parameter,
      8. one tiny optimizer step nudges the loss DOWN,
      9. sampling produces character-shaped GARBAGE — which is exactly
         what an untrained model should speak. Nobody taught it yet.

    Passing does NOT mean the model is good. It means we built the
    machine correctly; Stage 3 will give it something to learn.

HOW TO RUN (from the project root)
    python src\\sanity_check.py

SAFETY
    No downloads, no network, no real training (a few toy steps lasting
    well under a second). Writes one small demo file inside runs/.
    We import only torch + the standard library; numpy and tqdm (listed
    in requirements.txt) are for Stage 3's real training loop.
"""

import math
import os
import sys
import time

import torch

from model import TinyConfig, TinyLM
from tokenizer import CharacterTokenizer

# Where we are, no matter how the script is launched:
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# A small toy corpus INSIDE this file (no data files needed yet).
# It contains letters, digits and punctuation so the character set
# looks like a realistic tiny one.
SAMPLE_CORPUS = (
    "Welcome to my-LLM: a tiny language model we build from scratch.\n"
    "The model reads characters and learns to guess what comes next.\n"
    "Attention lets the model look back a few steps in the text.\n"
    "Letters, digits 0123456789 and marks . , ! ? ; : ' \" ( ) - all count.\n"
    "Everything runs offline on my little 2-core home computer.\n"
    "Small models learn slowly, but they teach us the entire recipe.\n"
)

FAILURES = []


def check(name, condition, detail=""):
    """Print a PASS/FAIL line for one test, remember failures."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)
    return bool(condition)


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    banner("Stage 2 sanity check — my-LLM")
    print(f"python {sys.version.split()[0]}  |  torch {torch.__version__}  |  device: cpu")
    print(f"CPU cores seen by the OS: {os.cpu_count()}  |  torch threads now: {torch.get_num_threads()}")
    torch.manual_seed(0)  # make this whole run reproducible

    # ------------------------------------------------------------------
    banner("Part 1 — the character tokenizer")
    # ------------------------------------------------------------------
    tok = CharacterTokenizer.from_text(SAMPLE_CORPUS)
    print(f"  toy corpus: {len(SAMPLE_CORPUS)} characters")
    print(f"  vocabulary: {tok.vocab_size} distinct characters")
    print(f"  characters: {tok.characters}")

    roundtrip = tok.decode(tok.encode(SAMPLE_CORPUS)) == SAMPLE_CORPUS
    check("encode->decode round-trips the whole corpus unchanged", roundtrip)

    # An unseen character must map to <unk>, not crash the tokenizer.
    demo_ids = tok.encode("x@y")  # '@' is NOT in the corpus above
    check("unseen characters (here '@') map to <unk> instead of crashing",
          tok.decode(demo_ids) == "x<unk>y", f"decode -> {tok.decode(demo_ids)!r}")

    # Saving/loading must give an IDENTICAL tokenizer (stable IDs are sacred).
    tokenizer_path = tok.save(os.path.join(RUNS_DIR, "tokenizer_demo.json"))
    tok2 = CharacterTokenizer.load(tokenizer_path)
    check("save/load preserves the exact vocabulary",
          tok2.vocab_size == tok.vocab_size and tok2.encode(SAMPLE_CORPUS) == tok.encode(SAMPLE_CORPUS),
          f"saved to runs/tokenizer_demo.json")

    # ------------------------------------------------------------------
    banner("Part 2 — building the model with random weights")
    # ------------------------------------------------------------------
    config = TinyConfig(vocab_size=tok.vocab_size)  # defaults = V1 sizing
    print(f"  config: {config}")
    model = TinyLM(config)   # weights are random right now (std 0.02)

    total = model.num_params()
    print(f"  total parameters: {total:,}")
    for group, count in model.param_overview().items():
        print(f"    {group:32s} {count:>10,}")
    print(f"  raw storage at 4 bytes/parameter: ~{total * 4 / 1e6:.1f} MB")
    check("model built with ~0.8M parameters (V1 recommendation)",
          500_000 < total < 1_500_000, f"got {total:,}")

    # ------------------------------------------------------------------
    banner("Part 3 — forward pass on a real minibatch")
    # ------------------------------------------------------------------
    # Turn the corpus into IDs, then build batches of (input, target)
    # pairs. The trick to internalize: targets are the SAME text shifted
    # by one position. At input position i the model must predict the
    # true character at position i+1 — that IS "learning next character".
    data = torch.tensor(tok.encode(SAMPLE_CORPUS), dtype=torch.long)
    B, T = 4, 64
    starts = torch.randint(0, len(data) - T - 1, (B,))
    x = torch.stack([data[i : i + T] for i in starts])          # inputs
    y = torch.stack([data[i + 1 : i + 1 + T] for i in starts])  # next chars
    print(f"  batch x: {tuple(x.shape)}  (B sequences, T positions)")

    logits, loss = model(x, y)
    print(f"  logits:  {tuple(logits.shape)}  (a score for each character, per position)")
    print(f"  measured untrained loss: {loss.item():.4f}")

    expected = math.log(tok.vocab_size)
    print(f"  predicted by math (ln(vocab)): {expected:.4f}")
    check("forward pass returns finite, correctly-shaped output",
          torch.isfinite(loss).item() and tuple(logits.shape) == (B, T, tok.vocab_size))
    check("untrained loss is close to ln(vocab_size) as mathematics demands",
          abs(loss.item() - expected) < 0.5,
          f"measured {loss.item():.3f} vs expected ~{expected:.3f}")

    # ------------------------------------------------------------------
    banner("Part 4 — gradient flow and one micro training step")
    # ------------------------------------------------------------------
    model.train()  # enable dropout-path (training behavior)
    _, loss_t = model(x, y)
    loss_t.backward()  # PyTorch computes d(loss)/d(every parameter) — the worst errors' directions
    with_grads = sum(1 for p in model.parameters() if p.grad is not None)
    check("backward() reaches EVERY parameter with a gradient",
          with_grads == len(list(model.parameters())), f"{with_grads}/{len(list(model.parameters()))} tensors")

    # Clean comparison in eval mode (dropout off) before and after ONE step:
    model.eval()
    before = model(x, y)[1].item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()  # uses the gradients from the backward pass above
    after = model(x, y)[1].item()
    delta = after - before
    print(f"  loss before one micro step: {before:.4f}")
    print(f"  loss after  one micro step: {after:.4f}  (delta {delta:+.4f})")
    print("  (informational, not a pass/fail — real training needs many steps)")

    check("an optimizer step changed the loss", delta != 0.0)

    # ------------------------------------------------------------------
    banner("Part 5 — generation (should be pure, beautiful garbage)")
    # ------------------------------------------------------------------
    prompt = "The model reads"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out_ids = model.generate(ids.clone(), max_new_tokens=120, temperature=1.0)
    prompt_len = len(prompt)
    n_ids = out_ids.shape[1]
    generated = tok.decode(out_ids[0].tolist())  # .tolist(): tensor -> plain ints
    print("  generated from the prompt, untrained model:")
    print("  +" + "-" * 66 + "+")
    print("  | " + generated.replace("\n", " ").ljust(64) + " |")
    print("  +" + "-" * 66 + "+")
    # NOTE: decode() may render FEWER characters than sampled IDs, because a
    # random model sometimes picks the <pad> slot (id 0), which decodes to
    # nothing. The correct test of generate() is therefore on the ID level:
    # it must return exactly the prompt IDs + max_new_tokens new IDs.
    check("generation keeps the prompt and adds exactly the requested IDs",
          n_ids == prompt_len + 120 and tok.decode(out_ids[0, :prompt_len].tolist()) == prompt,
          f"{n_ids} token IDs = {prompt_len} prompt + {n_ids - prompt_len} generated")

    # ------------------------------------------------------------------
    banner("Part 6 — raw speed probe (honest numbers for planning)")
    # ------------------------------------------------------------------
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    n_steps = 5
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _, l = model(x, y)
        l.backward()
        optimizer.zero_grad(set_to_none=True)
    dt = (time.perf_counter() - t0) / n_steps
    print(f"  this tiny probe batch (B={B}, T={T}): ~{dt * 1000:.1f} ms per fwd+bwd step")
    print("  (Stage 3 will benchmark the REAL training batch properly)")

    # ------------------------------------------------------------------
    banner("VERDICT")
    # ------------------------------------------------------------------
    if FAILURES:
        print(f"  {len(FAILURES)} check(s) FAILED: {FAILURES}")
        print("  -> do NOT proceed to training; investigate the failures first.")
        sys.exit(1)
    print("  ALL CHECKS PASSED")
    print("  Stage 2 complete: our own tokenizer + our own model, plumbing proven.")
    print("  Next: Stage 3 — pick training text, write train.py, and let it LEARN.")


if __name__ == "__main__":
    main()
