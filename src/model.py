"""
model.py — OUR OWN tiny decoder-only Transformer ("TinyLM").
=============================================================

WHAT THIS FILE IS
    The "brain" of my-LLM. It defines the model architecture from
    scratch on top of plain PyTorch tensor operations, with randomly
    initialized weights only. No training logic lives here — the
    training loop arrives in Stage 3 (train.py).

THE BIG PICTURE (read this first)
    A decoder-only Transformer predicts the NEXT character. For the
    input "c a t", the model produces, for every position, a guess of
    what character comes AFTER that position: after "c" predict "a",
    after "ca" predict "t". Position i may only look at positions <= i
    — that is what "causal" means, and it is what makes generation
    possible one character at a time.

    Data flow for one batch:

        token ids                (B, T)      integers 0..V-1
              |  + position info
              v
        "thought" vectors        (B, T, C)   one vector per character
              |  n_layer x [ (self-attention: exchange info BEHIND you)
              |                (feed-forward:  think about what you heard) ]
              v
        final LayerNorm
              |
              v
        logits                   (B, T, V)   a score for EVERY character,
                                             at EVERY position
        loss = cross-entropy(logits vs true next character)

SHAPE CONVENTION (used in every comment)
    B = batch      how many sequences we process at once
    T = time       how many positions (characters) are in one sequence
    C = channels   how many numbers describe one position (the embedding)
    V = vocab      how many distinct characters exist (vocab_size)

DESIGN CHOICES WE MADE (compare with the older reference script
"import os.py" in the project root — same family of thought, our own
cleaner implementation):
    * PRE-norm blocks. Normalization BEFORE attention/feed-forward.
      Small models train more stably this way; modern LLMs do this.
    * One fused qkv projection instead of three separate linears:
      identical math, fewer operations.
    * Attention written out by hand (q@k, mask, softmax, @v) because
      SEEING the math is a project goal. (A built-in fast version
      exists: torch.nn.functional.scaled_dot_product_attention.)
    * Learned absolute position embeddings: the simplest scheme that
      works. Alternatives (e.g. RoPE) are a Stage-5 upgrade.
    * Custom init with small random numbers (std = 0.02) so the very
      first predictions are near-uniform and training starts stable.

LEARNED vs NOT-LEARNED (frequent beginner question)
    WEIGHted parts (learnable): embeddings, all attention projections,
    the feed-forward layers, the output head.
    NOT learned: the causal mask, LayerNorm statistics (rescaled per
    position at runtime), the weight initialization itself.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    """All the knobs of one TinyLM. Defaults = our recommended V1 model
    (~0.8 M parameters) from hardware_report.md: small enough for a
    2-core CPU, big enough to produce readable character text."""

    vocab_size: int        # number of distinct characters (from the tokenizer)
    block_size: int = 128  # context length: how far back the model can see
    n_layer: int = 4       # how many Transformer blocks stacked
    n_head: int = 4        # attention heads per block (n_embd must divide evenly)
    n_embd: int = 128      # embedding channels C
    dropout: float = 0.1   # regularizer: fraction of activations randomly zeroed


class CausalSelfAttention(nn.Module):
    """
    Self-attention, written out by hand.

    Intuition (keep this!):
      - every position ASKS a question (query): "what am I looking for?"
      - every position ADVERTISES what it contains (key)
      - query . key = how interesting is this earlier position to me?
      - softmax turns the interest scores into percentages (rows sum to 1)
      - each position then RECEIVES a weighted average of value vectors:
        it "pulls in" information from the earlier characters it cares about.
    """

    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Compute Q, K, V for ALL heads in one multiplication,
        # then slice into n_head pieces. Output size: 3 * C.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)   # on attention weights
        self.resid_dropout = nn.Dropout(config.dropout)  # on the block output

        # The causal mask: lower-triangular True. This is bookkeeping
        # (NOT learned): position i may attend to positions <= i only.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
        )
        # 1/sqrt(head_dim) keeps the dot products from blowing up.
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)                          # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)    # 3 x (B, T, C)

        # Split C into n_head pieces so heads "think in parallel":
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # How much each position cares about each earlier position:
        att = (q @ k.transpose(-2, -1)) * self.scale   # (B, n_head, T, T)
        # Forbid looking into the FUTURE (those pairs get -inf -> ~0 probability):
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)                   # rows now sum to 1
        att = self.attn_dropout(att)

        # Pull in the information, then merge the heads back together:
        y = att @ v                                    # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, n_head*head_dim = C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """After looking around (attention), each position THINKS on its own:
    widen C -> 4C (room to combine ideas), nonlinear step (GELU),
    narrow back to C. Attention MOVES information; this part PROCESSES it."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One repeatable layer = (attention + feed-forward), each wrapped in a
    residual connection so gradients can flow easily through depth."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # "note what happened here, keep your memories"
        x = x + self.ffwd(self.ln2(x))  # "now think about it"
        return x


class TinyLM(nn.Module):
    """The whole model: embeddings -> n_layer blocks -> norm -> output head."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Lookup tables. Embeddings turn integer IDs into vectors.
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)

        # Maps every position's vector to V scores (one per character).
        # bias=False: enough freedom without it, saves parameters.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Our own initialization: small random numbers for all learnable
        # Linear/Embedding weights, zeros for biases.
        self.apply(self._init_weights)

    # --- weights ------------------------------------------------------

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self):
        """Total number of learnable numbers ('parameters')."""
        return sum(p.numel() for p in self.parameters())

    def param_overview(self):
        """Parameter count grouped by where they live in the model."""
        groups = {
            "token + position embeddings": 0,
            "transformer blocks": 0,
            "final norm + output head": 0,
        }
        for name, p in self.named_parameters():
            n = p.numel()
            if name.startswith("tok_emb") or name.startswith("pos_emb"):
                groups["token + position embeddings"] += n
            elif name.startswith("blocks"):
                groups["transformer blocks"] += n
            else:
                groups["final norm + output head"] += n
        return groups

    # --- the forward pass ---------------------------------------------

    def forward(self, idx, targets=None):
        """idx (B, T) integer ids -> logits (B, T, V) and optional loss."""
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Got sequence of {T} positions but block_size is "
                f"{self.config.block_size}. Crop input with generate()'s trick "
                f"or lower the config."
            )

        # What the character IS (token embedding) + WHERE it sits (position
        # embedding) -> a "thought" vector per position: (B, T, C).
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)   # (B,T,C) + (T,C) broadcast

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)                    # (B, T, V)

        if targets is None:
            return logits, None

        # The loss asks one question for every position: "was your guess
        # of the next character right?" Faster/simpler than iterating.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return logits, loss

    # --- generation -----------------------------------------------------

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Grow the sequence one token at a time:
        run the model on the (cropped) context, read the LAST position's
        scores, turn them into probabilities, sample once, append.
        temperature: <1 = more confident/repetitive, >1 = wilder.
        (top-k sampling is a Stage-4 upgrade.)
        """
        was_training = self.training
        self.eval()  # disable dropout so sampling is deterministic math
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        if was_training:
            self.train()
        return idx
