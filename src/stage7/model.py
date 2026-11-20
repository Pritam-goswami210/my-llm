"""
model.py — Stage 7 TinyLM for BPE tokenization.
===============================================

Self-contained decoder-only Transformer for Stage 7. Same architecture
as src/model.py (TinyLM) but defined independently so Stage 7 remains
fully isolated from Stages 1-6.

Architecture:
  token embeddings -> position embeddings -> N x (causal self-attention
  + feed-forward, pre-norm) -> final layer norm -> output head

The output head (lm_head) is a separate Linear layer (weights are NOT
tied with tok_emb), matching the original design.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Stage7Config:
    """Configuration for the Stage 7 TinyLM.

    Defaults are small and CPU-friendly for an i5-4570T-class machine
    with ~16 GB RAM and no GPU acceleration.
    """

    vocab_size: int = 1024       # BPE vocabulary size
    block_size: int = 128        # context length (tokens)
    n_layer: int = 4             # Transformer blocks
    n_head: int = 4              # attention heads
    n_embd: int = 128            # embedding channels
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Self-attention with a causal (lower-triangular) mask.

    Position i may attend only to positions j <= i. Implemented by hand
    (q @ k, mask, softmax, @ v) so the math is visible and auditable.
    """

    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
        )
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)                          # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)    # 3 x (B, T, C)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale   # (B, n_head, T, T)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                    # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """Position-wise feed-forward network: C -> 4C -> C with GELU."""

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
    """One pre-norm Transformer block: attention + feed-forward with
    residual connections."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Stage7LM(nn.Module):
    """The full Stage 7 model: embeddings -> blocks -> norm -> output head."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)

        # Output head: separate from tok_emb (weights NOT tied)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def param_overview(self):
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

    def forward(self, idx, targets=None):
        """idx (B, T) integer token IDs -> logits (B, T, V) and optional loss.

        Causal masking is enforced inside CausalSelfAttention so a
        position cannot attend to future tokens.
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Got sequence of {T} positions but block_size is "
                f"{self.config.block_size}."
            )

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)   # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)                    # (B, T, V)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation: predict one token at a time.

        Parameters
        ----------
        idx : torch.LongTensor (B, T)
            Starting token IDs.
        max_new_tokens : int
            How many new tokens to generate.
        temperature : float
            Sampling temperature. <1 = safer/more repetitive, >1 = wilder.
        top_k : int or None
            If set, only sample from the top-k most likely tokens.
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                # Keep only top-k logits, set others to -inf
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        if was_training:
            self.train()
        return idx
