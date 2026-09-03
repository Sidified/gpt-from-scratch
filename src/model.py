"""A decoder-only transformer (GPT-style), built from scratch.

Architecture per block: LayerNorm -> MultiHeadAttention -> residual,
then LayerNorm -> FeedForward -> residual. This is the "pre-norm" layout
used by GPT-2 onwards; the original 2017 paper put the norm after.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig


class MultiHeadAttention(nn.Module):
    """Causal self-attention with all heads computed in one batched op."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # one matrix produces query, key and value for every head at once
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # lower-triangular matrix of 1s: position t may attend to 0..t only.
        # registered as a buffer so it moves to GPU with .to(device) but is not a parameter.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)          # 3 x (B, T, C)
        # split the channel dim across heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # how much each position attends to each other position
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                                 # (B, nh, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # heads back together
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """Position-wise MLP. Widens 4x, then projects back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block. Ablation switches live here."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.use_residual = config.use_residual
        self.ln1 = nn.LayerNorm(config.n_embd) if config.use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(config.n_embd) if config.use_layernorm else nn.Identity()
        self.attn = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        if self.use_residual:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.attn(self.ln1(x))
            x = self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = (
            nn.Embedding(config.block_size, config.n_embd) if config.use_pos_emb else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) if config.use_layernorm else nn.Identity()
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}"
            )

        x = self.tok_emb(idx)                                   # (B, T, n_embd)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)                           # broadcast over batch
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                                   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.reshape(B * T)
            )
        return logits, loss
