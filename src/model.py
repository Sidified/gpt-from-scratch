"""A decoder-only transformer (GPT-style), built from scratch, with KV caching.

Architecture per block: LayerNorm -> MultiHeadAttention -> residual,
then LayerNorm -> FeedForward -> residual (pre-norm, as in GPT-2).
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

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x, past_kv=None):
        """past_kv: optional (k, v) from previous steps, each (B, n_head, T_past, head_dim).

        Returns (output, present_kv). When past_kv is given, x holds only the
        new tokens — their keys and values are appended to the cache instead of
        the whole prefix being recomputed.
        """
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v)

        T_full = k.size(2)
        if T_full > self.mask.size(-1):
            raise ValueError(f"total length {T_full} exceeds block_size {self.mask.size(-1)}")

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, nh, T, T_full)
        # the T query positions are the LAST T rows of the full causal mask
        att = att.masked_fill(
            self.mask[:, :, T_full - T : T_full, :T_full] == 0, float("-inf")
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y)), present


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

    def forward(self, x, past_kv=None):
        attn_out, present = self.attn(self.ln1(x), past_kv=past_kv)
        if self.use_residual:
            x = x + attn_out
            x = x + self.ffwd(self.ln2(x))
        else:
            x = attn_out
            x = self.ffwd(self.ln2(x))
        return x, present


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

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False):
        B, T = idx.shape
        n_past = past_kvs[0][0].size(2) if past_kvs is not None else 0
        if n_past + T > self.config.block_size:
            raise ValueError(
                f"sequence length {n_past + T} exceeds block_size {self.config.block_size}"
            )

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            # positions continue from where the cache left off
            pos = torch.arange(n_past, n_past + T, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, present = block(x, past_kv=past)
            if use_cache:
                presents.append(present)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.reshape(B * T))

        if use_cache:
            return logits, loss, presents
        return logits, loss
