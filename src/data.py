"""Character-level tokenizer and batch sampling."""

from pathlib import Path

import torch


class CharTokenizer:
    """Maps every unique character in the corpus to an integer, and back."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_data(path: str, train_frac: float = 0.9):
    """Read the corpus, tokenize it, split into train/val by position."""
    text = Path(path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    return tokenizer, data[:n], data[n:]


def get_batch(data, batch_size: int, block_size: int, device=None, generator=None):
    """Sample `batch_size` random chunks of `block_size` tokens.

    Returns (x, y) where y is x shifted one position right — so at every
    position, the target is the character that actually came next.
    """
    if len(data) < block_size + 1:
        raise ValueError(
            f"need at least {block_size + 1} tokens, got {len(data)}"
        )
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y


def tokenizer_from_stoi(stoi: dict) -> CharTokenizer:
    """Rebuild a tokenizer from a saved vocabulary.

    Works because CharTokenizer assigns ids in sorted character order, so
    feeding the vocab back in reproduces the identical mapping.
    """
    tok = CharTokenizer("".join(stoi.keys()))
    if tok.stoi != stoi:
        raise ValueError("checkpoint vocabulary does not match rebuilt tokenizer")
    return tok
