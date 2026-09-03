import torch

from src.config import GPTConfig
from src.model import GPT, KVCache
from src.benchmark import generate_naive
from src.sweep import generate_prealloc


def tiny():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4,
                    n_layer=2, dropout=0.0)
    return GPT(cfg).eval()


def test_prealloc_greedy_matches_naive():
    m = tiny()
    idx = torch.zeros((1, 3), dtype=torch.long)
    assert torch.equal(
        generate_naive(m, idx, 10, temperature=0.0),
        generate_prealloc(m, idx, 10),
    )


def test_prealloc_matches_naive_batched():
    """Batched decoding is where the cache indexing is easiest to get wrong."""
    m = tiny()
    idx = torch.zeros((4, 3), dtype=torch.long)
    assert torch.equal(
        generate_naive(m, idx, 10, temperature=0.0),
        generate_prealloc(m, idx, 10),
    )


def test_cache_position_advances_once_per_forward():
    m = tiny()
    cache = KVCache(m.config, batch_size=1, device="cpu")
    with torch.no_grad():
        m(torch.zeros((1, 3), dtype=torch.long), cache=cache)
        assert cache.pos == 3
        m(torch.zeros((1, 1), dtype=torch.long), cache=cache)
        assert cache.pos == 4


def test_cache_reports_expected_size():
    """2 tensors x n_layer x B x n_head x block_size x head_dim x 4 bytes."""
    m = tiny()
    c = m.config
    cache = KVCache(c, batch_size=2, device="cpu")
    expected = 2 * c.n_layer * 2 * c.n_head * c.block_size * (c.n_embd // c.n_head) * 4
    assert cache.nbytes() == expected
