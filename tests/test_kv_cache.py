import pytest
import torch

from src.config import GPTConfig
from src.model import GPT
from src.benchmark import generate_cached, generate_naive


def tiny():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
    return GPT(cfg).eval()


def test_cached_logits_match_uncached():
    """The core correctness property: incremental decoding must produce the
    same logits as recomputing the whole prefix."""
    m = tiny()
    idx = torch.randint(0, 20, (1, 6))
    with torch.no_grad():
        full, _ = m(idx)
        _, _, past = m(idx[:, :5], use_cache=True)
        step, _, _ = m(idx[:, 5:6], past_kvs=past, use_cache=True)
    assert torch.allclose(full[:, -1, :], step[:, -1, :], atol=1e-5)


def test_greedy_cached_equals_greedy_naive():
    m = tiny()
    idx = torch.zeros((1, 3), dtype=torch.long)
    assert torch.equal(
        generate_naive(m, idx, 10, temperature=0.0),
        generate_cached(m, idx, 10, temperature=0.0),
    )


def test_cache_grows_by_one_per_step():
    m = tiny()
    idx = torch.randint(0, 20, (1, 4))
    with torch.no_grad():
        _, _, past = m(idx, use_cache=True)
        assert past[0][0].shape[2] == 4
        _, _, past = m(idx[:, :1], past_kvs=past, use_cache=True)
        assert past[0][0].shape[2] == 5


def test_cache_is_per_layer():
    m = tiny()
    with torch.no_grad():
        _, _, past = m(torch.randint(0, 20, (1, 4)), use_cache=True)
    assert len(past) == m.config.n_layer
    assert not torch.equal(past[0][0], past[1][0])


def test_position_offset_is_applied():
    """If cached positions restarted at 0, this token would get position 0
    instead of position 5 and the logits would differ."""
    m = tiny()
    idx = torch.randint(0, 20, (1, 6))
    with torch.no_grad():
        full, _ = m(idx)
        _, _, past = m(idx[:, :5], use_cache=True)
        step, _, _ = m(idx[:, 5:6], past_kvs=past, use_cache=True)
        wrong, _ = m(idx[:, 5:6])          # same token, no cache, position 0
    assert torch.allclose(full[:, -1], step[:, -1], atol=1e-5)
    assert not torch.allclose(step[:, -1], wrong[:, -1], atol=1e-3)


def test_exceeding_block_size_raises():
    m = tiny()
    with pytest.raises(ValueError):
        generate_cached(m, torch.zeros((1, 2), dtype=torch.long), 20)


def test_backward_compatible_two_value_return():
    """Training code calls forward without use_cache and expects (logits, loss)."""
    m = tiny()
    idx = torch.randint(0, 20, (2, 8))
    out = m(idx, idx)
    assert len(out) == 2
