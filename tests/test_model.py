import math

import pytest
import torch

from src.config import GPTConfig
from src.model import GPT


def tiny(**kw):
    base = dict(vocab_size=20, block_size=8, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_output_shapes():
    cfg = tiny()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0


def test_initial_loss_is_near_uniform_entropy():
    """An untrained model should be maximally unsure: loss ~= ln(vocab_size)."""
    torch.manual_seed(0)
    cfg = tiny()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (32, cfg.block_size))
    _, loss = model(idx, idx)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3


def test_causality_future_cannot_leak_into_past():
    """Changing the last token must not change any earlier position's logits."""
    torch.manual_seed(0)
    cfg = tiny()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % cfg.vocab_size

    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(idx2)

    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_accepts_sequences_shorter_than_block_size():
    cfg = tiny()
    model = GPT(cfg)
    logits, _ = model(torch.randint(0, cfg.vocab_size, (2, 3)))
    assert logits.shape == (2, 3, cfg.vocab_size)


def test_rejects_sequences_longer_than_block_size():
    cfg = tiny()
    model = GPT(cfg)
    with pytest.raises(ValueError):
        model(torch.randint(0, cfg.vocab_size, (2, cfg.block_size + 1)))


def test_n_embd_must_divide_by_n_head():
    with pytest.raises(ValueError):
        GPT(tiny(n_embd=30, n_head=4))


def test_every_parameter_receives_gradient():
    cfg = tiny()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    _, loss = model(idx, idx)
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, f"no gradient reached: {dead}"


@pytest.mark.parametrize(
    "flag", ["use_layernorm", "use_residual", "use_pos_emb"]
)
def test_ablation_flags_change_the_model(flag):
    """Each switch must actually alter behaviour, or Block 6 measures nothing."""
    torch.manual_seed(0)
    on = GPT(tiny(**{flag: True})).eval()
    torch.manual_seed(0)
    off = GPT(tiny(**{flag: False})).eval()
    idx = torch.randint(0, 20, (2, 8))
    with torch.no_grad():
        a, _ = on(idx)
        b, _ = off(idx)
    assert not torch.allclose(a, b)
