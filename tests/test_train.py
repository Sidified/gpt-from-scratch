import torch

from src.config import GPTConfig
from src.model import GPT


def test_model_can_overfit_one_batch():
    """The standard sanity check: if a model cannot memorise a single batch,
    the training path is broken somewhere. Loss must collapse toward zero."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=8, n_embd=64, n_head=4, n_layer=2, dropout=0.0)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))

    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    assert loss.item() < 0.1, f"could not overfit a single batch: loss {loss.item():.3f}"


def test_loss_decreases_on_real_signal():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=8, n_embd=64, n_head=4, n_layer=2, dropout=0.0)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = torch.arange(2000) % cfg.vocab_size   # perfectly predictable sequence

    first = None
    for _ in range(200):
        i = torch.randint(0, len(data) - cfg.block_size - 1, (8,))
        x = torch.stack([data[j : j + cfg.block_size] for j in i])
        y = torch.stack([data[j + 1 : j + cfg.block_size + 1] for j in i])
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    assert loss.item() < first * 0.5


def test_eval_mode_is_deterministic():
    """With dropout disabled, two identical forward passes must match exactly."""
    cfg = GPTConfig(vocab_size=20, block_size=8, n_embd=32, n_head=4, n_layer=2, dropout=0.5)
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x)
    assert torch.equal(a, b)


def test_train_mode_dropout_is_active():
    cfg = GPTConfig(vocab_size=20, block_size=8, n_embd=32, n_head=4, n_layer=2, dropout=0.5)
    model = GPT(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x)
    assert not torch.equal(a, b)
