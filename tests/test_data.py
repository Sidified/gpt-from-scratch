import torch

from src.data import CharTokenizer, get_batch, load_data

TEXT = "hello world, this is a small test corpus for the tokenizer."


def test_encode_decode_roundtrip():
    tok = CharTokenizer(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_vocab_size_matches_unique_chars():
    tok = CharTokenizer(TEXT)
    assert tok.vocab_size == len(set(TEXT))


def test_batch_shapes():
    data = torch.arange(1000)
    x, y = get_batch(data, batch_size=8, block_size=16)
    assert x.shape == (8, 16)
    assert y.shape == (8, 16)


def test_targets_are_inputs_shifted_by_one():
    """The core invariant: y[t] must equal x[t+1] wherever they overlap."""
    data = torch.arange(1000)
    x, y = get_batch(data, batch_size=8, block_size=16)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_never_indexes_past_the_end():
    data = torch.arange(20)
    for _ in range(200):
        x, y = get_batch(data, batch_size=4, block_size=19)
        assert y.max() < len(data)


def test_seeded_batches_are_reproducible():
    data = torch.arange(1000)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    x1, _ = get_batch(data, 8, 16, generator=g1)
    x2, _ = get_batch(data, 8, 16, generator=g2)
    assert torch.equal(x1, x2)
