import torch

from src.config import GPTConfig
from src.model import GPT
from src.generate import apply_top_k, apply_top_p, generate, distinct_ngram_ratio


def tiny_model(block_size=8):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=block_size, n_embd=32,
                    n_head=4, n_layer=2, dropout=0.0)
    return GPT(cfg).eval()


def test_output_length_grows_by_exactly_max_new_tokens():
    m = tiny_model()
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = generate(m, idx, max_new_tokens=10)
    assert out.shape == (1, 13)
    assert torch.equal(out[:, :3], idx)      # prompt is preserved


def test_can_generate_past_the_context_window():
    """Generating more tokens than block_size must not crash — the context
    is cropped, not extended."""
    m = tiny_model(block_size=8)
    out = generate(m, torch.zeros((1, 5), dtype=torch.long), max_new_tokens=30)
    assert out.shape == (1, 35)


def test_all_sampled_tokens_are_valid_ids():
    m = tiny_model()
    out = generate(m, torch.zeros((1, 2), dtype=torch.long), max_new_tokens=50)
    assert out.min() >= 0 and out.max() < 20


def test_greedy_is_deterministic():
    m = tiny_model()
    idx = torch.zeros((1, 3), dtype=torch.long)
    a = generate(m, idx, 20, temperature=0.0)
    b = generate(m, idx, 20, temperature=0.0)
    assert torch.equal(a, b)


def test_top_k_1_equals_greedy():
    m = tiny_model()
    idx = torch.zeros((1, 3), dtype=torch.long)
    greedy = generate(m, idx, 20, temperature=0.0)
    topk1 = generate(m, idx, 20, temperature=1.0, top_k=1)
    assert torch.equal(greedy, topk1)


def test_top_k_keeps_exactly_k_candidates():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    out = apply_top_k(logits, 3)
    assert torch.isfinite(out).sum().item() == 3
    assert torch.equal(out[0, :3], logits[0, :3])


def test_top_p_never_masks_everything():
    """The degenerate case: one token holds more mass than p. A naive
    implementation masks the whole vocabulary and softmax returns NaN."""
    logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])   # first token ~= probability 1
    out = apply_top_p(logits, 0.5)
    assert torch.isfinite(out).sum().item() >= 1
    probs = torch.softmax(out, dim=-1)
    assert not torch.isnan(probs).any()
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_top_p_keeps_the_nucleus():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    kept = torch.isfinite(apply_top_p(logits, 0.9)).sum().item()
    assert kept == 3     # 0.5 + 0.3 + 0.15 crosses 0.9; the 0.05 tail is dropped


def test_same_seed_gives_same_sample():
    m = tiny_model()
    idx = torch.zeros((1, 3), dtype=torch.long)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = generate(m, idx, 20, temperature=1.0, generator=g1)
    b = generate(m, idx, 20, temperature=1.0, generator=g2)
    assert torch.equal(a, b)


def test_distinct_ngram_ratio_detects_looping():
    assert distinct_ngram_ratio("abababababababab", 4) < 0.3
    assert distinct_ngram_ratio("the quick brown fox jumps over", 4) > 0.9
