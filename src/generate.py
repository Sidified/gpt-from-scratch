"""Autoregressive generation with greedy, temperature, top-k and top-p sampling.

Run:  python -m src.generate --compare
"""

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import GPTConfig
from .data import tokenizer_from_stoi
from .model import GPT


def apply_top_k(logits, k: int):
    """Keep only the k highest-scoring tokens; everything else becomes impossible."""
    k = min(k, logits.size(-1))
    kth_value = torch.topk(logits, k, dim=-1).values[:, [-1]]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def apply_top_p(logits, p: float):
    """Nucleus sampling: keep the smallest set of tokens whose probabilities sum to p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    # cumsum minus the current token = cumulative mass *before* this token.
    # At index 0 that is always 0, so the top token can never be removed —
    # this is what stops the whole distribution being masked when one token
    # already carries more than p of the mass.
    mass_before = probs.cumsum(dim=-1) - probs
    sorted_logits = sorted_logits.masked_fill(mass_before > p, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter_(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
             generator=None):
    """Naive generation: re-runs the full forward pass on the whole context
    at every step. Block 5 replaces this with a KV-cached version."""
    model.eval()
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]          # never exceed the context window
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]                # only the final position predicts next

        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                logits = apply_top_k(logits, top_k)
            if top_p is not None:
                logits = apply_top_p(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)

        idx = torch.cat([idx, next_id], dim=1)

    return idx


def distinct_ngram_ratio(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are unique. Low values mean the model is looping."""
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer_from_stoi(ckpt["stoi"])


SETTINGS = [
    ("greedy",         dict(temperature=0.0)),
    ("temp 0.5",       dict(temperature=0.5)),
    ("temp 1.0",       dict(temperature=1.0)),
    ("temp 1.5",       dict(temperature=1.5)),
    ("top-k 10",       dict(temperature=1.0, top_k=10)),
    ("top-p 0.9",      dict(temperature=1.0, top_p=0.9)),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/baseline.pt")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--prompt", default="\n")
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_checkpoint(args.checkpoint, device)
    start = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)

    rows, blocks = [], []
    settings = SETTINGS if args.compare else [("default", dict(temperature=1.0, top_k=10))]

    for name, kw in settings:
        g = torch.Generator(device=device).manual_seed(1337)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = generate(model, start, args.tokens, generator=g, **kw)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        text = tok.decode(out[0].tolist())
        rows.append((name, dt, args.tokens / dt, distinct_ngram_ratio(text)))
        blocks.append(f"### {name}\n\n```\n{text}\n```\n")

    print(f"{'setting':<12} {'seconds':>8} {'tok/s':>8} {'distinct-4gram':>15}")
    for name, dt, tps, div in rows:
        print(f"{name:<12} {dt:>8.2f} {tps:>8.1f} {div:>15.3f}")

    Path("experiments").mkdir(exist_ok=True)
    header = "| setting | tok/s | distinct-4gram |\n|---|---|---|\n" + "".join(
        f"| {n} | {t:.1f} | {d:.3f} |\n" for n, _, t, d in rows
    )
    Path("experiments/samples.md").write_text(
        f"# Sampling comparison\n\n{header}\n" + "\n".join(blocks)
    )


if __name__ == "__main__":
    main()
