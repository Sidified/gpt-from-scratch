"""Naive vs KV-cached generation: correctness check and latency benchmark.

Run:  python -m src.benchmark
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .generate import apply_top_k, apply_top_p, load_checkpoint


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, temperature=1.0, top_k=None,
                    top_p=None, generator=None):
    """Prefill the prompt once, then feed one token at a time reusing the cache."""
    model.eval()
    logits, _, past = model(idx, use_cache=True)      # prefill
    logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            l = logits / temperature
            if top_k is not None:
                l = apply_top_k(l, top_k)
            if top_p is not None:
                l = apply_top_p(l, top_p)
            next_id = torch.multinomial(F.softmax(l, dim=-1), 1, generator=generator)

        idx = torch.cat([idx, next_id], dim=1)
        logits, _, past = model(next_id, past_kvs=past, use_cache=True)
        logits = logits[:, -1, :]

    return idx


@torch.no_grad()
def generate_naive(model, idx, max_new_tokens, temperature=1.0, top_k=None,
                   top_p=None, generator=None):
    """Recompute the entire prefix at every step."""
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.config.block_size :])
        logits = logits[:, -1, :]
        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            l = logits / temperature
            if top_k is not None:
                l = apply_top_k(l, top_k)
            if top_p is not None:
                l = apply_top_p(l, top_p)
            next_id = torch.multinomial(F.softmax(l, dim=-1), 1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def timeit(fn, warmup=3, repeats=7):
    """Warm the GPU, then take the median of several runs.

    Warmup exists because the first CUDA call pays context creation and kernel
    autotuning; without it the first configuration measured looks slowest
    regardless of how much work it does.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/baseline.pt")
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 127])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_checkpoint(args.checkpoint, device)
    prompt = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)

    # correctness first: greedy decoding must match exactly
    for n in args.lengths:
        a = generate_naive(model, prompt, n, temperature=0.0)
        b = generate_cached(model, prompt, n, temperature=0.0)
        assert torch.equal(a, b), f"cached output diverged from naive at {n} tokens"
    print("correctness: cached output identical to naive at all lengths\n")

    rows = []
    for n in args.lengths:
        nm, nlo, nhi = timeit(lambda: generate_naive(model, prompt, n, temperature=0.0))
        cm, clo, chi = timeit(lambda: generate_cached(model, prompt, n, temperature=0.0))
        rows.append({
            "tokens": n,
            "naive_ms": round(nm * 1000, 2),
            "cached_ms": round(cm * 1000, 2),
            "naive_tok_s": round(n / nm, 1),
            "cached_tok_s": round(n / cm, 1),
            "speedup": round(nm / cm, 2),
        })

    print(f"{'tokens':>7} {'naive ms':>10} {'cached ms':>10} "
          f"{'naive tok/s':>12} {'cached tok/s':>13} {'speedup':>8}")
    for r in rows:
        print(f"{r['tokens']:>7} {r['naive_ms']:>10.2f} {r['cached_ms']:>10.2f} "
              f"{r['naive_tok_s']:>12.1f} {r['cached_tok_s']:>13.1f} {r['speedup']:>7.2f}x")

    # KV cache memory cost at full context
    cfg = model.config
    cache_bytes = 2 * cfg.n_layer * cfg.n_head * cfg.block_size * (cfg.n_embd // cfg.n_head) * 4
    print(f"\nKV cache at full context, batch 1: {cache_bytes / 1024:.1f} KB (fp32)")

    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/kv_cache.json").write_text(json.dumps(
        {"results": rows, "kv_cache_bytes_at_full_context": cache_bytes,
         "method": "median of 7 runs after 3 warmup runs, greedy decoding"},
        indent=2))


if __name__ == "__main__":
    main()
