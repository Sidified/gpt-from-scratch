"""Where does KV caching actually pay off?

Compares naive recompute, cat-based cache, and preallocated cache across
model sizes and batch sizes. Latency is independent of weight values, so
randomly initialised models are used — no training required.

Run:  python -m src.sweep
"""

import json
from pathlib import Path

import torch

from .config import GPTConfig
from .model import GPT, KVCache
from .benchmark import generate_naive, generate_cached, timeit


@torch.no_grad()
def generate_prealloc(model, idx, max_new_tokens):
    """Greedy decoding with a preallocated cache."""
    model.eval()
    cache = KVCache(model.config, idx.size(0), idx.device)
    logits, _ = model(idx, cache=cache)
    for _ in range(max_new_tokens):
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        logits, _ = model(next_id, cache=cache)
    return idx


def launch_bound_probe(model, device):
    """If a 127-token forward costs the same as a 1-token forward, the model
    is bound by kernel launch overhead, not arithmetic."""
    out = {}
    for T in (1, 127):
        x = torch.randint(0, model.config.vocab_size, (1, T), device=device)
        med, _, _ = timeit(lambda: model(x), warmup=3, repeats=7)
        out[f"T{T}_ms"] = round(med * 1000, 3)
    out["ratio"] = round(out["T127_ms"] / out["T1_ms"], 2)
    return out


SIZES = {
    "small":  dict(n_embd=256, n_head=4, n_layer=4),
    "medium": dict(n_embd=512, n_head=8, n_layer=8),
}
BATCHES = [1, 8, 32]
NEW_TOKENS = 127


def main():
    device = "cuda"
    torch.manual_seed(0)
    results, probes = [], {}

    for size_name, dims in SIZES.items():
        cfg = GPTConfig(vocab_size=65, block_size=128, dropout=0.0, **dims)
        model = GPT(cfg).to(device).eval()
        probes[size_name] = {"params": model.num_params(),
                             **launch_bound_probe(model, device)}

        for B in BATCHES:
            prompt = torch.zeros((B, 1), dtype=torch.long, device=device)
            timings = {}
            for label, fn in [
                ("naive",    lambda: generate_naive(model, prompt, NEW_TOKENS, temperature=0.0)),
                ("cat",      lambda: generate_cached(model, prompt, NEW_TOKENS, temperature=0.0)),
                ("prealloc", lambda: generate_prealloc(model, prompt, NEW_TOKENS)),
            ]:
                med, _, _ = timeit(fn, warmup=2, repeats=5)
                timings[label] = med

            cache_mb = KVCache(cfg, B, device).nbytes() / 1e6
            results.append({
                "size": size_name, "params": model.num_params(), "batch": B,
                "naive_ms": round(timings["naive"] * 1000, 1),
                "cat_ms": round(timings["cat"] * 1000, 1),
                "prealloc_ms": round(timings["prealloc"] * 1000, 1),
                "cat_speedup": round(timings["naive"] / timings["cat"], 2),
                "prealloc_speedup": round(timings["naive"] / timings["prealloc"], 2),
                "kv_cache_mb": round(cache_mb, 2),
            })
            print(f"{size_name:>7} B={results[-1]['batch']:<3} "
                  f"naive {results[-1]['naive_ms']:>7.1f}ms  "
                  f"cat {results[-1]['cat_ms']:>7.1f}ms ({results[-1]['cat_speedup']:.2f}x)  "
                  f"prealloc {results[-1]['prealloc_ms']:>7.1f}ms "
                  f"({results[-1]['prealloc_speedup']:.2f}x)  "
                  f"cache {results[-1]['kv_cache_mb']:.1f}MB")
        del model
        torch.cuda.empty_cache()

    print("\nlaunch-bound probe (single forward pass, batch 1):")
    for name, p in probes.items():
        print(f"  {name:>7} {p['params']:>10,} params | "
              f"T=1 {p['T1_ms']:.3f}ms | T=127 {p['T127_ms']:.3f}ms | "
              f"ratio {p['ratio']}x")

    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/kv_sweep.json").write_text(json.dumps(
        {"results": results, "launch_bound_probe": probes,
         "new_tokens": NEW_TOKENS,
         "method": "median of 5 runs after 2 warmup, greedy, fp32, T4"},
        indent=2))


if __name__ == "__main__":
    main()
