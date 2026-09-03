# gpt-from-scratch

A decoder-only transformer (GPT-2 architecture) implemented from scratch in PyTorch,
trained on character-level Shakespeare, then used as a controlled testbed for
component ablations and inference-latency experiments.

The point of this repo is not that the model works. It is what the measurements say.

## Headline results

| Experiment | Result |
|---|---|
| Baseline model | 3.2M params, val loss **1.5571** (uniform guessing: 4.1744) |
| Residual connections ablation | **+1.80 val loss** — model collapses below the unigram baseline |
| KV caching at batch 1 | **0.73–1.04x** — no speedup; the GPU was never compute-bound |
| KV caching at batch 32 (25M model) | **7.48x** — 165 → 5,480 tokens/sec |
| Launch-bound probe | 127x more work costs **1.1x more time** |
| Greedy decoding | distinct-4gram **0.252** vs 0.930 for top-k — visible repetition loops |

All numbers produced on a single Colab T4, fp32 inference, fp16 mixed-precision
training. Raw data in [`experiments/`](experiments/).

## What I built

- Character tokenizer and positionally-split train/val loader (`src/data.py`)
- Multi-head causal self-attention, computed as one batched op rather than a
  loop over heads (`src/model.py`)
- Pre-norm transformer blocks with switchable LayerNorm / residuals /
  positional embeddings, so ablations are a config flag rather than an edit
- Training loop with AMP + GradScaler, gradient clipping, and per-run JSON
  experiment logs (`src/train.py`)
- Greedy, temperature, top-k and top-p sampling (`src/generate.py`)
- KV cache in two implementations — concatenation-based and preallocated —
  with equality tests against naive recomputation (`src/model.py`, `src/benchmark.py`)
- Benchmark harness with warmup and median-of-N timing (`src/benchmark.py`, `src/sweep.py`)
- 45 tests covering causality, off-by-one targets, cache correctness, and
  sampling edge cases (`tests/`)

## Why it is built this way

**Ablation flags live in the config dataclass, not in the code.** Each ablation
run is one CLI argument against an identical seed and identical hyperparameters.
A test asserts each flag actually changes model output — without it, a flag that
silently did nothing would produce three identical results and a false conclusion.

**Every experiment writes JSON.** Every claim in this README traces to a file
in `experiments/`.

**Tests target silent failures, not crashes.** The bugs that matter here do not
raise. A broken causal mask lets the model read the answer from the next
position: loss collapses, everything looks great, generation is noise.
An off-by-one in the target shifts the whole task. Both are one assertion each.

## Experiment 1: which components are load-bearing?

3,000 iterations, identical seed and hyperparameters, one component removed per run.

| configuration | val loss | vs baseline | vs unigram (3.3091) |
|---|---|---|---|
| baseline | 1.5571 | — | −1.7520 |
| no layer normalization | 1.7929 | +0.2358 | −1.5161 |
| no positional embeddings | 1.8466 | +0.2894 | −1.4625 |
| no residual connections | 3.3592 | **+1.8020** | **+0.0501** |

![ablation curves](experiments/ablations.png)

The final numbers understate what happened to the no-residual run. Its validation
loss moved 0.0055 across 2,750 iterations — it stopped learning at step 250 and
never resumed, while every other configuration was still descending at 3,000.

Comparing against the entropy of the character unigram distribution (3.3091 nats)
makes the failure precise: without residual connections the model ends up
*worse* than a static table of letter frequencies. It extracted nothing from
context at all. Each block computes `x = f(x)` instead of `x = x + f(x)`, so
the input is destroyed rather than refined, and gradients must survive four full
Jacobians to reach the embedding table.

Residuals are load-bearing. LayerNorm and positional embeddings are refinements
worth ~0.25 nats each at this scale.

**Methodology note:** this is an equal-compute-budget comparison, not a
converged-model comparison. All runs, including baseline, were still improving
at 3,000 iterations. The no_layernorm curve was steeper than baseline at the
end, so its gap would likely narrow with a longer budget.

## Experiment 2: when is KV caching worth it?

Naive autoregressive decoding recomputes the entire prefix at every step, so
generating *n* tokens does O(n²) work. Caching the keys and values makes it
O(n). Textbook. My first benchmark showed it was **slower**.

| tokens | naive | cached | speedup |
|---|---|---|---|
| 16 | 56.7 ms | 68.1 ms | 0.83x |
| 32 | 89.2 ms | 91.5 ms | 0.98x |
| 64 | 178.2 ms | 180.7 ms | 0.99x |
| 127 | 357.9 ms | 489.8 ms | 0.73x |

The cached output was bit-identical to naive, so the implementation was correct.
The diagnostic was in the naive column: per-token throughput was flat at
~355 tok/s regardless of sequence length. If recomputation were really quadratic,
throughput would fall as the prefix grew.

A direct probe confirmed it — a single forward pass, batch 1:

| model | T=1 | T=127 | ratio |
|---|---|---|---|
| 3.2M params | 3.417 ms | 3.775 ms | **1.1x** |
| 25M params | 6.226 ms | 7.632 ms | 1.23x |

**127x the arithmetic for 10% more wall-clock time.** The GPU was idle between
kernel launches, not computing. Caching removed work that was already free,
then added per-step `torch.cat` calls on the critical path.

So the question became: what pushes this into a regime where caching matters?
Not sequence length — batch size.

| model | batch | naive | cached | speedup |
|---|---|---|---|---|
| 3.2M | 1 | 390.9 ms | 375.1 ms | 1.04x |
| 3.2M | 8 | 442.2 ms | 438.7 ms | 1.01x |
| 3.2M | 32 | 836.3 ms | 402.5 ms | 2.08x |
| 25M | 1 | 855.1 ms | 770.1 ms | 1.11x |
| 25M | 8 | 1464.4 ms | 750.3 ms | 1.95x |
| 25M | 32 | 5544.8 ms | 741.6 ms | **7.48x** |

The cached column for the 25M model is flat: **770 → 750 → 742 ms**. Generating
127 tokens for 32 sequences costs the same wall-clock time as for one.
Throughput rises from 165 tok/s to 5,480 tok/s — 33x — on identical hardware.

That is the argument for continuous batching, measured rather than assumed:
batching is nearly free until the GPU becomes compute-bound, and caching is what
keeps per-step work constant so it stays free.

KV cache memory grows linearly with batch — 134 MB at 25M params, batch 32,
block size 128 — which is why cache memory, not weights, becomes the binding
constraint on serving concurrency at production scale.

## Experiment 3: sampling and repetition

250 tokens, fixed seed, from the trained baseline.

| setting | distinct-4gram |
|---|---|
| greedy | 0.252 |
| temperature 0.5 | 0.842 |
| temperature 1.0 | 0.930 |
| temperature 1.5 | 0.963 |
| top-k 10 | 0.930 |
| top-p 0.9 | 0.886 |

Greedy decoding locks into repetition loops — visible in
[`experiments/samples.md`](experiments/samples.md) as the same line of dialogue
repeating verbatim. Temperature 1.5 scores highest on diversity while producing
the least coherent text, so this metric measures novelty, not quality. It is
useful for detecting degeneracy, not for ranking output.

## What failed

**My first latency benchmark was confounded by GPU warmup.** Greedy decoding
measured slowest despite being the cheapest method, purely because it ran first
and absorbed CUDA context initialization. Fixed with 3 warmup iterations and
median-of-7 timing.

**My preallocated KV cache was slower than the naive `torch.cat` version**
(0.76x at batch 1) and never beat it at any configuration. The intent was to
remove the per-step reallocation and copy that `cat` performs. Untested
hypothesis: `cache.update()` returns a slice of a larger buffer, and feeding a
non-contiguous view into cuBLAS makes PyTorch materialize a contiguous copy —
reintroducing the copy I was trying to remove, plus a large upfront allocation.
Confirming this needs `torch.profiler`; not done. Kept in the repo as a
negative result.

**The first KV cache result looked like a bug and was not.** The temptation was
to assume the implementation was wrong. Tests said otherwise, which is what made
the launch-bound explanation findable.

## Limitations

- Learned absolute positional embeddings cap generation at `block_size` (128).
  Real systems use RoPE or sliding-window attention.
- Character-level tokenization, chosen to keep the vocabulary small and the
  focus on architecture. Not comparable to BPE-based models on any benchmark.
- No run trained to convergence; all comparisons are at a fixed 3,000-iteration budget.
- Single GPU, single hardware target. Launch-bound conclusions are specific to a T4.

## Next

Serve a 0.5B model under vLLM and compare against these hand-rolled numbers —
in particular whether the batch-size crossover point moves, and where the
KV cache memory ceiling caps concurrency.

## Reproduce

```bash
pip install -r requirements.txt
wget -O data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python -m pytest tests/ -q          # 45 tests
python -m src.train --run-name baseline
python -m src.generate --compare
python -m src.benchmark
python -m src.sweep
python -m src.train --run-name no_residual --no-residual
python -m src.report
python -m src.analyze_ablations
```
