"""What did each ablated model actually learn?

Compares final val loss against two reference points: uniform guessing, and
the entropy of the character unigram distribution — the loss a model would
reach by learning marginal character frequencies and nothing about context.
"""

import json
import math
from collections import Counter
from pathlib import Path

import torch

from .data import load_data
from .generate import load_checkpoint, generate, distinct_ngram_ratio

RUNS = ["baseline", "no_pos_emb", "no_residual", "no_layernorm"]


def unigram_entropy(data) -> float:
    counts = Counter(data.tolist())
    total = sum(counts.values())
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def main():
    tok, train_data, _ = load_data("data/input.txt")
    h_uniform = math.log(tok.vocab_size)
    h_unigram = unigram_entropy(train_data)

    print(f"uniform guessing       {h_uniform:.4f} nats")
    print(f"unigram distribution   {h_unigram:.4f} nats  "
          f"(context-free upper bound on a working model)\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, blocks = [], []

    for name in RUNS:
        ckpt = Path(f"checkpoints/{name}.pt")
        meta = Path(f"experiments/{name}.json")
        if not (ckpt.exists() and meta.exists()):
            continue

        model, t = load_checkpoint(str(ckpt), device)
        g = torch.Generator(device=device).manual_seed(1337)
        start = torch.tensor([[t.stoi["\n"]]], dtype=torch.long, device=device)
        text = t.decode(generate(model, start, 250, temperature=1.0,
                                 top_k=10, generator=g)[0].tolist())

        val = json.loads(meta.read_text())["final_val_loss"]
        rows.append({
            "run": name,
            "val_loss": round(val, 4),
            "vs_unigram": round(val - h_unigram, 4),
            "beats_unigram": bool(val < h_unigram),
            "distinct_4gram": round(distinct_ngram_ratio(text), 3),
        })
        blocks.append(f"### {name}  (val {val:.4f})\n\n```\n{text}\n```\n")

        verdict = "uses context" if val < h_unigram else "NO better than character frequencies"
        print(f"{name:<14} val {val:.4f}  vs unigram {val - h_unigram:+.4f}  -> {verdict}")

    Path("experiments").mkdir(exist_ok=True)
    table = ("| run | val loss | vs unigram | distinct-4gram |\n|---|---|---|---|\n" +
             "".join(f"| {r['run']} | {r['val_loss']} | {r['vs_unigram']:+} | "
                     f"{r['distinct_4gram']} |\n" for r in rows))
    Path("experiments/ablation_samples.md").write_text(
        f"# What each ablated model learned\n\n"
        f"- uniform guessing: {h_uniform:.4f} nats\n"
        f"- unigram distribution: {h_unigram:.4f} nats\n\n{table}\n\n" + "\n".join(blocks)
    )
    Path("experiments/entropy_reference.json").write_text(json.dumps(
        {"uniform_nats": round(h_uniform, 4),
         "unigram_nats": round(h_unigram, 4), "runs": rows}, indent=2))
    print("\nwrote experiments/ablation_samples.md")


if __name__ == "__main__":
    main()
