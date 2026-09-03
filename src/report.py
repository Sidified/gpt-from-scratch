"""Collect ablation runs into a comparison table and loss-curve plot.

Run:  python -m src.report
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = ["baseline", "no_pos_emb", "no_residual", "no_layernorm"]
LABELS = {
    "baseline": "baseline (all components)",
    "no_pos_emb": "no positional embeddings",
    "no_residual": "no residual connections",
    "no_layernorm": "no layer normalization",
}


def load(name):
    p = Path(f"experiments/{name}.json")
    return json.loads(p.read_text()) if p.exists() else None


def main():
    runs = {n: r for n in RUNS if (r := load(n))}
    if "baseline" not in runs:
        raise SystemExit("baseline.json missing — run training first")

    base_val = runs["baseline"]["final_val_loss"]
    uniform = math.log(runs["baseline"]["model_config"]["vocab_size"])

    lines = [
        "| configuration | params | final val loss | vs baseline | vs random |",
        "|---|---|---|---|---|",
    ]
    for name, r in runs.items():
        v = r["final_val_loss"]
        broken = not math.isfinite(v)
        shown = "diverged (NaN)" if broken else f"{v:.4f}"
        delta = "—" if name == "baseline" else ("—" if broken else f"+{v - base_val:.4f}")
        gain = "—" if broken else f"{uniform - v:.4f}"
        lines.append(f"| {LABELS[name]} | {r['params']:,} | {shown} | {delta} | {gain} |")

    table = "\n".join(lines)
    print(table)
    print(f"\nrandom-guess baseline (ln vocab_size) = {uniform:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in runs.items():
        h = r["history"]
        xs = [p["iter"] for p in h]
        ys = [p["val"] if math.isfinite(p["val"]) else None for p in h]
        ax.plot(xs, ys, marker="o", markersize=3, label=LABELS[name])
    ax.axhline(uniform, ls="--", c="grey", lw=1, label="random guessing")
    ax.set_xlabel("iteration")
    ax.set_ylabel("validation loss")
    ax.set_title("Component ablations, 3M-param char-level GPT (tinyshakespeare)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("experiments/ablations.png", dpi=150)

    Path("experiments/ablations.md").write_text(
        f"# Ablation results\n\nAll runs: 3000 iterations, identical seed, "
        f"identical hyperparameters. Only the named component differs.\n\n{table}\n\n"
        f"Random-guess baseline: {uniform:.4f}\n\n![](ablations.png)\n"
    )
    print("\nwrote experiments/ablations.md and ablations.png")


if __name__ == "__main__":
    main()
