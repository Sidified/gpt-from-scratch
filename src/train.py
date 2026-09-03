"""Training loop for the char-level GPT.

Run:  python -m src.train --run-name baseline
Ablations (Block 6) flip a flag:  python -m src.train --run-name no_residual --no-residual
"""

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import GPTConfig, TrainConfig
from .data import get_batch, load_data
from .model import GPT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_loss(model, splits, tcfg, mcfg, device, use_amp):
    """Average loss over several batches. eval() disables dropout so the
    number reflects the model, not the noise."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = get_batch(data, tcfg.batch_size, mcfg.block_size, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    tcfg = TrainConfig(max_iters=args.max_iters)
    set_seed(tcfg.seed)

    tokenizer, train_data, val_data = load_data(args.data)
    mcfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        use_layernorm=not args.no_layernorm,
        use_residual=not args.no_residual,
        use_pos_emb=not args.no_pos_emb,
    )

    model = GPT(mcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate)
    # fp16 on a T4: gradients can underflow, so the scaler multiplies the loss
    # up before backward and divides it back out before the optimizer step.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    splits = {"train": train_data, "val": val_data}
    history = []
    print(f"run={args.run_name}  device={device}  params={model.num_params():,}")
    print(f"layernorm={mcfg.use_layernorm} residual={mcfg.use_residual} pos_emb={mcfg.use_pos_emb}")

    t0 = time.time()
    for it in range(tcfg.max_iters + 1):
        if it % tcfg.eval_interval == 0 or it == tcfg.max_iters:
            losses = estimate_loss(model, splits, tcfg, mcfg, device, use_amp)
            elapsed = time.time() - t0
            history.append({"iter": it, **losses, "seconds": round(elapsed, 1)})
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {elapsed:6.1f}s"
            )

        x, y = get_batch(train_data, tcfg.batch_size, mcfg.block_size, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                      # undo scaling before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    Path("experiments").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    result = {
        "run_name": args.run_name,
        "model_config": asdict(mcfg),
        "train_config": asdict(tcfg),
        "params": model.num_params(),
        "final_train_loss": history[-1]["train"],
        "final_val_loss": history[-1]["val"],
        "total_seconds": round(time.time() - t0, 1),
        "history": history,
    }
    Path(f"experiments/{args.run_name}.json").write_text(json.dumps(result, indent=2))
    torch.save(
        {"model": model.state_dict(), "config": asdict(mcfg), "stoi": tokenizer.stoi},
        f"checkpoints/{args.run_name}.pt",
    )
    print(f"\nfinal val loss {result['final_val_loss']:.4f} in {result['total_seconds']}s")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--no-layernorm", action="store_true")
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--no-pos-emb", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
