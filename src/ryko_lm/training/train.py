"""Train a Ryko language model.

Production-style training loop sized for both CPU smoke runs and GPU runs:

* AdamW with weight decay (norm weights excluded), cosine LR with warmup.
* Gradient accumulation, gradient clipping, seeded RNG, exact resume.
* Periodic evaluation on the held-out split with checkpointing (last + best).
* JSONL metrics log consumed by ``ryko_lm.training.plot_loss``.

Data: flat ``uint16`` token arrays (``train.bin`` / ``val.bin``) read through
``numpy.memmap``; windows are sampled at random offsets — no padding, no masks.

Usage (resumable; re-run the same command until it reports completion):
    python -m ryko_lm.training.train --config configs/models/ryko-tiny.yaml \
        --data-dir data/processed --out-dir artifacts/training \
        [--total-steps 2400] [--time-budget 520] [--reset]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..model.ryko_model import RykoConfig, RykoLM


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class TokenDataset:
    """Random fixed windows from a flat uint16 token file."""

    def __init__(self, path: Path, block_size: int) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return max(1, len(self.data) - self.block_size - 1)

    def batch(self, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        ix = rng.integers(0, len(self.data) - self.block_size - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(self.data[i: i + self.block_size].astype(np.int64))
                         for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1: i + 1 + self.block_size].astype(np.int64))
                         for i in ix])
        return x, y


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
def lr_at(step: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.dim() < 2 else decay).append(param)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: RykoLM, dataset: TokenDataset, batch_size: int,
             max_batches: int, rng: np.random.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(max_batches):
        x, y = dataset.batch(batch_size, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    full = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_cfg = RykoConfig.from_dict(full["model"])
    tcfg = dict(full.get("train", {}))

    batch_size = int(tcfg.get("batch_size", 8))
    if args.batch_size:
        batch_size = int(args.batch_size)
    grad_accum = int(tcfg.get("grad_accum", 1))
    lr = float(tcfg.get("lr", 3e-4))
    min_lr = float(tcfg.get("min_lr", 3e-5))
    warmup = int(tcfg.get("warmup_steps", 100))
    weight_decay = float(tcfg.get("weight_decay", 0.1))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    total_steps = int(args.total_steps or tcfg.get("total_steps", 2400))
    eval_every = int(tcfg.get("eval_every", 200))
    seed = int(tcfg.get("seed", 1337))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)

    train_data = TokenDataset(Path(args.data_dir) / "train.bin", model_cfg.block_size)
    val_data = TokenDataset(Path(args.data_dir) / "val.bin", model_cfg.block_size)

    model = RykoLM(model_cfg)
    optimizer = make_optimizer(model, lr, weight_decay)

    step, best_val = 0, float("inf")
    tokens_seen_resume = None
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if last_path.exists() and not args.reset:
        try:
            blob = torch.load(last_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(
                f"[error] checkpoint '{last_path}' is unreadable ({type(exc).__name__}: {exc}). "
                f"Restore weights from best.pt / final.pt or delete the file to start fresh.",
                flush=True,
            )
            return 2
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        step = blob["step"]
        # Derive the sampling stream from the resumed step: re-seeding with the
        # bare seed would replay the SAME windows the model already saw in
        # earlier sessions — with frequent restarts this trains the model to
        # memorise a fixed prefix of the data (train loss drops, real quality
        # collapses). seed + step guarantees a fresh stream every session.
        rng = np.random.default_rng(seed + step)
        best_val = blob.get("best_val", float("inf"))
        tokens_seen_resume = blob.get("tokens_seen")
        if args.fresh_best:  # dataset changed: old best is not comparable
            best_val = float("inf")
            print("[resume] best_val reset (--fresh-best)", flush=True)
        print(f"[resume] step={step} best_val={best_val:.4f}", flush=True)

    model.train()
    deadline = time.time() + args.time_budget if args.time_budget else None
    t0 = time.time()
    start_step = step
    eval_every = int(args.eval_every or eval_every)
    # tokens_seen is persisted in the checkpoint so that changing batch size
    # mid-run keeps the accounting exact; the fallback covers older checkpoints.
    tokens_seen = tokens_seen_resume if tokens_seen_resume is not None else \
        step * batch_size * grad_accum * model_cfg.block_size

    def save_last() -> None:
        """Persist model + optimizer so interrupted sessions resume exactly."""
        blob = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "best_val": best_val, "total_steps": total_steps,
                "tokens_seen": tokens_seen}
        # atomic write: a kill mid-save must never truncate the destination
        tmp = last_path.with_suffix(".pt.tmp")
        torch.save(blob, tmp)
        os.replace(tmp, last_path)

    while step < total_steps:
        if deadline and time.time() > deadline:
            save_last()  # checkpoint on graceful stop, even between evals
            print(f"[budget] time budget reached at step {step}", flush=True)
            break

        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(grad_accum):
            x, y = train_data.batch(batch_size, rng)
            _, loss = model(x, y)
            (loss / grad_accum).backward()
            loss_total += loss.item() / grad_accum

        lr_now = lr_at(step, total_steps, warmup, lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        step += 1
        tokens_seen += batch_size * grad_accum * model_cfg.block_size

        if step % 50 == 0:
            elapsed = time.time() - t0
            session_tokens = (step - start_step) * batch_size * grad_accum * model_cfg.block_size
            print(
                f"step {step}/{total_steps} loss {loss_total:.4f} "
                f"lr {lr_now:.2e} tok/s {session_tokens / max(elapsed, 1e-9) / 1000:.0f}k "
                f"({elapsed:.0f}s)",
                flush=True,
            )

        if step % eval_every == 0 or step == total_steps:
            val_loss = evaluate(model, val_data, batch_size, args.eval_batches, rng)
            record = {"step": step, "train_loss": round(loss_total, 4),
                      "val_loss": round(val_loss, 4), "lr": lr_now,
                      "tokens": tokens_seen, "elapsed_s": round(time.time() - t0, 1)}
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            print(f"[eval] {json.dumps(record)}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                model.save(best_path, half=True)
            save_last()

    finished = step >= total_steps
    if finished:
        model.save(out_dir / "final.pt", half=True)
        val_loss = evaluate(model, val_data, batch_size, args.eval_batches, rng)
        summary = {"status": "completed", "steps": step, "final_val_loss": round(val_loss, 4),
                   "perplexity": round(math.exp(val_loss), 2),
                   "params": model.num_params(), "tokens": tokens_seen}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[done] {json.dumps(summary)}", flush=True)
    else:
        print(f"[paused] step {step}/{total_steps} — rerun the same command to resume", flush=True)
    return 0 if finished else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a Ryko model (resumable).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override batch_size from the config (experiments)")
    parser.add_argument("--time-budget", type=int, default=None,
                        help="stop gracefully after N seconds (checkpoint is kept)")
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=None,
                        help="override eval_every from the config")
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    parser.add_argument("--reset", action="store_true", help="ignore existing checkpoint")
    parser.add_argument("--fresh-best", action="store_true",
                        help="reset best_val on resume (use when the dataset changed)")
    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
