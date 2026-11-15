"""Plot the training loss curve from ``artifacts/training/log.jsonl``.

Usage:
    python -m ryko_lm.training.plot_loss --log artifacts/training/log.jsonl \
        --out docs/results/loss_curve.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot Ryko training curves.")
    parser.add_argument("--log", type=Path, default=Path("artifacts/training/log.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/loss_curve.png"))
    parser.add_argument("--title", default="Ryko-Tiny Training Curve")
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("[error] empty log", file=sys.stderr)
        return 1

    steps = [r["step"] for r in records]
    train = [r["train_loss"] for r in records]
    val = [r["val_loss"] for r in records]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(steps, train, color="#2563eb", linewidth=1.8, label="train loss")
    ax.plot(steps, val, color="#dc2626", linewidth=1.8, marker="o", markersize=3.5, label="val loss")
    ax.set_xlabel("optimizer steps")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(args.title)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper right")
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
