"""Evaluate a Ryko checkpoint: held-out loss / perplexity + generation samples.

Deterministic protocol:

* the perplexity pass slices ``val.bin`` into **sequential, non-overlapping**
  windows (no RNG) so every evaluation of the same split is byte-comparable;
* generation uses the model's KV-cache ``generate_iter`` with fixed seeds.

Usage:
    python -m ryko_lm.eval.evaluate \
        --config configs/models/ryko-tiny.yaml \
        --checkpoint artifacts/training_v3/last.pt \
        --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
        --val-bin data/processed_v3/val.bin \
        --out docs/results/eval_v3.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..model.ryko_model import RykoConfig, RykoLM
from ..tokenizer.loader import RykoTokenizer

DEFAULT_PROMPTS = [
    "قال رسول الله صلى الله عليه وسلم",
    "قال الإمام الشافعي في كتابه",
    "ومن فوائد هذا الباب",
    "معنى هذه الكلمة في اللغة",
    "وذهب الحنفية إلى أن",
]


@torch.no_grad()
def heldout_loss(model: RykoLM, val_bin: Path, block_size: int,
                 max_windows: int = 0, batch: int = 8,
                 skip_windows: int = 0) -> tuple[float, int]:
    """Mean cross-entropy over sequential non-overlapping windows."""
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    n_windows = (len(data) - 1) // block_size
    if max_windows:
        n_windows = min(n_windows, max_windows)
    n_windows = max(0, n_windows - skip_windows)
    if n_windows < 1:
        raise ValueError(f"val file too small: {len(data)} tokens")
    losses: list[float] = []
    for start in range(skip_windows, skip_windows + n_windows, batch):
        take = min(batch, skip_windows + n_windows - start)
        x = np.stack([data[(start + i) * block_size:(start + i + 1) * block_size]
                      for i in range(take)]).astype(np.int64)
        y = np.stack([data[(start + i) * block_size + 1:(start + i + 1) * block_size + 1]
                      for i in range(take)]).astype(np.int64)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        losses.append(loss.item() * take)
    mean_loss = sum(losses) / n_windows
    return float(mean_loss), n_windows


@torch.no_grad()
def sample_prompts(model: RykoLM, tok: RykoTokenizer, prompts: list[str],
                   max_new_tokens: int = 96, temperature: float = 0.8,
                   top_k: int = 50, seed: int = 1234,
                   eos_id: int | None = 2) -> list[dict]:
    results = []
    for i, prompt in enumerate(prompts):
        ids = tok.tokenizer.encode(prompt).ids
        idx = torch.tensor([[tok.bos_id] + ids], dtype=torch.long)
        gen = torch.Generator()
        gen.manual_seed(seed + i)
        out_ids: list[int] = []
        for t in model.generate_iter(idx, max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_k=top_k,
                                     eos_id=eos_id, generator=gen):
            token = int(t.item()) if torch.is_tensor(t) else int(t)
            if eos_id is not None and token == eos_id:
                break
            out_ids.append(token)
        text = tok.decode(out_ids)
        results.append({"prompt": prompt, "generation": text, "tokens": len(out_ids)})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Ryko checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--val-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=0,
                        help="cap windows (0 = full split)")
    parser.add_argument("--skip-windows", type=int, default=0,
                        help="skip the first N windows (distribution probes)")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args(argv)

    cfg = RykoConfig.from_dict(yaml.safe_load(args.config.read_text(encoding="utf-8"))["model"])
    model = RykoLM.load(args.checkpoint, device="cpu", config=cfg)
    model.eval()
    tok = RykoTokenizer.from_file(args.tokenizer)

    t0 = time.time()
    loss, n_windows = heldout_loss(model, args.val_bin, cfg.block_size,
                                   max_windows=args.max_windows,
                                   skip_windows=args.skip_windows)
    ppl = math.exp(loss)
    samples = sample_prompts(model, tok, DEFAULT_PROMPTS,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature)

    result = {
        "checkpoint": str(args.checkpoint),
        "val_bin": str(args.val_bin),
        "tag": args.tag,
        "heldout_loss": round(loss, 4),
        "perplexity": round(ppl, 2),
        "windows": int(n_windows),
        "window_size": cfg.block_size,
        "eval_seconds": round(time.time() - t0, 1),
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"},
                     ensure_ascii=False), flush=True)
    for s in samples[:2]:
        print(f"[sample] {s['prompt']} -> {s['generation'][:120]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
