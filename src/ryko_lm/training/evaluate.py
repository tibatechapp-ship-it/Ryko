"""Evaluate a trained Ryko checkpoint: perplexity on a split + text samples.

Usage:
    python -m ryko_lm.training.evaluate --checkpoint artifacts/training_v5c/final.pt \
        --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
        --data-dir data/processed --prompts "اللغة العربية" "كان يا ما كان"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from ..model.ryko_model import RykoLM
from ..tokenizer.loader import RykoTokenizer
from .train import TokenDataset


@torch.no_grad()
def perplexity(model: RykoLM, dataset: TokenDataset, batch_size: int,
               max_batches: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    losses = []
    model.eval()
    for _ in range(max_batches):
        x, y = dataset.batch(batch_size, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    return math.exp(float(np.mean(losses)))


def sample_prompts(model: RykoLM, tok: RykoTokenizer, prompts: list[str],
                   max_new_tokens: int, temperature: float, seed: int) -> list[dict]:
    results = []
    for i, prompt in enumerate(prompts):
        generator = torch.Generator().manual_seed(seed + i)
        ids = tok.encode(prompt, add_bos=True)
        idx = torch.tensor([ids], dtype=torch.long)
        out = model.generate(
            idx, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=50, top_p=0.95, repetition_penalty=1.15, eos_id=tok.eos_id,
            generator=generator,
        )
        completion = tok.decode(out[0].tolist())
        results.append({"prompt": prompt, "completion": completion})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Ryko checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("artifacts/tokenizer/ryko-tokenizer.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--prompts", nargs="*", default=[
        "اللغة العربية",
        "كان يا ما كان",
        "تعلم الذكاء الاصطناعي",
    ])
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    model = RykoLM.load(args.checkpoint)
    tok = RykoTokenizer.from_file(args.tokenizer)

    data_file = args.data_dir / f"{args.split}.bin"
    if data_file.exists():
        ppl = perplexity(model, TokenDataset(data_file, model.cfg.block_size),
                         args.batch_size, args.max_batches)
        print(f"[ppl] {args.split}: {ppl:.2f}")
    else:
        ppl = None

    samples = sample_prompts(model, tok, args.prompts, args.max_new_tokens,
                             args.temperature, seed=1337)
    for sample in samples:
        print(f"\n[prompt] {sample['prompt']}\n[ryko]   {sample['completion']}")

    if args.out:
        report = {"checkpoint": str(args.checkpoint), "perplexity": ppl,
                  "temperature": args.temperature, "samples": samples}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
