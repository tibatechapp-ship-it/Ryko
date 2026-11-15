"""Binarize the cleaned corpus into fixed-width uint16 token arrays.

Layout
------
``train.bin`` / ``val.bin`` are flat ``uint16`` arrays of token ids.  The
training loop slices fixed windows of ``block_size`` tokens out of them with
``numpy.memmap`` — zero-copy, no padding, no attention masks: the classic
nanoGPT-style data path, which keeps CPU training simple and fast.

Special tokens: every paragraph (one per line in ``train.txt``) ends with
``<eos>``; documents (blank-line separated) start with ``<bos>``.

Usage:
    python -m ryko_lm.data.binarize \
        --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
        --in-dir data/processed --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..tokenizer.loader import BOS_ID, EOS_ID, RykoTokenizer


def _encode_file(tok: RykoTokenizer, path: Path, out_path: Path,
                 batch_lines: int = 2000) -> int:
    """Encode ``path`` and stream uint16 tokens to ``out_path`` (memory-safe).

    Uses Rust-side batch encoding; BOS marks the first line of each document
    (blank-line separated), every line ends with EOS.  Returns token count.
    """
    t0 = time.time()
    total = 0
    in_document = False
    batch: list[str] = []
    first_flags: list[bool] = []

    def flush_batch() -> None:
        nonlocal total
        if not batch:
            return
        buf: list[int] = []
        for ids, is_first in zip(tok.encode_batch(batch), first_flags):
            if is_first:
                buf.append(BOS_ID)
            buf.extend(ids)
            buf.append(EOS_ID)
        if buf:
            sink.write(np.asarray(buf, dtype=np.uint16).tobytes())
            total += len(buf)
        batch.clear()
        first_flags.clear()

    with path.open(encoding="utf-8") as src, out_path.open("wb") as sink:
        for raw in src:
            if not raw.strip():  # document separator
                in_document = False
                continue
            first_flags.append(not in_document)
            batch.append(raw.rstrip("\n"))
            in_document = True
            if len(batch) >= batch_lines:
                flush_batch()
        flush_batch()
    print(f"[binarize] {path.name}: {total:,} tokens in {time.time() - t0:.1f}s", flush=True)
    return total


def binarize(tokenizer_path: Path, in_dir: Path, out_dir: Path) -> dict:
    tok = RykoTokenizer.from_file(tokenizer_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {"vocab_size": tok.vocab_size}
    for split in ("train", "val"):
        src = in_dir / f"{split}.txt"
        if not src.exists():
            continue
        out = out_dir / f"{split}.bin"
        n_tokens = _encode_file(tok, src, out)
        stats[split] = {"tokens": int(n_tokens), "file": str(out)}

    (out_dir / "binarize_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] {json.dumps({k: v for k, v in stats.items() if k != 'vocab_size'})}", flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Binarize cleaned corpus to uint16 arrays.")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/ryko-tokenizer.json"))
    parser.add_argument("--in-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)
    binarize(args.tokenizer, args.in_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
