"""Binarize the frozen RYKO-1000 SFT package into uint16 token streams.

Phase-5 data path: every prompt/completion pair becomes
``encode(prompt) + encode(completion) + [EOS]`` — exactly the conditional
``ryko-bench/1`` measures (``السؤال: …\\nالجواب: option`` scored by
``score_option``), so SFT optimises the same distribution the evaluation
reads. No BOS, no padding, no packing decisions: a flat EOS-separated
stream read through ``numpy.memmap`` by ``ryko_lm.training.train``.

Deterministic by construction (tokenization is deterministic, no RNG), so
the ``.bin`` files are byte-rebuildable from the sha256-frozen SFT package
and are *not* committed — the manifest records their sha256 and the source
package sha256 instead (same receipt discipline as the corpus bins).

Usage::

    python -m ryko_lm.training.pack_sft \\
        --train data/training/ryko1000_sft_train.jsonl \\
        --dev data/training/ryko1000_sft_dev.jsonl \\
        --tokenizer artifacts/tokenizer/ryko-tokenizer.json \\
        --out-dir artifacts/sft_ryko_small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from ..tokenizer.loader import EOS_ID, RykoTokenizer

SCHEMA = "ryko1000-sft-bin/1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_file(tok: RykoTokenizer, jsonl_path: Path, out_path: Path) -> dict:
    """Encode every pair and stream uint16 tokens to ``out_path``."""
    pairs, tokens, skipped = 0, 0, []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf: list[int] = []
    with open(jsonl_path, encoding="utf-8") as fh, \
            open(out_path, "wb") as sink:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            p_ids = tok.encode(rec["prompt"])
            c_ids = tok.encode(rec["completion"])
            ids = p_ids + c_ids
            if not p_ids or not c_ids or len(ids) > 2048:
                skipped.append(rec.get("id", f"line{pairs + 1}"))
                continue
            buf.extend(ids)
            buf.append(EOS_ID)
            pairs += 1
            tokens += len(ids) + 1
            if len(buf) >= 1_000_000:
                sink.write(np.asarray(buf, dtype=np.uint16).tobytes())
                buf.clear()
        if buf:
            sink.write(np.asarray(buf, dtype=np.uint16).tobytes())
    return {"pairs": pairs, "tokens": tokens, "skipped": skipped,
            "sha256": _sha256(out_path), "bytes": out_path.stat().st_size}


def build(train_path: Path, dev_path: Path, tokenizer_path: Path,
          out_dir: Path) -> dict:
    tok = RykoTokenizer.from_file(tokenizer_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for name, src in (("train", train_path), ("val", dev_path)):
        # "val" — the filename contract of ryko_lm.training.train
        out = out_dir / f"{name}.bin"
        stats = pack_file(tok, Path(src), out)
        files[name] = {"path": str(out), "source": str(src),
                       "source_sha256": _sha256(Path(src)), **stats}
    manifest = {
        "schema": SCHEMA,
        "layout": "encode(prompt) + encode(completion) + [EOS] per pair, "
                  "flat uint16 stream — matches score_option's conditional",
        "tokenizer_sha256": _sha256(Path(tokenizer_path)),
        "files": files,
        "tokens_total": files["train"]["tokens"] + files["val"]["tokens"],
    }
    (out_dir / "pack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ryko_pack_sft",
        description="Binarize the RYKO-1000 SFT package (phase 5).")
    p.add_argument("--train", type=Path,
                   default=Path("data/training/ryko1000_sft_train.jsonl"))
    p.add_argument("--dev", type=Path,
                   default=Path("data/training/ryko1000_sft_dev.jsonl"))
    p.add_argument("--tokenizer", type=Path,
                   default=Path("artifacts/tokenizer/ryko-tokenizer.json"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("artifacts/sft_ryko_small"))
    args = p.parse_args(argv)
    manifest = build(args.train, args.dev, args.tokenizer, args.out_dir)
    print(f"[pack] train {manifest['files']['train']['pairs']} pairs "
          f"{manifest['files']['train']['tokens']:,} tokens | "
          f"val {manifest['files']['val']['pairs']} pairs "
          f"{manifest['files']['val']['tokens']:,} tokens "
          f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
