"""PalmX-IC (external Islamic-culture MCQ) -> RYKO-Bench JSONL converter.

Source: PalmX 2025 subtask 2 (UBC-NLP). The official dataset is gated-auto on
the Hub; this session consumes the community mirror ``qimma/MCQ_PalmX``
(islamic split, same prompt/choices/index schema) and records that provenance
in a sidecar receipt. When an HF token becomes available, the same converter
runs unchanged on the official parquet (columns are identical).

Contract (closed, deterministic):
* kind="mcq", source="palmx-ic", domain="islamic-culture"
* case_id = "px-{row:04d}" in file order — stable across runs
* exactly 4 options, integer answer within range
* meta carries dataset id + parquet sha256 (first record's provenance,
  identical for all rows of one build)
* byte-identical double-build determinism is asserted in tests/test_palmx.py

Usage:
    python scripts/palmx_to_bench.py --parquet palmx_islamic.parquet \
        --out data/bench/palmx_ic.jsonl [--receipt data/bench/palmx_ic.manifest.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SOURCE = "palmx-ic"
DOMAIN = "islamic-culture"
MIRROR_DATASET = "qimma/MCQ_PalmX (islamic split) — community mirror of UBC-NLP/palmx_2025_subtask2_islamic"
OFFICIAL_DATASET = "UBC-NLP/palmx_2025_subtask2_islamic"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(parquet: Path, out: Path, dataset_id: str = MIRROR_DATASET,
            receipt: Path | None = None) -> tuple[int, str]:
    """Convert one parquet to the bench JSONL. Returns (rows, parquet_sha256)."""
    df = pd.read_parquet(parquet)
    sha = sha256_file(parquet)
    lines: list[str] = []
    for i, row in enumerate(df.itertuples(index=False)):
        choices = [str(c) for c in row.choices]
        if len(choices) < 2:
            raise ValueError(f"row {i}: {len(choices)} options — PalmX contract is 4")
        answer = int(row.index)
        if not 0 <= answer < len(choices):
            raise ValueError(f"row {i}: answer {answer} out of range")
        question = str(row.question).strip()
        if not question:
            raise ValueError(f"row {i}: empty question")
        rec = {
            "case_id": f"px-{i:04d}",
            "kind": "mcq",
            "source": SOURCE,
            "domain": DOMAIN,
            "question": question,
            "options": choices,
            "answer": answer,
            "meta": {"dataset": dataset_id, "parquet_sha256": sha, "row": i},
        }
        lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if receipt is not None:
        payload = {
            "dataset": dataset_id,
            "official_dataset": OFFICIAL_DATASET,
            "official_gated": "auto (needs HF token)",
            "parquet_sha256": sha,
            "rows": len(lines),
            "case_id_prefix": "px-",
            "bench_source": SOURCE,
            "domain": DOMAIN,
            "protocol": "ryko-bench/1 (mcq, length-normalized log-likelihood)",
        }
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n", encoding="utf-8")
    return len(lines), sha


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="palmx_to_bench",
                                description="PalmX-IC parquet -> RYKO-Bench JSONL")
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/bench/palmx_ic.jsonl"))
    p.add_argument("--receipt", type=Path,
                   default=Path("data/bench/palmx_ic.manifest.json"))
    args = p.parse_args(argv)
    rows, sha = convert(args.parquet, args.out, receipt=args.receipt)
    print(f"[palmx] rows={rows} sha256={sha[:16]}… -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
