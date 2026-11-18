#!/usr/bin/env python3
"""Build the linguistics pretraining pack (حزمة اللغة لـ ryko-small).

Streams the ``collect_lugha`` JSONL through the *standard* cleaning cascade
(``ryko_lm.data.clean_corpus``, identical options to corpus v5 for
comparability), deduplicates paragraphs with the same truncated-hash
fingerprints used by ``augment_corpus``, splits documents into train/val with
the same deterministic hash-bucket rule, and writes:

    data/processed_ling/train.txt   (paragraph per line, blank line per doc)
    data/processed_ling/val.txt
    data/processed_ling/stats.json  (per-section composition)
    data/processed_ling/receipt.json (signed provenance receipt)

The layout is byte-compatible with ``clean_corpus`` so ``binarize`` works
unchanged — this pack is the curriculum shard ready for a GPU (or CPU) warm
continuation of ryko-small/ryko-tiny exactly like the training_v5b shard.

Usage:
    python scripts/build_ling_pack.py --raw data/raw/lugha_shamela.jsonl \
        --out data/processed_ling
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ryko_lm.data.augment_corpus import doc_seed, fp_int  # noqa: E402
from ryko_lm.data.clean_corpus import CleanOptions, _clean_line  # noqa: E402
from ryko_lm.provenance import make_receipt  # noqa: E402

SECTION_IDS = ("nahw", "sarf", "balagha", "maajim", "fiqh-lugha", "adab", "munhaj")


def section_of(cat: str | None) -> str:
    """``'لغة/nahw'`` -> ``nahw`` (closed vocabulary, else ``other``)."""
    if cat and "/" in cat:
        sec = cat.rsplit("/", 1)[1]
        return sec if sec in SECTION_IDS else "other"
    return "other"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, nargs="*",
                    default=[Path("data/raw/lugha_shamela.jsonl")])
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed_ling"))
    ap.add_argument("--val-ratio", type=float, default=0.004,
                    help="fraction of documents diverted to val (same default "
                         "as augment_corpus)")
    ap.add_argument("--min-line-len", type=int, default=40)
    ap.add_argument("--min-arabic-letters", type=int, default=20)
    ap.add_argument("--min-arabic-ratio", type=float, default=0.55)
    args = ap.parse_args(argv)

    opts = CleanOptions(
        min_line_len=args.min_line_len,
        min_arabic_letters=args.min_arabic_letters,
        min_arabic_ratio=args.min_arabic_ratio,
        remove_diacritics=True,
        unify_alef=False,
        unify_yaa=False,
        taa_marbuta_to_haa=False,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_bucket = max(1, int(1 / args.val_ratio))

    seen: set[int] = set()
    per_section: dict[str, dict] = {}
    docs = {"train": 0, "val": 0}
    chars_kept = 0
    dup_paragraphs = 0

    with (args.out_dir / "train.txt").open("w", encoding="utf-8") as tr, \
            (args.out_dir / "val.txt").open("w", encoding="utf-8") as va:
        for raw_path in args.raw:
            if not raw_path.exists():
                print(f"[pack] missing raw file (skip): {raw_path}", flush=True)
                continue
            for line in raw_path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                sec = section_of(rec.get("cat"))
                st = per_section.setdefault(sec, {"docs": 0, "chars": 0,
                                                  "dup_paragraphs": 0})
                paragraphs: list[str] = []
                for page in rec.get("page_records") or [{"body": rec.get("text", "")}]:
                    body = (page.get("body") or "").strip()
                    if not body:
                        continue
                    for sub in body.splitlines():
                        if not sub.strip():
                            continue
                        cleaned = _clean_line(sub, opts)
                        if cleaned is not None:
                            paragraphs.append(cleaned)
                # paragraph dedup within the pack (first occurrence wins —
                # deterministic given the fixed record order)
                unique: list[str] = []
                for para in paragraphs:
                    h = fp_int(para)
                    if h in seen:
                        dup_paragraphs += 1
                        st["dup_paragraphs"] += 1
                        continue
                    seen.add(h)
                    unique.append(para)
                if not unique:
                    continue
                body = "\n".join(unique) + "\n\n"
                c = sum(len(p) for p in unique)
                chars_kept += c
                st["chars"] += c
                st["docs"] += 1
                if doc_seed(unique) % val_bucket == 0:
                    va.write(body)
                    docs["val"] += 1
                else:
                    tr.write(body)
                    docs["train"] += 1

    stats = {
        "pack": "linguistics-v1",
        "sections": per_section,
        "docs": docs,
        "chars_kept": chars_kept,
        "dup_paragraphs_dropped": dup_paragraphs,
        "val_ratio_new": args.val_ratio,
        "cleaning": {
            "min_line_len": args.min_line_len,
            "min_arabic_letters": args.min_arabic_letters,
            "min_arabic_ratio": args.min_arabic_ratio,
            "remove_diacritics": True,
        },
    }
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = make_receipt(args.out_dir, kind="corpus", meta={
        "pack": "linguistics-v1",
        "raw_sources": [str(p) for p in args.raw],
        "chars_kept": chars_kept,
        "docs": docs,
    })
    (args.out_dir / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("pack", "docs", "chars_kept",
                                            "dup_paragraphs_dropped")},
                     ensure_ascii=False))
    for sec, st in sorted(per_section.items()):
        print(f"[pack] {sec:10s} docs={st['docs']:3d} chars={st['chars'] / 1e6:6.2f}M "
              f"dup_dropped={st['dup_paragraphs']}")
    print(f"[pack] wrote {args.out_dir}/train.txt "
          f"({(args.out_dir / 'train.txt').stat().st_size / 1e6:.1f} MB) + "
          f"val.txt ({(args.out_dir / 'val.txt').stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
