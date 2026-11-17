#!/usr/bin/env python3
"""Measure knowledge-core serving latency on the current store (CPU).

Reusables the v1.14.0 measurement style: warm index, median over repeats,
three search modes + citation verify + the /knowledge/ask audit path.
Reports honest wall-clock numbers for docs/COMPETITIVE.md and CHANGELOG.

Usage:
    python scripts/perf_knowledge.py --db data/knowledge/ryko.sqlite \
        --checkpoint artifacts/training_v5c/final.pt \
        --tokenizer artifacts/tokenizer/ryko-tokenizer.json --repeats 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ryko_lm.knowledge.citation import verify_quote  # noqa: E402
from ryko_lm.knowledge.store import KnowledgeStore  # noqa: E402
from ryko_lm.knowledge.store import _Index  # noqa: E402

QUERIES = [
    "الزكاة دين المال في الإسلام",
    "الإسناد المتصل شرط صحة الحديث",
    "قاعدة لا ضرر ولا ضرار",
    "معنى العقل عند المتكلمين",
    "أحكام الوقف وحبس المال",
    "المقاصد الخمس في الشريعة",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=Path("data/knowledge/ryko.sqlite"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("artifacts/training_v5c/final.pt"))
    ap.add_argument("--tokenizer", type=Path,
                    default=Path("artifacts/tokenizer/ryko-tokenizer.json"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    store = KnowledgeStore(str(args.db), persist_vectors=False)
    stats = store.stats()

    # warm index (first build includes vector computation — excluded)
    t0 = time.perf_counter()
    index = _Index(store.conn, persist_vectors=False)
    warm_s = time.perf_counter() - t0

    timings: dict[str, list[float]] = {}
    for mode in ("bm25", "dense", "hybrid"):
        timings[mode] = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            for q in QUERIES:
                store.search(q, k=5, mode=mode)
            timings[mode].append((time.perf_counter() - t0) / len(QUERIES) * 1000)

    # citation verify on a real passage
    row = store.conn.execute(
        "SELECT p.text FROM passages p JOIN books b ON p.book_id=b.book_id "
        "WHERE b.source_id='turath' LIMIT 1").fetchone()
    quote = (row["text"][:180].strip()) if row else "لا شيء"
    cite_ms = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        verify_quote(store, quote)
        cite_ms.append((time.perf_counter() - t0) * 1000)

    report = {
        "store": {"db": str(args.db), **{k: stats[k] for k in
                                         ("passages", "books", "sources")}},
        "warm_index_seconds": round(warm_s, 2),
        "median_ms_per_query": {k: round(statistics.median(v), 1)
                                for k, v in timings.items()},
        "cite_verify_ms": round(statistics.median(cite_ms), 1),
        "queries": len(QUERIES),
        "repeats": args.repeats,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
