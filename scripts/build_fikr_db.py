#!/usr/bin/env python3
"""Build the منقحة fikr layer of the Knowledge Core from a raw collection.

Pipeline (idempotent — safe to re-run after any interruption):

    1. filter  — keep only records whose title is ``available`` in the manifest
    2. prune   — drop fikr books listed in ``policy.excluded_books`` (v2 تنقيح)
    3. ingest  — page-level passages with provenance (chunked, deduped)
    4. seed    — claims from ``ideas`` (``--replace`` drops manifest-sourced
                 claims of previous manifest versions first)
    5. core    — ``--core`` keeps only the committed-DB core titles
                 (seed + reading-order + kept classics) so the repo artifact
                 stays under GitHub's 100 MB blob limit; the FULL DB keeps
                 every in-scope book locally
    6. receipt — regenerate the provenance receipt

Usage:
    python scripts/build_fikr_db.py                      # core (committed) DB
    python scripts/build_fikr_db.py --no-core            # full DB (all in-scope)
    python scripts/build_fikr_db.py --db data/knowledge/ryko-full.sqlite --no-core
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ryko_lm.knowledge import ingest as ingest_mod  # noqa: E402
from ryko_lm.knowledge.ideas import load_manifest, prune_fikr, seed_ideas  # noqa: E402
from ryko_lm.knowledge.store import KnowledgeStore  # noqa: E402
from ryko_lm.provenance import make_receipt  # noqa: E402

# Committed-DB core: the seed + the منقحة reading order + the classics already
# committed since v1.7.0.  Everything else lives in the full local DB.
CORE_TITLES = {
    # reading order (docs/ISLAMIC_THOUGHT.md §6)
    "المنقذ من الضلال",
    "تهافت الفلاسفة",
    "الرد على المنطقيين",
    "الموافقات في أصول الشريعة",
    "مدارج السالكين",
    "درء تعارض العقل والنقل",
    "تاريخ ابن خلدون — كتاب العبر",
    # kept since v1.7.0 (السياسة الشرعية + المقاصد)
    "الأحكام السلطانية",
    "السياسة الشرعية في إصلاح الراعي والرعية",
    "الطرق الحكمية في السياسة الشرعية",
    "إعلام الموقعين عن رب العالمين",
    "الاعتصام",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=Path("data/knowledge/ryko.sqlite"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/knowledge/fikr_manifest.json"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw/fikr_shamela.jsonl"))
    ap.add_argument("--no-core", action="store_true",
                    help="keep ALL in-scope books (full DB, ~250 MB)")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    available = {b["title"] for b in manifest["books"] if b["status"] == "available"}

    if not args.db.exists():
        args.db.parent.mkdir(parents=True, exist_ok=True)
    store = KnowledgeStore(str(args.db))
    if "fikr" not in {r["source_id"] for r in store.conn.execute("SELECT source_id FROM sources")}:
        store.add_source(
            "fikr",
            "الفكر الإسلامي المنقح — كتب المدارس الأربع (مرآة الشاملة)",
            "MIT declared by dataset publisher",
            "declared-open",
        )

    # 1) filter
    kept, skipped = 0, 0
    records = []
    for line in args.raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("book") in available:
            records.append(rec)
            kept += 1
        else:
            skipped += 1
    print(f"[filter] {kept} in-scope records, {skipped} out-of-scope skipped")

    # 2) prune + 3) ingest
    rep = prune_fikr(store, manifest)
    print(f"[prune] {rep}")
    ing = ingest_mod.ingest_fikr_jsonl(store, records, classify=True)
    print(f"[ingest] books={ing['books']} passages=+{ing['passages']} "
          f"chars=+{ing['chars'] / 1e6:.1f}M dups={ing['duplicates_skipped']}")

    # 4) seed claims (منقحة)
    seeded = seed_ideas(store, manifest, replace=True)
    print(f"[seed] {seeded}")

    # 5) core shrink
    if not args.no_core:
        rows = store.conn.execute(
            "SELECT book_id, title_ar FROM books WHERE source_id='fikr'").fetchall()
        dropped = [r["book_id"] for r in rows if r["title_ar"] not in CORE_TITLES]
        for bid in dropped:
            store.conn.execute("DELETE FROM passages WHERE book_id=?", (bid,))
            store.conn.execute("DELETE FROM books WHERE book_id=?", (bid,))
        store.conn.execute(
            "DELETE FROM authors WHERE author_id NOT IN "
            "(SELECT DISTINCT author_id FROM books WHERE author_id IS NOT NULL)")
        store.conn.commit()
        store.conn.execute("VACUUM")
        print(f"[core] kept {len(rows) - len(dropped)} fikr books, "
              f"dropped {len(dropped)} (full DB variant: --no-core)")

    # 6) receipt
    meta = store.stats()
    out = args.db.with_name(args.db.name.replace(".sqlite", ".receipt.json"))
    receipt = make_receipt(args.db, kind="knowledge", meta=meta)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[receipt] {meta} -> {out}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
