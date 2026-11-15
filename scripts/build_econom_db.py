#!/usr/bin/env python3
"""Build the الاقتصاد والحقوق layer of the Knowledge Core from a raw collection.

Pipeline (idempotent — safe to re-run after any interruption):

    1. filter  — keep only records whose title is ``available`` in
                 ``data/knowledge/econom_manifest.json``
    2. prune   — drop econom books listed in ``policy.excluded_books``
                 (empty in v1 — kept for the تنقيح mechanism)
    3. ingest  — page-level passages under the ``econom`` source
    4. seed    — claims from the manifest's ``ideas`` (replace drops ONLY
                 econom-layer manifest claims — other layers untouched)
    5. core    — ``--core`` keeps only the committed-DB core titles so the
                 repo artifact stays under GitHub's 100 MB blob limit; the
                 FULL DB keeps every in-scope book locally
    6. receipt — regenerate the provenance receipt

Usage:
    python scripts/build_econom_db.py                      # core (committed) DB
    python scripts/build_econom_db.py --no-core            # full DB (all in-scope)
    python scripts/build_econom_db.py --db data/knowledge/ryko-full.sqlite --no-core
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

# Committed-DB core for the econom layer: the canonical start list of the
# research (التطبيق: كتاب الأموال ثم كتاب الخراج؛ الحقوق: حقوق الإنسان لجمل
# الليل) + the tools supports (الربا، مصارف الزكاة) — ~2.1M chars keeps the
# repo artifact well under GitHub's 100 MB blob limit.  كفاية الأخيار،
# الأموال لابن زنجويه، حقوق النبي، الزكاة في ضوء الكتاب والسنة stay in the
# full local DB (--no-core).
CORE_TITLES_ECONOM = {
    "كتاب الأموال",
    "كتاب الخراج",
    "الخراج ليحيى بن آدم",
    "حقوق الإنسان في الإسلام",
    "مصارف الزكاة في الإسلام",
    "الربا",
    "الاقتصاد الإسلامي",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=Path("data/knowledge/ryko.sqlite"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/knowledge/econom_manifest.json"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw/econom_shamela.jsonl"))
    ap.add_argument("--no-core", action="store_true",
                    help="keep ALL in-scope econom books (full DB)")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    available = {b["title"] for b in manifest["books"] if b["status"] == "available"}

    if not args.db.exists():
        args.db.parent.mkdir(parents=True, exist_ok=True)
    store = KnowledgeStore(str(args.db))
    if "econom" not in {r["source_id"] for r in store.conn.execute("SELECT source_id FROM sources")}:
        store.add_source(
            "econom",
            "الاقتصاد الإسلامي والحقوق — كتب المال والحق (مرآة الشاملة)",
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
    ing = ingest_mod.ingest_econom_jsonl(store, records, classify=True)
    print(f"[ingest] books={ing['books']} passages=+{ing['passages']} "
          f"chars=+{ing['chars'] / 1e6:.1f}M dups={ing['duplicates_skipped']}")

    # 4) seed claims (econom layer only — fikr claims untouched)
    seeded = seed_ideas(store, manifest, replace=True)
    print(f"[seed] {seeded}")

    # 5) core shrink — econom books only
    if not args.no_core:
        rows = store.conn.execute(
            "SELECT book_id, title_ar FROM books WHERE source_id='econom'").fetchall()
        dropped = [r["book_id"] for r in rows if r["title_ar"] not in CORE_TITLES_ECONOM]
        for bid in dropped:
            store.conn.execute("DELETE FROM passages WHERE book_id=?", (bid,))
            store.conn.execute("DELETE FROM books WHERE book_id=?", (bid,))
        store.conn.execute(
            "DELETE FROM authors WHERE author_id NOT IN "
            "(SELECT DISTINCT author_id FROM books WHERE author_id IS NOT NULL)")
        store.conn.commit()
        store.conn.execute("VACUUM")
        print(f"[core] kept {len(rows) - len(dropped)} econom books, "
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
