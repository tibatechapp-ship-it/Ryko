#!/usr/bin/env python3
"""Build the turath (wide stratified selection) layer of the Knowledge Core.

Pipeline (idempotent — safe to re-run after any interruption):

    1. load    — read the wide selection JSONL (``collect_shamela --per-cat 12
                 --max-books 600 --max-file-mb 3 --keep-pages``, the same
                 collection the corpus pipeline merged into v5.2)
    2. select  — ``--mode full`` ingests every collected record into the FULL
                 local store; ``--mode core`` applies the budgeted-breadth
                 policy so the committed DB stays under GitHub's 100 MB blob
                 limit (2.5M TEXT chars ≈ +7 MB calibrated on the 89.2 MB
                 committed store)
    3. ingest  — page-level passages with provenance under the ``turath``
                 source, namespaced ids (``turath-<dataset id>``),
                 category ``تراث/<cat>`` — never touching the two منقحة
                 layers' claims or relations (read-only on ``claims``)
    4. vacuum  — last step before commit; ``passage_vec`` stays derived/empty
    5. receipt — regenerate the provenance receipt (no search in between, so
                 the receipt sha256 matches the committed bytes)

Usage:
    python scripts/build_turath_db.py                     # core (committed) DB
    python scripts/build_turath_db.py --mode full         # full local store
    python scripts/build_turath_db.py --budget-chars 2500000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ryko_lm.knowledge import ingest as ingest_mod  # noqa: E402
from ryko_lm.knowledge.store import KnowledgeStore  # noqa: E402
from ryko_lm.knowledge.turath import (  # noqa: E402
    build_manifest,
    select_turath_core,
    write_manifest,
)
from ryko_lm.provenance import make_receipt  # noqa: E402

COLLECTOR_DOC = {
    "module": "ryko_lm.data.collect_shamela",
    "args": "--per-cat 12 --max-books 600 --max-file-mb 3 --keep-pages",
    "selection": "famous-first per category, remaining slots by dataset order",
    "note": "same wide stratified selection merged into corpus v5.2 (v1.15.0)",
}

# Calibrated on the committed store: 89,227,264 bytes / 32.0M chars = 2.788
# bytes per TEXT char (UTF-8 + row + index overhead).  2.5M chars ≈ +7.0 MB
# → ~96.2 MB committed DB, safely under GitHub's 100 MB blob limit.
DEFAULT_BUDGET_CHARS = 2_500_000


def load_records(raw: Path) -> list[dict]:
    records = []
    with raw.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def store_turath_rows(store: KnowledgeStore) -> list[dict]:
    """Cumulative layer state from the store — the manifest is rebuilt from
    these on every run, so re-running a completed build is a byte-identical
    no-op for the manifest too."""
    rows = store.conn.execute(
        "SELECT b.book_id AS book_id, b.title_ar AS title, "
        "b.category AS category, b.meta AS meta, "
        "COALESCE(SUM(LENGTH(p.text)), 0) AS chars "
        "FROM books b LEFT JOIN passages p ON p.book_id = b.book_id "
        "WHERE b.source_id='turath' GROUP BY b.book_id ORDER BY b.book_id"
    ).fetchall()
    out = []
    for r in rows:
        meta = json.loads(r["meta"]) if r["meta"] else {}
        out.append({"book_id": r["book_id"], "title": r["title"],
                    "category": r["category"], "raw_pages": meta.get("raw_pages"),
                    "chars": int(r["chars"])})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("core", "full"), default="core")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/shamela_wide.jsonl"))
    ap.add_argument("--db", type=Path, default=None,
                    help="default: data/knowledge/ryko.sqlite (core) / "
                         "data/knowledge/ryko-wide-full.sqlite (full)")
    ap.add_argument("--manifest-out", type=Path, default=None)
    ap.add_argument("--budget-chars", type=int, default=DEFAULT_BUDGET_CHARS,
                    help="core mode only: TEXT-character budget (default 2.5M)")
    args = ap.parse_args(argv)

    if args.db is None:
        args.db = (Path("data/knowledge/ryko.sqlite") if args.mode == "core"
                   else Path("data/knowledge/ryko-wide-full.sqlite"))
    if args.manifest_out is None:
        args.manifest_out = (Path("data/knowledge/turath_manifest.json")
                             if args.mode == "core"
                             else Path("data/knowledge/turath_manifest_full.json"))

    records = load_records(args.raw)
    print(f"[load] {len(records)} wide records from {args.raw}")

    store = KnowledgeStore(str(args.db))
    existing = {r["book_id"] for r in
                store.conn.execute("SELECT book_id FROM books")}
    if args.mode == "core":
        spent = store.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(p.text)), 0) AS c FROM passages p "
            "JOIN books b ON p.book_id = b.book_id "
            "WHERE b.source_id='turath'").fetchone()["c"]
        selected, report = select_turath_core(records, existing,
                                              args.budget_chars,
                                              spent_chars=int(spent))
        collector = dict(COLLECTOR_DOC, budget_chars=args.budget_chars)
    else:
        selected = records
        report = {"policy": "full tier: every collected record (local store, "
                            "receipt committed, file rebuildable)",
                  "budget_chars": None,
                  "budget_spent_chars": None,
                  "records_available": len(records),
                  "records_skipped_existing": len(records) - len(
                      [r for r in records
                       if ingest_mod.turath_book_id(r.get("id", ""))
                       not in existing]),
                  "records_selected": len(selected),
                  "categories_available": len({r.get("cat") or ""
                                               for r in records}),
                  "categories_covered": len({r.get("cat") or ""
                                             for r in selected}),
                  "rounds": None}
        collector = dict(COLLECTOR_DOC)

    before = store.stats()
    ing = ingest_mod.ingest_turath_jsonl(store, selected, classify=True)
    after = store.stats()
    print(f"[ingest] books={ing['books']} passages=+{ing['passages']} "
          f"chars=+{ing['chars'] / 1e6:.1f}M dups={ing['duplicates_skipped']}")
    print(f"[store] {before} -> {after}")

    # vacuum LAST (v1.9.0 lesson) then receipt immediately — no search in
    # between so the receipt sha256 matches the committed bytes
    store.conn.execute("VACUUM")
    store.conn.commit()
    meta = store.stats()
    receipt_out = args.db.with_name(
        args.db.name.replace(".sqlite", ".receipt.json"))
    receipt = make_receipt(args.db, kind="knowledge", meta=meta)
    receipt_out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    # the manifest documents the cumulative layer state: cover stats come
    # from the store, not from this run's selection
    rows = store_turath_rows(store)
    report = dict(report, categories_covered=len(
        {r["category"] for r in rows if r["category"]}))
    manifest = build_manifest(rows, report, collector)
    write_manifest(manifest, args.manifest_out)
    print(f"[receipt] {receipt_out}")
    print(f"[manifest] {args.manifest_out}: "
          f"{manifest['counts']['books']} books / "
          f"{manifest['counts']['chars'] / 1e6:.1f}M chars (cumulative layer)")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
