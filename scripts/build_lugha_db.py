#!/usr/bin/env python3
"""Build the لغة (Linguistics) layer of the Knowledge Core.

Pipeline (idempotent — safe to re-run after any interruption):

    1. filter  — keep only records whose title is ``available`` in the manifest
    2. ingest  — page-level passages with provenance under the ``lugha`` source
    3. seed    — claims from ``ideas`` (``--replace`` drops manifest-sourced
                 claims of previous manifest versions first)
    4. core    — the committed-DB shrink: lugha books are kept only under
                 ``--core-max-chars`` per book (matn + medium classics) so the
                 repo artifact stays under GitHub's 100 MB blob limit; every
                 book stays in the FULL DB with ``--no-core``
    5. receipt — regenerate the provenance receipt

Usage:
    python scripts/build_lugha_db.py                    # core (committed) DB
    python scripts/build_lugha_db.py --no-core          # keep every fetched book
    python scripts/build_lugha_db.py --core-max-chars 2500000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ryko_lm.knowledge import ingest as ingest_mod  # noqa: E402
from ryko_lm.knowledge import schema as sch  # noqa: E402
from ryko_lm.knowledge.ideas import load_manifest, prune_fikr, seed_ideas  # noqa: E402
from ryko_lm.knowledge.store import KnowledgeStore  # noqa: E402
from ryko_lm.provenance import make_receipt  # noqa: E402


# Canonical matn core — always kept in the committed DB (subject to the
# per-book cap), regardless of the greedy size ordering: these are the
# teaching matn the citation gate and RAG demos must verify against.
LUGHA_FORCE_KEEP: frozenset[str] = frozenset({
    "المقدمة الآجرومية",
    "متن قطر الندى وبل الصدى",
    "متن شذور الذهب في معرفة كلام العرب",
    "الكافية في علم النحو",
    "ألفية ابن مالك — الخلاصة",
    "لامية الأفعال",
    "الشافية في علم التصريف مع الوافية",
    "شذا العرف في فن الصرف",
    "الإيضاح في علل النحو",
    "أسرار البلاغة",
    "قواعد الإملاء",
})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=Path("data/knowledge/ryko.sqlite"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/knowledge/lugha_manifest.json"))
    ap.add_argument("--raw", type=Path, nargs="*",
                    default=[Path("data/raw/lugha_shamela.jsonl")])
    ap.add_argument("--no-core", action="store_true",
                    help="keep ALL fetched lugha books (full DB)")
    ap.add_argument("--claims-only", action="store_true",
                    help="committed-DB mode under the 100MiB wall (v1.17.0 "
                         "heritage ceiling): seed the lugha claims + register "
                         "book/author metadata rows WITHOUT passages — the "
                         "passages live in the full local DB only")
    ap.add_argument("--core-max-chars", type=int, default=600_000,
                    help="core keeps only lugha books under this many chars")
    ap.add_argument("--core-total-chars", type=int, default=5_500_000,
                    help="committed-DB budget for the lugha core")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    available = {b["title"] for b in manifest["books"] if b["status"] == "available"}

    if not args.db.exists():
        args.db.parent.mkdir(parents=True, exist_ok=True)
    store = KnowledgeStore(str(args.db))
    if "lugha" not in {r["source_id"] for r in store.conn.execute("SELECT source_id FROM sources")}:
        store.add_source(
            "lugha",
            "طبقة اللغة — كتب النحو والصرف والبلاغة والمعاجم واللسانيات والأدب والمقررات (مرآة الشاملة)",
            "MIT declared by dataset publisher",
            "declared-open",
        )

    # 1) filter
    kept, skipped = 0, 0
    records = []
    for raw_path in args.raw:
        if not raw_path.exists():
            print(f"[filter] missing raw file (skip): {raw_path}")
            continue
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("book") in available:
                records.append(rec)
                kept += 1
            else:
                skipped += 1
    print(f"[filter] {kept} in-scope records, {skipped} out-of-scope skipped")

    # 2) prune (scope changes) + ingest
    rep = prune_fikr(store, manifest)
    print(f"[prune] {rep}")
    if args.claims_only:
        # 100MiB-wall mode: catalog metadata only (books/authors/claims),
        # no passages — the full local DB carries the passages.
        n_books = 0
        for b in manifest["books"]:
            if b["status"] != "available":
                continue
            author = b.get("author")
            author_id = None
            if author:
                author_id = sch.stable_id("a", author)
                store.add_author(author_id, author,
                                 death_year_hijri=b.get("death_hijri"),
                                 meta={"school": b["school"]})
            # same id convention as the collector (shamela-<digits>) so the
            # committed catalog rows and the full local DB agree and a later
            # passage ingest is idempotent
            digits = b["shamela_path"].rsplit("/", 1)[1].split("__")[0]
            store.add_book(f"shamela-{digits}", "lugha", b["title"],
                           author_id=author_id,
                           category=f"لغة/{b['school']}",
                           rights_status="declared-open",
                           meta={"catalog": True, "shamela_path": b["shamela_path"]})
            n_books += 1
        store.conn.commit()
        print(f"[claims-only] registered {n_books} lugha catalog books (no passages)")
    else:
        ing = ingest_mod.ingest_lugha_jsonl(store, records, classify=True)
        print(f"[ingest] books={ing['books']} passages=+{ing['passages']} "
              f"chars=+{ing['chars'] / 1e6:.1f}M dups={ing['duplicates_skipped']}")

    # 3) seed claims (قواعد وتعريفات طبقة اللغة)
    seeded = seed_ideas(store, manifest, replace=True)
    print(f"[seed] {seeded}")

    # 4) core shrink (size policy, lugha source only) — skipped in
    # claims-only mode (no lugha passages in the committed DB at all)
    if not args.no_core and not args.claims_only:
        rows = store.conn.execute(
            "SELECT b.book_id AS book_id, b.title_ar AS title_ar, "
            "COALESCE(SUM(p.chars), 0) AS chars FROM books b "
            "LEFT JOIN passages p ON p.book_id = b.book_id "
            "WHERE b.source_id='lugha' GROUP BY b.book_id, b.title_ar").fetchall()
        # Deterministic core policy (lugha source only):
        #   force-keep    — the canonical matn (LUGHA_FORCE_KEEP) enter the
        #                   committed DB first, per-book <= force cap;
        #   greedy fill   — the remaining total budget is spent on the
        #                   smallest books so the DB stays under GitHub's
        #                   100 MB blob limit (the base DB is already ~86 MB).
        force_cap = min(args.core_max_chars, 500_000)
        ordered = sorted(rows, key=lambda r: (r["chars"], r["book_id"]))
        budget = args.core_total_chars
        kept_ids: set[str] = set()
        for r in ordered:
            if r["title_ar"] in LUGHA_FORCE_KEEP and r["chars"] <= force_cap:
                kept_ids.add(r["book_id"])
                budget -= r["chars"]
        for r in ordered:
            if r["book_id"] in kept_ids:
                continue
            if r["chars"] <= args.core_max_chars and budget >= r["chars"]:
                kept_ids.add(r["book_id"])
                budget -= r["chars"]
        dropped, kept_n = [], 0
        for r in rows:
            if r["book_id"] in kept_ids:
                kept_n += 1
            else:
                dropped.append(r["book_id"])
        for bid in dropped:
            store.conn.execute("DELETE FROM passages WHERE book_id=?", (bid,))
            store.conn.execute("DELETE FROM books WHERE book_id=?", (bid,))
        store.conn.execute(
            "DELETE FROM authors WHERE author_id NOT IN "
            "(SELECT DISTINCT author_id FROM books WHERE author_id IS NOT NULL)")
        store.conn.commit()
        store.conn.execute("VACUUM")
        kept_chars = sum(r["chars"] for r in rows if r["book_id"] in kept_ids)
        print(f"[core] kept {kept_n} lugha books ({kept_chars / 1e6:.1f}M chars, "
              f"per-book <= {args.core_max_chars:,}, total <= {args.core_total_chars:,}), "
              f"dropped {len(dropped)} (full DB variant: --no-core)")

    # 5) receipt
    meta = store.stats()
    out = args.db.with_name(args.db.name.replace(".sqlite", ".receipt.json"))
    receipt = make_receipt(args.db, kind="knowledge", meta=meta)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[receipt] {meta} -> {out}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
