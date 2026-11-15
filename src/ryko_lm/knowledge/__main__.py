"""Knowledge Core CLI::

    python -m ryko_lm.knowledge ingest --format shamela --jsonl data/raw/shamela.jsonl --db data/knowledge/ryko.sqlite
    python -m ryko_lm.knowledge stats   --db data/knowledge/ryko.sqlite
    python -m ryko_lm.knowledge search  --db data/knowledge/ryko.sqlite "الماء الطهور" -k 5
    python -m ryko_lm.knowledge cite    --db data/knowledge/ryko.sqlite "نص الاقتباس"
    python -m ryko_lm.knowledge receipt --db data/knowledge/ryko.sqlite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ryko_lm.knowledge import ingest as ingest_mod
from ryko_lm.knowledge.citation import verify_quote
from ryko_lm.knowledge.store import KnowledgeStore


def _open_db(path: str) -> KnowledgeStore:
    if path == ":memory:":
        return KnowledgeStore(":memory:")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return KnowledgeStore(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryko_knowledge",
        description="RYKO Knowledge Core — schema, ingest, hybrid search, citation gate.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="ingest a collector JSONL into the store")
    p_ing.add_argument("--db", required=True)
    p_ing.add_argument("--jsonl", type=Path, required=True)
    p_ing.add_argument("--format", choices=["shamela", "openiti", "fikr"], default="shamela")
    p_ing.add_argument("--no-classify", action="store_true",
                       help="skip rule-based discourse labelling")

    p_stats = sub.add_parser("stats", help="store statistics")
    p_stats.add_argument("--db", required=True)

    p_search = sub.add_parser("search", help="hybrid retrieval with provenance")
    p_search.add_argument("--db", required=True)
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    p_search.add_argument("--book", default=None, help="restrict to a book_id")

    p_cite = sub.add_parser("cite", help="verify a quote against the store")
    p_cite.add_argument("--db", required=True)
    p_cite.add_argument("quote")
    p_cite.add_argument("--book", default=None)

    p_rec = sub.add_parser("receipt", help="emit a provenance receipt for the store")
    p_rec.add_argument("--db", required=True)
    p_rec.add_argument("--out", type=Path, default=None)

    p_seed = sub.add_parser("ideas-seed",
                            help="seed the claims table from the fikr manifest")
    p_seed.add_argument("--db", required=True)
    p_seed.add_argument("--manifest", type=Path,
                        default=Path("data/knowledge/fikr_manifest.json"))

    p_ideas = sub.add_parser("ideas", help="search the idea layer (claims)")
    p_ideas.add_argument("--db", required=True)
    p_ideas.add_argument("query")
    p_ideas.add_argument("-k", type=int, default=8)

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        store = _open_db(args.db)
        fn = (ingest_mod.ingest_shamela_jsonl if args.format == "shamela"
              else ingest_mod.ingest_fikr_jsonl if args.format == "fikr"
              else ingest_mod.ingest_openiti_jsonl)
        report = fn(store, args.jsonl, classify=not args.no_classify)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[knowledge] stats: {json.dumps(store.stats(), ensure_ascii=False)}")
        return 0

    if args.cmd == "stats":
        store = KnowledgeStore(args.db)
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "search":
        store = KnowledgeStore(args.db)
        hits = store.search(args.query, k=args.k, mode=args.mode, book_id=args.book)
        print(json.dumps([h.as_dict() for h in hits], ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "cite":
        store = KnowledgeStore(args.db)
        verdict = verify_quote(store, args.quote, book_id=args.book)
        print(verdict.as_json())
        return 0 if verdict.status in {"verified", "partial"} else 1

    if args.cmd == "receipt":
        from ryko_lm.provenance import make_receipt

        store = KnowledgeStore(args.db)
        meta = store.stats()
        out = args.out or (Path(args.db).with_suffix(".receipt.json"))
        receipt = make_receipt(Path(args.db), kind="knowledge", meta=meta)
        out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[receipt] {meta} -> {out}")
        print(f"[receipt] sha256={receipt['receipt_sha256'][:16]}...")
        return 0

    if args.cmd == "ideas-seed":
        from ryko_lm.knowledge.ideas import load_manifest, seed_ideas

        store = _open_db(args.db)
        manifest = load_manifest(args.manifest)
        report = seed_ideas(store, manifest)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[knowledge] stats: {json.dumps(store.stats(), ensure_ascii=False)}")
        return 0

    if args.cmd == "ideas":
        from ryko_lm.knowledge.ideas import search_ideas

        store = KnowledgeStore(args.db)
        hits = search_ideas(store, args.query, k=args.k)
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
