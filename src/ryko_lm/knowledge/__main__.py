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
    p_ing.add_argument("--format", choices=["shamela", "openiti", "fikr", "econom"],
                       default="shamela")
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
                            help="seed the claims table from a curated manifest "
                                 "(fikr | econom)")
    p_seed.add_argument("--db", required=True)
    p_seed.add_argument("--manifest", type=Path,
                        default=Path("data/knowledge/fikr_manifest.json"))
    p_seed.add_argument("--replace", action="store_true",
                        help="drop this layer's manifest-sourced claims first")

    p_prune = sub.add_parser("fikr-prune",
                             help="remove books that fell outside the manifest's "
                                  "curated scope (layer-scoped)")
    p_prune.add_argument("--db", required=True)
    p_prune.add_argument("--manifest", type=Path,
                        default=Path("data/knowledge/fikr_manifest.json"))

    p_ideas = sub.add_parser("ideas", help="search the idea layers (claims)")
    p_ideas.add_argument("--db", required=True)
    p_ideas.add_argument("query")
    p_ideas.add_argument("-k", type=int, default=8)
    p_ideas.add_argument("--school", default=None,
                         help="restrict to one school/section id (e.g. naqd-hanbali, "
                              "econom-adawat)")
    p_ideas.add_argument("--layer", default=None,
                         help="restrict to one curated layer (fikr | econom)")

    p_rel_seed = sub.add_parser(
        "relations-seed",
        help="seed argument_relations from the relations manifest "
             "(referential integrity enforced)")
    p_rel_seed.add_argument("--db", required=True)
    p_rel_seed.add_argument("--manifest", type=Path,
                            default=Path("data/knowledge/relations_manifest.json"))
    p_rel_seed.add_argument("--replace", action="store_true",
                            help="drop all relations-manifest rows first")

    p_rel = sub.add_parser("relations",
                           help="list/search the argument-relations graph")
    p_rel.add_argument("--db", required=True)
    p_rel.add_argument("--layer", default=None,
                       help="restrict to one layer (fikr | econom)")
    p_rel.add_argument("--relation", default=None,
                       help="restrict to one verb (supports, responds, …)")
    p_rel.add_argument("--claim-id", default=None,
                       help="relations touching this claim (either endpoint)")
    p_rel.add_argument("-k", type=int, default=100)

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        store = _open_db(args.db)
        fn = (ingest_mod.ingest_shamela_jsonl if args.format == "shamela"
              else ingest_mod.ingest_fikr_jsonl if args.format == "fikr"
              else ingest_mod.ingest_econom_jsonl if args.format == "econom"
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
        report = seed_ideas(store, manifest, replace=args.replace)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[knowledge] stats: {json.dumps(store.stats(), ensure_ascii=False)}")
        return 0

    if args.cmd == "fikr-prune":
        from ryko_lm.knowledge.ideas import load_manifest, prune_fikr

        store = _open_db(args.db)
        manifest = load_manifest(args.manifest)
        report = prune_fikr(store, manifest)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[knowledge] stats: {json.dumps(store.stats(), ensure_ascii=False)}")
        return 0

    if args.cmd == "ideas":
        from ryko_lm.knowledge.ideas import search_ideas

        store = KnowledgeStore(args.db)
        hits = search_ideas(store, args.query, k=args.k,
                            schools={args.school} if args.school else None,
                            layers={args.layer} if args.layer else None)
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "relations-seed":
        from ryko_lm.knowledge.relations import (
            load_relations_manifest,
            relations_stats,
            seed_relations,
        )

        store = _open_db(args.db)
        manifest = load_relations_manifest(args.manifest)
        report = seed_relations(store, manifest, replace=args.replace)
        report["graph"] = relations_stats(store)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[knowledge] stats: {json.dumps(store.stats(), ensure_ascii=False)}")
        return 0

    if args.cmd == "relations":
        from ryko_lm.knowledge.relations import list_relations, relations_stats

        store = KnowledgeStore(args.db)
        hits = list_relations(store, layer=args.layer,
                              relation=args.relation,
                              claim_id=args.claim_id, k=args.k)
        print(json.dumps({"protocol": "ryko-relations/1",
                          "graph": relations_stats(store),
                          "relations": hits},
                         ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
