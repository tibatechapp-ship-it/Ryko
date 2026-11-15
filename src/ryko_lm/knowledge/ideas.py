"""Islamic-Thought idea layer (العقل الفكري) over the Knowledge Core claims table.

The research map of الفكر الإسلامي (12 schools, ~300 distinctive ideas) is
encoded in ``data/knowledge/fikr_manifest.json``.  This module:

1. **seeds** the ``claims`` table from the manifest's ``ideas`` list — every
   idea becomes a claim with deterministic id ``stable_id('c', ...)``, school
   and originator attribution in ``meta`` (idempotent: re-seeding is a no-op);
2. **links** the manifest's available books to ``authors`` rows so the graph
   schools -> authors -> books -> passages is queryable;
3. **searches** ideas with a deterministic hybrid score: 0.55 · cosine
   (blake2b feature-hashing embeddings) + 0.45 · token Jaccard
   (Arabic normalizing tokenizer) — no TF-IDF state needed at claim scale.

CLI:
    python -m ryko_lm.knowledge ideas-seed --db data/knowledge/ryko.sqlite \
        --manifest data/knowledge/fikr_manifest.json
    python -m ryko_lm.knowledge ideas --db data/knowledge/ryko.sqlite "الحاكمية" -k 5
"""

from __future__ import annotations

import json
from array import array
from pathlib import Path

from ryko_lm.knowledge import schema as sch
from ryko_lm.knowledge.embeddings import cosine, embed
from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.knowledge.textproc import ar_tokens

W_DENSE = 0.55
W_OVERLAP = 0.45


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("kind") != "ryko-fikr-manifest":
        raise ValueError(f"not a fikr manifest: {manifest.get('kind')!r}")
    if manifest.get("version") != 1:
        raise ValueError(f"unsupported manifest version: {manifest.get('version')!r}")
    for school in manifest["schools"]:
        for field in ("id", "name", "central_idea"):
            if not school.get(field):
                raise ValueError(f"school missing {field}: {school}")
    for idea in manifest["ideas"]:
        for field in ("text", "originator", "school"):
            if not idea.get(field):
                raise ValueError(f"idea missing {field}: {idea}")
    school_ids = {s["id"] for s in manifest["schools"]}
    for idea in manifest["ideas"]:
        if idea["school"] not in school_ids:
            raise ValueError(f"idea references unknown school: {idea['school']!r}")
    return manifest


def seed_ideas(store: KnowledgeStore, manifest: dict) -> dict:
    """Insert manifest ideas as claims (idempotent) + link available authors."""
    seeded = linked_authors = 0
    for idea in manifest["ideas"]:
        claim_id = store.add_claim(
            idea["text"],
            claim_type=idea.get("type"),
            confidence=1.0,  # attribution taken verbatim from the research map
            meta={
                "school": idea["school"],
                "originator": idea["originator"],
                "source": "fikr_manifest",
                "manifest_version": manifest["version"],
            },
        )
        if claim_id:
            seeded += 1
    for book in manifest.get("books", []):
        author = book.get("author")
        if not author:
            continue
        author_id = sch.stable_id("a", author)
        store.add_author(author_id, author,
                         death_year_hijri=book.get("death_hijri"),
                         meta={"school": book.get("school")})
        linked_authors += 1
        # back-link existing books ingested from the fikr collector
        store.conn.execute(
            "UPDATE books SET author_id=COALESCE(author_id, ?) "
            "WHERE title_ar=? AND author_id IS NULL",
            (author_id, book["title"]),
        )
    store.conn.commit()
    return {"claims_seeded": seeded, "authors_linked": linked_authors}


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def search_ideas(store: KnowledgeStore, query: str, k: int = 8) -> list[dict]:
    """Deterministic hybrid ranking over the claims table."""
    rows = [dict(r) for r in store.conn.execute(
        "SELECT claim_id, claim_text, claim_type, confidence, meta FROM claims")]
    q_vec: array = embed(query)
    q_toks = ar_tokens(query)
    scored = []
    for row in rows:
        dense = cosine(q_vec, embed(row["claim_text"]))
        overlap = _jaccard(q_toks, ar_tokens(row["claim_text"]))
        score = W_DENSE * dense + W_OVERLAP * overlap
        meta = json.loads(row["meta"]) if row["meta"] else {}
        scored.append({
            "claim_id": row["claim_id"],
            "claim_text": row["claim_text"],
            "claim_type": row["claim_type"],
            "originator": meta.get("originator"),
            "school": meta.get("school"),
            "score": round(score, 6),
        })
    scored.sort(key=lambda r: (-r["score"], r["claim_id"]))
    return scored[:k]
