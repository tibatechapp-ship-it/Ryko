"""Islamic-Philosophy idea layer (العقل الفكري المنقح) over the claims table.

The منقحة research map (4 schools, no disputed authors) is encoded in
``data/knowledge/fikr_manifest.json``.  Manifest **v2** carries an explicit
``policy`` block: excluded originators/books and the positioning principle
(«الفلسفة الإسلامية منطلق النموذج وحجته»).  This module:

1. **seeds** the ``claims`` table from the manifest's ``ideas`` list — every
   idea becomes a claim with deterministic id ``stable_id('c', ...)``, school
   and originator attribution in ``meta`` (idempotent: re-seeding is a no-op;
   ``replace=True`` drops every manifest-sourced claim first, for scope
   changes like the v1→v2 تنقيح);
2. **links** the manifest's available books to ``authors`` rows so the graph
   schools -> authors -> books -> passages is queryable;
3. **prunes** books that fell out of the curated scope (``policy.excluded_books``)
   — passages, claims and orphaned authors are removed;
4. **searches** ideas with a deterministic hybrid score: 0.55 · cosine
   (blake2b feature-hashing embeddings) + 0.45 · token Jaccard
   (Arabic normalizing tokenizer) — no TF-IDF state needed at claim scale.

CLI:
    python -m ryko_lm.knowledge ideas-seed --db data/knowledge/ryko.sqlite \
        --manifest data/knowledge/fikr_manifest.json [--replace]
    python -m ryko_lm.knowledge fikr-prune --db data/knowledge/ryko.sqlite
    python -m ryko_lm.knowledge ideas --db data/knowledge/ryko.sqlite "العقل والوحي" -k 5 \
        [--school naqd-hanbali]
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


SUPPORTED_MANIFEST_VERSIONS = (1, 2)


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("kind") != "ryko-fikr-manifest":
        raise ValueError(f"not a fikr manifest: {manifest.get('kind')!r}")
    if manifest.get("version") not in SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError(f"unsupported manifest version: {manifest.get('version')!r}")
    for school in manifest["schools"]:
        for field in ("id", "name", "central_idea"):
            if not school.get(field):
                raise ValueError(f"school missing {field}: {school}")
    school_ids = {s["id"] for s in manifest["schools"]}
    for idea in manifest["ideas"]:
        for field in ("text", "originator", "school"):
            if not idea.get(field):
                raise ValueError(f"idea missing {field}: {idea}")
        if idea["school"] not in school_ids:
            raise ValueError(f"idea references unknown school: {idea['school']!r}")
    for book in manifest.get("books", []):
        if book.get("school") not in school_ids:
            raise ValueError(f"book references unknown school: {book.get('school')!r}")
    policy = manifest.get("policy", {})
    if manifest.get("version", 1) >= 2:
        if not policy.get("principle"):
            raise ValueError("manifest v2 requires policy.principle")
        # تنقيح guarantee: no excluded originator sneaks back into the ideas
        banned = policy.get("excluded_originators", [])
        for idea in manifest["ideas"]:
            for name in banned:
                if name in idea["originator"]:
                    raise ValueError(
                        f"policy violation: excluded originator {name!r} in idea: "
                        f"{idea['text'][:40]}…")
        for book in manifest.get("books", []):
            for author_field in ("author",):
                for name in banned:
                    if name in book.get(author_field, ""):
                        raise ValueError(
                            f"policy violation: excluded originator {name!r} in book "
                            f"{book.get('title')!r}")
    return manifest


def seed_ideas(store: KnowledgeStore, manifest: dict, replace: bool = False) -> dict:
    """Insert manifest ideas as claims (idempotent) + link available authors.

    With ``replace=True`` every claim sourced from an earlier manifest is
    deleted first — the v1 → v2 تنقيح switch (old ideas belong to schools
    outside the منقحة scope).
    """
    seeded = linked_authors = removed = 0
    if replace:
        cur = store.conn.execute(
            "DELETE FROM claims WHERE json_extract(meta, '$.source') = 'fikr_manifest'")
        removed = cur.rowcount
        store.conn.commit()
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
    return {"claims_seeded": seeded, "authors_linked": linked_authors,
            "claims_removed": removed}


# v1 school ids that were renamed by the v2 تنقيح
_SCHOOL_RENAME = {
    "siyasa-shar'iyya": "siyasa-shariyya",
    "maqasid-wasatiyya": "maqasid-imran",
    "imran": "maqasid-imran",
}


def prune_fikr(store: KnowledgeStore, manifest: dict) -> dict:
    """Remove fikr-layer books that fell out of the منقحة scope.

    Deletes (source_id='fikr') books listed in ``policy.excluded_books`` with
    their passages, then any authors left with no books, then renames v1
    school categories to their v2 ids.  Idempotent.
    """
    titles = set(manifest.get("policy", {}).get("excluded_books", []))
    books = [dict(r) for r in store.conn.execute(
        "SELECT book_id FROM books WHERE source_id='fikr' AND title_ar IN (%s)"
        % ",".join("?" * len(titles)), tuple(titles))] if titles else []
    removed_books = removed_passages = 0
    for b in books:
        bid = b["book_id"]
        cur = store.conn.execute("DELETE FROM passages WHERE book_id=?", (bid,))
        removed_passages += cur.rowcount
        cur = store.conn.execute("DELETE FROM books WHERE book_id=?", (bid,))
        removed_books += cur.rowcount
    # orphaned authors (no remaining books)
    store.conn.execute(
        "DELETE FROM authors WHERE author_id NOT IN "
        "(SELECT DISTINCT author_id FROM books WHERE author_id IS NOT NULL)")
    # normalize v1 category/school ids on the kept books
    for old, new in _SCHOOL_RENAME.items():
        store.conn.execute(
            "UPDATE books SET category=REPLACE(category, ?, ?) "
            "WHERE category LIKE '%' || ? || '%'", (old, new, old))
    store.conn.commit()
    return {"books_removed": removed_books, "passages_removed": removed_passages}


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def search_ideas(store: KnowledgeStore, query: str, k: int = 8,
                 schools: set[str] | None = None) -> list[dict]:
    """Deterministic hybrid ranking over the claims table.

    ``schools`` optionally restricts the result to a subset of school ids
    (e.g. ``{"naqd-hanbali", "kalam-sunni"}``).
    """
    rows = [dict(r) for r in store.conn.execute(
        "SELECT claim_id, claim_text, claim_type, confidence, meta FROM claims")]
    if schools:
        rows = [r for r in rows
                if (json.loads(r["meta"]) if r["meta"] else {}).get("school") in schools]
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
