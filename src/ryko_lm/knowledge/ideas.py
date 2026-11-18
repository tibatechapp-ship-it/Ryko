"""Curated idea layers (العقل الفكري والاقتصاد والحقوق) over the claims table.

Each curated domain is encoded in its own manifest:

* ``data/knowledge/fikr_manifest.json`` — kind ``ryko-fikr-manifest`` →
  the منقحة Islamic-Philosophy layer (4 schools, no disputed authors);
* ``data/knowledge/econom_manifest.json`` — kind ``ryko-econom-manifest`` →
  the Islamic-Economics-and-Rights layer (المال مال الله والحقوق من إرادة الشرع).

Manifest **v2** carries an explicit ``policy`` block: excluded
originators/books and the layer's positioning principle.  This module:

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

# manifest kind -> knowledge layer (source_id prefix + claims meta.layer)
MANIFEST_KINDS = {
    "ryko-fikr-manifest": "fikr",
    "ryko-econom-manifest": "econom",
    "ryko-lugha-manifest": "lugha",
}


def manifest_layer(manifest: dict) -> str:
    """Knowledge layer for a manifest (``fikr`` | ``econom`` | ``lugha``)."""
    layer = MANIFEST_KINDS.get(manifest.get("kind"))
    if layer is None:
        raise ValueError(f"unknown manifest kind: {manifest.get('kind')!r}")
    return layer


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("kind") not in MANIFEST_KINDS:
        raise ValueError(f"not a curated manifest: {manifest.get('kind')!r}")
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
        # (policies are layer-scoped — the fikr ban list does not apply to
        # the econom layer and vice versa)
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

    With ``replace=True`` every claim sourced from *this layer's* earlier
    manifest is deleted first (e.g. the fikr v1 → v2 تنقيح switch) — claims
    of other layers (econom, …) are never touched.
    """
    layer = manifest_layer(manifest)
    source_marker = f"{layer}_manifest"
    seeded = linked_authors = removed = 0
    if replace:
        cur = store.conn.execute(
            "DELETE FROM claims WHERE json_extract(meta, '$.source') = ?",
            (source_marker,))
        removed = cur.rowcount
        store.conn.commit()
    for idea in manifest["ideas"]:
        claim_id = store.add_claim(
            idea["text"],
            claim_type=idea.get("type"),
            confidence=1.0,  # attribution taken verbatim from the research map
            meta={
                "layer": layer,
                "school": idea["school"],
                "originator": idea["originator"],
                "source": source_marker,
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
    """Remove this layer's books that fell out of the curated scope.

    Deletes books with ``source_id == layer`` listed in
    ``policy.excluded_books`` with their passages, then any authors left
    with no books, then renames v1 school categories to their v2 ids.
    Idempotent, and scoped to the manifest's own layer (fikr pruning never
    touches econom books and vice versa).
    """
    layer = manifest_layer(manifest)
    titles = set(manifest.get("policy", {}).get("excluded_books", []))
    books = [dict(r) for r in store.conn.execute(
        "SELECT book_id FROM books WHERE source_id=? AND title_ar IN (%s)"
        % ",".join("?" * len(titles)), (layer, *titles))] if titles else []
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


def _claim_layer(meta: dict) -> str | None:
    """Resolve the knowledge layer of a claim (meta.layer, with a fallback
    to the legacy fikr v2 source marker for claims seeded before layers)."""
    if meta.get("layer"):
        return meta["layer"]
    src = str(meta.get("source", ""))
    return src[:-9] if src.endswith("_manifest") else None


def search_ideas(store: KnowledgeStore, query: str, k: int = 8,
                 schools: set[str] | None = None,
                 layers: set[str] | None = None) -> list[dict]:
    """Deterministic hybrid ranking over the claims table.

    ``schools`` optionally restricts the result to a subset of school/section
    ids (e.g. ``{"naqd-hanbali", "econom-adawat"}``); ``layers`` restricts to
    a curated layer (``{"fikr"}``, ``{"econom"}``).
    """
    rows = [dict(r) for r in store.conn.execute(
        "SELECT claim_id, claim_text, claim_type, confidence, meta FROM claims")]
    metas = [json.loads(r["meta"]) if r["meta"] else {} for r in rows]
    if layers:
        rows = [r for r, m in zip(rows, metas) if _claim_layer(m) in layers]
        metas = [m for m in metas if _claim_layer(m) in layers]
    if schools:
        keep = [i for i, m in enumerate(metas) if m.get("school") in schools]
        rows = [rows[i] for i in keep]
        metas = [metas[i] for i in keep]
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
            "layer": _claim_layer(meta),
            "score": round(score, 6),
        })
    scored.sort(key=lambda r: (-r["score"], r["claim_id"]))
    return scored[:k]
