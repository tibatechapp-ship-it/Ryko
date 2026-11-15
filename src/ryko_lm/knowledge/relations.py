"""Argument-relations layer (علاقات الحجاج) over ``argument_relations``.

The six-table schema shipped the ``argument_relations`` table since v1.3.0
with a closed relation vocabulary (``schema.RELATIONS``) — but it stayed
empty.  This module seeds it from a dedicated manifest
(``data/knowledge/relations_manifest.json``) so the curated claims stop
being floating nodes and become a *graph*:

    «تهافت الفلاسفة يرد على ما عرضه مقاصد الفلاسفة» ·
    «حق الفقير حق معلوم يدعم فريضة الزكاة» ·
    «القاعدة الذهبية تلخص ثبوت حق الله وسقوط حق العبد بالعفو»

Design rules (same discipline as the idea layers):

* **Referential integrity at seed time** — every relation cites its two
  claims *verbatim by text*; ``seed_relations`` resolves them against the
  ``claims`` table and refuses (``ValueError``) any unknown text.  Nothing
  is guessed.
* **Deterministic ids** — ``relation_id = stable_id('r', claim_id,
  related_claim_id, relation)``; re-seeding reproduces ids byte for byte.
* **Closed vocabulary** — only ``schema.RELATIONS`` verbs are accepted;
  direction is always ``src --relation--> dst`` (src is the actor).
* **Documented evidence** — every row carries a human-written ``evidence``
  line pointing at the research document / book section it rests on; the
  sha256 of that line goes into ``meta`` for offline audit.
* **Layer ownership** — each relation declares the layer of its source
  claim (``fikr`` | ``econom``); cross-layer targets are allowed (المقاصد
  تحفظ المال — fikr→econom) and recorded in ``meta``.
* **Idempotent** — seeding twice inserts nothing new; ``replace=True``
  drops every relations-manifest row first (claims of any layer are never
  touched by relations operations — the graph layer is strictly
  read-only over ``claims``).

CLI:
    python -m ryko_lm.knowledge relations-seed --db data/knowledge/ryko.sqlite \
        --manifest data/knowledge/relations_manifest.json [--replace]
    python -m ryko_lm.knowledge relations --db data/knowledge/ryko.sqlite \
        [--layer econom] [--relation supports] [--claim-id c-...] [-k 50]
"""

from __future__ import annotations

import json
from pathlib import Path

from ryko_lm.knowledge import schema as sch
from ryko_lm.knowledge.ideas import _claim_layer
from ryko_lm.knowledge.store import KnowledgeStore

SUPPORTED_VERSIONS = (1,)

RELATIONS_KIND = "ryko-relations-manifest"

SOURCE_MARKER = "relations_manifest"

LAYERS = {"fikr", "econom"}

# human-readable verbs for the closed schema.RELATIONS vocabulary
RELATION_AR = {
    "supports": "يدعم",
    "objects": "يعترض",
    "responds": "يرد على",
    "cites": "ينقل عن",
    "prefers": "يرجّح",
    "summarizes": "يلخّص",
    "paraphrases": "يعيد الصياغة",
    "contradicts": "يناقض",
}


def load_relations_manifest(path: str | Path) -> dict:
    """Parse + validate the relations manifest (strict, no silent skips)."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("kind") != RELATIONS_KIND:
        raise ValueError(f"not a relations manifest: {manifest.get('kind')!r}")
    if manifest.get("version") not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported relations-manifest version: {manifest.get('version')!r}")
    seen: set[tuple[str, str, str, str]] = set()
    for rel in manifest.get("relations", []):
        for field in ("layer", "src", "dst", "relation", "evidence"):
            if not str(rel.get(field, "")).strip():
                raise ValueError(f"relation missing {field}: {rel}")
        if rel["relation"] not in sch.RELATIONS:
            raise ValueError(
                f"relation verb outside the closed vocabulary: "
                f"{rel['relation']!r} (allowed: {sch.RELATIONS})")
        if rel["layer"] not in LAYERS:
            raise ValueError(
                f"relation layer outside the curated layers: {rel['layer']!r}")
        if rel["src"] == rel["dst"]:
            raise ValueError(f"self-referencing relation: {rel['src'][:40]}…")
        key = (rel["layer"], rel["src"], rel["dst"], rel["relation"])
        if key in seen:
            raise ValueError(f"duplicate relation: {rel['src'][:40]}…")
        seen.add(key)
    return manifest


def _claim_index(store: KnowledgeStore) -> dict[str, list[tuple[str, str | None]]]:
    """claim_text -> [(claim_id, layer), …] (exact text keys)."""
    index: dict[str, list[tuple[str, str | None]]] = {}
    for row in store.conn.execute("SELECT claim_id, claim_text, meta FROM claims"):
        meta = json.loads(row["meta"]) if row["meta"] else {}
        index.setdefault(row["claim_text"], []).append(
            (row["claim_id"], _claim_layer(meta)))
    return index


def _resolve(index: dict, text: str, want_layer: str | None) -> str:
    """Resolve a claim text to one claim_id (layer-preferred, deterministic)."""
    hits = index.get(text, [])
    if not hits:
        raise ValueError(f"unknown claim text (not seeded?): {text[:60]}…")
    if want_layer:
        for cid, layer in hits:
            if layer == want_layer:
                return cid
    return sorted(cid for cid, _ in hits)[0]


def seed_relations(store: KnowledgeStore, manifest: dict,
                   replace: bool = False) -> dict:
    """Insert manifest relations into ``argument_relations`` (idempotent).

    With ``replace=True`` every relations-manifest-sourced row is deleted
    first — claims/passages/books of any layer are never touched.
    """
    removed = 0
    if replace:
        cur = store.conn.execute(
            "DELETE FROM argument_relations "
            "WHERE json_extract(meta, '$.source') = ?", (SOURCE_MARKER,))
        removed = cur.rowcount
        store.conn.commit()
    index = _claim_index(store)
    seeded = duplicates = 0
    for rel in manifest["relations"]:
        src_id = _resolve(index, rel["src"], rel["layer"])
        dst_id = _resolve(index, rel["dst"], None)
        relation_id = sch.stable_id("r", src_id, dst_id, rel["relation"])
        exists = store.conn.execute(
            "SELECT 1 FROM argument_relations WHERE relation_id=?",
            (relation_id,)).fetchone()
        if exists:
            duplicates += 1
            continue
        src_meta = json.loads(store.conn.execute(
            "SELECT meta FROM claims WHERE claim_id=?", (src_id,)
        ).fetchone()["meta"] or "{}")
        dst_meta = json.loads(store.conn.execute(
            "SELECT meta FROM claims WHERE claim_id=?", (dst_id,)
        ).fetchone()["meta"] or "{}")
        store.conn.execute(
            "INSERT INTO argument_relations "
            "(relation_id, claim_id, related_claim_id, relation, evidence, meta) "
            "VALUES (?,?,?,?,?,?)",
            (relation_id, src_id, dst_id, rel["relation"], rel["evidence"],
             json.dumps({
                 "layer": rel["layer"],
                 "src_layer": _claim_layer(src_meta),
                 "dst_layer": _claim_layer(dst_meta),
                 "src_originator": src_meta.get("originator"),
                 "dst_originator": dst_meta.get("originator"),
                 "source": SOURCE_MARKER,
                 "manifest_version": manifest["version"],
                 "evidence_sha256": sch.sha256_text(rel["evidence"])[:16],
             }, ensure_ascii=False)))
        seeded += 1
    store.conn.commit()
    return {"relations_seeded": seeded, "duplicates_skipped": duplicates,
            "relations_removed": removed}


def list_relations(store: KnowledgeStore, layer: str | None = None,
                   relation: str | None = None, claim_id: str | None = None,
                   k: int = 100) -> list[dict]:
    """Deterministic relation listing with both endpoints fully resolved.

    ``claim_id`` matches either endpoint (the graph is traversable from any
    node); ordering is fixed by ``relation_id`` — no RNG anywhere.
    """
    sql = (
        "SELECT r.relation_id, r.relation, r.evidence, r.meta, "
        "  s.claim_id AS src_id, s.claim_text AS src_text, s.claim_type AS src_type, "
        "  d.claim_id AS dst_id, d.claim_text AS dst_text, d.claim_type AS dst_type "
        "FROM argument_relations r "
        "JOIN claims s ON s.claim_id = r.claim_id "
        "JOIN claims d ON d.claim_id = r.related_claim_id "
        "WHERE 1=1"
    )
    params: list[object] = []
    if layer:
        sql += " AND json_extract(r.meta, '$.layer') = ?"
        params.append(layer)
    if relation:
        sql += " AND r.relation = ?"
        params.append(relation)
    if claim_id:
        sql += " AND (r.claim_id = ? OR r.related_claim_id = ?)"
        params.extend([claim_id, claim_id])
    sql += " ORDER BY r.relation_id"
    if k > 0:
        sql += f" LIMIT {int(k)}"
    out: list[dict] = []
    for row in store.conn.execute(sql, params):
        meta = json.loads(row["meta"]) if row["meta"] else {}
        out.append({
            "relation_id": row["relation_id"],
            "relation": row["relation"],
            "relation_ar": RELATION_AR.get(row["relation"], row["relation"]),
            "layer": meta.get("layer"),
            "src": {"claim_id": row["src_id"], "text": row["src_text"],
                    "claim_type": row["src_type"],
                    "originator": meta.get("src_originator"),
                    "school": None},
            "dst": {"claim_id": row["dst_id"], "text": row["dst_text"],
                    "claim_type": row["dst_type"],
                    "originator": meta.get("dst_originator")},
            "evidence": row["evidence"],
        })
    return out


def relations_stats(store: KnowledgeStore) -> dict:
    """Small deterministic profile of the relation graph."""
    by_layer = {row[0]: row[1] for row in store.conn.execute(
        "SELECT json_extract(meta, '$.layer'), COUNT(*) FROM argument_relations "
        "GROUP BY 1 ORDER BY 1")}
    by_relation = {row[0]: row[1] for row in store.conn.execute(
        "SELECT relation, COUNT(*) FROM argument_relations "
        "GROUP BY 1 ORDER BY 1")}
    cross = store.conn.execute(
        "SELECT COUNT(*) FROM argument_relations "
        "WHERE json_extract(meta, '$.src_layer') != "
        "json_extract(meta, '$.dst_layer')").fetchone()[0]
    return {"total": sum(by_layer.values()), "by_layer": by_layer,
            "by_relation": by_relation, "cross_layer": cross}
