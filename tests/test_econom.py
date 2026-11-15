"""الاقتصاد الإسلامي والحقوق layer tests: policy, seeding, isolation, search, API.

Manifest v2 scope (data/knowledge/econom_manifest.json):
  - kind ``ryko-econom-manifest`` → layer ``econom`` (source_id 'econom');
  - 4 curated sections (econom-usul / econom-adawat / econom-huquq /
    econom-siyasa);
  - policy: الرأسمالية والاشتراكية أهداف نقد لا مصادر منهج — ماركس وآدم سميث
    وكينز لا تدخل أقوالهم إلا في موضع نقدها;
  - layer isolation: fikr claims/books are never touched by econom ops.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ryko_lm.knowledge import KnowledgeStore
from ryko_lm.knowledge.ideas import (
    load_manifest,
    manifest_layer,
    prune_fikr,
    search_ideas,
    seed_ideas,
)

MANIFEST = Path("data/knowledge/econom_manifest.json")
FIKR_MANIFEST = Path("data/knowledge/fikr_manifest.json")


@pytest.fixture()
def store():
    s = KnowledgeStore(":memory:")
    s.add_source("econom", "الاقتصاد الإسلامي والحقوق", "MIT (declared)", "declared-open")
    s.add_source("fikr", "الفكر الإسلامي المنقح", "MIT (declared)", "declared-open")
    yield s
    s.close()


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST)


@pytest.fixture(scope="module")
def fikr_manifest():
    return load_manifest(FIKR_MANIFEST)


# ------------------------------------------------------------- manifest
def test_manifest_v2_curated_scope(manifest):
    assert manifest["kind"] == "ryko-econom-manifest"
    assert manifest["version"] == 2
    assert manifest_layer(manifest) == "econom"
    assert manifest["layer_label"] == "اقتصاد-حقوق"
    school_ids = {s["id"] for s in manifest["schools"]}
    assert school_ids == {"econom-usul", "econom-adawat",
                          "econom-huquq", "econom-siyasa"}
    assert len(manifest["ideas"]) >= 50
    for idea in manifest["ideas"]:
        assert idea["text"] and idea["originator"] and idea["school"]


def test_manifest_scale_block(manifest):
    scale = manifest["scale"]
    assert (scale["economics_books_original"] + scale["rights_books_original"]
            == scale["total_original"] == 300)
    assert (scale["economics_books_with_sharh"] + scale["rights_books_with_sharh"]
            == scale["total_with_sharh"] == 750)


def test_manifest_policy_blocks_excluded_originators(manifest):
    banned = manifest["policy"]["excluded_originators"]
    assert "ماركس" in banned and "آدم سميث" in banned
    for idea in manifest["ideas"]:
        for name in banned:
            assert name not in idea["originator"], idea["text"]
    for book in manifest["books"]:
        for name in banned:
            assert name not in book.get("author", ""), book["title"]
    assert "المال مال الله" in manifest["policy"]["principle"]


def test_manifest_rejects_excluded_originator_injection(tmp_path, manifest):
    """Policy enforcement is a load-time guarantee, not a convention."""
    bad = json.loads(json.dumps(manifest, ensure_ascii=False))
    bad["ideas"].append({"text": "أقول ماركس في التوزيع", "originator": "ماركس",
                         "school": "econom-usul"})
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="policy violation"):
        load_manifest(p)


def test_manifest_rejects_wrong_kind(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "ryko-unknown-manifest", "schools": []}),
                   encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(bad)


def test_manifest_rejects_unknown_school(tmp_path, manifest):
    bad = json.loads(json.dumps(manifest, ensure_ascii=False))
    bad["ideas"][0]["school"] = "no-such-school"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(p)


# --------------------------------------------------------------- seeding
def test_seed_ideas_is_idempotent(store, manifest):
    r1 = seed_ideas(store, manifest)
    r2 = seed_ideas(store, manifest)
    assert r1["claims_seeded"] == len(manifest["ideas"])
    assert r2["claims_seeded"] == len(manifest["ideas"])  # INSERT OR REPLACE
    assert store.stats()["claims"] == len(manifest["ideas"])


def test_seed_marks_layer_and_source(store, manifest):
    seed_ideas(store, manifest)
    row = store.conn.execute(
        "SELECT meta FROM claims WHERE claim_text LIKE 'حق الفقير في مال الغني%'"
    ).fetchone()
    assert row is not None
    meta = json.loads(row["meta"])
    assert meta["layer"] == "econom"
    assert meta["source"] == "econom_manifest"
    assert meta["school"] == "econom-usul"
    assert "المعارج 24" in meta["originator"]


def test_layer_isolation_replace_keeps_fikr_claims(store, manifest, fikr_manifest):
    """seed(replace=True) on the econom layer re-seeds only econom claims
    and never drops fikr claims."""
    seed_ideas(store, fikr_manifest)
    seed_ideas(store, manifest)
    assert store.stats()["claims"] == len(fikr_manifest["ideas"]) + len(manifest["ideas"])
    r = seed_ideas(store, manifest, replace=True)
    assert r["claims_removed"] == len(manifest["ideas"])  # dropped first…
    assert store.stats()["claims"] == (len(fikr_manifest["ideas"])
                                       + len(manifest["ideas"]))  # …then re-seeded
    n_fikr = store.conn.execute(
        "SELECT COUNT(*) FROM claims WHERE meta LIKE '%fikr_manifest%'").fetchone()[0]
    n_econom = store.conn.execute(
        "SELECT COUNT(*) FROM claims WHERE meta LIKE '%econom_manifest%'").fetchone()[0]
    assert n_fikr == len(fikr_manifest["ideas"])
    assert n_econom == len(manifest["ideas"])


def test_seed_links_authors_to_econom_books(store, manifest):
    store.add_book("shamela-4758", "econom", "كتاب الأموال",
                   category="اقتصاد-حقوق/econom-usul")
    seed_ideas(store, manifest)
    row = store.conn.execute(
        "SELECT b.author_id, a.name_ar FROM books b JOIN authors a "
        "ON a.author_id = b.author_id WHERE b.book_id='shamela-4758'").fetchone()
    assert row is not None and "أبو عبيد" in row["name_ar"]


# --------------------------------------------------------------- pruning
def test_prune_is_layer_scoped(store, manifest, fikr_manifest):
    """fikr pruning never touches econom books (and vice versa)."""
    store.add_book("shamela-7217", "fikr", "طبائع الاستبداد ومصارع الاستعباد",
                   category="فكر/islah-nahda")
    store.add_book("shamela-9999", "econom", "طبائع الاستبداد ومصارع الاستعباد",
                   category="اقتصاد-حقوق/econom-usul")
    rep = prune_fikr(store, fikr_manifest)
    assert rep["books_removed"] == 1
    titles = [r[0] for r in store.conn.execute(
        "SELECT title_ar FROM books WHERE source_id='econom'")]
    assert titles == ["طبائع الاستبداد ومصارع الاستعباد"]
    # econom manifest has no excluded books — prune is a no-op
    rep2 = prune_fikr(store, manifest)
    assert rep2["books_removed"] == 0 and rep2["passages_removed"] == 0


# --------------------------------------------------------------- search
def test_search_ideas_finds_haqq_malum(store, manifest):
    seed_ideas(store, manifest)
    hits = search_ideas(store, "حق الفقير في مال الغني هل هو صدقة", k=5)
    assert hits and "حق معلوم" in hits[0]["claim_text"]
    assert hits[0]["school"] == "econom-usul"
    assert hits[0]["layer"] == "econom"
    again = search_ideas(store, "حق الفقير في مال الغني هل هو صدقة", k=5)
    assert [h["claim_id"] for h in hits] == [h["claim_id"] for h in again]


def test_search_ideas_layer_filter(store, manifest, fikr_manifest):
    seed_ideas(store, fikr_manifest)
    seed_ideas(store, manifest)
    econom = search_ideas(store, "الزكاة والوقف إعادة توزيع الثروة", k=20,
                          layers={"econom"})
    assert econom and all(h["layer"] == "econom" for h in econom)
    fikr = search_ideas(store, "الزكاة والوقف إعادة توزيع الثروة", k=20,
                        layers={"fikr"})
    assert fikr and all(h["layer"] == "fikr" for h in fikr)
    assert not ({"claim_text"} & {h["claim_text"] for h in fikr}
                & {h["claim_text"] for h in econom})


def test_search_ideas_section_filter(store, manifest):
    seed_ideas(store, manifest)
    hits = search_ideas(store, "المضاربة والمشاركة والمرابحة والتمويل", k=10,
                        schools={"econom-adawat"})
    assert hits and all(h["school"] == "econom-adawat" for h in hits)


# --------------------------------------------------------------- ingest
def test_ingest_econom_registers_dedicated_source(store, tmp_path):
    from ryko_lm.knowledge.ingest import ingest_econom_jsonl

    rec = {"id": "shamela-TEST1", "book": "كتاب الأموال",
           "cat": "اقتصاد-حقوق/econom-usul", "pages": 1,
           "author": "أبو عبيد القاسم بن سلام", "school": "econom-usul",
           "page_records": [{"page_num": 7, "body": "باب الزكاة في أموال الغني. "
                                                   "والذين في أموالهم حق معلوم للسائل والمحروم."}]}
    p = tmp_path / "econom.jsonl"
    p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    rep = ingest_econom_jsonl(store, p)
    assert rep["format"] == "econom" and rep["books"] == 1 and rep["passages"] >= 1
    row = store.conn.execute(
        "SELECT b.source_id, b.category FROM books b WHERE b.book_id='shamela-TEST1'"
    ).fetchone()
    assert row["source_id"] == "econom"
    assert row["category"].startswith("اقتصاد-حقوق/")
    # idempotent re-ingest
    rep2 = ingest_econom_jsonl(store, p)
    assert rep2["passages"] == 0 and rep2["duplicates_skipped"] >= 1


# ------------------------------------------------------------------- API
def test_api_knowledge_ideas_endpoint_layer(monkeypatch, manifest):
    import ryko_lm.serving.api as api_mod

    store = KnowledgeStore(":memory:")
    seed_ideas(store, manifest)
    monkeypatch.setattr(api_mod, "state", {"store": store, "model": None})
    client = TestClient(api_mod.app)

    resp = client.get("/knowledge/ideas",
                      params={"q": "حق الفقير في مال الغني", "k": 3,
                              "layer": "econom"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["protocol"] == "ryko-ideas/1"
    assert body["layer"] == "econom"
    assert body["ideas"] and all(i["layer"] == "econom" for i in body["ideas"])

    # wrong layer → no results (isolation visible through the API)
    resp = client.get("/knowledge/ideas",
                      params={"q": "حق الفقير في مال الغني", "k": 3,
                              "layer": "fikr"})
    assert resp.status_code == 200
    assert resp.json()["ideas"] == []
