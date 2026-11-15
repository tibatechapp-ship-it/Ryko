"""منقحة Islamic-Philosophy layer tests: policy, seeding, pruning, search, API.

Manifest v2 scope (الفلسفة الإسلامية المنقحة):
  - 4 curated schools only (kalam-sunni, naqd-hanbali, maqasid-imran,
    siyasa-shariyya) — no disputed authors, no Greek-derived schools.
  - policy.excluded_originators / excluded_books enforce the تنقيح.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ryko_lm.knowledge import KnowledgeStore
from ryko_lm.knowledge.ideas import (
    load_manifest,
    prune_fikr,
    search_ideas,
    seed_ideas,
)

MANIFEST = Path("data/knowledge/fikr_manifest.json")


@pytest.fixture()
def store():
    s = KnowledgeStore(":memory:")
    s.add_source("fikr", "الفكر الإسلامي المنقح", "MIT (declared)", "declared-open")
    yield s
    s.close()


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST)


# ------------------------------------------------------------- manifest
def test_manifest_v2_curated_scope(manifest):
    assert manifest["kind"] == "ryko-fikr-manifest"
    assert manifest["version"] == 2
    school_ids = {s["id"] for s in manifest["schools"]}
    assert school_ids == {"kalam-sunni", "naqd-hanbali",
                          "maqasid-imran", "siyasa-shariyya"}
    assert len(manifest["ideas"]) >= 30
    for idea in manifest["ideas"]:
        assert idea["text"] and idea["originator"] and idea["school"]


def test_manifest_policy_blocks_excluded_originators(manifest):
    banned = manifest["policy"]["excluded_originators"]
    assert "المودودي" in banned and "سيد قطب" in banned
    for idea in manifest["ideas"]:
        for name in banned:
            assert name not in idea["originator"], idea["text"]
    for book in manifest["books"]:
        for name in banned:
            assert name not in book["author"], book["title"]


def test_manifest_rejects_excluded_originator_injection(tmp_path, manifest):
    """Policy enforcement is a load-time guarantee, not a convention."""
    bad = json.loads(json.dumps(manifest, ensure_ascii=False))
    bad["ideas"].append({"text": "فكرة خارج المنقحة", "originator": "سيد قطب",
                         "school": "naqd-hanbali"})
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="policy violation"):
        load_manifest(p)


def test_manifest_rejects_wrong_kind(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "something-else", "schools": []}), encoding="utf-8")
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


def test_seed_replace_switches_scopes(store, manifest):
    seed_ideas(store, manifest)
    # simulate a v1-era claim from a school outside the منقحة scope
    store.add_claim("الحاكمية: الحكم والتشريع لله وحده",
                    claim_type="فكرة محورية", confidence=1.0,
                    meta={"school": "harakat", "originator": "المودودي",
                          "source": "fikr_manifest"})
    assert store.stats()["claims"] == len(manifest["ideas"]) + 1
    r = seed_ideas(store, manifest, replace=True)
    assert r["claims_removed"] == len(manifest["ideas"]) + 1
    assert store.stats()["claims"] == len(manifest["ideas"])
    rows = store.conn.execute(
        "SELECT meta FROM claims WHERE meta LIKE '%harakat%'").fetchall()
    assert rows == []


def test_seed_ideas_attributions(store, manifest):
    seed_ideas(store, manifest)
    row = store.conn.execute(
        "SELECT claim_text, claim_type, confidence, meta FROM claims "
        "WHERE claim_text LIKE 'العقل الصحيح لا يعارض%'").fetchone()
    assert row is not None
    meta = json.loads(row["meta"])
    assert meta["school"] == "naqd-hanbali"
    assert "ابن تيمية" in meta["originator"]
    assert row["confidence"] == 1.0


def test_seed_links_authors_to_fikr_books(store, manifest):
    store.add_book("shamela-4281", "fikr", "الموافقات في أصول الشريعة",
                   category="فكر/maqasid-imran")
    seed_ideas(store, manifest)
    row = store.conn.execute(
        "SELECT b.author_id, a.name_ar FROM books b JOIN authors a "
        "ON a.author_id = b.author_id WHERE b.book_id='shamela-4281'").fetchone()
    assert row is not None and "الشاطبي" in row["name_ar"]


# --------------------------------------------------------------- pruning
def test_prune_fikr_removes_excluded_books(store, manifest):
    # v1-era books outside the منقحة scope + one kept book
    store.add_book("shamela-7217", "fikr", "طبائع الاستبداد ومصارع الاستعباد",
                   category="فكر/islah-nahda")
    store.add_book("shamela-4566", "fikr", "تفسير المنار",
                   category="فكر/salafiyya-islah")
    store.add_book("shamela-5994", "fikr", "الأحكام السلطانية",
                   category="فكر/siyasa-shar'iyya")
    for i in range(3):
        store.add_passage("shamela-7217", f"مقطع {i}", page_num=i)
        store.add_passage("shamela-5994", f"مقطع سياسي {i}", page_num=i)
    rep = prune_fikr(store, manifest)
    assert rep["books_removed"] == 2
    assert rep["passages_removed"] == 3  # only the excluded book's passages
    titles = [r[0] for r in store.conn.execute("SELECT title_ar FROM books")]
    assert titles == ["الأحكام السلطانية"]
    # v1 school id normalized to v2
    cat = store.conn.execute(
        "SELECT category FROM books WHERE book_id='shamela-5994'").fetchone()[0]
    assert cat == "فكر/siyasa-shariyya"
    # idempotent
    rep2 = prune_fikr(store, manifest)
    assert rep2["books_removed"] == 0 and rep2["passages_removed"] == 0


# --------------------------------------------------------------- search
def test_search_ideas_finds_aql_wa_nuql(store, manifest):
    seed_ideas(store, manifest)
    hits = search_ideas(store, "تعارض العقل مع النقل وأيهم مقدم", k=5)
    assert hits and "العقل الصحيح لا يعارض النقل الصحيح" in hits[0]["claim_text"]
    assert hits[0]["school"] == "naqd-hanbali"
    again = search_ideas(store, "تعارض العقل مع النقل وأيهم مقدم", k=5)
    assert [h["claim_id"] for h in hits] == [h["claim_id"] for h in again]


def test_search_ideas_school_filter(store, manifest):
    seed_ideas(store, manifest)
    hits = search_ideas(store, "العدل أساس الحكم والسياسة الشرعية", k=10,
                        schools={"siyasa-shariyya"})
    assert hits and all(h["school"] == "siyasa-shariyya" for h in hits)


def test_search_ideas_maqasid(store, manifest):
    seed_ideas(store, manifest)
    top = search_ideas(store, "مقاصد الشريعة حفظ النسل والمال", k=3)[0]
    assert top["school"] == "maqasid-imran"


# ------------------------------------------------------------------- API
def test_api_knowledge_ideas_endpoint(monkeypatch, manifest):
    import ryko_lm.serving.api as api_mod

    store = KnowledgeStore(":memory:")
    store.add_source("fikr", "الفكر الإسلامي المنقح", "MIT (declared)", "declared-open")
    seed_ideas(store, manifest)
    monkeypatch.setattr(api_mod, "state", {"store": store, "model": None})
    client = TestClient(api_mod.app)

    resp = client.get("/knowledge/ideas", params={"q": "العدل أساس الحكم", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["protocol"] == "ryko-ideas/1"
    assert len(body["ideas"]) == 3
    assert body["ideas"][0]["originator"] is not None

    # deterministic between calls
    resp2 = client.get("/knowledge/ideas", params={"q": "العدل أساس الحكم", "k": 3})
    assert resp2.json()["ideas"] == body["ideas"]

    # school filter
    resp3 = client.get("/knowledge/ideas",
                       params={"q": "العقل والنقل", "k": 5, "school": "naqd-hanbali"})
    assert resp3.status_code == 200
    assert resp3.json()["school"] == "naqd-hanbali"
    assert all(i["school"] == "naqd-hanbali" for i in resp3.json()["ideas"])

    # k clamped
    resp4 = client.get("/knowledge/ideas", params={"q": "المقاصد", "k": 999})
    assert resp4.status_code == 200 and len(resp4.json()["ideas"]) <= 20
