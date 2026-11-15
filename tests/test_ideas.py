"""Islamic-Thought idea layer tests: manifest validation, seeding, search, API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ryko_lm.knowledge import KnowledgeStore
from ryko_lm.knowledge.ideas import load_manifest, search_ideas, seed_ideas

MANIFEST = Path("data/knowledge/fikr_manifest.json")


@pytest.fixture()
def store():
    s = KnowledgeStore(":memory:")
    s.add_source("fikr", "الفكر الإسلامي", "MIT (declared)", "declared-open")
    yield s
    s.close()


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST)


# ------------------------------------------------------------- manifest
def test_manifest_loads_and_validates(manifest):
    assert manifest["kind"] == "ryko-fikr-manifest"
    assert len(manifest["schools"]) == 12
    assert len(manifest["ideas"]) >= 30
    for idea in manifest["ideas"]:
        assert idea["text"] and idea["originator"] and idea["school"]


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
    st = store.stats()
    assert st["claims"] == len(manifest["ideas"])


def test_seed_ideas_attributions(store, manifest):
    seed_ideas(store, manifest)
    row = store.conn.execute(
        "SELECT claim_text, claim_type, confidence, meta FROM claims "
        "WHERE claim_text LIKE 'الحاكمية%'").fetchone()
    assert row is not None
    meta = json.loads(row["meta"])
    assert meta["school"] == "harakat"
    assert "المودودي" in meta["originator"]
    assert row["confidence"] == 1.0


def test_seed_links_authors_to_fikr_books(store, manifest):
    # a book ingested earlier without an author gets back-linked
    store.add_book("shamela-7217", "fikr", "طبائع الاستبداد ومصارع الاستعباد",
                   category="فكر/islah-nahda")
    seed_ideas(store, manifest)
    row = store.conn.execute(
        "SELECT b.author_id, a.name_ar FROM books b JOIN authors a "
        "ON a.author_id = b.author_id WHERE b.book_id='shamela-7217'").fetchone()
    assert row is not None and "الكواكبي" in row["name_ar"]


# --------------------------------------------------------------- search
def test_search_ideas_finds_hakamiyya(store, manifest):
    seed_ideas(store, manifest)
    hits = search_ideas(store, "الحاكمية وتطبيق الشريعة", k=5)
    assert hits and hits[0]["claim_text"].startswith("الحاكمية")
    assert hits[0]["school"] == "harakat"
    # deterministic ordering: same query -> same ids
    again = search_ideas(store, "الحاكمية وتطبيق الشريعة", k=5)
    assert [h["claim_id"] for h in hits] == [h["claim_id"] for h in again]


def test_search_ideas_maqasid_vs_liberal(store, manifest):
    seed_ideas(store, manifest)
    top = search_ideas(store, "مقاصد الشريعة حفظ النسل والمال", k=3)[0]
    assert top["school"] == "maqasid-wasatiyya"
    top2 = search_ideas(store, "فصل الدين عن الدولة والخلافة ليست من الدين", k=3)[0]
    assert top2["school"] == "liberal"


# ------------------------------------------------------------------- API
def test_api_knowledge_ideas_endpoint(monkeypatch, manifest):
    import ryko_lm.serving.api as api_mod

    store = KnowledgeStore(":memory:")
    store.add_source("fikr", "الفكر الإسلامي", "MIT (declared)", "declared-open")
    seed_ideas(store, manifest)
    monkeypatch.setattr(api_mod, "state", {"store": store, "model": None})
    client = TestClient(api_mod.app)

    resp = client.get("/knowledge/ideas", params={"q": "الاستبداد أصل كل فساد", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["protocol"] == "ryko-ideas/1"
    assert len(body["ideas"]) == 3
    assert body["ideas"][0]["originator"] is not None

    # deterministic between calls
    resp2 = client.get("/knowledge/ideas", params={"q": "الاستبداد أصل كل فساد", "k": 3})
    assert resp2.json()["ideas"] == body["ideas"]

    # k clamped
    resp3 = client.get("/knowledge/ideas", params={"q": "النهضة", "k": 999})
    assert resp3.status_code == 200 and len(resp3.json()["ideas"]) <= 20
