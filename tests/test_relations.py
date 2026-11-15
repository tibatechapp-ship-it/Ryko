"""Argument-relations layer tests (علاقات الحجاج) — v1.10.0.

The ``argument_relations`` table shipped empty since v1.3.0; the relations
manifest (``data/knowledge/relations_manifest.json``) seeds it with
documented support/opposition links between the curated claims of the fikr
and econom layers.  These tests enforce:

  - strict manifest validation (closed verbs, known layers, no self-loops,
    no duplicates, evidence mandatory);
  - referential integrity at seed time (unknown claim text → ValueError);
  - deterministic ids + idempotence + layer-scoped ``--replace``;
  - cross-layer linking (المقاصد fikr → اقتصاد econom);
  - CLI (``relations-seed`` / ``relations``) and the
    ``GET /knowledge/relations`` endpoint (protocol ``ryko-relations/1``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ryko_lm.knowledge import KnowledgeStore
from ryko_lm.knowledge.ideas import load_manifest, seed_ideas
from ryko_lm.knowledge.relations import (
    RELATIONS_KIND,
    list_relations,
    load_relations_manifest,
    relations_stats,
    seed_relations,
)

REL_MANIFEST = Path("data/knowledge/relations_manifest.json")
FIKR_MANIFEST = Path("data/knowledge/fikr_manifest.json")
ECONOM_MANIFEST = Path("data/knowledge/econom_manifest.json")


@pytest.fixture(scope="module")
def rel_manifest():
    return load_relations_manifest(REL_MANIFEST)


@pytest.fixture()
def store():
    s = KnowledgeStore(":memory:")
    s.add_source("fikr", "الفكر الإسلامي المنقح", "MIT (declared)", "declared-open")
    s.add_source("econom", "الاقتصاد الإسلامي والحقوق", "MIT (declared)",
                 "declared-open")
    seed_ideas(s, load_manifest(FIKR_MANIFEST))
    seed_ideas(s, load_manifest(ECONOM_MANIFEST))
    yield s
    s.close()


# ------------------------------------------------------------- manifest
def test_relations_manifest_scope(rel_manifest):
    assert rel_manifest["kind"] == RELATIONS_KIND
    assert rel_manifest["version"] == 1
    rels = rel_manifest["relations"]
    assert len(rels) >= 30
    verbs = {"supports", "objects", "responds", "cites", "prefers",
             "summarizes", "paraphrases", "contradicts"}
    for rel in rels:
        assert rel["relation"] in verbs, rel["src"][:40]
        assert rel["layer"] in {"fikr", "econom"}
        assert rel["evidence"].strip()
        assert rel["src"] != rel["dst"]


def test_relations_manifest_rejects_bad_verb(tmp_path):
    bad = {"kind": RELATIONS_KIND, "version": 1, "relations": [
        {"layer": "fikr", "src": "أ", "dst": "ب", "relation": "يؤمن بـ",
         "evidence": "دليل"}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="closed vocabulary"):
        load_relations_manifest(p)


def test_relations_manifest_rejects_self_loop(tmp_path):
    bad = {"kind": RELATIONS_KIND, "version": 1, "relations": [
        {"layer": "fikr", "src": "أ", "dst": "أ", "relation": "supports",
         "evidence": "دليل"}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="self-referencing"):
        load_relations_manifest(p)


def test_relations_manifest_rejects_missing_evidence(tmp_path):
    bad = {"kind": RELATIONS_KIND, "version": 1, "relations": [
        {"layer": "fikr", "src": "أ", "dst": "ب", "relation": "supports",
         "evidence": " "}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing evidence"):
        load_relations_manifest(p)


def test_relations_manifest_rejects_duplicates(tmp_path):
    rel = {"layer": "fikr", "src": "أ", "dst": "ب", "relation": "supports",
           "evidence": "دليل"}
    bad = {"kind": RELATIONS_KIND, "version": 1, "relations": [rel, dict(rel)]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_relations_manifest(p)


def test_relations_manifest_rejects_wrong_kind(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"kind": "ryko-fikr-manifest", "version": 1,
                             "relations": []}, ensure_ascii=False),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="not a relations manifest"):
        load_relations_manifest(p)


# --------------------------------------------------------------- seeding
def test_seed_relations_deterministic_and_complete(store, rel_manifest):
    r1 = seed_relations(store, rel_manifest)
    total = len(rel_manifest["relations"])
    assert r1["relations_seeded"] == total
    assert r1["relations_removed"] == 0
    # deterministic ids: re-seeding is a no-op
    r2 = seed_relations(store, rel_manifest)
    assert r2["relations_seeded"] == 0
    assert r2["duplicates_skipped"] == total
    assert store.stats()["argument_relations"] == total


def test_seed_relations_replace_drops_only_relations(store, rel_manifest):
    seed_relations(store, rel_manifest)
    claims_before = store.stats()["claims"]
    r = seed_relations(store, rel_manifest, replace=True)
    assert r["relations_removed"] == len(rel_manifest["relations"])
    assert r["relations_seeded"] == len(rel_manifest["relations"])
    assert store.stats()["claims"] == claims_before  # claims never touched


def test_seed_relations_unknown_text_rejected(store, rel_manifest):
    tampered = json.loads(json.dumps(rel_manifest))
    tampered["relations"][0]["dst"] = "فكرة غير مزروعة إطلاقاً في القاعدة"
    with pytest.raises(ValueError, match="unknown claim text"):
        seed_relations(store, tampered)


def test_seed_relations_layers_and_cross_layer(store, rel_manifest):
    seed_relations(store, rel_manifest)
    graph = relations_stats(store)
    assert graph["total"] == len(rel_manifest["relations"])
    assert graph["by_layer"].get("fikr") and graph["by_layer"].get("econom")
    assert graph["cross_layer"] >= 1  # المقاصد تحفظ المال — fikr→econom
    rows = store.conn.execute(
        "SELECT meta FROM argument_relations "
        "WHERE json_extract(meta,'$.dst_layer') != "
        "json_extract(meta,'$.src_layer')").fetchall()
    assert rows and json.loads(rows[0]["meta"])["layer"] == "fikr"


def test_seed_relations_evidence_recorded(store, rel_manifest):
    seed_relations(store, rel_manifest)
    row = store.conn.execute(
        "SELECT evidence, relation FROM argument_relations "
        "WHERE relation='paraphrases'").fetchone()
    assert row is not None
    assert "الإبانة" in row["evidence"]


# ----------------------------------------------------------------- query
def test_list_relations_filters(store, rel_manifest):
    seed_relations(store, rel_manifest)
    econom = list_relations(store, layer="econom")
    assert econom and all(r["layer"] == "econom" for r in econom)
    summ = list_relations(store, relation="summarizes")
    assert summ and all(r["relation"] == "summarizes" for r in summ)
    # deterministic ordering
    assert [r["relation_id"] for r in econom] == [
        r["relation_id"] for r in list_relations(store, layer="econom")]


def test_list_relations_traverses_from_any_endpoint(store, rel_manifest):
    seed_relations(store, rel_manifest)
    row = store.conn.execute(
        "SELECT claim_id FROM claims WHERE claim_text LIKE 'الزكاة: فريضة%'"
    ).fetchone()
    touching = list_relations(store, claim_id=row["claim_id"])
    assert len(touching) >= 2  # حق الفقير يدعم الزكاة + الموضوعان يلخصانها
    ids = {r["src"]["claim_id"] for r in touching} | {
        r["dst"]["claim_id"] for r in touching}
    assert row["claim_id"] in ids
    for r in touching:
        assert r["src"]["text"] and r["dst"]["text"] and r["evidence"]


# ------------------------------------------------------------------- CLI
def test_cli_relations_seed_and_list(store, rel_manifest, tmp_path, monkeypatch,
                                     capsys):
    from ryko_lm.knowledge import __main__ as cli

    db_path = tmp_path / "rel.sqlite"
    # rebuild the store on disk (CLI opens its own connection)
    disk = KnowledgeStore(str(db_path))
    disk.add_source("fikr", "الفكر", "MIT", "declared-open")
    disk.add_source("econom", "الاقتصاد", "MIT", "declared-open")
    seed_ideas(disk, load_manifest(FIKR_MANIFEST))
    seed_ideas(disk, load_manifest(ECONOM_MANIFEST))
    disk.close()

    rc = cli.main(["relations-seed", "--db", str(db_path),
                   "--manifest", str(REL_MANIFEST), "--replace"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.split("[knowledge]")[0])
    assert out["relations_seeded"] == len(rel_manifest["relations"])

    rc = cli.main(["relations", "--db", str(db_path),
                   "--layer", "econom", "--relation", "supports", "-k", "5"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == "ryko-relations/1"
    assert len(payload["relations"]) <= 5
    assert all(r["relation"] == "supports" and r["layer"] == "econom"
               for r in payload["relations"])


# ------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client():
    from ryko_lm.serving.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_api_knowledge_relations_endpoint(client):
    body = client.get("/knowledge/relations",
                      params={"layer": "econom", "relation": "supports"}).json()
    assert body["protocol"] == "ryko-relations/1"
    assert body["layer"] == "econom"
    assert body["graph"]["total"] >= 30
    assert body["relations"]
    for rel in body["relations"]:
        assert rel["relation"] == "supports"
        assert rel["src"]["text"] and rel["dst"]["text"]
        assert rel["evidence"]


def test_api_knowledge_relations_claim_traversal(client):
    body = client.get("/knowledge/relations",
                      params={"layer": "fikr", "k": 3}).json()
    assert body["protocol"] == "ryko-relations/1"
    first = body["relations"][0]
    by_id = client.get("/knowledge/relations",
                       params={"claim_id": first["src"]["claim_id"]}).json()
    ids = {r["src"]["claim_id"] for r in by_id["relations"]} | {
        r["dst"]["claim_id"] for r in by_id["relations"]}
    assert first["src"]["claim_id"] in ids
