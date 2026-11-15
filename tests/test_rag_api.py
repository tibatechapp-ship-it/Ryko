"""RAG serving tests — /knowledge/* endpoints against committed artifacts."""

import pytest
from fastapi.testclient import TestClient

from ryko_lm.serving.api import app

from conftest import ROOT, requires_final_model


KNOWLEDGE_DB = ROOT / "data" / "knowledge" / "ryko.sqlite"

requires_knowledge = pytest.mark.skipif(
    not KNOWLEDGE_DB.exists(), reason="knowledge store not committed"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@requires_final_model
def test_health_reports_store(client):
    body = client.get("/health").json()
    assert body["knowledge_store_loaded"] is True


@requires_knowledge
def test_knowledge_search_provenance(client):
    body = client.get("/knowledge/search",
                      params={"q": "الإسناد المتصل", "k": 3}).json()
    assert body["mode"] == "hybrid"
    assert 1 <= len(body["hits"]) <= 3
    hit = body["hits"][0]
    for field in ("passage_id", "book_id", "title_ar", "page_num", "text", "score"):
        assert field in hit


@requires_knowledge
def test_knowledge_search_modes_deterministic(client):
    a = client.get("/knowledge/search", params={"q": "الماء الطهور", "k": 3}).json()
    b = client.get("/knowledge/search", params={"q": "الماء الطهور", "k": 3}).json()
    assert [h["passage_id"] for h in a["hits"]] == [h["passage_id"] for h in b["hits"]]


@requires_knowledge
def test_knowledge_cite_verified_and_not_found(client):
    hit = client.get("/knowledge/search",
                     params={"q": "الإسناد المتصل", "k": 1}).json()["hits"][0]
    quote = hit["text"][:100].rsplit(" ", 1)[0]
    verdict = client.post("/knowledge/cite",
                          json={"quote": quote}).json()
    assert verdict["status"] == "verified"
    assert len(verdict["quote_sha256"]) == 64
    bad = client.post("/knowledge/cite",
                      json={"quote": "هذا نص مركّب لا وجود له في المتن إطلاقاً وقطعاً"}).json()
    assert bad["status"] in {"not_found", "partial"}


@requires_final_model
@requires_knowledge
def test_knowledge_ask_grounding(client):
    body = client.post("/knowledge/ask", json={
        "question": "ما شروط الإسناد المتصل في الحديث؟",
        "k": 2, "max_new_tokens": 40, "seed": 7,
    }).json()
    assert body["answer"]
    assert 1 <= len(body["sources"]) <= 2
    assert "page_num" in body["sources"][0]
    assert body["guard"]["decision"] in {"ALLOW", "REVIEW"}
    assert 0.0 <= body["grounding_ratio"] <= 1.0
    assert "ليست فتوى" in body["disclaimer"]
    a = client.post("/knowledge/ask", json={
        "question": "ما شروط الإسناد المتصل في الحديث؟",
        "k": 2, "max_new_tokens": 40, "seed": 7,
    }).json()
    assert a["answer"] == body["answer"]  # seeded determinism end-to-end
