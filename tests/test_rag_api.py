"""RAG serving tests — /knowledge/* endpoints against committed artifacts.

The API is pointed at a *temporary copy* of the committed knowledge DB
(``RYKO_KNOWLEDGE_DB``): search warms the index and the cite endpoint books
verifications, and both must never dirty the committed repo-copy DB.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from conftest import ROOT, requires_final_model

KNOWLEDGE_DB = ROOT / "data" / "knowledge" / "ryko.sqlite"

# Isolation before the app module is imported (it reads the env at import).
_TMP_DB = None
if KNOWLEDGE_DB.exists():
    _TMP_DB = Path(tempfile.mkdtemp(prefix="ryko-rag-test-")) / "ryko.sqlite"
    shutil.copyfile(KNOWLEDGE_DB, _TMP_DB)
    os.environ["RYKO_KNOWLEDGE_DB"] = str(_TMP_DB)

from fastapi.testclient import TestClient  # noqa: E402

from ryko_lm.serving.api import app  # noqa: E402

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
    # v1.15.0 cite-audit loop: audited answer, deliver-or-abstain decision
    audit = body["citation_audit"]
    assert audit["protocol"] == "ryko-cite/1"
    assert audit["sentences_audited"] >= 0
    assert audit["status"] in {"attributed", "partial", "unattributed"}
    assert 0.0 <= audit["attributed_ratio"] <= 1.0
    # the decision follows the documented rule, independent of draft quality
    expected = "deliver" if audit["status"] in {"attributed", "partial"} else "abstain"
    assert body["decision"] == expected
    if body["decision"] == "deliver":
        assert body["delivered_answer"] == body["answer"]
        assert body["abstain_reason"] is None
    else:
        assert "أمتنع" in body["delivered_answer"]
        assert body["abstain_reason"]
    a = client.post("/knowledge/ask", json={
        "question": "ما شروط الإسناد المتصل في الحديث؟",
        "k": 2, "max_new_tokens": 40, "seed": 7,
    }).json()
    assert a["answer"] == body["answer"]  # seeded determinism end-to-end
    assert a["citation_audit"]["status"] == audit["status"]  # audit deterministic


@requires_final_model
@requires_knowledge
def test_knowledge_ask_strict_contract(client):
    body = client.post("/knowledge/ask", json={
        "question": "ما شروط الإسناد المتصل في الحديث؟",
        "k": 2, "max_new_tokens": 40, "seed": 7, "strict": True,
    }).json()
    status = body["citation_audit"]["status"]
    assert status in {"attributed", "partial", "unattributed"}
    expected = "deliver" if status == "attributed" else "abstain"
    assert body["decision"] == expected
    if body["decision"] == "abstain":
        assert "أمتنع" in body["delivered_answer"]
    else:
        assert body["delivered_answer"] == body["answer"]
