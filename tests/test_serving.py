"""Serving tests: FastAPI endpoints against the committed trained artifact."""

import pytest
from fastapi.testclient import TestClient

from ryko_lm.serving.api import app

from conftest import FINAL_MODEL, requires_final_model


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@requires_final_model
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


@requires_final_model
def test_model_info(client):
    info = client.get("/model").json()
    # v1.24.0: the serving default is the promoted round-2 point
    # (artifacts/dpo_round2_small/final.pt) — 52,179,456 params, the same
    # test-pinned ryko-small param contract (test_ryko_small_param_contract).
    assert info["params"] == 52_179_456
    assert info["vocab_size"] == 32768


@requires_final_model
def test_generate_endpoint(client):
    body = client.post("/generate", json={
        "prompt": "اللغة العربية", "max_new_tokens": 12, "seed": 5,
    }).json()
    assert body["prompt_tokens"] >= 1
    assert body["generated_tokens"] == 12
    assert len(body["completion"]) > 0
    assert body["finish_reason"] == "stop" and body["abstained"] is False


@requires_final_model
def test_generate_reports_immediate_eos_abstention(client):
    # QA v1.27.0-prep (FLAW_REPORT F1): with the published serving seed 315,
    # several in-domain prompts emit EOS immediately (empty completion, 1
    # generated token). The response must expose that as an explicit
    # abstention signal instead of a bare empty string.
    body = client.post("/generate", json={
        "prompt": "ما هي أركان الإسلام؟", "max_new_tokens": 12, "seed": 315,
    }).json()
    assert body["completion"] == "" and body["generated_tokens"] == 1
    assert body["abstained"] is True
    assert body["finish_reason"] == "eos_immediate"


@requires_final_model
def test_generate_is_deterministic_with_seed(client):
    payload = {"prompt": "كان يا ما كان", "max_new_tokens": 10, "seed": 9}
    a = client.post("/generate", json=payload).json()["completion"]
    b = client.post("/generate", json=payload).json()["completion"]
    assert a == b


@requires_final_model
def test_stream_endpoint_yields_sse(client):
    import json as _json

    with client.stream("POST", "/generate/stream",
                       json={"prompt": "العلم", "max_new_tokens": 8, "seed": 1}) as response:
        events = [line for line in response.iter_lines() if line.startswith("data:")]
    assert len(events) >= 8
    final = _json.loads(events[-1][5:].strip())  # last SSE payload
    assert final.get("done") is True and final["tokens"] == 8


def test_index_serves_html():
    html = TestClient(app).get("/").text
    assert "Ryko" in html and "generate/stream" in html
