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
    assert info["params"] == 11_536_640
    assert info["vocab_size"] == 32768


@requires_final_model
def test_generate_endpoint(client):
    body = client.post("/generate", json={
        "prompt": "اللغة العربية", "max_new_tokens": 12, "seed": 5,
    }).json()
    assert body["prompt_tokens"] >= 1
    assert body["generated_tokens"] == 12
    assert len(body["completion"]) > 0


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
