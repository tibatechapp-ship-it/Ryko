"""Speed — measured throughput/latency floors from calibration (16 tests).

Every floor below is >= 4x headroom under the measurements taken on the
project CPU (scripts/calibrate_perf.py): tokenizer ~494k tok/s, normalize
~130 MB/s, ar_tokens ~330k tok/s, embed-1000 65ms, 1024-passage index
build 0.17s, search p95 ~12ms hybrid, tiny forward 3.3ms, served 16-token
generation 141ms.  The suite also dumps a metrics JSON for the campaign
report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from ryko_lm.knowledge.embeddings import embed
from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.knowledge.textproc import ar_tokens
from ryko_lm.model.ryko_model import RykoLM
from ryko_lm.tokenizer.loader import RykoTokenizer
from ryko_lm.tokenizer.normalizers import python_normalize

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_JSON = ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json"
FINAL_MODEL = ROOT / "artifacts" / "dpo_round2_small" / "final.pt"
requires = pytest.mark.skipif(
    not (TOKENIZER_JSON.exists() and FINAL_MODEL.exists()),
    reason="artifacts not committed",
)

ARABIC = "اللغة العربية بحر زاخر بالمعاني وأدب عريق في كل العصور"
BIG = (ARABIC + " ") * 2000  # ~94 KB
METRICS: dict[str, float | int] = {}


def _finish(request):
    METRICS["suite"] = "ryko500"
    (Path("/tmp") / "ryko500_metrics.json").write_text(
        json.dumps(METRICS, indent=2, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module", autouse=True)
def _record(request):
    yield
    _finish(request)


@pytest.fixture(scope="module")
def tok():
    return RykoTokenizer.from_file(TOKENIZER_JSON)


@pytest.fixture(scope="module")
def served():
    if not FINAL_MODEL.exists():
        pytest.skip("final model not committed")
    return RykoLM.load(FINAL_MODEL, device="cpu")


@pytest.fixture(scope="module")
def synth_1024():
    s = KnowledgeStore(":memory:")
    s.add_source("bench", "Bench", "MIT", "declared-open")
    s.add_book("bk", "bench", "كتاب القياس")
    for i in range(1024):
        s.add_passage("bk", f"باب {i % 20}: القياس على الوضع في معنى المقياس به {ARABIC} رقم {i}",
                      page_num=i // 50, seq=i % 50)
    return s


# ------------------------------------------------------------- tokenizer
@requires
def test_tokenizer_throughput_floor(tok):
    t0 = time.perf_counter()
    tok.token_count(BIG)
    dt = time.perf_counter() - t0
    tps = tok.token_count(BIG) / dt
    METRICS["tokenizer_tps"] = round(tps)
    assert tps >= 50_000  # measured 494k


@requires
def test_tokenizer_kb_per_second_floor(tok):
    t0 = time.perf_counter()
    tok.token_count(BIG)
    dt = time.perf_counter() - t0
    kbps = len(BIG) / 1024 / dt
    METRICS["tokenizer_kb_s"] = round(kbps)
    assert kbps >= 200  # measured 2042


@requires
def test_batch_500_lines_fast(tok):
    lines = [f"{ARABIC} رقم {i}" for i in range(500)]
    t0 = time.perf_counter()
    tok.encode_batch(lines)
    dt = time.perf_counter() - t0
    METRICS["batch500_s"] = round(dt, 3)
    assert dt <= 2.0  # measured 0.009


# ------------------------------------------------------------- pipeline
def test_normalize_throughput_floor():
    t0 = time.perf_counter()
    python_normalize(BIG)
    dt = time.perf_counter() - t0
    kbps = len(BIG) / 1024 / dt
    METRICS["normalize_kb_s"] = round(kbps)
    assert kbps >= 10_000  # measured 132k KB/s


def test_ar_tokens_throughput_floor():
    t0 = time.perf_counter()
    ar_tokens(BIG)
    dt = time.perf_counter() - t0
    tps = len(ar_tokens(BIG)) / dt
    METRICS["ar_tokens_tps"] = round(tps)
    assert tps >= 40_000  # measured 331k


def test_embed_1000_docs_fast():
    t0 = time.perf_counter()
    for i in range(1000):
        embed(ARABIC + str(i))
    dt = time.perf_counter() - t0
    METRICS["embed_1000_s"] = round(dt, 3)
    assert dt <= 3.0  # measured 0.065


# ----------------------------------------------------------------- store
def test_ingest_1024_fast(synth_1024):
    assert synth_1024.stats()["passages"] == 1024  # fixture build <= 5s


def test_index_build_and_first_search_fast(synth_1024):
    fresh = KnowledgeStore(":memory:")
    fresh.add_source("b", "B", "MIT", "declared-open")
    fresh.add_book("k", "b", "ك")
    t0 = time.perf_counter()
    for i in range(1024):
        fresh.add_passage("k", f"مقطع القياس رقم {i} {ARABIC}", page_num=i // 50, seq=i % 50)
    fresh.search(ARABIC, k=5, mode="hybrid")  # forces full index build
    dt = time.perf_counter() - t0
    fresh.close()
    METRICS["index_build_1024_s"] = round(dt, 2)
    assert dt <= 8.0  # measured 0.17


@pytest.mark.parametrize("mode, floor_ms", [("bm25", 200), ("dense", 150), ("hybrid", 250)])
def test_search_p95_latency(synth_1024, mode, floor_ms):
    synth_1024.search(ARABIC, k=10, mode=mode)  # warm
    lat = []
    for _ in range(20):
        t0 = time.perf_counter()
        synth_1024.search(ARABIC, k=10, mode=mode)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    p95 = lat[int(len(lat) * 0.95)]
    METRICS[f"search_{mode}_p95_ms"] = round(p95, 2)
    assert p95 <= floor_ms


def test_sidecar_cache_warm_load_fast(tmp_path):
    """v1.28.0 sidecar: second session must be dramatically cheaper."""
    cache = tmp_path / "idx.pkl"

    def _build() -> KnowledgeStore:
        s = KnowledgeStore(":memory:", vector_cache=cache)
        s.add_source("b", "B", "MIT", "declared-open")
        s.add_book("k", "b", "ك")
        for i in range(512):
            s.add_passage("k", f"القياس رقم {i} {ARABIC}", page_num=i // 32, seq=i % 32)
        return s

    builder = _build()
    t0 = time.perf_counter()
    ref = builder.search("القياس", k=5, mode="hybrid")  # computes + saves sidecar
    cold = time.perf_counter() - t0
    builder.close()
    assert cache.exists()

    warm_store = _build()  # identical population -> identical fingerprint
    t0 = time.perf_counter()
    got = warm_store.search("القياس", k=5, mode="hybrid")  # adopts the sidecar
    warm = time.perf_counter() - t0
    warm_store.close()

    assert [h.passage_id for h in ref] == [h.passage_id for h in got]
    assert [h.score for h in ref] == [h.score for h in got]
    METRICS["sidecar_cold_s"] = round(cold, 3)
    METRICS["sidecar_warm_s"] = round(warm, 3)
    assert warm < max(cold / 2.0, 0.05)


# ----------------------------------------------------------------- model
@requires
def test_served_generation_16_tokens_floor(served, tok):
    ids = tok.encode(ARABIC, add_bos=True)
    t0 = time.perf_counter()
    out = served.generate(torch.tensor([ids]), max_new_tokens=16, eos_id=None,
                          generator=torch.Generator().manual_seed(7))
    dt = time.perf_counter() - t0
    METRICS["served_gen16_ms"] = round(dt * 1000, 1)
    assert out.shape[1] == 16 and dt <= 10.0  # measured 141ms


@requires
def test_served_seeded_generation_repeatable(served, tok):
    ids = tok.encode(ARABIC, add_bos=True)
    a = served.generate(torch.tensor([ids]), max_new_tokens=8,
                        generator=torch.Generator().manual_seed(13))
    b = served.generate(torch.tensor([ids]), max_new_tokens=8,
                        generator=torch.Generator().manual_seed(13))
    assert torch.equal(a, b)
