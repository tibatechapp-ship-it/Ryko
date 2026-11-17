"""PalmX-IC converter tests — schema, determinism, provenance receipt.

Offline: builds a synthetic parquet in tmp_path (same schema as PalmX 2025
subtask 2: question / choices / index) and exercises the closed contract of
``scripts/palmx_to_bench.convert``. The committed 981-case JSONL is validated
elsewhere (ryko_bench run on docs/results/palmx_ic_v1.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from palmx_to_bench import convert  # noqa: E402

from ryko_lm.eval.benchmarks import load_benchmark  # noqa: E402


@pytest.fixture()
def tiny_parquet(tmp_path):
    df = pd.DataFrame({
        "prompt": ["p1", "p2", "p3"],
        "question": ["سؤال التورث الأول؟", "سؤال الزكاة الثاني؟", "سؤال الوقف الثالث؟"],
        "choices": [
            ["أ", "ب", "ج", "د"],
            ["الظاهر", "الأرجح", "النادر", "المنهي"],
            ["نوعان", "ثلاثة", "أربعة", "خمسة"],
        ],
        "index": [2, 1, 0],
    })
    path = tmp_path / "palmx_tiny.parquet"
    df.to_parquet(path)
    return path


def test_convert_schema_and_load(tiny_parquet, tmp_path):
    out = tmp_path / "bench.jsonl"
    n, sha = convert(tiny_parquet, out)
    assert n == 3 and len(sha) == 64
    cases = load_benchmark(out)  # closed-schema validation must pass
    assert [c.case_id for c in cases] == ["px-0000", "px-0001", "px-0002"]
    assert cases[0].kind == "mcq"
    assert cases[0].source == "palmx-ic"
    assert cases[0].domain == "islamic-culture"
    assert cases[0].answer == 2
    assert len(cases[0].options) == 4
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[1])
    assert rec["answer"] == 1
    assert rec["meta"]["parquet_sha256"] == sha


def test_convert_double_build_byte_identical(tiny_parquet, tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    convert(tiny_parquet, a)
    convert(tiny_parquet, b)
    assert a.read_bytes() == b.read_bytes()


def test_convert_receipt_provenance(tiny_parquet, tmp_path):
    out = tmp_path / "bench.jsonl"
    receipt = tmp_path / "receipt.json"
    n, sha = convert(tiny_parquet, out, receipt=receipt)
    d = json.loads(receipt.read_text(encoding="utf-8"))
    assert d["rows"] == n and d["parquet_sha256"] == sha
    assert "UBC-NLP/palmx_2025_subtask2_islamic" in d["official_dataset"]
    assert d["bench_source"] == "palmx-ic"


def test_convert_rejects_bad_answer(tmp_path):
    df = pd.DataFrame({
        "prompt": ["p"], "question": ["س؟"],
        "choices": [["أ", "ب", "ج", "د"]],
        "index": [7],
    })
    path = tmp_path / "bad.parquet"
    df.to_parquet(path)
    with pytest.raises(ValueError, match="answer 7 out of range"):
        convert(path, tmp_path / "out.jsonl")


def test_committed_palmx_jsonl_loads():
    """The committed 981-case external benchmark must validate end-to-end."""
    committed = Path(__file__).resolve().parents[1] / "data" / "bench" / "palmx_ic.jsonl"
    if not committed.exists():
        pytest.skip("committed palmx_ic.jsonl not present")
    cases = load_benchmark(committed)
    assert len(cases) == 981
    assert all(c.source == "palmx-ic" for c in cases)
    assert len({c.case_id for c in cases}) == 981
